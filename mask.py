"""
SAM3 MVS Masking Tool
=====================
Black-out selected objects in images for MVS (Multi-View Stereo) pipelines.

Segments objects matching text prompts and covers them with solid black,
producing clean images suitable for photogrammetry / 3D reconstruction.

USAGE
-----
# Mask a single image  (blacks out people)
python mask.py --image photo.jpg --text person

# Mask multiple concepts
python mask.py --image photo.jpg --text person car tree

# Process an entire directory
python mask.py --dir images/ --text person car --ext jpg png

# Invert: keep only the matched objects, black out everything else
python mask.py --image photo.jpg --text building --invert

# Also export the binary mask (white = masked region)
python mask.py --image photo.jpg --text person --export-mask

# Custom output directory
python mask.py --dir images/ --text person --out-dir masked/

# Dilate masks to add safety margin (pixels)
python mask.py --image photo.jpg --text person --dilate 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image, ImageFilter

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("[ERROR] Sam3Model not found. Run:  pip install git+https://github.com/huggingface/transformers.git")
    sys.exit(1)

MODEL_ID = "facebook/sam3"
IMAGE_EXTS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(path: str) -> Image.Image:
    if path.startswith(("http://", "https://")):
        r = requests.get(path, stream=True, timeout=30)
        r.raise_for_status()
        return Image.open(r.raw).convert("RGB")
    return Image.open(path).convert("RGB")


def to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            out[k] = v
    return out


def outputs_to_cpu(outputs):
    """Move model outputs to CPU to avoid OOM during post-processing."""
    for k in vars(outputs):
        v = getattr(outputs, k)
        if isinstance(v, torch.Tensor):
            setattr(outputs, k, v.cpu())
    return outputs


def dilate_mask(mask_np: np.ndarray, pixels: int) -> np.ndarray:
    """Dilate a binary mask by the given number of pixels."""
    if pixels <= 0:
        return mask_np
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    # Use MaxFilter for dilation; apply multiple times for large values
    for _ in range(pixels):
        mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    return (np.array(mask_img) > 127).astype(np.uint8)


def segment_objects(image: Image.Image, text: str, processor, model,
                    device: str, dtype: torch.dtype, threshold: float):
    """Run SAM3 and return a combined binary mask of all detections."""
    inputs = to_device(
        processor(images=image, text=text, return_tensors="pt"),
        device, dtype
    )
    with torch.no_grad():
        outputs = model(**inputs)
    outputs = outputs_to_cpu(outputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    masks  = results["masks"]   # [N, H, W]
    scores = results["scores"]
    n = len(masks)

    if n == 0:
        return None, 0, []

    # Merge all masks into one binary mask
    combined = masks.any(dim=0).cpu().numpy().astype(np.uint8)  # [H, W]
    return combined, n, scores.tolist()


def apply_mask(image: Image.Image, mask_np: np.ndarray,
               invert: bool = False, fill_color: tuple = (0, 0, 0)) -> Image.Image:
    """Apply a binary mask to an image.
    
    mask_np: 1 = object region, 0 = background
    invert=False: black out the object (mask_np=1 → fill_color)
    invert=True:  black out everything EXCEPT the object
    """
    img_np = np.array(image)
    if invert:
        # Keep object, black out background
        img_np[mask_np == 0] = fill_color
    else:
        # Black out object, keep background
        img_np[mask_np == 1] = fill_color
    return Image.fromarray(img_np)


def process_single(image_path: str, text_prompt: str, processor, model,
                   device: str, dtype: torch.dtype, args) -> dict:
    """Process one image. Returns dict with stats."""
    image = load_image(image_path)

    combined_mask, n_objects, scores = segment_objects(
        image, text_prompt, processor, model, device, dtype, args.threshold
    )

    if combined_mask is None:
        return {"path": image_path, "found": 0, "skipped": True}

    # Dilate if requested
    if args.dilate > 0:
        combined_mask = dilate_mask(combined_mask, args.dilate)

    # Apply mask
    result = apply_mask(image, combined_mask, invert=args.invert)

    # Determine output path
    src = Path(image_path)
    if args.out and not args.dir:
        out_path = Path(args.out)
    elif args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / src.name
    else:
        out_path = src.parent / f"{src.stem}_masked{src.suffix}"

    result.save(out_path)

    # Export binary mask if requested
    if args.export_mask:
        mask_img = Image.fromarray((combined_mask * 255).astype(np.uint8))
        mask_out = out_path.parent / f"{out_path.stem}_binarymask.png"
        mask_img.save(mask_out)

    return {
        "path": image_path,
        "out": str(out_path),
        "found": n_objects,
        "scores": scores,
        "skipped": False,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SAM3 MVS Masking — black out selected objects for photogrammetry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="Single input image path or URL")
    src.add_argument("--dir",   help="Directory of images to process")

    p.add_argument("--text", nargs="+", required=True,
                   help="Object(s) to mask, e.g. --text person car")
    p.add_argument("--ext", nargs="+", default=None,
                   help="File extensions to include when using --dir (default: all image types)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Detection confidence threshold (default: 0.3)")
    p.add_argument("--dilate", type=int, default=0,
                   help="Dilate masks by N pixels for safety margin (default: 0)")
    p.add_argument("--invert", action="store_true",
                   help="Invert: keep matched objects, black out everything else")
    p.add_argument("--export-mask", action="store_true", dest="export_mask",
                   help="Also save the binary mask as a separate PNG")
    p.add_argument("--out", default=None,
                   help="Output path for single image mode")
    p.add_argument("--out-dir", default=None, dest="out_dir",
                   help="Output directory for batch mode")
    args = p.parse_args()

    text_prompt = " ".join(args.text)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\nSAM3 MVS Masking")
    print(f"  Prompt   : \"{text_prompt}\"")
    print(f"  Mode     : {'INVERT (keep objects)' if args.invert else 'BLACK OUT objects'}")
    print(f"  Dilate   : {args.dilate}px")
    print(f"  Device   : {device.upper()}")
    print(f"  Threshold: {args.threshold}\n")

    # Load model
    print("Loading model...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model     = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print("Model ready.\n")

    # Collect images
    if args.image:
        images = [args.image]
    else:
        exts = set(e.lower().lstrip(".") for e in args.ext) if args.ext else IMAGE_EXTS
        dir_path = Path(args.dir)
        images = sorted([
            str(f) for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower().lstrip(".") in exts
        ])
        if not images:
            print(f"  No images found in {args.dir}")
            sys.exit(1)
        print(f"Found {len(images)} images in {args.dir}\n")

    # Process
    total_masked = 0
    total_skipped = 0
    for i, img_path in enumerate(images, 1):
        prefix = f"[{i}/{len(images)}]" if len(images) > 1 else ""
        print(f"  {prefix} {Path(img_path).name}", end="", flush=True)

        result = process_single(img_path, text_prompt, processor, model, device, dtype, args)

        if result["skipped"]:
            print(f"  → no objects found, skipped")
            total_skipped += 1
        else:
            scores_str = ", ".join(f"{s:.2f}" for s in result["scores"])
            print(f"  → {result['found']} masked  [{scores_str}]  → {result['out']}")
            total_masked += 1

    print(f"\nDone! {total_masked} images masked, {total_skipped} skipped.\n")


if __name__ == "__main__":
    main()
