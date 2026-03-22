"""
SAM3 Segmentation Tool
======================
Segment objects in any image using Meta's SAM3 model.

USAGE EXAMPLES
--------------
# Text prompt (single or multiple words)
python segment.py --image photo.jpg --text person
python segment.py --image photo.jpg --text mirror window

# Bounding box (xyxy pixels)
python segment.py --image photo.jpg --box 100 50 400 300

# Text + negative box (exclude a region)
python segment.py --image photo.jpg --text "handle" --neg-box 40 183 318 204

# Segment EVERYTHING (no prompt needed)
python segment.py --image photo.jpg --everything

# Segment everything, denser grid
python segment.py --image photo.jpg --everything --grid-size 6

# URL input, custom output
python segment.py --image https://example.com/photo.jpg --text "dog" --out result.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("[ERROR] Sam3Model not found. Run:  pip install git+https://github.com/huggingface/transformers.git")
    sys.exit(1)

MODEL_ID = "facebook/sam3"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith(("http://", "https://")):
        r = requests.get(path_or_url, stream=True, timeout=30)
        r.raise_for_status()
        return Image.open(r.raw).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


def to_device(inputs: dict, device: str, dtype: torch.dtype) -> dict:
    """Move processor outputs to the correct device + dtype."""
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            out[k] = v
    return out


def outputs_to_cpu(outputs):
    """Move all tensor attributes of model outputs to CPU to avoid OOM during post-processing."""
    for k in vars(outputs):
        v = getattr(outputs, k)
        if isinstance(v, torch.Tensor):
            setattr(outputs, k, v.cpu())
    return outputs


def overlay_masks(image: Image.Image, masks: torch.Tensor, alpha: float = 0.5) -> Image.Image:
    """Overlay coloured semi-transparent masks on top of the image."""
    result = image.convert("RGBA")
    masks_np = (255 * masks.cpu().numpy().astype(np.uint8))
    n = masks_np.shape[0]
    cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n, 1))
    for i, (mask, color) in enumerate(zip(masks_np, [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)])):
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(Image.fromarray(mask).point(lambda v: int(v * alpha)))
        result = Image.alpha_composite(result, overlay)
    return result


def draw_boxes(image: Image.Image, boxes, scores) -> Image.Image:
    """Draw bounding boxes with confidence scores."""
    fig, ax = plt.subplots(1, figsize=(image.width / 80, image.height / 80))
    ax.imshow(image)
    ax.axis("off")
    n = len(boxes)
    colors = plt.cm.rainbow(np.linspace(0, 1, max(n, 1)))
    for i, (box, score, color) in enumerate(zip(boxes, scores, colors)):
        x1, y1, x2, y2 = [float(v) for v in box]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, lw=2, edgecolor=color, facecolor="none"))
        ax.text(x1, y1 - 4, f"#{i+1}  {score:.2f}", color="white", fontsize=8,
                bbox=dict(facecolor=color, alpha=0.75, pad=2, edgecolor="none"))
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
    plt.close(fig)
    return Image.fromarray(np.roll(buf, -1, axis=2), "RGBA")


def make_grid_boxes(img_w: int, img_h: int, grid: int, overlap: float = 0.3):
    """Generate a uniform grid of overlapping boxes covering the whole image."""
    step_x = img_w / grid
    step_y = img_h / grid
    pad_x  = step_x * overlap
    pad_y  = step_y * overlap
    boxes  = []
    for row in range(grid):
        for col in range(grid):
            x1 = max(0,     int(col * step_x - pad_x))
            y1 = max(0,     int(row * step_y - pad_y))
            x2 = min(img_w, int((col + 1) * step_x + pad_x))
            y2 = min(img_h, int((row + 1) * step_y + pad_y))
            boxes.append([x1, y1, x2, y2])
    return boxes


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Intersection-over-union between two binary masks."""
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a,  mask_b).sum()
    return float(inter) / float(union + 1e-6)


def nms_masks(all_masks: list, all_boxes: list, all_scores: list, iou_thresh: float = 0.5):
    """Greedy NMS over masks — keeps the highest-scoring mask among overlapping ones."""
    order   = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)
    kept    = []
    for idx in order:
        m = all_masks[idx]
        if not any(mask_iou(m, all_masks[k]) > iou_thresh for k in kept):
            kept.append(idx)
    masks  = np.stack([all_masks[i]  for i in kept])
    boxes  = [all_boxes[i]  for i in kept]
    scores = [all_scores[i] for i in kept]
    return masks, boxes, scores


def segment_everything(image, processor, model, device, dtype,
                       grid: int, threshold: float, iou_thresh: float = 0.5):
    """Sweep a grid of boxes over the image and collect all unique masks."""
    W, H    = image.width, image.height
    grid_boxes = make_grid_boxes(W, H, grid)
    total   = len(grid_boxes)

    all_masks, all_boxes, all_scores = [], [], []

    for i, box in enumerate(grid_boxes, 1):
        print(f"  Grid cell {i}/{total}  {box}", end="\r", flush=True)
        inputs = to_device(
            processor(images=image, input_boxes=[[box]],
                      input_boxes_labels=[[1]], return_tensors="pt"),
            device, dtype
        )
        with torch.no_grad():
            outputs = model(**inputs)
        outputs = outputs_to_cpu(outputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=threshold,
            target_sizes=inputs["original_sizes"].tolist()
        )[0]
        if len(results["masks"]) == 0:
            continue
        for mask, box_r, score in zip(
            results["masks"].cpu().numpy(),
            results["boxes"].tolist(),
            results["scores"].tolist(),
        ):
            all_masks.append(mask.astype(bool))
            all_boxes.append(box_r)
            all_scores.append(score)

    print(f"  Collected {len(all_masks)} raw masks, running NMS...           ")
    if not all_masks:
        return None, None, None

    masks_np, boxes_out, scores_out = nms_masks(all_masks, all_boxes, all_scores, iou_thresh)
    masks_tensor = torch.from_numpy(masks_np.astype(np.float32))
    return masks_tensor, boxes_out, scores_out


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SAM3 Segmentation Tool — segment any concept in an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image", required=True, help="Input image: local path or URL")
    p.add_argument("--text",  nargs="+", default=None,
                   help="Text concept(s) to segment, e.g. --text person  or  --text mirror window")
    p.add_argument("--box",   nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                   help="Positive bounding box in pixel coords (xyxy)")
    p.add_argument("--neg-box", nargs=4, type=int, metavar=("X1","Y1","X2","Y2"),
                   dest="neg_box", help="Negative bounding box to exclude a region")
    p.add_argument("--everything", action="store_true",
                   help="Segment everything in the image without any prompt")
    p.add_argument("--grid-size", type=int, default=4, dest="grid_size",
                   help="Grid density for --everything mode (default: 4 → 16 cells)")
    p.add_argument("--iou-thresh", type=float, default=0.5, dest="iou_thresh",
                   help="IoU threshold for NMS in --everything mode (default: 0.5)")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="Confidence threshold for detections (default: 0.3)")
    p.add_argument("--out", default=None,
                   help="Output path (default: <input_stem>_segmented.png)")
    p.add_argument("--save-masks", action="store_true",
                   help="Also save each individual mask as a separate PNG")
    p.add_argument("--boxes-only", action="store_true",
                   help="Draw bounding boxes instead of mask overlays")
    args = p.parse_args()

    if not args.text and not args.box and not args.neg_box and not args.everything:
        p.error("Provide at least one of --text, --box, --neg-box, or --everything")

    # ── Device setup ──────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    # ── Output path ───────────────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        stem = Path(args.image).stem if not args.image.startswith("http") else "image"
        out_path = Path(f"{stem}_segmented.png")

    print(f"\nSAM3 Segmentation")
    print(f"  Image     : {args.image}")
    if args.everything:
        print(f"  Mode      : EVERYTHING  (grid={args.grid_size}×{args.grid_size}, iou={args.iou_thresh})")
    else:
        print(f"  Text      : {args.text or '—'}")
        print(f"  Box       : {args.box or '—'}")
        print(f"  Neg box   : {args.neg_box or '—'}")
    print(f"  Threshold : {args.threshold}")
    print(f"  Device    : {device.upper()}")
    print(f"  Output    : {out_path}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model     = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print("Model ready.\n")

    # ── Load image ────────────────────────────────────────────────────────────
    print("Loading image...")
    image = load_image(args.image)
    print(f"  {image.width}×{image.height} px\n")

    # ── Inference ─────────────────────────────────────────────────────────────
    if args.everything:
        print(f"Sweeping {args.grid_size}×{args.grid_size} grid ({args.grid_size**2} cells)...")
        masks, boxes_out, scores_out = segment_everything(
            image, processor, model, device, dtype,
            grid=args.grid_size,
            threshold=args.threshold,
            iou_thresh=args.iou_thresh,
        )
        if masks is None:
            print("  No objects found. Try lowering --threshold or increasing --grid-size")
            sys.exit(0)
        boxes  = torch.tensor(boxes_out)   # keep as tensor for save_masks
        scores = torch.tensor(scores_out)
        n      = len(masks)
    else:
        # ── Build prompt inputs ───────────────────────────────────────────────
        proc_kwargs = dict(images=image, return_tensors="pt")
        if args.text:
            proc_kwargs["text"] = " ".join(args.text)

        boxes_list, labels_list = [], []
        if args.box:
            boxes_list.append(args.box)
            labels_list.append(1)       # positive
        if args.neg_box:
            boxes_list.append(args.neg_box)
            labels_list.append(0)       # negative
        if boxes_list:
            proc_kwargs["input_boxes"]        = [boxes_list]
            proc_kwargs["input_boxes_labels"] = [labels_list]

        inputs = to_device(processor(**proc_kwargs), device, dtype)

        print("Running segmentation...")
        with torch.no_grad():
            outputs = model(**inputs)

        # Move outputs to CPU before post-processing to avoid OOM on large images
        outputs = outputs_to_cpu(outputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]

        masks  = results["masks"]
        boxes  = results["boxes"]
        scores = results["scores"]
        n      = len(masks)

    print(f"  Found {n} object(s)\n")
    for i, (box, score) in enumerate(zip(boxes.tolist(), scores.tolist())):
        print(f"  [{i+1}] score={score:.3f}  box=[{', '.join(str(round(v)) for v in box)}]")

    if n == 0:
        print("\n  No objects found. Try lowering --threshold (e.g. --threshold 0.1)")
        sys.exit(0)

    # ── Visualise & save ─────────────────────────────────────────────────────
    if args.boxes_only:
        result_img = draw_boxes(image, boxes.tolist(), scores.tolist())
    else:
        result_img = overlay_masks(image, masks)
        # Also draw boxes on top of the mask overlay
        result_img = draw_boxes(result_img.convert("RGB"), boxes.tolist(), scores.tolist())

    result_img.convert("RGB").save(out_path)
    print(f"\n  Saved → {out_path}")

    if args.save_masks:
        mask_dir = out_path.parent / (out_path.stem + "_masks")
        mask_dir.mkdir(exist_ok=True)
        for i, mask in enumerate(masks):
            mask_img = Image.fromarray((255 * mask.cpu().numpy().astype(np.uint8)))
            mask_path = mask_dir / f"mask_{i+1:02d}_score{scores[i]:.2f}.png"
            mask_img.save(mask_path)
        print(f"  Individual masks → {mask_dir}/")

    print()


if __name__ == "__main__":
    main()
