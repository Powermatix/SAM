"""
SAM3 Basic Functionality Test
==============================
Tests SAM3 (Segment Anything Model 3) on a random COCO image using:
  1. Text-only prompt segmentation
  2. Bounding box prompt segmentation
  3. Combined text + negative box prompt

Outputs:
  - output_text_prompt.png     : masks from text prompt
  - output_box_prompt.png      : masks from bounding box prompt
  - output_combined_prompt.png : masks from combined prompt

Usage:
  python test_sam3.py [--text PROMPT] [--image IMAGE_URL_OR_PATH]
"""

import argparse
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image

# ── Check transformers version supports SAM3 ─────────────────────────────────
try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("\n[ERROR] Sam3Model not found in transformers.")
    print("  Make sure you installed transformers from source:")
    print("  pip install git+https://github.com/huggingface/transformers.git")
    sys.exit(1)

# ── Default test resources (COCO val2017) ──────────────────────────────────
DEFAULT_IMAGE_URL  = "http://images.cocodataset.org/val2017/000000077595.jpg"
KITCHEN_IMAGE_URL  = "http://images.cocodataset.org/val2017/000000136466.jpg"
MODEL_ID           = "facebook/sam3"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_image(url_or_path: str) -> Image.Image:
    """Load an image from a URL or local file path."""
    if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
        print(f"  Downloading image from: {url_or_path}")
        resp = requests.get(url_or_path, stream=True, timeout=30)
        resp.raise_for_status()
        return Image.open(resp.raw).convert("RGB")
    else:
        print(f"  Loading local image: {url_or_path}")
        return Image.open(url_or_path).convert("RGB")


def overlay_masks(image: Image.Image, masks: torch.Tensor) -> Image.Image:
    """Overlay coloured semi-transparent masks on the source image."""
    result = image.convert("RGBA")
    masks_np = (255 * masks.cpu().numpy().astype(np.uint8))
    n_masks  = masks_np.shape[0]
    cmap     = matplotlib.colormaps.get_cmap("rainbow").resampled(max(n_masks, 1))
    colors   = [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n_masks)]

    for mask, color in zip(masks_np, colors):
        mask_img = Image.fromarray(mask)
        overlay  = Image.new("RGBA", image.size, color + (0,))
        alpha    = mask_img.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        result = Image.alpha_composite(result, overlay)
    return result


def draw_boxes(image: Image.Image, boxes: list, scores: list) -> Image.Image:
    """Draw bounding boxes with confidence scores."""
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(image)
    ax.axis("off")

    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(boxes), 1)))
    for i, (box, score, color) in enumerate(zip(boxes, scores, colors)):
        x1, y1, x2, y2 = box
        rect = plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 5,
            f"#{i+1}: {score:.2f}",
            color="white",
            fontsize=9,
            bbox=dict(facecolor=color, alpha=0.7, pad=2, edgecolor="none"),
        )

    fig.tight_layout(pad=0)
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf  = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf  = buf.reshape(h, w, 4)
    buf  = np.roll(buf, -1, axis=2)          # ARGB → RGBA
    plt.close(fig)
    return Image.fromarray(buf, "RGBA")


def save_result(base_image: Image.Image, results: dict, out_path: str, title: str):
    """Overlay masks, print stats, and save to file."""
    masks  = results.get("masks",  torch.zeros(0))
    boxes  = results.get("boxes",  [])
    scores = results.get("scores", [])

    print(f"\n  ── {title} ──")
    print(f"     Objects found : {len(masks)}")
    for i, (box, score) in enumerate(zip(boxes.tolist() if hasattr(boxes, 'tolist') else boxes,
                                         scores.tolist() if hasattr(scores, 'tolist') else scores)):
        print(f"     [{i+1}] score={score:.3f}  box={[round(v) for v in box]}")

    if len(masks) > 0:
        composite = overlay_masks(base_image, masks)
    else:
        print("     (no masks returned — lowering threshold may help)")
        composite = base_image.convert("RGBA")

    composite.convert("RGB").save(out_path)
    print(f"     Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAM3 Basic Functionality Test")
    parser.add_argument("--text",  default="person",   help="Text prompt (default: 'person')")
    parser.add_argument("--image", default=DEFAULT_IMAGE_URL, help="Image URL or local path")
    args = parser.parse_args()

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 on GPU for efficiency, float32 on CPU to avoid dtype issues
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"\n{'='*55}")
    print(f"  SAM3 Functionality Test")
    print(f"{'='*55}")
    print(f"  Device  : {device.upper()} ({model_dtype})")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Prompt  : \"{args.text}\"")
    print(f"  Image   : {args.image}")
    print(f"{'='*55}\n")

    # ── Load model ────────────────────────────────────────────────────────────
    print("[1/4] Loading SAM3 model (first run downloads ~several GB)...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model     = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=model_dtype)
    model     = model.to(device)
    model.eval()
    print("      Model loaded ✓")

    def to_model(inputs: dict) -> dict:
        """Move all float tensors in a batch to the model's device + dtype."""
        result = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(device=device, dtype=model_dtype) if v.is_floating_point() else v.to(device=device)
            else:
                result[k] = v
        return result

    # ── Load test image ───────────────────────────────────────────────────────
    print("\n[2/4] Loading test image...")
    image = load_image(args.image)
    print(f"      Image size: {image.width}×{image.height}  ✓")

    # ── Test 1: Text-only prompt ──────────────────────────────────────────────
    print("\n[3/4] Running segmentation tests...")

    print(f"\n  (a) Text prompt: \"{args.text}\"")
    inputs = to_model(processor(images=image, text=args.text, return_tensors="pt"))
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=inputs.get("original_sizes").tolist()
    )[0]
    save_result(image, results, "output_text_prompt.png", f'Text: "{args.text}"')

    # ── Test 2: Bounding box prompt ────────────────────────────────────────────
    print("\n  (b) Bounding box prompt (centre quarter of image)")
    w, h = image.width, image.height
    box_xyxy    = [w//4, h//4, 3*w//4, 3*h//4]     # centre of image
    input_boxes = [[box_xyxy]]
    input_labels = [[1]]
    inputs_box = to_model(processor(
        images=image,
        input_boxes=input_boxes,
        input_boxes_labels=input_labels,
        return_tensors="pt"
    ))
    with torch.no_grad():
        outputs_box = model(**inputs_box)
    results_box = processor.post_process_instance_segmentation(
        outputs_box,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=inputs_box.get("original_sizes").tolist()
    )[0]
    save_result(image, results_box, "output_box_prompt.png", "Bounding box (centre quarter)")

    # ── Test 3: Kitchen image with combined prompt ────────────────────────────
    print("\n  (c) Kitchen image — text=\"handle\" + negative box")
    kitchen_image = load_image(KITCHEN_IMAGE_URL)
    oven_handle_box = [40, 183, 318, 204]
    inputs_combo = to_model(processor(
        images=kitchen_image,
        text="handle",
        input_boxes=[[oven_handle_box]],
        input_boxes_labels=[[0]],    # 0 = negative (exclude this region)
        return_tensors="pt"
    ))
    with torch.no_grad():
        outputs_combo = model(**inputs_combo)
    results_combo = processor.post_process_instance_segmentation(
        outputs_combo,
        threshold=0.3,
        mask_threshold=0.3,
        target_sizes=inputs_combo.get("original_sizes").tolist()
    )[0]
    save_result(kitchen_image, results_combo, "output_combined_prompt.png",
                'Text: "handle" + negative box')

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("  All tests complete! Output files:")
    print("    output_text_prompt.png")
    print("    output_box_prompt.png")
    print("    output_combined_prompt.png")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
