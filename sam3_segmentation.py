"""
SAM3 Visual Prompt Segmentation  (facebook/sam3)
=================================================
Uses Sam3TrackerModel for Promptable Visual Segmentation (PVS):
  - Point prompts  (foreground / background clicks)
  - Box prompts    ([x1, y1, x2, y2])
  - Combined       (points + box together)

Install:
    pip install transformers torch pillow matplotlib requests accelerate
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")   # save PNGs without needing tkinter / display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import requests
from io import BytesIO
from accelerate import Accelerator

from transformers import Sam3TrackerProcessor, Sam3TrackerModel


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam3"
device   = Accelerator().device


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
def load_model(model_id: str = MODEL_ID):
    print(f"Loading {model_id} on {device} …")
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    model     = Sam3TrackerModel.from_pretrained(model_id)
    model.to(device).eval()
    print("Model ready.\n")
    return processor, model


# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────
def segment_with_points(
    image: Image.Image,
    processor: Sam3TrackerProcessor,
    model: Sam3TrackerModel,
    input_points: list[list[list[list[float]]]],  # (img, obj, pt, xy) — 4-D
    input_labels: list[list[list[int]]],           # (img, obj, label)  — 3-D
    multimask_output: bool = True,
):
    """
    Point-prompted segmentation.

    input_points shape: [[[[x, y]]]]   — 1 image, 1 object, 1 point
    input_labels shape: [[[1]]]        — 1 = foreground, 0 = background
    """
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    # post_process_masks expects original_sizes from the inputs
    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )
    # masks[0] shape: (num_objects, num_multimask_outputs, H, W)
    # For single-object prompts, squeeze to (num_multimask, H, W)
    result = masks[0]
    if result.dim() == 4 and result.shape[0] == 1:
        result = result.squeeze(0)   # (num_multimask, H, W)
    return result, outputs.iou_scores[0].cpu()   # first image in batch


def segment_with_boxes(
    image: Image.Image,
    processor: Sam3TrackerProcessor,
    model: Sam3TrackerModel,
    input_boxes: list[list[list[float]]],  # (img, box, 4) — 3-D
    multimask_output: bool = False,
):
    """
    Box-prompted segmentation.

    input_boxes shape: [[[x1, y1, x2, y2]]]  — 1 image, 1 box
    """
    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )
    result = masks[0]
    if result.dim() == 4 and result.shape[0] == 1:
        result = result.squeeze(0)
    return result, outputs.iou_scores[0].cpu()


def segment_with_points_and_boxes(
    image: Image.Image,
    processor: Sam3TrackerProcessor,
    model: Sam3TrackerModel,
    input_points: list[list[list[list[float]]]],
    input_labels: list[list[list[int]]],
    input_boxes: list[list[list[float]]],
    multimask_output: bool = False,
):
    """Combined point + box prompts."""
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"],
    )
    result = masks[0]
    if result.dim() == 4 and result.shape[0] == 1:
        result = result.squeeze(0)
    return result, outputs.iou_scores[0].cpu()


# ─────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────
COLORS = [
    [30, 144, 255],   # dodger blue
    [255, 99,  71],   # tomato
    [50, 205,  50],   # lime green
    [255, 215,   0],  # gold
    [218, 112, 214],  # orchid
]


def _overlay_mask(ax, mask: np.ndarray, color_idx: int = 0, alpha: float = 0.45):
    c = np.array(COLORS[color_idx % len(COLORS)]) / 255.0
    h, w = mask.shape
    overlay = np.zeros((h, w, 4))
    overlay[mask] = [*c, alpha]
    ax.imshow(overlay)


def _draw_points(ax, points_4d, labels_3d):
    """points_4d: [[[[x,y]]]], labels_3d: [[[lbl]]]"""
    for obj_pts, obj_lbls in zip(points_4d[0], labels_3d[0]):
        for (x, y), lbl in zip(obj_pts, obj_lbls):
            marker = "*" if lbl == 1 else "X"
            color  = "lime" if lbl == 1 else "red"
            ax.plot(x, y, marker=marker, markersize=14, color=color,
                    markeredgecolor="black", markeredgewidth=1.2)


def _draw_boxes(ax, boxes_3d):
    """boxes_3d: [[[x1, y1, x2, y2]]]"""
    for x1, y1, x2, y2 in boxes_3d[0]:
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="yellow", facecolor="none",
        )
        ax.add_patch(rect)


def visualise(
    image: Image.Image,
    masks,              # tensor (N, H, W) bool
    scores,
    points_4d=None,
    labels_3d=None,
    boxes_3d=None,
    title: str = "SAM3 Segmentation",
    save_path: str | None = None,
):
    # Ensure masks is 3-D: (N, H, W)
    if masks.dim() == 4:
        masks = masks.view(-1, masks.shape[-2], masks.shape[-1])
    n = masks.shape[0]
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    if n + 1 == 1:
        axes = [axes]

    # — Original image panel
    axes[0].imshow(image)
    axes[0].set_title("Input", fontsize=11)
    axes[0].axis("off")
    if points_4d:
        _draw_points(axes[0], points_4d, labels_3d)
    if boxes_3d:
        _draw_boxes(axes[0], boxes_3d)

    # — One panel per mask
    for i in range(n):
        axes[i + 1].imshow(image)
        mask_np = masks[i].numpy().astype(bool)
        _overlay_mask(axes[i + 1], mask_np, color_idx=i)
        score_val = float(scores.flatten()[i]) if scores.numel() > i else 0.0
        axes[i + 1].set_title(f"Mask {i+1}  IoU={score_val:.3f}", fontsize=11)
        axes[i + 1].axis("off")
        if points_4d:
            _draw_points(axes[i + 1], points_4d, labels_3d)
        if boxes_3d:
            _draw_boxes(axes[i + 1], boxes_3d)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.close(fig)   # close instead of show when using Agg backend


# ─────────────────────────────────────────────────────────────
# Load image helper
# ─────────────────────────────────────────────────────────────
def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http"):
        resp = requests.get(path_or_url, stream=True)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    return Image.open(path_or_url).convert("RGB")


# ─────────────────────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────────────────────
def main():
    # ── 1. Image ─────────────────────────────────────────────
    IMAGE = "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    # Replace with a local path: IMAGE = "your_image.jpg"

    print(f"Loading image: {IMAGE}")
    image = load_image(IMAGE)
    W, H  = image.size
    print(f"Image size: {W}×{H}\n")

    # ── 2. Model ──────────────────────────────────────────────
    processor, model = load_model()

    # ── 3a. Point prompt ─────────────────────────────────────
    # SAM3 Tracker uses 4-D point arrays: (image, object, point, xy)
    # and 3-D label arrays:               (image, object, label)
    print("=== Point prompt ===")
    pt_points = [[[[500, 375]]]]   # 1 image, 1 object, 1 foreground click
    pt_labels = [[[1]]]
    masks_p, scores_p = segment_with_points(image, processor, model, pt_points, pt_labels)
    print(f"Masks shape: {masks_p.shape} | Scores: {scores_p}")
    visualise(
        image, masks_p, scores_p,
        points_4d=pt_points, labels_3d=pt_labels,
        title="Point Prompt – SAM3",
        save_path="output_point_prompt.png",
    )

    # ── 3b. Box prompt ────────────────────────────────────────
    print("\n=== Box prompt ===")
    margin    = 0.15
    box_coords = [[int(W * margin), int(H * margin),
                   int(W * (1 - margin)), int(H * (1 - margin))]]
    bx_boxes  = [box_coords]   # shape: (image, box, 4)
    masks_b, scores_b = segment_with_boxes(image, processor, model, bx_boxes)
    print(f"Masks shape: {masks_b.shape} | Scores: {scores_b}")
    visualise(
        image, masks_b, scores_b,
        boxes_3d=bx_boxes,
        title="Box Prompt – SAM3",
        save_path="output_box_prompt.png",
    )

    # ── 3c. Foreground + background point prompt ──────────────
    print("\n=== Multi-point prompt (fg + bg) ===")
    multi_points = [[[[500, 375], [10, 10]]]]   # obj has 2 points
    multi_labels = [[[1, 0]]]                    # first fg, second bg
    masks_m, scores_m = segment_with_points(
        image, processor, model, multi_points, multi_labels, multimask_output=False
    )
    print(f"Masks shape: {masks_m.shape} | Scores: {scores_m}")
    visualise(
        image, masks_m, scores_m,
        points_4d=multi_points, labels_3d=multi_labels,
        title="Multi-Point Prompt – SAM3",
        save_path="output_multipoint_prompt.png",
    )

    # ── 3d. Combined point + box ──────────────────────────────
    print("\n=== Combined point + box prompt ===")
    masks_c, scores_c = segment_with_points_and_boxes(
        image, processor, model,
        input_points=pt_points,
        input_labels=pt_labels,
        input_boxes=bx_boxes,
    )
    print(f"Masks shape: {masks_c.shape} | Scores: {scores_c}")
    visualise(
        image, masks_c, scores_c,
        points_4d=pt_points, labels_3d=pt_labels,
        boxes_3d=bx_boxes,
        title="Combined Prompt – SAM3",
        save_path="output_combined_prompt.png",
    )

    print("\nAll done! PNGs saved to the working directory.")


if __name__ == "__main__":
    main()