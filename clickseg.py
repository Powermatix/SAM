"""
SAM3 Click Segmentation
========================
Interactive point-click segmentation using SAM3 Tracker.

Open an image and click to segment objects:
  - LEFT CLICK  = positive point (include this region)
  - RIGHT CLICK = negative point (exclude this region)
  - MIDDLE CLICK / 'u' = undo last point
  - 'r' = reset all points
  - 's' = save current mask + overlay
  - 'm' = cycle mask display (overlay / mask only / off)
  - 'q' / ESC = quit

USAGE
-----
python click.py --image photo.jpg
python click.py --image photo.jpg --out result.png
python click.py --image https://example.com/photo.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import torch
from PIL import Image

try:
    from transformers import Sam3TrackerProcessor, Sam3TrackerModel
except ImportError as e:
    # Guard against confusing error if this file shadows a module via CWD
    print(f"[ERROR] Could not import Sam3TrackerModel: {e}")
    print("  Run: pip install git+https://github.com/huggingface/transformers.git")
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
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            out[k] = v
    return out


# ─── Interactive Session ─────────────────────────────────────────────────────

class ClickSegmenter:
    """Interactive matplotlib-based click segmentation session."""

    MASK_MODES = ["overlay", "mask_only", "off"]
    MASK_COLOR = np.array([0.3, 0.7, 1.0, 0.45])   # semi-transparent blue
    POS_COLOR  = "lime"
    NEG_COLOR  = "red"

    def __init__(self, image: Image.Image, processor, model, device, dtype, out_stem: str):
        self.image      = image
        self.img_np     = np.array(image)
        self.processor  = processor
        self.model      = model
        self.device     = device
        self.dtype      = dtype
        self.out_stem   = out_stem

        self.points     = []     # list of (x, y)
        self.labels     = []     # 1=positive, 0=negative
        self.mask       = None   # current best mask [H, W] bool
        self.mask_mode  = 0      # index into MASK_MODES

        # Matplotlib setup
        self.fig, self.ax = plt.subplots(1, figsize=(12, 8))
        self.fig.canvas.manager.set_window_title("SAM3 Click Segmentation")
        self.ax.set_title(self._title(), fontsize=10)
        self.ax.axis("off")
        self.im_artist = self.ax.imshow(self.img_np)

        # Overlay image for mask
        self.mask_overlay = self.ax.imshow(
            np.zeros((*self.img_np.shape[:2], 4)),
            alpha=1.0
        )

        # Point scatter
        self.scatter_pos = self.ax.scatter([], [], s=80, c=self.POS_COLOR,
                                           edgecolors="white", linewidths=1.5, zorder=5)
        self.scatter_neg = self.ax.scatter([], [], s=80, c=self.NEG_COLOR,
                                           marker="x", linewidths=2, zorder=5)

        # Legend
        legend_handles = [
            mpatches.Patch(color=self.POS_COLOR, label="Left click: include"),
            mpatches.Patch(color=self.NEG_COLOR, label="Right click: exclude"),
        ]
        self.ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                       framealpha=0.7, facecolor="black", labelcolor="white")

        # Connect events
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self.fig.tight_layout()

    def _title(self) -> str:
        n_pos = sum(1 for l in self.labels if l == 1)
        n_neg = sum(1 for l in self.labels if l == 0)
        mode  = self.MASK_MODES[self.mask_mode]
        return (f"Points: {len(self.points)} (+{n_pos} / -{n_neg})  |  "
                f"Display: {mode}  |  "
                f"Keys: u=undo  r=reset  s=save  m=mode  q=quit")

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = int(round(event.xdata)), int(round(event.ydata))

        if event.button == 1:       # left = positive
            self.points.append([x, y])
            self.labels.append(1)
        elif event.button == 3:     # right = negative
            self.points.append([x, y])
            self.labels.append(0)
        elif event.button == 2:     # middle = undo
            self._undo()
            self._refresh()
            return
        else:
            return

        self._run_inference()
        self._refresh()

    def _on_key(self, event):
        if event.key in ("q", "escape"):
            plt.close(self.fig)
        elif event.key == "u":
            self._undo()
            if self.points:
                self._run_inference()
            else:
                self.mask = None
            self._refresh()
        elif event.key == "r":
            self.points.clear()
            self.labels.clear()
            self.mask = None
            self._refresh()
        elif event.key == "s":
            self._save()
        elif event.key == "m":
            self.mask_mode = (self.mask_mode + 1) % len(self.MASK_MODES)
            self._refresh()

    def _undo(self):
        if self.points:
            self.points.pop()
            self.labels.pop()

    def _run_inference(self):
        """Run SAM3 Tracker with current points."""
        if not self.points:
            self.mask = None
            return

        # Format: [batch, objects, points, coords] and [batch, objects, labels]
        input_points = [[[p for p in self.points]]]
        input_labels = [[[l for l in self.labels]]]

        inputs = to_device(
            self.processor(
                images=self.image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ),
            self.device, self.dtype
        )

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process — get masks on CPU
        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"]
        )[0]  # [num_objects, num_multimask, H, W]

        # Pick the best mask (highest IoU score from model)
        if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
            best_idx = outputs.iou_scores[0, 0].argmax().item()
        else:
            best_idx = 0

        self.mask = masks[0, best_idx].numpy() > 0.0  # [H, W] bool

    def _refresh(self):
        """Redraw points and mask overlay."""
        # Update scatter
        pos_pts = np.array([[p[0], p[1]] for p, l in zip(self.points, self.labels) if l == 1])
        neg_pts = np.array([[p[0], p[1]] for p, l in zip(self.points, self.labels) if l == 0])
        self.scatter_pos.set_offsets(pos_pts if len(pos_pts) > 0 else np.empty((0, 2)))
        self.scatter_neg.set_offsets(neg_pts if len(neg_pts) > 0 else np.empty((0, 2)))

        # Update mask overlay
        mode = self.MASK_MODES[self.mask_mode]
        H, W = self.img_np.shape[:2]
        overlay = np.zeros((H, W, 4))

        if self.mask is not None and mode != "off":
            if mode == "overlay":
                overlay[self.mask] = self.MASK_COLOR
                self.im_artist.set_data(self.img_np)
            elif mode == "mask_only":
                self.im_artist.set_data(self.img_np)
                overlay[self.mask]  = [0.3, 0.7, 1.0, 0.8]
                overlay[~self.mask] = [0.0, 0.0, 0.0, 0.6]
        else:
            self.im_artist.set_data(self.img_np)

        self.mask_overlay.set_data(overlay)

        self.ax.set_title(self._title(), fontsize=10)
        self.fig.canvas.draw_idle()

    def _save(self):
        """Save current mask overlay and binary mask."""
        if self.mask is None:
            print("  Nothing to save (no mask yet)")
            return

        # 1. Overlay image
        rgba = np.zeros((*self.img_np.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = self.img_np
        rgba[..., 3]  = 255
        # Tint masked area
        mask_rgba = rgba.copy()
        mask_rgba[self.mask, 0] = np.clip(mask_rgba[self.mask, 0].astype(int) * 0.5 + 76, 0, 255).astype(np.uint8)
        mask_rgba[self.mask, 1] = np.clip(mask_rgba[self.mask, 1].astype(int) * 0.5 + 178, 0, 255).astype(np.uint8)
        mask_rgba[self.mask, 2] = np.clip(mask_rgba[self.mask, 2].astype(int) * 0.5 + 255, 0, 255).astype(np.uint8)
        overlay_path = f"{self.out_stem}_overlay.png"
        Image.fromarray(mask_rgba[..., :3]).save(overlay_path)
        print(f"  Saved overlay → {overlay_path}")

        # 2. Binary mask
        mask_path = f"{self.out_stem}_mask.png"
        Image.fromarray((self.mask * 255).astype(np.uint8)).save(mask_path)
        print(f"  Saved mask    → {mask_path}")

        # 3. Masked-out image (object blacked out)
        masked_img = self.img_np.copy()
        masked_img[self.mask] = 0
        blackout_path = f"{self.out_stem}_blackout.png"
        Image.fromarray(masked_img).save(blackout_path)
        print(f"  Saved blackout → {blackout_path}")

        # 4. Cutout (only the object, transparent background)
        cutout = np.zeros((*self.img_np.shape[:2], 4), dtype=np.uint8)
        cutout[self.mask, :3] = self.img_np[self.mask]
        cutout[self.mask, 3]  = 255
        cutout_path = f"{self.out_stem}_cutout.png"
        Image.fromarray(cutout).save(cutout_path)
        print(f"  Saved cutout  → {cutout_path}")

    def run(self):
        print("\n  Ready! Click on the image to segment.")
        print("  Left=include  Right=exclude  u=undo  r=reset  s=save  m=mode  q=quit\n")
        plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="SAM3 Click Segmentation — interactive point-click segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--image", required=True, help="Image path or URL")
    p.add_argument("--out",   default=None,  help="Output stem (default: <image_stem>_click)")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    print(f"\nSAM3 Click Segmentation")
    print(f"  Image  : {args.image}")
    print(f"  Device : {device.upper()}\n")

    # Load model
    print("Loading SAM3 Tracker model...")
    processor = Sam3TrackerProcessor.from_pretrained(MODEL_ID)
    model     = Sam3TrackerModel.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print("Model ready.\n")

    # Load image
    image = load_image(args.image)
    print(f"Image size: {image.width}×{image.height} px")

    # Output stem
    if args.out:
        out_stem = str(Path(args.out).with_suffix(""))
    else:
        stem = Path(args.image).stem if not args.image.startswith("http") else "image"
        out_stem = f"{stem}_click"

    # Launch interactive session
    session = ClickSegmenter(image, processor, model, device, dtype, out_stem)
    session.run()

    print("Done.")


if __name__ == "__main__":
    main()
