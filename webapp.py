"""
SAM3 Segmentation Web App
==========================
Combined text segmentation + click correction in the browser.
Uses a single Sam3Model for both text and point prompts.

Workflow:
  1. Upload an image
  2. Type a text prompt → get initial text-based segmentation
  3. Click to refine: positive/negative points correct the mask
  4. Download the result

Usage:
    source venv/bin/activate
    python webapp.py [--port 5000] [--host 0.0.0.0]
"""

import argparse
import base64
import io
import os
import sys
import uuid

import numpy as np
import torch
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    print("[ERROR] SAM3 not found.")
    print("  pip install git+https://github.com/huggingface/transformers.git")
    sys.exit(1)

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = "facebook/sam3"
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB

# ─── Global model state ─────────────────────────────────────────────────────
processor = None
model = None
device = None
dtype = None


def load_model():
    global processor, model, device, dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading Sam3Model on {device.upper()}...")
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID, torch_dtype=dtype).to(device)
    model.eval()
    print("Model ready.\n")


def to_device(inputs: dict) -> dict:
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)
        else:
            out[k] = v
    return out


def outputs_to_cpu(outputs):
    for k in vars(outputs):
        v = getattr(outputs, k)
        if isinstance(v, torch.Tensor):
            setattr(outputs, k, v.cpu())
    return outputs


def mask_to_b64(mask_np: np.ndarray, mode: str = "overlay") -> str:
    H, W = mask_np.shape
    if mode == "overlay":
        img = np.zeros((H, W, 4), dtype=np.uint8)
        img[mask_np] = [60, 140, 255, 128]
        pil = Image.fromarray(img, "RGBA")
    elif mode == "mask":
        pil = Image.fromarray((mask_np * 255).astype(np.uint8), "L")
    else:
        raise ValueError(mode)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def blackout_to_b64(image: Image.Image, mask_np: np.ndarray) -> str:
    img_np = np.array(image)
    img_np[mask_np] = 0
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify(error="No image file"), 400
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_id = str(uuid.uuid4())[:8]
    img_path = os.path.join(UPLOAD_DIR, f"{img_id}.png")
    img.save(img_path)
    return jsonify(id=img_id, width=img.width, height=img.height)


@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/segment_text", methods=["POST"])
def segment_text():
    """Text-based segmentation using Sam3Model."""
    data = request.json
    img_id = data.get("image_id")
    text = data.get("text", "").strip()
    threshold = data.get("threshold", 0.3)

    if not img_id or not text:
        return jsonify(error="Missing image_id or text"), 400

    img_path = os.path.join(UPLOAD_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        return jsonify(error="Image not found"), 404

    image = Image.open(img_path).convert("RGB")

    inputs = to_device(
        processor(images=image, text=text, return_tensors="pt")
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

    masks = results["masks"]
    boxes = results["boxes"]
    scores = results["scores"]
    n = len(masks)

    if n == 0:
        return jsonify(found=0, objects=[])

    objects = []
    for i in range(n):
        mask_np = masks[i].cpu().numpy().astype(bool)
        box = [round(float(v)) for v in boxes[i].tolist()]
        score = float(scores[i])
        objects.append({
            "id": i,
            "overlay": mask_to_b64(mask_np, "overlay"),
            "mask": mask_to_b64(mask_np, "mask"),
            "box": box,
            "score": score,
            "pixels": int(mask_np.sum()),
        })

    combined = masks.any(dim=0).cpu().numpy().astype(bool)

    return jsonify(
        found=n,
        objects=objects,
        combined_overlay=mask_to_b64(combined, "overlay"),
        combined_mask=mask_to_b64(combined, "mask"),
        combined_blackout=blackout_to_b64(image, combined),
        combined_pixels=int(combined.sum()),
        total_pixels=int(combined.shape[0] * combined.shape[1]),
    )


@app.route("/segment_boxes", methods=["POST"])
def segment_boxes():
    """Box-based segmentation using Sam3Model.

    Accepts user-drawn bounding boxes with labels (1=include, 0=exclude).
    """
    data = request.json
    img_id = data.get("image_id")
    boxes_data = data.get("boxes", [])   # [{x1, y1, x2, y2, label}]

    if not img_id or not boxes_data:
        return jsonify(error="Missing image_id or boxes"), 400

    img_path = os.path.join(UPLOAD_DIR, f"{img_id}.png")
    if not os.path.exists(img_path):
        return jsonify(error="Image not found"), 404

    image = Image.open(img_path).convert("RGB")

    boxes = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_data]
    labels = [b["label"] for b in boxes_data]

    inputs = to_device(
        processor(
            images=image,
            input_boxes=[boxes],
            input_boxes_labels=[labels],
            return_tensors="pt",
        )
    )

    with torch.no_grad():
        outputs = model(**inputs)

    outputs = outputs_to_cpu(outputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.1,
        mask_threshold=0.1,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    masks = results["masks"]
    scores_t = results["scores"]

    if len(masks) == 0:
        return jsonify(
            error="No mask found in drawn box",
            overlay=None, mask=None, blackout=None,
            score=0, mask_pixels=0, total_pixels=0,
        )

    # Pick the mask with highest score
    best_idx = scores_t.argmax().item()
    best_mask = masks[best_idx].cpu().numpy().astype(bool)
    best_score = float(scores_t[best_idx])

    return jsonify(
        overlay=mask_to_b64(best_mask, "overlay"),
        mask=mask_to_b64(best_mask, "mask"),
        blackout=blackout_to_b64(image, best_mask),
        score=best_score,
        mask_pixels=int(best_mask.sum()),
        total_pixels=int(best_mask.shape[0] * best_mask.shape[1]),
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="SAM3 Segmentation Web App")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--host", default="127.0.0.1",
                   help="Use 0.0.0.0 to allow remote access")
    args = p.parse_args()

    load_model()

    print(f"\n  → Open http://{args.host}:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
