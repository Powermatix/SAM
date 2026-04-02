import torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading on {device}...")
processor = Sam3Processor.from_pretrained("facebook/sam3")
model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=dtype).to(device)

# Create dummy image
image = Image.new("RGB", (300, 300), color=(128, 128, 128))

# Simulate two boxes drawn by user: one include, one exclude
boxes_data = [
    {"x1": 50, "y1": 50, "x2": 100, "y2": 100, "label": 1},
    {"x1": 150, "y1": 150, "x2": 200, "y2": 200, "label": 0}
]

boxes = [[b["x1"], b["y1"], b["x2"], b["y2"]] for b in boxes_data]
labels = [b["label"] for b in boxes_data]

inputs = processor(
    images=image,
    input_boxes=[boxes],
    input_boxes_labels=[labels],
    return_tensors="pt"
)

# move to device
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device=device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.1,
    mask_threshold=0.1,
    target_sizes=inputs["original_sizes"].tolist(),
)[0]

print("Masks shape:", results["masks"].shape)
print("Scores:", results["scores"])
