from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt

model_id = "IDEA-Research/grounding-dino-base"
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = r"/home/VLSP/data/VLSP2025/train_data/train_images/train_8_41.jpg"
image = Image.open(image_url).convert("RGB")
text = "traffic sign."

inputs = processor(images=image, text=text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.2,
    text_threshold=0.2,
    target_sizes=[image.size[::-1]]
)

   
# --- Draw results ---
result = results[0]
print(result)
draw = ImageDraw.Draw(image)

# Optional: specify a font
try:
    font = ImageFont.truetype("arial.ttf", 20)
except:
    font = ImageFont.load_default()

for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
    box = box.tolist()
    label_text = f"{label} ({score:.2f})"
    draw.rectangle(box, outline="red", width=3)
    draw.text((box[0], box[1]), label_text, fill="red", font=font)

# --- Show image ---
plt.figure(figsize=(12, 8))
plt.imshow(image)
plt.axis("off")
plt.show()