from models.backbone import CLIPBackbone
from PIL import Image
import torch

# Khởi tạo backbone CLIP
model = CLIPBackbone("openai/clip-vit-base-patch32")

# Mở ảnh test
image = Image.open("Testclip.png").convert("RGB")

# Trích đặc trưng
features = model([image])

print("Feature shape:", features.shape)
print("Feature sample:", features[0, :5, :5])  # In 1 phần nhỏ của tensor
