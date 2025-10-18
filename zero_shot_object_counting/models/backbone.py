# models/backbone.py
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPBackbone(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        print(f"Loading CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images):
        """
        Trích xuất đặc trưng ảnh (không dùng text)
        """
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        outputs = self.model.vision_model(**inputs)
        # Lấy patch-level features (chưa pooling)
        patch_features = outputs.last_hidden_state   # [B, num_patches, hidden_dim]
        return patch_features
