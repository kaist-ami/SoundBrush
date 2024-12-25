from __future__ import annotations

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image

class ClipSimilarity(nn.Module):
    def __init__(self, name: str = "ViT-L/14"):
        super().__init__()
        assert name in ("RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px")  # fmt: skip
        self.size = {"RN50x4": 288, "RN50x16": 384, "RN50x64": 448, "ViT-L/14@336px": 336}.get(name, 224)

        self.model, self.preprocessor = clip.load(name, device="cpu", download_root="./")
        self.model.eval().requires_grad_(False)

        self.register_buffer("mean", torch.tensor((0.48145466, 0.4578275, 0.40821073)))
        self.register_buffer("std", torch.tensor((0.26862954, 0.26130258, 0.27577711)))

    def encode_text(self, text: list[str]) -> torch.Tensor:
        text = clip.tokenize(text, truncate=True).to(next(self.parameters()).device)
        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    # def encode_image(self, image: torch.Tensor) -> torch.Tensor:  # Input images in range [0, 1].
    #     image = F.interpolate(image.float(), size=self.size, mode="bicubic", align_corners=False)
    #     image = image - rearrange(self.mean, "c -> 1 c 1 1")
    #     image = image / rearrange(self.std, "c -> 1 c 1 1")
    #     image_features = self.model.encode_image(image)
    #     image_features = image_features / image_features.norm(dim=1, keepdim=True)
    #     return image_features

    def encode_image(self, image_path):
        image = self.preprocessor(Image.open(image_path)).unsqueeze(0).to(next(self.parameters()).device)
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    # def forward(
    #     self, image_0: str, image_1: str, text_0: list[str], text_1: list[str]
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    def forward(self, image_0, image_1, text_0, text_1):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        text_features_0 = self.encode_text(text_0)
        text_features_1 = self.encode_text(text_1)
        sim_0 = F.cosine_similarity(image_features_0, text_features_0)
        sim_1 = F.cosine_similarity(image_features_1, text_features_1)
        sim_direction = F.cosine_similarity(image_features_1 - image_features_0, text_features_1 - text_features_0)
        sim_image = F.cosine_similarity(image_features_0, image_features_1)
        return sim_0, sim_1, sim_direction, sim_image

    def quantitative(self, image_0, image_1, image_1_gt, text):
        image_features_0 = self.encode_image(image_0)
        image_features_1 = self.encode_image(image_1)
        image_features_1_gt = self.encode_image(image_1_gt)
        text_features = self.encode_text(text)

        content_preserve = F.cosine_similarity(image_features_1, image_features_0)
        after_gt = F.cosine_similarity(image_features_1, image_features_1_gt)
        after_text = F.cosine_similarity(image_features_1, text_features)

        return content_preserve, after_gt, after_text
