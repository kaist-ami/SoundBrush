from __future__ import annotations

import json
import math
import pdb
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
import torchaudio
import os
from transformers import AutoProcessor
from tqdm import tqdm



class EditDataset(Dataset):
    def __init__(
            self,
            path: str,
            split: str = "train",
            # splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
            splits: tuple[float, float, float] = (1, 0.00, 0.00),
            min_resize_res: int = 256,
            max_resize_res: int = 256,
            crop_res: int = 256,
            flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        self.input_length = 5

        self.imgs_path = os.path.join(self.path, "imgs")
        self.wav_path = os.path.join(self.path, "wavs")

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        self.seeds = self.seeds[:]
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self) -> int:
        return len(self.seeds)


    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]

        # load wav_file
        wav_file = os.path.join(self.wav_path, name)
        wav_prompt = wav_file

        prompt = "Add <*>"
        category = seeds[2]

        image_0 = Image.open(os.path.join(self.imgs_path, seeds[0]))
        image_1 = Image.open(os.path.join(self.imgs_path, seeds[1]))

        reize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((reize_res, reize_res), Image.Resampling.LANCZOS)
        visual_feature = self.processor(images=image_1, return_tensors="pt").pixel_values

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        new_dict = {"edited": image_1,
                    "edit": {"c_concat": image_0, "c_crossattn": prompt, "c_wav": wav_prompt, "category": category,
                             "visual_feature": visual_feature}}
        return new_dict

