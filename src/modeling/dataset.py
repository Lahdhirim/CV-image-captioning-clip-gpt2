import json
import random
from typing import List, Tuple

import torch
from PIL import Image
from torch import device as TorchDevice
from torch.utils.data import Dataset
from transformers import CLIPModel, CLIPProcessor, PreTrainedTokenizer

from src.utils.schema import CaptionSchema


class ClipCaptionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        clip_processor: CLIPProcessor,
        clip_model: CLIPModel,
        device: TorchDevice,
        text_tokens_max_length: int,
        subset_ratio: float,
        image_size: List[int],
    ) -> None:

        with open(data_path, "r") as f:
            data = json.load(f)
        if subset_ratio < 1.0:
            random.seed(42)
            random.shuffle(data)
            keep = int(len(data) * subset_ratio)
            self.data = data[:keep]
        else:
            self.data = data

        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.device = device
        self.text_tokens_max_length = text_tokens_max_length
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        item = self.data[idx]
        caption = item[CaptionSchema.CAPTION]
        image_path = item[CaptionSchema.IMAGE]

        # Tokenize the caption
        text_tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_tokens_max_length,
        )

        # Load the image and calculate CLIP embedding
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size[0], self.image_size[1]))
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            clip_embed = self.clip_model.get_image_features(**inputs).cpu().squeeze(0)

        return text_tokens.input_ids.squeeze(0), clip_embed
