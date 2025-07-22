import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import json
from typing import Tuple
from PIL import Image


class ClipCaptionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        clip_processor,
        clip_model,
        device: str = "cpu",
    ) -> None:

        with open(data_path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        item = self.data[idx]
        caption = item["caption"]
        image_path = item["image"]

        # Tokenize the caption
        text_tokens = self.tokenizer(
            caption,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=30,
        )

        # Load the image and calculate CLIP embedding
        image = Image.open(image_path).convert("RGB")
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            clip_embed = self.clip_model.get_image_features(**inputs).cpu().squeeze(0)

        return text_tokens.input_ids.squeeze(0), clip_embed
