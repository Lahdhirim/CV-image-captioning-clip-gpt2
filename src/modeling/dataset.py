import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import pickle
from typing import Tuple


class ClipCaptionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
    ) -> None:

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.embeddings = data["clip_embedding"]
        self.captions = data["captions"]
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.captions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_tokens = self.tokenizer(
            self.captions[idx]["caption"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        clip_embed = torch.tensor(self.embeddings[idx]["clip_embedding"])
        return text_tokens.input_ids.squeeze(0), clip_embed
