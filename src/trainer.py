import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, PreTrainedTokenizer
from tqdm import tqdm
from typing import Dict, Any
from src.modeling.dataset import ClipCaptionDataset
from src.modeling.model import ClipCaptionModel


def train() -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    dataset = ClipCaptionDataset(data_path="data/captions.json", tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = ClipCaptionModel(
        clip_emb_dim=512,
        visual_tokens_length=10,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(2):

        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for tokens, prefix in loop:

            tokens, prefix = tokens.to(device), prefix.to(device)

            outputs = model(tokens, prefix, labels=tokens)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())
