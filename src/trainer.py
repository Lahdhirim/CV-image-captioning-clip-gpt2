import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from src.dataset.dataset import ClipCaptionDataset
from src.modeling.model import ClipCaptionModel


def train() -> None:

    # [MEDIUM] : add GPU calculation as option in config file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # [MEDIUM] : add clip model to config file
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = ClipCaptionDataset(
        data_path="data/coco_train_captions_processed.json",
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
        device=device,
    )
    # [MEDIUM] : add batch size and epoch in config file
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = ClipCaptionModel(
        clip_emb_dim=512,
        visual_tokens_length=10,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(2):

        loop = tqdm(dataloader, desc=f"Epoch {epoch}")
        for text_tokens, clip_embed in loop:

            text_tokens, clip_embed = text_tokens.to(device), clip_embed.to(device)

            outputs = model(
                text_tokens=text_tokens, clip_embed=clip_embed, labels=text_tokens
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

            # [HIGH] : add model validation and save the best model in pkl file

            # [HIGH] : add training monitoring (losses and metrics) and save plots
