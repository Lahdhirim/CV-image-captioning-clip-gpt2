import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel
from tqdm import tqdm
from src.dataset.dataset import ClipCaptionDataset
from src.modeling.model import ClipCaptionModel
from src.utils.utils_toolbox import save_model, plot_training_progress
from src.evaluators.bert_evaluator import semantic_similarity


def train() -> None:

    # [MEDIUM] : add GPU calculation as option in config file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # [MEDIUM] : add clip model to config file
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_dataset = ClipCaptionDataset(
        data_path="data/coco_train_captions_processed.json",
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
        device=device,
    )
    # [MEDIUM] : add batch size and epoch in config file
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = ClipCaptionDataset(
        data_path="data/coco_val_captions_processed.json",
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        clip_model=clip_model,
        device=device,
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    model = ClipCaptionModel(
        clip_emb_dim=512,
        visual_tokens_length=10,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    lowest_val_loss = None
    train_losses, val_losses = [], []
    val_bert_scores = []

    for epoch in range(2):

        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for text_tokens, clip_embed in loop:

            text_tokens, clip_embed = text_tokens.to(device), clip_embed.to(device)

            outputs = model(
                text_tokens=text_tokens, clip_embed=clip_embed, labels=text_tokens
            )

            loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        mean_train_loss = total_train_loss / len(train_loader)
        train_losses.append(mean_train_loss)
        print(f"Train Loss: {mean_train_loss:.4f}")

        # Evaluate the model on validation set
        model.eval()
        total_val_loss = 0

        predictions = []
        references = []

        with torch.no_grad():
            for text_tokens, clip_embed in val_loader:
                text_tokens, clip_embed = text_tokens.to(device), clip_embed.to(device)

                # Caculate validation loss
                outputs = model(
                    text_tokens=text_tokens, clip_embed=clip_embed, labels=text_tokens
                )
                total_val_loss += outputs.loss.item()

                # Generate predictions to calculate Bert Score
                generated_ids = model.generate(
                    clip_embed=clip_embed,
                    max_length=30,
                    num_beams=5,
                    early_stopping=True,
                )

                # Decode predictions and references
                decoded_preds = tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                decoded_refs = tokenizer.batch_decode(
                    text_tokens, skip_special_tokens=True
                )

                predictions.extend(decoded_preds)
                references.extend(decoded_refs)

            mean_val_loss = total_val_loss / len(val_loader)
            val_losses.append(mean_val_loss)
            print(f"Validation Loss: {mean_val_loss:.4f}")

            val_bert_score = semantic_similarity(predictions, references)
            val_bert_scores.append(val_bert_score)
            print(f"Validation Bert Score: {val_bert_score:.4f}")

            # Save the model with the lowest validation loss
            if lowest_val_loss is None or mean_val_loss < lowest_val_loss:
                lowest_val_loss = mean_val_loss
                save_model(
                    model, clip_model, tokenizer, path="trained_models/best_model.pkl"
                )

            # Save the training and validation loss curve after each epoch (to keep track of progress)
            if len(train_losses) >= 2:
                plot_training_progress(train_losses, val_losses, val_bert_scores)
