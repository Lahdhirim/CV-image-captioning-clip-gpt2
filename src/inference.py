import pickle
import torch
from pathlib import Path
import csv
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
from src.modeling.model import ClipCaptionModel
from src.config_loader.inference_config_loader import Config
from src.utils.schema import PickleSchema


class InferencePipeline:
    def __init__(self, config: Config):
        self.enable_GPU = config.enable_GPU
        self.images_dir = config.images_dir
        self.trained_models_dir = config.trained_models_dir
        self.output_filename = config.output_filename

    def run(self) -> None:

        # Select device
        if self.enable_GPU:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        print("Using device:", device)

        # Prepare image paths
        image_paths = (
            list(Path(self.images_dir).glob("*.jpg"))
            + list(Path(self.images_dir).glob("*.png"))
            + list(Path(self.images_dir).glob("*.jpeg"))
        )

        # Prepare models
        model_paths = list(Path(self.trained_models_dir).glob("*.pkl"))
        loaded_models = {}
        for model_path in model_paths:
            with open(model_path, "rb") as f:
                checkpoint = pickle.load(f)

            # Load tokenizer and GPT2
            decoder = checkpoint[PickleSchema.GPT2_CONFIG].model
            tokenizer = GPT2Tokenizer.from_pretrained(decoder)
            tokenizer.pad_token = tokenizer.eos_token
            gpt2_model = GPT2LMHeadModel.from_pretrained(decoder).to(device)

            # Load CLIP
            encoder = checkpoint[PickleSchema.CLIP_CONFIG].model
            clip_model = CLIPModel.from_pretrained(encoder).to(device)
            clip_processor = CLIPProcessor.from_pretrained(encoder)

            # Caption model
            caption_model = ClipCaptionModel(
                clip_emb_dim=512,
                visual_tokens_length=checkpoint[
                    PickleSchema.GPT2_CONFIG
                ].visual_tokens_length,
                gpt2_model=gpt2_model,
            )
            caption_model.load_state_dict(
                checkpoint[PickleSchema.CAPTION_MODEL_STATE_DICT]
            )
            caption_model = caption_model.to(device)
            caption_model.eval()

            # Store everything
            loaded_models[model_path.name] = {
                "caption_model": caption_model,
                "clip_model": clip_model,
                "clip_processor": clip_processor,
                "tokenizer": tokenizer,
                "image_size": checkpoint[PickleSchema.CLIP_CONFIG].image_size,
            }

        # Initialize output
        results = []

        # Loop over each image path
        for image_path in image_paths:

            image = Image.open(image_path).convert("RGB")

            for model_name, model_bundle in loaded_models.items():

                caption_model = model_bundle["caption_model"]
                clip_model = model_bundle["clip_model"]
                clip_processor = model_bundle["clip_processor"]
                tokenizer = model_bundle["tokenizer"]
                image_size = model_bundle["image_size"]

                # Resize and process the image
                image_resized = image.resize(tuple(image_size))
                inputs = clip_processor(images=image_resized, return_tensors="pt").to(
                    device
                )

                # Get the image embeddings and generate caption
                with torch.no_grad():
                    clip_embed = clip_model.get_image_features(**inputs)

                    generated_ids = caption_model.generate(
                        clip_embed=clip_embed,
                        max_length=30,
                        num_beams=5,
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    caption = tokenizer.decode(
                        generated_ids[0], skip_special_tokens=True
                    )

                results.append(
                    {
                        "image_id": image_path.name,
                        "model_name": model_name,
                        "caption": caption,
                    }
                )
                print(f"{image_path.name} | {model_name} → {caption}")

        # Write output to CSV file
        with open(self.output_filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["image_id", "model_name", "caption"]
            )
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {self.output_filename}")
