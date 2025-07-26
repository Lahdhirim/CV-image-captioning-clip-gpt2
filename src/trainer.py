import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor, GPT2LMHeadModel, GPT2Tokenizer
import datetime
from src.config_loader.config_loader import Config
from src.evaluators.bert_evaluator import semantic_similarity
from src.modeling.dataset import ClipCaptionDataset
from src.modeling.model import ClipCaptionModel
from src.utils.logging_config import logger
from src.utils.utils_toolbox import plot_training_progress, save_model


class Trainer:

    def __init__(self, config: Config):
        self.clip_config = config.clip_config
        self.gpt2_config = config.gpt2_config
        self.data_paths = config.data_paths
        self.training_config = config.training_config
        self.logger = logger

    def run(self) -> None:

        # Set up logging
        with open("logs/training.log", "w") as log_file:
            pass
        logger.info("Logging setup complete")
        logger.info(f"clip_config: {self.clip_config}")
        logger.info(f"gpt2_config: {self.gpt2_config}")
        logger.info(f"data_paths: {self.data_paths}")
        logger.info(f"training_config: {self.training_config}")

        # Select device
        if self.training_config.enable_GPU:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")
        logger.info(f"Selected device: {device}")

        # Initialize CLIP components
        clip_model = CLIPModel.from_pretrained(self.clip_config.model).to(device)
        clip_processor = CLIPProcessor.from_pretrained(self.clip_config.model)
        logger.info(f"CLIP model initialized using {self.clip_config.model}")

        # Initialize GPT-2 components
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_config.model)
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_model = GPT2LMHeadModel.from_pretrained(self.gpt2_config.model)
        logger.info(f"GPT-2 model initialized using {self.gpt2_config.model}")

        # Define DataLoaders
        train_dataset = ClipCaptionDataset(
            data_path=self.data_paths.train_captions_path,
            tokenizer=gpt2_tokenizer,
            clip_processor=clip_processor,
            clip_model=clip_model,
            device=device,
            text_tokens_max_length=self.gpt2_config.text_tokens_max_length,
            subset_ratio=self.training_config.subset_ratio,
            image_size=self.clip_config.image_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.training_config.batch_size, shuffle=True
        )
        logger.info("Train DataLoader initialized")

        val_dataset = ClipCaptionDataset(
            data_path=self.data_paths.val_captions_path,
            tokenizer=gpt2_tokenizer,
            clip_processor=clip_processor,
            clip_model=clip_model,
            device=device,
            text_tokens_max_length=self.gpt2_config.text_tokens_max_length,
            subset_ratio=self.training_config.subset_ratio,
            image_size=self.clip_config.image_size,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.training_config.batch_size, shuffle=True
        )
        logger.info("Validation DataLoader initialized")

        # Define the main model
        # [LOW]: load clip_emb_dim dynamically
        model = ClipCaptionModel(
            clip_emb_dim=512,
            visual_tokens_length=self.gpt2_config.visual_tokens_length,
            gpt2_model=gpt2_model,
            n_layers_to_freeze=self.gpt2_config.n_layers_to_freeze,
        ).to(device)
        logger.info("Main model initialized")

        # Define the optimizer
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(
            trainable_params, lr=self.training_config.learning_rate
        )
        # Log the names of the trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.debug(
                    f"Trainable parameter: {name} - shape: {tuple(param.shape)}"
                )

        # Create a unique timestamp to save the model and plots during training
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize the lowest validation loss and training/val losses lists
        # [LOW]: add perf metrics before training starts
        lowest_val_loss = None
        train_losses, val_losses = [], []
        val_bert_precisions = []
        val_bert_recalls = []
        val_bert_f1_scores = []
        logger.info(f"Starting training loop...")

        for epoch in range(self.training_config.num_epochs):
            logger.info(f"#### Epoch {epoch} started ####")

            model.train()
            total_train_loss = 0

            train_loop = tqdm(train_loader, desc=f"Epoch {epoch}")
            for text_tokens, clip_embed in train_loop:

                text_tokens, clip_embed = text_tokens.to(device), clip_embed.to(device)

                outputs = model(
                    text_tokens=text_tokens, clip_embed=clip_embed, labels=text_tokens
                )

                loss = outputs.loss
                total_train_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loop.set_postfix(loss=loss.item())

            mean_train_loss = total_train_loss / len(train_loader)
            train_losses.append(mean_train_loss)
            print(f"Train Loss: {mean_train_loss:.4f}")
            logger.info(f"Train Loss: {mean_train_loss:.4f}")

            # Evaluate the model on validation set
            model.eval()
            total_val_loss = 0

            predictions = []
            references = []

            with torch.no_grad():
                val_loop = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
                for text_tokens, clip_embed in val_loop:
                    text_tokens, clip_embed = text_tokens.to(device), clip_embed.to(
                        device
                    )

                    # Caculate validation loss
                    outputs = model(
                        text_tokens=text_tokens,
                        clip_embed=clip_embed,
                        labels=text_tokens,
                    )
                    total_val_loss += outputs.loss.item()

                    # Generate predictions to calculate Bert Score
                    # [LOW]: add max_length and num_beams to config file under gpt2_config
                    generated_ids = model.generate(
                        clip_embed=clip_embed,
                        max_length=20,
                        num_beams=5,
                        early_stopping=True,
                        pad_token_id=gpt2_tokenizer.pad_token_id,
                    )
                    # Decode predictions and references
                    decoded_preds = gpt2_tokenizer.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )
                    decoded_refs = gpt2_tokenizer.batch_decode(
                        text_tokens, skip_special_tokens=True
                    )
                    predictions.extend(decoded_preds)
                    references.extend(decoded_refs)

                mean_val_loss = total_val_loss / len(val_loader)
                val_losses.append(mean_val_loss)
                print(f"Validation Loss: {mean_val_loss:.4f}")
                logger.info(f"Validation Loss: {mean_val_loss:.4f}")

                val_bert_precision, val_bert_recall, val_bert_f1_score = (
                    semantic_similarity(predictions, references)
                )

                val_bert_precisions.append(val_bert_precision)
                print(f"Validation Bert Precision: {val_bert_precision:.4f}")
                logger.info(f"Validation Bert Precision: {val_bert_precision:.4f}")

                val_bert_recalls.append(val_bert_recall)
                print(f"Validation Bert Recall: {val_bert_recall:.4f}")
                logger.info(f"Validation Bert Recall: {val_bert_recall:.4f}")

                val_bert_f1_scores.append(val_bert_f1_score)
                print(f"Validation Bert F1 Score: {val_bert_f1_score:.4f}")
                logger.info(f"Validation Bert F1 Score: {val_bert_f1_score:.4f}")

                # Save the model with the lowest validation loss
                if lowest_val_loss is None or mean_val_loss < lowest_val_loss:
                    lowest_val_loss = mean_val_loss
                    save_model(
                        model=model,
                        path=self.training_config.trained_model_path.replace(
                            ".pkl", f"_{timestamp}.pkl"
                        ),
                        clip_config=self.clip_config,
                        gpt2_config=self.gpt2_config,
                        training_config=self.training_config,
                        mean_train_loss=mean_train_loss,
                        lowest_val_loss=lowest_val_loss,
                        val_bert_precision=val_bert_precision,
                        val_bert_recall=val_bert_recall,
                        val_bert_f1_score=val_bert_f1_score,
                    )
                    print(
                        f"Model saved with lowest validation loss: {lowest_val_loss:.4f} | "
                        f"BERT Precision: {val_bert_precision:.4f} | "
                        f"BERT Recall: {val_bert_recall:.4f} | "
                        f"BERT F1 Score: {val_bert_f1_score:.4f}"
                    )

                    logger.info(
                        f"Model saved with lowest validation loss: {lowest_val_loss:.4f} | "
                        f"BERT Precision: {val_bert_precision:.4f} | "
                        f"BERT Recall: {val_bert_recall:.4f} | "
                        f"BERT F1 Score: {val_bert_f1_score:.4f}"
                    )

                # Save the training and validation loss curve after each epoch (to keep track of progress)
                if len(train_losses) >= 2:
                    plot_training_progress(
                        train_losses=train_losses,
                        val_losses=val_losses,
                        val_bert_precisions=val_bert_precisions,
                        val_bert_recalls=val_bert_recalls,
                        val_bert_f1_scores=val_bert_f1_scores,
                        path=self.training_config.monitoring_plots_path.replace(
                            ".png", f"_{timestamp}.png"
                        ),
                    )
                    print(
                        f"Training and validation loss curves saved to {self.training_config.monitoring_plots_path}"
                    )
                    logger.info(
                        f"Training and validation loss curves saved to {self.training_config.monitoring_plots_path}"
                    )

        # [ENHANCEMENT]: add a function that updates a CSV file to ouput config, epoch duration and performance metrics across all runs (Synthesis file)
