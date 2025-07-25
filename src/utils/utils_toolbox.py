import pickle

import matplotlib.pyplot as plt
import torch

from src.utils.schema import PickleSchema


def save_model(
    model: torch.nn.Module, path: str, clip_config: dict, gpt2_config: dict
) -> None:
    with open(path, "wb") as f:
        pickle.dump(
            {
                PickleSchema.CAPTION_MODEL_STATE_DICT: model.state_dict(),
                PickleSchema.CLIP_CONFIG: clip_config,
                PickleSchema.GPT2_CONFIG: gpt2_config,
            },
            f,
        )


def plot_training_progress(
    train_losses: list, val_losses: list, val_bert_scores: list, path: str
):

    plt.figure(figsize=(12, 5))

    # Plot 1: Losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot 2: BERTScore F1
    plt.subplot(1, 2, 2)
    plt.plot(val_bert_scores, label="Val BERTScore F1", color="green")
    plt.title("BERTScore F1 on Validation Set")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
