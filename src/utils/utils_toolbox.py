import pickle
import matplotlib.pyplot as plt
import torch
from src.utils.schema import PickleSchema
import os


def save_model(
    model: torch.nn.Module,
    path: str,
    clip_config: dict,
    gpt2_config: dict,
    training_config: dict,
    mean_train_loss: float,
    lowest_val_loss: float,
    val_bert_precision: float,
    val_bert_recall: float,
    val_bert_f1_score: float,
) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the model state dictionary to a pickle file
    with open(path, "wb") as f:
        pickle.dump(
            {
                PickleSchema.CAPTION_MODEL_STATE_DICT: model.state_dict(),
                PickleSchema.CLIP_CONFIG: clip_config,
                PickleSchema.GPT2_CONFIG: gpt2_config,
                PickleSchema.TRAINING_CONFIG: training_config,
                PickleSchema.TRAIN_LOSS: mean_train_loss,
                PickleSchema.VALIDATION_LOSS: lowest_val_loss,
                PickleSchema.VALIDATION_BERT_PRECISION: val_bert_precision,
                PickleSchema.VALIDATION_BERT_RECALL: val_bert_recall,
                PickleSchema.VALIDATION_BERT_F1_SCORE: val_bert_f1_score,
            },
            f,
        )


def plot_training_progress(
    train_losses: list,
    val_losses: list,
    val_bert_precisions: list,
    val_bert_recalls: list,
    val_bert_f1_scores: list,
    path: str,
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

    # Plot 2: BERT Scores
    plt.subplot(1, 2, 2)
    plt.plot(val_bert_precisions, label="Precision", color="blue")
    plt.plot(val_bert_recalls, label="Recall", color="red")
    plt.plot(val_bert_f1_scores, label="F1", color="green")
    plt.title("BERT Scores on Validation Set")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
