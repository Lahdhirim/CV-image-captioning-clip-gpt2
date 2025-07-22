import pickle
import matplotlib.pyplot as plt


def save_model(model, clip_model, tokenizer, path="trained_models/best_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump(
            {
                "caption_model_state_dict": model.state_dict(),
                "clip_model_state_dict": clip_model.state_dict(),
                "tokenizer": tokenizer,
            },
            f,
        )


def plot_training_progress(train_losses: list, val_losses: list, val_bert_scores: list):

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
    plt.savefig("figs/training_plots.png")
    plt.close()
