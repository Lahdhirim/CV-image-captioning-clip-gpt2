import argparse

from src.config_loader.config_loader import load_config
from src.trainer import Trainer

from src.config_loader.inference_config_loader import load_inference_config
from src.inference import InferencePipeline

if __name__ == "__main__":

    # Parse command-line argument to determine which mode to run
    parser = argparse.ArgumentParser(description="CLIP-GPT2 Captioning Model")
    parser.add_argument(
        "mode",
        choices=["train", "inference"],
        default="train",
        nargs="?",
        help="Choose mode: train or inference",
    )
    args = parser.parse_args()

    if args.mode == "train":
        config_path = "config/training_config.json"
        config = load_config(config_path)
        Trainer(config=config).run()

    elif args.mode == "inference":
        config_path = "config/inference_config.json"
        inference_config = load_inference_config(config_path)
        InferencePipeline(config=inference_config).run()

    else:
        print("Invalid mode. Please choose 'train', 'inference'.")
