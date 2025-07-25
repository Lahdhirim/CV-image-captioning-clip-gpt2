from src.config_loader.config_loader import load_config
from src.trainer import Trainer

if __name__ == "__main__":
    config_path = "config/config.json"
    config = load_config(config_path)
    Trainer(config=config).run()
