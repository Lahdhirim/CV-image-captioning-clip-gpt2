import json
from typing import Optional
from pydantic import BaseModel, Field, validator
from pathlib import Path


class Config(BaseModel):
    enable_GPU: Optional[bool] = Field(
        default=False, description="Enable GPU during Inference"
    )
    images_dir: str = Field(
        ..., description="Path to the directory containing images for Inference"
    )
    trained_models_dir: str = Field(
        ..., description="Path to the directory containing trained models"
    )
    output_filename: str = Field(
        ..., description="Path to the output filename for Inference results"
    )

    @validator("images_dir")
    def validate_images_dir(cls, v):
        path = Path(v)
        if not path.is_dir():
            raise ValueError(f"{v} is not a valid directory.")
        image_files = (
            list(path.glob("*.jpg"))
            + list(path.glob("*.png"))
            + list(path.glob("*.jpeg"))
        )
        if not image_files:
            raise ValueError(f"No image files found in {v}.")
        return v

    @validator("trained_models_dir")
    def validate_models_dir(cls, v):
        path = Path(v)
        if not path.is_dir():
            raise ValueError(f"{v} is not a valid directory.")
        model_files = list(path.glob("*.pkl"))
        if not model_files:
            raise ValueError(f"No model (.pkl) files found in {v}.")
        return v


def load_inference_config(config_path: str) -> Config:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return Config(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
