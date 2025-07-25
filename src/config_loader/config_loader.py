import json
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class ClipConfig(BaseModel):
    model: str = Field(..., description="CLIP model name")
    image_size: Optional[List[int]] = Field(
        default_factory=lambda: [224, 224],
        description="Optional image resize dimensions [H, W]",
    )

    @validator("image_size")
    def validate_image_size(cls, v):
        if len(v) != 2:
            raise ValueError("image_size must have exactly two integers")
        if not all(isinstance(i, int) and i > 0 for i in v):
            raise ValueError("image_size values must be positive integers")
        return v


class GPT2Config(BaseModel):
    model: str = Field(..., description="GPT2 model name")
    visual_tokens_length: int = Field(
        ..., description="Length of visual tokens as a prefix"
    )
    text_tokens_max_length: int = Field(
        ..., description="Max length for tokenized captions"
    )
    n_layers_to_freeze: Optional[int] = Field(
        default=None, description="Number of GPT-2 transformer layers to freeze"
    )

    @validator("n_layers_to_freeze")
    def validate_n_layers_to_freeze(cls, v):
        if v < 0:
            raise ValueError("n_layers_to_freeze must be non-negative")
        return v


class DataPaths(BaseModel):
    train_captions_path: str = Field(
        ..., description="Path to training data captions JSON"
    )
    val_captions_path: str = Field(
        ..., description="Path to validation data captions JSON"
    )


class TrainingConfig(BaseModel):
    subset_ratio: Optional[float] = Field(
        default=1.0, description="Percentage of data to use per epoch"
    )
    batch_size: int = Field(..., description="Batch size")
    num_epochs: int = Field(..., description="Number of trainig epochs")
    learning_rate: float = Field(..., description="Learning rate")
    enable_GPU: Optional[bool] = Field(
        default=False, description="Enable using GPU during training"
    )
    trained_model_path: str = Field(..., description="Path to save best trained model")
    monitoring_plots_path: str = Field(
        ..., description="Path to save training monitoring plots"
    )


class Config(BaseModel):
    clip_config: ClipConfig = Field(..., description="Configuration CLIP encoder")
    gpt2_config: GPT2Config = Field(..., description="Configuration GPT-2 decoder")
    data_paths: DataPaths = Field(..., description="Paths to data files")
    training_config: TrainingConfig = Field(
        ..., description="Configuration for training loop"
    )


def load_config(config_path: str) -> Config:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return Config(**config)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find config file: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading config: {e}")
