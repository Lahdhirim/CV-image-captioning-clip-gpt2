# Image Captioning Using CLIP and GPT-2

This project implements a deep learning pipeline for **image captioning**, combining **CLIP** as a vision encoder and **GPT-2** as a language model. The model projects CLIP visual features into GPT-2's input space to generate natural language descriptions for images. The model is finetuned on the Microsoft **COCO** dataset, which contains over 82000 images for train and 40000 images for validation.\
The best trained model achieves a high Bert scores:

<div align="center">

| Metric | Value |
| --------- | ------ |
| Precision | 0.912 |
| Recall | 0.902 |
| F1-Score | 0.906 |

</div>

## Output Examples on Real Images

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="data/inference_images/amg_c63.jpg" alt="Caption 1" width="350" height="250" style="object-fit: cover;"/>
        <br><em>Caption: A car is parked in front of a car dealer.</em>
      </td>
      <td align="center">
        <img src="data/inference_images/antibes.jpg" alt="Caption 2" width="350" height="250" style="object-fit: cover;"/>
        <br><em>Caption: A view of a city street at night.</em>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="data/inference_images/baby_yoda.jpg" alt="Caption 3" width="350" height="250" style="object-fit: cover;"/>
        <br><em>Caption: A man holding a stuffed animal in his hand.</em>
      </td>
      <td align="center">
        <img src="data/inference_images/bibi.jpg" alt="Caption 4" width="350" height="250" style="object-fit: cover;"/>
        <br><em>Caption: A cat that is laying down on a bed.</em>
      </td>
    </tr>
  </table>
</div>

## Model Architecture

The model architecture is a hybrid design that combines **CLIP** (as a visual encoder) and **GPT-2** (as a language decoder). The main idea is to project the visual embeddings extracted from an image into the input embedding space of GPT-2, allowing the language model to generate captions conditioned on visual content.

<div style="text-align: center;">
    <img src="assets/model_architecture.png" alt="CV" width="950", height="550"/>
</div>

- **CLIP Encoder**
  The input image is processed using CLIP to extract high-level semantic features via `CLIPProcessor`. The processed image is then encoded using `CLIPModel` to a vector of dimensionality 512.

- **Projection Module**
  A simple MLP (a linear layer followed by a `Tanh` activation) is used to project the CLIP embedding into a sequence of GPT-2-compatible embeddings. These are treated as *visual tokens*.

- **GPT-2 Decoder**
  The visual tokens are concatenated with text tokens (caption inputs) at the embedding level. The combined sequence is passed through GPT-2 to predict the caption. During training, the loss is only computed on the text tokens, ignoring the visual prefix.

## Training Pipeline
The training process is fully configurable through a JSON configuration file ([training_config.json](config/training_config.json)). This allows for maximum flexibility — models, paths, and hyperparameters can be changed without modifying the core codebase.

### Configuration Parameters

| **Key** | **Description** | **Recommended Value** |
|--------|------------------|------------------------------|
| `clip_config.model` | Name of the pretrained CLIP model | `"openai/clip-vit-base-patch32"` |
| `clip_config.image_size` | Optional resize dimensions for input images `[H, W]` (optional default is [224, 224]) | `[224, 224]` |
| `gpt2_config.model` | Name of the pretrained GPT-2 model | `"gpt2"` |
| `gpt2_config.visual_tokens_length` | Number of visual prefix tokens passed to GPT-2 | `5` |
| `gpt2_config.text_tokens_max_length` | Maximum number of tokens for the caption text | `10` |
| `gpt2_config.n_layers_to_freeze` | Number of GPT-2 attention layers to freeze (optional default is None — means all layers are trainable) | `10` |
| `data_paths.train_captions_path` | Path to training captions JSON file | filename generated with the script [coco_captions.py](src/utils/coco_captions.py) when `split = train`|
| `data_paths.val_captions_path` | Path to validation captions JSON file | filename generated with the script [coco_captions.py](src/utils/coco_captions.py) when `split = val` |
| `training_config.subset_ratio` | Ratio of COCO dataset to use for each epoch (optional default is 1.0 — means all dataset is used) | `0.4` |
| `training_config.batch_size` | Batch size for training | `64` |
| `training_config.num_epochs` | Number of training epochs | `10` |
| `training_config.learning_rate` | Learning rate for the optimizer | `1e-4` |
| `training_config.enable_GPU` | Whether to use GPU if available (optional default is False) | `true` |
| `training_config.trained_model_path` | File path to save the best performing model | — |
| `training_config.monitoring_plots_path` | File path to save training curves image | — |

---

## Inference Pipeline (To be completed)

## Experimentations (To be completed)

## Installation

- Clone the repository:

    ```bash
    git clone https://github.com/Lahdhirim/CV-image-captioning-clip-gpt2.git
    cd CV-image-captioning-clip-gpt2
    ```

- Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
