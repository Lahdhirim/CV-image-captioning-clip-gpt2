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
<p align="center">
  <img src="data/inference_images/amg_c63.jpg" alt="Caption 1" width="350", height="250"/>
  <img src="data/inference_images/antibes.jpg" alt="Caption 2" width="350", height="250"/>
</p>
<p align="center">
  <em>Caption: A car is parked in front of a car dealer.</em>   <em>Caption: A view of a city street at night.</em>
</p>

<p align="center">
  <img src="data/inference_images/baby_yoda.jpg" alt="Caption 3" width="350", height="250"/>
  <img src="data/inference_images/bibi.jpg" alt="Caption 4" width="350", height="250"/>
</p>
<p align="center">
  <em>Caption: A man holding a stuffed animal in his hand.</em>   <em>Caption: A cat that is laying down on a bed.</em>
</p>

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

## Train Pipeline (To be completed)

## Inference Pipeline (To be completed)

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
