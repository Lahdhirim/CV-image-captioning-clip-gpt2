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
  <em>Caption: A car is parked in front of a car dealer.</em>     <em>Caption: A view of a city street at night.</em>
</p>

<p align="center">
  <img src="data/inference_images/baby_yoda.jpg" alt="Caption 3" width="350", height="250"/>
  <img src="data/inference_images/bibi.jpg" alt="Caption 4" width="350", height="250"/>
</p>
<p align="center">
  <em>Caption: A man holding a stuffed animal in his hand.</em>     <em>Caption: A cat that is laying down on a bed.</em>
</p>

## Model Architecture (To be completed)

<div style="text-align: center;">
    <img src="assets/model_architecture.png" alt="CV" width="950", height="550"/>
</div>

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
