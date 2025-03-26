# SonicDiffusion: Generating Images from Sound

This project implements **audio-conditioned image generation and editing** using pretrained diffusion models. Inspired by the [SonicDiffusion](https://arxiv.org/abs/2405.00878) paper, we extend the Stable Diffusion pipeline by injecting audio features into the image generation process via gated cross-attention.

---

## Project Structure

```
sonic_diffusion/
├── README.md
├── requirements.txt
├── dataset
├── download_pretrained.py       # Python script to download necessary pretrained models.
├── config/
│   └── model_config.yaml        # Model architecture, training hyperparameters, etc.
├── src/
│   ├── __init__.py
│   ├── audio_projector.py       # Audio encoder that maps raw audio features into token embeddings
│   ├── gated_attention.py       # Gated cross-attention layers to inject audio into the UNet
│   ├── diffusion_model.py       # Modified Stable Diffusion model with audio conditioning
│   ├── losses.py                # Custom loss functions (e.g., InfoNCE, MSE for projector training)
│   ├── train.py                 # Training pipeline (Stage 1: Audio Projector, Stage 2: Diffusion)
|   ├── tune.py                  # Optimizing hyperparameters using Optuna.
│   └── inference.py             # Image generation and editing scripts for inference
└── utils/
    ├── __init__.py
    ├── data_loader.py           # Audio/image dataset loading and preprocessing
    ├── preprocess.py            # Preprocessing the data (getting captions and tokenizing the captions)
    └── transforms.py            # Audio and image transformations (spectrograms, normalization, etc.)
```

---

## Key Components

### `audio_projector.py`
- Maps audio inputs (e.g., log-mel spectrograms or CLAP embeddings) into a sequence of latent tokens.
- Outputs token embeddings compatible with the Stable Diffusion UNet’s cross-attention layers.

### `gated_attention.py`
- Gated cross-attention mechanism that injects audio information into the decoder blocks of the UNet.
- Gradually learns how much influence audio should have during generation.

### `diffusion_model.py`
- Wraps the Stable Diffusion architecture and integrates the audio projector and gated attention.
- Supports conditioning on audio tokens (and optionally text) for multimodal generation.

### `losses.py`
- Implements:
  - InfoNCE loss: for contrastive learning of audio semantics.
  - MSE loss: for aligning audio tokens with text embeddings.
  - DDPM loss: for training the denoising network in the diffusion model.

### `train.py`
- Stage 1: Trains the audio projector to align audio and text semantics.
- Stage 2: Fine-tunes the new gated attention layers in the diffusion UNet.

### `inference.py`
- Loads trained models to generate or edit images from audio clips.
- Supports optional text prompts for stylized generation.

---

## Setup

```bash
git clone https://github.com/yourusername/sonic_diffusion.git
cd sonic_diffusion
pip install -r requirements.txt
python download_pretrained.py
```

Update `config/model_config.yaml` with the paths to your datasets and model settings.

---

## Future Work

- Add real-time audio input implementation.
- Extend to audio-to-video generation

---

## Citation

If you use this repo or base your work on it, please cite the original [SonicDiffusion paper](https://arxiv.org/abs/2405.00878) and this implementation.
