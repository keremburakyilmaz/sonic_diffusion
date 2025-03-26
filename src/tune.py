import os
import yaml
import torch
import optuna
from torch.utils.data import DataLoader
from utils import AudioImageDataset, get_image_transform, get_log_mel_transform
from utils.preprocess import get_caption, get_text_tokens
from src import AudioProjector, mse_loss, info_nce_loss

# Load config files
config_path = "config/model_config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

DATA_DIR = config["data_dir"]
DEVICE = config["device"]

# Optuna objective function
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    conv_channels = trial.suggest_categorical("conv_channels", [128, 256, 512])
    num_tokens = trial.suggest_int("num_tokens", 32, 128, step=16)
    mse_weight = trial.suggest_float("mse_weight", 0.5, 1.5)
    info_weight = trial.suggest_float("info_weight", 0.0, 1.0)

    # Load data
    dataset = AudioImageDataset(
        root_dir = DATA_DIR,
        image_transform = get_image_transform(),
        audio_transform = get_log_mel_transform()
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Init model
    model = AudioProjector(
        input_dim = 64,
        token_dim = 768,
        num_tokens = num_tokens,
        conv_channels = conv_channels,
        kernel_size = 3
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_loss = 0

    for step, batch in enumerate(dataloader):
        if step > 10:  # mini-training run
            break

        audio = batch["audio"].to(DEVICE)
        images = batch["image"]
        audio = audio.squeeze(1)
        audio_features = audio.unsqueeze(1)

        captions = [get_caption(img) for img in images]
        text_tokens = [get_text_tokens(cap) for cap in captions]
        text_tokens = torch.cat(text_tokens, dim=0).to(DEVICE)

        audio_tokens = model(audio_features)
        loss_mse_val = mse_loss(audio_tokens, text_tokens)

        anchor = audio_tokens[:, 0, :]
        positive = text_tokens[:, 0, :]
        negatives = anchor.roll(shifts=1, dims=0).unsqueeze(1)

        loss_info_val = info_nce_loss(anchor, positive, negatives)

        total = mse_weight * loss_mse_val + info_weight * loss_info_val

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        total_loss += total.item()

    return total_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

config.update(best_params)
with open(config_path, "w") as f:
    yaml.dump(config, f)
print(f"model_config.yaml updated with best parameters.")