import os
import subprocess
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AudioImageDataset, get_image_transform, get_log_mel_transform
from utils.preprocess import get_caption, get_text_tokens
from src import AudioProjector, mse_loss, info_nce_loss

# Run Optuna and get the best config
subprocess.run(["python", "src/tune.py"], check=True)
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Model hyperparameters
input_dim = config["input_dim"]
token_dim = config["token_dim"]
num_tokens = config["num_tokens"]
conv_channels = config["conv_channels"]
kernel_size = config["kernel_size"]

# Training settings
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
DATA_DIR = config["data_dir"]
SAVE_PATH = config["save_path"]

# Optional tuning weights
MSE_WEIGHT = config.get("mse_weight", 1.0)
INFO_WEIGHT = config.get("info_weight", 0.25)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
dataset = AudioImageDataset(
    root_dir=DATA_DIR,
    image_transform=get_image_transform(),
    audio_transform=get_log_mel_transform()
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize AudioProjector to tokenize audio
model = AudioProjector(
    input_dim=input_dim,
    token_dim=token_dim,
    num_tokens=num_tokens,
    conv_channels=conv_channels,
    kernel_size=kernel_size
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Train loop
for epoch in range(EPOCHS):
    model.train()
    total_mse, total_info = 0.0, 0.0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        audio = batch["audio"].to(DEVICE)
        images = batch["image"]

        audio = audio.squeeze(1)
        audio_features = audio.unsqueeze(1)

        captions = [get_caption(img) for img in images]
        text_tokens = [get_text_tokens(cap) for cap in captions]
        text_tokens = torch.cat(text_tokens, dim=0).to(DEVICE)

        audio_tokens = model(audio_features)
        loss_mse = mse_loss(audio_tokens, text_tokens)

        anchor = audio_tokens[:, 0, :]
        positive = text_tokens[:, 0, :]
        negatives = anchor.roll(shifts=1, dims=0).unsqueeze(1)

        loss_info = info_nce_loss(anchor, positive, negatives)
        loss = MSE_WEIGHT * loss_mse + INFO_WEIGHT * loss_info

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += loss_mse.item()
        total_info += loss_info.item()

    print(f"Epoch {epoch+1}: MSE={total_mse:.4f}, InfoNCE={total_info:.4f}")

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
