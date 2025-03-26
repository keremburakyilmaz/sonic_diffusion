import os
import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AudioImageDataset, get_image_transform, get_log_mel_transform
from utils.preprocess import get_caption, get_text_tokens
from src import AudioProjector, mse_loss, info_nce_loss

# Load congif file
with open("config/model_config.yaml", "r") as f:
    config = yaml.safe_load(f)

input_dim = config["input_dim"]
token_dim = config["token_dim"]
num_tokens = config["num_tokens"]
conv_channels = config["conv_channels"]
kernel_size = config["kernel_size"]

BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
LR = config["learning_rate"]
DATA_DIR = config["data_dir"]
SAVE_PATH = config["save_path"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the data
dataset = AudioImageDataset(
    root_dir=DATA_DIR,
    image_transform=get_image_transform(),
    audio_transform=get_log_mel_transform()
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initiate our AudioProjector to tokenize
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
        audio = batch["audio"].to(DEVICE)         # [B, 1, T]
        images = batch["image"]                   # PIL Image tensors

        # Prepare audio input
        audio = audio.squeeze(1)                  # [B, T]
        audio_features = audio.unsqueeze(1)       # [B, 1, T] for Conv1d

        # Text token generation (BLIP + CLIP)
        captions = [get_caption(img) for img in images]
        text_tokens = [get_text_tokens(cap) for cap in captions]
        text_tokens = torch.cat(text_tokens, dim=0).to(DEVICE)  # [B, num_tokens, 768]

        # Forward pass
        audio_tokens = model(audio_features.to(DEVICE))         # [B, num_tokens, 768]

        # Losses
        loss_mse = mse_loss(audio_tokens, text_tokens)

        anchor = audio_tokens[:, 0, :]             # [B, 768]
        positive = text_tokens[:, 0, :]            # [B, 768]
        negatives = anchor.roll(shifts=1, dims=0).unsqueeze(1)  # [B, 1, 768]

        loss_info = info_nce_loss(anchor, positive, negatives)
        loss = loss_mse + 0.25 * loss_info

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_mse += loss_mse.item()
        total_info += loss_info.item()

    # Epoch summary
    print(f"Epoch {epoch+1}: MSE={total_mse:.4f}, InfoNCE={total_info:.4f}")

    # Save model
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), SAVE_PATH)
