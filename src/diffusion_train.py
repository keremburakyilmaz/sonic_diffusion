import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from diffusers import StableDiffusionPipeline
from utils import AudioImageDataset, get_image_transform, get_log_mel_transform
from src.audio_projector import AudioProjector
from src.unet_with_audio import UNetWithAudioConditioning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5

# Load pretrained components from disk
PRETRAINED_PATH = "pretrained/stable_diffusion"

pipe = StableDiffusionPipeline.from_pretrained(PRETRAINED_PATH).to(DEVICE)
unet = UNetWithAudioConditioning(pipe.unet).to(DEVICE)
vae = pipe.vae
text_encoder = pipe.text_encoder
tokenizer = pipe.tokenizer

# Load trained AudioProjector
audio_projector = AudioProjector(
    input_dim=64, token_dim=768, num_tokens=77
)
audio_projector.load_state_dict(torch.load("checkpoints/audio_projector.pt"))
audio_projector = audio_projector.to(DEVICE).eval()

# Set up dataset & scheduler
dataset = AudioImageDataset(
    root_dir="your_dataset",
    image_transform=get_image_transform(),
    audio_transform=get_log_mel_transform()
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Optimizer (only train attention layers)
params_to_train = list(unet.attn_middle.parameters()) + list(unet.attn_out.parameters())
optimizer = torch.optim.Adam(params_to_train, lr=1e-4)

# Training Loop
for epoch in range(EPOCHS):
    for batch in dataloader:
        audio = batch["audio"].to(DEVICE)
        images = batch["image"].to(DEVICE)

        # 1. Encode image to latent space
        latents = vae.encode(images * 2.0 - 1.0).latent_dist.sample() * 0.18215  # scaling factor from SD

        # 2. Add noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=DEVICE).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # 3. Project audio to tokens
        audio = audio.squeeze(1)
        audio_features = audio.unsqueeze(1)
        audio_tokens = audio_projector(audio_features)

        # 4. Predict noise using UNet conditioned on audio
        noise_pred = unet(noisy_latents, timesteps, audio_tokens)

        # 5. Loss + Backprop
        loss = F.mse_loss(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")