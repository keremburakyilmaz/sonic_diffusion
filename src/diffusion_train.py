import torch
from diffusers import StableDiffusionPipeline
from src.audio_projector import AudioProjector
from src.unet_with_audio import UNetWithAudioConditioning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
