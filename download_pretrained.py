from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPTokenizer,
    CLIPTextModel,
)
import os
from diffusers import StableDiffusionPipeline


def download_and_save_blip(save_path="pretrained/blip"):
    print("Downloading BLIP...")
    os.makedirs(save_path, exist_ok=True)

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    processor.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"BLIP saved to {save_path}")

def download_and_save_clip(save_path="pretrained/clip"):
    print("Downloading CLIP...")
    os.makedirs(save_path, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    print(f"CLIP saved to {save_path}")

def download_and_save_stable_diffusion(save_path="pretrained/stable_diffusion"):
    print("Downloading Stable Diffusion (v1.4)...")
    os.makedirs(save_path, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
    pipe.save_pretrained(save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    download_and_save_blip()
    download_and_save_clip()
    download_and_save_stable_diffusion()
    print("All models downloaded and saved.")
