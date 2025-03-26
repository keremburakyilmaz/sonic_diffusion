from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPTokenizer, CLIPTextModel
from PIL import Image
import torch

# Load BLIP from local
blip_model = BlipForConditionalGeneration.from_pretrained("pretrained/blip")
blip_processor = BlipProcessor.from_pretrained("pretrained/blip")

# Load CLIP from local
clip_tokenizer = CLIPTokenizer.from_pretrained("pretrained/clip")
clip_text_model = CLIPTextModel.from_pretrained("pretrained/clip")

def get_caption(image: Image.Image) -> str:
# Generate a caption from an input image using BLIP.

    inputs = blip_processor(image, return_tensors="pt")
    with torch.no_grad():
        output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

def get_text_tokens(caption: str) -> torch.Tensor:
# Tokenize and encode a caption using CLIP's text encoder.

    inputs = clip_tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = clip_text_model(**inputs)
    return output.last_hidden_state
