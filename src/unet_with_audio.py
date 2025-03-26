import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from src.gated_attention import GatedCrossAttention

class UNetWithAudioConditioning(nn.Module):
    def __init__(self, base_unet: UNet2DConditionModel, audio_dim=768):
        super().__init__()
        self.unet = base_unet
        self.attn_middle = GatedCrossAttention(hidden_dim=768, audio_dim=audio_dim)
        self.attn_out = GatedCrossAttention(hidden_dim=768, audio_dim=audio_dim)

    def forward(self, latent, t, audio_tokens):
        # Inject audio in middle block
        self.unet_mid_input_hook(audio_tokens)

        # Inject audio in output block (hooking in later layer would be better)
        self.unet_out_input_hook(audio_tokens)

        # Use dummy text tokens
        dummy_encoder_hidden_states = torch.zeros(
            latent.size(0), 77, 768, device=latent.device
        )

        return self.unet(latent, t, encoder_hidden_states = dummy_encoder_hidden_states)

    def unet_mid_input_hook(self, audio_tokens):
        mid_block = self.unet.mid_block
        mid_block.forward = self._wrap_attention(mid_block.forward, self.attn_middle, audio_tokens)

    def unet_out_input_hook(self, audio_tokens):
        last_block = self.unet.output_blocks[-1]
        last_block.forward = self._wrap_attention(last_block.forward, self.attn_out, audio_tokens)

    def _wrap_attention(self, forward_fn, attn_layer, audio_tokens):
        def wrapper(x, *args, **kwargs):
            x = forward_fn(x, *args, **kwargs)
            return attn_layer(x, audio_tokens)
        return wrapper
