import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioProjector(nn.Module):
    def __init__(self, input_dim, token_dim = 768, num_tokens = 77, conv_channels = 512, kernel_size = 3):
        super(AudioProjector, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size // 2)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(conv_channels, token_dim * num_tokens, kernel_size, padding=kernel_size // 2)

        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.self_attn_layer = nn.TransformerEncoderLayer(d_model = token_dim, nhead = 8)
        self.transformer_encoder = nn.TransformerEncoder(self.self_attn_layer, num_layers = 1)

    def forward(self, audio_features):
        x = self.conv1(audio_features)
        x = self.activation(x)
        x = self.conv2(x)
        x = x.mean(dim=-1)
        tokens = x.view(x.size(0), self.num_tokens, self.token_dim)

        tokens = tokens.transpose(0, 1)
        tokens = self.transformer_encoder(tokens)
        return tokens.transpose(0, 1)