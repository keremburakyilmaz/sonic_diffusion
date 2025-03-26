import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention(nn.Module):
    def __init__(self, hidden_dim, audio_dim, n_heads=8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.audio_dim = audio_dim

        self.norm = nn.LayerNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(audio_dim, hidden_dim)
        self.v_proj = nn.Linear(audio_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Learnable gate for scaling audio influence
        self.gamma = nn.Parameter(torch.tensor(0.0))  # initialized near zero

    def forward(self, hidden_states, audio_tokens, attention_mask=None):
        """
        hidden_states: [B, N, D] from UNet
        audio_tokens:  [B, T, D] from AudioProjector
        """
        B, N, D = hidden_states.shape
        T = audio_tokens.shape[1]

        x = self.norm(hidden_states)

        # Project to Q/K/V
        Q = self.q_proj(x)
        K = self.k_proj(audio_tokens)
        V = self.v_proj(audio_tokens)

        # Multi-head attention shape: [B, heads, tokens, dim_per_head]
        Q = Q.view(B, N, self.n_heads, D // self.n_heads).transpose(1, 2)
        K = K.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        V = V.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

        # Attention scores
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (D // self.n_heads) ** 0.5
        attn_weights = F.softmax(attn_weights, dim=-1)

        attended = torch.matmul(attn_weights, V)  # [B, heads, N, dim]
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)

        out = self.out_proj(attended)

        # Gated injection
        return hidden_states + torch.tanh(self.gamma) * out
