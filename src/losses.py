import torch
import torch.nn.functional as F

def mse_loss(audio_tokens, text_tokens):
# aligns audio tokens to text tokens extracted from the image caption
    return F.mse_loss(audio_tokens, text_tokens)

def info_nce_loss(a0, a1, negatives, temperature = 0.07):
# makes sure that similar sounds stay close in latent space, while different ones are pushed apart

    # Normalize
    a0 = F.normalize(a0, dim=-1)
    a1 = F.normalize(a1, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Compute similarities
    pos_sim = torch.exp(torch.sum(a0 * a1, dim=-1) / temperature)  # shape [B]
    neg_sim = torch.exp(torch.bmm(negatives, a0.unsqueeze(-1)).squeeze(-1) / temperature)  # shape [B, N]

    denom = pos_sim + neg_sim.sum(dim=-1)  # shape [B]
    loss = -torch.log(pos_sim / denom)  # shape [B]
    return loss.mean()
