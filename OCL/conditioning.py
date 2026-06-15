import torch
from torch import nn


class LangConditioning_v2(nn.Module):
    """Creates slot initializers from biomarker name embeddings."""

    def __init__(self, object_dim: int, lang_dim: int, n_slots: int, dual_conditioning: bool = False):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim
        self.language = nn.Linear(lang_dim, object_dim)
        self.lang_norm = nn.LayerNorm(object_dim)
        self.slots = nn.Parameter(0.01 * torch.randn(1, n_slots, object_dim))
        self.lang_gate = nn.Parameter(torch.zeros(1, n_slots, 1))
        self.dual_conditioning = dual_conditioning

    def forward(self, name_embedding: torch.Tensor, mask: torch.Tensor, batch_size: int):
        slots = self.slots.expand(batch_size, -1, -1)
        sample_lang = self.lang_norm(self.language(name_embedding.float()))

        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(batch_size, -1)
        mask = mask.to(sample_lang.device).float()

        if self.dual_conditioning:
            return sample_lang, sample_lang

        gate = torch.sigmoid(self.lang_gate).expand(batch_size, -1, -1)
        conditioning = slots + gate * sample_lang * mask.unsqueeze(-1)
        return conditioning, sample_lang
