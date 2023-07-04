import torch
import torch.nn as nn
import torch.nn.functional as F

class Classification(nn.Module):
    def __init__(self, token_size: int, d_model: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.linear = nn.Linear(in_features=d_model, out_features=token_size)
    def forward(self, x: torch.Tensor):
        x = self.norm(x)
        x = self.linear(x)
        x = F.gelu(x)
        return x