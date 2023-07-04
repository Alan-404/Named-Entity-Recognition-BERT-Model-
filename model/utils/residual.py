import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualConnection(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.dropout_rate = dropout_rate
    def forward(self, x: torch.Tensor, pre_x: torch.Tensor):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x += pre_x
        x = self.norm(x)
        return x