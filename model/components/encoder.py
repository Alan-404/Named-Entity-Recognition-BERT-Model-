import torch
import torch.nn as nn
from typing import Union, Callable

from ..utils.layer import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout_rate, eps, activation) for _ in range(n)])

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]):
        for layer in self.layers:
            x = layer(x, mask)
        return x