import torch
import torch.nn as nn
from typing import Callable, Union

from .attention import MultiHeadAttention
from .ffn import PositionWiseFeedForwardNetworks
from .residual import ResidualConnection


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(heads=heads, d_model=d_model)
        self.ffn = PositionWiseFeedForwardNetworks(d_ff=d_ff, d_model=d_model, activation=activation)

        self.residual_connection_1 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)
        self.residual_connection_2 = ResidualConnection(d_model=d_model, dropout_rate=dropout_rate, eps=eps)

    def forward(self, x: torch.Tensor, mask: Union[torch.Tensor, None]) -> torch.Tensor:
        # sublayer 1
        attention_context, _ = self.attention(x, x, x, mask)
        attention_context = self.residual_connection_1(attention_context, x)

        # sublayer 3
        ffn_output = self.ffn(attention_context)
        output = self.residual_connection_2(ffn_output, attention_context)

        return output