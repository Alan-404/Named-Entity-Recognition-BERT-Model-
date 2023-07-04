import torch
import torch.nn as nn
from typing import Union, Callable
from .utils.mask import generate_padding_mask

from .components.encoder import Encoder


class BERT(nn.Module):
    def __init__(self, token_size: int, num_entities: int, n: int, d_model: int, heads: int, d_ff: int, dropout_rate: float, eps: float, activation: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.encoder = Encoder(n, d_model, heads, d_ff, dropout_rate, eps, activation)
        self.ner_layer = nn.Linear(in_features=d_model, out_features=num_entities)
    def forward(self, x: torch.Tensor):
        if self.training:
            mask = generate_padding_mask(x)
        x = self.embedding_layer(x)
        if self.training:
            x = self.encoder(x, mask)
        else:
            x = self.encoder(x, None)
        x = self.ner_layer(x)
        return x
