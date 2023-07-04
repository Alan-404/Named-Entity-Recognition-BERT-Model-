import torch
import torch.nn as nn
from typing import Union


class MultiHeadAttention(nn.Module):
    def __init__(self, heads: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.heads = heads

        self.head_samples = self.d_model // self.heads

        self.linear_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.linear_v = nn.Linear(in_features=d_model, out_features=d_model)

        self.linear_output = nn.Linear(in_features=d_model, out_features=d_model)

    def scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> tuple[torch.Tensor, torch.Tensor]:
        dk = torch.tensor(k.size(-1))

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(dk)

        if mask is not None:
            attention_scores += mask * (-1e15)
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_context = torch.matmul(attention_weights, v)

        return attention_context, attention_weights
    
    def split_head(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        n_ctx = x.size(1)

        x = x.reshape([batch_size, n_ctx, self.heads, self.head_samples])
        x = x.permute([0, 2, 1, 3])

        return x
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Union[torch.Tensor, None]) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)
        n_ctx = q.size(1)
        
        qw = self.linear_q(q)
        kw = self.linear_k(k)
        vw = self.linear_v(v)

        q_heads = self.split_head(qw)
        k_heads = self.split_head(kw)
        v_heads = self.split_head(vw)

        attention_context, attention_weights = self.scaled_dot_product_attention(q_heads, k_heads, v_heads, mask)

        attention_context = attention_context.permute([0, 2, 3, 1])
        attention_context = attention_context.reshape([batch_size, n_ctx, self.d_model])

        attention_context = self.linear_output(attention_context)
        return attention_context, attention_weights