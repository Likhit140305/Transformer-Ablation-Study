import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, bias: bool = True):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)."""
        batch, seq, d_model = x.shape
        q, k, v = self.qkv(x).view(batch, seq, 3, self.n_heads, -1).permute(2, 0, 3, 1, 4)
        context = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        return self.w_o(context.transpose(1, 2).reshape(batch, seq, d_model))
