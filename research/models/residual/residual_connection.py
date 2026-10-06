from typing import Callable

import torch
import torch.nn as nn


class ResidualConnection(nn.Module):
    """Wraps a sublayer with normalization, dropout and an optional skip path.

    Pre-LN:  x + Dropout(F(LN(x)))      Post-LN:  LN(x + Dropout(F(x)))
    ``use_residual=False`` drops the ``x +`` term and nothing else.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, pre_norm: bool = True, use_residual: bool = True):
        super().__init__()
        self.pre_norm = pre_norm
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        skip = x if self.use_residual else 0
        if self.pre_norm:
            return skip + self.dropout(sublayer(self.norm(x)))
        return self.norm(skip + self.dropout(sublayer(x)))
