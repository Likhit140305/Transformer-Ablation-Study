import torch
import torch.nn as nn
from typing import Optional, Callable


class ResidualConnection(nn.Module):
    """
    Residual connection with layer normalization and dropout.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        pre_norm: bool = True,
        use_residual: bool = True,
    ):
        """
        Args:
            d_model: Dimension of the model
            dropout: Dropout probability
            pre_norm: If True, apply LN before sublayer (Pre-LN).
                      If False, apply LN after sublayer (Post-LN).
            use_residual: If False, ablates the residual connection (returns only sublayer output).
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.use_residual = use_residual
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        sublayer: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor
            sublayer: Function/module to apply
        """
        if self.pre_norm:
            # Pre-LN: x + Dropout(Sublayer(LN(x)))
            sub_out = sublayer(self.norm(x))
            sub_out = self.dropout(sub_out)
            if self.use_residual:
                return x + sub_out
            return sub_out
        else:
            # Post-LN: LN(x + Dropout(Sublayer(x)))
            sub_out = sublayer(x)
            sub_out = self.dropout(sub_out)
            if self.use_residual:
                return self.norm(x + sub_out)
            return self.norm(sub_out)