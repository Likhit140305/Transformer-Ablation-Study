import torch
import torch.nn as nn
from typing import Optional


class LayerNorm(nn.Module):
    """Layer normalization wrapper.

    This is a thin wrapper around nn.LayerNorm to allow for easy substitution
    of different normalization layers if needed.
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ):
        """
        Args:
            normalized_shape: Input shape from an expected input of size
            eps: A value added to the denominator for numerical stability
            elementwise_affine: If True, this module has learnable per-element
                affine parameters initialized to ones (for weights) and zeros
                for biases.
            bias: If True, adds a learnable bias to the output.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (*, normalized_shape)

        Returns:
            Tensor of same shape as input
        """
        return self.layer_norm(x)