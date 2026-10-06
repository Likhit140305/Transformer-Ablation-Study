import torch
import torch.nn as nn

ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "linear": nn.Identity}


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    ``activation="linear"`` keeps both projections, so the ablation removes the
    non-linearity without changing width or parameter count.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"Unsupported activation: {activation}")
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
