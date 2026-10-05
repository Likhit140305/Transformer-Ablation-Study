import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Union


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
    ):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (inner dimension)
            dropout: Dropout probability
            activation: Activation function. Can be a string ("relu" or "gelu") or a callable.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        if isinstance(activation, str):
            if activation.lower() == "relu":
                self.activation = F.relu
            elif activation.lower() == "gelu":
                self.activation = F.gelu
            elif activation.lower() in {"linear", "identity", "none"}:
                # Keeping the two linear projections makes this a controlled
                # non-linearity ablation rather than a change in FFN width.
                self.activation = lambda x: x
            else:
                raise ValueError(f"Unsupported activation: {activation}")
        else:
            self.activation = activation

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Output tensor of same shape
        """
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
