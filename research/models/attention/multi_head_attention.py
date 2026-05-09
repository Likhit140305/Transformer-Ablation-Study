import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to use bias in linear projections
        """
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for linear in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.constant_(linear.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask tensor of shape (batch_size, 1, seq_len_q, seq_len_k)
                  or (batch_size, seq_len_q, seq_len_k) for broadcasting.
                  Should be a bool tensor where True indicates positions to keep,
                  False indicates positions to mask.

        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)

        # 1. Linear projections
        Q = self.w_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.w_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.w_v(value)  # (batch_size, seq_len_v, d_model)

        # 2. Split into multiple heads
        # Reshape to (batch_size, seq_len, n_heads, d_k) and transpose to (batch_size, n_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_q, d_k)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_k, d_k)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len_v, d_k)

        # 3. Compute attention scores
        # Scores = Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, n_heads, seq_len_q, seq_len_k)

        # 4. Apply mask (if provided)
        if mask is not None:
            # Mask should be broadcastable to the shape of scores
            # We expect mask to be of shape (batch_size, 1, seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
            # If mask is provided as (batch_size, seq_len_q, seq_len_k), we unsqueeze to (batch_size, 1, seq_len_q, seq_len_k)
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            # Now mask is (batch_size, 1, seq_len_q, seq_len_k)
            # We want to set the scores to -inf where mask is False
            scores = scores.masked_fill(~mask, float('-inf'))

        # 5. Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)  # (batch_size, n_heads, seq_len_q, seq_len_k)
        attn_weights = self.dropout(attn_weights)

        # 6. Compute weighted sum of values
        context = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len_q, d_k)

        # 7. Concatenate heads and apply final linear layer
        # Transpose to (batch_size, seq_len_q, n_heads, d_k) and then reshape to (batch_size, seq_len_q, d_model)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(context)

        return output