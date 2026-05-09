import torch
import torch.nn as nn
from typing import Optional


class TokenEmbedding(nn.Module):
    """Token embedding layer."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: Optional[int] = None):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimension of the model embeddings
            padding_idx: Index of the padding token (if any)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: Tensor of token indices of shape (batch_size, seq_len)

        Returns:
            Tensor of embeddings of shape (batch_size, seq_len, d_model)
        """
        return self.embedding(tokens) * (self.d_model ** 0.5)