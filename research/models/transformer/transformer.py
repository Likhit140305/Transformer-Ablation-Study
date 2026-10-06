import torch
import torch.nn as nn

from ..attention.multi_head_attention import MultiHeadAttention
from ..embedding.positional_embedding import SinusoidalPositionalEncoding
from ..embedding.token_embedding import TokenEmbedding
from ..ffn.feed_forward import FeedForward
from ..residual.residual_connection import ResidualConnection


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        activation: str,
        attn_dropout: float,
        ff_dropout: float,
        pre_norm: bool,
        use_residual: bool,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout=attn_dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout=ff_dropout, activation=activation)
        self.attn_residual = ResidualConnection(d_model, dropout, pre_norm, use_residual)
        self.ffn_residual = ResidualConnection(d_model, dropout, pre_norm, use_residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn_residual(x, self.self_attn)
        return self.ffn_residual(x, self.ffn)


class TransformerLM(nn.Module):
    """Decoder-only causal language model whose components can be ablated independently."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        emb_dropout: float = 0.1,
        activation: str = "gelu",
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        pos_encoding: str = "learned",
        pre_norm: bool = True,
        use_residual: bool = True,
        tie_weights: bool = True,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoding = pos_encoding
        if pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        elif pos_encoding == "sinusoidal":
            self.pos_embedding = SinusoidalPositionalEncoding(d_model, max_seq_length)
        elif pos_encoding != "none":
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, dropout, activation, attn_dropout, ff_dropout, pre_norm, use_residual)
            for _ in range(n_layers)
        )
        # Post-LN blocks already end in a LayerNorm; Pre-LN needs one before the head.
        self.final_norm = nn.LayerNorm(d_model) if pre_norm else nn.Identity()
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)
        if tie_weights:
            self.output_projection.weight = self.token_embedding.embedding.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding) and module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        x = self.token_embedding(input_ids)
        if self.pos_encoding == "learned":
            x = x + self.pos_embedding(torch.arange(seq_len, device=input_ids.device))
        elif self.pos_encoding == "sinusoidal":
            x = x + self.pos_embedding(seq_len)
        x = self.emb_dropout(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_projection(self.final_norm(x))
