import torch
import torch.nn as nn
from typing import Optional
from ...models.embedding.token_embedding import TokenEmbedding
from ...models.embedding.positional_embedding import PositionalEncoding
from ...models.attention.multi_head_attention import MultiHeadAttention
from ...models.ffn.feed_forward import FeedForward
from ...models.layernorm.layer_norm import LayerNorm
from ...models.residual.residual_connection import ResidualConnection


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        pre_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model, n_heads=n_heads, dropout=attn_dropout, bias=True
        )
        self.ffn = FeedForward(
            d_model=d_model, d_ff=d_ff, dropout=ff_dropout, activation=activation
        )
        self.attn_residual = ResidualConnection(
            d_model=d_model, dropout=dropout, pre_norm=pre_norm, use_residual=use_residual
        )
        self.ffn_residual = ResidualConnection(
            d_model=d_model, dropout=dropout, pre_norm=pre_norm, use_residual=use_residual
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn_residual(x, lambda _x: self.self_attn(_x, _x, _x, mask=mask))
        x = self.ffn_residual(x, self.ffn)
        return x


class TransformerLM(nn.Module):
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
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pos_embedding_type = pos_encoding
        self.pre_norm = pre_norm

        self.token_embedding = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, padding_idx=padding_idx)

        if pos_encoding == "learned":
            self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        elif pos_encoding == "sinusoidal":
            self.pos_embedding = PositionalEncoding(d_model=d_model, max_len=max_seq_length)
        elif pos_encoding == "none":
            self.pos_embedding = None
        else:
            raise ValueError(f"Unknown pos_encoding: {pos_encoding}")

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                activation=activation,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                pre_norm=pre_norm,
                use_residual=use_residual,
            ) for _ in range(n_layers)
        ])

        # If pre_norm, we typically apply a final layer norm before the projection
        self.final_norm = LayerNorm(d_model) if pre_norm else nn.Identity()
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.output_projection.weight = self.token_embedding.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = input_ids.size(1)
        device = input_ids.device

        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1) 

        if attention_mask is not None:
            padding_mask = attention_mask.unsqueeze(1).unsqueeze(2) 
            mask = causal_mask & padding_mask
        else:
            mask = causal_mask

        x = self.token_embedding(input_ids)

        if self.pos_embedding_type == "learned":
            positions = torch.arange(0, seq_len, device=device).unsqueeze(0)
            x = x + self.pos_embedding(positions)
        elif self.pos_embedding_type == "sinusoidal":
            x = self.pos_embedding(x)
        
        x = self.emb_dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits