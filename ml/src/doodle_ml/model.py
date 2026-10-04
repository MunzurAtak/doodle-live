"""Stroke-sequence transformer classifier (SPEC §5).

Input ``strokes (B, L, 3)`` of stroke-3 rows and ``mask (B, L)`` (1 = real row), with
``L <= max_len``. Output ``logits (B, n_classes)``.

Attention is written out explicitly instead of using ``nn.TransformerEncoder``: the
built-in module takes a different inference "fast path" with padding masks and its ONNX
export is fragile. Padded keys get a large negative bias (not -inf), so even an all-padding
row stays finite, and padding never changes the output for the real rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

MASK_BIAS = -1e9


@dataclass(frozen=True)
class ModelConfig:
    n_classes: int = 50
    max_len: int = 200
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 256
    dropout: float = 0.1


class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> Tensor:
        """``key_bias``: (B, 1, 1, L) with 0 for real rows and MASK_BIAS for padding."""
        b, length, d = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.n_heads, self.d_head).unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, H, L, Dh)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head) + key_bias
        attn = self.dropout(scores.softmax(dim=-1))
        context = (attn @ v).transpose(1, 2).reshape(b, length, d)
        return self.out(context)


class EncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = MaskedSelfAttention(config.d_model, config.n_heads, config.dropout)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, key_bias: Tensor) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), key_bias))
        return x + self.dropout(self.ff(self.norm2(x)))


class StrokeTransformer(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()) -> None:  # noqa: B008
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(3, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_len, config.d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.layers = nn.ModuleList(EncoderLayer(config) for _ in range(config.n_layers))
        self.head_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.n_classes)

    def forward(self, strokes: Tensor, mask: Tensor) -> Tensor:
        length = strokes.shape[1]
        x = self.input_proj(strokes) + self.pos_emb[:, :length]
        key_bias = ((1.0 - mask) * MASK_BIAS)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, key_bias)
        weights = mask.unsqueeze(-1)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.head(self.head_norm(pooled))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
