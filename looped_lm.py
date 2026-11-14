"""A tiny decoder-only language model with recurrent (looped) depth.

The important bit is that ``self.core`` is one Transformer block, but its
``forward`` method is called ``num_loops`` times.  More loops therefore add
compute/effective depth without adding a second set of core parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class LoopedLMConfig:
    vocab_size: int
    block_size: int = 128
    d_model: int = 256
    n_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_loops: int = 4

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.num_loops < 1:
            raise ValueError("num_loops must be at least 1")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = config.dropout

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (batch, time, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # Causal attention prevents a position from reading future tokens.
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self.proj(y.transpose(1, 2).contiguous().view(batch, time, channels))


class TransformerBlock(nn.Module):
    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        hidden = config.mlp_ratio * config.d_model
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class LoopedLanguageModel(nn.Module):
    """Decoder-only LM with a shared Transformer core applied repeatedly."""

    def __init__(self, config: LoopedLMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.core = TransformerBlock(config)  # Intentionally only one block.
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # standard weight tying
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, tokens: Tensor, targets: Optional[Tensor] = None, *, loops: Optional[int] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Return next-token logits and, when targets are given, cross-entropy loss.

        ``loops`` may be changed at inference to spend more or less compute.
        Training across a range of loop counts is usually preferable if that
        inference-time flexibility matters.
        """
        batch, time = tokens.shape
        if time > self.config.block_size:
            raise ValueError(f"sequence length {time} exceeds block_size {self.config.block_size}")
        loops = self.config.num_loops if loops is None else loops
        if loops < 1:
            raise ValueError("loops must be at least 1")

        positions = torch.arange(time, device=tokens.device)
        input_state = self.embedding_dropout(
            self.token_embedding(tokens) + self.position_embedding(positions)[None, :, :]
        )
        state = input_state
        for _ in range(loops):
            # Re-injection keeps token/position evidence available each loop.
            state = self.core(state + input_state)
        logits = self.lm_head(self.final_norm(state))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self, tokens: Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: Optional[int] = None, *, loops: Optional[int] = None,
    ) -> Tensor:
        """Autoregressively sample tokens (simple reference implementation)."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            context = tokens[:, -self.config.block_size :]
            logits, _ = self(context, loops=loops)
            logits = logits[:, -1] / temperature
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
