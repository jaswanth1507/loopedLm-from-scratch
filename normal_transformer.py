"""A conventional (untied-depth) Transformer baseline for the experiment."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looped_lm import LoopedLMConfig, TransformerBlock


class NormalTransformer(nn.Module):
    """Same decoder block, but every depth has its own parameters."""

    def __init__(self, config: LoopedLMConfig, layers: int = 4) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.block_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        _, time = tokens.shape
        if time > self.config.block_size:
            raise ValueError("sequence is longer than block_size")
        positions = torch.arange(time, device=tokens.device)
        state = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        for block in self.blocks:
            state = block(state)
        logits = self.lm_head(self.final_norm(state))
        loss = None if targets is None else F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss
