"""Tiny random-init MDLM for CPU-phase testing.

Outputs are garbage — the point is exercising every code path end to end.
Architecture: small transformer encoder (~5-10M params) with GPT-2 tokenizer.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from .base import MDLM


class TinyTransformerBlock(nn.Module):
    """Minimal pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h = self.ff(h)
        return x + h


class TinyMDLM(MDLM):
    """Small random-init transformer that implements the MDLM interface.

    Default config: d_model=256, n_heads=4, n_layers=4, d_ff=512
    ~5M params with GPT-2 vocab (50257).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 128,
        device: Optional[str] = None,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.device = torch.device(device or "cpu")

        # Tokenizer setup
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            tokenizer_name
        )
        # GPT-2 tokenizer has no [MASK] token. Add one.
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self.mask_id: int = self.tokenizer.mask_token_id  # type: ignore[assignment]
        self.max_len: int = max_len
        self.vocab_size: int = len(self.tokenizer)

        # Model
        self.embed = nn.Embedding(self.vocab_size, d_model).to(self.device)
        self.pos_embed = nn.Embedding(max_len, d_model).to(self.device)
        self.blocks = nn.ModuleList(
            [TinyTransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        ).to(self.device)
        self.ln_final = nn.LayerNorm(d_model).to(self.device)
        self.head = nn.Linear(d_model, self.vocab_size, bias=False).to(self.device)

        # Init weights
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def parameters(self):
        """Yield all model parameters."""
        for module in [self.embed, self.pos_embed, self.blocks, self.ln_final, self.head]:
            yield from module.parameters()

    @torch.no_grad()
    def logits(self, canvas: Tensor) -> Tensor:
        """Forward pass: canvas [L] -> logits [L, V]."""
        L = canvas.shape[0]
        assert L <= self.max_len, f"Canvas length {L} exceeds max_len {self.max_len}"

        canvas = canvas.to(self.device)
        positions = torch.arange(L, device=self.device)

        x = self.embed(canvas) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.head(x)  # [L, V]

    def param_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
