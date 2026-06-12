"""Abstract interface for masked diffusion language models (MDLMs)."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase


class MDLM(ABC):
    """Base interface all MDLM adapters must implement.

    Attributes:
        tokenizer: HuggingFace tokenizer (or compatible).
        mask_id: Token ID used for [MASK].
        max_len: Maximum sequence length the model supports.
        device: torch device for tensors.
    """

    tokenizer: PreTrainedTokenizerBase
    mask_id: int
    max_len: int
    device: torch.device

    @abstractmethod
    def logits(self, canvas: Tensor) -> Tensor:
        """Compute per-position logits for the given canvas.

        Args:
            canvas: [L] int tensor. Committed positions hold token IDs,
                    masked positions hold self.mask_id.

        Returns:
            Tensor of shape [L, V] — logits over vocabulary at every position.
            Only logits at masked positions are used by decoders, but the model
            may return logits everywhere.
        """
        ...

    def logits_batch(self, canvases: Tensor) -> Tensor:
        """Batched forward pass: canvases [B, L] → logits [B, L, V].

        Default: loops over batch calling logits(). Override for true batching.
        """
        return torch.stack([self.logits(c) for c in canvases])

    def is_masked(self, canvas: Tensor) -> Tensor:
        """Return boolean mask: True where canvas == mask_id.

        Args:
            canvas: [L] int tensor.

        Returns:
            [L] bool tensor.
        """
        return canvas == self.mask_id

    def param_count(self) -> int:
        """Total number of model parameters. Override in subclasses."""
        return 0

    def fully_masked_canvas(self, length: Optional[int] = None) -> Tensor:
        """Create a canvas of all [MASK] tokens.

        Args:
            length: Sequence length. Defaults to self.max_len.

        Returns:
            [L] int tensor filled with self.mask_id on self.device.
        """
        L = length if length is not None else self.max_len
        return torch.full((L,), self.mask_id, dtype=torch.long, device=self.device)
