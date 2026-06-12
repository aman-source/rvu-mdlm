"""Base decoder interface. All decoders (B0, B1, B2, RVU) implement this."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rvu.models.base import MDLM


@dataclass
class StepTrace:
    """Record of a single denoising step."""
    step: int
    positions_committed: List[int]  # indices into canvas
    tokens_committed: List[int]     # token IDs placed at those positions
    confidences: List[float]        # confidence scores at committed positions
    masks_remaining_before: int
    masks_remaining_after: int


@dataclass
class RVUStepTrace(StepTrace):
    """Extended trace for RVU steps (paper figures come from this)."""
    agreement_scores: List[float] = field(default_factory=list)
    reward_values: List[float] = field(default_factory=list)  # K values
    weights: List[float] = field(default_factory=list)          # K values


@dataclass
class DecodeResult:
    """Output of any decoder."""
    token_ids: Tensor          # [L] final canvas, no masks
    text: str                  # detokenized completion text (prompt excluded, EOS-stripped)
    trace: List[StepTrace]     # per-step trace
    reward_calls_used: int     # 0 for B0
    forward_passes: int = 0    # number of model forward calls (batched counts as batch_size)
    prompt_len: int = 0        # number of prompt tokens (completion = token_ids[prompt_len:])
    config: Dict[str, Any] = field(default_factory=dict)


def commit_schedule(n_masks: int, steps_remaining: int) -> int:
    """How many positions to commit this step.

    Linear schedule: n = ceil(remaining_masks / remaining_steps).
    Guarantees all masks filled in exactly S steps.
    """
    assert steps_remaining > 0, "No steps remaining"
    assert n_masks > 0, "No masks to commit"
    return ceil(n_masks / steps_remaining)


def expected_steps(prompt_len: int, max_len: int, steps: int) -> int:
    """How many denoising steps RVU/B2 will actually take for a given prompt.

    n_masks = max_len - prompt_len. The commit_schedule guarantees completion
    in min(steps, n_masks) steps (can't take more steps than masks).
    """
    n_masks = max_len - prompt_len
    if n_masks <= 0:
        return 0
    return min(steps, n_masks)


class Decoder(ABC):
    """Abstract decoder interface."""

    @abstractmethod
    def decode(
        self,
        model: MDLM,
        prompt_ids: Optional[Tensor],
        config: Dict[str, Any],
    ) -> DecodeResult:
        """Run decoding.

        Args:
            model: MDLM model providing logits.
            prompt_ids: Optional [P] tensor of prompt token IDs to prepend
                        (these are pre-committed, never masked).
            config: Dict with keys: steps, max_len, device, and decoder-specific params.

        Returns:
            DecodeResult with final tokens, text, trace, and reward call count.
        """
        ...
