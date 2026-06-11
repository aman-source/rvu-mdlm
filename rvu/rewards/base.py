"""Abstract interface for reward functions."""

from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor


class Reward(ABC):
    """Black-box reward callable: sequence in, scalar out.

    RVU never sees internals — just calls __call__.
    """

    @abstractmethod
    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        """Score a single complete sequence.

        Args:
            token_ids: [L] int tensor of token IDs (no masks).
            tokenizer: Tokenizer for decoding to text if needed.

        Returns:
            Scalar reward. May be binary, partial-credit, or continuous.
        """
        ...

    def batch_score(self, sequences: Tensor, tokenizer) -> List[float]:
        """Score a batch of sequences. Default: loop over __call__.

        Args:
            sequences: [K, L] int tensor.
            tokenizer: Tokenizer for decoding.

        Returns:
            List of K reward scalars.
        """
        return [self(seq, tokenizer) for seq in sequences]
