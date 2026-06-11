"""Shared utilities for RVU decoders."""

from torch import Tensor
from transformers import PreTrainedTokenizerBase


def detokenize(token_ids: Tensor, tokenizer: PreTrainedTokenizerBase) -> str:
    """Decode token IDs to text, stripping everything after first EOS.

    Args:
        token_ids: [L] int tensor.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Decoded string, truncated at first EOS token if present.
    """
    ids = token_ids.tolist()

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in ids:
        ids = ids[:ids.index(eos_id)]

    return tokenizer.decode(ids, skip_special_tokens=True)


def detokenize_completion(
    token_ids: Tensor, tokenizer: PreTrainedTokenizerBase, prompt_len: int
) -> str:
    """Decode only the completion region (token_ids[prompt_len:]) to text.

    Args:
        token_ids: [L] int tensor (full canvas).
        tokenizer: HuggingFace tokenizer.
        prompt_len: Number of prompt tokens to skip.

    Returns:
        Decoded completion string, EOS-stripped.
    """
    ids = token_ids[prompt_len:].tolist()

    eos_id = tokenizer.eos_token_id
    if eos_id is not None and eos_id in ids:
        ids = ids[:ids.index(eos_id)]

    return tokenizer.decode(ids, skip_special_tokens=True)
