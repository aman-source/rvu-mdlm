"""B1: Best-of-N decoding with batched rollouts.

Generate N full sequences using sampled decoding, score each with reward,
pick argmax. Tie-break: first generated (lowest rollout index).

Rollouts are decoded in chunks of b1_batch_size using batched forward passes.
"""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from rvu.models.base import MDLM
from rvu.decoding.base import DecodeResult, Decoder, StepTrace, commit_schedule, expected_steps
from rvu.rewards.base import Reward
from rvu.utils import detokenize_completion


def matched_n(steps: int, k: int) -> int:
    """Compute N for B1 to match RVU's reward-call budget (config-level)."""
    return steps * k


def matched_n_per_case(prompt_len: int, max_len: int, steps: int, k: int) -> int:
    """Compute per-case N for B1 to match RVU's actual reward calls."""
    es = expected_steps(prompt_len, max_len, steps)
    return max(1, es * k)


@dataclass
class BestOfNTrace:
    """Trace for B1: per-rollout rewards and chosen index."""
    rollout_rewards: List[float]
    chosen_index: int
    chosen_reward: float


def _decode_batch_sampled(
    model: MDLM,
    prompt_ids: Optional[Tensor],
    batch_size: int,
    steps: int,
    max_len: int,
    device: torch.device,
    temperature: float,
    seeds: List[int],
) -> List[Tensor]:
    """Decode a batch of sampled rollouts in parallel.

    Args:
        seeds: One seed per rollout in this batch (len == batch_size).

    Returns:
        List of [L] token ID tensors, one per rollout.
    """
    B = batch_size
    L = max_len

    # Build batch of canvases
    canvases = model.fully_masked_canvas(length=L).unsqueeze(0).expand(B, -1).clone().to(device)

    # Pre-commit prompt
    prompt_len = 0
    if prompt_ids is not None and len(prompt_ids) > 0:
        prompt_len = len(prompt_ids)
        canvases[:, :prompt_len] = prompt_ids.to(device)

    # Per-rollout generators
    gens = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(s)
        gens.append(g)

    for step_idx in range(steps):
        # Check masks — all rollouts share same mask structure at start,
        # but diverge after first commit. Use per-rollout mask.
        # However, commit schedule is the same for all (same initial mask count,
        # same steps). After step 0 they may diverge slightly if different tokens
        # get committed, but mask COUNT should stay identical because we commit
        # the same number per step in all rollouts.
        # Use first rollout's mask count for schedule (they're all identical).
        mask_0 = (canvases[0] == model.mask_id)
        n_masked = int(mask_0.sum().item())
        if n_masked == 0:
            break

        steps_left = steps - step_idx
        n_commit = commit_schedule(n_masked, steps_left)

        # Batched forward pass
        all_logits = model.logits_batch(canvases)  # [B, L, V]

        # Suppress mask token
        all_logits[:, :, model.mask_id] = float("-inf")

        # Confidence: max softmax prob per position
        probs = F.softmax(all_logits, dim=-1)  # [B, L, V]
        max_probs, _ = probs.max(dim=-1)  # [B, L]

        # Temperature-scaled sampling probs
        sampling_logits = all_logits / temperature
        sampling_probs = F.softmax(sampling_logits, dim=-1)  # [B, L, V]

        for b in range(B):
            mask_b = (canvases[b] == model.mask_id)
            confidence_b = torch.where(mask_b, max_probs[b], torch.tensor(0.0, device=device))

            n_masked_b = int(mask_b.sum().item())
            if n_masked_b == 0:
                continue

            _, top_indices = confidence_b.topk(min(n_commit, n_masked_b))

            # Sample tokens at committed positions
            selected = torch.zeros(len(top_indices), dtype=torch.long, device=device)
            for idx_i, pos in enumerate(top_indices):
                selected[idx_i] = torch.multinomial(
                    sampling_probs[b, pos].unsqueeze(0), 1, generator=gens[b]
                ).squeeze()

            canvases[b, top_indices] = selected

    return [canvases[b] for b in range(B)]


class BestOfNDecoder(Decoder):
    """Best-of-N: generate N sampled rollouts, pick highest reward.

    Rollouts decoded in chunks of b1_batch_size for GPU efficiency.
    """

    def __init__(self, reward: Reward) -> None:
        self.reward = reward

    def decode(
        self,
        model: MDLM,
        prompt_ids: Optional[Tensor],
        config: Dict[str, Any],
    ) -> DecodeResult:
        N: int = config["N"]
        steps: int = config["steps"]
        max_len: int = config.get("max_len", model.max_len)
        device = torch.device(config.get("device", "cpu"))
        temperature: float = config.get("temperature", 0.7)
        seed: int = config.get("seed", 42)
        batch_size: int = config.get("b1_batch_size", 16)

        prompt_len = len(prompt_ids) if prompt_ids is not None else 0

        all_canvases: List[Tensor] = []
        all_rewards: List[float] = []

        # Process in chunks
        for chunk_start in range(0, N, batch_size):
            chunk_end = min(chunk_start + batch_size, N)
            chunk_n = chunk_end - chunk_start
            chunk_seeds = [seed * 100000 + i for i in range(chunk_start, chunk_end)]

            chunk_canvases = _decode_batch_sampled(
                model=model,
                prompt_ids=prompt_ids,
                batch_size=chunk_n,
                steps=steps,
                max_len=max_len,
                device=device,
                temperature=temperature,
                seeds=chunk_seeds,
            )

            for canvas in chunk_canvases:
                all_canvases.append(canvas)
                # Score completion-only
                r = self.reward(canvas[prompt_len:], model.tokenizer)
                all_rewards.append(r)

        # Pick argmax reward; tie-break: first generated
        best_idx = 0
        best_reward = all_rewards[0]
        for i in range(1, N):
            if all_rewards[i] > best_reward:
                best_reward = all_rewards[i]
                best_idx = i

        chosen_canvas = all_canvases[best_idx]
        text = detokenize_completion(chosen_canvas, model.tokenizer, prompt_len)

        bon_trace = BestOfNTrace(
            rollout_rewards=all_rewards,
            chosen_index=best_idx,
            chosen_reward=best_reward,
        )

        return DecodeResult(
            token_ids=chosen_canvas,
            text=text,
            trace=[bon_trace],  # type: ignore[arg-type]
            reward_calls_used=N,
            prompt_len=prompt_len,
            config=config,
        )
