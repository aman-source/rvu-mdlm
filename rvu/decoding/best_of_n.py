"""B1: Best-of-N decoding.

Generate N full sequences using sampled B0 decoding, score each with reward,
pick argmax. Tie-break: first generated (lowest rollout index).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from rvu.models.base import MDLM
from rvu.decoding.base import DecodeResult, Decoder, expected_steps
from rvu.decoding.vanilla import VanillaDecoder
from rvu.rewards.base import Reward


def matched_n(steps: int, k: int) -> int:
    """Compute N for B1 to match RVU's reward-call budget (config-level).

    For per-case matching use expected_steps(prompt_len, L, S) * K.
    """
    return steps * k


def matched_n_per_case(prompt_len: int, max_len: int, steps: int, k: int) -> int:
    """Compute per-case N for B1 to match RVU's actual reward calls.

    N = expected_steps(prompt_len, max_len, steps) * K.
    """
    es = expected_steps(prompt_len, max_len, steps)
    return max(1, es * k)  # at least 1 rollout


@dataclass
class BestOfNTrace:
    """Trace for B1: per-rollout rewards and chosen index."""
    rollout_rewards: List[float]
    chosen_index: int
    chosen_reward: float


class BestOfNDecoder(Decoder):
    """Best-of-N: generate N sampled rollouts, pick highest reward.

    Each rollout uses VanillaDecoder in sampled mode with its own
    torch.Generator seed derived from (run_seed, rollout_index).
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
        device_str: str = config.get("device", "cpu")
        temperature: float = config.get("temperature", 0.7)
        seed: int = config.get("seed", 42)

        prompt_len = len(prompt_ids) if prompt_ids is not None else 0

        inner_decoder = VanillaDecoder()

        rollout_results: List[DecodeResult] = []
        rollout_rewards: List[float] = []

        for i in range(N):
            rollout_seed = seed * 100000 + i
            inner_config = {
                "steps": steps,
                "max_len": max_len,
                "device": device_str,
                "sample": True,
                "temperature": temperature,
                "seed": rollout_seed,
            }
            result = inner_decoder.decode(model, prompt_ids, inner_config)
            rollout_results.append(result)

            # Score completion-only (exclude prompt tokens)
            r = self.reward(result.token_ids[prompt_len:], model.tokenizer)
            rollout_rewards.append(r)

        # Pick argmax reward; tie-break: first generated (lowest index)
        best_idx = 0
        best_reward = rollout_rewards[0]
        for i in range(1, N):
            if rollout_rewards[i] > best_reward:
                best_reward = rollout_rewards[i]
                best_idx = i

        chosen = rollout_results[best_idx]
        bon_trace = BestOfNTrace(
            rollout_rewards=rollout_rewards,
            chosen_index=best_idx,
            chosen_reward=best_reward,
        )

        return DecodeResult(
            token_ids=chosen.token_ids,
            text=chosen.text,
            trace=[bon_trace],  # type: ignore[arg-type]
            reward_calls_used=N,
            prompt_len=prompt_len,
            config=config,
        )
