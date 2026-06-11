"""Tests for B1 best-of-N decoding."""

from typing import List
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from rvu.decoding.best_of_n import BestOfNDecoder, BestOfNTrace, matched_n
from rvu.decoding.vanilla import VanillaDecoder
from rvu.models.base import MDLM
from rvu.rewards.base import Reward


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FixedLogitsModel(MDLM):
    """Returns predetermined logits."""

    def __init__(self, fixed_logits: Tensor, mask_id: int, vocab_size: int):
        self.device = torch.device("cpu")
        self.max_len = fixed_logits.shape[0]
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self._fixed_logits = fixed_logits
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = None
        self.tokenizer.decode = lambda ids, **kw: " ".join(str(i) for i in ids)

    def logits(self, canvas: Tensor) -> Tensor:
        return self._fixed_logits.clone()


class SequenceReward(Reward):
    """Returns rewards from a list, one per call."""

    def __init__(self, values: List[float]):
        self.values = values
        self.call_count = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        idx = self.call_count % len(self.values)
        self.call_count += 1
        return self.values[idx]


class ConstantReward(Reward):
    def __init__(self, value: float):
        self.value = value
        self.call_count = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        self.call_count += 1
        return self.value


class TokenSumReward(Reward):
    """Reward = sum of token IDs. Creates natural variation among rollouts."""

    def __init__(self):
        self.call_count = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        self.call_count += 1
        return float(token_ids.sum().item())


# ---------------------------------------------------------------------------
# matched_n
# ---------------------------------------------------------------------------


class TestMatchedN:
    def test_basic(self):
        assert matched_n(steps=4, k=8) == 32

    def test_single_step(self):
        assert matched_n(steps=1, k=5) == 5

    def test_single_k(self):
        assert matched_n(steps=10, k=1) == 10

    def test_matches_rvu_budget(self):
        """N from matched_n should equal RVU's S × K reward calls."""
        S, K = 16, 8
        N = matched_n(S, K)
        assert N == S * K == 128


# ---------------------------------------------------------------------------
# Reward accounting
# ---------------------------------------------------------------------------


class TestRewardAccounting:
    def test_reward_calls_equals_n(self):
        L, V = 4, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        N = 7
        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": N, "steps": 4, "max_len": 4,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert result.reward_calls_used == N
        assert reward.call_count == N

    def test_reward_calls_with_matched_budget(self):
        """matched_n gives N that matches RVU's S*K budget."""
        S, K = 3, 5
        N = matched_n(S, K)

        L, V = 6, 8
        mask_id = 7
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": N, "steps": S, "max_len": 6,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        assert result.reward_calls_used == S * K


# ---------------------------------------------------------------------------
# Identical rewards → picks first
# ---------------------------------------------------------------------------


class TestIdenticalRewards:
    def test_picks_first_on_tie(self):
        L, V = 3, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": 5, "steps": 3, "max_len": 3,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert len(result.trace) == 1
        trace = result.trace[0]
        assert isinstance(trace, BestOfNTrace)
        assert trace.chosen_index == 0
        assert trace.chosen_reward == 0.5
        assert len(trace.rollout_rewards) == 5
        assert all(r == 0.5 for r in trace.rollout_rewards)


# ---------------------------------------------------------------------------
# Best rollout selected
# ---------------------------------------------------------------------------


class TestBestRolloutSelected:
    def test_picks_highest_reward(self):
        L, V = 3, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        # 5 rollouts with rewards [0.1, 0.3, 0.9, 0.2, 0.5]
        reward = SequenceReward([0.1, 0.3, 0.9, 0.2, 0.5])
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": 5, "steps": 3, "max_len": 3,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        trace = result.trace[0]
        assert isinstance(trace, BestOfNTrace)
        assert trace.chosen_index == 2  # reward 0.9
        assert trace.chosen_reward == 0.9


# ---------------------------------------------------------------------------
# Sampled variant produces diverse rollouts
# ---------------------------------------------------------------------------


class TestSampledDiversity:
    def test_rollouts_vary(self):
        """With moderate temperature and non-peaked logits, rollouts should differ."""
        L, V = 6, 10
        mask_id = 9
        # Spread logits so sampling has entropy
        logits = torch.ones(L, V) * 1.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        # Use TokenSumReward to capture diversity through reward variation
        reward = TokenSumReward()
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": 10, "steps": 3, "max_len": 6,
            "temperature": 1.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        trace = result.trace[0]
        assert isinstance(trace, BestOfNTrace)
        # With uniform logits and 10 rollouts, rewards should vary
        unique_rewards = set(trace.rollout_rewards)
        assert len(unique_rewards) > 1, "Rollouts should produce diverse sequences"

    def test_each_rollout_gets_unique_seed(self):
        """Different rollout indices should produce different sequences
        even with same base seed."""
        L, V = 4, 8
        mask_id = 7
        logits = torch.ones(L, V) * 1.0  # uniform → high entropy
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        # Run two single-rollout B1 with different rollout seeds
        # (simulated by running N=2 and checking they differ)
        collected: List[List[int]] = []

        class CollectorReward(Reward):
            def __init__(self):
                self.call_count = 0
                self.sequences: List[List[int]] = []

            def __call__(self, token_ids, tokenizer):
                self.call_count += 1
                self.sequences.append(token_ids.tolist())
                return 0.5

        reward = CollectorReward()
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": 5, "steps": 2, "max_len": 4,
            "temperature": 1.0, "seed": 42, "device": "cpu",
        }
        decoder.decode(model, prompt_ids=None, config=config)

        # At least some rollouts should differ
        unique_seqs = set(tuple(s) for s in reward.sequences)
        assert len(unique_seqs) > 1, "Different rollouts should produce different sequences"


# ---------------------------------------------------------------------------
# Determinism per seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_result(self):
        L, V = 4, 6
        mask_id = 5
        logits = torch.ones(L, V) * 1.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        config = {
            "N": 5, "steps": 2, "max_len": 4,
            "temperature": 1.0, "seed": 77, "device": "cpu",
        }

        reward1 = ConstantReward(0.5)
        result1 = BestOfNDecoder(reward=reward1).decode(model, None, config)

        reward2 = ConstantReward(0.5)
        result2 = BestOfNDecoder(reward=reward2).decode(model, None, config)

        assert torch.equal(result1.token_ids, result2.token_ids)

    def test_different_seed_different_result(self):
        L, V = 4, 10
        mask_id = 9
        logits = torch.ones(L, V) * 1.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        config1 = {
            "N": 5, "steps": 2, "max_len": 4,
            "temperature": 1.0, "seed": 42, "device": "cpu",
        }
        config2 = {**config1, "seed": 999}

        reward1 = ConstantReward(0.5)
        result1 = BestOfNDecoder(reward=reward1).decode(model, None, config1)

        reward2 = ConstantReward(0.5)
        result2 = BestOfNDecoder(reward=reward2).decode(model, None, config2)

        # With high entropy, collision very unlikely
        assert not torch.equal(result1.token_ids, result2.token_ids)


# ---------------------------------------------------------------------------
# Greedy B0 regression — unchanged behavior
# ---------------------------------------------------------------------------


class TestGreedyB0Regression:
    """VanillaDecoder without sample flag still works as greedy B0."""

    def test_greedy_default(self):
        L, V = 4, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        logits[3, 3] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        decoder = VanillaDecoder()
        config = {"steps": 4, "max_len": 4, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Greedy: each position committed to its argmax
        assert result.token_ids.tolist() == [0, 1, 2, 3]
        assert result.reward_calls_used == 0

    def test_greedy_deterministic_regardless_of_seed(self):
        """Greedy mode ignores seed — always same result."""
        L, V = 3, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        decoder = VanillaDecoder()
        config1 = {"steps": 3, "max_len": 3, "device": "cpu", "seed": 42}
        config2 = {"steps": 3, "max_len": 3, "device": "cpu", "seed": 999}
        r1 = decoder.decode(model, None, config1)
        r2 = decoder.decode(model, None, config2)
        assert torch.equal(r1.token_ids, r2.token_ids)

    def test_sampled_mode_differs_from_greedy(self):
        """With moderate logits, sampled and greedy should occasionally differ."""
        L, V = 6, 10
        mask_id = 9
        logits = torch.ones(L, V) * 1.0
        logits[:, 0] = 2.0  # slight preference
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        greedy = VanillaDecoder()
        greedy_result = greedy.decode(model, None, {
            "steps": 3, "max_len": 6, "device": "cpu",
        })

        sampled = VanillaDecoder()
        sampled_result = sampled.decode(model, None, {
            "steps": 3, "max_len": 6, "device": "cpu",
            "sample": True, "temperature": 1.0, "seed": 42,
        })

        # Greedy always picks token 0 (highest logit). Sampled should sometimes differ.
        greedy_all_zero = all(t == 0 for t in greedy_result.token_ids.tolist())
        sampled_all_zero = all(t == 0 for t in sampled_result.token_ids.tolist())
        # At least one should differ (high probability with temperature=1.0)
        # This could theoretically fail but probability is ~(1/10)^6 ≈ 1e-6
        assert greedy_all_zero  # greedy always picks token 0
        assert not sampled_all_zero  # sampled almost certainly varies


# ---------------------------------------------------------------------------
# No mask tokens in output
# ---------------------------------------------------------------------------


class TestNoMaskInOutput:
    def test_no_mask_in_best_of_n_output(self):
        L, V = 5, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(1.0)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": 3, "steps": 5, "max_len": 5,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        assert (result.token_ids != mask_id).all()
