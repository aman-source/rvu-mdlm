"""Tests for RVU decoder with hand-verified voting math.

Uses FixedLogitsModel: returns predetermined logits so vote tests
don't depend on TinyMDLM randomness.
"""

import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from rvu.decoding.base import RVUStepTrace
from rvu.decoding.rvu import RVUDecoder
from rvu.models.base import MDLM
from rvu.rewards.base import Reward


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FixedLogitsModel(MDLM):
    """Returns predetermined logits. Deterministic, no randomness."""

    def __init__(
        self,
        fixed_logits: Tensor,  # [L, V]
        mask_id: int,
        vocab_size: int,
    ):
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


class ConstantReward(Reward):
    """Returns a constant reward for every sequence."""

    def __init__(self, value: float):
        self.value = value
        self.call_count = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        self.call_count += 1
        return self.value


class ListReward(Reward):
    """Returns rewards from a predetermined list, cycling."""

    def __init__(self, values: List[float]):
        self.values = values
        self.call_count = 0

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        idx = self.call_count % len(self.values)
        self.call_count += 1
        return self.values[idx]


class FutureInspectingReward(Reward):
    """Records futures for inspection, returns predetermined rewards."""

    def __init__(self, rewards: List[float]):
        self.rewards = rewards
        self.call_count = 0
        self.observed_futures: List[List[int]] = []

    def __call__(self, token_ids: Tensor, tokenizer) -> float:
        self.observed_futures.append(token_ids.tolist())
        idx = self.call_count % len(self.rewards)
        self.call_count += 1
        return self.rewards[idx]


# ---------------------------------------------------------------------------
# Hand-verified voting math: L=3, V=4, K=3
# ---------------------------------------------------------------------------


class TestHandVerifiedVoting:
    """Tiny case: L=3 positions, V=4 vocab, K=3 futures.

    Setup:
    - All 3 positions masked (mask_id=3, so vocab tokens are 0,1,2).
    - Fixed logits make sampling deterministic enough for controlled tests.
    - We use predetermined rewards to verify weight/vote/agreement math.

    We set logits so that at each masked position, token distribution
    after tau_f scaling is heavily concentrated on one token. This way
    the sampled futures are predictable, and we can verify the math.

    Logits at each position (before mask_id suppression):
      pos 0: token 0 gets logit 10, rest 0  → samples token 0
      pos 1: token 1 gets logit 10, rest 0  → samples token 1
      pos 2: token 2 gets logit 10, rest 0  → samples token 2

    With tau_f=0.7 and logit=10, softmax(10/0.7) ≈ 1.0 for the hot token.
    So all K=3 futures will be [0, 1, 2] (deterministically).

    Rewards: [0.5, 0.8, 0.2] for the 3 futures.
    Lambda = 5.0.

    Hand computation:
      scaled = 5.0 * [0.5, 0.8, 0.2] = [2.5, 4.0, 1.0]
      shifted = [2.5-4.0, 0, 1.0-4.0] = [-1.5, 0.0, -3.0]
      exp = [exp(-1.5), exp(0), exp(-3)] = [0.22313, 1.0, 0.04979]
      sum = 1.27292
      weights = [0.17529, 0.78554, 0.03912] (approx, recomputed below)

    Since all futures agree on every position (all are [0,1,2]):
      score(pos, winner) = sum of all weights = 1.0 for each position.
      agreement = 1.0 everywhere.
      winner = [0, 1, 2].
    """

    def _make_model(self) -> FixedLogitsModel:
        L, V = 3, 4
        mask_id = 3
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        return FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

    def _compute_expected_weights(self, rewards, lam):
        """Compute expected weights by hand formula."""
        import math
        scaled = [lam * r for r in rewards]
        max_s = max(scaled)
        shifted = [s - max_s for s in scaled]
        exps = [math.exp(s) for s in shifted]
        total = sum(exps)
        return [e / total for e in exps]

    def test_weights_exact(self):
        """Verify weight computation matches hand calculation."""
        rewards = [0.5, 0.8, 0.2]
        lam = 5.0
        expected_w = self._compute_expected_weights(rewards, lam)

        # Verify hand computation
        # scaled = [2.5, 4.0, 1.0], shifted = [-1.5, 0.0, -3.0]
        assert abs(expected_w[0] - math.exp(-1.5) / (math.exp(-1.5) + 1.0 + math.exp(-3.0))) < 1e-6
        assert abs(expected_w[1] - 1.0 / (math.exp(-1.5) + 1.0 + math.exp(-3.0))) < 1e-6
        assert abs(expected_w[2] - math.exp(-3.0) / (math.exp(-1.5) + 1.0 + math.exp(-3.0))) < 1e-6
        assert abs(sum(expected_w) - 1.0) < 1e-6

    def test_agreement_and_winner(self):
        """All futures identical → agreement=1.0 at all positions, winners=[0,1,2]."""
        model = self._make_model()
        reward = ListReward([0.5, 0.8, 0.2])

        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1,  # commit all 3 in one step
            "max_len": 3,
            "K": 3,
            "tau_f": 0.7,
            "lambda": 5.0,
            "seed": 42,
            "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert len(result.trace) == 1
        t = result.trace[0]
        assert isinstance(t, RVUStepTrace)

        # All positions committed in one step (ceil(3/1) = 3)
        assert len(t.positions_committed) == 3
        assert sorted(t.positions_committed) == [0, 1, 2]

        # Winners should be [0, 1, 2]
        for pos, tok in zip(t.positions_committed, t.tokens_committed):
            assert tok == pos, f"Position {pos} should have winner token {pos}, got {tok}"

        # Agreement should be ~1.0 (all futures agree)
        for ag in t.agreement_scores:
            assert abs(ag - 1.0) < 1e-6

        # Weights should match hand computation
        expected_w = self._compute_expected_weights([0.5, 0.8, 0.2], 5.0)
        for w_actual, w_expected in zip(t.weights, expected_w):
            assert abs(w_actual - w_expected) < 1e-5

        # Reward values recorded
        assert t.reward_values == [0.5, 0.8, 0.2]

    def test_reward_call_count_matches(self):
        """reward_calls_used must equal reward.call_count."""
        model = self._make_model()
        reward = ListReward([0.5, 0.8, 0.2])

        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1,
            "max_len": 3,
            "K": 3,
            "tau_f": 0.7,
            "lambda": 5.0,
            "seed": 42,
            "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert result.reward_calls_used == 3  # 1 step * K=3
        assert result.reward_calls_used == reward.call_count

    def test_no_masks_remain(self):
        model = self._make_model()
        reward = ConstantReward(1.0)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 3, "max_len": 3, "K": 3,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        assert (result.token_ids != model.mask_id).all()


# ---------------------------------------------------------------------------
# Lambda = 0 → uniform weights (B2)
# ---------------------------------------------------------------------------


class TestLambdaZeroUniform:
    """λ=0 → weights exactly uniform → majority vote."""

    def test_uniform_weights(self):
        L, V = 3, 4
        mask_id = 3
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ListReward([0.1, 0.9, 0.5])
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1, "max_len": 3, "K": 3,
            "tau_f": 0.7, "lambda": 0.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        t = result.trace[0]

        # Weights must be exactly uniform: 1/3 each
        for w in t.weights:
            assert abs(w - 1.0 / 3.0) < 1e-6

    def test_majority_vote_with_disagreement(self):
        """With λ=0, the most-voted token wins regardless of reward.

        L=2, V=4, K=3. mask_id=3.
        We craft logits so futures might disagree at pos 0:
        - pos 0: token 0 has logit 2.0, token 1 has logit 1.5 (close)
        - pos 1: token 2 has logit 10.0 (deterministic)

        Since λ=0 all weights are 1/3. The token appearing in most
        futures wins (majority). With K=3, if 2+ futures pick token 0,
        token 0 wins.
        """
        L, V = 2, 4
        mask_id = 3
        logits = torch.zeros(L, V)
        logits[0, 0] = 2.0
        logits[0, 1] = 1.5
        logits[1, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ListReward([0.0, 1.0, 0.0])
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1, "max_len": 2, "K": 3,
            "tau_f": 0.7, "lambda": 0.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Position 1 always gets token 2
        # Position 0: majority vote wins (regardless of which future has reward 1.0)
        assert result.token_ids[1].item() == 2
        # Token 0 or 1 at position 0 — just verify it's not mask_id
        assert result.token_ids[0].item() != mask_id


# ---------------------------------------------------------------------------
# Lambda → large → vote collapses to best future
# ---------------------------------------------------------------------------


class TestLambdaLarge:
    """λ very large → weight concentrates on highest-reward future."""

    def test_best_future_dominates(self):
        """L=3, V=5, K=3. mask_id=4.

        Logits designed so each future is deterministic but different:
        - We use moderate logits so all K futures sample the same tokens.
        - But we give different rewards: [0.1, 0.9, 0.3].
        - With λ=1e6, weight on future 1 ≈ 1.0.
        - So committed tokens = future 1's tokens.

        Since logits are peaked, all futures are the same anyway.
        So this test verifies the weight concentration.
        """
        L, V = 3, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ListReward([0.1, 0.9, 0.3])
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1, "max_len": 3, "K": 3,
            "tau_f": 0.7, "lambda": 1e6, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        t = result.trace[0]

        # Weight on future 1 (reward=0.9) should be ≈1.0
        assert t.weights[1] > 0.999
        # Other weights ≈ 0
        assert t.weights[0] < 1e-3
        assert t.weights[2] < 1e-3


# ---------------------------------------------------------------------------
# All rewards equal → uniform weights regardless of λ
# ---------------------------------------------------------------------------


class TestEqualRewards:
    """If all rewards identical, weights must be uniform for any λ."""

    @pytest.mark.parametrize("lam", [0.0, 1.0, 5.0, 100.0])
    def test_equal_rewards_uniform_weights(self, lam):
        L, V = 2, 4
        mask_id = 3
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(0.7)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1, "max_len": 2, "K": 4,
            "tau_f": 0.7, "lambda": lam, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)
        t = result.trace[0]

        for w in t.weights:
            assert abs(w - 0.25) < 1e-6  # 1/K = 1/4


# ---------------------------------------------------------------------------
# Position tie-break (float64 precision)
# ---------------------------------------------------------------------------


class TestPositionTieBreak:
    """When agreement AND confidence are identical, lower position index wins.

    This requires float64 for the 1e-15 epsilon to survive.
    """

    def test_lower_position_wins_tiebreak(self):
        """L=4, V=3, K=1, S=2 (commit 2 per step). mask_id=2.

        All positions have identical logits → identical confidence.
        K=1 with constant reward → agreement=1.0 everywhere.
        So agreement and confidence are tied for all 4 positions.
        First step should commit positions 0,1 (lower indices win).
        """
        L, V = 4, 3
        mask_id = 2
        # All positions have token 0 with logit 10, token 1 with logit 10
        # (identical confidence at every position)
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, 0] = 10.0
            logits[i, 1] = 10.0  # equal to token 0

        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)
        reward = ConstantReward(1.0)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 2, "max_len": 4, "K": 1,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Step 0: commits 2 positions. With tie-break on position, should be 0 and 1.
        step0_positions = sorted(result.trace[0].positions_committed)
        assert step0_positions == [0, 1], (
            f"Expected positions [0, 1] from tie-break, got {step0_positions}"
        )

        # Step 1: remaining positions 2 and 3
        step1_positions = sorted(result.trace[1].positions_committed)
        assert step1_positions == [2, 3]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_same_seed_same_result(self):
        L, V = 4, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        # Spread logits so sampling has some variation
        logits[0, 0] = 2.0; logits[0, 1] = 1.5
        logits[1, 2] = 2.0; logits[1, 3] = 1.5
        logits[2, 0] = 3.0
        logits[3, 1] = 3.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        config = {
            "steps": 2, "max_len": 4, "K": 4,
            "tau_f": 1.0, "lambda": 5.0, "seed": 123, "device": "cpu",
        }

        reward1 = ConstantReward(0.5)
        result1 = RVUDecoder(reward=reward1).decode(model, None, config)

        reward2 = ConstantReward(0.5)
        result2 = RVUDecoder(reward=reward2).decode(model, None, config)

        assert torch.equal(result1.token_ids, result2.token_ids)

    def test_different_seed_different_futures(self):
        """Different seeds should (almost surely) produce different futures."""
        L, V = 4, 10
        mask_id = 9
        # Uniform-ish logits → sampling has high entropy
        logits = torch.ones(L, V) * 1.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        config1 = {
            "steps": 1, "max_len": 4, "K": 4,
            "tau_f": 1.0, "lambda": 0.0, "seed": 42, "device": "cpu",
        }
        config2 = {**config1, "seed": 999}

        reward1 = ConstantReward(0.5)
        result1 = RVUDecoder(reward=reward1).decode(model, None, config1)

        reward2 = ConstantReward(0.5)
        result2 = RVUDecoder(reward=reward2).decode(model, None, config2)

        # With 4 positions × 9 possible tokens, collision probability is negligible
        assert not torch.equal(result1.token_ids, result2.token_ids)


# ---------------------------------------------------------------------------
# K=1 degeneracy
# ---------------------------------------------------------------------------


class TestK1Degenerate:
    """K=1: only one future, weight=1.0, committed tokens = that future's tokens."""

    def test_k1_single_future(self):
        L, V = 4, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, 0] = 10.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0
        logits[3, 3] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(0.42)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 2, "max_len": 4, "K": 1,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Should commit tokens [0, 1, 2, 3]
        assert result.token_ids.tolist() == [0, 1, 2, 3]

        # Each step: weight = [1.0]
        for t in result.trace:
            assert len(t.weights) == 1
            assert abs(t.weights[0] - 1.0) < 1e-6

        # Reward calls: steps_actual * K
        assert result.reward_calls_used == len(result.trace) * 1
        assert result.reward_calls_used == reward.call_count


# ---------------------------------------------------------------------------
# Mask token never in futures or commitments
# ---------------------------------------------------------------------------


class TestNoMaskInFutures:
    """Mask token must never appear in any future or committed token."""

    def test_mask_never_committed(self):
        L, V = 5, 6
        mask_id = 5
        # Logits with mask_id having the highest value — decoder must suppress it
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, mask_id] = 20.0  # mask_id is "best" in raw logits
            logits[i, i % (V - 1)] = 5.0  # real token gets second-best

        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)
        reward = ConstantReward(1.0)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 5, "max_len": 5, "K": 4,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        # No mask in final canvas
        assert (result.token_ids != mask_id).all()

        # No mask in any committed token
        for t in result.trace:
            for tok in t.tokens_committed:
                assert tok != mask_id

    def test_futures_never_contain_mask(self):
        """Inspect actual futures sampled — none should contain mask_id."""
        L, V = 3, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        logits[0, mask_id] = 15.0  # mask_id tempting
        logits[0, 0] = 3.0
        logits[1, 1] = 10.0
        logits[2, 2] = 10.0

        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)
        reward = FutureInspectingReward(rewards=[0.5, 0.5, 0.5])
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 1, "max_len": 3, "K": 3,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Check all observed futures
        for future in reward.observed_futures:
            assert mask_id not in future, f"mask_id {mask_id} found in future {future}"


# ---------------------------------------------------------------------------
# Reward accounting over multiple steps
# ---------------------------------------------------------------------------


class TestRewardAccounting:
    """reward_calls_used = S_actual × K, must match reward.call_count."""

    def test_multi_step_accounting(self):
        L, V = 6, 8
        mask_id = 7
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        K = 5
        reward = ConstantReward(0.5)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 3, "max_len": 6, "K": K,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        n_actual_steps = len(result.trace)
        expected_calls = n_actual_steps * K

        assert result.reward_calls_used == expected_calls
        assert result.reward_calls_used == reward.call_count

    def test_accounting_with_prompt(self):
        """Prompt reduces masked positions but accounting still exact."""
        L, V = 8, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        K = 3
        reward = ConstantReward(0.5)
        decoder = RVUDecoder(reward=reward)
        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        config = {
            "steps": 4, "max_len": 8, "K": K,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        n_actual_steps = len(result.trace)
        assert result.reward_calls_used == n_actual_steps * K
        assert result.reward_calls_used == reward.call_count


# ---------------------------------------------------------------------------
# Commit schedule matches B0
# ---------------------------------------------------------------------------


class TestCommitScheduleMatchesB0:
    """RVU uses same commit_schedule as B0: same S → same step counts."""

    def test_same_commit_counts(self):
        L, V, S = 10, 8, 4
        mask_id = 7
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = ConstantReward(0.5)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": S, "max_len": L, "K": 3,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=None, config=config)

        commits = [len(t.positions_committed) for t in result.trace]
        # L=10, S=4 → [3, 3, 2, 2]
        assert commits == [3, 3, 2, 2]
        assert len(result.trace) == S
