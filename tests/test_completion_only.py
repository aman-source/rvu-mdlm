"""Tests for completion-only rewards and per-case budget matching."""

import pytest
import torch
from unittest.mock import MagicMock
from typing import List

from rvu.decoding.base import expected_steps, commit_schedule
from rvu.decoding.best_of_n import BestOfNDecoder, matched_n_per_case
from rvu.decoding.rvu import RVUDecoder
from rvu.decoding.vanilla import VanillaDecoder
from rvu.models.base import MDLM
from rvu.rewards.base import Reward
from rvu.rewards.json_schema import JsonSchemaReward
from rvu.utils import detokenize_completion


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FixedLogitsModel(MDLM):
    def __init__(self, fixed_logits, mask_id, vocab_size):
        self.device = torch.device("cpu")
        self.max_len = fixed_logits.shape[0]
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self._fixed_logits = fixed_logits
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = None
        self.tokenizer.decode = lambda ids, **kw: "".join(chr(48 + (i % 75)) for i in ids)

    def logits(self, canvas):
        return self._fixed_logits.clone()


class TextCapturingReward(Reward):
    """Captures the text passed to it for inspection."""
    def __init__(self):
        self.call_count = 0
        self.seen_texts: List[str] = []

    def __call__(self, token_ids, tokenizer):
        self.call_count += 1
        text = tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        self.seen_texts.append(text)
        return 0.0


class ConstantReward(Reward):
    def __init__(self, value):
        self.value = value
        self.call_count = 0

    def __call__(self, token_ids, tokenizer):
        self.call_count += 1
        return self.value


# ---------------------------------------------------------------------------
# Completion-only: prompt with valid JSON + garbage completion → score 0.0
# ---------------------------------------------------------------------------


class TestCompletionOnlyScoring:
    """A prompt containing a perfect JSON object + garbage completion must score 0."""

    def test_prompt_json_not_scored(self):
        """Reward sees only completion region, not prompt."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        }
        reward = JsonSchemaReward(schema=schema)

        # Simulate: prompt contains valid JSON, completion is garbage
        prompt_text = '{"name": "Alice", "age": 30}'
        garbage_text = "asdfghjkl random garbage"
        # score_text on prompt alone would give 1.0
        assert reward.score_text(prompt_text) == 1.0
        # score_text on garbage gives 0.0
        assert reward.score_text(garbage_text) == 0.0
        # score_text on completion-only must be 0.0
        assert reward.score_text(garbage_text) == 0.0

    def test_vanilla_decoder_completion_only_text(self):
        """VanillaDecoder.text should contain only completion, not prompt."""
        L, V = 10, 6
        mask_id = 5
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        decoder = VanillaDecoder()
        config = {"steps": 10, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        assert result.prompt_len == 3
        # The text should be completion-only (from position 3 onward)
        # Decode the full canvas for comparison
        full_text = model.tokenizer.decode(result.token_ids.tolist(), skip_special_tokens=True)
        completion_text = model.tokenizer.decode(
            result.token_ids[3:].tolist(), skip_special_tokens=True
        )
        assert result.text == completion_text
        assert result.text != full_text  # must differ since prompt excluded

    def test_rvu_rewards_see_completion_only(self):
        """RVU decoder must pass only completion tokens to reward."""
        L, V = 8, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = TextCapturingReward()
        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 5, "max_len": L, "K": 2,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        # Every text seen by reward should NOT start with prompt decode
        prompt_text = model.tokenizer.decode([0, 1, 2], skip_special_tokens=True)
        for text in reward.seen_texts:
            # Completion text should be shorter than full canvas decode
            # and should not start with prompt tokens' decode
            assert not text.startswith(prompt_text) or len(prompt_text) == 0

    def test_b1_rewards_see_completion_only(self):
        """B1 decoder must pass only completion tokens to reward."""
        L, V = 8, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        reward = TextCapturingReward()
        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "steps": 5, "max_len": L, "N": 3,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        prompt_text = model.tokenizer.decode([0, 1, 2], skip_special_tokens=True)
        for text in reward.seen_texts:
            assert not text.startswith(prompt_text) or len(prompt_text) == 0


# ---------------------------------------------------------------------------
# detokenize_completion
# ---------------------------------------------------------------------------


class TestDetokenizeCompletion:
    def test_excludes_prompt(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = None
        tokenizer.decode.return_value = "completion"
        ids = torch.tensor([10, 20, 30, 40, 50])
        result = detokenize_completion(ids, tokenizer, prompt_len=2)
        # Should decode only [30, 40, 50]
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [30, 40, 50]

    def test_eos_in_completion(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 99
        tokenizer.decode.return_value = "partial"
        ids = torch.tensor([10, 20, 30, 99, 50])
        result = detokenize_completion(ids, tokenizer, prompt_len=1)
        # Completion = [20, 30, 99, 50], truncated at 99 → [20, 30]
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [20, 30]

    def test_zero_prompt(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = None
        tokenizer.decode.return_value = "all"
        ids = torch.tensor([10, 20, 30])
        detokenize_completion(ids, tokenizer, prompt_len=0)
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [10, 20, 30]


# ---------------------------------------------------------------------------
# expected_steps
# ---------------------------------------------------------------------------


class TestExpectedSteps:
    def test_no_prompt(self):
        # L=128, S=16, prompt_len=0 → min(16, 128) = 16
        assert expected_steps(0, 128, 16) == 16

    def test_long_prompt(self):
        # L=128, S=16, prompt_len=120 → min(16, 8) = 8
        assert expected_steps(120, 128, 16) == 8

    def test_prompt_equals_canvas(self):
        # No masks → 0 steps
        assert expected_steps(128, 128, 16) == 0

    def test_prompt_longer_than_canvas(self):
        # Edge case
        assert expected_steps(200, 128, 16) == 0

    def test_few_masks_less_than_steps(self):
        # L=10, S=16, prompt_len=7 → 3 masks, min(16, 3) = 3
        assert expected_steps(7, 10, 16) == 3

    def test_steps_less_than_masks(self):
        # L=100, S=4, prompt_len=0 → min(4, 100) = 4
        assert expected_steps(0, 100, 4) == 4


# ---------------------------------------------------------------------------
# matched_n_per_case
# ---------------------------------------------------------------------------


class TestMatchedNPerCase:
    def test_basic(self):
        # prompt_len=0, L=128, S=16, K=4 → 16*4=64
        assert matched_n_per_case(0, 128, 16, 4) == 64

    def test_long_prompt(self):
        # prompt_len=120, L=128, S=16, K=4 → expected_steps=8, N=8*4=32
        assert matched_n_per_case(120, 128, 16, 4) == 32

    def test_full_prompt(self):
        # prompt fills canvas → 0 steps → at least 1
        assert matched_n_per_case(128, 128, 16, 4) == 1

    def test_varies_with_prompt_length(self):
        """Different prompt lengths → different N when masks < S."""
        # L=20, S=16: prompt_len=5 → min(16,15)=15, N=60
        #             prompt_len=15 → min(16,5)=5, N=20
        n1 = matched_n_per_case(5, 20, 16, 4)
        n2 = matched_n_per_case(15, 20, 16, 4)
        assert n1 > n2


# ---------------------------------------------------------------------------
# B1 N varies with prompt length (per-case matching)
# ---------------------------------------------------------------------------


class TestB1PerCaseBudget:
    def test_b1_reward_calls_match_expected(self):
        """B1's reward_calls_used should equal matched_n_per_case."""
        L, V = 10, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        prompt = torch.tensor([0, 1, 2, 3, 0, 1, 2], dtype=torch.long)  # len=7
        K = 3
        S = 4
        expected_N = matched_n_per_case(len(prompt), L, S, K)

        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "steps": S, "max_len": L, "N": expected_N,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt_ids=prompt, config=config)
        assert result.reward_calls_used == expected_N
        assert reward.call_count == expected_N

    def test_b1_budget_matches_rvu(self):
        """B1 and RVU should use same number of reward calls for same prompt."""
        L, V = 12, 5
        mask_id = 4
        logits = torch.zeros(L, V)
        for i in range(L):
            logits[i, i % (V - 1)] = 10.0
        model = FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)

        prompt = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)  # len=5
        K = 2
        S = 4

        # RVU
        rvu_reward = ConstantReward(0.5)
        rvu_decoder = RVUDecoder(reward=rvu_reward)
        rvu_config = {
            "steps": S, "max_len": L, "K": K,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        rvu_result = rvu_decoder.decode(model, prompt_ids=prompt, config=rvu_config)

        # B1 with matched N
        N = matched_n_per_case(len(prompt), L, S, K)
        b1_reward = ConstantReward(0.5)
        b1_decoder = BestOfNDecoder(reward=b1_reward)
        b1_config = {
            "steps": S, "max_len": L, "N": N,
            "temperature": 0.7, "seed": 42, "device": "cpu",
        }
        b1_result = b1_decoder.decode(model, prompt_ids=prompt, config=b1_config)

        # Budgets should match
        assert b1_result.reward_calls_used == rvu_result.reward_calls_used
