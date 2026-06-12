"""Tests for forward-pass accounting across all decoders."""

import torch
from unittest.mock import MagicMock

from rvu.decoding.base import expected_steps
from rvu.decoding.vanilla import VanillaDecoder
from rvu.decoding.rvu import RVUDecoder
from rvu.decoding.best_of_n import BestOfNDecoder, matched_n_per_case
from rvu.models.base import MDLM
from rvu.rewards.base import Reward


class FixedLogitsModel(MDLM):
    def __init__(self, fixed_logits, mask_id, vocab_size):
        self.device = torch.device("cpu")
        self.max_len = fixed_logits.shape[0]
        self.mask_id = mask_id
        self.vocab_size = vocab_size
        self._fixed_logits = fixed_logits
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = None
        self.tokenizer.decode = lambda ids, **kw: " ".join(str(i) for i in ids)

    def logits(self, canvas):
        return self._fixed_logits.clone()


class ConstantReward(Reward):
    def __init__(self, value):
        self.value = value
        self.call_count = 0

    def __call__(self, token_ids, tokenizer):
        self.call_count += 1
        return self.value


def _make_model(L=8, V=5, mask_id=4):
    logits = torch.zeros(L, V)
    for i in range(L):
        logits[i, i % (V - 1)] = 10.0
    return FixedLogitsModel(fixed_logits=logits, mask_id=mask_id, vocab_size=V)


class TestB0ForwardPasses:
    def test_equals_expected_steps(self):
        model = _make_model(L=10)
        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        decoder = VanillaDecoder()
        config = {"steps": 8, "max_len": 10, "device": "cpu"}
        result = decoder.decode(model, prompt, config)
        es = expected_steps(len(prompt), 10, 8)
        assert result.forward_passes == es

    def test_no_prompt(self):
        model = _make_model(L=6)
        decoder = VanillaDecoder()
        config = {"steps": 4, "max_len": 6, "device": "cpu"}
        result = decoder.decode(model, None, config)
        assert result.forward_passes == expected_steps(0, 6, 4)


class TestRVUForwardPasses:
    def test_equals_expected_steps(self):
        model = _make_model(L=8)
        prompt = torch.tensor([0, 1], dtype=torch.long)
        reward = ConstantReward(0.5)
        decoder = RVUDecoder(reward=reward)
        config = {
            "steps": 6, "max_len": 8, "K": 3,
            "tau_f": 0.7, "lambda": 5.0, "seed": 42, "device": "cpu",
        }
        result = decoder.decode(model, prompt, config)
        es = expected_steps(len(prompt), 8, 6)
        assert result.forward_passes == es


class TestB1ForwardPasses:
    def test_equals_n_times_expected_steps(self):
        """B1 forward passes = N × expected_steps (each rollout does expected_steps passes)."""
        model = _make_model(L=8)
        prompt = torch.tensor([0, 1, 2], dtype=torch.long)
        N = 6
        S = 4
        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": N, "steps": S, "max_len": 8,
            "temperature": 0.7, "seed": 42, "device": "cpu",
            "b1_batch_size": 4,
        }
        result = decoder.decode(model, prompt, config)
        es = expected_steps(len(prompt), 8, S)
        assert result.forward_passes == N * es

    def test_batch_size_1(self):
        """Sequential: same forward pass count."""
        model = _make_model(L=6)
        N = 5
        S = 3
        reward = ConstantReward(0.5)
        decoder = BestOfNDecoder(reward=reward)
        config = {
            "N": N, "steps": S, "max_len": 6,
            "temperature": 0.7, "seed": 42, "device": "cpu",
            "b1_batch_size": 1,
        }
        result = decoder.decode(model, None, config)
        es = expected_steps(0, 6, S)
        assert result.forward_passes == N * es
