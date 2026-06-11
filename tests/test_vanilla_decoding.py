"""Tests for B0 vanilla confidence-based decoding."""

import pytest
import torch
import torch.nn.functional as F
from math import ceil
from unittest.mock import MagicMock
from typing import List

from rvu.decoding.base import commit_schedule, StepTrace
from rvu.decoding.vanilla import VanillaDecoder
from rvu.models.base import MDLM
from rvu.utils import detokenize, detokenize_completion


# ---------------------------------------------------------------------------
# commit_schedule unit tests (hand-verified)
# ---------------------------------------------------------------------------


class TestCommitSchedule:
    """Hand-verify: L=10 masks, S=4 → commits 3,3,2,2."""

    def test_10_masks_4_steps(self):
        remaining = 10
        steps = 4
        commits = []
        for s in range(steps):
            n = commit_schedule(remaining, steps - s)
            commits.append(n)
            remaining -= n
        assert commits == [3, 3, 2, 2]
        assert remaining == 0

    def test_5_masks_5_steps(self):
        """One per step."""
        remaining = 5
        steps = 5
        commits = []
        for s in range(steps):
            n = commit_schedule(remaining, steps - s)
            commits.append(n)
            remaining -= n
        assert commits == [1, 1, 1, 1, 1]
        assert remaining == 0

    def test_7_masks_3_steps(self):
        """ceil(7/3)=3, ceil(4/2)=2, ceil(2/1)=2."""
        remaining = 7
        steps = 3
        commits = []
        for s in range(steps):
            n = commit_schedule(remaining, steps - s)
            commits.append(n)
            remaining -= n
        assert commits == [3, 2, 2]
        assert remaining == 0

    def test_1_mask_1_step(self):
        assert commit_schedule(1, 1) == 1

    def test_12_masks_4_steps(self):
        """Exact division: 3,3,3,3."""
        remaining = 12
        steps = 4
        commits = []
        for s in range(steps):
            n = commit_schedule(remaining, steps - s)
            commits.append(n)
            remaining -= n
        assert commits == [3, 3, 3, 3]
        assert remaining == 0

    def test_always_completes(self):
        """For many (masks, steps) combos, always exhausts all masks."""
        for masks in range(1, 30):
            for steps in range(1, masks + 1):
                remaining = masks
                for s in range(steps):
                    n = commit_schedule(remaining, steps - s)
                    remaining -= n
                assert remaining == 0, f"Failed: masks={masks}, steps={steps}"

    def test_zero_steps_raises(self):
        with pytest.raises(AssertionError):
            commit_schedule(5, 0)

    def test_zero_masks_raises(self):
        with pytest.raises(AssertionError):
            commit_schedule(0, 3)


# ---------------------------------------------------------------------------
# Mock model for controlled testing
# ---------------------------------------------------------------------------


class MockMDLM(MDLM):
    """Deterministic mock model returning pre-set logits.

    For each position, logits are set so that the argmax and confidence
    are fully controlled by the test.
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        mask_id: int,
        logits_fn=None,
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.mask_id = mask_id
        self.device = torch.device("cpu")
        self._logits_fn = logits_fn

        # Minimal mock tokenizer
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = None
        self.tokenizer.decode = lambda ids, **kw: " ".join(str(i) for i in ids)

    def logits(self, canvas):
        if self._logits_fn is not None:
            return self._logits_fn(canvas)
        # Default: uniform logits (all positions equal confidence)
        return torch.zeros(canvas.shape[0], self.vocab_size)


def make_confidence_logits(L: int, V: int, mask_id: int):
    """Create a logits_fn that gives position i confidence proportional to i.

    Position 0 gets highest confidence, position L-1 lowest.
    This means the decoder should commit positions in order 0, 1, 2, ...
    Argmax token for position i is token (i + 1) to avoid mask_id=0 collisions.
    """
    def logits_fn(canvas):
        logits = torch.zeros(L, V)
        for i in range(L):
            if canvas[i] == mask_id:
                # Make token (i+1) the argmax with confidence decreasing by position
                confidence_logit = float(L - i)
                logits[i, (i + 1) % V] = confidence_logit
        return logits
    return logits_fn


# ---------------------------------------------------------------------------
# VanillaDecoder integration tests
# ---------------------------------------------------------------------------


class TestVanillaDecoderBasic:
    def test_decode_returns_decode_result(self):
        model = MockMDLM(vocab_size=10, max_len=8, mask_id=0)
        decoder = VanillaDecoder()
        config = {"steps": 4, "max_len": 8, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)
        assert result.token_ids.shape == (8,)
        assert result.reward_calls_used == 0
        assert isinstance(result.trace, list)
        assert isinstance(result.text, str)

    def test_no_masks_remain(self):
        """After decoding, no mask tokens in canvas."""
        # Use mask_id=9 so uniform-logits argmax (token 0) won't collide
        model = MockMDLM(vocab_size=10, max_len=8, mask_id=9)
        decoder = VanillaDecoder()
        config = {"steps": 8, "max_len": 8, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)
        assert (result.token_ids != model.mask_id).all()

    def test_mask_id_suppressed(self):
        """Even if model puts max logit on mask_id, decoder must never commit it."""
        L, V = 6, 10
        mask_id = 3

        def logits_putting_max_on_mask(canvas):
            logits = torch.zeros(L, V)
            for i in range(L):
                if canvas[i] == mask_id:
                    # mask_id token gets highest logit
                    logits[i, mask_id] = 10.0
                    # next-best token gets a lower logit
                    logits[i, (mask_id + 1) % V] = 5.0
            return logits

        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=logits_putting_max_on_mask,
        )
        decoder = VanillaDecoder()
        config = {"steps": L, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        # No position should hold mask_id
        assert (result.token_ids != mask_id).all()
        # Should have committed to the next-best token (mask_id+1)
        expected_token = (mask_id + 1) % V
        assert (result.token_ids == expected_token).all()


class TestVanillaDecoderSchedule:
    """Verify commit counts per step match commit_schedule."""

    def test_10_masks_4_steps_trace(self):
        L, V, S = 10, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        commits_per_step = [len(t.positions_committed) for t in result.trace]
        assert commits_per_step == [3, 3, 2, 2]

    def test_completes_in_exactly_S_steps(self):
        L, V, S = 10, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert len(result.trace) == S
        assert result.trace[-1].masks_remaining_after == 0
        assert (result.token_ids != mask_id).all()

    def test_5_masks_5_steps(self):
        L, V, S = 5, 10, 5
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        commits_per_step = [len(t.positions_committed) for t in result.trace]
        assert commits_per_step == [1, 1, 1, 1, 1]

    def test_more_steps_than_masks(self):
        """S > L: should finish early, fewer trace entries than S."""
        L, V, S = 4, 10, 10
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        # Should still complete with no masks
        assert (result.token_ids != mask_id).all()
        # Fewer steps used than S since each step commits at least 1
        assert len(result.trace) <= L


class TestVanillaDecoderConfidence:
    """Verify highest-confidence positions are committed first."""

    def test_confidence_ordering(self):
        """Positions with higher confidence logits should commit first."""
        L, V, S = 6, 10, 6
        mask_id = 0

        # Position 0 has highest confidence, 5 has lowest
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        # With 1 commit per step, positions should be committed in order 0,1,2,3,4,5
        committed_order = [t.positions_committed[0] for t in result.trace]
        assert committed_order == [0, 1, 2, 3, 4, 5]

    def test_argmax_tokens_correct(self):
        """Each position should be committed to its argmax token."""
        L, V, S = 4, 10, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        # make_confidence_logits sets argmax for position i = (i+1) % V
        for pos in range(L):
            expected_token = (pos + 1) % V
            assert result.token_ids[pos].item() == expected_token

    def test_confidences_in_trace_are_decreasing_within_step(self):
        """Within a step, committed positions should be ordered by confidence (desc)."""
        L, V, S = 8, 20, 2
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        for step_trace in result.trace:
            confs = step_trace.confidences
            # topk returns in descending order
            assert confs == sorted(confs, reverse=True)


class TestVanillaDecoderTrace:
    """Verify trace metadata consistency."""

    def test_trace_masks_remaining_consistent(self):
        L, V, S = 10, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        for i, t in enumerate(result.trace):
            n_committed = len(t.positions_committed)
            assert t.masks_remaining_after == t.masks_remaining_before - n_committed
            if i > 0:
                assert t.masks_remaining_before == result.trace[i - 1].masks_remaining_after

        assert result.trace[0].masks_remaining_before == L
        assert result.trace[-1].masks_remaining_after == 0

    def test_no_position_committed_twice(self):
        L, V, S = 10, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        all_positions = []
        for t in result.trace:
            all_positions.extend(t.positions_committed)
        assert len(all_positions) == len(set(all_positions)) == L

    def test_step_indices_sequential(self):
        L, V, S = 6, 10, 3
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=None, config=config)

        assert [t.step for t in result.trace] == list(range(S))


class TestVanillaDecoderPrompt:
    """Verify prompt pre-commitment."""

    def test_prompt_positions_preserved(self):
        L, V, S = 8, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        prompt = torch.tensor([5, 6, 7], dtype=torch.long)
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        # Prompt tokens untouched
        assert result.token_ids[0].item() == 5
        assert result.token_ids[1].item() == 6
        assert result.token_ids[2].item() == 7

    def test_prompt_reduces_masks(self):
        """Prompt positions are pre-committed, so fewer masks to fill."""
        L, V, S = 8, 20, 4
        mask_id = 0
        model = MockMDLM(
            vocab_size=V, max_len=L, mask_id=mask_id,
            logits_fn=make_confidence_logits(L, V, mask_id),
        )
        prompt = torch.tensor([5, 6, 7], dtype=torch.long)
        decoder = VanillaDecoder()
        config = {"steps": S, "max_len": L, "device": "cpu"}
        result = decoder.decode(model, prompt_ids=prompt, config=config)

        # First trace entry should show 5 masks (8 - 3 prompt)
        assert result.trace[0].masks_remaining_before == 5

        # No prompt positions in any committed set
        all_committed = []
        for t in result.trace:
            all_committed.extend(t.positions_committed)
        assert 0 not in all_committed
        assert 1 not in all_committed
        assert 2 not in all_committed


class TestDetokenize:
    """Test EOS-stripping utility."""

    def test_strip_after_eos(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 50256
        tokenizer.decode.return_value = "hello world"
        ids = torch.tensor([100, 200, 50256, 300, 400])
        result = detokenize(ids, tokenizer)
        tokenizer.decode.assert_called_once()
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [100, 200]

    def test_no_eos(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 50256
        tokenizer.decode.return_value = "all tokens"
        ids = torch.tensor([100, 200, 300])
        detokenize(ids, tokenizer)
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [100, 200, 300]

    def test_eos_none(self):
        tokenizer = MagicMock()
        tokenizer.eos_token_id = None
        tokenizer.decode.return_value = "all tokens"
        ids = torch.tensor([100, 200, 300])
        detokenize(ids, tokenizer)
        call_args = tokenizer.decode.call_args
        assert call_args[0][0] == [100, 200, 300]
