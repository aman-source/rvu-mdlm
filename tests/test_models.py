"""Tests for MDLM interface and tiny stand-in model."""

import pytest
import torch

from rvu.models.tiny import TinyMDLM


@pytest.fixture(scope="module")
def model():
    """Create a small TinyMDLM for testing (smaller than default for speed)."""
    return TinyMDLM(
        d_model=64,
        n_heads=2,
        n_layers=2,
        d_ff=128,
        max_len=32,
        device="cpu",
    )


class TestTinyMDLMInit:
    def test_has_tokenizer(self, model: TinyMDLM):
        assert model.tokenizer is not None

    def test_has_mask_id(self, model: TinyMDLM):
        assert isinstance(model.mask_id, int)
        assert model.mask_id >= 0

    def test_mask_token_decodes(self, model: TinyMDLM):
        decoded = model.tokenizer.decode([model.mask_id])
        assert "[MASK]" in decoded

    def test_device_is_cpu(self, model: TinyMDLM):
        assert model.device == torch.device("cpu")

    def test_max_len(self, model: TinyMDLM):
        assert model.max_len == 32

    def test_vocab_size_matches_tokenizer(self, model: TinyMDLM):
        assert model.vocab_size == len(model.tokenizer)

    def test_param_count_positive(self, model: TinyMDLM):
        count = model.param_count()
        assert count > 0
        # Should be smallish for our test config
        assert count < 10_000_000


class TestTinyMDLMLogits:
    def test_logits_shape(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=16)
        logits = model.logits(canvas)
        assert logits.shape == (16, model.vocab_size)

    def test_logits_dtype(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=8)
        logits = model.logits(canvas)
        assert logits.dtype == torch.float32

    def test_logits_finite(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=8)
        logits = model.logits(canvas)
        assert torch.isfinite(logits).all()

    def test_logits_with_partial_mask(self, model: TinyMDLM):
        """Canvas with some committed tokens and some masks."""
        canvas = model.fully_masked_canvas(length=10)
        # Commit first 3 positions to token ID 100
        canvas[:3] = 100
        logits = model.logits(canvas)
        assert logits.shape == (10, model.vocab_size)

    def test_logits_fully_committed(self, model: TinyMDLM):
        """Canvas with no masks — model should still return logits."""
        canvas = torch.full((8,), 100, dtype=torch.long)
        logits = model.logits(canvas)
        assert logits.shape == (8, model.vocab_size)

    def test_logits_max_len(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas()  # default = max_len
        logits = model.logits(canvas)
        assert logits.shape == (model.max_len, model.vocab_size)

    def test_logits_exceeds_max_len_raises(self, model: TinyMDLM):
        canvas = torch.full((model.max_len + 1,), model.mask_id, dtype=torch.long)
        with pytest.raises(AssertionError):
            model.logits(canvas)


class TestMDLMHelpers:
    def test_fully_masked_canvas_default(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas()
        assert canvas.shape == (model.max_len,)
        assert (canvas == model.mask_id).all()
        assert canvas.device == model.device

    def test_fully_masked_canvas_custom_length(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=5)
        assert canvas.shape == (5,)
        assert (canvas == model.mask_id).all()

    def test_is_masked_all_masked(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=8)
        mask = model.is_masked(canvas)
        assert mask.all()

    def test_is_masked_none_masked(self, model: TinyMDLM):
        canvas = torch.full((8,), 100, dtype=torch.long)
        mask = model.is_masked(canvas)
        assert not mask.any()

    def test_is_masked_partial(self, model: TinyMDLM):
        canvas = model.fully_masked_canvas(length=6)
        canvas[0] = 100
        canvas[3] = 200
        mask = model.is_masked(canvas)
        assert mask.tolist() == [False, True, True, False, True, True]


class TestDeterminism:
    def test_same_input_same_output(self, model: TinyMDLM):
        """Same canvas should produce identical logits (no dropout in eval)."""
        canvas = model.fully_masked_canvas(length=10)
        logits1 = model.logits(canvas.clone())
        logits2 = model.logits(canvas.clone())
        assert torch.allclose(logits1, logits2)
