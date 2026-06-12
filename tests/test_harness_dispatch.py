"""Tests for harness model dispatch based on config['model'] field."""

import pytest
from unittest.mock import MagicMock, patch

from rvu.eval.harness import build_model, VALID_MODELS


class TestBuildModel:
    def test_tiny_routes_to_tiny_mdlm(self):
        """model: tiny -> TinyMDLM (regression)."""
        config = {"model": "tiny", "max_len": 32, "device": "cpu"}
        model = build_model(config)
        from rvu.models.tiny import TinyMDLM
        assert isinstance(model, TinyMDLM)
        assert model.max_len == 32

    def test_tiny_is_default(self):
        """Missing model key defaults to tiny."""
        config = {"max_len": 32, "device": "cpu"}
        model = build_model(config)
        from rvu.models.tiny import TinyMDLM
        assert isinstance(model, TinyMDLM)

    def test_llada_calls_constructor(self):
        """model: llada -> LLaDAMDLM constructor called with model_path and device."""
        mock_cls = MagicMock()
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        config = {
            "model": "llada",
            "model_path": "some/model/path",
            "max_len": 256,
            "device": "cuda:0",
        }

        with patch("rvu.eval.harness._import_llada", return_value=mock_cls):
            result = build_model(config)

        mock_cls.assert_called_once_with(
            model_path="some/model/path",
            max_len=256,
            device="cuda:0",
        )
        assert result is mock_instance

    def test_llada_default_model_path(self):
        """model: llada without model_path uses default."""
        mock_cls = MagicMock()
        mock_cls.return_value = MagicMock()

        config = {"model": "llada", "max_len": 128, "device": "cuda:0"}

        with patch("rvu.eval.harness._import_llada", return_value=mock_cls):
            build_model(config)

        mock_cls.assert_called_once_with(
            model_path="GSAI-ML/LLaDA-8B-Instruct",
            max_len=128,
            device="cuda:0",
        )

    def test_unknown_model_raises(self):
        """Unknown model name -> ValueError with valid options listed."""
        config = {"model": "gpt5", "max_len": 64, "device": "cpu"}
        with pytest.raises(ValueError, match="Unknown model.*gpt5"):
            build_model(config)

    def test_unknown_model_lists_valid_options(self):
        """Error message includes valid model names."""
        config = {"model": "bad", "max_len": 64, "device": "cpu"}
        with pytest.raises(ValueError, match="tiny.*llada"):
            build_model(config)
