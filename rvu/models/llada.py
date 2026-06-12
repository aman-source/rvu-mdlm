"""LLaDA-8B-Instruct adapter behind the MDLM interface.

Model: GSAI-ML/LLaDA-8B-Instruct
Mask token ID: 126336
Uses the model's chat template for Instruct formatting.
Single forward pass on canvas → logits. bf16, device from config.
"""

from typing import Optional

import torch
from torch import Tensor

from .base import MDLM

# LLaDA constants
LLADA_MASK_TOKEN_ID = 126336
LLADA_MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


class LLaDAMDLM(MDLM):
    """Adapter for LLaDA-8B-Instruct as an MDLM.

    Args:
        model_path: HuggingFace model name or local path.
        max_len: Maximum sequence length.
        device: torch device string (e.g., "cuda:0").
    """

    def __init__(
        self,
        model_path: str = LLADA_MODEL_NAME,
        max_len: int = 256,
        device: str = "cuda:0",
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = torch.device(device)
        self.max_len = max_len

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.mask_id = LLADA_MASK_TOKEN_ID
        self.vocab_size = len(self.tokenizer)

        # Load model in bf16
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self._model.eval()

    @torch.no_grad()
    def logits(self, canvas: Tensor) -> Tensor:
        """Forward pass: canvas [L] → logits [L, V].

        Single forward pass. Returns float32 logits for stable softmax.
        """
        canvas = canvas.to(self.device)
        # Model expects batch dimension: [1, L]
        input_ids = canvas.unsqueeze(0)

        outputs = self._model(input_ids=input_ids)
        # outputs.logits: [1, L, V] in bf16 → cast to float32
        return outputs.logits[0].float()

    @torch.no_grad()
    def logits_batch(self, canvases: Tensor) -> Tensor:
        """Batched forward pass: canvases [B, L] → logits [B, L, V].

        True batched inference. Returns float32 logits.
        """
        canvases = canvases.to(self.device)
        outputs = self._model(input_ids=canvases)
        return outputs.logits.float()  # [B, L, V]

    def format_prompt(self, user_message: str) -> str:
        """Format a user message using the chat template.

        Returns the formatted string ready for tokenization.
        """
        messages = [{"role": "user", "content": user_message}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def param_count(self) -> int:
        """Total number of model parameters."""
        return sum(p.numel() for p in self._model.parameters())
