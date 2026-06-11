"""B0: Vanilla confidence-based decoding (no reward).

Supports two token-selection modes:
  - greedy (default): argmax at each committed position.
  - sampled: sample from softmax(logits / temperature) at each committed position.
    Used internally by B1 (best-of-N) to produce diverse rollouts.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from rvu.models.base import MDLM
from rvu.decoding.base import DecodeResult, Decoder, StepTrace, commit_schedule
from rvu.utils import detokenize_completion


class VanillaDecoder(Decoder):
    """Confidence-based unmasking.

    Each step:
    1. Forward pass -> logits at all positions.
    2. At masked positions, confidence = max softmax prob over vocab.
    3. Commit top-n by confidence to their selected tokens.
    4. n = ceil(remaining_masks / remaining_steps).

    Token selection at committed positions:
    - greedy (sample=False): argmax token.
    - sampled (sample=True): draw from softmax(logits / temperature).
    """

    def decode(
        self,
        model: MDLM,
        prompt_ids: Optional[Tensor],
        config: Dict[str, Any],
    ) -> DecodeResult:
        steps: int = config["steps"]
        max_len: int = config.get("max_len", model.max_len)
        device = torch.device(config.get("device", "cpu"))
        sample: bool = config.get("sample", False)
        temperature: float = config.get("temperature", 0.7)
        seed: Optional[int] = config.get("seed", None)

        # Build initial canvas
        canvas = model.fully_masked_canvas(length=max_len).to(device)

        # Pre-commit prompt positions
        prompt_len = 0
        if prompt_ids is not None and len(prompt_ids) > 0:
            prompt_len = len(prompt_ids)
            canvas[:prompt_len] = prompt_ids.to(device)

        # Generator for sampled mode
        gen: Optional[torch.Generator] = None
        if sample and seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(seed)

        trace: List[StepTrace] = []

        for step_idx in range(steps):
            mask = model.is_masked(canvas)
            n_masked = int(mask.sum().item())

            if n_masked == 0:
                break

            steps_left = steps - step_idx
            n_commit = commit_schedule(n_masked, steps_left)

            # Forward pass
            logits = model.logits(canvas)  # [L, V]

            # Suppress mask token: never commit a mask_id token
            logits[:, model.mask_id] = float("-inf")

            probs = F.softmax(logits, dim=-1)  # [L, V]

            # Confidence and argmax at masked positions only
            max_probs, argmax_tokens = probs.max(dim=-1)  # [L], [L]

            # Zero out non-masked positions so they're never selected
            confidence = torch.where(mask, max_probs, torch.tensor(0.0, device=device))

            # Top-n by confidence among masked positions
            _, top_indices = confidence.topk(min(n_commit, n_masked))

            if sample:
                # Sample tokens from temperature-scaled distribution
                sampling_logits = logits / temperature
                sampling_probs = F.softmax(sampling_logits, dim=-1)
                selected_tokens = torch.zeros(len(top_indices), dtype=torch.long, device=device)
                for idx_i, pos in enumerate(top_indices):
                    selected_tokens[idx_i] = torch.multinomial(
                        sampling_probs[pos].unsqueeze(0), 1, generator=gen
                    ).squeeze()
            else:
                selected_tokens = argmax_tokens[top_indices]

            positions = top_indices.tolist()
            tokens = selected_tokens.tolist()
            confs = max_probs[top_indices].tolist()

            # Commit
            canvas[top_indices] = selected_tokens

            masks_after = int(model.is_masked(canvas).sum().item())
            trace.append(StepTrace(
                step=step_idx,
                positions_committed=positions,
                tokens_committed=tokens,
                confidences=confs,
                masks_remaining_before=n_masked,
                masks_remaining_after=masks_after,
            ))

        text = detokenize_completion(canvas, model.tokenizer, prompt_len)

        return DecodeResult(
            token_ids=canvas,
            text=text,
            trace=trace,
            reward_calls_used=0,
            prompt_len=prompt_len,
            config=config,
        )
