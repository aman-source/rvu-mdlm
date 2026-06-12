"""RVU: Reward-Voted Unmasking decoder.

B2 (uniform voting ablation) is RVU with lambda=0 — no separate class needed.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from rvu.models.base import MDLM
from rvu.decoding.base import DecodeResult, Decoder, RVUStepTrace, commit_schedule
from rvu.rewards.base import Reward
from rvu.utils import detokenize_completion


class RVUDecoder(Decoder):
    """Reward-Voted Unmasking.

    Per step:
    1. Forward pass → logits. Suppress mask_id to -inf.
    2. Sample K futures from softmax(logits / tau_f) at masked positions.
    3. Score each future with reward R → K scalars.
    4. Weights = softmax(lambda * rewards) with max-subtraction stability.
    5. Vote: score(i,v) = sum_k w_k * 1[future_k[i]==v].
       agreement(i) = max_v score(i,v); winner(i) = argmax_v.
    6. Commit top-n by agreement (tie-break: model confidence, then position).
    """

    def __init__(self, reward: Reward) -> None:
        self.reward = reward

    def decode(
        self,
        model: MDLM,
        prompt_ids: Optional[Tensor],
        config: Dict[str, Any],
    ) -> DecodeResult:
        steps: int = config["steps"]
        max_len: int = config.get("max_len", model.max_len)
        device = torch.device(config.get("device", "cpu"))
        K: int = config.get("K", 8)
        tau_f: float = config.get("tau_f", 0.7)
        lam: float = config.get("lambda", 5.0)
        seed: int = config.get("seed", 42)

        # Build initial canvas
        canvas = model.fully_masked_canvas(length=max_len).to(device)

        # Pre-commit prompt positions
        prompt_len = 0
        if prompt_ids is not None and len(prompt_ids) > 0:
            prompt_len = len(prompt_ids)
            canvas[:prompt_len] = prompt_ids.to(device)

        trace: List[RVUStepTrace] = []
        total_reward_calls = 0
        total_forward_passes = 0

        for step_idx in range(steps):
            mask = model.is_masked(canvas)
            n_masked = int(mask.sum().item())

            if n_masked == 0:
                break

            steps_left = steps - step_idx
            n_commit = commit_schedule(n_masked, steps_left)

            # --- 1. Forward pass (1 per step) ---
            logits = model.logits(canvas)  # [L, V]
            total_forward_passes += 1
            logits[:, model.mask_id] = float("-inf")
            probs = F.softmax(logits, dim=-1)  # [L, V]

            # Model confidence for tie-breaking
            model_confidence, _ = probs.max(dim=-1)  # [L]

            # --- 2. Sample K futures ---
            # Temperature-scaled logits for sampling
            sampling_logits = logits / tau_f  # [L, V]
            sampling_probs = F.softmax(sampling_logits, dim=-1)  # [L, V]

            masked_indices = mask.nonzero(as_tuple=True)[0]  # [M]
            L = canvas.shape[0]
            V = probs.shape[1]

            futures = canvas.unsqueeze(0).expand(K, -1).clone()  # [K, L]

            # Reproducible per-step generator
            gen = torch.Generator(device=device)
            gen.manual_seed(seed * 10000 + step_idx)

            for pos_idx in masked_indices:
                pos = pos_idx.item()
                pos_probs = sampling_probs[pos]  # [V]
                # Sample K tokens for this position
                sampled = torch.multinomial(
                    pos_probs.unsqueeze(0).expand(K, -1),
                    num_samples=1,
                    generator=gen,
                ).squeeze(-1)  # [K]
                futures[:, pos] = sampled

            # --- 3. Score futures (completion-only: exclude prompt) ---
            rewards_list: List[float] = []
            for k in range(K):
                r = self.reward(futures[k][prompt_len:], model.tokenizer)
                rewards_list.append(r)
            total_reward_calls += K

            rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)

            # --- 4. Weights: softmax(lambda * r) with max-subtraction ---
            scaled = lam * rewards
            scaled = scaled - scaled.max()  # stability
            weights = F.softmax(scaled, dim=0)  # [K]

            # --- 5. Vote ---
            # score(i, v) = sum_k w_k * 1[futures[k][i] == v]
            # Only need to vote at masked positions
            agreement = torch.zeros(L, device=device)
            winner = torch.zeros(L, dtype=torch.long, device=device)

            for pos_idx in masked_indices:
                pos = pos_idx.item()
                tokens_at_pos = futures[:, pos]  # [K]

                # Accumulate weighted votes per token
                vote_scores: Dict[int, float] = {}
                for k in range(K):
                    tok = tokens_at_pos[k].item()
                    vote_scores[tok] = vote_scores.get(tok, 0.0) + weights[k].item()

                # Find winner and agreement
                best_tok = -1
                best_score = -1.0
                for tok, sc in vote_scores.items():
                    if sc > best_score:
                        best_score = sc
                        best_tok = tok
                agreement[pos] = best_score
                winner[pos] = best_tok

            # --- 6. Commit: top-n by agreement ---
            # Tie-break: higher model confidence, then lower position index.
            # Build sort key: (agreement, model_confidence, -position)
            # We want descending agreement, descending confidence, ascending position.
            # Create a composite score for topk among masked positions only.
            # Use agreement as primary, add tiny tiebreakers.

            # Only consider masked positions
            # float64 required: 1e-15 vanishes in float32 (eps ~ 1.2e-7)
            eps_conf = 1e-9  # scale for confidence tiebreak
            eps_pos = 1e-15  # scale for position tiebreak

            commit_score = torch.full((L,), -float("inf"), dtype=torch.float64, device=device)
            for pos_idx in masked_indices:
                pos = pos_idx.item()
                commit_score[pos] = (
                    float(agreement[pos].item())
                    + eps_conf * model_confidence[pos].item()
                    - eps_pos * pos
                )

            _, top_indices = commit_score.topk(min(n_commit, n_masked))

            positions = top_indices.tolist()
            tokens = winner[top_indices].tolist()
            agreements = agreement[top_indices].tolist()
            confs = model_confidence[top_indices].tolist()

            # Commit
            canvas[top_indices] = winner[top_indices]

            masks_after = int(model.is_masked(canvas).sum().item())
            trace.append(RVUStepTrace(
                step=step_idx,
                positions_committed=positions,
                tokens_committed=tokens,
                confidences=confs,
                masks_remaining_before=n_masked,
                masks_remaining_after=masks_after,
                agreement_scores=agreements,
                reward_values=rewards_list,
                weights=weights.tolist(),
            ))

        text = detokenize_completion(canvas, model.tokenizer, prompt_len)

        return DecodeResult(
            token_ids=canvas,
            text=text,
            trace=trace,
            reward_calls_used=total_reward_calls,
            forward_passes=total_forward_passes,
            prompt_len=prompt_len,
            config=config,
        )
