"""Reward-variance diagnostic for RVU.

Runs RVU on N cases, extracts per-step reward stats from traces,
outputs a table of reward std/min/max by step index.

Usage:
    python scripts/reward_variance_diag.py --config configs/kill_test_llada.yaml --limit 30 --subset-per-tier 8
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml

from rvu.eval.harness import build_model, build_prompt_ids, load_dataset
from rvu.decoding.rvu import RVUDecoder
from rvu.decoding.base import RVUStepTrace, expected_steps
from rvu.rewards.json_schema import JsonSchemaReward


def main():
    parser = argparse.ArgumentParser(description="RVU reward-variance diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--subset-per-tier", type=int, default=8)
    parser.add_argument("--output", type=str, default="results/reward_variance_diag.jsonl")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    seed = config.get("seeds", [42])[0] if isinstance(config.get("seeds"), list) else config.get("seed", 42)
    ds_path = config.get("dataset_path", "data/kill_test_cases.jsonl")
    device = config.get("device", "cpu")
    max_len = config.get("max_len", 256)
    steps = config.get("steps", 32)
    K = config.get("K", 8)
    lam = config.get("lambda", 5.0)
    tau_f = config.get("tau_f", 0.7)

    torch.manual_seed(seed)

    # Load + filter dataset
    cases = load_dataset(ds_path, limit=args.limit)
    if args.subset_per_tier:
        tier_counts: Dict[int, int] = {}
        filtered = []
        for c in cases:
            t = c["tier"]
            tier_counts[t] = tier_counts.get(t, 0) + 1
            if tier_counts[t] <= args.subset_per_tier:
                filtered.append(c)
        cases = filtered

    print(f"Running RVU reward-variance diagnostic on {len(cases)} cases")
    print(f"Config: K={K}, lambda={lam}, tau_f={tau_f}, S={steps}, L={max_len}")

    model = build_model(config)
    print(f"Model loaded: {model.param_count():,} params")

    # Per-step aggregation
    step_rewards: Dict[int, List[List[float]]] = defaultdict(list)  # step_idx -> list of K-reward vectors
    per_case_results = []

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        for i, case in enumerate(cases):
            schema = case["schema"]
            prompt_text = case["prompt"]
            prompt_ids = build_prompt_ids(prompt_text, model.tokenizer)
            if len(prompt_ids) >= max_len - 1:
                prompt_ids = prompt_ids[:max_len - 2]

            reward = JsonSchemaReward(schema=schema)
            decoder = RVUDecoder(reward=reward)
            decode_config = {
                "steps": steps, "max_len": max_len, "device": device,
                "seed": seed, "K": K, "tau_f": tau_f, "lambda": lam,
            }

            result = decoder.decode(model, prompt_ids, decode_config)

            # Extract per-step rewards from trace
            case_steps = []
            for trace in result.trace:
                if isinstance(trace, RVUStepTrace):
                    rv = trace.reward_values  # K floats
                    std = float(np.std(rv))
                    mn = float(np.min(rv))
                    mx = float(np.max(rv))
                    mean = float(np.mean(rv))
                    step_rewards[trace.step].append(rv)
                    case_steps.append({
                        "step": trace.step,
                        "rewards": rv,
                        "std": round(std, 6),
                        "min": round(mn, 4),
                        "max": round(mx, 4),
                        "mean": round(mean, 4),
                        "masks_before": trace.masks_remaining_before,
                    })

            row = {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "n_steps": len(case_steps),
                "final_reward": reward.call_count,  # not the score, just tracking
                "steps_detail": case_steps,
            }
            per_case_results.append(row)
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            if (i + 1) % 5 == 0 or i == len(cases) - 1:
                print(f"  [{i+1}/{len(cases)}] case_id={case['case_id']} steps={len(case_steps)}")

    # Aggregate table
    print("\n=== REWARD VARIANCE BY STEP INDEX ===")
    print(f"{'step':>5} {'n_cases':>8} {'mean_std':>10} {'mean_min':>10} {'mean_max':>10} {'mean_mean':>10} {'frac_std>0.05':>14}")

    total_steps = 0
    steps_with_variance = 0

    for step_idx in sorted(step_rewards.keys()):
        all_rvecs = step_rewards[step_idx]
        stds = [float(np.std(rv)) for rv in all_rvecs]
        mins = [float(np.min(rv)) for rv in all_rvecs]
        maxs = [float(np.max(rv)) for rv in all_rvecs]
        means = [float(np.mean(rv)) for rv in all_rvecs]
        frac_high_std = sum(1 for s in stds if s > 0.05) / len(stds)

        total_steps += len(stds)
        steps_with_variance += sum(1 for s in stds if s > 0.05)

        print(f"{step_idx:>5} {len(all_rvecs):>8} {np.mean(stds):>10.4f} {np.mean(mins):>10.4f} {np.mean(maxs):>10.4f} {np.mean(means):>10.4f} {frac_high_std:>14.3f}")

    if total_steps > 0:
        overall_frac = steps_with_variance / total_steps
        print(f"\nOverall: {steps_with_variance}/{total_steps} step-instances ({overall_frac:.1%}) have std(rewards) > 0.05")
    else:
        print("\nNo steps recorded.")

    print(f"\nDetailed results: {output_path}")


if __name__ == "__main__":
    main()
