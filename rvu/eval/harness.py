"""Eval harness: runs (model, decoder, reward, dataset) → JSONL results.

Usage:
    python -m rvu.eval.harness --config configs/kill_test.yaml --decoder rvu --limit 20
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

from rvu.models.tiny import TinyMDLM
from rvu.decoding.base import expected_steps
from rvu.decoding.vanilla import VanillaDecoder
from rvu.decoding.rvu import RVUDecoder
from rvu.decoding.best_of_n import BestOfNDecoder, matched_n_per_case
from rvu.rewards.json_schema import JsonSchemaReward


def load_dataset(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load JSONL dataset, optionally truncated."""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            cases.append(json.loads(line))
            if limit is not None and len(cases) >= limit:
                break
    return cases


def dataset_hash(path: str) -> str:
    """Short hash of dataset file for run_id."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def make_run_id(decoder_name: str, ds_path: str, seed: int) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dh = dataset_hash(ds_path)
    return f"{decoder_name}_{dh}_{seed}_{ts}"


def build_prompt_ids(prompt_text: str, tokenizer) -> torch.Tensor:
    """Tokenize prompt into tensor of token IDs."""
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    return torch.tensor(ids, dtype=torch.long)


def run_harness(
    config_path: str,
    decoder_name: str,
    limit: Optional[int] = None,
    dataset_path_override: Optional[str] = None,
) -> str:
    """Run eval harness. Returns path to results directory."""
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seeds", [42])[0] if isinstance(config.get("seeds"), list) else config.get("seed", 42)
    ds_path = dataset_path_override or config.get("dataset_path", "data/kill_test_cases.jsonl")
    device = config.get("device", "cpu")
    max_len = config.get("max_len", 128)
    steps = config.get("steps", 16)
    K = config.get("K", 8)
    lam = config.get("lambda", 5.0)
    tau_f = config.get("tau_f", 0.7)

    # Seed everything
    torch.manual_seed(seed)

    # Load dataset
    cases = load_dataset(ds_path, limit=limit)
    print(f"Loaded {len(cases)} cases from {ds_path}")

    # Build model
    print("Building tiny model...")
    model = TinyMDLM(max_len=max_len, device=device)
    print(f"Model: {model.param_count():,} params, vocab={model.vocab_size}, mask_id={model.mask_id}")

    # Results directory
    run_id = make_run_id(decoder_name, ds_path, seed)
    results_dir = Path("results") / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    run_config = {
        **config,
        "decoder": decoder_name,
        "run_id": run_id,
        "limit": limit,
        "dataset_path": ds_path,
    }
    with open(results_dir / "config.yaml", "w") as f:
        yaml.dump(run_config, f)

    # Run
    results_path = results_dir / "results.jsonl"
    results_data: List[Dict[str, Any]] = []
    nonzero_outputs: List[Dict[str, Any]] = []

    print(f"Running {decoder_name} on {len(cases)} cases...")
    with open(results_path, "w", encoding="utf-8") as results_f:
        for i, case in enumerate(cases):
            schema = case["schema"]
            prompt_text = case["prompt"]

            # Build prompt IDs
            prompt_ids = build_prompt_ids(prompt_text, model.tokenizer)
            # Truncate prompt if too long for canvas
            if len(prompt_ids) >= max_len - 1:
                prompt_ids = prompt_ids[:max_len - 2]

            prompt_len = len(prompt_ids)

            # Fresh decode reward (budget-counted)
            decode_reward = JsonSchemaReward(schema=schema)

            # Build decode config per case
            decode_config: Dict[str, Any] = {
                "steps": steps,
                "max_len": max_len,
                "device": device,
                "seed": seed,
            }

            # Build decoder per case (needs fresh reward)
            if decoder_name == "b0":
                case_decoder = VanillaDecoder()
            elif decoder_name in ("b2", "rvu"):
                lam_val = 0.0 if decoder_name == "b2" else lam
                decode_config.update({"K": K, "tau_f": tau_f, "lambda": lam_val})
                case_decoder = RVUDecoder(reward=decode_reward)
            elif decoder_name == "b1":
                # Per-case budget matching
                N = matched_n_per_case(prompt_len, max_len, steps, K)
                decode_config.update({"N": N, "temperature": tau_f})
                case_decoder = BestOfNDecoder(reward=decode_reward)
            else:
                raise ValueError(f"Unknown decoder: {decoder_name}")

            t0 = time.time()
            result = case_decoder.decode(model, prompt_ids, decode_config)
            wall_time = time.time() - t0

            # Assert reward accounting (skip B0 which uses no rewards)
            if decoder_name != "b0":
                assert result.reward_calls_used == decode_reward.call_count, (
                    f"Case {case['case_id']}: decoder reports {result.reward_calls_used} "
                    f"reward calls but reward.call_count={decode_reward.call_count}"
                )

            # Score final output with SEPARATE scoring-only reward (completion-only text)
            scoring_reward = JsonSchemaReward(schema=schema)
            final_score = scoring_reward.score_text(result.text)
            schema_valid = final_score == 1.0

            row = {
                "case_id": case["case_id"],
                "tier": case["tier"],
                "decoder": decoder_name,
                "final_reward": final_score,
                "schema_valid": schema_valid,
                "reward_calls_used": result.reward_calls_used,
                "prompt_len": prompt_len,
                "steps": steps,
                "wall_time_s": round(wall_time, 4),
                "seed": seed,
                "output_text": result.text[:500],  # truncate for sanity
            }
            results_data.append(row)

            # Write + flush immediately
            results_f.write(json.dumps(row) + "\n")
            results_f.flush()

            if final_score > 0.0:
                nonzero_outputs.append(row)

            if (i + 1) % 5 == 0 or i == len(cases) - 1:
                print(f"  [{i+1}/{len(cases)}] case_id={case['case_id']} "
                      f"tier={case['tier']} reward={final_score:.2f} "
                      f"calls={result.reward_calls_used} time={wall_time:.2f}s")

    # Print nonzero outputs
    if nonzero_outputs:
        print(f"\n*** {len(nonzero_outputs)} cases scored > 0 (unexpected with random model):")
        for row in nonzero_outputs:
            print(f"  case_id={row['case_id']} tier={row['tier']} "
                  f"reward={row['final_reward']:.3f}")
            print(f"  output: {row['output_text'][:200]}")
    else:
        print(f"\nAll {len(cases)} cases scored 0.0 (expected with random model).")

    # Summary
    from rvu.eval.metrics import compute_metrics
    summary = compute_metrics(results_data)
    summary["run_id"] = run_id
    summary["decoder"] = decoder_name
    summary["n_cases"] = len(cases)

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {results_path}")
    print(f"Summary: {results_dir / 'summary.json'}")

    return str(results_dir)


def main():
    parser = argparse.ArgumentParser(description="RVU eval harness")
    parser.add_argument("--config", type=str, required=True, help="Config YAML path")
    parser.add_argument("--decoder", type=str, required=True,
                        choices=["b0", "b1", "b2", "rvu"],
                        help="Decoder to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max cases to evaluate")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset path")
    args = parser.parse_args()

    run_harness(
        config_path=args.config,
        decoder_name=args.decoder,
        limit=args.limit,
        dataset_path_override=args.dataset,
    )


if __name__ == "__main__":
    main()
