"""Compare results across multiple runs.

Usage:
    python -m rvu.eval.compare results/run1 results/run2 [results/run3 ...]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_summary(run_dir: str) -> Dict[str, Any]:
    """Load summary.json from a run directory."""
    p = Path(run_dir) / "summary.json"
    with open(p, "r") as f:
        return json.load(f)


def load_config(run_dir: str) -> Dict[str, Any]:
    """Load config.yaml from a run directory."""
    import yaml
    p = Path(run_dir) / "config.yaml"
    with open(p, "r") as f:
        return yaml.safe_load(f)


def load_results(run_dir: str) -> List[Dict[str, Any]]:
    """Load per-case results from results.jsonl."""
    p = Path(run_dir) / "results.jsonl"
    rows = []
    with open(p, "r") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compare(run_dirs: List[str]) -> str:
    """Build comparison table. Returns formatted string."""
    summaries = []
    configs = []
    all_results = []
    for d in run_dirs:
        summaries.append(load_summary(d))
        configs.append(load_config(d))
        all_results.append(load_results(d))

    # Detect tiers present
    all_tiers = set()
    for s in summaries:
        all_tiers.update(s.get("per_tier_valid_rate", {}).keys())
    tier_cols = sorted(all_tiers)

    # Header
    header = ["decoder", "valid%", "mean_rwd"]
    for t in tier_cols:
        header.append(f"{t}_%")
    header.extend(["rwd_calls", "fwd_pass", "time_s"])

    rows = []
    for s, c in zip(summaries, configs):
        decoder = s.get("decoder", c.get("decoder", "?"))
        valid_pct = f"{s['schema_valid_rate']*100:.1f}"
        mean_rwd = f"{s['mean_reward']:.4f}"

        tier_vals = []
        for t in tier_cols:
            rate = s.get("per_tier_valid_rate", {}).get(t, 0.0)
            tier_vals.append(f"{rate*100:.1f}")

        rwd_calls = f"{s['mean_reward_calls']:.0f}"
        fwd_pass = f"{s.get('mean_forward_passes', 0):.0f}"
        time_s = f"{s['mean_wall_time_s']:.2f}"

        row = [decoder, valid_pct, mean_rwd] + tier_vals + [rwd_calls, fwd_pass, time_s]
        rows.append(row)

    # Format table
    col_widths = [max(len(header[i]), max(len(r[i]) for r in rows)) for i in range(len(header))]
    sep = "  "

    lines = []
    header_line = sep.join(h.rjust(w) for h, w in zip(header, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))

    for row in rows:
        lines.append(sep.join(v.rjust(w) for v, w in zip(row, col_widths)))

    # Per-case budget match check between B1 and RVU/B2
    lines.append("")
    b1_results = None
    rvu_results = None
    b1_decoder = None
    rvu_decoder = None

    for s, results in zip(summaries, all_results):
        d = s.get("decoder", "?")
        if d == "b1":
            b1_results = results
            b1_decoder = d
        if d in ("rvu", "b2"):
            rvu_results = results
            rvu_decoder = d

    if b1_results is not None and rvu_results is not None:
        # Check per-case budget match
        mismatches = 0
        total = min(len(b1_results), len(rvu_results))
        for b1_row, rvu_row in zip(b1_results, rvu_results):
            if b1_row["reward_calls_used"] != rvu_row["reward_calls_used"]:
                mismatches += 1

        if mismatches == 0:
            lines.append(f"Budget MATCHED: B1 and {rvu_decoder} have identical "
                         f"per-case reward calls ({total} cases checked)")
        else:
            lines.append(f"*** BUDGET MISMATCH: {mismatches}/{total} cases differ "
                         f"between B1 and {rvu_decoder}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare eval runs")
    parser.add_argument("run_dirs", nargs="+", help="Paths to result directories")
    args = parser.parse_args()

    table = compare(args.run_dirs)
    print(table)


if __name__ == "__main__":
    main()
