"""Metrics computation from eval results."""

from typing import Any, Dict, List


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute metrics from a list of result rows.

    Returns dict with:
      - schema_valid_rate: fraction of cases with final_reward == 1.0
      - mean_reward: mean final_reward across all cases
      - per_tier_valid_rate: {tier: rate}
      - per_tier_mean_reward: {tier: mean}
      - mean_reward_calls: mean reward_calls_used
      - mean_wall_time_s: mean wall_time_s
    """
    if not results:
        return {
            "schema_valid_rate": 0.0,
            "mean_reward": 0.0,
            "per_tier_valid_rate": {},
            "per_tier_mean_reward": {},
            "mean_reward_calls": 0.0,
            "mean_forward_passes": 0.0,
            "mean_wall_time_s": 0.0,
        }

    n = len(results)
    valid_count = sum(1 for r in results if r["schema_valid"])
    total_reward = sum(r["final_reward"] for r in results)
    total_calls = sum(r["reward_calls_used"] for r in results)
    total_fwd = sum(r.get("forward_passes", 0) for r in results)
    total_time = sum(r["wall_time_s"] for r in results)

    # Per-tier
    tiers: Dict[int, List[Dict[str, Any]]] = {}
    for r in results:
        t = r["tier"]
        tiers.setdefault(t, []).append(r)

    per_tier_valid: Dict[str, float] = {}
    per_tier_reward: Dict[str, float] = {}
    for t, rows in sorted(tiers.items()):
        t_n = len(rows)
        t_valid = sum(1 for r in rows if r["schema_valid"])
        t_reward = sum(r["final_reward"] for r in rows)
        per_tier_valid[f"T{t}"] = round(t_valid / t_n, 4) if t_n > 0 else 0.0
        per_tier_reward[f"T{t}"] = round(t_reward / t_n, 4) if t_n > 0 else 0.0

    return {
        "schema_valid_rate": round(valid_count / n, 4),
        "mean_reward": round(total_reward / n, 4),
        "per_tier_valid_rate": per_tier_valid,
        "per_tier_mean_reward": per_tier_reward,
        "mean_reward_calls": round(total_calls / n, 2),
        "mean_forward_passes": round(total_fwd / n, 2),
        "mean_wall_time_s": round(total_time / n, 4),
    }
