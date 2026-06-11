#!/usr/bin/env python3
"""Generate (prompt, schema) pairs for the JSON schema kill test."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rvu.data.json_schema_gen import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate kill-test dataset")
    parser.add_argument("--n", type=int, default=200, help="Number of eval cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/kill_test_cases.jsonl")
    args = parser.parse_args()

    generate_dataset(n=args.n, seed=args.seed, output_path=args.output)
    print(f"Wrote {args.n} cases to {args.output}")


if __name__ == "__main__":
    main()
