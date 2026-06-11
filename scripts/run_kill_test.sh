#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

# Parse args
LIMIT="${1:-20}"
CONFIG="${2:-configs/kill_test.yaml}"
DATASET="data/kill_test_cases.jsonl"

echo "=== RVU Kill Test (CPU dry run) ==="
echo "Config: $CONFIG"
echo "Limit: $LIMIT cases"
echo ""

# Step 1: Generate dataset
echo "--- Generating dataset ---"
python scripts/make_dataset.py --n 200 --seed 42 --output "$DATASET"
echo ""

# Step 2: Run all four decoders
RUNS=()

echo "--- Running B0 (vanilla greedy) ---"
B0_DIR=$(python -m rvu.eval.harness --config "$CONFIG" --decoder b0 --limit "$LIMIT" --dataset "$DATASET" 2>&1 | tee /dev/stderr | grep "^Results:" | sed 's|Results: ||' | xargs dirname)
RUNS+=("$B0_DIR")
echo ""

echo "--- Running B2 (RVU with lambda=0) ---"
B2_DIR=$(python -m rvu.eval.harness --config "$CONFIG" --decoder b2 --limit "$LIMIT" --dataset "$DATASET" 2>&1 | tee /dev/stderr | grep "^Results:" | sed 's|Results: ||' | xargs dirname)
RUNS+=("$B2_DIR")
echo ""

echo "--- Running RVU (lambda=5) ---"
RVU_DIR=$(python -m rvu.eval.harness --config "$CONFIG" --decoder rvu --limit "$LIMIT" --dataset "$DATASET" 2>&1 | tee /dev/stderr | grep "^Results:" | sed 's|Results: ||' | xargs dirname)
RUNS+=("$RVU_DIR")
echo ""

echo "--- Running B1 (best-of-N, matched budget) ---"
B1_DIR=$(python -m rvu.eval.harness --config "$CONFIG" --decoder b1 --limit "$LIMIT" --dataset "$DATASET" 2>&1 | tee /dev/stderr | grep "^Results:" | sed 's|Results: ||' | xargs dirname)
RUNS+=("$B1_DIR")
echo ""

# Step 3: Compare
echo "=== COMPARISON TABLE ==="
python -m rvu.eval.compare "${RUNS[@]}"
