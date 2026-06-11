#!/usr/bin/env bash
# Pod setup script for RunPod GPU instances.
# Run this after SSH-ing into the pod.
set -euo pipefail

echo "=== RVU Pod Setup ==="

# 1. Clone repo
if [ ! -d "rvu-mdlm" ]; then
    echo "--- Cloning repo ---"
    git clone https://github.com/aman-source/rvu-mdlm.git
fi
cd rvu-mdlm

# 2. Install dependencies
echo "--- Installing GPU requirements ---"
pip install -r requirements-gpu.txt

# 3. Download LLaDA-8B-Instruct
echo "--- Downloading LLaDA-8B-Instruct ---"
if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN not set. Set it with: export HF_TOKEN=hf_..."
    echo "Some gated models may require authentication."
fi
python -c "
from huggingface_hub import snapshot_download
snapshot_download('GSAI-ML/LLaDA-8B-Instruct', token=None)
print('LLaDA-8B-Instruct downloaded successfully')
"

# 4. Verify: run CPU tests (must pass on pod too)
echo "--- Running unit tests ---"
python -m pytest tests/ -q --tb=short
echo ""

# 5. Quick GPU sanity check
echo "--- GPU sanity check ---"
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
echo ""

# 6. Generate dataset if not present
if [ ! -f "data/kill_test_cases.jsonl" ]; then
    echo "--- Generating dataset ---"
    python scripts/make_dataset.py --n 200 --seed 42 --output data/kill_test_cases.jsonl
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run the smoke test (20 cases):"
echo "  python -m rvu.eval.harness --config configs/llada_smoke.yaml --decoder rvu --limit 20"
echo ""
echo "To run the full kill test (200 cases, all decoders):"
echo "  bash scripts/run_kill_test.sh 200 configs/llada_smoke.yaml"
echo ""
echo "IMPORTANT: After every run, commit and push results:"
echo "  git add results/ && git commit -m 'results: <decoder> <description>' && git push"
