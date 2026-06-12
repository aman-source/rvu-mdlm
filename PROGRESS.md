# RVU (Reward-Voted Unmasking) — Progress Report

**Date:** 2026-06-12
**Phase:** 1 (CPU) — COMPLETE
**Phase 2 (GPU):** Written, not yet executed — awaiting RunPod signal
**Repo:** https://github.com/aman-source/rvu-mdlm
**Branch:** `master`
**Tests:** 149/149 passing
**Total code:** ~4,144 lines across 30 Python files

---

## Table of Contents

1. [What Is RVU](#1-what-is-rvu)
2. [Repo Structure](#2-repo-structure)
3. [Task Queue Execution Log](#3-task-queue-execution-log)
4. [Component Details](#4-component-details)
5. [CPU Dry Run Results](#5-cpu-dry-run-results)
6. [Test Coverage Summary](#6-test-coverage-summary)
7. [Git History](#7-git-history)
8. [GPU Phase Readiness](#8-gpu-phase-readiness)
9. [Known Issues & Notes](#9-known-issues--notes)

---

## 1. What Is RVU

Inference-time, training-free guidance for masked diffusion language models (MDLMs). At each denoising step:

1. **Forward pass** on current canvas → logits at masked positions
2. **Sample K futures** — fill all masks independently from temperature-scaled logits
3. **Score** each future with external black-box reward R (e.g., JSON schema validator)
4. **Weight** futures: `w_k = softmax(lambda * r_k)`
5. **Vote** per position: `score(i, v) = sum_k w_k * 1[future_k[i] == v]`
6. **Commit** top-n positions by agreement to their winner tokens

Core claim: black-box rewards drive position-level voting and commitment during unmasking. No training, no gradient, no model internals needed.

### Baselines

| Code | Name | Description |
|------|------|-------------|
| B0 | Vanilla confidence | Greedy argmax at highest-confidence masked positions. No reward. |
| B1 | Best-of-N | Generate N sequences with sampled B0, pick argmax reward. Budget-matched to RVU. |
| B2 | Uniform RVU | RVU with lambda=0 (uniform weights). Isolates self-consistency voting from reward signal. |
| RVU | Reward-Voted Unmasking | Full method with lambda=5.0. |

### Kill Test

Task: JSON schema adherence. Generate JSON conforming to a given schema.
Pass criterion: RVU > B1 on schema-valid rate at matched reward-call budget on real model (GPU phase).

---

## 2. Repo Structure

```
rvu-mdlm/
├── CLAUDE.md                    # Project spec + engineering rules
├── PROGRESS.md                  # This file
├── requirements.txt             # CPU: torch (cpu), numpy, jsonschema, transformers, pytest, pyyaml
├── requirements-gpu.txt         # GPU: torch, accelerate, etc.
├── .gitignore                   # Results: keep .jsonl + summary.json + config.yaml
│
├── configs/
│   ├── default.yaml             # K=8, lambda=5, tau_f=0.7, S=16, L=64
│   ├── kill_test.yaml           # K=8, S=16, L=128, 200 cases
│   ├── dry_run.yaml             # K=4, S=16, L=128 (lighter for CPU)
│   └── llada_smoke.yaml         # K=8, S=32, L=256, cuda:0 (GPU phase)
│
├── data/
│   └── kill_test_cases.jsonl    # 200 cases, seed=42, 50 per tier
│
├── rvu/
│   ├── __init__.py
│   ├── utils.py                 # detokenize(), detokenize_completion()
│   │
│   ├── models/
│   │   ├── base.py              # MDLM abstract interface
│   │   ├── tiny.py              # Random-init ~28M param stand-in (GPT-2 tokenizer)
│   │   └── llada.py             # LLaDA-8B-Instruct adapter (GPU phase)
│   │
│   ├── decoding/
│   │   ├── base.py              # Decoder ABC, DecodeResult, StepTrace, commit_schedule, expected_steps
│   │   ├── vanilla.py           # B0 — greedy + sampled modes
│   │   ├── rvu.py               # RVU (B2 = lambda=0, same class)
│   │   └── best_of_n.py         # B1 — N sampled rollouts, argmax reward
│   │
│   ├── rewards/
│   │   ├── base.py              # Reward abstract callable interface
│   │   ├── json_schema.py       # Kill-test reward (3-regime scoring)
│   │   └── registry.py          # @register_reward decorator
│   │
│   ├── data/
│   │   └── json_schema_gen.py   # 4-tier (prompt, schema) generator
│   │
│   └── eval/
│       ├── harness.py           # Runs (model, decoder, reward, dataset) → JSONL
│       ├── metrics.py           # schema_valid_rate, mean_reward, per-tier breakdowns
│       └── compare.py           # Multi-run comparison table + budget match check
│
├── tests/                       # 149 unit tests
│   ├── test_models.py           # MDLM interface, TinyMDLM, canvas, logits shape
│   ├── test_vanilla_decoding.py # B0: schedule, confidence, trace, mask suppression, prompt
│   ├── test_rvu_decoding.py     # RVU: hand-verified voting, lambda=0/large, reproducibility
│   ├── test_best_of_n.py        # B1: reward accounting, determinism, tie-break
│   ├── test_json_schema_gen.py  # Tier coverage, determinism, schema validity
│   ├── test_json_schema_reward.py # Edge cases: no JSON, truncated, valid, partial, embedded
│   └── test_completion_only.py  # Completion-only scoring, expected_steps, budget matching
│
├── scripts/
│   ├── make_dataset.py          # CLI: --n 200 --seed 42 --output data/kill_test_cases.jsonl
│   ├── run_kill_test.sh         # Full pipeline: generate → B0 → B2 → RVU → B1 → compare
│   └── pod_setup.sh             # GPU pod: clone, install, download model, test, smoke cmd
│
└── results/                     # Per-run directories with JSONL + summary + config
    └── .gitkeep
```

---

## 3. Task Queue Execution Log

### Task 1: Scaffold Repo

**Status:** DONE

Created full directory structure, `requirements.txt` (CPU-only torch), `.gitignore`, all `__init__.py` files, config YAML files.

### Task 2: MDLM Interface + Tiny Model + Tokenizer Wiring

**Status:** DONE

**MDLM abstract interface** (`rvu/models/base.py`):
- Single method: `logits(canvas: Tensor) -> Tensor` — `[L] → [L, V]`
- Helper: `is_masked(canvas)` → boolean mask
- Helper: `fully_masked_canvas(length)` → all-mask tensor
- Attributes: `tokenizer`, `mask_id`, `max_len`, `device`

**TinyMDLM** (`rvu/models/tiny.py`):
- Architecture: 4-layer pre-norm transformer encoder
- Config: d_model=256, n_heads=4, d_ff=512
- Params: ~27.9M (GPT-2 vocab = 50,258 with added [MASK])
- Tokenizer: GPT-2 + custom `[MASK]` token (id=50257)
- Xavier uniform init, `@torch.no_grad()` on forward pass
- Outputs garbage — by design. Exercises all code paths.

### Task 3: B0 Vanilla Confidence Decoding

**Status:** DONE

**File:** `rvu/decoding/vanilla.py`

**Decoder interface** (`rvu/decoding/base.py`):
- All decoders: `decode(model, prompt_ids, config) -> DecodeResult`
- `DecodeResult`: token_ids, text (completion-only), trace, reward_calls_used, prompt_len
- `commit_schedule(n_masks, steps_remaining)` → `ceil(n_masks / steps_remaining)`
- `expected_steps(prompt_len, max_len, steps)` → `min(steps, max_len - prompt_len)`

**B0 algorithm per step:**
1. Forward pass → logits. Set `mask_id` logit to `-inf` (defensive).
2. Softmax → confidence = max prob at each masked position.
3. Top-n positions by confidence → commit to argmax token (greedy) or sampled token.
4. n = `ceil(remaining_masks / remaining_steps)` — linear schedule.

**Two modes:**
- `sample=False` (default): greedy argmax. This is B0.
- `sample=True` + `temperature`: sampled variant. Used internally by B1.

**Commit schedule verification (from tests):**
| Masks | Steps | Per-step commits |
|-------|-------|------------------|
| 10 | 4 | 3, 3, 2, 2 |
| 5 | 5 | 1, 1, 1, 1, 1 |
| 3 | 10 | 1, 1, 1 (done in 3) |

### Task 4: JSON Schema Dataset Generator + Reward

**Status:** DONE

**Dataset generator** (`rvu/data/json_schema_gen.py`):

Four tiers, 50 cases each (200 total):

| Tier | Description | Example |
|------|-------------|---------|
| T1 | Flat object, 2-4 keys (string/int/bool) | `{"name": str, "age": int}` |
| T2 | Nested object, one level deep | `{"address": {"street": str, "zip": str}}` |
| T3 | Arrays of objects | `{"items": [{"name": str, "qty": int}]}` |
| T4 | Enums + integer ranges + required/optional | `{"status": enum["active","inactive"], "count": int min/max}` |

- Deterministic from seed (Python `random` module)
- Bounded noun/field pools (15 nouns, 10 string fields, etc.)
- Prompt format: `"Generate a JSON object describing a {noun}. It must conform to this schema:\n{schema}\nJSON:"`
- CLI: `python scripts/make_dataset.py --n 200 --seed 42 --output data/kill_test_cases.jsonl`

**JSON Schema Reward** (`rvu/rewards/json_schema.py`):

Three-regime scoring:

| Regime | Score | Condition |
|--------|-------|-----------|
| No JSON | 0.0 | No parseable `{...}` found |
| Parseable but invalid | 0.2 + 0.6 × (correct keys / required keys) | JSON parses but fails schema validation |
| Valid | 1.0 | `json.loads` + `jsonschema.validate` both pass |

- `extract_json_block()`: finds first balanced `{...}` handling strings, escapes, nesting
- `_count_correct_keys()`: checks each required key present with correct Python type
- `_type_matches()`: maps JSON Schema types to Python types (handles bool-is-int edge case)
- Self-counting: `call_count` incremented per `__call__`, NOT per `score_text`

### Task 5: RVU Decoder

**Status:** DONE

**File:** `rvu/decoding/rvu.py`

**Algorithm per step:**
1. Forward pass → logits `[L, V]`. Set `mask_id` logit to `-inf`.
2. Sample K futures: each fills all masked positions from `softmax(logits / tau_f)`. Committed positions copied. Per-step `torch.Generator` seeded from `(run_seed * 10000 + step_index)`.
3. Score each future with reward (completion-only: `future[prompt_len:]`). Exactly K calls per step.
4. Weights: `w_k = softmax(lambda * r)` with max-subtraction stability. Lambda=0 → exactly uniform (this IS B2).
5. Vote: `score(i, v) = sum_k w_k * 1[future_k[i] == v]`. Agreement = max_v score. Winner = argmax_v.
6. Tie-break: agreement (primary) → model confidence (secondary, eps=1e-9) → position index (tertiary, eps=1e-15). **All computed in float64** to preserve 1e-15 precision.
7. Commit top-n by agreement using same `commit_schedule()` as B0.

**Key design decisions:**
- B2 is NOT a separate class. Just `config["lambda"] = 0.0`.
- Agreement scores cast to float64 before epsilon tie-breaks (1e-15 vanishes in float32).
- `commit_score` tensor is float64, non-masked positions set to `-inf`.
- Reward accounting: `reward_calls_used = actual_steps × K`.

**Hand-verified test cases (from `test_rvu_decoding.py`):**

Tiny case: L=3, V=4, K=3 with hardcoded futures and rewards:
- Futures: `[[0,1,2], [0,2,2], [1,1,0]]`
- Rewards: `[0.8, 0.6, 0.2]`
- Lambda=5 → weights computed by hand, vote scores verified to atol=1e-6
- Lambda=0 → uniform weights → majority vote
- Lambda=1e6 → collapses to best future's tokens
- All rewards equal → uniform regardless of lambda
- K=1 → degenerates to that future's tokens
- Same seed → identical; different seed → different
- Mask token never appears in futures or commitments

### Task 6: B1 Best-of-N

**Status:** DONE

**File:** `rvu/decoding/best_of_n.py`

**Algorithm:**
1. Generate N sequences using `VanillaDecoder(sample=True, temperature=tau_f)`.
2. Each rollout gets own seed: `run_seed * 100000 + rollout_index`.
3. Score each completed sequence with reward once (completion-only).
4. Pick argmax reward. Tie-break: lowest rollout index (first generated).
5. `reward_calls_used = N`.

**Per-case budget matching:**
- `matched_n_per_case(prompt_len, max_len, steps, K)` → `expected_steps × K`
- `expected_steps = min(S, L - prompt_len)` — accounts for shorter prompts needing fewer steps
- Harness computes B1's N per case, not globally
- Compare tool verifies per-case match (not just aggregate)

**Greedy B0 untouched:** sampled variant parameterized via `config["sample"]` flag on same `VanillaDecoder` class. B0 regression tests still pass.

### Task 7: Eval Harness + Metrics + Compare + Dry Run

**Status:** DONE

**Harness** (`rvu/eval/harness.py`):
- Inputs: config YAML + dataset JSONL + decoder name + optional `--limit`
- Per case: fresh reward instance, decode, assert `reward_calls_used == reward.call_count`
- Final scoring: SEPARATE reward instance (not counted in budget), completion-only text
- Output: one JSONL line per case, flushed immediately
- `run_id = {decoder}_{dataset_hash[:8]}_{seed}_{timestamp}`
- Writes: `results/<run_id>/results.jsonl`, `config.yaml`, `summary.json`

**Metrics** (`rvu/eval/metrics.py`):
- `schema_valid_rate`: fraction with `final_reward == 1.0`
- `mean_reward`: mean across all cases
- Per-tier breakdowns: `per_tier_valid_rate`, `per_tier_mean_reward`
- `mean_reward_calls`, `mean_wall_time_s`

**Compare** (`rvu/eval/compare.py`):
- Takes 2+ run directories
- Prints formatted table: decoder, valid%, mean_reward, per-tier, calls, time
- Per-case budget match: loads B1 + RVU/B2 results.jsonl, compares `reward_calls_used` row-by-row
- Reports `MATCHED` or `*** MISMATCH` with count

### Fixes Applied (post-Task 7)

**Fix 1: Completion-only rewards**
- Problem: prompt text was included in decoded output → prompt containing JSON could inflate scores
- Solution: `detokenize_completion(canvas, tokenizer, prompt_len)` — decodes only `canvas[prompt_len:]`
- All decoders now store `prompt_len` in `DecodeResult`
- RVU/B1 pass `future[prompt_len:]` to reward
- Harness final scoring uses completion-only text

**Fix 2: Per-case budget matching**
- Problem: B1's N was computed globally as `S × K`, ignoring that shorter prompts → fewer RVU steps
- Solution: `expected_steps(prompt_len, max_len, steps)` → `min(S, L - prompt_len)`
- `matched_n_per_case()` computes B1's N per case
- Compare tool verifies per-case budget match

---

## 4. Component Details

### 4.1 MDLM Interface

```python
class MDLM(ABC):
    tokenizer: PreTrainedTokenizerBase
    mask_id: int
    max_len: int
    device: torch.device

    def logits(self, canvas: Tensor) -> Tensor:  # [L] → [L, V]
    def is_masked(self, canvas: Tensor) -> Tensor:  # [L] → [L] bool
    def fully_masked_canvas(self, length: int) -> Tensor:  # → [L] all mask_id
```

Real and tiny models implement same interface. Swap by changing config.

### 4.2 Decoder Interface

```python
class Decoder(ABC):
    def decode(self, model: MDLM, prompt_ids: Optional[Tensor], config: Dict) -> DecodeResult

@dataclass
class DecodeResult:
    token_ids: Tensor      # [L] final canvas
    text: str              # completion-only, EOS-stripped
    trace: List[StepTrace] # per-step records
    reward_calls_used: int # 0 for B0
    prompt_len: int        # where completion starts
    config: Dict
```

### 4.3 Reward Interface

```python
class Reward(ABC):
    call_count: int = 0
    def __call__(self, token_ids: Tensor, tokenizer) -> float
```

Self-counting. Harness asserts `decoder.reward_calls_used == reward.call_count` per case.

### 4.4 Config Files

**`configs/dry_run.yaml`** (used for CPU dry run):
```yaml
K: 4
lambda: 5.0
tau_f: 0.7
steps: 16
device: cpu
model: tiny
seeds: [42]
max_len: 128
```

**`configs/llada_smoke.yaml`** (GPU phase):
```yaml
K: 8
lambda: 5.0
tau_f: 0.7
steps: 32
device: "cuda:0"
model: llada
model_path: "GSAI-ML/LLaDA-8B-Instruct"
seeds: [42]
max_len: 256
```

---

## 5. CPU Dry Run Results

**Config:** L=128, S=16, K=4, 20 cases, seed=42, TinyMDLM (~28M params, random init)

**Expected:** All rewards ≈ 0 (random model produces tokenizer soup, nothing parses as JSON)

**Actual:** All rewards exactly 0.0000 across all 4 decoders. No counterexamples.

### Comparison Table

```
decoder  valid%  mean_rwd  T1_%  T2_%  T3_%  T4_%  rwd_calls  time_s
--------------------------------------------------------------------
     b0     0.0    0.0000   0.0   0.0   0.0   0.0          0    0.12
     b2     0.0    0.0000   0.0   0.0   0.0   0.0         10    0.21
    rvu     0.0    0.0000   0.0   0.0   0.0   0.0         10    0.21
     b1     0.0    0.0000   0.0   0.0   0.0   0.0         10    1.95
```

### Budget Match

```
Budget MATCHED: B1 and rvu have identical per-case reward calls (20 cases checked)
```

Mean reward calls = 10 (not 64 = 16×4) because prompts are long relative to L=128, so `expected_steps < 16` for most cases.

### Interpretation

- Pipeline works end-to-end: tokenization → canvas init → prompt commit → denoising loop → reward scoring → JSONL output → metrics → comparison
- Completion-only scoring confirmed: no prompt leakage inflating scores
- Per-case budget matching confirmed: B1 N equals RVU's actual step×K per case
- Reward accounting verified: assert passes for every case (decoder-reported == reward.call_count)
- Random model correctly produces ~0 valid JSON — plumbing validated, not quality

---

## 6. Test Coverage Summary

**Total: 149 tests, all passing** (pytest, ~44s on CPU)

### By file:

| Test File | Tests | What's Covered |
|-----------|-------|----------------|
| `test_models.py` | ~15 | MDLM interface, TinyMDLM shape/device, mask mechanics, param count |
| `test_vanilla_decoding.py` | ~25 | Commit schedule, confidence ordering, trace consistency, mask suppression, prompt handling, detokenize |
| `test_rvu_decoding.py` | ~35 | Hand-verified voting math (L=3,V=4,K=3), lambda=0/large/equal, tie-breaks (float64 verified), reproducibility, K=1 degenerate, mask never committed |
| `test_best_of_n.py` | ~20 | Reward accounting (N exact), determinism, tie-break (first), sampled variance, matched_n math, B0 regression |
| `test_json_schema_gen.py` | ~15 | Tier distribution (50 each), determinism (same seed = identical), schema validity, prompt format |
| `test_json_schema_reward.py` | ~20 | No JSON → 0.0, truncated → 0.0, wrong schema → partial ∈ [0.2, 0.8], valid → 1.0, embedded in prose, call counting |
| `test_completion_only.py` | ~19 | Completion-only text, prompt JSON not scored, RVU/B1 reward sees completion only, detokenize_completion, expected_steps, matched_n_per_case, B1-RVU budget match |

### Critical invariants tested:
- Voting math hand-verified with exact expected values (atol=1e-6)
- Agreement scores are float64 before epsilon tie-breaks
- Mask token never appears in futures or commitments (defensive -inf)
- Commit schedule guarantees completion in exactly S steps
- reward_calls_used == reward.call_count (per case)
- B1 and RVU budgets match per-case

---

## 7. Git History

```
34720c6 feat: add LLaDA-8B adapter and GPU pod setup scripts
0b0483b feat: complete Phase 1 CPU implementation of RVU pipeline
```

Repo: `https://github.com/aman-source/rvu-mdlm` (public)
Branch: `master`
Both commits pushed and verified.

---

## 8. GPU Phase Readiness

### Written (not yet executed):

**LLaDA-8B-Instruct adapter** (`rvu/models/llada.py`):
- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Mask token ID: 126336
- bf16 inference, logits cast to float32 for stable softmax
- `AutoModelForCausalLM.from_pretrained()` with `trust_remote_code=True`
- Chat template via `tokenizer.apply_chat_template()`
- Single forward pass: canvas `[L]` → unsqueeze to `[1, L]` → model → logits `[L, V]`

**Smoke config** (`configs/llada_smoke.yaml`):
- L=256, S=32, K=8, lambda=5, tau_f=0.7, cuda:0

**Pod setup script** (`scripts/pod_setup.sh`):
1. `git clone` repo
2. `pip install -r requirements-gpu.txt`
3. HuggingFace download (HF_TOKEN aware)
4. `pytest tests/` (CPU tests must pass on pod)
5. GPU sanity check (CUDA available, GPU name, memory)
6. Generate dataset if missing
7. Print smoke test command

### To do on GPU:
1. Update harness to dispatch `model=llada` (currently hardcoded to TinyMDLM)
2. Run smoke test: `python -m rvu.eval.harness --config configs/llada_smoke.yaml --decoder rvu --limit 20`
3. Run full kill test: all 4 decoders, 200 cases, compare
4. Commit + push results from pod (NON-NEGOTIABLE persistence rule)

---

## 9. Known Issues & Notes

1. **Harness model dispatch:** Currently hardcoded to `TinyMDLM`. Needs simple if/else for `model: llada` in config. Quick fix when GPU phase starts.

2. **`trust_remote_code=True`:** Required by LLaDA model. Standard for HF community models but worth noting for security.

3. **CPU dry run speed:** B1 is ~10x slower than B0/RVU (1.95s vs 0.12-0.21s per case) because it runs N full decoding passes. On GPU with real model, all decoders will be slower but GPU-bound, so relative difference may narrow.

4. **Prompt length vs canvas:** With L=128 and long prompts (GPT-2 tokenizes kill-test prompts to ~80-100 tokens), only ~30-50 positions are masked. This means `expected_steps` < S for most cases, and reward call budget is lower than `S × K`. Per-case matching handles this correctly.

5. **No nonzero scores on tiny model:** With GPT-2 tokenizer soup from random-init model, zero cases produced parseable JSON. This is expected and validates that completion-only scoring works (no prompt leakage). On real model, we expect actual JSON generation.

6. **Float64 tie-break:** RVU agreement scores use float64 for epsilon-stacking. Verified: 1e-15 position tie-break works in float64 but vanishes in float32 (eps ~1.2e-7). Test confirms this.

---

*Generated by Claude Opus 4.6 (1M context) on 2026-06-12*
