# CLAUDE.md — Reward-Voted Unmasking (RVU)

## What this project is

Inference-time, training-free guidance for masked diffusion language models (MDLMs).
At each denoising step we sample K complete "futures" from the model's per-position
logits, score each future with an external black-box reward R (schema validator,
constraint checker, reward model), and use reward-weighted position-level voting to
decide (a) which masked positions to commit this step and (b) to which tokens.
Positions where high-reward futures disagree stay masked (deferred).

Core claim: external black-box rewards used for position-level voting and commitment
decisions during unmasking. Nobody has this. Target: match RL-tuned dLLM results
(d1 / diffu-GRPO) with zero training.

This is a research codebase. Correctness and reproducibility beat cleverness.

## Roles

- **Manager (Aman):** relays prompts and outputs between architect and you. Decides
  when GPU spend starts. Never make him debug — surface problems with full context.
- **Architect (Claude chat):** owns method design, experiment design, and all
  decisions about what to build next. You receive prompts written by the architect.
- **You (Claude Code):** implement, test, run, and report. Do not silently change
  the method spec. If the spec seems wrong or ambiguous, implement the literal spec,
  flag the concern in your report, and propose the fix. The architect decides.

After every task, end your output with a short REPORT block:
```
REPORT
- What was built/run
- Results (numbers, file paths)
- Surprises / concerns
- Suggested next step
```
This block is what the manager pastes back to the architect. Make it self-contained.

## Method spec (v1 — do not deviate without architect sign-off)

State: canvas x_t of length L, some positions unmasked (committed), rest = [MASK].

One RVU step:
1. **Forward pass:** model(x_t) → logits over vocab at every masked position.
2. **Sample K futures:** for k = 1..K, sample a complete sequence x̂₀ᵏ by drawing
   each masked position independently from its logits at temperature τ_f.
   Committed positions are copied as-is. (K, τ_f are config values; defaults K=8, τ_f=0.7.)
3. **Score:** rᵏ = R(x̂₀ᵏ). R is a pluggable callable, sequence-in scalar-out.
   R may be binary, partial-credit, or a learned model. RVU never sees R's internals.
4. **Weights:** wᵏ = exp(λ · rᵏ) / Σⱼ exp(λ · rʲ). λ is config (default 5.0).
5. **Vote:** for each masked position i and token v:
   score(i, v) = Σₖ wᵏ · 1[x̂₀ᵏ[i] = v]
   agreement(i) = max_v score(i, v); winner(i) = argmax_v score(i, v)
6. **Commit:** unmask the top-n positions by agreement(i) to their winner tokens,
   where n follows the same per-step budget as the baseline scheduler (so total
   step count matches vanilla decoding). Tie-break by model confidence.
7. Repeat until no masks remain or max steps reached.

Required baselines (same model, same total reward-call budget where applicable):
- B0: vanilla confidence-based decoding (no reward).
- B1: best-of-N at sequence level — generate N full sequences with B0, pick
  argmax R. Match N so total reward calls ≈ RVU's (steps × K).
- B2: RVU with uniform weights (λ=0) — isolates "does the reward matter or is
  it just self-consistency voting".

B2 is an ablation that doubles as a sanity check. If RVU ≤ B1 at matched reward
calls, the kill test fails — report it plainly, do not massage.

## Kill test (Week 1 gate)

Task: JSON schema adherence. Given a prompt + JSON schema, generate an object
conforming to the schema. Reward: 1.0 if json.loads succeeds AND schema validates
(use `jsonschema` lib); partial credit = fraction of required keys present with
correct types when full validation fails; 0 if not parseable.

Dataset: build a generator producing (prompt, schema) pairs across difficulty
tiers (flat object → nested → arrays of objects → enums/patterns). 200 eval cases,
fixed seed, saved to disk as JSONL so every run uses identical cases.

Pass criterion: RVU > B1 on schema-valid rate at matched reward-call budget,
on the real model (GPU phase). CPU phase only proves the pipeline runs.

## Two-phase workflow

### Phase 1 — CPU (now)
Everything must run on a laptop-class CPU with no GPU assumptions.
- Implement the full pipeline against a **tiny stand-in model**: a small
  random-init MDLM-style transformer (~5-10M params, vocab can be GPT-2 tokenizer)
  wrapped in the same interface as the real model. Outputs are garbage; that is fine.
  The point is exercising every code path end to end.
- The model interface is a single abstraction:
  `class MDLM: def logits(self, canvas: Tensor) -> Tensor  # [L, V] at masked positions`
  plus `tokenizer`, `mask_id`, `max_len`. Real and tiny models implement the same class.
- All decoding logic (B0, B1, B2, RVU), rewards, eval harness, logging, and plots
  are written and unit-tested in this phase.
- `device` is a config value everywhere. No hardcoded `.cuda()`. torch CPU wheel only
  in Phase 1 requirements.

### Phase 2 — GPU (RunPod, on manager's signal)
- Manager provides SSH to a RunPod pod. You connect, `git clone` this repo,
  `pip install -r requirements-gpu.txt`, download LLaDA-8B-Instruct
  (GSAI-ML/LLaDA-8B-Instruct) from HuggingFace, run the real experiments.
- Implement the real-model adapter (`models/llada.py`) behind the same MDLM
  interface. Dream-7B and DiffusionGemma adapters come later, same interface.
- Pods are ephemeral and can be deleted at any moment. Therefore:

### NON-NEGOTIABLE persistence rules (GPU phase)
1. Every experiment writes results incrementally to `results/<run_id>/` as JSONL,
   one line per eval case, flushed after every write. Never hold results only in memory.
2. After every experiment (and at minimum every 30 minutes during long runs),
   `git add results/ && git commit && git push` from the pod. Results live on
   GitHub, never only on the pod.
3. Any artifact too big for git (>50MB) gets pushed to HuggingFace Hub immediately
   after creation, then the run continues.
4. A run that finishes without its results pushed is a failed run. Verify the push
   succeeded before reporting completion.

## Repo structure

```
rvu/
  CLAUDE.md
  README.md
  requirements.txt           # CPU: torch (cpu), numpy, jsonschema, transformers, pytest
  requirements-gpu.txt       # adds: accelerate, flash-attn if applicable
  configs/
    default.yaml             # K, lambda, tau_f, steps, device, model, seeds
    kill_test.yaml
  rvu/
    models/
      base.py                # MDLM abstract interface
      tiny.py                # random-init stand-in (Phase 1)
      llada.py               # real adapter (Phase 2)
    decoding/
      vanilla.py             # B0 confidence decoding
      best_of_n.py           # B1
      rvu.py                 # the method (B2 = rvu with lambda=0)
    rewards/
      base.py                # Reward callable interface
      json_schema.py         # kill-test reward
      registry.py
    data/
      json_schema_gen.py     # (prompt, schema) pair generator, tiered difficulty
    eval/
      harness.py             # runs (model, decoder, reward, dataset) -> JSONL
      metrics.py             # schema-valid rate, partial credit, reward calls used
      compare.py             # tables + matched-budget comparison
  tests/                     # pytest; every decoding path, voting math, reward,
                             # matched-budget accounting must have tests
  scripts/
    run_kill_test.sh
    make_dataset.py
  results/                   # gitignored EXCEPT *.jsonl and summary.json
```

## Engineering rules

- Config-driven. No magic numbers in code; everything in YAML, logged into every
  run's `results/<run_id>/config.yaml`.
- Seed everything (python, numpy, torch) from config. Every run records its seed,
  git commit hash, and timestamp.
- Reward-call accounting is a first-class metric. Every decoder reports exactly
  how many reward calls it used. Matched-budget comparisons depend on this being exact.
- Unit tests for the voting math with hand-computed expected values (small L, V, K).
  The vote/commit logic is the heart of the paper; it must be provably correct.
- Keep diffs small and focused. One concern per commit. Push after every working unit.
- Python 3.10+, type hints, no notebook-only code. Everything runs via scripts.
- If something fails on the pod, capture full traceback + nvidia-smi + config into
  the REPORT. Do not retry-loop silently.

## What NOT to do

- Do not implement SMC, multi-particle resampling, refinement steps on futures, or
  any extension beyond the v1 spec. Those are architect decisions for v2.
- Do not benchmark on GSM8K/math yet. Kill test is JSON schema only.
- Do not fine-tune anything. This method is training-free; that is the claim.
- Do not optimize for speed in Phase 1. Correctness first; profiling comes after
  the kill test passes.

## Current task queue (architect updates this section via the manager)

1. Scaffold repo per structure above. CPU requirements only.
2. Implement MDLM interface + tiny stand-in model + tokenizer wiring.
3. Implement B0 vanilla confidence decoding + tests.
4. Implement JSON schema dataset generator + reward + tests.
5. Implement RVU (and B2 via λ=0) + hand-verified voting tests.
6. Implement B1 + reward-call accounting + tests.
7. Eval harness + metrics + compare script. Full CPU dry run of the kill test
   with the tiny model. Report.