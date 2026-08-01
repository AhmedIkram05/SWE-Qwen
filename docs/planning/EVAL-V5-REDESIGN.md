# Eval v5 — Single-Pipeline Redesign

**Document Type:** Phase Plan Revision (Level 4)
**Status:** Draft v1.0
**Parent:** `docs/planning/MASTER-PLAN.md` (Phase 5), `docs/planning/PHASE-5-EVALUATION-HARNESS.md`
**Supersedes:** the dual-pipeline state (old harness + mini-SWE-agent)
**Date:** 2026-08-01

---

## 1. Why This Redesign Exists

Two evaluation pipelines coexisted (old Modal harness + new mini-SWE-agent path).
Neither was trusted end-to-end, and the agentic path was 10–50× the token cost of
single-turn generation. Decision (user, 2026-08-01): **one pipeline, single-turn,
execution-based, all-Modal, tiered sampling.** Aligned with ADR-005 (execution-based
F2P is the primary metric; text similarity secondary) and the Vision's out-of-scope
("no benchmark-leadership chasing, no multi-agent frameworks").

**What the model is trained on (reminder, from ADR-003/004 + Phase 4):**
SWE-bench issue→patch pairs (Issue Description → Code Patch, single-turn; execution
feedback deferred to v2). Golden = 2,056 SWE-bench Verified+Test+Dev with
FAIL_TO_PASS/PASS_TO_PASS. Evaluation measures patch correctness against real test
suites. General-knowledge benchmarks (MMLU etc.) are **off-task and rejected** —
the golden set IS the capability regression check (per-repo F2P + P2P).

## 2. Design: One Pipeline, Four Tiers

```
eval/  (ONE package, ONE CLI)
├─ generate   → Modal vLLM + LoRA (existing inference.py: generate_patches_batch)
├─ apply      → existing patch_applier.py (git apply → unidiff fallback)
├─ test       → REWORKED test_runner.py — repo-batched containers on official
│              swebench images (bottleneck fix, see §2.1)
├─ score      → existing metrics.py + NEW stats.py (Wilson CIs, McNemar, bootstrap)
├─ log        → existing harness.py (W&B artifacts, per-repo resume)
└─ compare    → existing comparison.py (re-validate P4 proxy champion) + NEW
               stats + cost per run
```

## 2.1 Test Execution Architecture — Bottleneck Fix (primary redesign driver)

**Problem:** old plan = 1 Modal container per instance, each cloning the repo and
running `pip install -e .` (~2-5 min setup) → ~10 min/instance → 2,056 instances
= ~340 container-hours of pure setup. Untenable.

**Fix — official SWE-bench images, one Modal function per repo (implemented):**

- Official SWE-bench eval images are per-*instance*:
  `swebench/sweb.eval.x86_64.<instance_id munged>:latest` (`django__django-10554`
  → `django_1776_django-10554`). Each contains `/testbed` checked out at the
  instance's base commit (plus a "SWE-bench" marker commit on top — so
  `git reset --hard <base_sha>` works offline; base_sha is always an ancestor),
  and a conda env `testbed` with the project installed **editable**.
- Modal 1.5.3 `Function.with_options` has **no `image` param** (image is fixed at
  `@app.function` decoration) → `test_runner.py` registers **one function per
  repo** (`swebench_fn` lazy registry, keyed by repo) using the repo's first
  instance's image. Valid because every image for a repo contains the repo's
  **full git history** → any instance's `base_sha` resolves, and reset+editable
  install makes per-instance source state exact. Ground-truth verification
  (F2P must be 100% on `test_patch`) catches any env mismatch.
- Per instance, inside the container (cheap ops only):
  1. `git reset --hard <base_sha>` + `git clean -fd` (~2-5 s)
  2. apply ground-truth `test_patch` → `conda run -n testbed pytest -k "<f2p tests>"`
  3. ground-truth verification (F2P = 100% on test_patch — catches env drift)
  4. revert → apply generated patch → run same tests → collect F2P/P2P
- One-time per container: `conda run -n testbed python -m pip install
  pytest-json-report pytest-timeout` (~10-20 s; probe import first).
- **No pip install of the project, no full clones per instance.** ~60-90 s/instance.
- Concurrency: `max_parallel=16` instances in flight; all repos parallel.
- **Fallback (existing, volume-cached):** if the swebench path fails (image
  unavailable, bad env), `run_batch` falls back to the original
  clone/checkout/install path — full clone + venv built once per repo **ever**
  (`.installed` marker in `/test_cache/{repo}/.venv`, `.git` presence check),
  reused by every container and every instance.

| Tier | Containers | Wall time |
|------|-----------|-----------|
| smoke (20) | ~4-6 | ~3-5 min |
| dev (100) | ~10 | ~8-10 min |
| final (500) | 18 | ~20-25 min |
| full (2,056) | 18 | ~40-60 min |

**Risks:** image pull size (~3-10 GB/repo, pulled once per repo — Modal caches);
entrypoint quirks → ground-truth verification catches broken envs; conda
activation inside Modal container (`conda run -n testbed`); Test/Dev images
partially missing → those instances hit the clone/install fallback.

| Tier | Size | Source | When | Est. cost |
|------|------|--------|------|-----------|
| `smoke` | 20 | Verified, seed 42 | CI gate, every PR touching eval/training | **~$0.10** |
| `dev` | 100 | Verified, seed 42 | Variant/prompt iteration | **~$0.50** |
| `final` | 500 | Verified | Champion selection + README numbers | **~$2–3** |
| `full` | 2,056 | Golden (all) | Champion evidence once (S1: F2P ≥ 30%) | **~$8–12** |

Deterministic subsetting (fixed seed 42) → identical instances across all
variants → **paired McNemar significance**, so 100 dev instances have the
statistical power of ~2× unpaired samples. This is the cheap-reliable trade.

**Total project eval spend (3 variants + baseline):** ~$20–25 one-time, plus
~$0.10 per CI smoke run. Iteration on prompts = dev tier only.

**Cost math (Modal, honest):**
- Generation A10G-24GB vLLM, 14B QLoRA ~2K out tokens/instance: 500 ≈ 1M tokens
  ≈ 5–10 min GPU ≈ **$0.20–0.30**
- Test exec: Modal CPU containers (~2 vCPU × ~3 min/instance, parallel 20–50):
  500 ≈ 50 vCPU-hrs ≈ **$1.50–2.00** (CPU ≈ $0.008/vCPU-hr)
- Dominated by repo clone + `pip install` per repo (cached in Modal volumes)

## 3. Changes

### Delete (mini-SWE-agent path — user-confirmed)
- `evaluation/swe_agent.py` (16.6K)
- `evaluation/serve.py` (3.2K)
- `tests/test_swe_agent.py` (14.5K, 13 tests)
- `run-swe-agent` CLI command in `evaluation/cli.py`
- Any `EVAL_SWE_AGENT_*` config/env remnants; sweep `.github/workflows/` + docs for references

### Keep as-is (old path, plan-aligned, 51 unit tests green)
- `config.py`, `schema.py`, `patch_applier.py`, `test_runner.py`, `metrics.py`,
  `harness.py` (resume + W&B), `inference.py`, `prompt_ab_test.py`, `comparison.py`, `cli.py`
- Golden = 2,056, `ci_sample_size=50` → replaced by explicit tiers (below)
- `scripts/local_e2e_smoke.py`, `scripts/debug_eval_one.py` (dev helpers)

### Add (small, recruiter-visible)
1. **`evaluation/stats.py`** (~80 lines): Wilson 95% CI on F2P/P2P rates, McNemar
   paired test (exact binomial) for variant-vs-variant, paired bootstrap option.
2. **Tier plumbing in `cli.py`**: `--mode smoke|dev|final|full` → resolves sample
   size + subset (seed 42). `smoke` returns non-zero exit code on F2P drop vs
   stored baseline → CI-gate ready (Phase 7 wires the workflow).
3. **Cost estimate in `harness.py`** (~25 lines): generation GPU-minutes × rate +
   test container vCPU-hrs × rate → `cost_usd` logged to W&B + printed in
   `compare` report. Feeds README "cost per eval run" number (recruiter signal).

### Metrics reported (per tier run, W&B)
- F2P rate (primary, Wilson CI), P2P rate (regression safety, Wilson CI)
- Per-repo F2P breakdown, flaky rate, patch-apply success rate
- McNemar p-values vs baseline/champion in `compare`
- `cost_usd`, latency p50/p95

### Quality gates (unchanged from Phase 5 plan / Master Plan S1/S2)
- `min_f2p 0.15` quality floor, `min_p2p 0.90` regression ceiling
- Final champion: F2P ≥ 30% on `full` golden (S1), P2P ≥ 90% (S2)

## 4. CLI Surface After Redesign

```
python -m evaluation.cli run      --mode smoke|dev|final|full [--models ...] [--prompts ...]
python -m evaluation.cli run-golden / run-swebench / run-baseline   (existing, tier-aware)
python -m evaluation.cli run-prompt-ab   (existing)
python -m evaluation.cli compare --run-ids a,b,c   (existing + CI/McNemar/cost columns)
```

## 5. Implementation Steps (topological)

| # | Step | Est. |
|---|------|------|
| 1 | Delete agentic files + CLI command + test; fix imports/refs | 20 min |
| 2 | Add `stats.py` (Wilson CI, McNemar, bootstrap) + unit tests | 1 hr |
| 3 | Tier plumbing in `cli.py`/`config.py` (`--mode`, seed-42 subsets) | 45 min |
| 4 | Cost estimate in `harness.py` + `compare` report columns | 30 min |
| 5 | Unit tests: stats (incl. paired-significance correctness), tiers, cost | 1 hr |
| 6 | Run `pytest` full suite → all green (**663 tests passing**: 645 pre-review + 18 review-round-2 regression tests) | 15 min |
| 7 | `smoke` e2e on Modal (20 instances, baseline_14b) → W&B artifacts | 1 hr |
| 8 | Update IMPLEMENTATION-LOG (deviation) + README eval section | 20 min |

**Total: ~4 hrs**

## 6. Definition of Done

- [ ] Zero mini-SWE-agent references in repo (code, tests, workflows, docs)
- [ ] `python -m evaluation.cli run --mode smoke` runs end-to-end on Modal, logs W&B
- [ ] `compare` reports F2P (Wilson CI), P2P, McNemar p-values, cost
- [ ] Tier subsets deterministic (seed 42) — same instances across variants
- [ ] All unit + integration tests pass
- [ ] IMPLEMENTATION-LOG deviation entry + README eval section updated

## 7. Deferred (not this redesign)

- CI eval workflow (`.github/workflows/eval.yml`) — Phase 7 per Master Plan
- Execution feedback (multi-turn) — v2 per Phase 5 plan §16
- Optuna HPO — v2 per Master Plan 4.12
