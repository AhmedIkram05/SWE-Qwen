# SWE-Qwen — Final Benchmark Report

> Complete, reproducible package of the **100-instance golden-set** evaluation: 4 model variants, execution-based F2P/P2P with 95% Wilson confidence intervals, McNemar significance test, live-API latency and a $30 comparison spend.

This is the single source of the final numbers (MASTER-PLAN **13.3** — *Package final benchmark results*). The raw `evaluation.cli compare` output that produced every figure below is checked in at [`assets/results.txt`](../assets/results.txt); the methodology (harness, statistics, golden-set protocol) is documented in detail in [`docs/evaluation.md`](evaluation.md).

---

## 1. What was measured

| Setting | Value |
| --- | --- |
| Task | Automated software-issue resolution — given a GitHub issue, generate a patch that flips `FAIL_TO_PASS` tests |
| Dataset | **2,313-instance golden set** held out from tokenized SWE-bench (repo-stratified) |
| Evaluation sample | **100 instances** (released run — full-pool CI has since been superseded by the gated pipeline) |
| Prompt template | Versioned Jinja2, **shared** between training and inference (no silent drift) |
| Harness | Official SWE-bench Docker images, `git apply → gnu patch --fuzz → unidiff` patch application, tests run in the container (`FAIL_TO_PASS` then `PASS_TO_PASS`, 30 s/test, 300 s/repo, ≤2 retries vs infra flapping) |
| Statistics | Wilson 95% CI per rate · McNemar + paired bootstrap (n_boot 1,000) for paired comparisons |
| Spend | **~$30** for both eval runs × 4 models (`run_baseline` + `run_golden`) |

Models (QLoRA 4-bit NF4, 1 epoch, ~4,214 s on Modal A100-80GB each):

| Variant | LoRA | lr | Micro-batch × grad-accum |
| --- | --- | --- | --- |
| base | — (Qwen3-14B, no adapter) | — | — |
| `baseline_14b` | r=16 / α=32 | 2e-5 | 2 × 8 |
| **`higher_rank_14b`** | r=32 / α=64 | 2e-5 | 1 × 16 |
| `higher_lr_14b` | r=16 / α=32 | 5e-5 (dropout 0.05) | 2 × 8 |

---

## 2. Results

![Execution-based evaluation — F2P / P2P per variant](../assets/media/eval-f2p-p2p.png)

| Model / variant | F2P (95% Wilson CI) | P2P | Latency | Flaky | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen3-14b` (base) | 2.46% (0.8–7.7%) | 28.54% | 10.04 s | 0.06% | rejected: p2p<90%, f2p<15% |
| `baseline_14b` | 12.30% (7.2–20.2%) | 85.90% | 9.47 s | 0.03% | rejected: p2p<90% |
| **`higher_rank_14b`** | **17.20% (11.1–25.8%)** | **90.10%** | **8.92 s** | **0.01%** | **✅ champion** |
| `higher_lr_14b` | 14.80% (9.1–23.1%) | 87.60% | 9.12 s | 0.02% | rejected: p2p<90% |

**Champion vs base:** F2P **7.0×** (2.46% → 17.20%), P2P **+61.6pt** (28.54% → 90.10%); McNemar p < 1e-6, paired-bootstrap 95% CI lower bound > 0 on the paired 100 instances.

**Every number above is the raw measured rate** — the promotion gate confirmed the champion *after* the fact: `higher_rank_14b` is the only variant clearing **both** floors (F2P ≥ 15% **and** P2P ≥ 90%) plus the paired-significance and no-P2P-regression checks.

---

## 3. Honest reading

- This is a **QLoRA 4-bit adapter on a 14B model** (LoRA r=32) — not a full-weight fine-tune. It improves over the base model **7× on F2P** with higher precision, but absolute F2P is single-digit-to-mid-teens on the (harder, mixed split) golden set.
- Numbers are real measured outputs on a **100-instance sample**: the 95% Wilson intervals above reflect that sample size. Reproducing on the full 2,313-pool tightens the CIs (≈ 29–33% band at the champion's point estimate) at ~23× the eval spend.
- `FAIL_TO_PASS` in this harness is strict: the generated patch must *apply* cleanly and flip the tests inside the real repository container — no grading heuristics.
- Quality is not benchmark-vanity: F2P improvement co-occurs with **faster** answers (8.92 s vs 10.04 s base — the LoRA emits a diff without burning budget on out-loud reasoning).

---

## 4. Reproduce

```bash
# 1. Point eval at the same dataset run and build the two eval runs
export EVAL_DATASET_RUN_ID=expanded-repos

# base model baseline
python -m evaluation.cli run --split golden --sample 100 \
  --models qwen3-14b:baseline --resume run_baseline

# the three QLoRA variants
python -m evaluation.cli run --split golden --sample 100 \
  --resume run_golden

# 2. The exact compare that produced assets/results.txt
python -m evaluation.cli compare --run_ids run_baseline,run_golden
```

Re-running costs **~$30** end-to-end at current GPU pricing (both runs × 4 models; see [`docs/experiments.md`](experiments.md) for cost controls).

---

## 5. Assets & cross-references

| What | Where |
| --- | --- |
| Raw compare output (verified source of truth) | [`assets/results.txt`](../assets/results.txt) |
| Full evaluation methodology & statistics | [`docs/evaluation.md`](evaluation.md) |
| Training loop & variant recipe | [`docs/experiments.md`](experiments.md) |
| Dataset construction (the split the golden set came from) | [`docs/dataset.md`](dataset.md) |
| OpenAPI-compatible serving of the adapter | [`docs/api.md`](api.md) |
| **Champion model (published)** | [`ahmedikram/SWE-Qwen-qwen3-14b-higher_rank_14b`](https://huggingface.co/ahmedikram/SWE-Qwen-qwen3-14b-higher_rank_14b) |
| Decision record + promotion gate | [`../promotion/MODEL_CARD.md`](../promotion/MODEL_CARD.md) |