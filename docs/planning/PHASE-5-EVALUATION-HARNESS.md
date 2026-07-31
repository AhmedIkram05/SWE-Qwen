# Phase 5 Implementation Plan: Evaluation Harness

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Draft v1.0
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 3 complete (data pipeline, golden eval set), Phase 4 complete (training pipeline, tokenized shards, W&B artifacts)

---

## 1. Objective

Build an **execution-based evaluation harness** that:
- Generates patches from model (baseline + 3 QLoRA variants) on golden eval set (2,056 SWE-bench Verified+Test+Dev)
- Applies patches to repos at `base_sha`, runs test suites in isolated Modal containers
- Computes **F2P** (Fail-to-Pass) and **P2P** (Pass-to-Pass) metrics per model
- Logs per-example + aggregate results to W&B artifacts
- Re-validates P4 proxy champion (`baseline_14b`) with real F2P (not fresh selection)
- Runs SWE-bench Verified (500) as secondary benchmark
- Runs prompt A/B test (2-3 templates) against golden set
- Supports resume, lightweight CI sampling, full merge evaluation

---

## 2. Inputs (from completed phases)

| Source | Artifact | Location |
|--------|----------|----------|
| Phase 3 | Golden eval set (2,056 records with `FAIL_TO_PASS`/`PASS_TO_PASS`) | GCS `gs://swe-qwen-datasets/datasets/{run_id}/golden.jsonl` + `metadata.base_sha`, `metadata.head_sha`, `metadata.test_patch` |
| Phase 3 | SWE-bench Verified subset (500) | Same GCS location, filter by `metadata.is_verified==true` |
| Phase 4 | Tokenized shards for inference | `data/tokenized/shards/test/` |
| Phase 4 | Trained LoRA adapters (3 variants) | W&B artifacts: `model-qwen3-14b-{variant}` (type: `model_checkpoint`) |
| Phase 4 | Baseline model (Qwen3-14B) | HF Hub `Qwen/Qwen3-14B` |
| Phase 4 | Prompt templates (4 Jinja2) | `training/prompts/*.j2` |
| Phase 4 | `f2p_proxy.py` | P4 proxy champion selection (reuse `select_champion()`, re-validate with real F2P) |

---

## 3. Module Structure (Flat, matching `data_engineering/`)

```
evaluation/
├── __init__.py
├── schema.py              # Pydantic models: EvalInput, EvalResult, F2PMetrics, EvalRun
├── config.py              # EvalConfig (Pydantic Settings)
├── test_runner.py         # Core: run tests in Modal container, apply patch, collect results
├── patch_applier.py       # git apply → unidiff fallback
├── metrics.py             # F2P/P2P computation
├── harness.py             # Orchestrator + entry points (golden, swebench, baseline) + resume + W&B logging
├── prompt_ab_test.py      # Prompt A/B testing (2-3 templates)
├── inference.py           # Batch inference for patch generation (Modal + vLLM + LoRA)
├── comparison.py          # Re-validate P4 proxy champion with real F2P
└── cli.py                 # Typer CLI: run, run-golden, run-swebench, run-prompt-ab, run-baseline
```

**Consolidation rationale:** `golden_runner`, `swebench_runner`, `baseline_runner` are thin wrappers around `harness.run_batch()` with different data filters → folded into `harness.py` entry points. `resume.py` and `wandb_logger.py` are single-class modules → folded into `harness.py`. `comparison.py` stays separate (Phase 9 dependency). `prompt_ab_test.py` stays separate (user-facing entry point).

---

## 4. Detailed Module Specifications

### 4.1 `evaluation/config.py`

```python
class EvalConfig(BaseSettings):
    # Data
    golden_data_path: str = "gs://swe-qwen-datasets/datasets/{run_id}/golden.jsonl"  # GCS source
    swebench_verified_filter: str = "metadata.is_verified==true"  # filter from golden

    # Models
    baseline_model: str = "Qwen/Qwen3-14B"
    wandb_entity: str = "2571642-university-of-dundee"  # override via WANDB_ENTITY env var
    wandb_project: str = "swe-qwen"
    lora_artifact_pattern: str = "model-qwen3-14b-{variant}"  # W&B artifact naming

    # Modal
    modal_volumes: dict[str, str] = {
        "repo_cache": "eval-repo-cache",
        "test_cache": "eval-test-cache",
    }
    docker_image_base: str = "python:3.11-slim"
    gpu_type: str = "a10g-24gb"  # for inference

    # Test execution
    test_timeout_seconds: int = 30
    repo_timeout_seconds: int = 300
    max_retries: int = 2
    flaky_threshold: float = 0.5  # if pass rate < 0.5 across retries → flaky

    # Quality Gates (ADR-005, Master Plan S2)
    min_f2p_threshold: float = 0.15  # Quality floor: minimum F2P to pass
    min_p2p_threshold: float = 0.90  # Regression ceiling: P2P ≥ 90% (no regressions)

    # Sampling
    ci_sample_size: int = 50  # lightweight PR eval
    ci_random_seed: int = 42

    # Resume
    checkpoint_dir: Path = Path("data/eval_checkpoints")
    resume_from: str | None = None  # run_id to resume

    # Output
    output_dir: Path = Path("data/eval_results")
    wandb_log_per_example: bool = True
    wandb_log_aggregate: bool = True

    # Comparison
    comparison_run_ids: str = ""  # Comma-separated run IDs for champion selection

    model_config = SettingsConfigDict(env_prefix="EVAL_", env_file=".env")
```

---

### 4.2 `evaluation/schema.py`

```python
# Pydantic models for evaluation I/O


class TestResult(BaseModel):
    name: str
    status: Literal["passed", "failed", "errored", "skipped", "flaky"]
    duration: float
    output: str = ""
    retry_count: int = 0


class PatchApplicationResult(BaseModel):
    success: bool
    method_used: Literal["git_apply", "unidiff_fallback", "failed"]
    error: str | None = None
    files_modified: list[str] = []


class EvalInput(BaseModel):
    instance_id: str
    repo: str
    issue_body: str
    base_sha: str
    head_sha: str
    test_patch: str  # ground truth test changes
    fail_to_pass: list[str]  # test names that should fail→pass
    pass_to_pass: list[str]  # test names that should pass→pass
    repo_domain: str
    metadata: dict[str, Any] = {}


class EvalResult(BaseModel):
    instance_id: str
    repo: str
    model_name: str
    variant: str
    prompt_template: str
    generated_patch: str
    patch_application: PatchApplicationResult
    tests_before: list[TestResult]  # at base_sha
    tests_after: list[TestResult]  # at head_sha + generated patch
    f2p: float  # 0.0-1.0
    p2p: float  # 0.0-1.0
    latency_seconds: float
    timestamp: datetime
    error: str | None = None


class F2PMetrics(BaseModel):
    model_name: str
    variant: str
    prompt_template: str
    total_examples: int
    successful_patches: int
    f2p_rate: float
    f2p_count: int
    p2p_rate: float
    p2p_count: int
    avg_latency: float
    flaky_test_rate: float
    per_repo_breakdown: dict[str, dict]


class EvalRun(BaseModel):
    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    config: EvalConfig
    models_evaluated: list[str]
    results: list[EvalResult]
    aggregate: list[F2PMetrics]
    status: Literal["running", "completed", "failed", "partial"]
```

---

### 4.3 `evaluation/patch_applier.py`

```python
def apply_patch_git(repo_path: Path, patch: str, base_sha: str) -> PatchApplicationResult:
    """Try git apply first (most faithful)."""
    # 1. git checkout base_sha
    # 2. git apply --check patch
    # 3. git apply patch
    # Return result


def apply_patch_unidiff(repo_path: Path, patch: str) -> PatchApplicationResult:
    """Fallback: parse with unidiff, apply manually."""
    # Use unidiff.PatchSet to parse hunks
    # Apply each hunk to target file
    # Return result


def apply_patch(repo_path: Path, patch: str, base_sha: str) -> PatchApplicationResult:
    """Main entry: git apply → unidiff fallback."""
    result = apply_patch_git(repo_path, patch, base_sha)
    if not result.success:
        logger.warning(f"git apply failed: {result.error}, trying unidiff")
        result = apply_patch_unidiff(repo_path, patch)
        result.method_used = "unidiff_fallback"
    return result
```

---

### 4.4 `evaluation/test_runner.py` — **Core Module**

```python
# Modal function for isolated test execution


@app.function(
    image=Image.from_registry("python:3.11-slim").pip_install(["pytest", "gitpython"]),
    volumes={"/repo_cache": repo_volume, "/test_cache": test_volume},
    timeout=300,
    gpu=None,  # CPU only for test execution
)
def run_tests_in_container(
    repo: str,
    base_sha: str,
    test_patch: str | None,
    generated_patch: str | None,
    fail_to_pass: list[str] = [],
    pass_to_pass: list[str] = [],
    timeout: int = 30,
    max_retries: int = 2,
) -> dict:
    """
    Execute test suite in isolated container.
    Uses pytest -k "test_name1 or test_name2" to run specific tests from FAIL_TO_PASS/PASS_TO_PASS.
    Returns: {tests_before: [...], tests_after: [...], patch_application: {...}, ground_truth: {...}}
    """
    # 1. Clone repo to /repo_cache/{repo} if not cached
    # 2. Checkout base_sha
    # 3. Run pytest -k "f2p_test1 or f2p_test2 or ..." → tests_before
    # 4. If test_patch: apply it (ground truth head state)
    # 5. Run pytest → tests_head (for verification)
    # 6. If generated_patch: revert to base_sha, apply generated_patch
    # 7. Run pytest → tests_after
    # 8. Ground truth verification: compute F2P on test_patch → should be 100%
    # 1. Clone repo to /repo_cache/{repo} if not cached
    # 2. Checkout base_sha
    # 3. Run pytest on test_dirs → tests_before
    # 4. If test_patch: apply it (ground truth head state)
    # 5. Run pytest → tests_head (for verification)
    # 6. If generated_patch: revert to base_sha, apply generated_patch
    # 7. Run pytest → tests_after
    # 8. Ground truth verification: compute F2P on test_patch → should be 100%
    #    if test_patch:
    #        gt_f2p, gt_p2p = compute_f2p(tests_before, tests_head, fail_to_pass, pass_to_pass)
    #        if gt_f2p < 1.0:
    #            logger.warning(f"Ground truth F2P={gt_f2p:.2%} < 100% - test patch may be incomplete")
    # 9. Return all results with retry logic for flaky detection

    ground_truth = {}
    if test_patch:
        tests_head = collect_test_results(repo_path, test_dirs, timeout, max_retries)
        gt_f2p, gt_p2p = compute_f2p(tests_before, tests_head, fail_to_pass, pass_to_pass)
        ground_truth = {"f2p": gt_f2p, "p2p": gt_p2p, "warning": gt_f2p < 1.0}
    return {
        "tests_before": tests_before,
        "tests_after": tests_after,
        "patch_application": patch_app_result,
        "ground_truth": ground_truth,
    }


def collect_test_results(
    repo_path: Path, test_dirs: list[str], timeout: int, max_retries: int
) -> list[TestResult]:
    """Run pytest with retries, return TestResult list."""
    # Use pytest --json-report or parse stdout
    # Retry failed/errored tests up to max_retries
    # Mark as flaky if inconsistent across retries
```

**Key implementation details:**
- Cache repos in Modal volume `/repo_cache` keyed by `{repo}@{base_sha}`
- Use `pytest -x --tb=short --json-report` for structured output
- Handle pytest not installed → install in container
- Handle missing test dirs gracefully
- **Ground truth verification**: After applying `test_patch`, run tests and verify F2P=100% on `fail_to_pass` tests. If not, log warning — test patch may be incomplete or repo state drifted.

---

### 4.5 `evaluation/metrics.py`

```python
def compute_f2p(
    tests_before: list[TestResult],
    tests_after: list[TestResult],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> tuple[float, float, int, int]:
    """
    F2P = |{t ∈ fail_to_pass : t failed before ∧ t passed after}| / |fail_to_pass|
    P2P = |{t ∈ pass_to_pass : t passed before ∧ t passed after}| / |pass_to_pass|
    """
    before_map = {t.name: t.status for t in tests_before}
    after_map = {t.name: t.status for t in tests_after}

    f2p_passed = sum(
        1 for t in fail_to_pass if before_map.get(t) == "failed" and after_map.get(t) == "passed"
    )
    f2p_total = len(fail_to_pass)

    p2p_passed = sum(
        1 for t in pass_to_pass if before_map.get(t) == "passed" and after_map.get(t) == "passed"
    )
    p2p_total = len(pass_to_pass)

    return (
        f2p_passed / f2p_total if f2p_total else 0.0,
        p2p_passed / p2p_total if p2p_total else 1.0,
        f2p_passed,
        p2p_passed,
    )


def aggregate_metrics(results: list[EvalResult]) -> F2PMetrics:
    """Aggregate per-example results into F2PMetrics."""
    # Group by model/variant/prompt
    # Compute rates, latency stats, flaky rates
```

---

### 4.6 `evaluation/harness.py` — **Orchestrator + Entry Points**

```python
class CheckpointManager:
    """Per-repo checkpoint resume (was resume.py)."""

    def __init__(self, checkpoint_dir: Path): ...
    def get_checkpoint_key(self, run_id, repo, model, variant): ...
    def is_completed(self, key): ...
    def save_result(self, key, result): ...
    def load_results(self, run_id): ...


class WandbLogger:
    """W&B artifact logging (was wandb_logger.py)."""

    def log_eval_run(run, config): ...
    def log_per_example(results, run_id): ...
    def log_aggregate(metrics, run_id): ...


class EvaluationHarness:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: list[EvalResult] = []
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        self.logger = WandbLogger(config)

    def load_examples(self, split: str = "golden") -> list[EvalInput]:
        """Load EvalInput from GCS golden.jsonl + metadata."""
        # Read JSONL from GCS, reconstruct EvalInput from metadata

    def run_example(
        self, example: EvalInput, model_name: str, variant: str, prompt_template: str
    ) -> EvalResult:
        """Single example: generate patch → apply → run tests → compute metrics."""

    def run_batch(
        self,
        examples: list[EvalInput],
        model_name: str,
        variant: str,
        prompt_template: str,
        resume_from: str | None = None,
    ) -> list[EvalResult]:
        """Run batch with checkpoint resume per-repo."""

    def run_golden(
        self, models: list[tuple[str, str]], prompt_templates: list[str] = ["chat"]
    ) -> EvalRun:
        """Entry: run golden eval on all model/variant/prompt combos."""
        # Load golden examples from GCS
        # For each model+variant+prompt: run_batch
        # Aggregate, log to W&B

    def run_swebench_verified(self, models: list[tuple[str, str]]) -> EvalRun:
        """Entry: filter golden by is_verified, run same harness."""

    def run_baseline(self, model: str = "Qwen/Qwen3-14B") -> EvalRun:
        """Entry: evaluate unfine-tuned base model (no LoRA)."""
```

---

### 4.7 `evaluation/prompt_ab_test.py`

```python
@app.local_entrypoint()
def run_prompt_ab_test(
    model: str = "qwen3-14b:baseline_14b",
    templates: str = "system,user,assistant,chat",  # or subset
    sample: int = 200,
):
    """Prompt A/B: test 2-3 prompt templates against golden sample."""
    # Load prompt templates from training/prompts/
    # Run harness with each template
    # Compare F2P/P2P across templates
    # Log to W&B: "eval-prompt-ab"
```

---

### 4.8 `evaluation/inference.py` — **Batch Inference for Patch Generation**

```python
# Modal function for vLLM + LoRA batch inference

vllm_image = Image.from_registry("vllm/vllm-openai:latest").pip_install(["peft", "wandb"])


@app.function(
    image=vllm_image,
    gpu="a10g-24gb",
    volumes={"/models": model_volume},
    timeout=600,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("hf-secret")],
)
def generate_patches_batch(
    model_name: str,  # key from models.yaml
    variant: str,  # determines LoRA adapter path
    prompt_template: str,  # "chat", "system", etc.
    examples: list[EvalInput],
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> list[str]:
    """Batch inference for patch generation."""
    # 1. Resolve LoRA adapter path from W&B artifact or local volume
    # 2. Load base model + LoRA via vLLM (LoraRequest)
    # 3. Load PromptLoader, render prompts for each example
    # 4. vLLM batch generate with sampling params
    # 5. Extract generated patches from completions
    # 6. Return list of generated patches (same order as examples)
```

---

### 4.12 `evaluation/wandb_logger.py` — **W&B Logging (inline in harness.py)**

*Note: WandbLogger class is now inline in `harness.py` (see §4.6). This section documents the logging interface.*

```python
# Methods on WandbLogger class in harness.py:
def log_eval_run(run: EvalRun, config: EvalConfig):
    """Log EvalRun to W&B as artifact + summary metrics."""
    # 1. Per-example: JSONL artifact "eval-results-{run_id}"
    # 2. Aggregate: summary metrics to W&B run
    # 3. Per-repo breakdown: table artifact
    # 4. Link to model checkpoint artifact (lineage)


def log_per_example(results: list[EvalResult], run_id: str):
    """Write JSONL, upload as artifact."""


def log_aggregate(metrics: list[F2PMetrics], run_id: str):
    """Log summary scalars to W&B run."""
```

---

### 4.13 `evaluation/comparison.py` — **Re-validate P4 Proxy Champion**

```python
"""
Re-validate P4 proxy champion with real F2P.
Imports select_champion from scripts.f2p_proxy (P4 artifact).
"""

from evaluation.schema import F2PMetrics, EvalRun
from evaluation.config import EvalConfig
from scripts.f2p_proxy import select_champion, compute_proxy_f2p_scores  # P4 reuse
import wandb


def load_all_eval_runs(run_ids: list[str]) -> list[EvalRun]:
    """Download and parse EvalRun from W&B artifacts."""


def extract_model_metrics(runs: list[EvalRun]) -> dict[str, F2PMetrics]:
    """Aggregate per-model metrics across runs."""


def revalidate_champion(
    metrics: dict[str, F2PMetrics],
    proxy_champion: str,  # from P4: "baseline_14b"
    min_f2p: float,
    min_p2p: float,
) -> tuple[str, F2PMetrics] | None:
    """
    Re-validate P4 proxy champion with real F2P:
    1. Filter: P2P >= min_p2p (regression ceiling - ADR-005)
    2. Filter: F2P >= min_f2p (quality floor)
    3. Rank remaining by F2P descending
    4. If proxy_champion is still #1 → confirmed
    5. If another model is #1 → proxy was wrong, promote new champion
    """
    candidates = [
        (model, m)
        for model, m in metrics.items()
        if m.p2p_rate >= min_p2p and m.f2p_rate >= min_f2p
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1].f2p_rate, reverse=True)
    return candidates[0]
```

---

### 4.14 `evaluation/cli.py` — **Typer CLI**

```python
@app.command()
def run(
    models: str = typer.Option("qwen3-14b:baseline_14b,qwen3-14b:higher_rank_14b,qwen3-14b:higher_lr_14b"),
    split: str = typer.Option("golden", help="golden|swebench_verified|test"),
    prompts: str = typer.Option("chat"),
    sample: int = typer.Option(0, help="0 = all"),
    resume: str | None = typer.Option(None),
    ci_mode: bool = typer.Option(False, help="Lightweight sample for CI"),
):
    """Main evaluation entry point."""
    # Parse models, prompts
    # If ci_mode: sample = 50
    # Dispatch to appropriate runner

@app.command()
def run_golden(...): ...

@app.command()
def run_swebench(...): ...

@app.command()
def run_prompt_ab(...): ...

@app.command()
def run_baseline(...): ...

@app.command()
def compare(
    run_ids: str = typer.Option(..., help="Comma-separated run_ids to compare"),
):
    """Compare multiple eval runs, output markdown table."""
```

---

## 5. Implementation Steps (Dependency-Topological Order)

| Step | Task | File(s) | Estimate | Dependencies |
|------|------|---------|----------|--------------|
| 5.0 | Verify P3/P4 artifacts + extend `swebench_ingest.py` | swebench_ingest.py | 45 min | P3, P4 |
| 5.1 | Create `evaluation/config.py` | config.py | 30 min | 5.0 |
| 5.2 | Create `evaluation/schema.py` | schema.py | 45 min | 5.1 |
| 5.3 | Create `evaluation/patch_applier.py` | patch_applier.py | 1 hr | 5.1 |
| 5.4 | Create `evaluation/test_runner.py` (Modal function) | test_runner.py | 3 hrs | 5.1, 5.3 |
| 5.5 | Create `evaluation/metrics.py` | metrics.py | 45 min | 5.2 |
| 5.6 | Create `evaluation/harness.py` | harness.py | 2.5 hrs | 5.2, 5.4, 5.5 |
| 5.7 | Create `evaluation/prompt_ab_test.py` | prompt_ab_test.py | 1.5 hrs | 5.6 |
| 5.8 | Create `evaluation/inference.py` | inference.py | 2 hrs | 5.4, 5.6 |
| 5.9 | Create `evaluation/comparison.py` | comparison.py | 1 hr | 5.6 |
| 5.10 | Create `evaluation/cli.py` | cli.py | 45 min | 5.6-5.9 |
| 5.11 | Unit tests: schema, metrics, patch_applier | test_eval_unit.py | 1.5 hrs | 5.2, 5.3, 5.5 |
| 5.12 | Integration test: harness mock + smoke test | test_eval_integration.py | 1.5 hrs | 5.6 |
| 5.13 | Baseline model evaluation run | Manual Modal run | 2 hrs | 5.10 |
| 5.14 | Golden eval: all 3 variants | Manual Modal run | 4 hrs | 5.10 |
| 5.15 | SWE-bench Verified eval | Manual Modal run | 1 hr | 5.10 |
| 5.16 | Prompt A/B test run | Manual Modal run | 2 hrs | 5.7 |
| 5.17 | Re-validate P4 proxy champion | Manual | 30 min | 5.14-5.16 |

**Total: ~24 hours** (reduced from 30 via consolidation, removed CI/cost tracking)

---

## 6. Test Execution Architecture Details

**Modal Container Strategy:**
- Pre-built base image with common test deps (build once, reuse):
  ```python
  BASE_IMAGE = (
      Image.from_registry("python:3.11-slim")
      .pip_install(
          [
              "pytest>=8.0",
              "pytest-timeout>=2.3",
              "pytest-json-report>=1.5",
              "gitpython>=3.1",
              "unidiff>=0.7",
          ]
      )
      .apt_install(["git"])
  )
  ```
- Cache repos in Modal volume `/repo_cache/{repo}@{sha}`
- Cache test dependencies in `/test_cache/{repo}` (per-repo `pip install -e .`)
- This avoids 18 Docker builds; clone + install is fast enough on Modal CPU

**Patch Application Flow:**
```
generated_patch + base_sha
       │
       ▼
git apply --check (dry run)
       │
       ├─ success → git apply ✓
       │
       └─ fail → unidiff.PatchSet parse → manual hunk apply
                     │
                     ├─ success ✓
                     └─ fail → PatchApplicationResult(success=False)
```

**Flaky Detection:**
- Run pytest 3x total (1 initial + 2 retries)
- If status changes across runs → mark `flaky`
- Flaky tests excluded from F2P/P2P denominator (or counted as 0.5)
- Report flaky rate per repo

**Timeouts:**
- `@app.function(timeout=300)` = 5 min per repo
- `pytest --timeout=30` per test
- If repo exceeds 5 min → mark as timeout, return partial results

**Ground Truth Verification (NEW):**
- After applying `test_patch` (ground truth), run tests and verify F2P=100% on `fail_to_pass` tests
- If F2P < 100%, log warning: test patch may be incomplete or repo state drifted
- Return `ground_truth` dict in test runner output for sanity checking

---

## 7. Inference for Patch Generation

**Decision:** Build lightweight inference Modal function in Phase 5 (Option B).
**Engine:** vLLM (confirmed per ADR & Vision requirement — CV/resume targeting for AI engineering roles).

```python
# evaluation/inference.py (new module)
@app.function(
    image=vllm_image,
    gpu="a10g-24gb",
    volumes={"/models": model_volume},
    timeout=600,
)
def generate_patches_batch(
    model_name: str,
    variant: str,  # determines LoRA adapter path
    prompt_template: str,
    examples: list[EvalInput],
    max_new_tokens: int = 2048,
    temperature: float = 0.1,
) -> list[str]:
    """Batch inference for patch generation."""
    # Load base model + LoRA adapter from W&B artifact or volume
    # Render prompts using PromptLoader
    # vLLM batch generate
    # Return list of generated patches
```

---

## 8. W&B Logging Structure

**Artifacts per eval run:**
| Artifact | Type | Contents |
|----------|------|----------|
| `eval-results-{run_id}` | `eval_results` | JSONL: one EvalResult per line |
| `eval-aggregate-{run_id}` | `eval_metrics` | F2PMetrics per model/variant/prompt |
| `eval-per-repo-{run_id}` | `eval_breakdown` | CSV: repo-level F2P/P2P |
| `eval-baseline-{run_id}` | `eval_results` | Baseline model results |
| `eval-prompt-ab-{run_id}` | `eval_results` | Prompt A/B results |
| `eval-swebench-{run_id}` | `eval_results` | SWE-bench Verified results |

**Summary metrics logged to W&B run:**
- `eval/f2p_rate`, `eval/p2p_rate` per model/variant
- `eval/latency_p50`, `eval/latency_p95`
- `eval/flaky_rate`
- `eval/successful_patches` / `eval/total_examples`

**Lineage:** Each eval artifact references:
- Model checkpoint artifact (from Phase 4)
- Dataset artifact (golden split from Phase 3)
- Prompt template version (from Phase 4 W&B artifact)

---

## 9. Acceptance Criteria (Phase Exit Gate)

| # | Criterion | Verification |
|---|-----------|--------------|
| 1 | `evaluation/` package exists with all 15 modules (includes `comparison.py`) | `ls evaluation/` |
| 2 | `python -m evaluation.cli run --models "qwen3-14b:baseline_14b" --sample 10` runs without error | Manual test |
| 3 | Golden eval completes on all 3 variants (2,056 examples each) | Modal run logs |
| 4 | F2P/P2P metrics computed and logged to W&B per model/variant | W&B dashboard |
| 5 | Per-example results artifact exists in W&B with `eval_results` type | W&B UI |
| 6 | SWE-bench Verified (500) eval runs and logs separate artifact | W&B UI |
| 7 | Baseline model eval runs and logs artifact | W&B UI |
| 8 | Prompt A/B test runs (2-3 templates) and logs comparison | W&B UI |
| 9 | Resume works: interrupt + `--resume run_id` continues from last repo | Manual test |
| 10 | `swebench_ingest.py` populates `metadata.base_sha`, `metadata.test_patch` | Unit test |
| 11 | Unit tests pass: `pytest tests/test_eval_unit.py` | CI |
| 12 | Integration tests pass: `pytest tests/test_eval_integration.py` | CI |
| 13 | Patch application: git apply → unidiff fallback verified on real patches | Unit test |
| 14 | Flaky detection marks inconsistent tests correctly | Unit test |
| 15 | `evaluation/comparison.py` re-validates P4 proxy champion | Manual test |
| 16 | Ground truth verification runs and logs warning if F2P < 100% | Modal run logs |
| 17 | `scripts.f2p_proxy.select_champion` imported (not re-implemented) | Code review |

---

## 11. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Test suite fails to install in container | Medium | High | Pre-install common deps in base image; cache pip cache in volume |
| Modal volume cache corruption | Low | Medium | Version cache key with repo+sha; invalidate on failure |
| vLLM + LoRA loading fails in inference function | Medium | High | Test inference function separately (Step 5.0 infra check) |
| Generated patches have syntax errors | High | Medium | Patch validation in `patch_applier` catches; log, continue |
| F2P rates very low (< 5%) on all variants | Medium | High | Baseline establishes floor; if all low, data/training issue |
| W&B artifact download fails | Low | Medium | Retry logic in artifact loader; cache locally |

---

## 12. Definition of Done

1. All 17 implementation steps complete
2. All 17 acceptance criteria verified
3. `pytest tests/test_eval_unit.py` and `tests/test_eval_integration.py` pass (≥ 25 test cases)
4. Golden eval run ID recorded, W&B artifacts visible
5. SWE-bench Verified run ID recorded
6. Baseline eval run ID recorded
7. Prompt A/B run ID recorded
8. Champion model identified from real F2P (not proxy) → W&B Registry `champion` alias updated
9. `swebench_ingest.py` extended with `base_sha` and `test_patch` population

---

## 13. Next Phase Dependency

**Phase 6 (Inference API)** consumes:
- Champion model from W&B Registry `champion` alias (validated by real F2P)
- LoRA adapter path from eval artifacts
- vLLM config benchmarks (deferred to Phase 6)

**Phase 9 (Promotion Pipeline)** consumes:
- Evaluation harness as library (`from evaluation.harness import EvaluationHarness`)
- W&B artifact naming conventions established here
- Quality gate logic from `scripts/check_eval_gate.py`

---

## 14. File Manifest After Phase 5

```
swe-qwen/
├── evaluation/
│   ├── __init__.py
│   ├── config.py
│   ├── schema.py
│   ├── patch_applier.py
│   ├── test_runner.py
│   ├── metrics.py
│   ├── harness.py              # includes: runners, resume, wandb_logger
│   ├── prompt_ab_test.py
│   ├── inference.py            # batch inference for patch gen
│   ├── comparison.py           # re-validate P4 proxy champion
│   └── cli.py                  # Typer CLI entrypoint
├── data_engineering/
│   ├── swebench_ingest.py      # extended: base_sha, test_patch
├── scripts/
│   ├── f2p_proxy.py            # P4 artifact (imported by comparison.py)
│   └── run_3config_comparison.py
├── tests/
│   ├── test_eval_unit.py       # schema, metrics, patch_applier
│   └── test_eval_integration.py  # harness mock, smoke test
└── data/eval_checkpoints/      # Created at runtime
```

---

## 15. Grilling Decisions Record

| Question | Decision | Rationale |
|----------|----------|-----------|
| Q1: Test execution architecture | D — Modal functions with cached Docker images | Matches Phase 4 Modal infra, parallelizable, clean isolation |
| Q2: Patch application | C — git apply → unidiff fallback | Faithful to real workflow, robust fallback |
| Q3: Flaky test handling | A — Retry 2x, mark flaky | Balance of rigor and speed |
| Q4: Timeout strategy | A — Per-test 30s, per-repo 5min | Prevents hangs |
| Q5: Golden eval scope | B — Verified+Test+Dev (2,056) | Execution-verifiable ground truth |
| Q6: SWE-bench Verified secondary | A — Yes | Enables comparison to published results |
| Q7: Baseline evaluation | A — Run actual baseline model | Real F2P for Champion/Challenger |
| Q8: Fine-tuned evaluation | B — All 3 variants | Re-validate P4 proxy champion (`baseline_14b`) with real F2P |
| Q9: Output format | C — Both per-example + aggregate | Debugging + dashboards |
| Q10: W&B integration | A — Read from Registry `champion` alias | Clean separation, enables Phase 9 |
| Q11: Prompt A/B testing | A — Yes | MASTER-PLAN 4.3 requirement |
| Q12: SWE-bench runner | A — Same harness, filtered data | DRY, consistent metrics |
| Q13: Module structure | A — Flat like data_engineering/ | Proven pattern |
| Q14: Resume support | A — Yes, per-repo checkpoints | Long runs need resilience |
| Q15: Cost tracking | DEFERRED | Not needed for P5; add when eval runs become cost-significant |

---

## 16. Review Gap Fixes (Post-Grilling Review)

The following 7 gaps were identified during plan review against Master Plan/ADR and are now addressed:

| Gap | Fix | Location |
|-----|-----|----------|
| 1. Missing comparison framework for champion selection (Master Plan 4.12, 5.8) | Added `evaluation/comparison.py` — reuses P4 `f2p_proxy.py` `select_champion()`, re-validates proxy champion with real F2P | Section 4.13, Step 5.9 |
| 2. Missing P2P ≥ 90% regression ceiling gate (ADR-005, S2) | Added `min_p2p_threshold` to config; enforced in `comparison.py` | Section 4.1, Section 4.13 |
| 3. Missing ground truth test_patch verification | Added ground truth verification in `test_runner.py` — computes F2P on test_patch, warns if < 100% | Section 4.4 |
| 4. Missing cost tracking | DEFERRED — not needed for P5; add when eval runs become cost-significant | N/A |
| 5. Golden set size discrepancy (2,056 vs 8-12k) | Clarified: Golden = 2,056 (Verified+Test+Dev, execution-verifiable); Training = ~10,882 (includes Train split) | Below |
| 6. Execution feedback deferral undocumented | Documented: deferred to v2 (Phase 11+); Phase 5 uses single-turn issue→patch→evaluate | Below |
| 7. Docker caching strategy underspecified | Specified: pre-built base image with common deps; per-repo pip cache in Modal volume | Section 6 |

### Golden Set Size Clarification

| Split | Source | Count | Purpose |
|-------|--------|-------|---------|
| **Golden (eval)** | SWE-bench Verified + Test + Dev | **2,056** | Execution-verifiable F2P evaluation |
| **Training** | SWE-bench Train (Python) + Verified + Test + Dev | **~10,882** | Model training (issue→patch) |
| **SWE-bench Verified (benchmark)** | SWE-bench Verified only | **500** | Published benchmark comparison |

**Master Plan "8-12k" = training set. Phase 5 golden = 2,056.** Discrepancy resolved.

### Execution Feedback Deferral

> **Note on Execution Feedback (ADR & Vision §3):** The vision specifies "Issue Description + Execution Feedback → Code Patch" where test failure output conditions patch generation. This requires a multi-turn refinement loop (run tests → feed failures → generate patch → repeat). **Deferred to v2 (Phase 11+).** Phase 5 uses single-turn: issue → patch → evaluate. See Phase 11+ for iterative refinement.

### Docker Caching Strategy

```python
# Pre-built base image with common test deps (build once, reuse)
BASE_IMAGE = (
    Image.from_registry("python:3.11-slim")
    .pip_install(
        [
            "pytest>=8.0",
            "pytest-timeout>=2.3",
            "pytest-json-report>=1.5",
            "gitpython>=3.1",
            "unidiff>=0.7",
        ]
    )
    .apt_install(["git"])
)

# Per-repo: clone → pip install -e . (cached in /test_cache/{repo})
# Subsequent runs: reuse cached env
```

---

*End of Phase 5 Implementation Plan*
