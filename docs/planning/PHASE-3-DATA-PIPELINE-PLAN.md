# Phase 3 Implementation Plan: Data Pipeline Engine

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Final v1.0 (grilled and approved)
**Parent Document:** `docs/MASTER-PLAN.md`
**Dependencies:** Phase 1 complete (infra, Modal, W&B), Phase 2 complete (`repos/manifest.json`)

---

## 1. Objective

Ingest issue-PR pairs from the 10+ verified repositories in `repos/manifest.json`, extract unified diffs with parsed hunks, validate against strict schema, deduplicate and filter for quality, produce repo-stratified train/val/test splits + a golden F2P eval subset, version all stages in W&B, archive to GCS, and generate a dataset card.

**Outputs:**
- `data_engineering/` Python package (11 modules + CLI)
- W&B versioned artifacts: raw, validated, cleaned, train, val, test, golden, validation_errors
- GCS archive: `gs://{bucket}/datasets/{run_id}/{stage}.jsonl` + manifest + dataset_card.md
- JSONL splits ready for Phase 4 training: `train.jsonl`, `val.jsonl`, `test.jsonl`, `golden.jsonl`

---

## 2. Selection Criteria (Locked from Manifest)

The manifest defines per-repo `ingestion_config` that Phase 3 MUST respect:

| Config Field | Source | Phase 3 Usage |
|--------------|--------|---------------|
| `default_branch` | `verify_repos.py` detected actual branch | GitHub API fetch branch |
| `issue_labels_to_include` | Manifest (default: `["bug", "defect", "fix"]`) | Filter issues by label (case-insensitive substring) |
| `pr_merge_commits_only` | Manifest (default: `true`) | Only process PRs with `merged_at` populated |
| `max_issues_per_repo` | Manifest (default: `2000`) | Cap fetch count (by `updated_at` desc) |
| `exclude_paths` | Manifest (globs: `docs/`, `examples/`, `*.md`, etc.) | Skip files matching patterns during diff extraction |
| `test_directories` | Manifest (default: `["tests/", "test/"]`) | Detect test file changes; used by Phase 5 test runner |

---

## 3. Domain Diversity (From Manifest)

The manifest guarantees domain spread (enforced in Phase 2):

| Domain Bucket | Minimum Repos | Example Repos (from manifest) |
|---------------|---------------|------------------------------|
| **web-api** | 2 | fastapi/fastapi, headroomlabs-ai/headroom |
| **cli** | 2 | fastapi/typer, textualize/textual |
| **data-ml** | 2 | mlflow/mlflow, huggingface/datasets |
| **utils** | 2 | psf/black, pydantic/pydantic, marimo-team/marimo |
| **testing** | 2 | joke2k/faker, pytest-dev/pytest |

Max 2 repos per GitHub organization (enforced in Phase 2 selection).

---

## 4. Ingestion Strategy

### 4.1 GitHub API Approach
- **Library:** `pygithub` (already in project deps)
- **Auth:** `GITHUB_TOKEN` env var (validated at pipeline start; fail fast if missing)
- **API:** REST (not GraphQL) — simpler, well-supported, matches Phase 2

### 4.2 Query Pattern
```
For each repo in manifest:
  1. GET /repos/{owner}/{repo}/issues?state=all&labels={labels}&per_page=100&sort=updated&direction=desc
  2. For each issue: GET /repos/{owner}/{repo}/issues/{issue_number}/timeline
     - Filter: event == "cross-referenced" AND source.issue.pull_request == true
     - Get PR: source.issue.as_pull_request()
     - Keep only if PR.merged == true
  3. For each linked merged PR:
     - GET /repos/{owner}/{repo}/pulls/{pr_number}/files → files changed + patch
     - GET /repos/{owner}/{repo}/pulls/{pr_number}/commits → commit messages
     - GET /repos/{owner}/{repo}/pulls/{pr_number} → PR title, body, metadata
```

### 4.3 Rate Limit Handling
- Exponential backoff: 1s → 2s → 4s → 8s → 16s → 32s → 60s (max)
- Respect `Retry-After` header from GitHub
- Conditional requests: `If-None-Match` with ETag for unchanged data
- Proactive throttle: check `X-RateLimit-Remaining` after each API call; if < 100, inject delay proportional to `X-RateLimit-Reset` to pre-empt 429s
- Batch 50 issues, process with `ThreadPoolExecutor(max_workers=4)` per repo

### 4.4 Large Repo Handling
- Respect `manifest.ingestion_config.max_issues_per_repo` (default 2000)
- Stop after limit, log warning with count, continue to next repo
- Prevents runaway on huge repos (e.g., mlflow 2382 commits/6mo)

---

## 5. Data Quality Filters (Cleaning Stage)

| Filter | Action | Condition | Rationale |
|--------|--------|-----------|-----------|
| **Test file change** | **REMOVE** | No file in `test_files_changed` matches `test_directories` patterns | Ensures F2P signal exists |
| **Patch size** | **REMOVE** | `patch_diff` lines > `max_patch_lines` (default 500) | Too large for model context |
| **Binary files** | **REMOVE** | Any hunk contains binary markers (`Binary files differ`) | Not usable for code generation |
| **Non-Python files (strict)** | **REMOVE** | >50% of files in `files_changed` don't end with `.py` | Python-only training objective |
| **Non-Python files (lenient)** | **WARN** | Any file in `files_changed` doesn't end with `.py` | Mixed-language PRs; keep record, log warning |
| **Empty issue body** | **REMOVE** | `issue_body` is empty or whitespace | No problem description = no training signal |
| **No F2P signal (V1 proxy)** | **REMOVE** | `test_files_changed` is empty OR no F2P keywords (`"fix"`, `"close"`, `"resolve"`) in commit messages/PR description | V1 proxy uses keywords + test file changes, NOT `test_results.failed` (final state = still failing, not "was failing") |

**Filter Logic:** All **REMOVE** filters are applied in sequence. A record is kept only if it passes ALL **REMOVE** filters. **WARN** filters log a warning but don't remove the record.

**Note:** The "No failed tests" filter from earlier drafts (`test_results.failed == 0`) has been REMOVED — it incorrectly discarded PRs where all tests pass after the fix (i.e., successful fixes). The V1 proxy for F2P is: PR modified test files (`test_files_changed` non-empty) AND commit/PR text contains F2P keywords.

---

## 6. Splits Strategy

### 6.1 Train/Val/Test (Repo-Stratified)
- **Constraint:** Each repo appears in EXACTLY ONE split (no leakage)
- **Ratios:** 80% train / 10% val / 10% test (by repo count)
- **Algorithm:** Shuffle repos (seeded), assign sequentially to buckets
- **Result:** ~8 repos train, 1 val, 1 test (for 10 repos)

### 6.2 Golden Eval Subset
- **Source:** `test` split only (configurable via `golden_source_split`, default `"test"`)
  - **Rationale:** Sourcing from `train` or `val` would create data leakage — the model would see golden examples during training.
  - The `all` option requires explicit opt-in with a logged warning about data leakage.
- **Criterion:** Verified F2P = tests were failing before fix AND passing after fix
- **V1 Proxy (heuristic):** Select examples where `test_files_changed` is non-empty (PR modified tests) AND commit messages/PR description contain F2P keywords (`"fix"`, `"close"`, `"resolve"` with test references).
  - **Note:** `test_results.failed > 0` in the final state means tests are *still failing* — this is NOT a valid "was failing before" signal.
  - Phase 5 will refine with actual test execution at base vs head SHAs for ground truth.
- **Target:** ≥200 golden examples (use ALL verified F2P; log warning if < 200)

---

## Decisions Summary (from grilling session)

| Area | Decision |
|------|----------|
| **Data Schema** | Standard: issue_id, repo, issue_body, patch_diff, test_results, PR_title, PR_description, commit_messages, files_changed, test_files_changed, issue_labels, repo_domain, metadata |
| **Patch Format** | Both: raw unified_diff string + parsed hunks (file, old_start, old_lines, new_start, new_lines, diff_lines) |
| **Ingestion** | PyGithub REST API: paginate issues (100/page) with labels [bug, defect, fix] → timeline events → linked merged PRs → fetch diff/files/commits |
| **Rate Limits** | Exponential backoff (1s, 2s, 4s, max 60s) + respect Retry-After + conditional requests (If-None-Match) |
| **Validation** | Pydantic v2 strict=True + custom validators (patch parses, test_results structure, non-empty issue_body, list fields) |
| **Validation Errors** | Log to W&B artifact (validation_errors.jsonl) + local file; continue processing |
| **Dedup** | Primary: hash by (repo, issue_id, PR_number); Secondary: patch content hash |
| **Cleaning Filters** | Remove: no F2P signal (test_files_changed empty or no F2P keywords), patches >500 lines, binary files, non-.py patches (>50%), empty issue_body |
| **Splits** | Repo-stratified 80/10/10 (no leakage); Golden = V1 F2P proxy (test_files_changed + keywords); Target ≥200 golden |
| **W&B Versioning** | Versioned artifact per stage (raw, validated, cleaned, split); metadata: manifest hash, split ratios, golden size, pass/fail counts |
| **GCS Archival** | All stages + manifest + dataset_card.md to gs://bucket/datasets/{run_id}/{stage}.jsonl |
| **Dataset Card** | Auto-generated: size, schema, source repos, quality stats, splits, golden stats, date, git SHA, W&B links |
| **CLI** | Typer: --manifest, --output-dir, --run-id (auto UUID), --stages (all/raw/validated/cleaned/train/val/test/golden), --parallel (4), --resume-from |
| **Module Structure** | Flat: schema.py, ingest.py, validate.py, clean.py, split.py, golden.py, version.py, archive.py, card.py, run_pipeline.py, cli.py |
| **Config** | Pydantic Settings: DataPipelineConfig (batch_size, max_patch_lines, min_golden, parallel_workers, gcs_bucket, wandb_project, manifest_path, output_dir, golden_source_split, max_events_per_issue, issue_labels_to_include, test_directories) |
| **Logging** | Structured JSON via observability.logging (Phase 8 interface); fallback stdlib for Phase 3 |
| **Tests** | Unit per module (mocked GitHub/W&B/GCS) + integration @pytest.mark.integration on 1-2 test repos + property-based (hypothesis) for schema/clean/split |
| **Fixtures** | tests/fixtures/sample_issues.json (5-10 real pairs), tests/fixtures/mock_github_responses/, tests/fixtures/expected_outputs/ |
| **Phase 4 Format** | JSONL (train/val/test/golden.jsonl) matching schema.IssueRecord; training reads directly |
| **Pagination** | 100/page, batch 50 issues, ThreadPoolExecutor(4) per repo |
| **Large Repos** | Respect manifest.max_issues_per_repo (default 2000), warn if truncated |
| **Progress** | Rich progress bars per repo per stage; emit metrics dict to W&B per stage |
| **Edge Cases** | Skip individual records on error (log to validation_errors.jsonl); skip repo only on clone/auth failure |

---

## Module Specifications

### 0. `data_engineering/config.py`
**Purpose:** Centralized configuration via Pydantic Settings.

**Fields:**
```python
class DataPipelineConfig(BaseSettings):
    batch_size: int = 50
    max_patch_lines: int = 500
    min_golden_examples: int = 200
    parallel_workers: int = 4
    gcs_bucket: str = ""
    wandb_project: str = "swe-qwen-data"
    manifest_path: Path = Path("repos/manifest.json")
    output_dir: Path = Path("data/")
    golden_source_split: str = "test"  # "test" default; "all" requires opt-in with leakage warning
    max_issues_per_repo: int = 2000
    issue_labels_to_include: list[str] = ["bug", "defect", "fix"]
    test_directories: list[str] = ["tests/", "test/"]
    max_events_per_issue: int = 100  # timeline event cap

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATA_PIPELINE_")
```

**Behavior:**
- Layered resolution: CLI args > env vars > .env file > defaults
- Startup validation: `GITHUB_TOKEN`, `WANDB_API_KEY`, GCP ADC all checked on load
- All filter thresholds are configurable via env overrides

### 1. `data_engineering/schema.py`
**Purpose:** Core Pydantic models for all data structures.

**Models:**
```python
class ParsedHunk(BaseModel):
    file: str
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    diff_lines: list[str]

class TestResults(BaseModel):
    passed: list[str]
    failed: list[str]
    errored: list[str]

class IssueRecord(BaseModel):
    issue_id: str
    repo: str
    issue_body: str
    patch_diff: str              # raw unified diff
    parsed_hunks: list[ParsedHunk]
    test_results: TestResults
    pr_title: str
    pr_description: str
    commit_messages: list[str]
    files_changed: list[str]
    test_files_changed: list[str]
    issue_labels: list[str]
    repo_domain: str
    metadata: dict[str, Any] = {}  # timestamps, urls, etc.

    @field_validator("patch_diff")
    def validate_patch(cls, v):
        # Must parse as unified diff
        parse_unified_diff(v)  # raises if invalid
        return v
```

### 2. `data_engineering/ingest.py`
**Purpose:** GitHub API ingestion per repo.

**Key Functions:**
```python
def fetch_issues_for_repo(repo_config: IngestionConfig, gh: Github, max_issues: int) -> list[RawIssue]:
    # Paginate issues with labels, return raw issue data

def fetch_linked_prs(issue: RawIssue, gh: Github) -> list[RawPR]:
    # Get timeline events → find cross-referenced merged PRs

def fetch_pr_details(pr: RawPR, gh: Github) -> PRDetails:
    # Fetch diff, files, commits, test file detection

def ingest_repo(repo: Repository, config: DataPipelineConfig) -> list[IssueRecord]:
    # Orchestrate full ingestion for one repo
```

**Implementation Notes:**
- Use `gh.get_repo(f"{owner}/{name}").get_issues(state="all", labels=config.issue_labels_to_include)`
- Timeline: `issue.get_timeline()` → filter `event == "cross-referenced"` and `source.issue.pull_request` and `merged == True`
- Timeline pagination: set `max_events_per_issue` cap (default 100) to prevent runaway on issues with many cross-references; PyGithub paginates by default
- PR diff: `pr.get_files()` for files + `pr.get_commits()` for messages
- Test file detection: match against `manifest.ingestion_config.test_directories` (tests/, test/)
- Rate limit: exponential backoff decorator on all GitHub API calls
- Proactive throttle: check `X-RateLimit-Remaining` header after each API call; if < 100, inject delay proportional to `X-RateLimit-Reset` to pre-empt 429s

### 3. `data_engineering/validate.py`
**Purpose:** Schema validation with detailed error logging.

**Key Functions:**
```python
def validate_record(record: dict) -> ValidationResult:
    # Pydantic validation + custom validators
    # Returns ValidationResult(valid=bool, record=IssueRecord|None, errors=list[ValidationError])

def validate_batch(records: list[dict]) -> tuple[list[IssueRecord], list[ValidationError]]:
    # Process batch, separate valid/invalid

class ValidationError(BaseModel):
    record_id: str
    field: str
    error: str
    raw_value: Any
```

**Custom Validators:**
- `patch_diff`: parse as unified diff (use `unidiff` library)
- `test_results`: must have passed/failed/errored as lists
- `issue_body`: non-empty string
- `files_changed`, `test_files_changed`, `commit_messages`, `issue_labels`: list[str]

### 4. `data_engineering/clean.py`
**Purpose:** Deduplication and filtering.

**Key Functions:**
```python
def deduplicate(records: list[IssueRecord]) -> tuple[list[IssueRecord], DedupStats]:
    # Primary: hash by (repo, issue_id, pr_number)
    # Secondary: patch content hash
    # Returns deduplicated list + stats

def clean_records(records: list[IssueRecord], config: DataPipelineConfig) -> tuple[list[IssueRecord], CleanStats]:
    # Filter: no F2P signal, patch > max_patch_lines, binary files, non-.py (>50%), empty issue_body
    # Returns cleaned list + stats
```

**Filter Logic:**
- F2P signal: `test_files_changed` non-empty AND F2P keywords in commit messages/PR description (matches Section 5 filter table)
- Patch size: count lines in `patch_diff` (split by `\n`)
- Binary: check `ParsedHunk.diff_lines` for binary markers
- Non-Python: >50% of files in `files_changed` must end with `.py` (WARN if any non-.py)
- Empty issue body: `issue_body` is empty or whitespace

### 5. `data_engineering/split.py`
**Purpose:** Repo-stratified train/val/test splits + golden subset.

**Key Functions:**
```python
def stratified_split(records: list[IssueRecord], ratios: SplitRatios) -> Splits:
    # Group by repo, assign each repo to train/val/test (80/10/10)
    # Ensure no repo appears in multiple splits

def extract_golden(records: list[IssueRecord]) -> list[IssueRecord]:
    # V1 Proxy: test_files_changed non-empty AND F2P keywords in commit messages/PR description
    # (test_results.failed is final state only — cannot represent "was failing")

class Splits(BaseModel):
    train: list[IssueRecord]
    val: list[IssueRecord]
    test: list[IssueRecord]
    golden: list[IssueRecord]
```

**Note on Golden Set:** The `test_results` in IssueRecord represents the *final* state (after fix). `test_results.failed > 0` means tests are *still failing*, not "were failing before." The V1 F2P proxy (Section 6.2) uses `test_files_changed` presence + F2P keywords in commit messages/PR description. Phase 5 will refine with actual test execution at base vs head SHAs for ground truth.

### 6. `data_engineering/golden.py`
**Purpose:** Golden eval subset extraction and verification.

**Key Functions:**
```python
def build_golden_set(splits: Splits, min_size: int) -> GoldenSet:
    # Extract from splits.test (or all splits) where F2P verified
    # If < min_size, log warning

class GoldenSet(BaseModel):
    records: list[IssueRecord]
    f2p_verified_count: int
    source_split: str  # "test" or "all"
```

### 7. `data_engineering/version.py`
**Purpose:** W&B dataset versioning and artifact management.

**Key Functions:**
```python
def log_dataset_artifacts(run_id: str, stages: dict[str, list[IssueRecord]], config: DataPipelineConfig, manifest_hash: str) -> dict[str, wandb.Artifact]:
    # Create artifact per stage: raw, validated, cleaned, train, val, test, golden
    # Metadata: manifest_hash, repo_list, split_ratios, counts, golden_size, validation_stats
    # Auto-tag: "latest"

def log_validation_errors(run_id: str, errors: list[ValidationError]) -> wandb.Artifact:
    # Log validation_errors.jsonl artifact
```

### 8. `data_engineering/archive.py`
**Purpose:** GCS upload for durable storage.

**Key Functions:**
```python
def upload_to_gcs(run_id: str, stages: dict[str, list[IssueRecord]], manifest: Manifest, dataset_card: str, config: DataPipelineConfig) -> dict[str, str]:
    # Upload each stage as JSONL to gs://{bucket}/datasets/{run_id}/{stage}.jsonl
    # Upload manifest.json and dataset_card.md
    # Return dict of gcs_paths
```

### 9. `data_engineering/card.py`
**Purpose:** Auto-generate dataset card.

**Key Functions:**
```python
def generate_dataset_card(manifest: Manifest, stats: PipelineStats, run_id: str, git_sha: str) -> str:
    # Returns markdown string with:
    # - Total examples, schema fields
    # - Source repos table (name, count, domain)
    # - Validation: pass/fail/dedup/filter counts
    # - Split ratios & counts
    # - Golden set size & F2P stats
    # - Generation date, pipeline version (git SHA)
    # - W&B artifact links
```

### 10. `data_engineering/run_pipeline.py`
**Purpose:** Orchestrator — runs full pipeline per repo, handles checkpoints, parallelism.

**Key Functions:**
```python
def run_pipeline_for_repo(repo: Repository, config: DataPipelineConfig, run_id: str, resume_from: str | None) -> RepoResult:
    # Load checkpoint if resume_from
    # Stage 1: ingest → checkpoint raw.jsonl
    # Stage 2: validate → checkpoint validated.jsonl + validation_errors.jsonl
    # Stage 3: clean → checkpoint cleaned.jsonl
    # Stage 4: split → checkpoint splits/
    # Stage 5: golden → checkpoint golden.jsonl
    # Return RepoResult with stats

def run_pipeline(config: DataPipelineConfig) -> PipelineResult:
    # Load manifest
    # Generate run_id (UUID)
    # Parallel: ThreadPoolExecutor(config.parallel_workers) over repos
    # Aggregate results, create final splits (merge per-repo splits)
    # Build golden set
    # Log to W&B, upload to GCS, generate card
```

**Checkpoint Format:**
- Each stage writes `{stage}.jsonl` to `output_dir/{run_id}/{repo_id}/`
- `--resume-from validated` skips ingest, loads `validated.jsonl`

### 11. `data_engineering/cli.py`
**Purpose:** Typer CLI entry point.

```python
@app.command()
def run(
    manifest: Path = typer.Option("repos/manifest.json", "--manifest", "-m"),
    output_dir: Path = typer.Option("data/", "--output-dir", "-o"),
    run_id: str | None = typer.Option(None, "--run-id"),
    stages: str = typer.Option("all", "--stages"),  # comma-separated
    parallel: int = typer.Option(4, "--parallel", "-p"),
    resume_from: str | None = typer.Option(None, "--resume-from"),
):
    # Build config, run pipeline, print summary
```

---

## 7. Implementation Tasks

### Task 3.0: Environment & Config Setup
- **Action:** Create `data_engineering/` package structure, `config.py` with `DataPipelineConfig`, verify deps
- **Details:**
  - Package: `data_engineering/__init__.py` + 11 modules
  - Config: Pydantic Settings class with all tunable params (CLI > env > .env > defaults)
  - Fields: `batch_size`, `max_patch_lines`, `min_golden_examples`, `parallel_workers`, `gcs_bucket`, `wandb_project`, `manifest_path`, `output_dir`, `golden_source_split` (default: `"test"`)
  - Auth: `GITHUB_TOKEN`, `WANDB_API_KEY`, `GCP_CREDENTIALS` validated at startup
- **Owner:** Human
- **Estimate:** 1 hour

### Task 3.1: Schema Module (`schema.py`)
- **File:** `data_engineering/schema.py`
- **Models:** `ParsedHunk`, `TestResults`, `IssueRecord` (with validators), `ValidationError`, `ValidationResult`, `DedupStats`, `CleanStats`, `SplitRatios`, `Splits`, `GoldenSet`, `RepoResult`, `PipelineResult`, `PipelineStats`
- **Validators:**
  - `patch_diff`: parse as unified diff (use `unidiff` library)
  - `test_results`: must have `passed`, `failed`, `errored` as `list[str]`
  - `issue_body`: non-empty string
  - List fields: must be `list[str]`
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.2: Ingestion Module (`ingest.py`)
- **File:** `data_engineering/ingest.py`
- **Functions:**
  - `fetch_issues_for_repo(repo_config, gh, max_issues) → list[RawIssue]`
  - `fetch_linked_prs(issue, gh) → list[RawPR]`
  - `fetch_pr_details(pr, gh) → PRDetails`
  - `ingest_repo(repo, config) → list[IssueRecord]`
- **Implementation:**
  - Paginate issues (100/page) with label filter from `issue_labels_to_include`
  - Timeline events → cross-referenced merged PRs
  - PR files → patches + file list; PR commits → messages
  - Test file detection: match `files_changed` against `test_directories` globs
  - Rate limit: exponential backoff decorator on all GitHub calls
  - Batch 50 issues, `ThreadPoolExecutor(max_workers=4)`
- **Output:** Raw `IssueRecord` dicts (not yet validated)
- **Owner:** Human
- **Estimate:** 4 hours

### Task 3.3: Validation Module (`validate.py`)
- **File:** `data_engineering/validate.py`
- **Functions:**
  - `validate_record(record: dict) → ValidationResult`
  - `validate_batch(records: list[dict]) → tuple[list[IssueRecord], list[ValidationError]]`
- **Models:** `ValidationError(record_id, field, error, raw_value)`, `ValidationResult(valid, record, errors)`
- **Behavior:** Strict Pydantic validation + custom validators; collect ALL errors per record; continue on failure
- **Output:** Validated `IssueRecord` objects + `ValidationError` list for artifact logging
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.4: Cleaning Module (`clean.py`)
- **File:** `data_engineering/clean.py`
- **Functions:**
  - `deduplicate(records) → tuple[list[IssueRecord], DedupStats]`
  - `clean_records(records, config) → tuple[list[IssueRecord], CleanStats]`
- **Dedup:**
  - Primary: hash `(repo, issue_id, pr_number)` — exact same fix for same issue
  - Secondary: SHA256 of `patch_diff` content — catch same fix applied elsewhere
  - Stats: `total_input`, `exact_duplicates_removed`, `content_duplicates_removed`, `unique_output`
- **Filters (all configurable via config):**
  - No test file changes (per `test_directories`)
  - Patch lines > `max_patch_lines` (default 500)
  - Binary file in diff
  - Non-.py file in `files_changed` (REMOVE if >50% non-.py; WARN if any non-.py)
  - Empty `issue_body`
  - No F2P signal: `test_files_changed` empty OR no F2P keywords (`"fix"`, `"close"`, `"resolve"`) in commit messages/PR description
- **Zero-output warning:** If a repo yields 0 records after cleaning, log WARNING with repo name and filter breakdown — continue to next repo (don't fail pipeline)
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.5: Splitting Module (`split.py`)
- **File:** `data_engineering/split.py`
- **Functions:**
  - `stratified_split(records, ratios) → Splits`
  - `extract_golden(records, min_size) → GoldenSet`
- **Stratification:**
  - Group records by `repo`
  - Shuffle repos (seeded: `run_id` hash)
  - Assign repos to train/val/test by ratio (80/10/10)
  - Verify: no repo appears in multiple splits
- **Golden:**
  - Filter: `test_files_changed` non-empty AND F2P keywords in commit messages/PR description (V1 proxy)
  - If count < `min_golden_examples`: log warning, use all available
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.6: Golden Module (`golden.py`)
- **File:** `data_engineering/golden.py`
- **Functions:** `build_golden_set(splits, min_size) → GoldenSet`
- **Purpose:** Extract and verify golden eval subset; compute F2P stats
- **Owner:** Human
- **Estimate:** 1 hour

### Task 3.7: Versioning Module (`version.py`)
- **File:** `data_engineering/version.py`
- **Functions:**
  - `log_dataset_artifacts(run_id, stages, config, manifest_hash) → dict[str, wandb.Artifact]`
  - `log_validation_errors(run_id, errors) → wandb.Artifact`
- **Artifacts per stage:** `raw`, `validated`, `cleaned`, `train`, `val`, `test`, `golden`, `validation_errors`
- **Metadata per artifact:**
  - `manifest_hash`, `repo_list`, `split_ratios`, `counts`, `golden_size`, `validation_pass`, `validation_fail`, `dedup_exact`, `dedup_content`, `filter_counts`
  - Auto-tag: `"latest"`
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.8: Archival Module (`archive.py`)
- **File:** `data_engineering/archive.py`
- **Functions:**
  - `upload_to_gcs(run_id, stages, manifest, dataset_card, config) → dict[str, str]`
- **GCS Structure:** `gs://{bucket}/datasets/{run_id}/{stage}.jsonl` + `manifest.json` + `dataset_card.md`
- **Auth:** GCP ADC (from Phase 1 Terraform)
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.9: Dataset Card Module (`card.py`)
- **File:** `data_engineering/card.py`
- **Function:** `generate_dataset_card(manifest, stats, run_id, git_sha) → str`
- **Card Sections:**
  - Dataset overview (size, schema fields)
  - Source repos table (name, domain, count, stars)
  - Quality stats (validation pass/fail, dedup counts, filter counts per type)
  - Split ratios & counts
  - Golden set size & F2P verification stats
  - Generation metadata (date, git SHA, pipeline version)
  - W&B artifact links
- **Owner:** Human
- **Estimate:** 1.5 hours

### Task 3.10: Orchestrator (`run_pipeline.py`)
- **File:** `data_engineering/run_pipeline.py`
- **Functions:**
  - `run_pipeline_for_repo(repo, config, run_id, resume_from) → RepoResult`
  - `run_pipeline(config) → PipelineResult`
- **Checkpointing:**
  - Each stage writes `{stage}.jsonl` to `output_dir/{run_id}/{repo_id}/`
  - `--resume-from validated` loads `validated.jsonl`, skips ingest
  - Failed repo: log error, continue others
- **Parallelism:** `ThreadPoolExecutor(config.parallel_workers)` over repos
- **Aggregation:** Merge per-repo splits → final train/val/test/golden
- **Progress:** Rich progress bars per repo per stage; emit metrics dict to W&B per stage
- **Owner:** Human
- **Estimate:** 4 hours

### Task 3.11: CLI (`cli.py`)
- **File:** `data_engineering/cli.py`
- **Command:** `python -m data_engineering.run_pipeline [options]`
- **Options:**
  - `--manifest PATH` (default: `repos/manifest.json`)
  - `--output-dir PATH` (default: `data/`)
  - `--run-id STR` (default: auto UUID)
  - `--stages STR` (comma-separated: `all`, `raw`, `validated`, `cleaned`, `train`, `val`, `test`, `golden`; default: `all`)
  - `--parallel INT` (default: 4)
  - `--resume-from STR` (stage name to resume from)
- **Behavior:** Load config, run orchestrator, print summary table
- **Owner:** Human
- **Estimate:** 1.5 hours

### Task 3.12: Unit Tests
- **File:** `tests/test_data_*.py` (one per module)
- **Strategy:** Mock `pygithub`, `wandb`, `google-cloud-storage`; use `pytest-mock`
- **Fixtures:** `tests/fixtures/sample_issues.json` (5-10 real pairs), `mock_github_responses/`, `expected_outputs/`
- **Coverage:** Each validator, each filter, split logic, dedup logic
- **Owner:** Human
- **Estimate:** 3 hours

### Task 3.13: Integration Test
- **File:** `tests/test_data_integration.py`
- **Marker:** `@pytest.mark.integration`
- **Run:** Full pipeline on 1-2 small test repos (from fixtures or real API with `@pytest.mark.requires_github_token`)
- **Verify:** Raw → validated → cleaned → split → golden → W&B artifacts → GCS upload
- **Checkpoint test:** `--resume-from validated` works
- **Owner:** Human
- **Estimate:** 2 hours

### Task 3.14: Property-Based Tests
- **File:** `tests/test_data_property.py`
- **Marker:** `@pytest.mark.hypothesis`
- **Tests:**
  - Schema: random dicts → accept/reject correctly
  - Dedup: idempotent, commutative
  - Filters: commutative (order doesn't matter)
  - Split: no repo in multiple splits, ratios ≈ 80/10/10
- **Owner:** Human
- **Estimate:** 1.5 hours

### Task 3.15: PoC Run (1-2 Repos)
- **Action:** `python -m data_engineering.run_pipeline --manifest repos/manifest.json --stages all --parallel 2`
- **Verify:** All stages complete, artifacts in W&B, files in GCS, dataset card generated
- **Owner:** Human
- **Estimate:** 30 min (plus compute time)

### Task 3.16: Full Pipeline Run (All 10 Repos)
- **Action:** Run with all repos, `--parallel 4`
- **Verify:** Acceptance criteria met
- **Owner:** Human
- **Estimate:** 2-4 hours (depends on API latency)

---

## 8. Output Schemas

### 8.1 IssueRecord (JSONL line)
```json
{
  "issue_id": "12345",
  "repo": "owner/repo",
  "issue_body": "Bug description...",
  "patch_diff": "@@ -1,3 +1,4 @@\n def foo():\n+    return 1\n     return 0",
  "parsed_hunks": [
    {"file": "foo.py", "old_start": 1, "old_lines": 3, "new_start": 1, "new_lines": 4, "diff_lines": [" def foo():", "+    return 1", "     return 0"]}
  ],
  "test_results": {"passed": ["test_foo"], "failed": ["test_bar"], "errored": []},
  "pr_title": "Fix foo return value",
  "pr_description": "This PR fixes...",
  "commit_messages": ["fix: return 1 in foo", "test: add test_bar"],
  "files_changed": ["foo.py"],
  "test_files_changed": ["tests/test_foo.py"],
  "issue_labels": ["bug"],
  "repo_domain": "web-api",
  "metadata": {"issue_url": "...", "pr_url": "...", "merged_at": "...", "base_sha": "...", "head_sha": "..."}
}
```

### 8.2 ValidationError (JSONL line)
```json
{
  "record_id": "owner/repo#12345",
  "field": "patch_diff",
  "error": "Invalid unified diff: missing @@ header",
  "raw_value": "not a diff"
}
```

### 8.3 Manifest.json (Phase 2 output — reference)
See `docs/PHASE-2-REPOSITORY-CURATION.md` section 6 for full schema. Key fields Phase 3 uses:
- `repositories[].ingestion_config` (all fields)
- `repositories[].test_command_actual`
- `repositories[].install_command`
- `repositories[].install_success`

---

## 9. Field Notes (Phase 3 Specific)

| Field | Purpose | Semantics | Used By |
|-------|---------|-----------|---------|
| `parsed_hunks` | Structured diff for context-aware training | Each hunk: file, line ranges, diff lines. Enables hunk-level masking/attention. | Phase 4 training, Phase 6 inference |
| `test_files_changed` | Test file targeting | Subset of `files_changed` matching `test_directories` globs. | Phase 5 eval (run only changed tests) |
| `test_results.failed` | Final test state | Tests failing after fix (means fix is incomplete). V1 F2P proxy uses `test_files_changed` + keywords instead. Phase 5 runs actual before/after test execution at base/head SHAs. | Phase 5 eval (before/after comparison) |
| `metadata.base_sha` / `head_sha` | Reproducible test execution | Phase 5 checks out these SHAs to run tests before/after fix. | Phase 5 `test_runner.py` |
| `repo_domain` | Domain conditioning | From manifest `domain_category`. Enables domain-aware sampling. | Phase 4 training, Phase 6 routing |

---

## 10. File Structure After Phase 3

```
swe-qwen/
├── data_engineering/
│   ├── __init__.py
│   ├── schema.py
│   ├── config.py
│   ├── ingest.py
│   ├── validate.py
│   ├── clean.py
│   ├── split.py
│   ├── golden.py
│   ├── version.py
│   ├── archive.py
│   ├── card.py
│   ├── run_pipeline.py
│   └── cli.py
├── tests/
│   ├── test_data_schema.py
│   ├── test_data_ingest.py
│   ├── test_data_validate.py
│   ├── test_data_clean.py
│   ├── test_data_split.py
│   ├── test_data_golden.py
│   ├── test_data_version.py
│   ├── test_data_archive.py
│   ├── test_data_card.py
│   ├── test_data_integration.py
│   └── test_data_property.py
└── tests/fixtures/
    ├── sample_issues.json
    ├── mock_github_responses/
    │   ├── issues_page1.json
    │   ├── timeline_events.json
    │   ├── pr_files.json
    │   └── pr_commits.json
    └── expected_outputs/
        ├── validated.jsonl
        ├── cleaned.jsonl
        ├── train.jsonl
        ├── val.jsonl
        ├── test.jsonl
        └── golden.jsonl
```

---

## 11. Task Dependencies & Ordering

```
3.0 → 3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6
                      │         │
                      │         └→ 3.7 → 3.8 → 3.9
                      │
                      └→ 3.10 → 3.11
                                 │
                                 ├→ 3.12 (unit)
                                 ├→ 3.13 (integration)
                                 ├→ 3.14 (property)
                                 │
                                 ├→ 3.15 (PoC 1-2 repos)
                                 │
                                 └→ 3.16 (full 10 repos)
```

- **3.0** prerequisites: project venv, `GITHUB_TOKEN`, `WANDB_API_KEY`, GCP ADC
- **3.1** (schema) must exist before 3.2-3.6 consume it
- **3.2** (ingest) and **3.3** (validate) can be developed in parallel after 3.1
- **3.4** (clean) needs 3.3 output; **3.5** (split) needs 3.4 output
- **3.7-3.9** (version/archive/card) need all stage outputs
- **3.10** (orchestrator) wires everything; **3.11** (CLI) wraps it
- **3.12-3.14** (tests) can run after respective modules exist
- **3.15** (PoC) validates end-to-end before **3.16** (full run)

---

## 12. Acceptance Criteria (Phase Exit Gate)

Phase 3 is **complete** when ALL are true:

- [ ] `data_engineering/` package exists with all 11 modules + CLI
- [ ] `python -m data_engineering.run_pipeline --manifest repos/manifest.json` runs without error
- [ ] W&B project shows versioned artifacts: `raw`, `validated`, `cleaned`, `train`, `val`, `test`, `golden`, `validation_errors` (each with `latest` tag)
- [ ] Each artifact metadata includes: `manifest_hash`, `repo_list`, `split_ratios`, `counts`, `golden_size`, `validation_pass`, `validation_fail`, `dedup_exact`, `dedup_content`, `filter_counts`
- [ ] GCS bucket contains `datasets/{run_id}/{stage}.jsonl` for all stages + `manifest.json` + `dataset_card.md`
- [ ] Dataset card includes: size, schema, source repos table, quality stats, split ratios/counts, golden set size/F2P stats, git SHA, W&B links
- [ ] Golden eval subset: all records pass schema validation + test-verification (F2P heuristic) + dedup checks (zero duplicates)
- [ ] JSONL splits (`train.jsonl`, `val.jsonl`, `test.jsonl`, `golden.jsonl`) exist and are readable
- [ ] `tests/test_data_*.py` passes (all unit, integration, property tests)
- [ ] Checkpoint resume works: `--resume-from validated` skips ingest, loads `validated.jsonl`
- [ ] Per-repo progress bars emit metrics to W&B per stage
- [ ] No repo appears in more than one of train/val/test splits
- [ ] All 10 manifest repos processed (or failures logged with reason)
- [ ] Repo yielding zero records after cleaning: logged as warning with repo name, not treated as pipeline failure

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GitHub API rate limit during full run | Medium | High | Exponential backoff + conditional requests + batch 50/4 workers; monitor rate limit remaining |
| Golden set < 200 examples | Medium | Medium | Log warning, proceed with available; expand repo pool in Phase 2 if needed |
| OOM during large repo processing | Low | High | Batch processing, stream from API, disk-backed checkpoints (JSONL per stage) |
| Patch parsing failures on weird diffs | Medium | Low | Robust parser (try `unidiff`, fallback custom); log failures, continue |
| Test result inference (before/after) inaccurate | High | Medium | V1 uses `test_files_changed` + F2P keywords as proxy; Phase 5 runs actual tests at base/head SHAs for ground truth |
| W&B/GCS auth failures in CI | Low | High | Phase 1 Terraform sets up ADC; validate creds at pipeline start; clear error messages |
| Manifest `ingestion_config` drift | Low | Medium | `build_manifest.py` validates; `run_pipeline` validates config on load; CI catches drift |
| Non-deterministic splits | None | High | Seeded shuffle (hash of `run_id`); same `run_id` = same splits |
| Temp disk space exhausted | Low | Low | Each repo cleaned after processing; max ~500MB at once; checkpoints are JSONL (streaming) |

---

## 14. Definition of Done

1. All 17 tasks completed with deliverables in repository
2. All 13 acceptance criteria verified (checkboxes above)
3. `pytest tests/test_data_*.py` passes (≥ 25 test cases across unit/integration/property)
4. `repos/manifest.json` is the **only** input required — no manual handoff
5. W&B artifacts show complete lineage: manifest → raw → validated → cleaned → split → versioned
6. GCS archive is queryable and complete
7. Dataset card is human-readable and comprehensive
8. No hardcoded assumptions (branches, labels, paths) — all from manifest

---

## 15. Next Phase Dependency

**Phase 4 (Fine-Tuning)** consumes the JSONL splits directly:

1. Reads `train.jsonl`, `val.jsonl` from GCS (via W&B artifact path) or local `data/{run_id}/`
2. Schema matches `IssueRecord` exactly — no conversion needed
3. Uses `repo_domain` for domain-aware sampling/curriculum
4. Uses `parsed_hunks` for hunk-level training objectives
5. Phase 5 evaluation uses `test.jsonl` + `golden.jsonl` with `metadata.base_sha`/`head_sha` for before/after test execution

No additional coordination needed — the dataset artifacts are the complete contract.
