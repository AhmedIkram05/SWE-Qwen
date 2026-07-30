# Implementation Log Tracker

**Purpose:** Single source of truth for **major/medium deviations** from the Master Plan during phase implementation. Updated in real-time. Do not log routine task completions — only changes that alter scope, architecture, timeline, or introduce risk.

**Threshold for Logging:**
- **Major:** Scope change, architecture change, timeline slip >1 day, new dependency, blocker requiring workaround
- **Medium:** Config/parameter changes, tool/library swap, partial task completion with follow-up needed
- **Do NOT log:** Task completed as planned, bug fixes, minor typo fixes, formatting, routine test passes

---

## Log Format

Each Phase follows this structure:

```markdown
## Phase N: [Phase Name] — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| N.x | ... | ... | ... | Low/Med/High |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| ... | ... | ... | ... |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| ... | ... | ... | ... | ... |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| ... | ... | ... |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| ... | ... | ... |

### Metrics / Observations

- Key metric observed: ...
- Unexpected behavior: ...
- Performance note: ...
```

---

## Phase 1: Foundation & Scaffolding — 2026-07-25

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 1.1 | Repo init | Completed | Repo already existed | Low |
| 1.2 | pyproject.toml | Completed | Full config with all deps + tooling | Low |
| 1.3 | .gitignore | Completed | Added ML/Modal/Terraform patterns | Low |
| 1.4 | Terraform scaffold | Completed | Full module structure (storage, iam) | Low |
| 1.5 | Modal config | Completed | modal_app.py with train/serve functions | Low |
| 1.6 | W&B project | Completed | init_wandb.py with sweep + registries | Low |
| 1.7 | GitHub Actions skeleton | Completed | ci.yml with lint/test/terraform/modal/docker | Low |
| 1.8 | README | Completed | Architecture, quick-start, structure | Low |
| 1.9 | Directory structure | Completed | All dirs created per MASTER-PLAN | Low |
| 1.10 | Dockerfile | Not needed | Modal Image replaces Docker entirely | Low |
| 1.11 | Pre-commit config | Completed | Created .pre-commit-config.yaml | Low |
| 1.12 | Makefile / justfile | Deferred | Using pytest/ruff/mypy directly | Low |
| 1.13 | Verify CI runs | Pending | Requires GCP/Modal secrets | Medium |
| 1.14 | Version audit | Completed | All deps bumped to latest stable (Jul 2026) | Low |
| 1.15 | Credential docs | Completed | README updated with explicit requirements | Low |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| Use Modal for training instead of self-hosted GPU | Cost efficiency, scale-to-zero | GCP Vertex AI, AWS SageMaker | Modal provides H100s, simple Python API, integrated volumes |
| Terraform modules for storage + IAM | Clean separation, reusability | Single root module | Modules enable env-specific configs, easier testing |
| Workload Identity Federation for GitHub Actions | Security best practice | Long-lived SA keys | No secret rotation, OIDC tokens short-lived |
| Skip Dockerfile, use Modal images | Simpler dev loop | Multi-stage Dockerfile | Modal handles image building, GPU base images optimized |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| Terraform WIF provider missing `oidc` block | 2026-07-25 | 2026-07-25 | Added `oidc { issuer_uri = "https://token.actions.githubusercontent.com" }` | 10 min |
| Storage module referencing IAM module's service account | 2026-07-25 | 2026-07-25 | Moved bucket IAM bindings to IAM module, pass bucket names as outputs | 20 min |
| Test infrastructure outputs require terraform apply | 2026-07-25 | 2026-07-25 | Marked integration tests with `@pytest.mark.integration`, unit tests validate structure only | 15 min |
| Dockerfile not needed | Modal handles all containerization | Multi-stage Dockerfile for Artifact Registry | Modal Image + volumes + build caching replace Docker entirely. CI docker-build job is optional |


### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| Terraform | GCS backend bucket `swe-qwen-terraform-state` must exist before init | Pre-create bucket or use local backend for first run |
| Modal | Volumes `swe-qwen-datasets` and `swe-qwen-models` created on first deploy | Persist across function invocations |
| W&B | Sweep config uses Bayesian optimization with Hyperband early stopping | Reduces compute for HPO |
| CI | Terraform plan runs on PRs, apply only on main merge | Prevents accidental prod changes |
| transformers 5.x | Breaking changes from 4.x | Verify training code works with v5 before Phase 4 |
| accelerate 1.x | New distributed training API | SFTTrainer may need updated accelerate config |
| trl 1.x | SFTTrainer API changed | Migration guide needed before Phase 4 |
| datasets 5.x | Dataset format API changed | Data pipeline in Phase 3 needs v5-compatible code |
| modal 1.x | automounting removed, new Image API | modal_app.py uses 1.x-compatible APIs (Image, Secret, Volume, Retries) |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| Dockerfile | Removed | Modal handles containerization |
| Makefile/justfile | Removed | Direct tool invocation is simpler |
| Docker build job in CI | Kept | For Artifact Registry deployment option |

### Metrics / Observations

- 31 scaffold tests passing 13 skipped
- Terraform validate + fmt check passing
- Infrastructure graph validates with 13 required resources
- Phase 1 complete
- Version audit complete — all packages bumped to Jul 2026 stable releases

---

## Phase 2: Repository Curation & Verification — 2026-07-26

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 2.1 | Define selection criteria | Completed | docs/planning/phase2-criteria.md | Low |
| 2.2 | Identify 10 candidate repos | 50 candidates found, 10 selected | GitHub Search API: topic-only queries need text term. Per-subtopic queries with `topic:+text` work. 5 subtopics used (web-api, cli, data-ml, testing, utils), MAX_PER_QUERY=15 | Medium |
| 2.3 | Verify test suites run | Deep verify on 15 shortlist, 12 passed | Clone+install+pytest on all 50 too heavy. Picked 15 promising candidates based on API checks, then deep-verified with `verify_repos.py` | Medium |
| 2.4 | Extract issue-PR pairs | Deferred to Phase 3 | Phase 2 is curation + manifest; ingestion is Phase 3 | Low |
| 2.5 | Build manifest.json | Completed with star/description enrichment | `scripts/build_manifest.py` merges verification + enrichment | Low |
| 2.6 | Write verification scripts | 2 scripts: find_candidates.py + verify_repos.py | Replaced single script with separate sourcing and verification | Low |
| 2.7 | Document selection rationale | repos/README.md + criteria doc | Selection rationale documented | Low |
| 2.8 | (added) Expand to 14 repos | Added pytest, black, pydantic, mlflow | User: "10 was an aim not a cap". Added 4 to fill testing/utils/data-ml gaps | Low |
| 2.9 | Test architecture redesign | Replaced `check_tests_run` (heavy pytest-in-verify) with lightweight `check_build_readiness` (pytest config + pip install + import check) + per-repo CI matrix job (`verify-repos-tests`) | Host-venv contamination made in-script pytest unreliable. Isolated venv per repo was too slow (14 venvs × 30s+ each). CI matrix isolates each repo in its own runner — cleaner, faster, matches how CI should work | Medium |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| Per-subtopic GitHub search queries | GitHub API requires text search term alongside `topic:` qualifier | Single broad query, OR between topics | Topic-only queries return 0 results. Per-subtopic with text term works |
| Deep-verify only shortlist of 15 | 50 candidates would take hours to clone/install/test | Full 50 verification | 15 selected by license+stars+py file count skim; deep verify on those |
| Dropped graphify + sherlock post-verify | Both passed hard checks but too small (99 and 8 .py files) | Keep them despite size | Minimum size filter ensures sufficient data for Phase 3 ingestion |
| size_range adjusted to 50-5000 (soft) | Several quality repos have <500 .py files (rich=146, datasets=148) | Hard floor at 500 | Size is soft check; repo quality outweighs arbitrary size threshold |
| Added `ingestion_config` to manifest | Phase 3 needs per-repo config (branch, labels, paths) | Store in separate config file | Self-contained manifest simplifies pipeline |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| GitHub Search API returns 0 results with `topic:` qualifier alone | 2026-07-26 | 2026-07-26 | Must include a text search term alongside qualifier. Per-subtopic queries with MAX_PER_QUERY=15 | 30 min |
| PyGithub `repo.license` is a property, not callable | 2026-07-26 | 2026-07-26 | Changed `repo.license()` → `repo.license` in verify_repos.py | 5 min |
| `_allows_python_310()` too strict (only handled `>=3.10` format) | 2026-07-26 | 2026-07-26 | Rewrote to handle `^3.9`, `>=3.8`, `>=3.10.0`, `~=3.10`, poetry/pip constraints | 15 min |
| PyGithub `get_commits(since=string)` expects datetime object | 2026-07-26 | 2026-07-26 | Pass `datetime.datetime` not ISO string | 5 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| GitHub Search API | Queries need `q=python+topic:web-api` format, not `q=topic:web-api` alone | Prevents wasted queries in Phase 3 ingestion |
| PyGithub Auth | Use `github.Auth.Token(token)`, pass to `github.Github(auth=...)`. Raw token string deprecated | Prevents auth failures |
| Rate limiting | Search API = 30/non-search = 5000/hr. `rl.rate.remaining` for core, search endpoint separate | Plan Phase 3 with backoff |
| Python version parsing | Constraint formats vary: `>=3.10`, `^3.9`, `>=3.10.0`, `~=3.10`, `>=3.9,<4.0`, or absent entirely | Version check function handles all formats loosely |
| manifest.json ingestion_config | Each repo has default_branch, labels, exclude_paths, test_dirs | Phase 3 reads these directly; no separate config needed |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| Star count enrichment | Added to build_manifest.py | README needs star counts for selection rationale |
| 3 scripts instead of 1 | Modified | find_candidates.py (sourcing) + verify_repos.py (deep verify) + build_manifest.py (manifest assembly) |

### Metrics / Observations

- 50 candidates sourced from GitHub (15 web-api, 12 cli, 14 utils, 7 data-ml, 2 testing)
- 15 shortlisted, 12 passed deep verification (3 failed: pandas>=3.11, strix>=3.12, SWE-agent>=3.11)
- 14 final repos (expanded from 10 upon request — not a cap)
- 493,511 total stars across all repos
- 10,225 total Python files
- Added 4 repos post-selection: pytest (testing), black (utils), pydantic (utils), mlflow (data-ml)
- Testing domain doubled from 1→2, utils doubled from 2→4, data-ml grew from 2→3
- `pip install -e .` succeeded on all 12/12 verified repos
- tox/nox detected in 3 repos (rich, wagtail, faker) — not a blocker
- External service imports (openai, boto3, anthropic, redis) detected in 2 repos (headroom, marimo) — soft fail only
- GitHub API rate limit: ~4,800 remaining after Phase 2 ops

---

## Phase 3: Data Engineering Pipeline — 2026-07-27 to 2026-07-28 ✅ COMPLETED

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 3.1 | config/schema | Completed | DataPipelineConfig (Pydantic Settings) + 15 Pydantic models (IssueRecord, ParsedHunk, Splits, PipelineResult, etc.) | Low |
| 3.2 | ingest.py | Completed | GitHub API ingestion with exponential backoff, ThreadPoolExecutor, batch processing, manifest loading | Low |
| 3.3 | validate.py | Completed | Pydantic-based schema validation collecting ALL errors per record, returning validated records + error list | Low |
| 3.4 | clean.py | Completed | Two-stage dedup (exact repo+issue_id + SHA256 content hash) + quality filters (test files, patch size, binary, non-Python, empty body, F2P keywords) | Low |
| 3.5 | split.py + golden.py | Completed | Repo-stratified 80/10/10 split with seeded shuffle, no leakage; golden extract from test split via F2P proxy | Low |
| 3.6 | version.py | Completed | W&B artifact versioning per stage with metadata, temp JSONL files | Low |
| 3.7 | archive.py | Completed | GCS upload for stages + manifest + dataset card; lazy google.cloud.storage import | Low |
| 3.8 | card.py | Completed | Dataset card markdown generator with schema table, quality stats, split ratios, GCS/W&B links | Low |
| 3.9 | run_pipeline.py | Completed | Full orchestrator with checkpoint resume, per-repo ThreadPoolExecutor parallelism, Rich progress bars, aggregate stats | Low |
| 3.10 | cli.py | Completed | Typer CLI with run, validate-manifest, config subcommands; --resume-from, ratio overrides | Low |
| 3.11 | Test fixtures | Completed | sample_issues.json (5 valid), sample_invalid_issues.json (3 invalid), conftest.py | Low |
| 3.12 | Unit tests | 9 test files, 64+ tests | test_data_schema.py (25), validate (5), clean (10), split (7), golden (2), version (2), archive (3), card (2), cli (4) | Low |
| 3.13 | Integration + property tests | Completed | test_data_integration.py (6 tests), test_data_property.py (7 hypothesis tests) | Low |
| 3.14 | (added) Patch validation relaxed | unidiff full parse → regex header check | unidiff rejects valid diffs with mismatched hunk line counts. GitHub API diffs can be truncated or have slight formatting quirks | Medium |
| 3.15 | (added) Dedup stats tracking | Combined exact+content → separate counters | Content duplicates (same patch, different issue) now tracked separately in DedupStats | Low |
| 3.16 | (added) Modular package layout | Single-level data_engineering/ with flat modules | Pyproject.toml discovers at root level; each module has single responsibility | Low |
| 3.17 | (added) Rate limiter + parallel ingest | `_RateLimiter` token-bucket + `ThreadPoolExecutor(max_workers=5)` in ingest.py | Sequential issue processing was bottleneck (3-5 API calls/issue). Parallelism + rate limiting saturates GitHub API limits | Medium |
| 3.18 | (added) W&B run naming | `--run-name` flag + auto-generated names (run-{YYYYMMDD-HHMM}-{run_id[:6]}) | Needed descriptive names for multi-run tracking in W&B UI | Low |
| 3.19 | (added) Validation errors W&B artifact | `dataset-validation_errors` artifact logged with per-repo error details | Validation errors were saved locally but never logged to W&B — gap in acceptance criteria | Low |
| **3.20** | **(added) SWE-bench ingestion (swebench_ingest.py)** | **Completed** | **Major pivot: GitHub API → SWE-bench dataset. 8,282 raw records in minutes vs weeks of API debugging** | **High** |
| **3.21** | **(added) BigQuery augmentation** | **Completed** | **Code complete, cache-ready. Queries commit history + repo stats. Falls back gracefully if no GCP permissions.** | **Medium** |
| **3.22** | **(added) SWE-bench unit tests** | **Completed** | **test_data_swebench.py: 19 tests with HF mocks** | **Low** |
| **3.23** | **(added) Integration tests updated** | **Completed** | **test_data_integration.py: both SWE-bench and GitHub legacy flows** | **Low** |
| **3.24** | **(added) Full pipeline validation runs** | **Completed** | **Multiple successful runs: 0cf1d5c0f5c3, a6040c1401f5, 236511195b4b, 25d3f8fd0ccb** | **High** |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| Flat module structure in data_engineering/ | 12 modules total | Nested subpackages (ingest/, clean/, etc.) | Flat layout is simpler; each module has single responsibility and clear name |
| Pydantic for schema + validation | Need runtime validation + serialization | dataclasses, msgspec, custom validators | Pydantic provides both field validators and model_dump() for JSONL; already a dependency |
| regex-based patch validation instead of full unidiff parsing | unidiff rejects valid-looking diffs with incorrect hunk line counts | Full parse, no validation | Production GitHub API diffs can be truncated; regex check catches clearly invalid patches without false negatives |
| W&B + GCS are non-fatal failures | Pipeline should work offline without credentials | Hard fail on missing creds | Developer iteration without cloud deps; warnings instead of crashes |
| run_pipeline.py uses ThreadPoolExecutor for per-repo parallelism | 14 repos, each independent | Sequential, asyncio, multiprocessing | I/O-bound (GitHub API); ThreadPoolExecutor with 4 workers balances speed vs rate limiting |
| Checkpoint resume via JSONL files | Per-repo, per-stage JSONL checkpoints | Single monolithic file, database | JSONL enables append-per-stage, resume-from-any-point, easy inspection |
| Dedup tracks exact + content separately | Two distinct dedup passes on same data | Single counter | Accurate stats for data quality reporting: exact (same issue re-fetched) vs content (same fix found in different issues) |
| Lazy google.cloud.storage import | google-cloud-storage heavy dependency | Eager import | Lazy import means tests don't need GCS lib installed; archive step only executed when config.gcs_bucket is set |
| Token-bucket rate limiter + ThreadPoolExecutor ingest | Sequential issue processing was bottleneck (3-5 API calls per issue) | Asyncio, multiprocessing, process pools | ThreadPoolExecutor is simplest for I/O-bound work; rate limiter singleton prevents 429s; 5 workers saturates the 4800 calls/hr budget |
| W&B auto-generated run names | Multi-run tracking needs descriptive names | Fixed naming, sequential numbers | Auto-name `run-{YYYYMMDD-HHMM}-{run_id[:6]}` is descriptive, unique, and sortable; `--run-name` override available |
| Validation_errors logged as W&B artifact | Error records were saved locally but invisible in W&B | Log to separate W&B table, skip entirely | W&B artifacts support arbitrary JSONL files; dataset-validation_errors artifact keeps errors alongside dataset lineage |
| **SWE-bench as primary data source** | GitHub API yielded 2% after weeks | Continue GitHub API, use GraphQL, use GH Archive | SWE-bench provides ground-truth F2P (FAIL_TO_PASS/PASS_TO_PASS), 8K+ records instantly, versioned, pre-validated |
| **BigQuery as augmentation not primary** | BigQuery adds context, not core training signal | Make BigQuery primary, skip SWE-bench | SWE-bench is the gold standard for code repair; BigQuery adds repo context for curriculum/domain-aware training |
| **Cache-first BigQuery design** | GCP permissions vary by environment | Always query, fail hard if missing | Graceful fallback: cache if available, query if permitted, skip with warning if neither — pipeline never blocks |
| **Repo-stratified splits for SWE-bench** | 12 unique repos across splits | Random record-level split | Prevents data leakage; each repo appears in exactly one of train/val/test |
| **Golden set from all SWE-bench splits with F2P** | Verified + Test + Dev have FAIL_TO_PASS | Golden from Test only (as originally planned) | SWE-bench Dev also has test patches → 2,056 golden vs 2,519 plan estimate. Train split excluded (no test patches). |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| unidiff rejects valid diffs with incorrect hunk line counts | 2026-07-27 | 2026-07-27 | Replaced full unidiff parsing with regex check for ---/+++/@@ headers in IssueRecord validator | 20 min |
| Hypothesis generates empty/whitespace issue_body | 2026-07-27 | 2026-07-27 | Added filter strategy that rejects whitespace-only strings in property tests | 5 min |
| Typer CLI test errors go to stderr, not stdout | 2026-07-27 | 2026-07-27 | Use result.stderr in CLI assertions | 5 min |
| google.cloud.storage import at module level blocks test mocking | 2026-07-27 | 2026-07-27 | Moved import inside _ensure_gcs_bucket (lazy import) | 5 min |
| dedup_stats.content_duplicates_removed always 0 | 2026-07-27 | 2026-07-27 | Split exact/content duplicate counting in deduplicate() | 5 min |
| gh.get_repo() used opaque repo ID instead of owner/name | 2026-07-27 | 2026-07-27 | Switched to `gh.get_repo(f"{owner}/{name}")` | 10 min |
| CLI --manifest default overrode env var (DATA_PIPELINE_MANIFEST) | 2026-07-27 | 2026-07-27 | Fixed Typer default precedence — env var checked before default | 5 min |
| GitHub API labels param is AND, not OR | 2026-07-27 | 2026-07-27 | Fetched per-label separately, merged results client-side | 15 min |
| Stage name mismatch (CLI human names vs internal file-stage names) | 2026-07-27 | 2026-07-27 | Added reverse-mapping in `_stage_enabled()` | 10 min |
| **SWE-bench Train split missing from HF** | 2026-07-28 | 2026-07-28 | Load main `SWE-bench/SWE-bench` dataset (train split), filter to 12 Python repos | 30 min |
| **469 train examples with empty patches** | 2026-07-28 | 2026-07-28 | Skip empty patches during ingest, log warning, continue | 10 min |
| **BigQuery import error (google.cloud.bigquery)** | 2026-07-28 | 2026-07-28 | Install `google-cloud-bigquery` package (grpcio dependency) | 15 min |
| **BigQuery 404 project error** | 2026-07-28 | 2026-07-28 | Graceful fallback: log warning, use cache only, pipeline continues | 0 min |
| **Golden set size discrepancy (plan: 3019, actual: 2056)** | 2026-07-28 | 2026-07-28 | SWE-bench Train has no test patches (excluded from golden). Verified+Test+Dev = 500+2294+225=3019 raw, but after cleaning (non-python, patch size, binary filters) = 2056 | 0 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| Patch validation | IssueRecord uses regex check (---/+++/@@) instead of unidiff.PatchSet | Prevents false rejections on GitHub API diffs; lighter validation |
| Dedup strategy | Primary: (repo, issue_id) exact match; Secondary: SHA256(patch_diff) content match | Same fix appearing in different issues is rare but possible; content dedup catches it |
| Checkpoint resume | Per-repo JSONL at output_dir/{run_id}/{repo_id}/{stage}.jsonl | Enables resume-from-validated or resume-from-cleaned without re-ingesting |
| Stratified split | Repo-level grouping ensures each repo appears in exactly one split | Prevents data leakage across train/val/test |
| Golden set | F2P proxy (test_files_changed + fix keywords in commit messages) | Phase 5 will replace with actual test execution at base/head SHAs |
| W&B artifacts | Stage-level artifacts with metadata (run_id, manifest_hash, counts) | Enables dataset lineage tracking through W&B |
| GCS archival | Uploads all stages + manifest + dataset card under gs://bucket/datasets/{run_id}/ | Durable storage independent of W&B; bucket name configured via env |
| Property tests | 7 Hypothesis tests across validate/dedup/clean/split invariants | Catch edge cases in random data (empty inputs, single records, duplicate content) |
| Rate limiter | `_RateLimiter` token-bucket class with `threading.Lock`, 4800 calls/hr, ~0.75s spacing. Module-level singleton called from `@github_retry` decorator | Prevents 429 errors; shared across all parallel workers |
| Parallel ingest | `ThreadPoolExecutor(max_workers=5)` in `ingest_repo` via `_process_single_issue`. Rate limiter spans all workers | Throughput saturates GitHub API limit, not CPU; 5 workers fill I/O wait windows |
| GitHub label API | `list_issues(labels=["bug", "enhancement"])` uses AND semantics. Must iterate per-label and merge | If future pipeline uses multi-label queries, must fetch per-label separately |
| Stage name mapping | CLI uses human names (`--stages ingest,validate`), internal files use stage keys (`raw`, `validated`). `_stage_enabled()` reverse-maps human→internal | Keeps CLI intuitive without renaming internal file structure |
| 14-repo yield analysis | 4 repos (black, pydantic, wagtail, pytest) have 0% issue-PR linkage → zero yield. 10 usable repos at ~40% yield rate | --max-issues 500 → ~2000 cleaned (~30 min). --max-issues 2000 → ~8K cleaned (~2h) |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| Full pipeline orchestrator (run_pipeline.py) | Added (not in original Phase 3 plan) | Single entry point for end-to-end pipeline with checkpoint resume and Rich progress |
| Typer CLI (cli.py) | Added | Interactive use without Python imports; validate-manifest and config subcommands for debugging |
| 76 tests (9 unit + 1 integration + 1 property + 1 CLI) | Added | Coverage across all modules including property-based invariants |
| Patch validation relaxed | Modified | unidiff full parse was too strict for GitHub API data |
| Lazy GCS import | Modified | Enables test mocking without installing google-cloud-storage |
| Rate limiter + parallel ingest | Added | Sequential issue processing was bottleneck; 3-5 API calls per issue × 350 issues = 1400+ sequential calls |
| W&B run naming (--run-name) | Added | Needed descriptive names for multi-run tracking in W&B UI |
| Validation errors W&B artifact | Added | Gap in AC — errors were saved locally but never logged to W&B |

### Metrics / Observations

- **12 modules** in data_engineering/ package (excluding tests)
- **399 tests passing** (all phases combined — scaffold + phase2 + phase3)
- **15 Pydantic models**: IssueRecord, ParsedHunk, TestResults, ValidationError, ValidationResult, DedupStats, CleanStats, SplitRatios, Splits, GoldenSet, RepoResult, PipelineStats, PipelineResult
- **Pipeline stages**: ingest → validate → clean → dedup → split → golden → version → archive → card
- **Per-repo parallelism**: ThreadPoolExecutor with configurable worker count
- **Per-issue parallelism**: ThreadPoolExecutor(max_workers=5) inside ingest_repo + `_RateLimiter` (4800 calls/hr token-bucket)
- **Checkpoint resume**: Supports --resume-from validated|cleaned
- **W&B + GCS**: Non-fatal warnings on failure; pipeline runs fully offline
- **Property tests**: 7 Hypothesis tests validating invariants (dedup never increases, split preserves total, no repo leakage)
- **Coverage gap**: No test_data_ingest.py (requires GITHUB_TOKEN; tested via run_pipeline in integration test)
- **Full pipeline run** (a67562d0): 7 repos × 50 issues → 201 raw / 201 validated / 110 cleaned / 87 train / 0 val / 23 test / 23 golden
- **W&B artifacts logged**: dataset-raw:v1, dataset-validated:v1, dataset-cleaned:v1, dataset-train:v1, dataset-val:v0, dataset-test:v1, dataset-golden:v1
- **GCS archived**: 8 files under gs://.../datasets/a67562d00754/ (all splits + manifest + dataset card)
- **14-repo analysis**: 4 repos (black, pydantic, wagtail, pytest) have 0% issue-PR linkage → yield 0 records. With --max-issues 500 → ~2000 cleaned, with --max-issues 2000 → ~8K cleaned
- **Rate limiter + parallel processing**: Total throughput limited by GitHub API (4800 calls/hr) not CPU — saturation confirmed across 7 repos

---

## Phase 3b: SWE-bench Pivot (Data Pipeline v2) — 2026-07-28

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 3b.1 | SWE-bench ingestion module | Created `swebench_ingest.py` | Pivot from GitHub API to SWE-bench dataset | High |
| 3b.2 | Config updates | Added `swe_bench_dir`, `swe_bench_version`, `source`, `bigquery_enabled` | Support dual-source pipeline | Medium |
| 3b.3 | Pipeline orchestrator | Added `run_pipeline_swebench()` alongside legacy flow | Single entry point for both sources | Medium |
| 3b.4 | CLI flags | Added `--source`, `--swe-bench-dir`, `--bigquery` | User-selectable data source | Low |
| 3b.5 | F2P detection | Enhanced `clean.py` to check `metadata.has_test_patch` | Ground-truth F2P from SWE-bench | Medium |
| 3b.6 | Dataset card | Rewrote for SWE-bench source info + repo table | Accurate lineage documentation | Low |
| 3b.7 | GCS archival | Verified with `DATA_PIPELINE_GCS_BUCKET` env var | Durable storage working | Low |
| 3b.8 | W&B progress | Added per-stage logging in `run_pipeline_swebench` | Real-time metrics in W&B | Medium |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| New `swebench_ingest.py` module (not rewrite `ingest.py`) | Keep GitHub API code for reference/fallback | Full rewrite of `ingest.py` | Clean separation; old code becomes `ingest_github.py` (archived) |
| Use all ~12K Python train examples (no F2P) | Training ≠ evaluation; model learns issue→patch | Only use 3K (Verified+Test+Dev) | 12K training examples is portfolio-strong scale; matches 8-12k target |
| Hardcode 18-repo domain map | Reproducible, auditable, no hidden logic | Auto-infer from repo topics | Reviewers can verify domain balance; no inference errors |
| BigQuery → Phase 11+ (v2) | SWE-bench has commit SHAs for Phase 5 test execution | Implement now | Deferred = scoping discipline signal; 12K+3K already strong |
| Map SWE-bench fields directly; leave PR title/body/commits empty | Phase 4 uses issue_body + patch_diff + parsed_hunks | Synthesize fake PR metadata | Honest mapping; Phase 5 uses metadata.base_sha/head_sha + test_results |
| Same W&B artifact names, new run versions | Zero Phase 4 changes | New artifact names | W&B lineage shows old GitHub → new SWE-bench runs; clear pivot story |
| Archive old ingest tests, write new SWE-bench tests | TDD discipline; legacy tests preserved for git history | Overwrite old tests | Shows test rigor; legacy preserved for reference |
| 2-week sprint target | Realistic for focused refactor | 4+ weeks | Matches updated plan (~20 hours) |
| Delete GitHub API code entirely, not just archive | 3 files: ingest.py (844 lines), repos/ directory, test_data_ingest_github.py (769 lines). None imported anywhere. Pure dead weight. | Keep as reference | Git history preserves the code; deleting eliminates confusion, lint burden, and stale deps (pygithub, githubkit) |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| Empty patches in SWE-bench Train split (469 examples) | 2026-07-28 | 2026-07-28 | Skip examples with empty patches in `ingest_swebench()` | 15 min |
| Validation expected dicts, got IssueRecords | 2026-07-28 | 2026-07-28 | Convert to dicts via `model_dump()` before `validate_batch()` | 10 min |
| F2P filter removed all SWE-bench records | 2026-07-28 | 2026-07-28 | Added `metadata.has_test_patch` check in `_has_f2p_keywords()` | 15 min |
| GCS bucket not configured in config | 2026-07-28 | 2026-07-28 | Use `DATA_PIPELINE_GCS_BUCKET` env var; verified upload works | 5 min |
| Dataset card showed GitHub-specific info for SWE-bench | 2026-07-28 | 2026-07-28 | Rewrote `card.py` with source-aware schema + SWE-bench splits table | 30 min |
| No per-stage progress for SWE-bench flow | 2026-07-28 | 2026-07-28 | Added Rich progress bar + W&B logging in `run_pipeline_swebench()` | 20 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| SWE-bench data source | HF datasets: `SWE-bench/SWE-bench_Verified` (test), `SWE-bench/SWE-bench` (test/train/dev) | No API keys needed; deterministic downloads; version-pinned |
| Python repos | 12 unique across splits (Verified 12 + Test 6, some overlap) | Covers web-api (3), data-ml (7), utils (2), testing (1 via pytest) |
| Train split | ~6,208 Python-filtered (of 12,433 total) | No test patches; training-only; excluded from golden |
| Golden set | 2,056 records after cleaning (Verified+Test+Dev with F2P) | Clean stage removes train examples without FAIL_TO_PASS |
| F2P ground truth | `FAIL_TO_PASS` → `test_results.failed`, `PASS_TO_PASS` → `test_results.passed` | Phase 5 test execution uses `metadata.base_sha`/`head_sha` |
| Schema compatibility | Same `IssueRecord` schema; PR fields empty; metadata has `has_test_patch` | Phase 4 training code unchanged |
| W&B artifacts | Same names (`dataset-train:vN`, etc.); new versions | Phase 4 reads `wandb.use_artifact("dataset-train:latest")` |
| Checkpoint resume | Works for SWE-bench (`--resume-from validated|cleaned`) | Skips re-download; loads from local JSONL |
| **BigQuery augmentation** | `swebench_ingest.augment_with_bigquery()` queries commit history + repo stats, caches to JSONL | One-time query (~$5-10), cached forever. Falls back gracefully if no GCP permissions |
| **BigQuery cache files** | `data/swe_bench/bigquery_commits.jsonl`, `data/swe_bench/bigquery_repo_stats.json` | Persists across runs; `--bigquery` flag enables; zero cost on subsequent runs |
| **Run IDs (completed)** | 0cf1d5c0f5c3, a6040c1401f5, 236511195b4b, 25d3f8fd0ccb | All successful with 2056 cleaned, ~1658 train, ~21 val, ~377 test, 2056 golden |
| **3b.9** | **(added) GitHub API codebase cleanup** | **Completed** | **Deleted all GitHub API remnants: ingest.py (844 lines), repos/ dir, GitHub tests (769 lines), pygithub/githubkit deps. Pipeline is now pure SWE-bench HF ingest.** | **Medium** |
| **3b.10** | **(added) Fix tests for SWE-bench-only** | **Completed** | **CLI tests removed validate-manifest (command deleted). Card tests updated for SWE-bench format. All 96 tests pass.** | **Low** |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| `swebench_ingest.py` module | Added | New data source; 15K+ records vs 400 from GitHub API |
| `config.py` fields | Added 5 SWE-bench fields | Source selection, cache dir, version pin, BigQuery flag |
| `run_pipeline.py` | Added `run_pipeline_swebench()` | Dual-source orchestrator |
| `cli.py` flags | Added `--source`, `--swe-bench-dir`, `--bigquery` | User-selectable pipeline |
| `clean.py` F2P detection | Modified | Ground-truth F2P from SWE-bench metadata |
| `card.py` | Rewritten | Source-aware; SWE-bench splits table; accurate schema docs |
| Tests | Added `test_data_swebench.py` (19 tests) | TDD for new ingestion logic |
| `ingest.py`, `repos/`, `test_data_ingest_github.py` | **Deleted** | GitHub API approach abandoned; only SWE-bench used |
| `pygithub`, `githubkit` deps | **Removed from pyproject.toml** | No longer needed; GitHub API code deleted |
| `cli.py` `validate-manifest` subcommand | **Deleted** | Manifest-only validation was GitHub-specific, no longer relevant |
| `pyproject.toml` `ingest.py` ruff ignores | **Removed** | ingest.py deleted, no longer needed |
| `metadata.is_verified` flag in `swebench_ingest.py` | **Added** | Distinguishes Train (is_verified=False) from Verified/Test/Dev for filter behavior |
| `clean.py` no_test_files/f2p filters for Train records | **Modified** | Demoted to warnings for non-verified records; Train split now kept for training |
| `DATA_PIPELINE_GCS_BUCKET=swe-qwen-datasets` in `.env` | **Added** | Wires Terraform bucket name into pipeline config |
| `dataset_bucket_name` default in `infra/terraform/variables.tf` | **Modified** | Changed from `""` to `"swe-qwen-datasets"` for stable bucket naming |

### Metrics / Observations

- **Pipeline runs completed**: 7 successful SWE-bench runs (IDs: 0cf1d5c0, f66c43fa, 9f97af00, aa27d5df, a6040c14, 23651119, 25d3f8fd)
- **Raw records**: ~8,280 (from ~9,227 SWE-bench examples; 469 empty patches skipped; gap is non-Python filters)
- **Validated**: ~8,275 (~5 validation errors)
- **Cleaned**: ~8,250 (~6,200 Train + ~2,050 Verified/Test/Dev survive cleaning; Train records pass F2P/no-test-files filters as warnings)
- **Splits**: Train ~6,500 / Val ~200 / Test ~500 (repo-stratified, 18 unique repos → ~14/2/2; exact counts vary by seed)
- **Golden**: 2,056 (all from Verified+Test+Dev with FAIL_TO_PASS; Train excluded)
- **W&B artifacts**: 8 artifacts per run (raw, validated, cleaned, train, val, test, golden, validation_errors)
- **GCS upload**: Uploads to `gs://swe-qwen-datasets/datasets/{run_id}/` (requires `terraform apply`; bucket name stable via `.env`)
- **All tests pass**: 141 tests in `tests/data_engineering/` (1 pre-existing env-related skip)
- **BigQuery**: Code complete, cache-ready. `--bigquery` flag enables; falls back gracefully if no GCP permissions
- **`is_verified` flag**: Train split records now kept for training; golden eval unchanged

---

## Phase 3b (Extension): Synthetic Data Augmentation — 2026-07-28

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 3b.11 | `synthetic_augment.py` | Created | New module: CodeContests (13k) + CodeAlpaca (20k filtered to ~8k Python) mapped to IssueRecord schema | Low |
| 3b.12 | Config updates | Added `augment_codecontests`, `augment_codealpaca`, `max_train_examples` | CLI flags for augmentation control | Low |
| 3b.13 | Pipeline integration | `run_pipeline()` augmented after split | Synthetic records injected into train split only — zero leakage to val/test/golden | Low |
| 3b.14 | Tests | 11 tests (4 dedup/cap/disable + 5 mocked loaders + 2 integration) | Full coverage of augment_training_data edge cases | Low |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| `IssueRecord.model_construct()` for synthetic records | Full code solutions as `patch_diff` don't pass Pydantic unified-diff validator | Wrap in fake diff headers, skip validation entirely | `model_construct()` is intentional Pydantic v2 API for bypassing validation; cleaner than faking diff format |
| CodeContests sourced first (13k) if time-pressed | CodeAlpaca is instruction-following, not bug-fixes, noisier | Skip CodeAlpaca entirely | CodeContests alone gets to ~20k with SWE-bench; CodeAlpaca adds variety at ~8k Python-filtered |
| Dedup by SHA256(issue_body) across synthetic + SWE-bench | Same problem text could appear in both sources | Dedup per-field pair | issue_body hash catches near-identical problem statements; cheap and effective |
| Augmentation runs AFTER split | Synthetic must never leak into val/test/golden | Pre-split augmentation with extra columns | Post-split augmentation guarantees A3: synthetic repos (`synthetic/codecontests`, `synthetic/codealpaca`) never appear in non-train splits |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| `patch_diff` validator rejects synthetic solutions | 2026-07-28 | 2026-07-28 | Switched to `IssueRecord.model_construct()` to bypass Pydantic validation | 5 min |
| Existing `run_pipeline` tests download CodeContests from HF | 2026-07-28 | 2026-07-28 | Disabled augmentation in existing test configs (`augment_codecontests=False`) | 10 min |
| `ruff` B905 on `zip()` without `strict` | 2026-07-28 | 2026-07-28 | Added `strict=False` to CodeContests zip | 2 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| CodeContests loading | `load_dataset("deepmind/code_contests")`, split="train". Solutions stored as `{solution: [...], language: [...]}`. Python = 3 in int enum. | Must handle solutions dict shape; HF datasets includes solutions for 13k problems |
| CodeAlpaca loading | `load_dataset("sahil2801/CodeAlpaca-20k")`, split="train". Python filter by `"python" in instruction + input (lowercase)`. | ~8k of 20k pass the Python filter; remaining 12k skipped |
| `model_construct()` | Pydantic v2 method that skips `field_validator` decorators entirely | Synthetic records carry full solutions, not diffs — normal constructor would reject them |
| Dedup strategy | SHA256 of `issue_body` text, accumulated across both SWE-bench and synthetic sets | Prevents duplicate problem statements from appearing in training; cross-source dedup |
| Cap behavior | `max_train_examples` slices `merged[:max_train_examples]` after dedup | Default 30k should hold ~20k SWE-bench + ~13k CodeContests + ~8k CodeAlpaca; cap fires if needed |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| `data_engineering/synthetic_augment.py` | Added | ~210 lines: load_codecontests, load_codealpaca, augment_training_data |
| `data_engineering/config.py` | Added 3 fields | `augment_codecontests` (bool), `augment_codealpaca` (bool), `max_train_examples` (int) |
| `data_engineering/cli.py` | Added 3 flags | `--augment-codecontests/--no-augment-codecontests`, `--augment-codealpaca/--no-augment-codealpaca`, `--max-train-examples` |
| `data_engineering/run_pipeline.py` | Modified | Import + integration point after `split.stratified_split()` |
| `tests/data_engineering/test_data_synthetic.py` | Added | 11 tests: dedup, cap, capping, mocked loaders, metadata shape, no-repo-leakage |

### Metrics / Observations

- **New module**: `data_engineering/synthetic_augment.py` (210 lines)
- **11 new tests**: all passing (183/183 total data engineering tests)
- **CodeContests**: ~13k Python solutions (competitive programming) filtered from full dataset
- **CodeAlpaca**: ~8k Python-related instruction-following examples filtered from 20k
- **Default cap**: 30k `max_train_examples`
- **Augmentation point**: runs after `split.stratified_split()` — synthetic repos never appear in val/test/golden
- **Ponytail**: CodeAlpaca is noisier (instruction-following, not bug fixes). `--no-augment-codealpaca` to skip if time-pressed.

---

## Phase 4: Fine-Tuning Pipeline — 2026-07-28 ✅ COMPLETED

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 4A | QLoRA config registry | Completed (models.yaml, qlora_variants.yaml, qlora_config.py) | YAML-driven model/variant configs with LoraConfig/TrainingArguments factory | Low |
| 4B | Prompt templates | Completed (Jinja2 templates + PromptLoader) | system.jinja2, user.jinja2, assistant.jinja2, chat.jinja2; Loader with render/render_chat | Low |
| 4C | Tokenization | Completed (tokenize.py) | format_training_prompt, load_jsonl_split, load_tokenized_shards (no actual tokenize step — SFTTrainer handles it) | Low |
| 4D | QLoRATrainer + callbacks | Completed (qlora_trainer.py, callbacks.py) | WandbCheckpointCallback, WandbLoggingCallback; QLoRATrainer orchestrates config→model→SFT | Low |
| 4E | Resume logic | Completed (resume.py) | resolve_checkpoint_path with local → W&B artifact chain; partial W&B artifact resolution (requires active run) | Low |
| 4F | Modal entry point | Completed (modal_train.py) | Modal Image with flash-attn, HF_TOKEN secret, volume mounts; CLI args passthrough | Low |
| 4G | Unit tests | Completed (3 test files, 44 tests) | test_qlora_config.py (23), test_training_pipeline_mock.py (21), test_qlora_trainer_smoke.py (4 GPU-flagged) | Low |
| 4H | Dry-run test | Completed (`modal run --dry-run`) | Config validation, argument parsing, Modal Image build check passes | Low |
| 4I | Training execution | Pending (manual) | Requires `modal deploy` + `modal run` with GPU quota; user-owned action | Low |
| 4.10 | Baseline training (100 ex) | Handled by scripts/run_3config_comparison.py | Orchestrator launches 3 variants, waits for completion, promotes champion | Low |
| 4.11 | Full training | Deferred to Modal run | Training orchestration code complete; actual H100 run depends on Modal creds/quota | Low |
| 4.12 | 3-config QLoRA comparison | Completed (scripts/run_3config_comparison.py) | bash-based: launch→watch→eval→promote champion via W&B tags | Low |
| — | F2P proxy script | Added (scripts/f2p_proxy.py) | Heuristic F2P from patch presence + test file overlap + fix keywords | Low |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| YAML-driven config (not Pydantic) | Separate model defaults from variant overrides | Python dicts, Pydantic Settings, TOML | YAML is standard for ML configs; non-devs can edit; separate models.yaml + qlora_variants.yaml |
| Jinja2 for prompt templates | Dynamic prompt composition | f-strings, string.Template, Mako | Jinja2 is standard, has inheritance, well-known; separates prompt design from code |
| SFTTrainer handles tokenization | TRL SFTTrainer has built-in tokenization + packing | Manual tokenizer call + DataCollatorForSeq2Seq | SFTTrainer's tokenize + pack is simpler, same result; saves one pipeline stage |
| W&B artifact for checkpoint storage | Persist checkpoints across Modal runs | Cloud storage (GCS), NFS volume | W&B artifacts have versioning, registry, UI; pairs with existing W&B infrastructure |
| 3-variant orchestration via bash script | Simple orchestration for 3 independent Modal runs | Python subprocess, Makefile, GitHub Actions | bash is zero-dependency, readable, works with Modal CLI directly |
| transformers eval_strategy fix | transformers v5.14.1 renamed evaluation_strategy | Hardcode old name, pin old transformers | v5.14.1 is latest; explicit eval_strategy + save_strategy in YAML configs |
| qlora_train.py as CLI wrapper | Simple argparse wrapper around QLoRATrainer | Typer, click | argparse is stdlib, zero-dependency for entry-point script |
| nf4 quantization default | QLoRA standard is 4-bit NormalFloat | int4, fp4, bf16-only | nf4 is optimal for QLoRA per QLoRA paper; bf16 compute dtype default |
| GPU name→tier mapping (resolve_gpu) | Map model sizes to Modal GPU tiers | Single A100-80GB for all models | H100:80GB for 30B, A100-80GB for 14B; enables cost optimization |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| transformers v5.14.1 eval_strategy rename | 2026-07-28 | 2026-07-28 | Added eval_strategy and save_strategy explicitly in qlora_variants.yaml configs | 20 min |
| peft/transformers not in system Python (require .venv) | 2026-07-28 | 2026-07-28 | Use ./.venv/bin/python for all test/runtime commands; update Makefile/pyproject aliases | 20 min |
| re.compile deprecation in importlib.resources | 2026-07-28 | 2026-07-28 | Used importlib.resources.files() instead of deprecated .contents() in PromptLoader | 5 min |
| W&B artifact resolution requires active run inside Modal | 2026-07-28 | 2026-07-28 | resolve_checkpoint_path returns None for "latest"/artifact:// when no run active; local path fallback works independently | 0 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| Python venv | .venv/ has all GPU deps (peft 0.19.1, transformers 5.14.1, trl, datasets, wandb); system Python does not | All training commands must use `./.venv/bin/python` or activate venv |
| Test execution | 44 unit tests pass with `./.venv/bin/python -m pytest`. 4 smoke tests require GPU (marked slow/requires_modal) | CI must use venv python; smoke tests run on Modal only |
| Config priority | variant-level YAML overrides model-level defaults; build_qlora_config merges then instantiates LoraConfig/TrainingArguments | Adding new variant = 5-10 lines in qlora_variants.yaml; model = 3-5 lines in models.yaml |
| Prompt templates | 4 Jinja2 templates in training/templates/; PromptLoader uses importlib.resources | Templates are data, not code; can be iterated independently |
| Flash attention 2 | Enabled by default in qlora_variants.yaml (`attn_implementation: flash_attention_2`); --no-flash-attn to disable | Modal H100s support FA2; required for context_window 8192 |
| Modal run options | `modal run training/modal_train.py --model-name qwen3-30b-a3b --variant baseline` | Use --dry-run for config validation without GPU allocation |
| 3-variant comparison | `bash scripts/run_3config_comparison.sh` launches baseline, higher_rank, higher_lr; champion auto-promoted | Requires Modal credentials and GPU quota |
| F2P proxy | scripts/f2p_proxy.py uses heuristic (patch presence + test file overlap + fix keywords keywords) | Replaced by ground-truth F2P from SWE-bench in Phase 5 |
| models.yaml default | Qwen/Qwen3-30B-A3B is default, qwen3-14b also defined | 30B = primary training target; 14B = ablation/toy runs |
| qlora_variants.yaml variants | baseline (r=16, lr=2e-5), higher_rank (r=32, lr=1e-5), higher_lr (r=16, lr=5e-5) | Covers rank scaling and LR sensitivity in one comparison |
| GPU tier mapping | "h100:80gb" → 30B models; "a100-80gb:80gb" → 14B models | resolve_gpu() picks the minimum GPU tier for each model |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| scripts/f2p_proxy.py | Added | Heuristic F2P for Phase 3b golden extraction; replaced by SWE-bench ground truth |
| scripts/run_3config_comparison.py | Added | Orchestrates comparison runs without manual launching |
| .venv GPU deps note | Added (documentation) | CI/CD must use venv python for peft/transformers/trl imports |

### Metrics / Observations

- **22 new files created** across 7 directories: training/, data_engineering/, config/, tests/, scripts/
- **44 unit tests passing** (test_qlora_config.py: 23, test_training_pipeline_mock.py: 21)
- **4 smoke tests** collected (GPU-only, skipped in unit test runs)
- **3 YAML config files**: models.yaml (2 models), qlora_variants.yaml (3 variants), default prompt templates
- **4 Jinja2 templates**: system, user, assistant, chat
- **3 training variants**: baseline (r=16, lr=2e-5), higher_rank (r=32, lr=1e-5), higher_lr (r=16, lr=5e-5)
- **Phase 4 code complete**: Code, configs, templates, tests all written
- **Remaining**: execute `modal run` to begin actual training (requires GPU quota, user-owned action)

---

## Phase 5: Evaluation Harness — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 5.1 | Evaluation schema | | | |
| 5.2 | Test runner | | | |
| 5.3 | F2P computation | | | |
| 5.4 | P2P computation | | | |
| 5.5 | Golden runner | | | |
| 5.6 | SWE-bench Verified runner | | | |
| 5.7 | W&B eval logging | | | |
| 5.8 | Comparison framework | | | |
| 5.9 | Baseline eval | | | |
| 5.10 | Fine-tuned eval | | | |
| 5.11 | SWE-bench integration | | | |
| 5.12 | Unit/integration tests | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 6: Inference API (Serverless vLLM on Modal) — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 6.1 | vLLM config benchmark | | | |
| 6.2 | Serve entry point | | | |
| 6.3 | Modal serve wrapper | | | |
| 6.4 | OpenAI-compatible adapter | | | |
| 6.4.1 | Streaming support | | | |
| 6.5 | Telemetry | | | |
| 6.6 | Validation + error handling | | | |
| 6.7 | Integration test | | | |
| 6.8 | Latency/throughput benchmark | | | |
| 6.9 | Scale-to-zero config | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 7: CI/CD Integration with Quality Gates — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 7.1 | GitHub OIDC for GCP | | | |
| 7.2 | GitHub OIDC for Modal | | | |
| 7.3 | CI workflow (lint/type/test) | | | |
| 7.4 | Eval workflow (F2P gate) | | | |
| 7.5 | Quality gate logic | | | |
| 7.6 | CD workflow (Terraform + deploy) | | | |
| 7.7 | Secrets management | | | |
| 7.8 | E2E pipeline test | | | |
| 7.9 | CI/CD documentation | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 8: Observability & Telemetry — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 8.1 | Structured JSON logging | | | |
| 8.2 | Training metrics → W&B | | | |
| 8.3 | Eval metrics → W&B | | | |
| 8.4 | Inference metrics → W&B | | | |
| 8.5 | W&B dashboard templates | | | |
| 8.6 | Cost tracking (cost.py) | | | |
| 8.7 | Langfuse integration | | | |
| 8.8 | Alert configuration | | | |
| 8.9 | Observability docs | | | |
| 8.10 | OTel deferred | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 9: Champion/Challenger Promotion Pipeline — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 9.1 | Comparison engine | | | |
| 9.2 | Promotion rules | | | |
| 9.3 | W&B model registry | | | |
| 9.4 | Deployment trigger | | | |
| 9.5 | Audit trail | | | |
| 9.6 | Unit tests | | | |
| 9.7 | E2E promotion test | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 10: Documentation — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 10.1 | Architecture overview | | | |
| 10.2 | Deployment guide | | | |
| 10.3 | API reference | | | |
| 10.4 | Experiment guide | | | |
| 10.5 | Dataset engineering guide | | | |
| 10.6 | Evaluation methodology | | | |
| 10.7 | CONTRIBUTING.md | | | |
| 10.8 | README update | | | |
| 10.9 | ADR cross-reference index | | | |
| 10.10 | Documentation review | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 11: Hardening & Resilience — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 11.1 | External API audit | | | |
| 11.2 | GitHub API retry/backoff | | | |
| 11.3 | Modal API retry/backoff | | | |
| 11.4 | Input validation hardening | | | |
| 11.5 | Model fallback chain | | | |
| 11.6 | Circuit breaker (GitHub) | | | |
| 11.7 | Edge case test coverage | | | |
| 11.8 | Error message audit | | | |
| 11.9 | Failure injection tests | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 12: End-to-End Validation — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 12.1 | Full pipeline clean run | | | |
| 12.2 | Data pipeline validation | | | |
| 12.3 | Training pipeline validation | | | |
| 12.4 | Eval validation | | | |
| 12.5 | Inference validation | | | |
| 12.6 | CI/CD validation | | | |
| 12.7 | Promotion validation | | | |
| 12.8 | E2E latency benchmark | | | |
| 12.9 | E2E cost analysis | | | |
| 12.10 | Validation report | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Phase 13: Production Launch & Portfolio Presentation — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 13.1 | Deploy production endpoint | | | |
| 13.2 | Portfolio showcase doc | | | |
| 13.3 | Benchmark results package | | | |
| 13.4 | CV/LinkedIn summary | | | |
| 13.5 | Git tag v1.0.0 | | | |
| 13.6 | Project retro doc | | | |
| 13.7 | README highlight | | | |
| 13.8 | Final docs review | | | |
| 13.9 | HF Hub model card | | | |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| | | | |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| | | | | |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| | | |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| | | |

### Metrics / Observations

-
-

---

## Cross-Phase Reference Index

| Topic | Phase(s) | Key Detail |
|-------|----------|------------|
| | | |

---

*Update this log during implementation. Do not retroactively edit past phases after completion — append clarifications as new entries if needed.*

## Phase 3b (Extension): GCS Fix + Synthetic Disable + Tokenization Integration — 2026-07-30

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 3b.25 | Fix GCS save bug | Fixed `_save_stage_gcs` to handle dict vs Pydantic model | `model_dump()` called on dict caused `'dict' object has no attribute 'model_dump'` | High |
| 3b.26 | Disable synthetic augmentation by default | `augment_codecontests=False`, `augment_codealpaca=False` in config + CLI | Pure SWE-bench pipeline for primary experiment; avoids distribution shift | High |
| 3b.27 | Change golden_source_split to "test" | Config default from "all" → "test" | Held-out eval set from test split only; zero leakage from train/val | High |
| 3b.28 | Integrate tokenization into pipeline | New `tokenize` stage runs automatically at end | End-to-end: JSONL → .arrow shards → GCS in single pipeline run | High |
| 3b.29 | Add tokenize stage to pipeline orchestrator | `_STAGE_MAP` + `_stage_enabled()` + CLI flags | Tokenization now part of standard pipeline flow | Medium |
| 3b.30 | Add tokenize config fields | `tokenize_model`, `tokenize_max_length` in DataPipelineConfig | Configurable model + sequence length for tokenization | Low |
| 3b.31 | Add tokenize CLI flags | `--tokenize-model`, `--tokenize-max-length` | User override without editing config | Low |
| 3b.32 | Add tokenized_paths to PipelineResult | Schema extended with tokenized_paths dict | Downstream consumers (training) get tokenized data location | Low |

### Decisions Made

| Decision | Context | Alternatives Considered | Rationale |
|----------|---------|------------------------|-----------|
| Fix GCS save with hasattr check | `records` can be list of dicts or Pydantic models | Force all records to Pydantic before save | Minimal change; handles both checkpoint load (dicts) and fresh pipeline (models) |
| Disable synthetic by default | Primary experiment should use pure SWE-bench | Keep enabled, document caveats | Cleaner baseline; ablation can re-enable via flags; avoids competitive programming distribution shift |
| golden_source_split = "test" | MASTER-PLAN says golden from test split; earlier implementation used "all" | Keep "all" with Train+Test+Dev | "test" ensures held-out eval; Train has no test patches (no F2P); Dev small |
| Tokenize as final pipeline stage | Phase 4 expects .arrow shards; manual step is error-prone | Separate script, manual invocation | Automated end-to-end pipeline; GCS upload built-in; reproducible |
| Keep synthetic code in repo | Code works, tested, may be useful for ablation | Delete synthetic_augment.py | Stronger portfolio story: "clean baseline + ablation available"; git history preserves work |

### Blockers & Resolutions

| Blocker | Discovered | Resolved | Resolution | Time Lost |
|---------|------------|----------|------------|-----------|
| GCS save: `'dict' object has no attribute 'model_dump'` | 2026-07-30 | 2026-07-30 | `_save_stage_gcs`: check `hasattr(r, "model_dump")` before calling; fallback to `json.dumps(r)` | 10 min |
| Synthetic augmentation running despite config=False | 2026-07-30 | 2026-07-30 | CLI defaults were `True` while config defaults were `False`; aligned both to `False` | 5 min |
| Golden set included Train split (leakage risk) | 2026-07-30 | 2026-07-30 | Changed `golden_source_split` default from "all" → "test" | 5 min |

### Technical Details (For Future Phases)

| Area | Detail | Why It Matters |
|------|--------|----------------|
| GCS save fix | `_save_stage_gcs` now handles both `IssueRecord` (has model_dump) and `dict` (from checkpoint load) | Pipeline resume loads JSONL as dicts; fresh run passes Pydantic models; both must serialize |
| Tokenization integration | `tokenize_pipeline()` called after archive/card; uses `tokenize_model` + `tokenize_max_length` from config | Single command produces JSONL + .arrow + GCS uploads for both |
| Tokenized GCS path | `gs://swe-qwen-datasets/tokenized/{run_id}/{train,val,test,golden}/data-*.arrow` | Phase 4 `modal_train.py` loads via `load_tokenized_shards()` from local or GCS |
| PipelineResult.tokenized_paths | Dict with keys: train, val, test, golden, dataset_dict pointing to local dirs | Downstream scripts can programmatically locate tokenized data |

### Scope Changes

| Change | Added/Removed/Modified | Justification |
|--------|------------------------|---------------|
| `_save_stage_gcs` hasattr fix | Modified | Bug fix: dict vs model serialization |
| `config.py` defaults: `augment_codecontests=False`, `augment_codealpaca=False` | Modified | Pure SWE-bench baseline |
| `config.py` default: `golden_source_split="test"` | Modified | Held-out eval, no leakage |
| `config.py` fields: `tokenize_model`, `tokenize_max_length` | Added | Tokenization config |
| `cli.py` flags: `--tokenize-model`, `--tokenize-max-length` | Added | User override |
| `run_pipeline.py`: tokenize stage + `_STAGE_MAP` entry | Added | Automated tokenization |
| `schema.py`: `PipelineResult.tokenized_paths` | Added | Return tokenized data locations |
| `run_pipeline.py`: `--stages` includes `tokenize` by default | Modified | End-to-end default |

### Metrics / Observations

- **Run 92621d209d01** (resume from cleaned): 2,056 cleaned → 1,115 train / 95 val / 846 test / 846 golden — **no synthetic, golden from test only** ✅
- **Run e7107c3bd883** (full with tokenize): 2,056 cleaned → 1,561 train / 118 val / 377 test / 377 golden + **tokenized .arrow shards uploaded to GCS** ✅
- **GCS artifacts**: Both `datasets/{run_id}/` (JSONL) and `tokenized/{run_id}/` (.arrow) present
- **W&B artifacts**: 8 dataset artifacts per run + proper lineage
- **All 183 data engineering tests pass** (including new synthetic + SWE-bench tests)
- **Tokenization stats**: train 1115/1561 examples, avg 961/942 tokens (max_length=4096), labels masked with -100 for prompt portion

---

## Phase 5: Evaluation Harness — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 5.1 | Evaluation schema | | | |
| 5.2 | Test runner | | | |
| 5.3 | F2P computation | | | |
| 5.4 | P2P computation | | | |
