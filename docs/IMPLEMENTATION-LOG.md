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

## Phase 3: Data Engineering Pipeline — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 3.1 | Ingest from GitHub API | | | |
| 3.2 | Validate schema | | | |
| 3.3 | Clean/normalize data | | | |
| 3.4 | Train/val/test split | | | |
| 3.5 | Golden eval subset | | | |
| 3.6 | W&B dataset artifacts | | | |
| 3.7 | Archive raw data | | | |
| 3.8 | Version dataset | | | |

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

## Phase 4: Fine-Tuning Pipeline — YYYY-MM-DD

### Deviation Log

| Task | Planned | Actual | Reason | Impact |
|------|---------|--------|--------|--------|
| 4.1 | Model selection + baseline eval | | | |
| 4.2 | QLoRA config | | | |
| 4.3 | Prompt engineering workstream | | | |
| 4.4 | Training entry point | | | |
| 4.5 | Modal training wrapper | | | |
| 4.6 | W&B callbacks | | | |
| 4.7 | Checkpoint versioning | | | |
| 4.8 | Experiment resumption | | | |
| 4.9 | Unit tests | | | |
| 4.10 | Baseline training (100 ex) | | | |
| 4.11 | Full training | | | |
| 4.12 | 3-config QLoRA comparison | | | |

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
