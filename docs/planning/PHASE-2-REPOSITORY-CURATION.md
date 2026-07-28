# Phase 2: Repository Curation — Implementation Plan (DEPRECATED)

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Superseded by SWE-bench + BigQuery pivot (see ADR-004)
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 1 complete

---

## ⚠️ DEPRECATED — Superseded by SWE-bench Strategy

**This phase is no longer executed.** The original approach (manual GitHub API curation of 10 repos) was abandoned in favor of **SWE-bench + BigQuery** which provides:

| Split | Python Examples | Execution-Verifiable | Test Patches |
|-------|-----------------|---------------------|--------------|
| **Verified** (golden eval) | 500 | ✅ 500 | ✅ 500 |
| **Test** | 2,294 | ✅ 2,294 | ✅ 2,294 |
| **Dev** | 225 | ✅ 225 | ✅ 225 |
| **Train** | 7,863 | ❌ 0 (no test patches) | ❌ 0 |

**Total execution-verifiable: 3,019** across 18 Python repos (12 Verified + 6 Test)
**Total training data: ~10,882** (Verified + Test + Dev + Train Python)

This delivers 10k+ records in **hours** instead of **weeks** of GitHub API debugging.

---

## Original Plan (Archived for Reference)

### 1. Objective (Original)

Select, document, and prepare **10+ Python repositories** for Phase 3 Data Pipeline ingestion.

**Output:** `repos/manifest.json` — single source of truth for Phase 3.

---

### 2. Selection Criteria (Locked - Original)

| Criterion | Threshold | Verification Method |
|-----------|-----------|---------------------|
| **Language** | Pure Python 3.10+ | `pyproject.toml` / `setup.py` `requires-python` |
| **License** | MIT \| Apache-2.0 \| BSD-3-Clause only | GitHub License API (SPDX ID) |
| **Activity** | Release ≤ 365 days ago AND ≥ 10 commits in last 6 months | GitHub Releases API + Commits API |
| **Test Framework** | pytest only, no tox, no external services | Scan config files + `requirements*.txt` for forbidden deps |
| **Issue-PR Linkage** | ≥ 35% of last 30 merged PRs reference issues | GitHub PRs API + Issues API |
| **Repo Size** | 500–5000 Python files (excl. tests) | `find . -name "*.py" ... \| wc -l` |
| **Tests Run Clean** | `pip install -e . && pytest -x` exits 0 in ≤ 180s | Actual execution in temp venv |

---

### 3. Domain Diversity Requirements (Enforced at Selection)

| Domain Bucket | Minimum Repos | Description |
|---------------|---------------|-------------|
| **web-api** | 2 | FastAPI, Django, Flask, Starlette, aiohttp applications |
| **cli** | 2 | Click, Typer, argparse-based command-line tools |
| **data-ml** | 2 | Pandas, NumPy, scikit-learn, Polars, DuckDB data/ML libraries |
| **utils** | 2 | General-purpose utility libraries, helpers, toolkits |
| **testing** | 2 | Testing frameworks, linting, formatting, developer tools |

**Additional constraint:** Maximum 2 repositories per GitHub organization.

---

### 4. Sourcing Strategy (Original)

#### 4.1 GitHub Search Queries (5 queries, one per domain)

```python
queries = [
    # web-api
    'language:python stars:>300 pushed:>2025-01-01 archived:false '
    'topic:fastapi OR topic:django OR topic:flask OR topic:starlette OR topic:aiohttp',
    # cli
    'language:python stars:>300 pushed:>2025-01-01 archived:false '
    'topic:cli OR topic:click OR topic:typer OR topic:argparse',
    # data-ml
    'language:python stars:>300 pushed:>2025-01-01 archived:false '
    'topic:pandas OR topic:numpy OR topic:scikit-learn OR topic:polars OR topic:duckdb',
    # utils
    'language:python stars:>300 pushed:>2025-01-01 archived:false '
    'topic:utility OR topic:helpers OR topic:toolkit',
    # testing
    'language:python stars:>300 pushed:>2025-01-01 archived:false '
    'topic:testing OR topic:pytest OR topic:linting OR topic:formatting',
]
```

#### 4.2 Excluded Organizations

`pallets`, `pydantic`, `tiangolo`, `django`, `psf`, `scipy`, `numpy`, `pandas`, `matplotlib`, `sphinx`

#### 4.3 Candidate Pool Sizing

- **Pull 50 candidates** total across all queries
- **Manually review** to 20–30 verified candidates
- **Select final 10** meeting all criteria + domain spread

---

### 5. Implementation Tasks (Archived - Not Executed)

All tasks below were planned but **not executed** due to pivot:

- **Task 2.0**: Script Environment Setup
- **Task 2.1**: Define Selection Criteria Document → `docs/planning/phase2-criteria.md`
- **Task 2.2**: Build Candidate Sourcing Script → `scripts/find_candidates.py`
- **Task 2.3**: Build Validation Script → `scripts/verify_repos.py`
- **Task 2.4**: Manual Selection & Rationale → `repos/selection-rationale.md`
- **Task 2.5**: Build Manifest Script → `scripts/build_manifest.py` + `repos/manifest.json`
- **Task 2.6**: Write Per-Repo Documentation → `repos/README.md`
- **Task 2.7**: Final Verification Run → `repos/verification-log.txt`
- **Task 2.8**: Write Tests → `tests/test_phase2.py`

---

### 6. Manifest Schema (Original - Not Used)

```json
{
  "version": "1.0",
  "created_at": "2026-07-25T...",
  "selection_criteria": { ... },
  "repositories": [
    {
      "id": "owner-repo",
      "url": "https://github.com/owner/repo",
      "owner": "owner",
      "name": "repo",
      "description": "One-liner from GitHub",
      "license": "MIT",
      "primary_language": "Python",
      "stars": 1234,
      "forks": 56,
      "open_issues": 23,
      "latest_release": "2026-03-15",
      "commits_last_6mo": 47,
      "python_version": ">=3.10",
      "test_command": "pytest -x",
      "test_command_actual": "pytest -x -m \"not slow\"",
      "install_command": "pip install -e .",
      "install_success": true,
      "py_file_count": 1234,
      "domain_category": "web-api",
      "issue_pr_linkage_ratio": 0.83,
      "verification": { ... },
      "ingestion_config": {
        "default_branch": "main",
        "issue_labels_to_include": ["bug", "defect", "fix"],
        "pr_merge_commits_only": true,
        "max_issues_per_repo": 2000,
        "exclude_paths": ["docs/", "examples/", "benchmarks/", "scripts/", "*.md", "*.rst"],
        "test_directories": ["tests/", "test/"]
      }
    }
  ],
  "summary": { ... }
}
```

---

### 7. File Structure (Original Plan - Not Created)

```
swe-qwen/
├── docs/
│   ├── MASTER-PLAN.md
│   ├── PHASE-2-REPOSITORY-CURATION.md
│   └── phase2-criteria.md
├── repos/
│   ├── manifest.json
│   ├── README.md
│   ├── selection-rationale.md
│   └── verification-log.txt
├── scripts/
│   ├── find_candidates.py
│   ├── verify_repos.py
│   └── build_manifest.py
├── tests/
│   └── test_phase2.py
└── ... (Phase 1 files)
```

---

### 8. Acceptance Criteria (Original - Not Met)

Phase 2 was complete when:
- [ ] `repos/manifest.json` exists, valid JSON, contains ≥10 repositories
- [ ] All repos: license ∈ {MIT, Apache-2.0, BSD-3-Clause}
- [ ] All repos: Python ≥ 3.10, pytest-only, no external service dependencies
- [ ] All repos: `pip install -e . && pytest -x` passes in ≤ 180s
- [ ] Domain spread: 2 web-api, 2 cli, 2 data-ml, 2 utils, 2 testing
- [ ] Maximum 2 repos per GitHub organization
- [ ] Issue-PR linkage ratio ≥ 0.3 for each repository
- [ ] `repos/selection-rationale.md` documents all repos
- [ ] `repos/verification-log.txt` exists — all repos `overall_pass: true`
- [ ] `tests/test_phase2.py` passes (≥ 12 test cases)
- [ ] Default branch detected correctly for each repo

---

### 9. Why This Was Deprecated

| Factor | GitHub API Approach | SWE-bench + BigQuery |
|--------|---------------------|----------------------|
| **Time to 10k records** | Weeks (rate limits, debugging) | Hours (dataset download) |
| **Execution-verifiable examples** | ~400 (2% yield from 20k issues) | 3,019 (golden + test + dev) |
| **Training examples** | ~400 (same) | 10,882 (includes train split) |
| **Test patches available** | Manual extraction, error-prone | Pre-extracted, verified |
| **F2P ground truth** | Heuristic only (Phase 5 to verify) | Built-in (tests run at base/head) |
| **Maintenance** | Ongoing API breakage | Static dataset versions |
| **Cost** | Compute + API time | One-time download |

**Decision recorded in:** `docs/adr/ADR-004-data-source-selection.md`

---

### 10. Replacement: Phase 2B — SWE-bench Ingestion

**New Phase 2 objective:** Ingest and prepare SWE-bench dataset for Phase 3 training pipeline.

| Task | Description | Output |
|------|-------------|--------|
| **2B.1** | Download SWE-bench (Verified, Test, Dev, Train) from HF | Local parquet/jsonl files |
| **2B.2** | Filter to Python examples only | Python-subset datasets |
| **2B.3** | Extract test patches + issue context for all splits | Structured records |
| **2B.4** | Query BigQuery for additional training context (commits, files) | Enriched records |
| **2B.5** | Create train/val/test/golden splits (repo-stratified) | `data/swe_bench/{split}.jsonl` |
| **2B.6** | Validate schema compatibility with Phase 3 `IssueRecord` | Schema mapping verified |
| **2B.7** | Version in W&B, archive to GCS | Artifacts + dataset card |

**Estimated time:** 2-4 hours (vs 2-3 weeks for original Phase 2)

---

### 11. SWE-bench Data Structure

```python
# SWE-bench example (from HF datasets)
{
    "instance_id": "django__django-12345",
    "repo": "django/django",
    "base_commit": "abc123...",
    "problem_statement": "Issue description...",
    "patch": "unified diff...",
    "test_patch": "test file changes...",
    "version": "4.2",
    "environment_setup_commit": "def456...",
    "FAIL_TO_PASS": ["test_module.TestClass.test_method"],
    "PASS_TO_PASS": ["test_module.TestClass.test_other"],
    "created_at": "2024-01-15T...",
    "hints_text": "Optional hints...",
}
```

**Key fields mapping to Phase 3 `IssueRecord`:**
- `instance_id` → `issue_id`
- `repo` → `repo`
- `problem_statement` → `issue_body`
- `patch` → `patch_diff`
- `test_patch` → `test_files_changed` (parsed)
- `FAIL_TO_PASS` → `test_results.failed` (before fix)
- `PASS_TO_PASS` → `test_results.passed` (before fix)
- `base_commit` → `metadata.base_sha`
- `environment_setup_commit` → `metadata.head_sha` (after fix)

---

### 12. BigQuery Augmentation

Query GitHub Archive / BigQuery public datasets for:
- Full commit history per repo (context beyond SWE-bench instances)
- File-level change patterns
- Additional issue-PR pairs not in SWE-bench
- Repository metadata (stars, forks, contributors)

```sql
-- Example: Get all commits for SWE-bench repos
SELECT repo_name, commit_sha, author, message, files_changed
FROM `bigquery-public-data.github_repos.commits`
WHERE repo_name IN (SELECT DISTINCT repo FROM swe_bench_instances)
  AND commit_date > '2023-01-01'
```

---

### 13. Next Phase Dependency

**Phase 3 (Data Pipeline Engine)** now consumes:
1. `data/swe_bench/train.jsonl` — ~10,882 training examples
2. `data/swe_bench/val.jsonl` — validation split (from Test/Dev)
3. `data/swe_bench/test.jsonl` — test split (from Test)
4. `data/swe_bench/golden.jsonl` — 3,019 execution-verifiable (Verified + Test + Dev)

No manual manifest needed — SWE-bench provides the complete contract.

---

## References

- ADR-004: `docs/adr/ADR-004-data-source-selection.md`
- SWE-bench paper: https://arxiv.org/abs/2310.06770
- SWE-bench HF: https://huggingface.co/datasets/SWE-bench/SWE-bench
- BigQuery GitHub public datasets: https://console.cloud.google.com/marketplace/product/github/github-archive
