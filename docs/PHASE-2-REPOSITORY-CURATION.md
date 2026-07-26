# Phase 2: Repository Curation — Implementation Plan

**Document Type:** Phase Plan (Level 4 in project hierarchy)
**Status:** Final v1.1 (reviewed and patched)
**Parent Document:** `docs/planning/MASTER-PLAN.md`
**Dependencies:** Phase 1 complete (repo structure, Terraform scaffold, Modal, W&B configured)

---

## 1. Objective

Select, document, and prepare **10+ Python repositories** that will serve as the data source for the training dataset. Each repository is chosen for:
- Test quality (pytest, runs cleanly in isolation)
- Issue clarity (bug/defect/fix labels)
- Permissive license (MIT, Apache-2.0, BSD-3-Clause)
- Clear issue→PR linkage (GitHub `Fixes #`/`Closes #`/`Resolves #` pattern)
- Domain diversity (web/API, CLI, data/ML, utils, testing)

**Output:** `repos/manifest.json` — the single source of truth for Phase 3 Data Pipeline ingestion.

---

## 2. Selection Criteria (Locked)

| Criterion | Threshold | Verification Method |
|-----------|-----------|---------------------|
| **Language** | Pure Python 3.10+ (for candidates; our project runs 3.11+) | `pyproject.toml` / `setup.py` `requires-python` |
| **License** | MIT \| Apache-2.0 \| BSD-3-Clause only | GitHub License API (SPDX ID) |
| **Activity** | Release ≤ 365 days ago AND ≥ 10 commits in last 6 months | GitHub Releases API + Commits API |
| **Test Framework** | pytest only, no tox, no external services | Scan config files + `requirements*.txt` for forbidden deps (psycopg2, redis, pymongo, sqlalchemy[asyncpg], boto3, google-cloud-*, anthropic, openai) |
| **Issue-PR Linkage** | ≥ 35% of last 30 merged PRs reference issues via GitHub-recognized keywords (fix/fixed/fixes, close/closed/closes, resolve/resolved/resolves) in title, body, or commit messages | GitHub PRs API + Issues API |
| **Repo Size** | 500–5000 Python files (excl. tests, .venv, .git) | `find . -name "*.py" -not -path "./.venv/*" -not -path "./tests/*" -not -path "./test/*" -not -path "./.git/*" \| wc -l` |
| **Tests Run Clean** | `pip install -e . && pytest -x` exits 0 in ≤ 180s | Actual execution in temp venv |

**Note on Python version:** The candidate threshold is `>=3.10` to be as permissive as possible during sourcing. Our project itself runs on `>=3.11` (per `pyproject.toml`). This is intentional — we want diverse candidates, not to restrict ourselves.

---

## 3. Domain Diversity Requirements (Enforced at Selection)

| Domain Bucket | Minimum Repos | Description |
|---------------|---------------|-------------|
| **web-api** | 2 | FastAPI, Django, Flask, Starlette, aiohttp applications |
| **cli** | 2 | Click, Typer, argparse-based command-line tools |
| **data-ml** | 2 | Pandas, NumPy, scikit-learn, Polars, DuckDB data/ML libraries |
| **utils** | 2 | General-purpose utility libraries, helpers, toolkits |
| **testing** | 2 | Testing frameworks, linting, formatting, developer tools |

**Additional constraint:** Maximum 2 repositories per GitHub organization.

**If a bucket cannot be filled:** Relax to min 1 per bucket. Document which bucket is short and why in `selection-rationale.md`. Then fill remaining slots from next-best candidates regardless of domain.

---

## 4. Sourcing Strategy

### 4.1 GitHub Search Queries (5 queries, one per domain)

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

### 4.2 Excluded Organizations (Framework-centric, less application code)

`pallets`, `pydantic`, `tiangolo`, `django`, `psf`, `scipy`, `numpy`, `pandas`, `matplotlib`, `sphinx`

### 4.3 Candidate Pool Sizing

- **Pull 50 candidates** total across all queries
- **Manually review** to 20–30 verified candidates
- **Select final 10** meeting all criteria + domain spread

### 4.4 GitHub Authentication

All scripts use `pygithub` (already in project deps) authenticated via `GITHUB_TOKEN` environment variable. The token must have `repo` and `public_repo` scopes. If `GITHUB_TOKEN` is unset, scripts fail with a clear error message.

---

## 5. Implementation Tasks

### Task 2.0: Script Environment Setup
- **Action:** Ensure all scripts can run in the project's Python environment
- **Details:**
  - Scripts live under `scripts/` and use project deps (`pygithub`, `rich`, `pydantic`, `pyyaml`) — no additional `scripts/requirements.txt` needed
  - Auth via `GITHUB_TOKEN` env var (validated at script start; fail fast if missing)
  - All scripts use `argparse` for CLI args (not raw sys.argv)
  - All scripts produce structured output via `rich.console.Console` for progress/table display
  - All scripts write their primary output (JSON) to stdout or specified `--output` path
- **Owner:** Human
- **Estimate:** 15 min

### Task 2.1: Define Selection Criteria Document
- **File:** `docs/planning/phase2-criteria.md`
- **Content:** This criteria table + rationale
- **Owner:** Human
- **Estimate:** 30 min

### Task 2.2: Build Candidate Sourcing Script
- **File:** `scripts/find_candidates.py`
- **Function:** Execute 5 GitHub Search API queries via `pygithub`, apply filters (stars≥300, pushed>2025-01-01, archived=false, exclude orgs), deduplicate, output top 50
- **Deduplication strategy:**
  - Dedup by `owner/name` (canonical, not URL)
  - When same repo matches multiple queries, assign domain based on which query had the most topic overlap (fallback: first query match)
  - Domain assignment recorded in `candidates.json` for manual override in Task 2.4
- **Output schema (`candidates.json`):**
  ```json
  [
    {
      "owner": "owner",
      "name": "repo",
      "url": "https://github.com/owner/repo",
      "stars": 1234,
      "description": "One-liner from GitHub",
      "topics": ["fastapi", "api", "async"],
      "license": "MIT",
      "pushed_at": "2026-06-15T...",
      "domain": "web-api",
      "matched_query": "fastapi OR django OR flask OR starlette OR aiohttp"
    }
  ]
  ```
- **Logging:** Print domain distribution summary table after completion
- **Owner:** Human (script provided)
- **Estimate:** 2 hours

### Task 2.3: Build Validation Script
- **File:** `scripts/verify_repos.py`
- **Function:** Read `candidates.json`, for each repo:
  1. Shallow clone (`--depth=1`) to temp dir managed via `tempfile.mkdtemp()` at `/tmp/swe-qwen-verify-{uuid}/`
  2. Query GitHub API for `default_branch` (capture actual branch, don't assume `main`)
  3. Store the actual value in `default_branch`
  4. Run 9 checks in parallel (4 workers):
     - License (GitHub API) — **hard fail**
     - Python version (parse pyproject.toml) — **hard fail**
     - Recent release (GitHub API)
     - Commit activity (GitHub API, last 6 months)
     - pytest only (scan configs)
     - No external services (grep requirements for forbidden deps)
     - Issue-PR linkage (sample 30 PRs via GitHub API; check if linked issues have *any* label matching configured `issue_labels_to_include`, not just exact `bug`/`defect`/`fix`)
     - Size (count .py files using exact command from section 2)
     - Tests run (`pip install -e . && pytest -x` with `-m "not slow and not requires_modal and not requires_gcp"` to skip tests needing external resources; timeout 180s)
   5. Use `sys.executable -m pytest` (not bare `pytest`) to avoid picking up the host project's venv
   6. After all checks complete, capture the *actual* test command that worked (e.g., `python3 -m pytest -x -m "not slow"`) — store in `test_command_actual`
   7. Delete temp clone directory after verification
  7. Print remaining GitHub API rate limit after completion
  8. Collect ALL results (don't stop on first failure)
  9. Output structured `verified.json`
- **Hard fail** vs **Soft fail:**
  - **Hard fail** (license, Python version): `overall_pass` is `false` regardless of other checks
  - **Soft fail** (all others): logged with reason, but `overall_pass` can still be `true` (human decides in Task 2.4)
- **Failure handling:**
  - `pip install -e .` failure: capture stderr, log first 500 chars of error in `check_details`
  - `pytest -x` failure: capture stdout+stderr, log last 50 lines in `check_details`
  - GitHub API failure (rate limit, network): retry once with 5s backoff, then mark check as `"error"` with message
- **Output schema (`verified.json`):**
  ```json
  {
    "verified_at": "2026-07-25T...",
    "github_rate_limit_remaining": 4250,
    "repos": [
      {
        "url": "https://github.com/owner/repo",
        "overall_pass": true,
        "default_branch": "main",
        "test_command_actual": "pytest -x -m \"not slow\"",
        "install_success": true,
        "install_error_snippet": null,
        "checks": {
          "license":           {"passed": true,  "value": "MIT",                          "hard_fail": true},
          "python_version":    {"passed": true,  "value": ">=3.10",                      "hard_fail": true},
          "recent_release":    {"passed": true,  "value": "2026-03-15",                  "hard_fail": false},
          "commit_activity":   {"passed": true,  "value": 47,                            "hard_fail": false},
          "pytest_only":       {"passed": true,  "value": "pytest only",                 "hard_fail": false},
          "no_services":       {"passed": true,  "value": [],                            "hard_fail": false},
          "issue_pr_linkage":  {"passed": true,  "value": 0.83,                          "hard_fail": false},
          "size":              {"passed": true,  "value": 1234,                           "hard_fail": false},
          "tests_run":         {"passed": true,  "value": "12 passed in 45.2s",           "hard_fail": false}
        }
      }
    ],
    "summary": {
      "total": 25,
      "passed_all": 18,
      "failed_hard": 3,
      "failed_soft": 4
    }
  }
  ```
- **Logging:** `rich` table showing per-repo check results during execution. Final summary table printed.
- **Owner:** Human (script provided)
- **Estimate:** 3 hours

### Task 2.4: Manual Selection & Rationale
- **Input:** `verified.json` and `candidates.json`
- **Process (step-by-step):**
  1. Filter `verified.json` to repos where `overall_pass == true`
  2. Group by `domain`. Sort each group by: stars (desc) → issue-PR linkage ratio (desc) → commit activity (desc)
  3. Select top 2 per domain, ensuring ≤2 per org total
  4. If any domain has <2 qualifying repos:
     - Check if any candidates from *other* domains would fit the underrepresented bucket (based on GitHub topics/description)
     - If no, re-run `find_candidates.py` with lowered star threshold (200) for underrepresented bucket only
     - If still insufficient, document in rationale and fill from next-best candidate regardless of domain
  5. Review the selected 10 and confirm no org limit violation
  6. Write rationale
- **Output:** `repos/selection-rationale.md` — one paragraph per repo explaining:
  - Why this repo was chosen
  - Key domain and org
  - Any notable checks behavior (e.g., "tests run but needed `-m 'not slow'`")
  - How it contributes to dataset diversity
- **Owner:** Human
- **Estimate:** 1 hour

### Task 2.5: Build Manifest Script
- **File:** `scripts/build_manifest.py`
- **Function:** Read `verified.json` + selected repo list (provided as CLI arg or JSON input), enrich each with `ingestion_config`, validate output against Pydantic model (defined inline or in `data_engineering/schema.py`), write `repos/manifest.json`
- **Pydantic validation:** The script defines a Pydantic `Manifest` model covering the full schema below. Output is validated before writing. If validation fails, script prints errors and exits non-zero.
- **Output:** `repos/manifest.json` (schema below)
- **Logging:** Print summary: "Manifest written with 10 repos, 5 domains, N orgs"
- **Owner:** Human (script provided)
- **Estimate:** 1 hour

### Task 2.6: Write Per-Repo Documentation
- **File:** `repos/README.md`
- **Content:** Table with all 10 repos: name, URL, domain category, stars, py_file_count, test_command, one-line why. Include a cross-reference: "See `selection-rationale.md` for detailed rationale per repo."
- **Owner:** Human
- **Estimate:** 30 min

### Task 2.7: Final Verification Run
- **Action:** Re-run `verify_repos.py` against final 10 repos (from manifest)
- **Output:** Save verification log to `repos/verification-log.txt` (plain text capture of the `rich` output table). Acceptance: all 10 repos show `overall_pass: true`.
- **Owner:** Human
- **Estimate:** 15 min

### Task 2.8: Write Tests
- **File:** `tests/test_phase2.py`
- **Mock strategy:** Use `pytest-mock` for GitHub API calls (mock `pygithub` client responses) and `subprocess.run` (for clone/test execution). Use `tmp_path` fixture for filesystem operations. Use `json.loads` to verify output file structure. Do NOT use VCR recording — too fragile for Phase 2's unique data.
- **Test structure:**
  ```python
  # tests/test_phase2.py

  class TestFindCandidates:
      """scripts/find_candidates.py"""
      def test_deduplicates_same_repo_across_queries(self, mocker):
          """Two queries returning same repo → one entry in output"""
      def test_excludes_orgs_in_blocklist(self, mocker):
          """Query returns repo from excluded org → filtered out"""
      def test_outputs_50_candidates_json(self, mocker):
          """Mock 5 queries returning 10 each → 50 entries in output"""
      def test_domain_assignment_fallback(self, mocker):
          """Repo matching 2 queries → assigned to most-overlapping-topic query"""

  class TestVerifyRepos:
      """scripts/verify_repos.py"""
      def test_all_checks_pass(self, mocker, tmp_path):
          """All 9 checks pass → overall_pass=true"""
      def test_license_hard_fail(self, mocker, tmp_path):
          """GPL license → overall_pass=false, check_details has reason"""
      def test_python_version_hard_fail(self, mocker, tmp_path):
          """Python <3.10 → overall_pass=false"""
      def test_test_run_timeout(self, mocker, tmp_path):
          """pytest hangs → timeout, check has error, overall_pass unaffected (soft)"""
      def test_collects_all_results_without_stopping(self, mocker, tmp_path):
          """Mixed pass/fail → all 9 checks present in output"""
      def test_temp_dir_cleaned_after_run(self, mocker, tmp_path):
          """Cloned repo deleted after verification completes"""
      def test_detects_default_branch(self, mocker, tmp_path):
          """Repo with 'master' default branch → captured in output"""

  class TestBuildManifest:
      """scripts/build_manifest.py"""
      def test_generates_valid_manifest(self, mocker, tmp_path):
          """Output validates against Pydantic Manifest model"""
      def test_ingestion_config_applied_per_repo(self, mocker, tmp_path):
          """exclude_paths, test_directories copied to each entry"""
      def test_default_branch_from_verified_json(self, mocker, tmp_path):
          """Repos with 'master' branch → manifest has 'master'"""
      def test_exit_nonzero_on_pydantic_validation_fail(self, mocker, tmp_path):
          """Malformed input → script exits non-zero"""
  ```
- **Note:** Tests mock the GitHub API and subprocess calls — they do NOT make real API calls or clone repos. This keeps tests fast and offline.
- **Owner:** Human
- **Estimate:** 1.5 hours

---

## 6. Manifest Schema (`repos/manifest.json`)

```json
{
  "version": "1.0",
  "created_at": "2026-07-25T...",
  "selection_criteria": {
    "license": ["MIT", "Apache-2.0", "BSD-3-Clause"],
    "python_version": ">=3.10",
    "min_stars": 300,
    "min_commits_6mo": 10,
    "max_age_days": 365,
    "test_framework": "pytest",
    "issue_pr_linkage_min_ratio": 0.35,
    "size_range_py_files": [500, 5000]
  },
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
      "verification": {
        "verified_at": "2026-07-25T...",
        "all_checks_passed": true,
        "check_details": { ... }
      },
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
  "summary": {
    "total_repos": 14,
    "by_domain": {"web-api": 2, "cli": 2, "data-ml": 2, "utils": 2, "testing": 2},
    "total_py_files": 12340,
    "total_stars": 15000,
    "avg_issue_pr_linkage": 0.78
  }
}
```

### Field Notes

| Field | Purpose | Semantics / Notes | Used By |
|-------|---------|-------------------|---------|
| `ingestion_config.exclude_paths` | Glob patterns to skip during patch extraction | Uses `pathlib.PurePath.match()` (Python stdlib). Directory entries like `docs/` match any path starting with that prefix (recursive). Glob entries like `*.md` match against the final filename component only. Phase 3 resolves all paths relative to repo root, stripping leading `./`. | Phase 3 `ingest.py` |
| `ingestion_config.test_directories` | Where pytest discovers tests | Relative to repo root. Phase 5 uses these to run test suites in isolation. | Phase 5 `evaluation/test_runner.py` |
| `ingestion_config.max_issues_per_repo` | Cap to prevent runaway ingestion on huge repos | Phase 3 stops fetching after this count (by `updated_at` order, most recent first). | Phase 3 `ingest.py` |
| `ingestion_config.pr_merge_commits_only` | Only ingest merged PRs | Only PRs with `merged_at` populated. Unmerged/closed PRs skipped — they don't have a diff. | Phase 3 `ingest.py` |
| `ingestion_config.default_branch` | Actual default branch of the repo | Detected via GitHub API (`GET /repos/{owner}/{name}`). Populated by `verify_repos.py`. Do NOT hardcode `main`. | Phase 3 `ingest.py` |
| `ingestion_config.issue_labels_to_include` | Issue labels to filter on | Per-repo configurable. Phase 3 matches issues where any label name from this list appears (case-insensitive substring match). Default: `["bug", "defect", "fix"]`. | Phase 3 `ingest.py` |
| `test_command_actual` | The *actual* test command that passed | May differ from default `pytest -x` if repo needed `-m "not slow"` or `--ignore`. Phase 5 uses this for evaluation. | Phase 5 `evaluation/test_runner.py` |
| `install_success` | Whether `pip install -e .` succeeded | If `false`, repo needs special install steps. Phase 3/5 handles specially. | Phase 3 `ingest.py`, Phase 5 `test_runner.py` |

---

## 7. File Structure After Phase 2

```
swe-qwen/
├── docs/
│   ├── MASTER-PLAN.md
│   ├── PHASE-2-REPOSITORY-CURATION.md    # THIS FILE
│   └── phase2-criteria.md                # Task 2.1 output
├── repos/
│   ├── manifest.json                     # Task 2.5 output (Phase 3 input)
│   ├── README.md                         # Task 2.6 output
│   ├── selection-rationale.md            # Task 2.4 output
│   └── verification-log.txt              # Task 2.7 output
├── scripts/
│   ├── find_candidates.py                # Task 2.2
│   ├── verify_repos.py                   # Task 2.3
│   └── build_manifest.py                 # Task 2.5
├── tests/
│   └── test_phase2.py                    # Task 2.8
└── ... (Phase 1 files)
```

---

## 8. Task Dependencies & Ordering

```
2.0 → 2.1 → 2.2 → 2.3 → 2.4 ──→ 2.5 ──→ 2.7
                              │         │
                              └──→ 2.6  │
                                        ↓
                                      2.8
```

- **2.0** prerequisites: project venv is set up, `GITHUB_TOKEN` is available
- **2.1** can be done immediately (documentation)
- **2.2** depends on 2.1 + 2.0 (criteria inform query filters; env must work)
- **2.3** depends on 2.2 (needs candidate list)
- **2.4** depends on 2.3 (needs verification results)
- **2.5** depends on 2.4 (needs final selection list) AND creates `manifest.json`
- **2.6** depends on 2.4 (needs rationale content), independent of 2.5
- **2.7** depends on 2.5 (runs against final `manifest.json` repos, uses `verify_repos.py` from 2.3)
- **2.8** can run anytime after scripts exist (2.2, 2.3, 2.5 are written)

---

## 9. Acceptance Criteria (Phase Exit Gate)

Phase 2 is **complete** when ALL are true:

- [ ] `repos/manifest.json` exists, valid JSON (validated against Pydantic model), contains ≥10 repositories
- [ ] All repos: license ∈ {MIT, Apache-2.0, BSD-3-Clause} ([automated check via manifest schema])
- [ ] All repos: Python ≥ 3.10, pytest-only test framework, no external service dependencies
- [ ] All repos: `pip install -e . && pytest -x` passes in ≤ 180 seconds on clean environment
- [ ] Domain spread achieved: 2 web-api, 2 cli, 2 data-ml, 2 utils, 2 testing (or documented exception)
- [ ] Maximum 2 repositories per GitHub organization
- [ ] Issue-PR linkage ratio ≥ 0.35 for each repository (lowered from 0.7 after fixing regex to match all GitHub-recognized keyword forms + commit message scanning)
- [ ] `repos/selection-rationale.md` documents selection rationale for all repos
- [ ] `repos/verification-log.txt` exists — all repos show `overall_pass: true`
- [ ] `tests/test_phase2.py` passes (all unit tests green, > 12 test cases covering 3 script classes)
- [ ] Default branch detected correctly for each repo (not hardcoded to `main`)

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| < 10 repos pass all checks | Medium | High | Candidate pool = 50 (not 25); can lower stars to 200 for underrepresented domains only |
| Domain spread impossible with passed repos | Low | Medium | Relax to min 1 per bucket; document in rationale |
| GitHub API rate limit during sourcing | Low | Medium | Authenticated `pygithub` (5k req/hr); 50 repos × ~15 calls = 750 req; track and print remaining after each run |
| Test suites fail on clean env despite passing in CI | Medium | High | Phase 2.3 catches this; retry once with `pip install -e ".[dev,test]"` if base fails; only verified repos enter manifest |
| Shallow clone misses history for commit count | None | N/A | Commit activity fetched via GitHub API, not local git |
| Issue label mismatch (repo uses `kind/bug` not `bug`) | Medium | Low | `issue_labels_to_include` is per-repo configurable; Task 2.3 samples and human corrects in selection |
| Temp disk space exhausted by 50 clones | Low | Low | Each clone cleaned after verify; max ~500MB at any time |
| Pydantic model changes before manifest write | Low | Low | `build_manifest.py` validates inline; CI catches drift |

---

## 11. Definition of Done

1. All 9 tasks completed and deliverables present in repository
2. All acceptance criteria verified (11 checkboxes)
3. `pytest tests/test_phase2.py` passes (≥ 12 test cases)
4. `repos/manifest.json` is the **only** artifact required by Phase 3 — no manual handoff needed
5. All repos verified: `overall_pass: true` in final verification log
6. No hard-coded `default_branch` assumptions — each repo's actual branch recorded

---

## 12. Next Phase Dependency

**Phase 3 (Data Pipeline Engine)** consumes `repos/manifest.json` directly. Its `ingest.py` will:
1. Read `manifest.json`
2. For each repo, use `ingestion_config` to configure GitHub API ingestion:
   - `default_branch` → branch to fetch files from
   - `issue_labels_to_include` → filter issues by label
   - `pr_merge_commits_only` → only process merged PRs
   - `max_issues_per_repo` → cap on fetch count
   - `exclude_paths` → `pathlib.PurePath.match()` patterns to skip during diff extraction
   - `test_directories` → used later by Phase 5 evaluation harness
3. Extract issues with matching labels, resolve linked merged PRs, extract diffs
4. Apply `exclude_paths` when extracting file patches
5. Use `install_command` + `test_command_actual` from manifest for Phase 5 test execution

No additional coordination needed — the manifest is the complete contract referenced from `ingestion_config`.
