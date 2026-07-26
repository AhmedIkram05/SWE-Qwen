# Phase 2: Repository Selection Criteria

**Document Type:** Selection Criteria (Task 2.1)
**Status:** Final
**Phase Plan Reference:** `docs/PHASE-2-REPOSITORY-CURATION.md` §2

---

## 1. Mandatory Criteria (Hard Gates)

A repository MUST satisfy ALL of these to enter the final manifest:

| # | Criterion | Threshold | Verification |
|---|-----------|-----------|-------------|
| 1 | **Language** | Pure Python 3.10+ | Parse `pyproject.toml` / `setup.py` `requires-python` |
| 2 | **License** | MIT, Apache-2.0, or BSD-3-Clause only | GitHub License API (SPDX ID) |
| 3 | **Activity** | Release ≤ 365 days ago **AND** ≥ 10 commits in last 6 months | GitHub Releases API + Commits API |
| 4 | **Test Framework** | pytest only — no tox, no external service dependencies | Scan config files; forbid `psycopg2`, `redis`, `pymongo`, `boto3`, `google-cloud-*`, `anthropic`, `openai` |
| 5 | **Issue–PR Linkage** | ≥ 70% of last 30 merged PRs reference issues via `Fixes #`/`Closes #`/`Resolves #` | GitHub PRs API + Issues API |
| 6 | **Repo Size** | 500–5000 Python files (excl. tests, `.venv`, `.git`) | `find . -name '*.py' -not -path './.venv/*' -not -path './tests/*' -not -path './.git/*' \| wc -l` |
| 7 | **Tests Run Clean** | `pip install -e . && pytest -x` exits 0 in ≤ 180s | Actual execution in temp venv |

## 2. Soft Criteria (Guides Selection, Not Gates)

| # | Criterion | Notes |
|---|-----------|-------|
| 8 | **Domain diversity** | Spread across web-api, cli, data-ml, utils, testing (2 per bucket) |
| 9 | **Org diversity** | Maximum 2 repos per GitHub organization |
| 10 | **Documentation quality** | Clear README, issue templates, contribution guide |
| 11 | **Active maintenance** | Responsive to issues/PRs (qualitative assessment) |

## 3. Domain Buckets

| Bucket | Min Repos | Example Frameworks/Libraries |
|--------|-----------|------------------------------|
| **web-api** | 2 | FastAPI, Django, Flask, Starlette, aiohttp |
| **cli** | 2 | Click, Typer, argparse-based tools |
| **data-ml** | 2 | Pandas, NumPy, scikit-learn, Polars, DuckDB |
| **utils** | 2 | General-purpose utility libraries |
| **testing** | 2 | Testing frameworks, linting, developer tools |

If a bucket cannot be filled: relax to min 1 per bucket, document why in
`selection-rationale.md`, then fill remaining slots from next-best candidates.

## 4. Excluded Organizations

Repos from these orgs are excluded (framework-centric, less application code):

`pallets`, `pydantic`, `tiangolo`, `django`, `psf`, `scipy`, `numpy`,
`pandas`, `matplotlib`, `sphinx`

## 5. Candidate Sourcing Parameters

- **Queries:** 5 GitHub search queries (one per domain)
- **Filters:** `stars:>300`, `pushed:>2025-01-01`, `archived:false`
- **Pool size:** 50 candidates across all queries
- **Review:** Narrow to 20–30 verified candidates
- **Final:** 10 repos meeting all criteria + domain spread

---

*Reference: Phase 2 plan — Selection Criteria (§2), Domain Diversity (§3), Sourcing Strategy (§4)*
