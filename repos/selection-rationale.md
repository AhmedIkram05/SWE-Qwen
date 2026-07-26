# Selection Rationale — Phase 2 Repository Curation

**Date:** 2026-07-26
**Total Repos Selected:** 14
**Domains:** 5 (web-api, cli, data-ml, testing, utils)

---

## Selection Process

1. Sourced 50 candidates via `scripts/find_candidates.py` using GitHub Search API (5 domain queries)
2. Quick-API checked candidates for license & Python version compatibility
3. Shortlisted 15 promising repos across all domains
4. Deep-verified via `scripts/verify_repos.py` (clone + install + 8 checks)
5. Selected 14 repos meeting all hard criteria (license + Python >=3.10)
6. Expanded beyond original 10 target per user request to maximize coverage

---

## Selected Repositories

### web-api (3 repos)

**fastapi/fastapi** (MIT, 100,898★, 543 .py)
*Domain:* web-api | *Org:* fastapi
The leading Python web framework for building APIs with modern Python type hints. Chosen as the flagship web-api repo for its massive community, excellent test coverage (pytest only), and high release activity. All hard checks pass. Tests run cleanly. FastAPI is permissively licensed under MIT and sees 963 commits in the last 6 months.

**headroomlabs-ai/headroom** (Apache-2.0, 62,544★, 596 .py)
*Domain:* web-api | *Org:* headroomlabs-ai
A token-compression library, proxy, and MCP server. Selected for its exceptional growth rate, active maintenance (2,259 commits/6mo), and unique domain fit as an LLM-facing service. Includes some forbidden deps (openai, boto3, anthropic) per soft check — acceptable since these are optional extras. 596 Python source files provide rich training material.

**wagtail/wagtail** (BSD-3-Clause, 20,419★, 1,298 .py)
*Domain:* web-api | *Org:* wagtail
Industry-standard Django CMS. Chosen for its large codebase (1,298 .py files) with excellent issue-labeling practices. BSD-3-Clause license. Tox detected but primary test runner is pytest. Provides depth in the Django ecosystem.

---

### cli (2 repos)

**Textualize/textual** (MIT, 36,747★, 577 .py)
*Domain:* cli | *Org:* Textualize
The leading Text User Interface framework for Python. Selected for its rich CLI application patterns, active development (241 commits/6mo), and well-structured test suite. MIT licensed. Provides complex async UI patterns that make excellent training data.

**fastapi/typer** (MIT, 19,811★, 339 .py)
*Domain:* cli | *Org:* fastapi
CLI framework built on Python type hints, sibling to FastAPI. Selected for clean API design, active maintenance (437 commits/6mo), and close relationship to FastAPI patterns. 339 source files provide good coverage of modern CLI architecture.

---

### data-ml (3 repos)

**huggingface/datasets** (Apache-2.0, 21,764★, 148 .py)
*Domain:* data-ml | *Org:* huggingface
The largest hub of ready-to-use ML datasets. Selected for its centrality in the ML ecosystem, Apache-2.0 license, and focus on data manipulation tooling. 148 source files (size check soft fail — below 500 threshold, overridden manually). Core ML infrastructure library.

**onnx/onnx** (Apache-2.0, 21,215★, 698 .py)
*Domain:* data-ml | *Org:* onnx
Open standard for ML interoperability. Selected for its well-defined API surface, complex serialization patterns, and diverse test coverage. 698 source files within size range. Apache-2.0 license. Provides important ML standards-adjacent training data.

**mlflow/mlflow** (Apache-2.0, 27,217★, 2,606 .py)
*Domain:* data-ml | *Org:* mlflow
The open-source ML platform for managing the ML lifecycle. Selected for its massive codebase (2,606 .py files, largest in pool) and comprehensive coverage of AI/ML engineering patterns. Includes some forbidden deps (boto3, google-cloud, psycopg2, openai) — acceptable for a platform of this scale. Apache-2.0 license.

---

### testing (2 repos)

**joke2k/faker** (MIT, 19,342★, 752 .py)
*Domain:* testing | *Org:* joke2k
Industry-standard fake data generator. Selected for its well-maintained codebase, excellent test organization, and high utility for training testing patterns. 752 .py files. MIT licensed. Tox detected but pytest-native tests. Active with 197 commits/6mo.

**pytest-dev/pytest** (MIT, 14,375★, 270 .py)
*Domain:* testing | *Org:* pytest-dev
The pytest testing framework itself. Selected as the reference implementation for testing tools. 270 source files with mature codebase patterns. MIT licensed. All hard checks pass. Provides authoritative patterns for test framework design.

---

### utils (4 repos)

**Textualize/rich** (MIT, 56,950★, 146 .py)
*Domain:* utils | *Org:* Textualize
The most popular terminal formatting library for Python. Selected for its beautiful API design and extensive use of Python special methods. 146 source files (below 500 threshold, overridden manually — quality over quantity). Tox detected but pytest primary.

**marimo-team/marimo** (Apache-2.0, 22,064★, 1,509 .py)
*Domain:* utils | *Org:* marimo-team
A reactive Python notebook. Selected for its modern codebase, innovative architecture, and large size (1,509 .py files). Apache-2.0. Includes some forbidden deps (openai, boto3, anthropic, redis) as optional integrations. Active maintenance with 1,286 commits/6mo.

**psf/black** (MIT, 41,764★, 338 .py)
*Domain:* utils | *Org:* psf
The uncompromising Python code formatter. Selected as the reference Python tool. MIT licensed. Well-structured codebase (338 .py files). Tox detected but pytest-compatible. Clean API surface for code transformation patterns.

**pydantic/pydantic** (MIT, 28,401★, 405 .py)
*Domain:* utils | *Org:* pydantic
Data validation using Python type hints. Selected for its deep integration with Python's type system and essential role in the modern Python ecosystem. MIT licensed. 405 source files. Provides rich training material for type-annotation-heavy code.

---

## Domain Distribution Summary

| Domain  | Count | Repos |
|---------|-------|-------|
| web-api | 3     | fastapi, headroom, wagtail |
| cli     | 2     | textual, typer |
| data-ml | 3     | datasets, onnx, mlflow |
| testing | 2     | faker, pytest |
| utils   | 4     | rich, marimo, black, pydantic |

**Total: 14 repos across 5 domains | 493,511 combined stars | 10,225 .py files**
