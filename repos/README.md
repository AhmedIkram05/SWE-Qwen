# Repository Collection

This directory contains the curated set of 14 open-source Python repositories selected for training SWE-Qwen. These repos span 5 domains and provide real-world Issue-PR pairs for fine-tuning a code-generation model.

## Selection Criteria

See [docs/planning/phase2-criteria.md](../docs/planning/phase2-criteria.md) for the full criteria document.

**Hard Requirements:**
- License: MIT, Apache-2.0, or BSD-3-Clause
- Python >= 3.10
- Minimum 300 GitHub stars
- At least 10 commits in the last 6 months
- Latest release within the last year
- Uses pytest (no unittest/nose-only)
- No external service dependencies (DBs, cloud APIs)
- 500–5000 Python source files

## Manifest

| # | Repo | Domain | License | Stars | Python Files | Description |
|---|------|--------|---------|-------|-------------|-------------|
| 1 | **[fastapi/fastapi](https://github.com/fastapi/fastapi)** | web-api | MIT | 100,898 | 543 | High-performance web framework |
| 2 | **[headroomlabs-ai/headroom](https://github.com/headroomlabs-ai/headroom)** | web-api | Apache-2.0 | 62,544 | 596 | LLM token compression |
| 3 | **[wagtail/wagtail](https://github.com/wagtail/wagtail)** | web-api | BSD-3-Clause | 20,419 | 1,298 | Django CMS |
| 4 | **[Textualize/rich](https://github.com/Textualize/rich)** | utils | MIT | 56,950 | 146 | Terminal formatting |
| 5 | **[marimo-team/marimo](https://github.com/marimo-team/marimo)** | utils | Apache-2.0 | 22,064 | 1,509 | Reactive notebooks |
| 6 | **[psf/black](https://github.com/psf/black)** | utils | MIT | 41,764 | 338 | Code formatter |
| 7 | **[pydantic/pydantic](https://github.com/pydantic/pydantic)** | utils | MIT | 28,401 | 405 | Data validation |
| 8 | **[Textualize/textual](https://github.com/Textualize/textual)** | cli | MIT | 36,747 | 577 | TUI framework |
| 9 | **[fastapi/typer](https://github.com/fastapi/typer)** | cli | MIT | 19,811 | 339 | CLI builder |
| 10 | **[onnx/onnx](https://github.com/onnx/onnx)** | data-ml | Apache-2.0 | 21,215 | 698 | ML interoperability |
| 11 | **[huggingface/datasets](https://github.com/huggingface/datasets)** | data-ml | Apache-2.0 | 21,764 | 148 | ML datasets hub |
| 12 | **[mlflow/mlflow](https://github.com/mlflow/mlflow)** | data-ml | Apache-2.0 | 27,217 | 2,606 | ML engineering platform |
| 13 | **[joke2k/faker](https://github.com/joke2k/faker)** | testing | MIT | 19,342 | 752 | Fake data generation |
| 14 | **[pytest-dev/pytest](https://github.com/pytest-dev/pytest)** | testing | MIT | 14,375 | 270 | Python testing framework |

**Totals:** 14 repos, 5 domains, 493,511 stars, 10,225 Python files

## Domain Distribution

| Domain | Count | Repos |
|--------|-------|-------|
| web-api | 3 | fastapi, headroom, wagtail |
| utils | 4 | rich, marimo, black, pydantic |
| cli | 2 | textual, typer |
| data-ml | 3 | datasets, onnx, mlflow |
| testing | 2 | faker, pytest |

## Verification Status

All 14 repos passed hard checks:
- **License:** MIT, Apache-2.0, or BSD-3-Clause ✓
- **Python version:** >=3.10 ✓
- **Recent release:** within the last year ✓
- **Active development:** >=10 commits in 6 months ✓
- **Test framework:** pytest ✓
- **No service dependencies:** no DB/cloud API requirements ✓
- **Install success:** pip install -e . passed ✓

## Structure

```
repos/
├── README.md           # This file
├── manifest.json       # Machine-readable manifest with verification data
├── fastapi-fastapi/    # Cloned during Phase 3
├── headroomlabs-ai-headroom/
└── ...
```

Each repo will be cloned into its own directory during the Phase 3 ingestion pipeline.
