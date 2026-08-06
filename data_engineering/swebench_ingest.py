"""SWE-bench ingestion module.

Downloads SWE-bench dataset from Hugging Face, maps to IssueRecord schema,
and provides repo-stratified splits for training/evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from pydantic import BaseModel

from data_engineering.config import DataPipelineConfig
from data_engineering.schema import IssueRecord, ParsedHunk, TestResults

logger = logging.getLogger(__name__)

# ── SWE-bench Python repos (expanded to cover all Python repos in train split) ────────────

SWE_BENCH_PYTHON_REPOS: set[str] = {
    # Original 18 from Verified + Test + Dev
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/black",
    "pytest-dev/pytest",
    "pydantic/pydantic",
    "scipy/scipy",
    "sphinx-doc/sphinx",
    "sympy/sympy",
    "tkinter/tkinter",
    "dask/dask",
    "huggingface/transformers",
    "pallets/jinja",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "tensorflow/tensorflow",
    # Additional Python repos from train split (major ones)
    "numpy/numpy",
    "googleapis/google-cloud-python",
    "pantsbuild/pants",
    "ipython/ipython",
    "pypa/pip",
    "conda/conda",
    "docker/compose",
    "apache/airflow",
    "wagtail/wagtail",
    "PrefectHQ/prefect",
    "Lightning-AI/lightning",
    "pyca/cryptography",
    "ray-project/ray",
    "google/jax",
    "ytdl-org/youtube-dl",
    "celery/celery",
    "jupyterlab/jupyterlab",
    "dagster-io/dagster",
    "open-mmlab/mmdetection",
    "twisted/twisted",
    "gitpython-developers/GitPython",
    "DataDog/integrations-core",
    "tensorflow/models",
    "explosion/spaCy",
    "Qiskit/qiskit",
    "mesonbuild/meson",
    "pypa/setuptools",
    "pypa/virtualenv",
    "ansible/ansible",
    "saltstack/salt",
    "home-assistant/core",
    "psf/requests",
    "psf/urllib3",
    "pallets/click",
    "encode/httpx",
    "encode/starlette",
    "fastapi/fastapi",
    "tiangolo/fastapi",
    "pytest-dev/pluggy",
    "pytest-dev/pytest-asyncio",
    "pytest-dev/pytest-mock",
    "tox-dev/tox",
    "pypa/pipenv",
    "pypa/poetry",
    "astral-sh/ruff",
    "astral-sh/uv",
    "python-poetry/poetry",
    "pydantic/pydantic-core",
    "encode/databases",
    "encode/orm",
    "sqlalchemy/sqlalchemy",
    "pallets/werkzeug",
    "pallets/itsdangerous",
    "pallets/markupsafe",
    "psf/typing-extensions",
    "python/typing_extensions",
    # Dev-split repos (6, from princeton-nlp/SWE-bench dev split)
    "marshmallow-code/marshmallow",
    "pvlib/pvlib-python",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}

# Repo → domain mapping for curriculum/domain-aware training
REPO_DOMAIN_MAP: dict[str, str] = {
    "astropy/astropy": "data-ml",
    "django/django": "web-api",
    "matplotlib/matplotlib": "data-ml",
    "mwaskom/seaborn": "data-ml",
    "pallets/flask": "web-api",
    "psf/black": "utils",
    "pytest-dev/pytest": "testing",
    "pydantic/pydantic": "utils",
    "scipy/scipy": "data-ml",
    "sphinx-doc/sphinx": "utils",
    "sympy/sympy": "data-ml",
    "tkinter/tkinter": "utils",
    "dask/dask": "data-ml",
    "huggingface/transformers": "data-ml",
    "pallets/jinja": "web-api",
    "pandas-dev/pandas": "data-ml",
    "scikit-learn/scikit-learn": "data-ml",
    "tensorflow/tensorflow": "data-ml",
    # Additional repos
    "numpy/numpy": "data-ml",
    "googleapis/google-cloud-python": "web-api",
    "pantsbuild/pants": "utils",
    "ipython/ipython": "utils",
    "pypa/pip": "utils",
    "conda/conda": "utils",
    "docker/compose": "utils",
    "apache/airflow": "web-api",
    "wagtail/wagtail": "web-api",
    "PrefectHQ/prefect": "utils",
    "Lightning-AI/lightning": "data-ml",
    "pyca/cryptography": "utils",
    "ray-project/ray": "data-ml",
    "google/jax": "data-ml",
    "ytdl-org/youtube-dl": "utils",
    "celery/celery": "web-api",
    "jupyterlab/jupyterlab": "utils",
    "dagster-io/dagster": "utils",
    "open-mmlab/mmdetection": "data-ml",
    "twisted/twisted": "web-api",
    "gitpython-developers/GitPython": "utils",
    "DataDog/integrations-core": "utils",
    "tensorflow/models": "data-ml",
    "explosion/spaCy": "data-ml",
    "Qiskit/qiskit": "data-ml",
    "mesonbuild/meson": "utils",
    "pypa/setuptools": "utils",
    "pypa/virtualenv": "utils",
    "ansible/ansible": "utils",
    "saltstack/salt": "utils",
    "home-assistant/core": "web-api",
    "psf/requests": "web-api",
    "psf/urllib3": "web-api",
    "pallets/click": "utils",
    "encode/httpx": "web-api",
    "encode/starlette": "web-api",
    "fastapi/fastapi": "web-api",
    "tiangolo/fastapi": "web-api",
    "pytest-dev/pluggy": "testing",
    "pytest-dev/pytest-asyncio": "testing",
    "pytest-dev/pytest-mock": "testing",
    "tox-dev/tox": "testing",
    "pypa/pipenv": "utils",
    "pypa/poetry": "utils",
    "astral-sh/ruff": "utils",
    "astral-sh/uv": "utils",
    "python-poetry/poetry": "utils",
    "pydantic/pydantic-core": "utils",
    "encode/databases": "web-api",
    "encode/orm": "web-api",
    "sqlalchemy/sqlalchemy": "web-api",
    "pallets/werkzeug": "web-api",
    "pallets/itsdangerous": "utils",
    "pallets/markupsafe": "utils",
    "psf/typing-extensions": "utils",
    "python/typing_extensions": "utils",
    # Dev-split repos
    "marshmallow-code/marshmallow": "web-api",
    "pvlib/pvlib-python": "data-ml",
    "pydicom/pydicom": "data-ml",
    "pylint-dev/astroid": "utils",
    "pyvista/pyvista": "data-ml",
    "sqlfluff/sqlfluff": "utils",
}


class SWEBenchConfig(BaseModel):
    """Configuration for SWE-bench dataset loading."""

    verified_path: str = "SWE-bench/SWE-bench_Verified"
    test_path: str = "SWE-bench/SWE-bench"
    train_path: str = "SWE-bench/SWE-bench"
    version: str = "2025-04-29"
    cache_dir: Path | None = None


def _parse_test_list(test_str: str) -> list[str]:
    """Parse FAIL_TO_PASS / PASS_TO_PASS string into list of test names."""
    if not test_str:
        return []
    return [t.strip() for t in test_str.split() if t.strip()]


def _parse_files_from_patch(patch: str) -> list[str]:
    """Extract file paths from a unified diff patch."""
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[6:].strip()
            if path not in files:
                files.append(path)
    return files


def _parse_unified_diff(diff_str: str) -> list[ParsedHunk]:
    """Parse a unified diff string into a list of ParsedHunk.

    Tries unidiff first; falls back to empty list on parse failure
    (some SWE-bench patches have formatting quirks).
    """
    from unidiff.patch import PatchSet

    try:
        patch_set = PatchSet(diff_str)
        hunks: list[ParsedHunk] = []
        for patched_file in patch_set:
            for hunk in patched_file:
                hunks.append(
                    ParsedHunk(
                        file=patched_file.path,
                        old_start=hunk.source_start,
                        old_lines=hunk.source_length,
                        new_start=hunk.target_start,
                        new_lines=hunk.target_length,
                        diff_lines=[str(line) for line in hunk],
                    )
                )
        return hunks  # noqa: TRY300 -- early return deliberate; except path returns []
    except Exception:
        logger.warning("unidiff parse failed, returning empty hunks")
        return []


def load_swebench_splits(config: DataPipelineConfig) -> dict[str, list[dict[str, Any]]]:
    """Load SWE-bench splits from Hugging Face.

    Returns dict with keys: "verified", "test", "dev", "train"
    Train split is filtered to Python repos only (~12K examples).
    """
    sb_config = SWEBenchConfig(version=config.swe_bench_version, cache_dir=config.swe_bench_dir)

    logger.info("Loading SWE-bench Verified (test split)...")
    verified = load_dataset(
        sb_config.verified_path,
        split="test",
        cache_dir=str(sb_config.cache_dir) if sb_config.cache_dir else None,
    )

    logger.info("Loading SWE-bench Test (test split)...")
    test = load_dataset(
        sb_config.test_path,
        split="test",
        cache_dir=str(sb_config.cache_dir) if sb_config.cache_dir else None,
    )

    logger.info("Loading SWE-bench Dev (dev split)...")
    dev = load_dataset(
        sb_config.test_path,
        split="dev",
        cache_dir=str(sb_config.cache_dir) if sb_config.cache_dir else None,
    )

    logger.info("Loading SWE-bench Train (train split, filtering Python repos)...")
    train = load_dataset(
        sb_config.train_path,
        split="train",
        cache_dir=str(sb_config.cache_dir) if sb_config.cache_dir else None,
    )

    # Filter train to Python repos only
    train_python = train.filter(lambda x: x["repo"] in SWE_BENCH_PYTHON_REPOS)

    logger.info(
        "SWE-bench loaded: verified=%d, test=%d, dev=%d, train_python=%d",
        len(verified),
        len(test),
        len(dev),
        len(train_python),
    )

    return {
        "verified": [dict(r) for r in verified],
        "test": [dict(r) for r in test],
        "dev": [dict(r) for r in dev],
        "train": [dict(r) for r in train_python],
    }


def swebench_to_issue_record(
    example: dict[str, Any], repo_domain: str, source_split: str = ""
) -> IssueRecord:
    """Map a SWE-bench example to IssueRecord schema.

    SWE-bench fields:
    - instance_id → issue_id
    - repo → repo
    - problem_statement → issue_body
    - patch → patch_diff
    - test_patch → test_files_changed (parsed)
    - FAIL_TO_PASS → test_results.failed
    - PASS_TO_PASS → test_results.passed
    - base_commit → metadata.base_sha
    - environment_setup_commit → metadata.head_sha
    - version → metadata.version
    - hints_text → metadata.hints
    - created_at → metadata.created_at
    """
    has_test_patch = bool(example.get("FAIL_TO_PASS"))

    # Parse test files from test_patch
    test_files = _parse_files_from_patch(example.get("test_patch", ""))

    # Parse all files from patch
    all_files = _parse_files_from_patch(example.get("patch", ""))

    hints = example.get("hints_text") or ""
    return IssueRecord(
        issue_id=example["instance_id"],
        repo=example["repo"],
        issue_body=example["problem_statement"],
        patch_diff=example["patch"],
        parsed_hunks=_parse_unified_diff(example["patch"]),
        test_results=TestResults(
            failed=_parse_test_list(example.get("FAIL_TO_PASS", "")),
            passed=_parse_test_list(example.get("PASS_TO_PASS", "")),
            errored=[],
        ),
        pr_title="",  # not in SWE-bench
        pr_description="",  # not in SWE-bench
        commit_messages=[],  # not in SWE-bench
        files_changed=all_files,
        test_files_changed=test_files,
        issue_labels=[],  # not in SWE-bench
        repo_domain=repo_domain,
        metadata={
            "base_sha": example["base_commit"],
            "head_sha": example["environment_setup_commit"],
            "test_patch": example.get("test_patch", ""),
            "source_split": source_split,
            "version": example["version"],
            "hints": hints,
            "created_at": example["created_at"],
            "has_test_patch": has_test_patch,
            # "Verified" = official SWE-bench Verified split, NOT "has ground
            # truth" (every F2P split has ground truth). Conflating them let
            # test/dev instances through the eval's --split swebench_verified
            # filter and sample instances with no official eval image.
            "is_verified": source_split == "verified",
            "instance_id": example["instance_id"],
        },
    )


def ingest_swebench(config: DataPipelineConfig) -> list[IssueRecord]:
    """Main ingestion function: load SWE-bench, map to IssueRecords.

    Returns list of IssueRecord for all splits combined.
    Caller is responsible for splitting (train/val/test/golden).
    """
    splits = load_swebench_splits(config)
    records: list[IssueRecord] = []
    skipped_empty_patch = 0

    for split_name, examples in splits.items():
        logger.info("Processing SWE-bench split: %s (%d examples)", split_name, len(examples))
        for ex in examples:
            repo = ex["repo"]
            if repo not in SWE_BENCH_PYTHON_REPOS:
                continue
            # Skip examples with empty patches (some train examples lack patches)
            if not ex.get("patch") or not ex["patch"].strip():
                skipped_empty_patch += 1
                continue
            repo_domain = REPO_DOMAIN_MAP.get(repo, "unknown")
            record = swebench_to_issue_record(ex, repo_domain, split_name)
            records.append(record)

    if skipped_empty_patch:
        logger.warning("Skipped %d examples with empty patches", skipped_empty_patch)
    logger.info("Total SWE-bench records ingested: %d", len(records))
    return records


def save_swebench_splits(
    records: list[IssueRecord],
    output_dir: Path,
    config: DataPipelineConfig,
) -> dict[str, Path]:
    """Save records as JSONL split files for checkpoint/resume."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # Group by split source (we need to track which split each record came from)
    # For simplicity, save all raw records; splitting happens in split.py
    raw_path = output_dir / "swebench_raw.jsonl"
    with raw_path.open("w") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
    paths["raw"] = raw_path

    logger.info("Saved SWE-bench raw records to %s", raw_path)
    return paths


# ── BigQuery augmentation ────────────────────────────────────────────────────

_BQ_STATS_CACHE_FILE = "bigquery_repo_stats.json"


def _get_bq_stats_cache_path(config: DataPipelineConfig) -> Path:
    """Get path to BigQuery stats cache file."""
    return config.swe_bench_dir / _BQ_STATS_CACHE_FILE


def _load_bq_stats_cache(cache_path: Path) -> dict[str, dict]:
    """Load BigQuery stats cache from JSON file."""
    if not cache_path.exists():
        return {}
    import json

    with cache_path.open() as f:
        return cast(dict[str, dict], json.load(f))


def _save_bq_stats_cache(cache_path: Path, data: dict[str, dict]) -> None:
    """Save BigQuery stats cache to JSON file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with cache_path.open("w") as f:
        json.dump(data, f, indent=2)


def _fetch_repo_stats(config: DataPipelineConfig, repos: set[str]) -> dict[str, dict]:
    """Fetch repository metadata from BigQuery.

    Uses sample_repos table (no full repositories table in github_repos dataset).
    Returns basic stats; full stars/forks require GitHub Archive dataset.
    """
    from google.cloud import bigquery

    client = bigquery.Client(project=config.bigquery_project)

    repo_list = ", ".join(f"'{r}'" for r in sorted(repos))
    query = f"""
    SELECT
      s.repo_name,
      s.watch_count as stars,
      l.license
    FROM `bigquery-public-data.github_repos.sample_repos` s
    LEFT JOIN `bigquery-public-data.github_repos.licenses` l
      ON s.repo_name = l.repo_name
    WHERE s.repo_name IN ({repo_list})
    """

    logger.info("Querying BigQuery for repo stats...")
    rows = client.query(query).result()

    stats: dict[str, dict] = {}
    for row in rows:
        stats[row.repo_name] = {
            "stars": row.stars or 0,
            "forks": 0,
            "contributors": 0,
            "primary_language": "Python",
            "license": row.license or "",
        }

    logger.info("Fetched stats for %d repos", len(stats))
    return stats


def augment_with_bigquery(
    records: list[IssueRecord], config: DataPipelineConfig
) -> list[IssueRecord]:
    """Augment records with BigQuery repo stats only.

    Queries (cached):
    1. Repository metadata (stars, license) from sample_repos + licenses tables
    Note: Commit history unavailable - github_repos.commits has fork names only,
    sample_commits only covers 6 large repos (linux, swift, etc.)

    Attaches to metadata:
    - bigquery_repo_stats: stars, forks, contributors, primary_language, license
    """
    if not config.bigquery_enabled:
        logger.info("BigQuery augmentation disabled (--no-bigquery)")
        return records

    # Collect unique repos from records
    repos = {r.repo for r in records}
    logger.info("BigQuery augmentation for %d repos", len(repos))

    stats_cache_path = _get_bq_stats_cache_path(config)

    # Try loading from cache first
    cached_stats = _load_bq_stats_cache(stats_cache_path)

    # Determine what we need to fetch
    missing_stats = repos - set(cached_stats.keys())

    fetched_stats = {}

    if missing_stats:
        try:
            fetched_stats = _fetch_repo_stats(config, missing_stats)

            # Merge with cache
            all_stats = {**cached_stats, **fetched_stats}

            # Save updated cache
            _save_bq_stats_cache(stats_cache_path, all_stats)

            cached_stats = all_stats

        except Exception as exc:
            logger.warning("BigQuery query failed: %s. Using cache only.", exc)
            if not cached_stats:
                logger.warning("No cache available, skipping BigQuery augmentation")
                return records

    # Attach to records
    for record in records:
        repo = record.repo
        if repo in cached_stats:
            record.metadata["bigquery_repo_stats"] = cached_stats[repo]

    logger.info("BigQuery augmentation attached to %d records", len(records))
    return records
