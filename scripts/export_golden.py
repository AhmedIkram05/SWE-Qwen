"""Export golden.jsonl for the eval harness from the SWE-bench HF cache.

The eval harness reads ``golden_data_path`` (default
``gs://swe-qwen-datasets/datasets/{run_id}/golden.jsonl``), but no run ever
uploaded a golden.jsonl to GCS — the archive only ships the stages that were
non-empty in that run. This script regenerates the golden set (the project
definition: official SWE-bench verified+test+dev records with F2P ground
truth, i.e. ``test_files_changed`` non-empty) and writes it to a local file
so the eval can run without GCS:

    python -m scripts.export_golden --out data/golden.jsonl
    EVAL_GOLDEN_DATA_PATH=$PWD/data/golden.jsonl python -m evaluation.cli run ...

Records are ``IssueRecord.model_dump_json()`` lines — the exact shape
``EvalInput.from_swebench_record`` consumes (metadata.is_verified included).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from data_engineering.config import DataPipelineConfig
from data_engineering.swebench_ingest import (
    REPO_DOMAIN_MAP,
    SWE_BENCH_PYTHON_REPOS,
    load_swebench_splits,
    swebench_to_issue_record,
)
from observability.logging import configure_logging

logger = logging.getLogger(__name__)

GOLDEN_SPLITS = ("verified", "test", "dev")  # all carry FAIL_TO_PASS ground truth


def main() -> None:
    configure_logging(level=logging.INFO)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/golden.jsonl"))
    ap.add_argument("--swe-bench-dir", type=Path, default=Path("data/swe_bench"))
    args = ap.parse_args()

    config = DataPipelineConfig(swe_bench_dir=args.swe_bench_dir, output_dir=Path("data"))
    splits = load_swebench_splits(config)

    records = []
    for split_name in GOLDEN_SPLITS:
        for ex in splits.get(split_name, []):
            repo = ex["repo"]
            if repo not in SWE_BENCH_PYTHON_REPOS or not (ex.get("patch") or "").strip():
                continue
            rec = swebench_to_issue_record(ex, REPO_DOMAIN_MAP.get(repo, "unknown"), split_name)
            if rec.test_files_changed:
                records.append(rec)
        logger.info("%s: %d golden records so far", split_name, len(records))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    print(f"golden.jsonl written: {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()
