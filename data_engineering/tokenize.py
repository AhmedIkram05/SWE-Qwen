"""Tokenize Phase 3 JSONL splits into HuggingFace ``.arrow`` shards.

Phase 3 produces JSONL files (``train.jsonl``, ``val.jsonl``, ``test.jsonl``,
``golden.jsonl``). This module reads them, renders training prompts via
``PromptLoader``, tokenizes with the model's tokenizer, and saves as
HuggingFace ``DatasetDict`` with ``.arrow`` shards ready for ``SFTTrainer``.

Supports saving to local disk and/or GCS.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, cast

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer

from data_engineering.config import DataPipelineConfig
from training.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)

# Max errors to log before suppressing
_MAX_LOG_ERRORS = 5

# Column names in the output dataset
INPUT_IDS_COL = "input_ids"
ATTENTION_MASK_COL = "attention_mask"
LABELS_COL = "labels"

# Expected JSONL split files
SPLIT_FILES = {
    "train": "train.jsonl",
    "val": "val.jsonl",
    "test": "test.jsonl",
    "golden": "golden.jsonl",
}


def _save_dataset_to_gcs(
    ds: DatasetDict,
    bucket_name: str,
    prefix: str,
) -> dict[str, str]:
    """Save DatasetDict to GCS as parquet/arrow shards."""
    try:
        from google.cloud import storage

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        if not bucket.exists():
            logger.warning(f"GCS bucket {bucket_name} does not exist, skipping GCS save")
            return {}

        gcs_paths = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Save locally first (to arrow format)
            ds.save_to_disk(str(tmp_path))

            # Upload all files
            for file_path in Path(tmpdir).rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(tmp_path)
                    key = f"tokenized/{prefix}/{rel_path}"
                    blob = bucket.blob(key)
                    blob.upload_from_filename(str(file_path))
                    logger.info(f"Uploaded {key} to GCS")

            prefix = f"tokenized/{bucket_name}/{prefix}"
            gcs_paths["dataset"] = f"gs://{bucket_name}/tokenized/{prefix}"

    except Exception as e:
        logging.warning(f"GCS save failed (non-fatal): {e}")
        return {}
    else:
        return gcs_paths


def load_jsonl_split(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    records: list[dict[str, Any]] = []
    with path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line:
                records.append(json.loads(line))
    return records


def format_training_prompt(record: dict[str, Any], prompt_loader: PromptLoader) -> str:
    """Format a record into a training prompt string.

    Uses the ``chat.j2`` template with issue body as the user message.
    """
    issue_id = record.get("issue_id", "unknown")
    issue_body = record.get("issue_body", "")
    repo = record.get("repo", "unknown")
    repo_domain = record.get("repo_domain", "unknown")
    patch_diff = record.get("patch_diff", "")
    files_changed: list[str] = record.get("files_changed", [])
    test_files: list[str] = record.get("test_files_changed", [])

    # Build user message
    user_content = prompt_loader.render(
        "user",
        issue_title=issue_id,
        issue_body=issue_body,
        repo_name=repo,
        repo_domain=repo_domain,
        context_files=files_changed[:20],  # cap context files
        test_files=test_files[:10],
    )

    # Build assistant response (the patch)
    assistant_content = patch_diff

    return prompt_loader.render_chat(
        system_prompt=prompt_loader.render(
            "system",
            task_description="Fix the bug described in the issue by generating a correct patch.",
            language="Python",
            style_guide="Follow PEP 8 and the repository's existing code style.",
        ),
        messages=[
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
    )


def tokenize_split(
    records: list[dict[str, Any]],
    tokenizer: Any,
    prompt_loader: PromptLoader,
    max_length: int,
    split_name: str = "train",
) -> Dataset:
    """Tokenize a split of records into a HuggingFace Dataset.

    For causal LM training with SFTTrainer, we need to mask the prompt
    portion of labels to -100 so loss is only computed on the assistant response.

    Args:
        records: List of issue records as dicts.
        tokenizer: HF tokenizer for the model.
        prompt_loader: For rendering prompts.
        max_length: Maximum sequence length (from ``models.yaml.context_window``).
        split_name: Split name for logging.

    Returns:
        HuggingFace Dataset with ``input_ids``, ``attention_mask``, ``labels``.
    """
    texts: list[str] = []
    prompt_ends: list[int] = []  # token index where prompt ends (response starts)
    errors = 0

    for i, rec in enumerate(records):
        try:
            text = format_training_prompt(rec, prompt_loader)
            texts.append(text)

            # Find the boundary between prompt and response
            # The chat template uses "### Response" as the delimiter
            # We'll tokenize the prompt part separately to find its length
            issue_id = rec.get("issue_id", "unknown")
            issue_body = rec.get("issue_body", "")
            repo = rec.get("repo", "unknown")
            repo_domain = rec.get("repo_domain", "unknown")
            files_changed: list[str] = rec.get("files_changed", [])
            test_files: list[str] = rec.get("test_files_changed", [])

            user_content = prompt_loader.render(
                "user",
                issue_title=issue_id,
                issue_body=issue_body,
                repo_name=repo,
                repo_domain=repo_domain,
                context_files=files_changed[:20],
                test_files=test_files[:10],
            )

            prompt_only = prompt_loader.render_chat(
                system_prompt=prompt_loader.render(
                    "system",
                    task_description="Fix the bug described in the issue"
                    " by generating a correct patch.",
                    language="Python",
                    style_guide="Follow PEP 8 and the repository's existing code style.",
                ),
                messages=[{"role": "user", "content": user_content}],
            )
            # Tokenize prompt to find its length
            prompt_tokens = tokenizer(prompt_only, add_special_tokens=False)["input_ids"]
            prompt_ends.append(len(prompt_tokens))

        except Exception as exc:
            errors += 1
            if errors <= _MAX_LOG_ERRORS:
                logger.warning(
                    "Failed to format record %s (%s): %s",
                    rec.get("issue_id", f"index-{i}"),
                    split_name,
                    exc,
                )
            continue

    if errors:
        logger.warning(
            "Tokenization: %d/%d records failed formatting in split '%s'",
            errors,
            len(records),
            split_name,
        )

    if not texts:
        logger.error("No valid texts for split '%s'", split_name)
        return Dataset.from_list([])

    # Tokenize all texts
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,  # return python lists
    )

    # Build labels with prompt masking (-100 for prompt tokens)
    labels = []
    for i, input_ids in enumerate(tokenized["input_ids"]):
        prompt_end = prompt_ends[i] if i < len(prompt_ends) else 0
        # Mask prompt tokens with -100
        label = [-100] * prompt_end + input_ids[prompt_end:]
        # Ensure same length
        if len(label) != len(input_ids):
            label = label[: len(input_ids)] + [-100] * max(0, len(input_ids) - len(label))
        labels.append(label)

    # Build dataset dict
    data = {
        INPUT_IDS_COL: tokenized["input_ids"],
        ATTENTION_MASK_COL: tokenized["attention_mask"],
        LABELS_COL: labels,
    }

    dataset = Dataset.from_dict(data)
    logger.info(
        "Tokenized '%s' split: %d records → %d examples (max_length=%d, errors=%d)",
        split_name,
        len(records),
        len(dataset),
        max_length,
        errors,
    )
    return dataset


def tokenize_pipeline(  # noqa: PLR0913, PLR0917
    data_dir: str | Path,
    output_dir: str | Path,
    model_name: str = "qwen3-30b-a3b",
    max_length: int | None = None,
    prompt_template_dir: str | Path | None = None,
    config: DataPipelineConfig | None = None,
    run_id: str | None = None,
) -> DatasetDict:
    """Run the full tokenization pipeline: JSONL → .arrow shards.

    Args:
        data_dir: Directory containing JSONL split files
            (``train.jsonl``, ``val.jsonl``, ``test.jsonl``, ``golden.jsonl``).
        output_dir: Where to save the ``DatasetDict``.
        model_name: Key from ``models.yaml`` used to select tokenizer.
        max_length: Override max sequence length. If ``None``, reads from
            ``models.yaml`` context_window for the given model.
        prompt_template_dir: Override prompt template directory.

    Returns:
        The saved ``DatasetDict``.
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)

    # Resolve tokenizer and max_length
    from training.qlora_config import _get_model_config

    model_cfg = _get_model_config(model_name)
    hf_id: str = model_cfg["hf_id"]
    ctx_length = max_length or int(model_cfg.get("context_window", 32768))

    logger.info(
        "Tokenizing for model %s (hf_id=%s, max_length=%d)",
        model_name,
        hf_id,
        ctx_length,
    )

    # Load tokenizer (uses fast tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load prompt loader
    prompt_loader = PromptLoader(
        template_dir=prompt_template_dir,
    )

    # Process each split
    dataset_dict: dict[str, Dataset] = {}
    for split_name, filename in SPLIT_FILES.items():
        split_path = data_path / filename
        if not split_path.exists():
            logger.warning("Split file not found, skipping: %s", split_path)
            continue

        records = load_jsonl_split(split_path)
        logger.info("Loaded %d records from %s", len(records), split_path)

        ds = tokenize_split(
            records,
            tokenizer,
            prompt_loader,
            max_length=ctx_length,
            split_name=split_name,
        )
        if len(ds) > 0:
            dataset_dict[split_name] = ds

    if not dataset_dict:
        raise ValueError(
            f"No tokenized datasets produced from {data_path}. "
            f"Expected JSONL files: {list(SPLIT_FILES.values())}"
        )

    # Build DatasetDict
    full_ds = DatasetDict(dataset_dict)  # type: ignore[operator]

    # Save to disk
    out_path.mkdir(parents=True, exist_ok=True)
    full_ds.save_to_disk(str(out_path))

    logger.info(
        "Saved tokenized DatasetDict to %s with splits: %s",
        out_path,
        {k: len(v) for k, v in full_ds.items()},
    )

    # Upload to GCS if configured
    if config and config.gcs_bucket:
        gcs_paths = _save_dataset_to_gcs(full_ds, config.gcs_bucket, run_id or "tokenized")
        logger.info("Uploaded tokenized data to GCS: %s", gcs_paths)

    return full_ds


def load_tokenized_shards(
    data_dir: str | Path,
) -> DatasetDict:
    """Load pre-tokenized ``DatasetDict`` from disk.

    Args:
        data_dir: Path where ``save_to_disk()`` was called.

    Returns:
        The loaded ``DatasetDict``.
    """
    ds = load_from_disk(str(data_dir))
    ds = cast(DatasetDict, ds)
    logger.info(
        "Loaded tokenized DatasetDict from %s: %s",
        data_dir,
        {k: len(v) for k, v in ds.items()},
    )
    return ds


def tokenize_dataset(
    run_id: str | None = None,
    model_name: str = "qwen3-14b",
    max_seq_length: int = 8192,
    config: DataPipelineConfig | None = None,
) -> dict[str, Any]:
    """Tokenize a completed dataset run.

    This is a CLI-friendly wrapper around ``tokenize_pipeline``.

    Args:
        run_id: The run ID of the dataset to tokenize. If None, uses the latest run.
        model_name: Model name from models.yaml (default: qwen3-14b).
        max_seq_length: Maximum sequence length for tokenization.
        config: Optional pipeline config for GCS upload.

    Returns:
        Dict with tokenization results and output path.
    """
    from data_engineering.config import DataPipelineConfig

    # Use provided config or create default
    if config is None:
        config = DataPipelineConfig()

    # Determine run_id
    if run_id is None:
        # Find the latest run directory in output_dir
        runs = sorted(config.output_dir.glob("*/"))
        if not runs:
            raise ValueError("No run directories found in output_dir")
        run_id = runs[-1].name
        logger.info(f"Using latest run: {run_id}")

    data_dir = config.output_dir / run_id / "swebench"
    output_dir = config.output_dir / f"{run_id}_tokenized"

    if not data_dir.exists():
        raise ValueError(f"Run directory not found: {data_dir}")

    logger.info(f"Tokenizing run {run_id} for model {model_name}")

    ds = tokenize_pipeline(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=model_name,
        max_length=8192,  # Use context window from model config in qlora_config
        config=config,
        run_id=run_id,
    )

    return {
        "run_id": run_id,
        "model_name": model_name,
        "output_dir": str(output_dir),
        "splits": {k: len(v) for k, v in ds.items()},
        "total_examples": sum(len(v) for v in ds.values()),
    }
