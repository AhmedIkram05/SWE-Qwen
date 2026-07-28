"""GCS archival for durable dataset storage.

Uploads all pipeline stages + manifest + dataset card to GCS under
``gs://{bucket}/datasets/{run_id}/``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from data_engineering.config import DataPipelineConfig

logger = logging.getLogger(__name__)


def _ensure_gcs_bucket(bucket_name: str) -> Any:
    """Get or create a GCS bucket and return it."""
    from google.cloud import storage  # type: ignore[attr-defined]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    if not bucket.exists():
        logger.info("GCS bucket '%s' does not exist — creating", bucket_name)
        bucket = client.create_bucket(bucket_name, location="US-CENTRAL1")
    return bucket


def _upload_jsonl(
    bucket: Any,
    prefix: str,
    name: str,
    records: list[Any],
) -> str:
    """Upload records as JSONL to GCS, return the full gs:// path."""
    import hashlib

    key = f"{prefix}/{name}.jsonl"
    blob = bucket.blob(key)

    lines = [json.dumps(r.model_dump(), default=str) + "\n" for r in records]
    content = "".join(lines)

    blob.upload_from_string(content, content_type="application/jsonl")

    gs_path = f"gs://{bucket.name}/{key}"
    content_hash = hashlib.md5(content.encode()).hexdigest()
    logger.debug("Uploaded %s (%d records, md5=%s)", gs_path, len(records), content_hash)
    return gs_path


def _upload_text(bucket: Any, prefix: str, name: str, content: str) -> str:
    """Upload a plain-text file to GCS."""
    key = f"{prefix}/{name}"
    blob = bucket.blob(key)
    blob.upload_from_string(content, content_type="text/markdown; charset=utf-8")
    gs_path = f"gs://{bucket.name}/{key}"
    logger.debug("Uploaded %s (%d bytes)", gs_path, len(content))
    return gs_path


def upload_to_gcs(
    run_id: str,
    stages: dict[str, Any],
    manifest: dict[str, Any],
    dataset_card: str,
    config: DataPipelineConfig,
) -> dict[str, str]:
    """Upload all pipeline stages + manifest + dataset card to GCS.

    Args:
        run_id: Unique run ID used as the GCS prefix.
        stages: Mapping of stage name → records list.
        manifest: The loaded manifest dict (uploaded as JSON).
        dataset_card: Dataset card markdown string (may be empty placeholder).
        config: Pipeline config.

    Returns:
        Mapping of stage name → ``gs://`` URL.
    """
    if not config.gcs_bucket:
        logger.warning("No GCS bucket configured — skipping archival")
        return {}

    bucket = _ensure_gcs_bucket(config.gcs_bucket)
    prefix = f"datasets/{run_id}"
    gcs_paths: dict[str, str] = {}

    # Upload each stage
    for stage_name, records in stages.items():
        if records:
            gcs_paths[stage_name] = _upload_jsonl(bucket, prefix, stage_name, records)

    # Upload manifest
    manifest_json = json.dumps(manifest, default=str, indent=2)
    _upload_text(bucket, prefix, "manifest.json", manifest_json)
    gcs_paths["manifest"] = f"gs://{bucket.name}/{prefix}/manifest.json"

    # Upload dataset card (may be placeholder, will be overwritten after full generation)
    _upload_text(bucket, prefix, "dataset_card.md", dataset_card)
    gcs_paths["dataset_card"] = f"gs://{bucket.name}/{prefix}/dataset_card.md"

    logger.info(
        "GCS upload complete: %d files under gs://%s/%s",
        len(gcs_paths),
        config.gcs_bucket,
        prefix,
    )
    return gcs_paths


def upload_text_to_gcs(
    bucket_name: str,
    prefix: str,
    name: str,
    content: str,
) -> str:
    """Upload a text file to GCS, returning the ``gs://`` path.

    Useful for overwriting a previously uploaded file (e.g. dataset card
    after W&B/GCS paths are known).
    """
    from google.cloud import storage  # type: ignore[attr-defined]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    key = f"{prefix}/{name}"
    blob = bucket.blob(key)
    blob.upload_from_string(content, content_type="text/markdown; charset=utf-8")
    gs_path = f"gs://{bucket.name}/{key}"
    logger.debug("Uploaded %s (%d bytes)", gs_path, len(content))
    return gs_path
