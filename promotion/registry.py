"""Source-of-truth champion registry (Phase 9, task 9.3, spec §4.3).

The champion-of-record is a JSON file (spec decision 5): in production
``gs://swe-qwen-datasets/ci/champion.json`` (mirror of the Phase 7
``smoke_baseline.json`` pattern — reads public, writes require GCP WIF auth),
in dev/tests any local path.  The W&B ``champion`` alias is a derived,
human-facing pointer synced from this record — never a second source of
truth.

Schema (spec §4.3, exact field order — stable, not alphabetized):

    {
      "variant": "higher_lr_14b",
      "model_ref": "qwen3-14b:higher_lr_14b",
      "f2p_rate": 0.169,
      "p2p_rate": 0.912,
      "dataset_run_id": "expanded-repos",
      "tier": "full",
      "seed": 42,
      "promoted_at": "2026-08-06T12:00:00+00:00",
      "previous": {... or null}
    }

``previous`` is self-referential (spec decision 7): rollback re-promotes the
record's own predecessor through the same pipeline — no extra state.

Only ``evaluation.comparison`` is composed (lazy ``wandb`` import inside
:func:`sync_alias` — repo convention: offline tests pass without cloud).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from evaluation import comparison
from evaluation.config import EvalConfig

__all__ = [
    "ChampionRecord",
    "read_champion",
    "write_champion",
    "to_dict",
    "from_dict",
    "sync_alias",
]


@dataclass(frozen=True)
class ChampionRecord:
    """Immutable snapshot of the champion-of-record (spec §4.3 schema).

    ``previous`` holds the record this one replaced (or ``None`` for the
    seeded baseline), so rollback needs no extra state (spec decision 7).
    """

    variant: str
    model_ref: str
    f2p_rate: float
    p2p_rate: float
    dataset_run_id: str
    tier: str
    seed: int
    promoted_at: str
    previous: ChampionRecord | None


def _require(data: dict[str, Any], key: str, types: tuple[type, ...]) -> Any:
    """Return *data[key]* validated against *types* (bool never accepted)."""
    if key not in data:
        raise ValueError(f"champion record missing required field {key!r}")
    value = data[key]
    if not isinstance(value, types) or isinstance(value, bool):
        # spec §4.3 mandates ValueError on bad types (TRY004 would prefer TypeError).
        raise ValueError(  # noqa: TRY004
            f"champion record field {key!r} has invalid type {type(value).__name__}"
        )
    return value


def to_dict(record: ChampionRecord) -> dict[str, Any]:
    """Serialize a record to a plain dict (nested ``previous`` recursed)."""
    return asdict(record)


def from_dict(data: dict[str, Any]) -> ChampionRecord:
    """Parse a plain dict into a :class:`ChampionRecord`.

    All nine schema fields must be present with the right types; ``previous``
    is either null or itself a valid record dict.  Unknown extra keys are
    tolerated (forward-compatible).  Raises ``ValueError`` on any violation.
    """
    if not isinstance(data, dict):
        # spec §4.3: malformed champion.json must raise ValueError.
        raise ValueError(  # noqa: TRY004
            f"champion record must be a JSON object, got {type(data).__name__}"
        )
    previous = data.get("previous")
    return ChampionRecord(
        variant=_require(data, "variant", (str,)),
        model_ref=_require(data, "model_ref", (str,)),
        f2p_rate=float(_require(data, "f2p_rate", (int, float))),
        p2p_rate=float(_require(data, "p2p_rate", (int, float))),
        dataset_run_id=_require(data, "dataset_run_id", (str,)),
        tier=_require(data, "tier", (str,)),
        seed=_require(data, "seed", (int,)),
        promoted_at=_require(data, "promoted_at", (str,)),
        previous=None if previous is None else from_dict(previous),
    )


def read_champion(path: str | Path) -> ChampionRecord:
    """Read and validate the champion record at *path*.

    Raises:
        FileNotFoundError: if *path* does not exist.
        ValueError: if the file is malformed JSON or fails schema validation.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"champion record not found at {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"champion record {path} is malformed JSON: {exc}") from exc
    return from_dict(data)


def write_champion(path: str | Path, record: ChampionRecord) -> None:
    """Serialize *record* to *path* as pretty plain JSON (schema field order).

    Parent directories are created on demand; the file mirrors
    ``smoke_baseline.json`` style (indent=2, gsutil-friendly).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(record), indent=2) + "\n", encoding="utf-8")


def sync_alias(champion_key: str, config: EvalConfig) -> str | None:
    """Sync the W&B human-facing ``champion`` alias to *champion_key*.

    Idempotent: ``comparison._clear_champion_alias`` first removes any stale
    ``champion`` alias, then ``comparison.promote_champion_to_registry``
    (which never raises) links the champion artifact.  ``wandb`` is imported
    lazily; on any failure (not installed, no network, API error) this
    returns ``None`` — the alias is derived from champion.json, never a
    second source of truth (spec §4.3, decision 5).

    Args:
        champion_key: ``"model:variant"`` key, e.g. ``"qwen3-14b:higher_lr_14b"``.
        config: Eval config (wandb entity/project + artifact pattern).

    Returns:
        Human-readable summary from the promotion call, or ``None`` on failure.
    """
    try:
        import wandb

        api = wandb.Api(timeout=30)
    except Exception:  # noqa: BLE001 — W&B must never break the pipeline
        return None
    comparison._clear_champion_alias(api, config)
    return comparison.promote_champion_to_registry(champion_key, config)
