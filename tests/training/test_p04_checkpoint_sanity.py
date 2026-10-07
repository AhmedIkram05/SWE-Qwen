"""P0-4 checkpoint sanity: per-run output dirs, callback run filtering, config.

No GPU, no Modal runtime, no W&B API — unit tests with ``tmp_path`` and
mocks only. Guards against the stale ``checkpoint-108`` contamination class
of bug (shared output dirs + global-max checkpoint selection).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import training.callbacks as cbs
from training.qlora_trainer import (
    QLoRATrainer,
    prepare_run_output_dir,
    resolve_run_output_dir,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ── Per-variant-run output dir resolution (modal + local paths) ──────────────


@pytest.mark.unit
class TestModalOutputDir:
    """``resolve_run_output_dir`` / ``prepare_run_output_dir`` / trainer wiring."""

    def test_modal_output_dir_unique_per_variant(self, tmp_path):
        base = tmp_path / "qlora-output"
        t1 = QLoRATrainer(output_dir=str(base), variant="baseline_14b", run_name="r")
        t2 = QLoRATrainer(output_dir=str(base), variant="higher_rank_14b", run_name="r")
        t3 = QLoRATrainer(output_dir=str(base), variant="baseline_14b", run_name="other")

        dirs = {t1.output_dir, t2.output_dir, t3.output_dir}
        assert len(dirs) == 3
        assert t1.output_dir == base / "baseline_14b-r"
        assert t2.output_dir == base / "higher_rank_14b-r"
        assert t3.output_dir == base / "baseline_14b-other"
        # Created on construction, per run.
        assert all(d.is_dir() for d in dirs)

    def test_modal_run_output_dir_uses_run_id_fallback(self, tmp_path):
        base = tmp_path / "out"
        t = QLoRATrainer(output_dir=str(base), variant="v", run_id="expanded-repos")
        assert t.output_dir == base / "v-expanded-repos"
        assert t.output_dir.is_dir()

    def test_modal_prepare_cleans_stale_checkpoints(self, tmp_path):
        base = tmp_path / "out"
        stale_dir = base / "checkpoint-108"
        stale_dir.mkdir(parents=True)
        (stale_dir / "adapter_model.safetensors").write_text("stale")
        stale_file = base / "checkpoint-99.txt"
        stale_file.write_text("stale")

        result = prepare_run_output_dir(str(base))

        assert Path(result).is_dir()
        assert not (Path(result) / "checkpoint-108").exists()
        assert not (Path(result) / "checkpoint-99.txt").exists()

    def test_modal_prepare_keeps_checkpoints_when_resuming(self, tmp_path):
        base = tmp_path / "out"
        stale_dir = base / "checkpoint-108"
        stale_dir.mkdir(parents=True)

        prepare_run_output_dir(str(base), resume_from_checkpoint=str(stale_dir))

        assert stale_dir.is_dir()

    def test_modal_resolve_run_output_dir_idempotent(self, tmp_path):
        base = tmp_path / "out"
        once = resolve_run_output_dir(str(base), "v", "r")
        twice = resolve_run_output_dir(once, "v", "r")
        assert once == twice


# ── WandbCheckpointCallback: per-run filtering ───────────────────────────────


def _fake_run(run_id: str = "run-1", name: str = "my-run") -> SimpleNamespace:
    return SimpleNamespace(id=run_id, name=name)


def _state(global_step: int) -> SimpleNamespace:
    return SimpleNamespace(
        global_step=global_step,
        epoch=0.5,
        log_history=[],
        max_steps=global_step,
    )


def _args(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(output_dir=str(output_dir))


@pytest.mark.unit
class TestCallbackRunFiltering:
    """``WandbCheckpointCallback`` uploads the current run's checkpoint only."""

    def _invoke(self, callback, base, global_step, mocker):
        mocker.patch.object(cbs.wandb, "run", _fake_run())
        artifact_cls = mocker.patch.object(cbs.wandb, "Artifact")
        artifact = artifact_cls.return_value
        log_artifact = mocker.patch.object(cbs.wandb, "log_artifact")
        mocker.patch("training.callbacks.time.sleep")
        callback.on_save(_args(base), _state(global_step), SimpleNamespace())
        return artifact_cls, artifact, log_artifact

    def test_callback_uploads_current_run_checkpoint_not_stale_max(self, tmp_path, mocker):
        base = tmp_path / "out"
        (base / "checkpoint-108").mkdir(parents=True)  # stale, higher step
        (base / "checkpoint-20").mkdir(parents=True)  # this run's save

        callback = cbs.WandbCheckpointCallback(run_id="run-1")
        artifact_cls, _, log_artifact = self._invoke(callback, base, 20, mocker)

        log_artifact.assert_called_once()
        # The stale checkpoint-108 must NOT be uploaded.
        artifact_cls.assert_called_once()
        assert "step-20" in artifact_cls.call_args.kwargs["name"]

    def test_callback_run_id_mismatch_skips_upload(self, tmp_path, mocker):
        base = tmp_path / "out"
        (base / "checkpoint-20").mkdir(parents=True)

        callback = cbs.WandbCheckpointCallback(run_id="other-run")
        artifact_cls, _, log_artifact = self._invoke(callback, base, 20, mocker)

        log_artifact.assert_not_called()
        artifact_cls.assert_not_called()

    def test_callback_resume_ignores_stale_higher_step(self, tmp_path, mocker):
        base = tmp_path / "out"
        (base / "checkpoint-108").mkdir(parents=True)  # resumable (<= step)
        (base / "checkpoint-200").mkdir(parents=True)  # stale from another run

        callback = cbs.WandbCheckpointCallback(run_id="run-1")
        artifact_cls, _, log_artifact = self._invoke(callback, base, 128, mocker)

        log_artifact.assert_called_once()
        # Fallback picks the newest checkpoint at or below the current step.
        assert "step-108" in artifact_cls.call_args.kwargs["name"]

    def test_callback_no_checkpoints_is_quiet_noop(self, tmp_path, mocker):
        base = tmp_path / "out"
        base.mkdir(parents=True)

        callback = cbs.WandbCheckpointCallback(run_id="run-1")
        artifact_cls, _, log_artifact = self._invoke(callback, base, 20, mocker)

        log_artifact.assert_not_called()
        artifact_cls.assert_not_called()

    def test_trainer_wires_run_id_into_checkpoint_callback(self, tmp_path, mocker):
        import training.qlora_trainer as qt

        trainer = QLoRATrainer(output_dir=str(tmp_path / "out"), variant="v", run_name="r")
        trainer.training_args = SimpleNamespace()
        trainer.model = object()
        trainer.tokenizer = object()
        sft_trainer = mocker.patch.object(qt, "SFTTrainer")
        mocker.patch.object(cbs.wandb, "run", _fake_run(run_id="wandb-abc"))

        trainer._setup_callbacks()

        callbacks = sft_trainer.call_args.kwargs["callbacks"]
        checkpoint_cb = next(c for c in callbacks if isinstance(c, cbs.WandbCheckpointCallback))
        assert checkpoint_cb.run_id == "wandb-abc"


# ── config/qlora_variants.yaml sanity ────────────────────────────────────────


@pytest.mark.unit
class TestConfig:
    """``qlora_variants.yaml``: save cadence + eval readiness."""

    _VARIANTS_PATH = _PROJECT_ROOT / "config" / "qlora_variants.yaml"

    def _training(self) -> dict:
        cfg = yaml.safe_load(self._VARIANTS_PATH.read_text(encoding="utf-8"))
        return cfg["variants"]["baseline_14b"]["training"]

    def test_config_save_steps_below_typical_total_steps(self):
        # 1-epoch 14B runs total ~16-76 steps; save_steps must be below the
        # smallest expected total so intermediates actually exist (was 500).
        training = self._training()
        assert training["save_steps"] == 10
        assert training["save_steps"] < 16

    def test_config_eval_ready_but_off_by_default(self):
        training = self._training()
        # A10G-safe default: eval stays off (it OOMs), but eval_steps is
        # preset so overriding eval_strategy to "steps" works immediately.
        assert training["eval_strategy"] == "no"
        assert training["eval_steps"] == training["save_steps"]
        # Separate-eval-worker-ready keys are present (commented, since
        # load_best_model_at_end requires eval to be enabled).
        raw = self._VARIANTS_PATH.read_text(encoding="utf-8")
        assert "# load_best_model_at_end: true" in raw
        assert "# metric_for_best_model: eval_loss" in raw
        assert "# greater_is_better: false" in raw
        assert "separate-eval-worker-ready" in raw
