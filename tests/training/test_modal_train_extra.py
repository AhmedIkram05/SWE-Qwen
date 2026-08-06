"""Tests for ``training/modal_train.py``.

Follows the same order-independent convention as ``tests/test_modal_app.py``:
``@app.function``-decorated callables may be ``_ModalFunctionWrapper``
instances or plain functions depending on which test imported the module
first. ``_unwrap`` normalises both. No network / Modal / W&B / torch calls.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

import pytest

_MODULE_NAME = "training.modal_train"


def _mod():
    """Import (or return cached) ``training.modal_train``."""
    return importlib.import_module(_MODULE_NAME)


def _unwrap(func):
    """Return the underlying callable of a ``_ModalFunctionWrapper`` if any."""
    wrapped = getattr(func, "_func", None)
    return wrapped if wrapped is not None else func


def _call(func, *args, **kwargs):
    return _unwrap(func)(*args, **kwargs)


def _reload():
    """Freshly execute the module (for the ``unsloth`` import branch)."""
    if _MODULE_NAME not in sys.modules:
        importlib.import_module(_MODULE_NAME)
    return importlib.reload(sys.modules[_MODULE_NAME])


class _FakeHttpResponse:
    """Context-manager stand-in for ``urllib.request.urlopen`` results."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, *args) -> bytes:
        return self._payload

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(self, *exc) -> bool:
        return False


# ── GCS download helper ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestDownloadGcsPublic:
    def test_downloads_items_and_skips_prefix_only(self, mocker, tmp_path):
        mod = _mod()
        payload = (
            b'{"items": ['
            b'{"name": "tokenized/run-1/train/data-00000.arrow", "mediaLink": "http://x/1"},'
            b'{"name": "tokenized/run-1/", "mediaLink": "http://x/2"}'
            b"]}"
        )
        copyfileobj = mocker.patch("shutil.copyfileobj")
        urlopen = mocker.patch(
            "urllib.request.urlopen",
            side_effect=[_FakeHttpResponse(payload), _FakeHttpResponse(b"bytes")],
        )

        result = mod._download_gcs_public("tokenized/run-1/", str(tmp_path))

        assert result == str(tmp_path)
        data_file = tmp_path / "train" / "data-00000.arrow"
        assert data_file.exists()
        assert not (tmp_path / "train").is_dir() or list((tmp_path / "train").iterdir()) == [
            data_file
        ]
        assert copyfileobj.call_count == 1
        assert urlopen.call_count == 2

        list_url = urlopen.call_args_list[0][0][0]
        assert "storage/v1/b/swe-qwen-datasets/o?prefix=tokenized/run-1/" in list_url

    def test_no_items_raises(self, mocker, tmp_path):
        mod = _mod()
        mocker.patch(
            "urllib.request.urlopen",
            return_value=_FakeHttpResponse(b'{"items": []}'),
        )

        with pytest.raises(RuntimeError, match="No objects found"):
            mod._download_gcs_public("tokenized/ghost/", str(tmp_path))

    def test_missing_items_key_raises(self, mocker, tmp_path):
        mod = _mod()
        mocker.patch(
            "urllib.request.urlopen",
            return_value=_FakeHttpResponse(b"{}"),
        )

        with pytest.raises(RuntimeError, match="No objects found"):
            mod._download_gcs_public("tokenized/ghost/", str(tmp_path))


# ── unsloth module-level import branch ───────────────────────────────────────


@pytest.mark.unit
class TestUnslothImportBranch:
    def test_unsloth_installed(self, mocker):
        fake_unsloth = types.ModuleType("unsloth")
        mocker.patch.dict(sys.modules, {"unsloth": fake_unsloth})

        mod = _reload()
        assert mod._unsloth_patched is True

    def test_unsloth_missing(self, mocker):
        mocker.patch.dict(sys.modules, {})

        sys.modules.pop("unsloth", None)
        mod = _reload()
        assert mod._unsloth_patched is False


# ── train_qlora ────────────────────────────────────────────────────────────


def _fake_training_imports(mocker, *, with_torch_shutdown=True):
    """Implant fake ``training.qlora_config`` / ``training.qlora_trainer`` /
    ``torch`` modules and return recording handles."""
    model = mocker.MagicMock()
    tokenizer = mocker.MagicMock()
    trainer = mocker.MagicMock()
    trainer.train.return_value = {"train_loss": 0.42}

    fake_qcfg = types.ModuleType("training.qlora_config")
    fake_qcfg.resolve_gpu_type = mocker.MagicMock(return_value="A100:1")
    fake_qcfg.build_model_and_peft = mocker.MagicMock(return_value=(model, tokenizer))

    fake_qtr = types.ModuleType("training.qlora_trainer")
    fake_qtr.QLoRATrainer = mocker.MagicMock(return_value=trainer)

    modules = {
        "training.qlora_config": fake_qcfg,
        "training.qlora_trainer": fake_qtr,
    }

    shutdown_workers = mocker.MagicMock()
    if with_torch_shutdown:
        fake_torch = types.ModuleType("torch")
        inductor = types.ModuleType("torch._inductor")
        async_compile = types.ModuleType("torch._inductor.async_compile")
        async_compile.shutdown_compile_workers = shutdown_workers
        inductor.async_compile = async_compile
        fake_torch._inductor = inductor
        modules.update(
            {
                "torch": fake_torch,
                "torch._inductor": inductor,
                "torch._inductor.async_compile": async_compile,
            }
        )
    else:
        modules["torch"] = types.ModuleType("torch")

    mocker.patch.dict(sys.modules, modules)
    return types.SimpleNamespace(
        qcfg=fake_qcfg,
        qtr=fake_qtr,
        trainer=trainer,
        model=model,
        tokenizer=tokenizer,
        shutdown_workers=shutdown_workers,
    )


@pytest.mark.unit
class TestTrainQlora:
    def test_missing_api_key_raises(self, mocker):
        mod = _mod()
        mocker.patch.dict("os.environ", {"WANDB_API_KEY": ""})

        with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
            _call(mod.train_qlora)

    def test_full_path_unsloth_from_env(self, mocker, caplog):
        mod = _mod()
        ns = _fake_training_imports(mocker)
        mocker.patch.dict("os.environ", {"WANDB_API_KEY": "key", "UNSLOTH_ENABLED": "1"})
        download = mocker.patch.object(
            mod, "_download_gcs_public", return_value="/tmp/data/tokenized/final"
        )
        run_stub = mocker.MagicMock()
        run_stub.id = "wandb-run-77"
        wandb_mock = mocker.patch.object(mod, "wandb")
        wandb_mock.run = run_stub

        with caplog.at_level(logging.INFO, logger="training.modal_train"):
            result = _call(
                mod.train_qlora,
                model_name="qwen3-14b",
                variant="baseline_14b",
                run_id="run-abc",
                output_dir="/models/qlora-output",
                run_name="my-run",
                wandb_entity="acme",
                max_train_samples=99,
            )

        assert "Unsloth" in caplog.text
        download.assert_called_once_with("tokenized/run-abc/", "/tmp/data")
        ns.qcfg.resolve_gpu_type.assert_called_once_with("qwen3-14b")
        ns.qcfg.build_model_and_peft.assert_called_once()

        train_call = ns.qtr.QLoRATrainer.call_args
        assert train_call[1]["run_name"] == "my-run"
        assert train_call[1]["wandb_entity"] == "acme"
        assert train_call[1]["data_dir"] == "/tmp/data/tokenized/final"
        assert train_call[1]["max_train_samples"] == 99
        assert train_call[1]["gpu_type"] == "A100:1"

        ns.shutdown_workers.assert_called_once()
        assert result == {
            "status": "completed",
            "wandb_run_id": "wandb-run-77",
            "artifact_name": "model-qwen3-14b-baseline_14b",
            "model_name": "qwen3-14b",
            "variant": "baseline_14b",
            "output_dir": "/models/qlora-output",
            "metrics": {"train_loss": 0.42},
        }

    def test_explicit_gpu_and_no_unsloth(self, mocker, caplog):
        mod = _mod()
        ns = _fake_training_imports(mocker)
        mocker.patch.dict("os.environ", {"WANDB_API_KEY": "key"})
        mocker.patch.object(mod, "_download_gcs_public", return_value="/tmp/data/tokenized/f")
        mocker.patch("wandb.run", None)
        mocker.patch("wandb.finish")

        with caplog.at_level(logging.INFO, logger="training.modal_train"):
            result = _call(
                mod.train_qlora,
                gpu_type="A10G:1",
                use_unsloth=False,
                resume="/models/ckpt/step-500",
            )

        assert "standard TRL" in caplog.text
        ns.qcfg.resolve_gpu_type.assert_not_called()
        assert result["wandb_run_id"] is None
        assert result["metrics"] == {"train_loss": 0.42}
        assert ns.qtr.QLoRATrainer.call_args[1]["resume_from_checkpoint"] == "/models/ckpt/step-500"

    def test_torch_shutdown_failure_suppressed(self, mocker):
        mod = _mod()
        ns = _fake_training_imports(mocker, with_torch_shutdown=False)
        mocker.patch.dict("os.environ", {"WANDB_API_KEY": "key"})
        mocker.patch.object(mod, "_download_gcs_public", return_value="/tmp/data/tokenized/f")
        mocker.patch("wandb.run", None)

        result = _call(mod.train_qlora)

        assert result["status"] == "completed"
        ns.shutdown_workers.assert_not_called()

    def test_wandb_finish_error_suppressed(self, mocker):
        mod = _mod()
        _fake_training_imports(mocker)
        mocker.patch.dict("os.environ", {"WANDB_API_KEY": "key"})
        mocker.patch.object(mod, "_download_gcs_public", return_value="/tmp/data/tokenized/f")
        mocker.patch("wandb.run", mocker.MagicMock(id="run-1"))
        mocker.patch("wandb.finish", side_effect=RuntimeError("network down"))

        result = _call(mod.train_qlora)

        assert result["status"] == "completed"
        assert result["wandb_run_id"] == "run-1"


# ── get_gpu_for_model / alias ──────────────────────────────────────────────


@pytest.mark.unit
class TestGetGpuForModel:
    def test_returns_resolved_gpu(self):
        mod = _mod()
        gpu = _call(mod.get_gpu_for_model, "qwen3-14b")
        assert isinstance(gpu, str)
        assert gpu

    def test_modal_entrypoint_alias(self):
        mod = _mod()
        assert mod.modal_entrypoint is mod.train_qlora


if __name__ == "__main__":
    pass
