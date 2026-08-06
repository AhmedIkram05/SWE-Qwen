"""Tests for ``src/swe_qwen/modal_app.py``.

None of these tests contact Modal, W&B, HuggingFace, or any network.
The root ``tests/conftest.py`` installs a fake ``modal`` module; functions
decorated with ``@app.function`` may arrive as ``_ModalFunctionWrapper``
instances or as plain functions depending on which test file imported the
module first. ``_unwrap`` preserves both shapes so the suite is
order-independent. The module is pre-imported here under a MagicMock-based
``modal`` so the shared cached module state matches what
``test_infra_coverage`` expects regardless of collection order.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from unittest import mock as unittest_mock

import pytest

_MODULE_NAME = "src.swe_qwen.modal_app"


def _import_plain():
    """Import the module under a MagicMock ``modal``/``wandb``.

    ``test_infra_coverage`` and this file both exercise this module, and its
    tests assume functions are not wrapped by Modal. Do the same importing so
    whichever file runs first produces the same cached module.
    """
    plain_modal = unittest_mock.MagicMock()
    plain_app = unittest_mock.MagicMock()
    plain_app.function = lambda **_kw: lambda f: f
    plain_app.local_entrypoint = lambda **_kw: lambda f: f
    plain_modal.App = lambda *_a, **_kw: plain_app
    with unittest_mock.patch.dict(
        sys.modules, {"modal": plain_modal, "wandb": unittest_mock.MagicMock()}
    ):
        return importlib.import_module(_MODULE_NAME)


if _MODULE_NAME not in sys.modules:
    _import_plain()


def _mod():
    """Import (or return cached) ``src.swe_qwen.modal_app``."""
    return importlib.import_module(_MODULE_NAME)


def _unwrap(func):
    """Return the underlying callable of a ``_ModalFunctionWrapper`` if any."""
    wrapped = getattr(func, "_func", None)
    return wrapped if wrapped is not None else func


def _call(func, *args, **kwargs):
    """Call a (possibly wrapper-wrapped) callable like the original function."""
    return _unwrap(func)(*args, **kwargs)


def _instance(cls, **kwargs):
    """Instantiate a (possibly wrapper-wrapped) class."""
    return _unwrap(cls)(**kwargs)


# ── hello_modal ────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestHelloModal:
    def test_tokens_set(self, mocker):
        mod = _mod()
        with unittest_mock.patch.dict("os.environ", {}):
            os.environ["MODAL_TOKEN"] = "t"
            os.environ["WANDB_API_KEY"] = "k"
            result = _call(mod.hello_modal)
        assert result["status"] == "ok"
        assert result["modal_token_set"] is True
        assert result["wandb_key_set"] is True

    def test_tokens_missing(self, mocker):
        mod = _mod()
        with unittest_mock.patch.dict("os.environ", {}):
            os.environ.pop("MODAL_TOKEN", None)
            os.environ.pop("WANDB_API_KEY", None)
            result = _call(mod.hello_modal)
        assert result["modal_token_set"] is False
        assert result["wandb_key_set"] is False


# ── train_swe_qwen ──────────────────────────────────────────────────────────


def _fake_train_deps(mocker):
    """Install fake torch/datasets/peft/transformers/trl into ``sys.modules``.

    Returns a namespace of recording fakes for assertions.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.bfloat16 = object()

    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_from_disk = mocker.MagicMock()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = mocker.MagicMock()
    fake_transformers.BitsAndBytesConfig = mocker.MagicMock()
    fake_transformers.AutoModelForCausalLM = mocker.MagicMock()
    fake_transformers.TrainingArguments = mocker.MagicMock()

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = mocker.MagicMock()
    fake_peft.get_peft_model = mocker.MagicMock(side_effect=lambda model, *a, **k: model)
    fake_peft.prepare_model_for_kbit_training = mocker.MagicMock(side_effect=lambda model: model)

    fake_trl = types.ModuleType("trl")
    fake_trl.SFTTrainer = mocker.MagicMock()

    mocker.patch.dict(
        sys.modules,
        {
            "torch": fake_torch,
            "datasets": fake_datasets,
            "peft": fake_peft,
            "transformers": fake_transformers,
            "trl": fake_trl,
        },
    )

    wandb_mock = mocker.patch.object(_mod(), "wandb")

    return types.SimpleNamespace(
        torch=fake_torch,
        datasets=fake_datasets,
        peft=fake_peft,
        transformers=fake_transformers,
        trl=fake_trl,
        wandb=wandb_mock,
    )


@pytest.mark.unit
class TestTrainSweQwen:
    def test_default_path(self, mocker):
        mod = _mod()
        ns = _fake_train_deps(mocker)
        sample = [{"text": "hello"}]
        ns.datasets.load_from_disk.return_value = {"train": sample, "validation": sample}

        result = _call(
            mod.train_swe_qwen,
            model_name="Qwen/Qwen3-14B",
            dataset_path="/data/train",
            output_dir="/out",
            num_epochs=2,
            batch_size=2,
            learning_rate=1e-4,
            lora_rank=16,
            lora_alpha=32,
            lora_dropout=0.1,
            gradient_accumulation_steps=2,
            max_seq_length=2048,
            warmup_ratio=0.01,
            weight_decay=0.01,
            logging_steps=2,
            save_steps=10,
            eval_steps=10,
            wandb_project="swe-qwen",
            push_to_hub=False,
        )

        assert result["status"] == "completed"
        assert result["output_dir"] == "/out"
        assert result["run_name"].startswith("swe-qwen-")

        ns.wandb.login.assert_called_once()
        ns.wandb.init.assert_called_once()
        assert ns.wandb.init.call_args[1]["project"] == "swe-qwen"
        assert ns.wandb.init.call_args[1]["name"] == result["run_name"]

        tok = ns.transformers.AutoTokenizer.from_pretrained.return_value
        assert tok.pad_token == tok.eos_token
        assert tok.padding_side == "right"

        model_kwargs = ns.transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert model_kwargs["attn_implementation"] == "flash_attention_2"
        assert model_kwargs["quantization_config"] is not None

        ns.peft.prepare_model_for_kbit_training.assert_called_once()
        ns.peft.LoraConfig.assert_called_once()
        ns.peft.get_peft_model.assert_called_once()

        train_args = ns.transformers.TrainingArguments.call_args[1]
        assert train_args["evaluation_strategy"] == "steps"
        assert train_args["load_best_model_at_end"] is True
        assert train_args["metric_for_best_model"] == "eval_loss"
        assert train_args["run_name"] == result["run_name"]

        trainer_kwargs = ns.trl.SFTTrainer.call_args[1]
        assert trainer_kwargs["eval_dataset"] is sample
        assert trainer_kwargs["max_seq_length"] == 2048
        assert trainer_kwargs["packing"] is True

        tok.save_pretrained.assert_called_once_with("/out")
        ns.wandb.log_artifact.assert_called_once()
        ns.wandb.finish.assert_called_once()
        # push_to_hub branch not taken
        ns.transformers.AutoModelForCausalLM.from_pretrained.return_value.push_to_hub.assert_not_called()

    def test_push_to_hub(self, mocker):
        mod = _mod()
        ns = _fake_train_deps(mocker)
        ns.datasets.load_from_disk.return_value = {"train": [{"text": "x"}]}

        result = _call(
            mod.train_swe_qwen,
            wandb_run_name="custom-run",
            push_to_hub=True,
            hub_model_id="acme/swe-qwen",
        )

        assert result["run_name"] == "custom-run"
        model = ns.transformers.AutoModelForCausalLM.from_pretrained.return_value
        model.push_to_hub.assert_called_once_with("acme/swe-qwen", use_temp_dir=True)
        tokenizer = ns.transformers.AutoTokenizer.from_pretrained.return_value
        tokenizer.push_to_hub.assert_called_once_with("acme/swe-qwen", use_temp_dir=True)

    def test_no_4bit_no_flash_attn(self, mocker):
        mod = _mod()
        ns = _fake_train_deps(mocker)
        ns.datasets.load_from_disk.return_value = {"train": [{"text": "x"}]}

        _call(mod.train_swe_qwen, use_4bit=False, use_flash_attn=False)

        model_kwargs = ns.transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
        assert model_kwargs["quantization_config"] is None
        assert model_kwargs["attn_implementation"] == "eager"
        ns.transformers.BitsAndBytesConfig.assert_not_called()
        ns.peft.prepare_model_for_kbit_training.assert_not_called()

    def test_no_eval_dataset(self, mocker):
        mod = _mod()
        ns = _fake_train_deps(mocker)
        ns.datasets.load_from_disk.return_value = {"train": [{"text": "x"}]}

        _call(mod.train_swe_qwen)

        train_args = ns.transformers.TrainingArguments.call_args[1]
        assert train_args["evaluation_strategy"] == "no"
        assert train_args["load_best_model_at_end"] is False
        assert train_args["metric_for_best_model"] is None
        assert ns.trl.SFTTrainer.call_args[1]["eval_dataset"] is None


# ── serve_swe_qwen ──────────────────────────────────────────────────────────


class _FakeRouteApp:
    """Stand-in FastAPI app that records route handlers."""

    instances: list[_FakeRouteApp] = []

    def __init__(self, *args, **kwargs):
        self.creation_args = (args, kwargs)
        self.routes = {}
        _FakeRouteApp.instances.append(self)

    def get(self, path, **kwargs):
        def _deco(fn):
            self.routes[("GET", path)] = fn
            return fn

        return _deco

    def post(self, path, **kwargs):
        def _deco(fn):
            self.routes[("POST", path)] = fn
            return fn

        return _deco


@pytest.mark.unit
class TestServeSweQwen:
    def test_serve_builds_app_and_streams(self, mocker):
        mod = _mod()
        _FakeRouteApp.instances.clear()

        fake_fastapi = types.ModuleType("fastapi")
        fake_fastapi.FastAPI = _FakeRouteApp
        responses_mod = types.ModuleType("fastapi.responses")
        fake_fastapi.responses = responses_mod

        streaming_seen = []

        class _FakeStreamingResponse:
            def __init__(self, content, media_type=None, **kwargs):
                self.content = content
                self.media_type = media_type
                streaming_seen.append(self)

        responses_mod.StreamingResponse = _FakeStreamingResponse

        protocol_mod = types.ModuleType("vllm.entrypoints.openai.protocol")

        class _ProtoReq:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        protocol_mod.CompletionRequest = _ProtoReq
        protocol_mod.ChatCompletionRequest = _ProtoReq

        openai_mod = types.ModuleType("vllm.entrypoints.openai")
        openai_mod.protocol = protocol_mod
        entrypoints_mod = types.ModuleType("vllm.entrypoints")
        entrypoints_mod.openai = openai_mod

        engine = mocker.MagicMock()
        engine.generate.return_value = "gen-stream-1"
        engine.chat.return_value = "gen-stream-2"

        engine_args_seen = {}

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.AsyncEngineArgs = lambda **kwargs: engine_args_seen.update(kwargs)
        fake_vllm.AsyncLLMEngine = types.SimpleNamespace(from_engine_args=lambda a: engine)
        fake_vllm.entrypoints = entrypoints_mod

        served = {"started": False}

        def fake_uvicorn_server():
            class _Srv:
                def __init__(self, config):
                    self.config = config

                async def serve(self):
                    served["started"] = True

            return _Srv

        fake_uvicorn = types.ModuleType("uvicorn")
        fake_uvicorn.Config = lambda *a, **k: k
        fake_uvicorn.Server = fake_uvicorn_server()

        mocker.patch.dict(
            sys.modules,
            {
                "fastapi": fake_fastapi,
                "fastapi.responses": responses_mod,
                "vllm": fake_vllm,
                "vllm.entrypoints": entrypoints_mod,
                "vllm.entrypoints.openai": openai_mod,
                "vllm.entrypoints.openai.protocol": protocol_mod,
                "uvicorn": fake_uvicorn,
            },
        )

        result = asyncio.run(
            _call(
                mod.serve_swe_qwen,
                model_path="/models/finetuned",
                host="0.0.0.0",
                port=8000,
                max_model_len=4096,
                tensor_parallel_size=2,
                gpu_memory_utilization=0.9,
            )
        )

        assert result == {"status": "serving", "model": "/models/finetuned"}
        assert served["started"] is True
        assert _FakeRouteApp.instances[0].creation_args[1] == {
            "title": "SWE-Qwen API",
            "version": "1.0.0",
        }
        assert engine_args_seen["tensor_parallel_size"] == 2
        assert engine_args_seen["gpu_memory_utilization"] == 0.9
        assert engine_args_seen["model"] == "/models/finetuned"

        health = asyncio.run(_FakeRouteApp.instances[0].routes[("GET", "/health")]())
        assert health == {"status": "healthy", "model": "/models/finetuned"}

        asyncio.run(
            _FakeRouteApp.instances[0].routes[("POST", "/v1/completions")](
                {"prompt": "fix this bug", "sampling_params": {}, "request_id": "r1"}
            )
        )
        asyncio.run(
            _FakeRouteApp.instances[0].routes[("POST", "/v1/chat/completions")](
                {
                    "messages": [{"role": "user", "content": "hi"}],
                    "sampling_params": {},
                    "request_id": "r2",
                }
            )
        )

        assert engine.generate.call_args[1]["prompt"] == "fix this bug"
        assert engine.chat.call_args[1]["messages"] == [{"role": "user", "content": "hi"}]
        assert len(streaming_seen) == 2
        assert {s.content for s in streaming_seen} == {"gen-stream-1", "gen-stream-2"}
        assert all(s.media_type == "application/json" for s in streaming_seen)


# ── DataPipelineConfig / run_data_pipeline ──────────────────────────────────


@pytest.mark.unit
class TestDataPipelineConfig:
    def test_defaults(self):
        mod = _mod()
        cfg = _instance(mod.DataPipelineConfig)
        assert cfg.swe_bench_dir == "/data/swe_bench"
        assert cfg.output_dir == "/data/pipeline_output"
        assert cfg.run_id is None
        assert cfg.stages == "all"
        assert cfg.bigquery is True
        assert cfg.wandb_project == "swe-qwen-data"
        assert cfg.parallel == 4

    def test_custom_fields(self):
        mod = _mod()
        cfg = _instance(
            mod.DataPipelineConfig,
            swe_bench_dir="/custom/sb",
            output_dir="/custom/out",
            run_id="run-77",
            stages="tokenize",
            bigquery=False,
            wandb_project="p2",
            parallel=9,
        )
        assert cfg.run_id == "run-77"
        assert cfg.stages == "tokenize"
        assert cfg.bigquery is False
        assert cfg.parallel == 9


@pytest.mark.unit
class TestRunDataPipeline:
    def test_success_with_run_id(self, mocker, tmp_path):
        mod = _mod()
        cfg = _instance(
            mod.DataPipelineConfig,
            swe_bench_dir=str(tmp_path / "sb"),
            output_dir=str(tmp_path / "out"),
            run_id="run-9",
            stages="all",
            bigquery=False,
            wandb_project="proj",
            parallel=3,
        )
        proc = mocker.MagicMock()
        proc.returncode = 0
        proc.stdout = "ok"
        proc.stderr = ""
        run = mocker.patch("subprocess.run", return_value=proc)
        mocker.patch("sys.executable", "/usr/bin/python")

        with unittest_mock.patch.dict("os.environ", {}):
            result = mod.run_data_pipeline(
                augment_codecontests=True,
                augment_codealpaca=True,
                max_train_examples=321,
                cfg=cfg,
            )
            assert os.environ["DATA_PIPELINE_WANDB_PROJECT"] == "proj"

        assert result["status"] == "completed"
        assert result["run_id"] == "run-9"
        assert result["max_train_examples"] == 321

        cmd = run.call_args[0][0]
        assert cmd[0] == "/usr/bin/python"
        assert cmd[cmd.index("--run-id") + 1] == "run-9"
        assert "--augment-codecontests" in cmd
        assert "--augment-codealpaca" in cmd
        assert "--no-bigquery" in cmd
        assert cmd[cmd.index("--max-train-examples") + 1] == "321"
        assert cmd[cmd.index("--stages") + 1] == "all"
        assert cmd[cmd.index("--parallel") + 1] == "3"
        assert run.call_args[1]["cwd"] == "/app"
        assert "PYTHONPATH" in run.call_args[1]["env"]

    def test_success_auto_run_id_disabled_flags(self, mocker, tmp_path):
        mod = _mod()
        cfg = _instance(
            mod.DataPipelineConfig,
            swe_bench_dir=str(tmp_path / "sb"),
            output_dir=str(tmp_path / "out"),
            run_id=None,
            stages="split",
            bigquery=True,
            wandb_project="proj",
            parallel=2,
        )
        proc = mocker.MagicMock()
        proc.returncode = 0
        proc.stdout = "done"
        proc.stderr = ""
        run = mocker.patch("subprocess.run", return_value=proc)

        result = mod.run_data_pipeline(
            augment_codecontests=False,
            augment_codealpaca=False,
            max_train_examples=10,
            cfg=cfg,
        )

        assert result["status"] == "completed"
        cmd = run.call_args[0][0]
        auto_id = cmd[cmd.index("--run-id") + 1]
        assert auto_id.startswith("modal-")
        assert "--no-augment-codecontests" in cmd
        assert "--no-augment-codealpaca" in cmd
        assert "--bigquery" in cmd

    def test_failure_raises(self, mocker, tmp_path):
        mod = _mod()
        cfg = _instance(
            mod.DataPipelineConfig,
            swe_bench_dir=str(tmp_path / "sb"),
            output_dir=str(tmp_path / "out"),
        )
        proc = mocker.MagicMock()
        proc.returncode = 3
        proc.stdout = "boom out"
        proc.stderr = "boom err"
        mocker.patch("subprocess.run", return_value=proc)

        with pytest.raises(RuntimeError, match="Pipeline failed with exit code 3"):
            mod.run_data_pipeline(max_train_examples=10, cfg=cfg)


# ── entrypoints ─────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestEntrypoints:
    def test_run_data_pipeline_local(self, mocker, capsys):
        mod = _mod()
        fake = mocker.MagicMock()
        fake.remote.return_value = {
            "status": "completed",
            "run_id": "local-1",
            "output_dir": "/data/out",
        }
        mocker.patch.object(mod, "run_data_pipeline", fake)

        result = mod.run_data_pipeline_local(
            augment_codecontests=False,
            augment_codealpaca=False,
            max_train_examples=50,
        )

        fake.remote.assert_called_once_with(
            augment_codecontests=False,
            augment_codealpaca=False,
            max_train_examples=50,
        )
        assert result["run_id"] == "local-1"
        out = capsys.readouterr().out
        assert "Launching data pipeline" in out
        assert "Pipeline completed" in out
        assert "Run ID: local-1" in out

    def test_run_data_pipeline_local_defaults(self, mocker):
        mod = _mod()
        fake = mocker.MagicMock()
        fake.remote.return_value = {"status": "completed", "run_id": "x", "output_dir": "/o"}
        mocker.patch.object(mod, "run_data_pipeline", fake)

        mod.run_data_pipeline_local()

        assert fake.remote.call_args[1]["augment_codecontests"] is True
        assert fake.remote.call_args[1]["max_train_examples"] == 30000

    def test_main(self, mocker, capsys):
        mod = _mod()
        fake = mocker.MagicMock()
        fake.remote.return_value = {"status": "ok", "message": "Modal is configured correctly!"}
        mocker.patch.object(mod, "hello_modal", fake)

        mod.main()

        fake.remote.assert_called_once()
        out = capsys.readouterr().out
        assert "Testing Modal setup" in out
        assert "status" in out
