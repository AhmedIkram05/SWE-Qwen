"""Unit tests covering remaining lines in scripts/ (f2p_proxy, init_wandb,
prepare_training_data, run_3config_comparison).  No network, no Modal, no W&B.
"""

from __future__ import annotations

import importlib.util
import io
import json
import logging
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def _load_scripts_module(name: str):
    path = _PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load scripts/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── scripts/f2p_proxy.py ────────────────────────────────────────────────────


class TestF2PProxyGaps:
    def _load(self):
        return _load_scripts_module("f2p_proxy")

    def test_wandb_project_entity_no_entity(self, mocker):
        wandb_fake = mocker.MagicMock()
        wandb_fake.Api.return_value.default_entity = None
        mocker.patch.dict(sys.modules, {"wandb": wandb_fake})
        mod = self._load()
        with pytest.raises(RuntimeError, match="entity not found"):
            mod._wandb_project_entity()

    def test_compute_proxy_loss_range_zero(self, tmp_path, mocker):
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        wandb_fake = mocker.MagicMock()
        wandb_fake.Api.return_value.default_entity = "entity"

        def _runs(project, filters=None):  # noqa: ARG001
            r = mocker.MagicMock()
            r.state = "finished"
            r.created_at = "2024-01-01"
            r.summary = {"train/loss": 5.0}
            return [r]

        wandb_fake.Api.return_value.runs.side_effect = _runs
        mocker.patch.dict(sys.modules, {"wandb": wandb_fake})
        mod = self._load()
        out = mod.compute_proxy_f2p_scores(golden, {"a": "p1", "b": "p2"})
        assert out["a"]["mean_f2p"] == 1.0
        assert out["b"]["mean_f2p"] == 1.0
        assert "loss" in out["a"]

    def test_compute_missing_loss_key(self, mocker, tmp_path):
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        wandb_fake = mocker.MagicMock()
        wandb_fake.Api.return_value.default_entity = "entity"

        def _runs(project, filters=None):  # noqa: ARG001
            r = mocker.MagicMock()
            r.state = "finished"
            r.created_at = "2024-01-01"
            r.summary = {}
            return [r]

        wandb_fake.Api.return_value.runs.side_effect = _runs
        mocker.patch.dict(sys.modules, {"wandb": wandb_fake})
        mod = self._load()
        out = mod.compute_proxy_f2p_scores(golden, {"a": "p1"})
        assert out["a"]["mean_f2p"] == 0.0
        assert out["a"]["warning"] == "no train/loss in summary"


# ── scripts/init_wandb.py ───────────────────────────────────────────────────


class TestInitWandbMain:
    def _load(self):
        return _load_scripts_module("init_wandb")

    @pytest.fixture(autouse=True)
    def _wandb_fake(self, mocker):
        mocker.patch.dict(sys.modules, {"wandb": mocker.MagicMock()})

    def test_main_success_flags(self, mocker):
        mod = self._load()
        mocker.patch.object(mod, "init_wandb_project", return_value=True)
        create_sweep = mocker.patch.object(mod, "create_sweep_config", return_value="abc")
        setup = mocker.patch.object(mod, "setup_artifact_registries")
        mocker.patch(
            "sys.argv",
            [
                "init_wandb",
                "--project",
                "proj",
                "--entity",
                "team",
                "--description",
                "custom desc",
                "--create-sweep",
                "--setup-registries",
            ],
        )
        assert mod.main() == 0
        create_sweep.assert_called_once_with("proj", "team")
        setup.assert_called_once_with("proj", "team")
        kwargs = mod.init_wandb_project.call_args.kwargs
        assert kwargs["description"] == "custom desc"
        assert kwargs["tags"] == [
            "swe",
            "qwen",
            "qwen3-moe",
            "code-generation",
            "fine-tuning",
            "llm",
        ]

    def test_main_default_description(self, mocker):
        mod = self._load()
        mocker.patch.object(mod, "init_wandb_project", return_value=True)
        create_sweep = mocker.patch.object(mod, "create_sweep_config")
        setup = mocker.patch.object(mod, "setup_artifact_registries")
        mocker.patch("sys.argv", ["init_wandb", "--entity", "team"])
        assert mod.main() == 0
        create_sweep.assert_not_called()
        setup.assert_not_called()
        assert "SWE-Qwen" in mod.init_wandb_project.call_args.kwargs["description"]
        assert mod.init_wandb_project.call_args.kwargs["config"]["model"] == "Qwen/Qwen3-30B-A3B"

    def test_main_init_fails(self, mocker):
        mod = self._load()
        mocker.patch.object(mod, "init_wandb_project", return_value=False)
        mocker.patch("sys.argv", ["init_wandb"])
        assert mod.main() == 1


# ── scripts/prepare_training_data.py ────────────────────────────────────────


class TestPrepareTrainingData:
    def _load(self):
        return _load_scripts_module("prepare_training_data")

    @staticmethod
    def _fake_tokenize_module(mocker, *, raise_error: bool = False):
        fake = ModuleType("data_engineering.tokenize")
        if raise_error:
            fake.tokenize_pipeline = mocker.MagicMock(side_effect=RuntimeError("boom"))
        else:
            fake.tokenize_pipeline = mocker.MagicMock(return_value=True)
        mocker.patch.dict(sys.modules, {"data_engineering.tokenize": fake})
        return fake.tokenize_pipeline

    def test_tokenize_datasets_ok(self, mocker, tmp_path):
        mod = self._load()
        tp = self._fake_tokenize_module(mocker)
        assert mod.tokenize_datasets(tmp_path / "in", tmp_path / "out", "model") is True
        tp.assert_called_once()

    def test_tokenize_datasets_failure(self, mocker, tmp_path):
        mod = self._load()
        self._fake_tokenize_module(mocker, raise_error=True)
        assert mod.tokenize_datasets(tmp_path, tmp_path / "out", "model") is False

    @staticmethod
    def _records():
        return [
            {"repo": "sympy/sympy", "issue_id": "s-1"},
            {"repo": "django/django", "issue_id": "d-1"},
        ]

    def test_main_success(self, mocker, tmp_path, monkeypatch):
        mod = self._load()
        raw = tmp_path / "raw.jsonl"
        raw.write_text("")
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(raw),
                "--output-dir",
                str(tmp_path / "out"),
                "--model-name",
                "qwen3-14b",
                "--max-length",
                "2048",
            ],
        )
        recs = self._records()
        mocker.patch.object(mod, "load_jsonl", return_value=recs * 2)
        mocker.patch.object(mod, "basic_quality_filters", side_effect=lambda r: r)
        mocker.patch.object(mod, "repo_stratified_split", return_value=(recs, recs, recs))
        mocker.patch.object(mod, "extract_golden", return_value=(recs, recs))
        tokenize = mocker.patch.object(mod, "tokenize_datasets", return_value=True)
        summary = mocker.patch.object(mod, "log_summary")
        monkeypatch.chdir(tmp_path)
        mod.main()
        tokenize.assert_called_once()
        args = tokenize.call_args
        assert args.kwargs["max_length"] == 2048
        assert args.kwargs["model_name"] == "qwen3-14b"
        summary.assert_called_once()
        assert (tmp_path / "out" / "jsonl" / "train.jsonl").exists()
        assert (tmp_path / "out" / "jsonl" / "golden.jsonl").exists()

    def test_main_skip_tokenize(self, mocker, tmp_path, monkeypatch):
        mod = self._load()
        raw = tmp_path / "raw.jsonl"
        raw.write_text("")
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(raw),
                "--output-dir",
                str(tmp_path / "out"),
                "--skip-tokenize",
            ],
        )
        recs = self._records()
        mocker.patch.object(mod, "load_jsonl", return_value=recs)
        mocker.patch.object(mod, "basic_quality_filters", side_effect=lambda r: r)
        mocker.patch.object(mod, "repo_stratified_split", return_value=(recs, recs, recs))
        mocker.patch.object(mod, "extract_golden", return_value=(recs, recs))
        tokenize = mocker.patch.object(mod, "tokenize_datasets")
        summary = mocker.patch.object(mod, "log_summary")
        monkeypatch.chdir(tmp_path)
        mod.main()
        tokenize.assert_not_called()
        assert summary.call_args.args[-1] is True

    def test_main_tokenize_fails(self, mocker, tmp_path, monkeypatch):
        mod = self._load()
        raw = tmp_path / "raw.jsonl"
        raw.write_text("")
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(raw),
                "--output-dir",
                str(tmp_path / "out"),
            ],
        )
        mocker.patch.object(mod, "load_jsonl", return_value=self._records())
        mocker.patch.object(mod, "basic_quality_filters", side_effect=lambda r: r)
        mocker.patch.object(mod, "repo_stratified_split", return_value=(self._records(), [], []))
        mocker.patch.object(mod, "extract_golden", return_value=([], []))
        mocker.patch.object(mod, "tokenize_datasets", return_value=False)
        mocker.patch.object(mod, "log_summary")
        monkeypatch.chdir(tmp_path)
        mod.main()

    def test_main_input_missing(self, mocker, tmp_path):
        mod = self._load()
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(tmp_path / "a" / "b" / "c" / "raw.jsonl"),
            ],
        )
        mod.main()

    def test_main_input_missing_empty_data_dir(self, mocker, tmp_path):
        # data dir exists but holds no run dirs (no digit-prefixed entries)
        mod = self._load()
        (tmp_path / "ploads").mkdir()
        missing = tmp_path / "ploads" / "runs" / "rs" / "raw.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(missing),
            ],
        )
        mod.main()

    def test_main_input_missing_with_runs(self, mocker, tmp_path, caplog):
        mod = self._load()
        (tmp_path / "12345").mkdir()
        (tmp_path / "54321").mkdir()
        missing = tmp_path / "runs" / "abc" / "raw.jsonl"
        mocker.patch(
            "sys.argv",
            [
                "prepare_training_data",
                "--input",
                str(missing),
            ],
        )
        with caplog.at_level(logging.INFO):
            mod.main()
        assert any("Available runs" in r.message for r in caplog.records)

    def test_main_no_records(self, mocker, tmp_path):
        mod = self._load()
        raw = tmp_path / "raw.jsonl"
        raw.write_text("")
        mocker.patch("sys.argv", ["prepare_training_data", "--input", str(raw)])
        mocker.patch.object(mod, "load_jsonl", return_value=[])
        mocker.patch.object(mod, "basic_quality_filters", return_value=[])
        mod.main()


# ── scripts/run_3config_comparison.py ───────────────────────────────────────


class Test3ConfigGaps:
    def _load(self):
        return _load_scripts_module("run_3config_comparison")

    @pytest.fixture
    def fake_wandb(self, mocker):
        fake = mocker.MagicMock()
        fake.__spec__ = mocker.MagicMock()
        mocker.patch.dict(sys.modules, {"wandb": fake})
        return fake

    @staticmethod
    def _run(mocker, *, name="run", state="finished", rid="r", variant=None, summary=None):
        r = mocker.MagicMock()
        r.name = name
        r.state = state
        r.id = rid
        r.config = {} if variant is None else {"variant": variant}
        r.created_at = "2024-01-01"
        r.summary = summary or {}
        return r

    # ── _reconcile_state_with_wandb ──

    def test_reconcile_api_error(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.side_effect = Exception("w&b down")
        state = mod._new_state("r")
        assert mod._reconcile_state_with_wandb(state) is state
        assert state["variants"] == {}

    def test_reconcile_no_runs_for_variant(self, fake_wandb):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = []
        state = mod._new_state("r")
        out = mod._reconcile_state_with_wandb(state, requested_variants=["missing"])
        assert "missing" not in out["variants"]
        assert out["completed_variants"] == []

    def test_reconcile_run_without_variant(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = [
            self._run(mocker, name="plain", state="finished", rid="n1")
        ]
        state = mod._new_state("r")
        state["variants"]["v1"] = {"run_name": "", "result": {}}
        out = mod._reconcile_state_with_wandb(state)
        # no run carries this variant → state entry untouched
        assert out["variants"]["v1"] == {"run_name": "", "result": {}}

    def test_reconcile_fallback_suffix_match(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        r1 = self._run(mocker, name="other-look", state="running", rid="r1", variant="v1")
        r2 = self._run(mocker, name="3config-v1-20240101", state="running", rid="r2", variant="v1")
        fake_wandb.Api.return_value.runs.return_value = [r1, r2]
        state = mod._new_state("r")
        out = mod._reconcile_state_with_wandb(state, requested_variants=["v1"])
        assert out["variants"]["v1"]["status"] == "running"
        assert out["variants"]["v1"]["run_name"] == "3config-v1-20240101"

    def test_reconcile_fallback_first_run(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        r1 = self._run(mocker, name="odd", state="finished", rid="r3", variant="v1")
        fake_wandb.Api.return_value.runs.return_value = [r1]
        state = mod._new_state("r")
        out = mod._reconcile_state_with_wandb(state, requested_variants=["v1"])
        assert out["variants"]["v1"]["status"] == "completed"
        assert out["variants"]["v1"]["result"]["wandb_run_id"] == "r3"

    def test_reconcile_match_by_wandb_id_dedup(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        r1 = self._run(mocker, name="saved-run", state="finished", rid="r9", variant="v1")
        fake_wandb.Api.return_value.runs.return_value = [r1]
        state = mod._new_state("r")
        state["completed_variants"] = ["v1"]
        state["variants"]["v1"] = {"run_name": "", "result": {"wandb_run_id": "r9"}}
        out = mod._reconcile_state_with_wandb(state)
        assert out["completed_variants"] == ["v1"]

    def test_reconcile_empty_runs_for_variant_in_state(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        r_other = self._run(mocker, name="x", state="finished", rid="n2")
        fake_wandb.Api.return_value.runs.return_value = [r_other]
        state = mod._new_state("r")
        state["variants"]["ghost"] = {"run_name": ""}
        out = mod._reconcile_state_with_wandb(state, requested_variants=["ghost"])
        assert out["variants"]["ghost"] == {"run_name": ""}

    def test_reconcile_unknown_state(self, fake_wandb, mocker):
        # matched run in a state that's neither finished/crashed/running
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        killed = self._run(mocker, name="k", state="killed", rid="rk", variant="v1")
        fake_wandb.Api.return_value.runs.return_value = [killed]
        state = mod._new_state("r")
        out = mod._reconcile_state_with_wandb(state, requested_variants=["v1"])
        assert "v1" not in out["variants"]

    def test_reconcile_running_loss_already_completed(self, fake_wandb, mocker):
        # running with train/loss already logged, variant recorded as completed
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        run = self._run(
            mocker, name="dup", state="running", rid="rd", variant="v1", summary={"train/loss": 0.5}
        )
        fake_wandb.Api.return_value.runs.return_value = [run]
        state = mod._new_state("r")
        state["completed_variants"] = ["v1"]
        state["variants"]["v1"] = {"run_name": "dup", "result": {"wandb_run_id": "rd"}}
        out = mod._reconcile_state_with_wandb(state)
        assert out["completed_variants"] == ["v1"]
        assert out["variants"]["v1"]["status"] == "completed"

    # ── _wandb_run_finished ──

    def test_wandb_run_finished_no_entity(self, fake_wandb):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = None
        with pytest.raises(RuntimeError, match="entity not found"):
            mod._wandb_run_finished("run")

    def test_wandb_run_finished_live_but_unknown(self, fake_wandb, mocker):
        mod = self._load()
        fake_wandb.Api.return_value.default_entity = "e"
        stale = self._run(mocker, name="w", state="running", rid="stale", variant="v")
        fake_wandb.Api.return_value.runs.return_value = [stale]
        assert mod._wandb_run_finished("run") is None

    # ── _ensure_golden binary download ──

    def test_ensure_golden_binary_item(self, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path

        class _CtxBytesIO(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        list_data = json.dumps(
            {
                "items": [
                    {"name": "datasets/run1/swebench/", "mediaLink": "https://s/dir"},
                    {"name": "datasets/run1/swebench/tokenizer.bin", "mediaLink": "https://s/bin"},
                    {
                        "name": "datasets/run1/swebench/golden.jsonl",
                        "mediaLink": "https://s/golden",
                    },
                ]
            }
        ).encode()
        list_resp = mocker.MagicMock()
        list_resp.read.return_value = list_data
        list_resp.__enter__.return_value = list_resp
        bin_resp = _CtxBytesIO(b"\x00binary")
        json_resp = mocker.MagicMock()
        json_resp.read.return_value = b'{"test": "data"}'
        json_resp.__enter__.return_value = json_resp
        mocker.patch("urllib.request.urlopen", side_effect=[list_resp, bin_resp, json_resp])
        dst = mod._ensure_golden("run1")
        assert dst.read_text() == '{"test": "data"}'
        assert (
            tmp_path / "data" / "run1" / "swebench" / "tokenizer.bin"
        ).read_bytes() == b"\x00binary"

    # ── launch_modal_training ──

    def test_launch_fresh_completes(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        finished = self._run(
            mocker,
            name="3config-baseline_14b-20240101",
            state="finished",
            rid="W1",
            variant="baseline_14b",
        )
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = [finished]

        proc = mocker.MagicMock()
        proc.pid = 42
        mocker.patch.object(mod.subprocess, "Popen", return_value=proc)

        state = mod._new_state("run123")
        result = mod.launch_modal_training(
            "baseline_14b",
            "run123",
            "3config-baseline_14b-20240101",
            state,
            train_kwargs={"model_name": "qwen3-14b", "gpu_type": None},
        )
        assert result["wandb_run_id"] == "W1"
        assert state["variants"]["baseline_14b"]["status"] == "completed"
        assert state["variants"]["baseline_14b"]["result"]["wandb_run_id"] == "W1"
        cmd = mod.subprocess.Popen.call_args.args[0]
        assert cmd[0] == "modal"

    def test_launch_saved_run_not_finished(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        finished = self._run(
            mocker,
            name="3config-baseline_14b-20240101",
            state="finished",
            rid="W2",
            variant="baseline_14b",
        )
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.side_effect = [[], [finished]]
        mocker.patch.object(mod.subprocess, "Popen", return_value=mocker.MagicMock())
        state = mod._new_state("r")
        state["variants"]["baseline_14b"] = {"status": "launched", "run_name": "saved-run"}
        result = mod.launch_modal_training("baseline_14b", "r", "new-run", state)
        assert result["wandb_run_id"] == "W2"

    def test_launch_crashed_saved_run(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        crashed = self._run(
            mocker, name="saved-run", state="crashed", rid="WX", variant="baseline_14b"
        )
        finished = self._run(
            mocker, name="new-run", state="finished", rid="W3", variant="baseline_14b"
        )
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.side_effect = [[crashed], [finished]]
        mocker.patch.object(mod.subprocess, "Popen", return_value=mocker.MagicMock())
        state = mod._new_state("r")
        state["variants"]["baseline_14b"] = {"status": "launched", "run_name": "saved-run"}
        result = mod.launch_modal_training("baseline_14b", "r", "new-run", state)
        assert result["wandb_run_id"] == "W3"

    def test_launch_failed_state_relaunch(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        finished = self._run(mocker, name="r", state="finished", rid="W4", variant="v")
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = [finished]
        mocker.patch.object(mod.subprocess, "Popen", return_value=mocker.MagicMock())
        state = mod._new_state("r")
        state["variants"]["v"] = {"status": "failed"}
        result = mod.launch_modal_training("v", "r", "rn", state)
        assert result["wandb_run_id"] == "W4"

    def test_launch_poll_crash_raises(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        crashed = self._run(mocker, name="rn", state="crashed", rid="WC", variant="v")
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = [crashed]
        mocker.patch.object(mod.subprocess, "Popen", return_value=mocker.MagicMock())
        state = mod._new_state("r")
        with pytest.raises(RuntimeError, match="Variant 'v'"):
            mod.launch_modal_training("v", "r", "rn", state)

    def test_launch_process_exited(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        mocker.patch.object(mod.time, "time", side_effect=[0, 0, 200])
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = []
        proc = mocker.MagicMock()
        proc.poll.return_value = 1
        mocker.patch.object(mod.subprocess, "Popen", return_value=proc)
        state = mod._new_state("r")
        with pytest.raises(RuntimeError, match="exited with code 1"):
            mod.launch_modal_training("v", "r", "rn", state)

    def test_launch_poll_timeout(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        mocker.patch.object(mod.time, "time", side_effect=[0, 0, 22000])
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = []
        proc = mocker.MagicMock()
        proc.poll.return_value = None
        mocker.patch.object(mod.subprocess, "Popen", return_value=proc)
        state = mod._new_state("r")
        with pytest.raises(RuntimeError, match="Poll timeout"):
            mod.launch_modal_training("v", "r", "rn", state)

    def test_launch_keyboard_interrupt(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep", side_effect=KeyboardInterrupt)
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = []
        mocker.patch.object(mod.subprocess, "Popen", return_value=mocker.MagicMock())
        state = mod._new_state("r")
        with pytest.raises(SystemExit) as e:
            mod.launch_modal_training("v", "r", "rn", state)
        assert e.value.code == 130

    def test_launch_resume_finished(self, fake_wandb, mocker, tmp_path):
        # previous run finished via W&B while we were asleep → no re-launch
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        finished = self._run(mocker, name="saved-run", state="finished", rid="WS", variant="v")
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.return_value = [finished]
        popen = mocker.patch.object(mod.subprocess, "Popen")
        state = mod._new_state("r")
        state["variants"]["v"] = {"status": "launched", "run_name": "saved-run"}
        result = mod.launch_modal_training("v", "r", "new-run", state)
        assert result["wandb_run_id"] == "WS"
        popen.assert_not_called()

    def test_launch_min_poll_wait_completes(self, fake_wandb, mocker, tmp_path):
        # elapsed crosses MIN_POLL_WAIT but not POLL_TIMEOUT → probe process, then finish
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "_save_state")
        mocker.patch.object(mod.time, "sleep")
        mocker.patch.object(mod.time, "time", side_effect=[0, 0, 130])
        finished = self._run(mocker, name="new-run", state="finished", rid="WT", variant="v")
        fake_wandb.Api.return_value.default_entity = "e"
        fake_wandb.Api.return_value.runs.side_effect = [[], [finished]]
        proc = mocker.MagicMock()
        proc.poll.return_value = None
        mocker.patch.object(mod.subprocess, "Popen", return_value=proc)
        state = mod._new_state("r")
        result = mod.launch_modal_training("v", "r", "new-run", state)
        assert result["wandb_run_id"] == "WT"

    # ── evaluate_proxy_f2p (live) ──

    def test_evaluate_proxy_f2p_live(self, fake_wandb, mocker, tmp_path):
        mod = self._load()
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        finished = self._run(
            mocker, name="n", state="finished", rid="W", variant="v", summary={"train/loss": 0.3}
        )
        fake_wandb.Api.return_value.default_entity = "entity"
        fake_wandb.Api.return_value.runs.return_value = [finished]
        result = mod.evaluate_proxy_f2p("v", golden, "/tmp/adapter", dry_run=False)
        assert result["mean_f2p"] == 1.0
        assert result["loss"] == 0.3

    def test_evaluate_proxy_f2p_missing_script(self, mocker, tmp_path):
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        mocker.patch("importlib.util.spec_from_file_location", return_value=None)
        with pytest.raises(RuntimeError, match="Failed to load f2p_proxy.py"):
            mod.evaluate_proxy_f2p("v", golden, "/tmp/adapter", dry_run=False)

    # ── main() orchestration ──

    @pytest.fixture
    def main_mocks(self, mocker, tmp_path):
        """Patch everything main() touches; copy f2p_proxy under tmp so the
        importlib reload in main resolves without touching the real repo."""
        mod = self._load()
        mod._REPO_ROOT = tmp_path
        (tmp_path / "scripts").mkdir(parents=True)
        shutil.copyfile(
            _PROJECT_ROOT / "scripts" / "f2p_proxy.py",
            tmp_path / "scripts" / "f2p_proxy.py",
        )
        golden = tmp_path / "golden.jsonl"
        golden.write_text('{"id": 1}\n')
        mocker.patch.object(mod, "_STATE_PATH", tmp_path / "state.json")
        mocker.patch.object(mod, "signal")
        mocker.patch.object(mod, "_ensure_golden", return_value=golden)
        mocker.patch.object(mod, "_load_state", return_value=mod._new_state("run123"))
        reconcile = mocker.patch.object(
            mod, "_reconcile_state_with_wandb", side_effect=lambda s, **kw: s
        )
        save_state = mocker.patch.object(mod, "_save_state")
        cleanup = mocker.patch.object(mod, "_cleanup_state")
        mocker.patch.object(mod, "_variant_run_name", return_value="3config-baseline_14b-20240101")
        launch = mocker.patch.object(
            mod,
            "launch_modal_training",
            return_value={"wandb_run_id": "wb-1", "artifact_name": "model-qwen3-14b-baseline_14b"},
        )
        download = mocker.patch.object(
            mod, "download_adapter", return_value=str(tmp_path / "adapter")
        )
        evaluate = mocker.patch.object(mod, "evaluate_proxy_f2p", return_value={"mean_f2p": 0.5})
        return {
            "mod": mod,
            "golden": golden,
            "reconcile": reconcile,
            "save_state": save_state,
            "cleanup": cleanup,
            "launch": launch,
            "download": download,
            "evaluate": evaluate,
        }

    @pytest.fixture
    def wandb_for_compute(self, mocker):
        """wandb returning a finished run per variant for the real proxy call."""
        fake = mocker.MagicMock()
        fake.__spec__ = mocker.MagicMock()
        fake.Api.return_value.default_entity = "entity"

        def _runs(project, filters=None):  # noqa: ARG001
            r = mocker.MagicMock()
            r.id = "id-1"
            r.state = "finished"
            r.created_at = "2024-01-01"
            r.summary = {"train/loss": 1.0}
            return [r]

        fake.Api.return_value.runs.side_effect = _runs
        mocker.patch.dict(sys.modules, {"wandb": fake})
        return fake

    def test_main_dry_run_full(self, main_mocks, wandb_for_compute, mocker, tmp_path):
        mod = main_mocks["mod"]
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--output",
                str(report),
            ],
        )
        mod.main()
        main_mocks["cleanup"].assert_called_once()
        main_mocks["reconcile"].assert_not_called()
        data = json.loads(report.read_text())
        assert data["dry_run"] is True
        assert data["champion"] in {"baseline_14b", "higher_rank_14b", "higher_lr_14b"}
        assert set(data["variants"]) == {"baseline_14b", "higher_rank_14b", "higher_lr_14b"}
        assert data["champion_f2p"] is not None
        assert main_mocks["launch"].call_count == 3

    def test_main_skip_eval(self, main_mocks, mocker, tmp_path, capsys):
        mod = main_mocks["mod"]
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--skip-eval",
                "--output",
                str(report),
            ],
        )
        mod.main()
        data = json.loads(report.read_text())
        assert data["champion"] is None
        assert "Skipping evaluation" in capsys.readouterr().out
        assert main_mocks["launch"].call_count == 3

    def test_main_force_retrain(self, main_mocks, mocker, tmp_path):
        mod = main_mocks["mod"]
        state = mod._new_state("run123")
        state["completed_variants"] = ["baseline_14b"]
        main_mocks["mod"]._load_state.return_value = state
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--force-retrain",
                "--skip-eval",
                "--output",
                str(report),
            ],
        )
        mod.main()
        main_mocks["reconcile"].assert_not_called()
        assert main_mocks["launch"].call_count == 3

    def test_main_completed_skip_and_max_samples(self, main_mocks, mocker, tmp_path):
        mod = main_mocks["mod"]
        state = mod._new_state("run123")
        state["completed_variants"] = ["baseline_14b"]
        state["variants"]["baseline_14b"] = {
            "result": {"wandb_run_id": "prev", "artifact_name": "prev-art"}
        }
        mod._load_state.return_value = state
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--skip-eval",
                "--max-train-samples",
                "4",
                "--output",
                str(report),
            ],
        )
        mod.main()
        data = json.loads(report.read_text())
        assert data["variants"]["baseline_14b"]["wandb_run_id"] == "prev"
        assert main_mocks["launch"].call_count == 2
        # train_kwargs grew max_train_samples=4 for every launch
        train_kwargs = main_mocks["launch"].call_args_list[0].args[4]
        assert train_kwargs["max_train_samples"] == 4

    def test_main_all_fail(self, main_mocks, mocker, tmp_path):
        mod = main_mocks["mod"]
        main_mocks["launch"].side_effect = RuntimeError("boom")
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--output",
                str(report),
            ],
        )
        mod.main()
        data = json.loads(report.read_text())
        assert data["champion"] is None
        assert len(data["failed_variants"]) == 3
        assert data["variants"] == {}

    def test_main_f2p_proxy_missing(self, main_mocks, mocker, tmp_path):
        # results exist but the f2p_proxy module fails to load → hard error
        mod = main_mocks["mod"]
        report = tmp_path / "report.json"
        mocker.patch("importlib.util.spec_from_file_location", return_value=None)
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--dry-run",
                "--output",
                str(report),
            ],
        )
        with pytest.raises(RuntimeError, match="Failed to load f2p_proxy.py"):
            mod.main()

    def test_main_reconcile_live(self, main_mocks, mocker, tmp_path):
        mod = main_mocks["mod"]
        report = tmp_path / "report.json"
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--skip-eval",
                "--output",
                str(report),
            ],
        )
        mod.main()
        main_mocks["reconcile"].assert_called_once()
        main_mocks["save_state"].assert_called()
        assert main_mocks["launch"].call_count == 3

    def test_main_promote_non_dry(self, main_mocks, wandb_for_compute, mocker, tmp_path, capsys):
        mod = main_mocks["mod"]
        report = tmp_path / "report.json"
        promote = mocker.patch.object(mod, "promote_champion")
        mocker.patch(
            "sys.argv",
            [
                "run_3config_comparison",
                "--run-id",
                "run123",
                "--output",
                str(report),
            ],
        )
        mod.main()
        data = json.loads(report.read_text())
        assert data["champion"] is not None
        promote.assert_called_once()
        assert promote.call_args[0][0] == data["champion"]
