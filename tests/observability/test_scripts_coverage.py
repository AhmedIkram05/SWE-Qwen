"""Phase 8 coverage top-up for the three observability scripts (offline).

``scripts/log_deploy.py``, ``scripts/seed_dashboards.py`` and
``scripts/build_dashboards.py`` are driven end-to-end with wandb /
wandb-workspaces fakes in ``sys.modules`` — no real W&B client is built,
so these tests never touch the network or read credentials.
"""

from __future__ import annotations

import random
import sys
import types
from typing import Any

import pytest

from observability.dashboards import PANELS, PanelSpec
from observability.metrics import METRIC_REGISTRY, assert_registered
from scripts import build_dashboards, log_deploy, seed_dashboards

pytestmark = pytest.mark.unit


# ── scripts/log_deploy.py ───────────────────────────────────────────────────


def test_main_payload_and_finish(monkeypatch):
    inits: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    finished: list[str] = []

    def _init(**kwargs: Any) -> None:
        inits.append(kwargs)

    def _log(data: dict[str, Any]) -> None:
        logged.append(data)

    def _finish() -> None:
        finished.append("finish")

    wandb = types.SimpleNamespace(run=types.SimpleNamespace(finish=_finish), init=_init, log=_log)
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    code = log_deploy.main(["--status", "1", "--duration-s", "342", "--sha", "abc123"])
    assert code == 0
    assert logged == [{"deploy/status": 1, "deploy/duration_s": 342.0}]
    assert inits == [
        {
            "project": "swe-qwen",
            "entity": "2571642-university-of-dundee",
            "job_type": "deploy",
            "config": {
                "deploy_status": 1,
                "deploy_duration_s": 342.0,
                "sha": "abc123",
            },
        }
    ]
    assert finished == ["finish"]


def test_main_without_optional_args_and_no_run(monkeypatch):
    def _init(**kwargs: Any) -> None:
        _ = kwargs

    def _log(data: dict[str, Any]) -> None:
        _ = data

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(run=None, init=_init, log=_log))
    assert log_deploy.main(["--status", "0"]) == 0  # no active run -> finish skipped


def test_main_rejects_invalid_status(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace())
    with pytest.raises(SystemExit) as excinfo:
        log_deploy.main(["--status", "2"])
    assert excinfo.value.code == 2


def test_build_payload_sha_skipped_when_unregistered():
    payload = log_deploy.build_payload(1, None, "abc123")
    assert payload == {"deploy/status": 1}  # deploy/sha is not a metric
    assert_registered(payload)


# ── scripts/seed_dashboards.py ──────────────────────────────────────────────


def test_expected_keys_matches_registry():
    flat = {
        f"{domain}/{metric}"
        for domain, metrics in METRIC_REGISTRY.items()
        for metric in metrics
        if not metric.startswith("{key}/")
    }
    hierarchical = {
        f"eval/{seed_dashboards._SEGMENT}/latency_p50",
        f"eval/{seed_dashboards._SEGMENT}/latency_p95",
    }
    assert seed_dashboards.expected_keys() == flat | hierarchical


def test_build_step_cadences():
    mid = seed_dashboards.build_step(1, 20, random.Random(0))  # no cadence fires
    assert "data/records_ingested" not in mid
    assert "deploy/status" not in mid
    assert "eval/f2p_rate" not in mid
    assert_registered(mid)

    zero = seed_dashboards.build_step(0, 20, random.Random(0))  # data + deploy cadence
    assert zero["deploy/status"] == 1
    assert "data/records_ingested" in zero

    failure = seed_dashboards.build_step(30, 45, random.Random(0))  # red dot in cycle
    assert failure["deploy/status"] == 0

    eval_step = seed_dashboards.build_step(7, 20, random.Random(0))  # % 15 == 7
    assert "eval/f2p_rate" in eval_step
    assert f"eval/{seed_dashboards._SEGMENT}/latency_p50" in eval_step
    assert_registered(eval_step)

    late_final = seed_dashboards.build_step(5, 6, random.Random(0))  # final fires all
    assert "data/records_ingested" in late_final
    assert "eval/f2p_rate" in late_final
    assert_registered(late_final)

    single = seed_dashboards.build_step(0, 1, random.Random(0))  # total == 1 edge
    assert single["deploy/status"] == 1


def _make_seed_run(
    logged: list[dict[str, Any]],
    finished: list[str],
    url: str = "https://wandb.ai/acme/seed",
) -> types.SimpleNamespace:
    def _log(data: dict[str, Any]) -> None:
        logged.append(data)

    def _finish() -> None:
        finished.append("finish")

    return types.SimpleNamespace(log=_log, finish=_finish, url=url)


def _make_wandb(init_run: Any) -> tuple[types.SimpleNamespace, list[dict[str, Any]]]:
    inits: list[dict[str, Any]] = []

    def _init(**kwargs: Any) -> Any:
        inits.append(kwargs)
        return init_run

    return types.SimpleNamespace(init=_init), inits


def test_seed_success(monkeypatch, capsys):
    logged: list[dict[str, Any]] = []
    finished: list[str] = []
    wandb, inits = _make_wandb(_make_seed_run(logged, finished, "https://wandb.ai/acme/seed"))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    assert seed_dashboards.seed("swe-qwen", "acme", 3) == 0
    assert inits == [
        {
            "project": "swe-qwen",
            "entity": "acme",
            "name": "dashboard-seed",
            "job_type": "seed",
            "tags": ["seed"],
        }
    ]
    assert len(logged) == 3  # one run.log per synthetic step
    assert finished == ["finish"]
    out = capsys.readouterr().out
    assert "Run URL: https://wandb.ai/acme/seed" in out


def test_seed_missing_keys_warns(monkeypatch, capsys):
    wandb, _ = _make_wandb(_make_seed_run([], []))
    monkeypatch.setitem(sys.modules, "wandb", wandb)

    assert seed_dashboards.seed("swe-qwen", None, 0) == 1  # zero steps -> nothing emitted
    assert "did not emit" in capsys.readouterr().err


def test_seed_wandb_missing(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb", None)
    assert seed_dashboards.seed("swe-qwen", None, 60) == 1
    assert "wandb is not installed" in capsys.readouterr().err


def test_seed_init_failure(monkeypatch, capsys):
    def _init(**kwargs: Any) -> Any:
        _ = kwargs
        raise RuntimeError("no login")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(init=_init))
    assert seed_dashboards.seed("swe-qwen", None, 5) == 1
    assert "could not start the W&B run" in capsys.readouterr().err


def test_seed_main(monkeypatch):
    calls: list[tuple[str, str | None, int]] = []

    def _fake_seed(project: str, entity: str | None, steps: int) -> int:
        calls.append((project, entity, steps))
        return 7

    monkeypatch.setattr(seed_dashboards, "seed", _fake_seed)
    monkeypatch.setattr(sys, "argv", ["seed_dashboards.py"])
    assert seed_dashboards.main() == 7
    monkeypatch.setattr(
        sys,
        "argv",
        ["seed_dashboards.py", "--project", "p", "--entity", "acme", "--steps", "10"],
    )
    assert seed_dashboards.main() == 7
    assert calls == [("swe-qwen", None, 60), ("p", "acme", 10)]


# ── scripts/build_dashboards.py ─────────────────────────────────────────────


def _install_workspace_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    save_raises: bool = False,
) -> tuple[types.ModuleType, list[tuple[str, dict[str, Any]]], list[types.SimpleNamespace]]:
    """Put recording wandb_workspaces fakes in sys.modules (no real import)."""
    panel_calls: list[tuple[str, dict[str, Any]]] = []
    saved: list[types.SimpleNamespace] = []

    def _panel_ctor(kind: str) -> Any:
        def _ctor(**kwargs: Any) -> str:
            panel_calls.append((kind, kwargs))
            return f"{kind}-panel"

        return _ctor

    wr = types.ModuleType("wandb_workspaces.reports.v2")
    wr.__dict__["LinePlot"] = _panel_ctor("LinePlot")
    wr.__dict__["BarPlot"] = _panel_ctor("BarPlot")
    wr.__dict__["ScalarChart"] = _panel_ctor("ScalarChart")

    def _make_section(**kwargs: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(kwargs=kwargs)

    def _make_workspace(**kwargs: Any) -> types.SimpleNamespace:
        workspace = types.SimpleNamespace(kwargs=kwargs, url="https://wandb.ai/ws")

        if save_raises:

            def _save() -> None:
                raise RuntimeError("API down")
        else:

            def _save() -> None:
                saved.append(workspace)

        vars(workspace)["save"] = _save
        return workspace

    ws = types.ModuleType("wandb_workspaces.workspaces")
    ws.__dict__["Section"] = _make_section
    ws.__dict__["Workspace"] = _make_workspace

    pkg = types.ModuleType("wandb_workspaces")
    reports = types.ModuleType("wandb_workspaces.reports")
    monkeypatch.setitem(sys.modules, "wandb_workspaces", pkg)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports", reports)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.reports.v2", wr)
    monkeypatch.setitem(sys.modules, "wandb_workspaces.workspaces", ws)
    return wr, panel_calls, saved


@pytest.fixture
def fake_ws_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[tuple[str, dict[str, Any]]], list[types.SimpleNamespace]]:
    _wr, panel_calls, saved = _install_workspace_modules(monkeypatch)
    return panel_calls, saved


def test_panel_line_with_target_and_custom_aggregate(monkeypatch):
    wr, _panel_calls, _saved = _install_workspace_modules(monkeypatch)

    line = build_dashboards._panel(
        wr, PanelSpec("serve/cold_start_s", "line", "Cold Start", target=10.0)
    )
    assert line is not None  # LinePlot accepted the custom_expressions kwarg

    custom = build_dashboards._panel(
        wr, PanelSpec("deploy/status", "custom", "Deploy Status", aggregate="mean")
    )
    assert custom is not None  # ScalarChart accepted the groupby_aggfunc kwarg


def test_panel_unsupported_type_raises(monkeypatch):
    wr, _panel_calls, _saved = _install_workspace_modules(monkeypatch)
    with pytest.raises(ValueError, match="no as-code mapping for panel type 'run-table'"):
        build_dashboards._panel(wr, PanelSpec("deploy/status", "run-table", "bad"))


def test_build_success(monkeypatch, capsys, fake_ws_modules):
    panel_calls, saved = fake_ws_modules

    assert build_dashboards.build("swe-qwen", "acme") == 0
    assert len(saved) == len(PANELS)
    saved_names = {workspace.kwargs["name"] for workspace in saved}
    assert saved_names == {name.replace("/", "-") for name in PANELS}
    for workspace in saved:
        assert workspace.kwargs["entity"] == "acme"
        assert workspace.kwargs["project"] == "swe-qwen"
        assert [section.kwargs["is_open"] for section in workspace.kwargs["sections"]] == [True]

    by_kind: dict[str, list[dict[str, Any]]] = {}
    for kind, kwargs in panel_calls:
        by_kind.setdefault(kind, []).append(kwargs)
    # S9 10s cold-start target renders as a custom_expressions line
    assert any(kwargs.get("custom_expressions") == ["10.0"] for kwargs in by_kind["LinePlot"])
    # cumulative cost panels aggregate across runs with groupby_aggfunc="sum"
    assert any(kwargs.get("groupby_aggfunc") == "sum" for kwargs in by_kind["BarPlot"])
    assert any(kwargs["title"] == "Deploy Status" for kwargs in by_kind["ScalarChart"])

    out = capsys.readouterr().out
    assert "Creating workspace" in out
    assert out.count("https://wandb.ai/ws") == len(PANELS)


def test_build_resolves_entity_from_api(monkeypatch, fake_ws_modules):
    _panel_calls, saved = fake_ws_modules

    def _api() -> types.SimpleNamespace:
        return types.SimpleNamespace(default_entity="acme-entity")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_api))
    assert build_dashboards.build("swe-qwen", None) == 0
    assert saved[0].kwargs["entity"] == "acme-entity"


def test_build_entity_resolution_failure(monkeypatch, capsys, fake_ws_modules):
    monkeypatch.setattr(build_dashboards, "_default_entity", lambda: None)
    assert build_dashboards.build("swe-qwen", None) == 1
    assert "could not determine the W&B entity" in capsys.readouterr().err


def test_build_missing_workspaces_dependency(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "wandb_workspaces", None)
    assert build_dashboards.build("swe-qwen", "acme") == 1
    assert "uv sync --extra dashboards" in capsys.readouterr().err


def test_build_save_failure(monkeypatch, capsys):
    _wr, _panel_calls, _saved = _install_workspace_modules(monkeypatch, save_raises=True)
    assert build_dashboards.build("swe-qwen", "acme") == 1
    assert "failed to save workspace" in capsys.readouterr().err


def test_default_entity_success(monkeypatch):
    def _api() -> types.SimpleNamespace:
        return types.SimpleNamespace(default_entity="acme-entity")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_api))
    assert build_dashboards._default_entity() == "acme-entity"


def test_default_entity_failure(monkeypatch):
    def _api() -> None:
        raise RuntimeError("no auth")

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Api=_api))
    assert build_dashboards._default_entity() is None


def test_build_main(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    def _fake_build(project: str, entity: str | None) -> int:
        calls.append((project, entity))
        return 0

    monkeypatch.setattr(build_dashboards, "build", _fake_build)
    monkeypatch.setattr(sys, "argv", ["build_dashboards.py"])
    assert build_dashboards.main() == 0
    monkeypatch.setattr(sys, "argv", ["build_dashboards.py", "--project", "p", "--entity", "me"])
    assert build_dashboards.main() == 0
    assert calls == [("swe-qwen", None), ("p", "me")]
