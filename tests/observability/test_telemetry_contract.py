"""Phase 8 CI contract tests (plan §5.9) — ``tests/observability/test_telemetry_contract.py``.

All tests are offline and credential-free: wandb/langfuse SDK clients are
patched (never constructed), and the registry walk is pure AST.

Scope boundary (documented, see ``test_all_wandb_log_keys_registered``): the
walk covers ``wandb.log({...})`` plus ``log_metrics``/``assert_registered``
dict literals — exactly the call shapes §5.9 scans for. Wandb-bound ALIAS
receivers whose keys are already contract-clean are walked through the
explicit, marker-asserted ``_WANDB_ALIASES`` table (evaluation/harness.py,
scripts/seed_dashboards.py, observability/cost.py). ``data_engineering/
run_pipeline.py``'s ``wandb_run.log`` legacy ``stage_*``/``dedup_*``/``clean_*``
scalars are NOT walked: they predate the registry (retrofits from earlier
phases), each is collocated with its registered ``log_metrics`` counterpart or
carries breakdown detail the plan's ``data/*`` namespace intentionally does not
define. Migrating them is a data-pipeline decision out of §5.9 scope — the
registry test fails (naming the key) the moment they are brought in-scope.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import os
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from observability.cost import cost_per_fix, estimate_cost_usd
from observability.dashboards import PANELS, PanelSpec, assert_panels_registered
from observability.logging import configure_logging
from observability.metrics import (
    METRIC_REGISTRY,
    _is_registered,
    assert_registered,
)
from observability.slo import (
    SLO_TARGETS,
    attainment,
    burn_level,
    burn_rate,
    maybe_alert_burn,
)
from scripts.log_deploy import build_payload

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WALK_DIRS = (
    "training",
    "evaluation",
    "inference",
    "data_engineering",
    "observability",
    "scripts",
    "promotion",
)

# Documented wandb-alias receivers: (relative path, receiver name) -> the
# marker string that MUST appear in the module source (so a rename still fails
# the walk instead of silently narrowing it). Each alias's emitted keys are
# already contract-clean; walking them is strictly additive enforcement (they
# also carry the plan's marquee hierarchical eval pattern and f-string keys).
_WANDB_ALIASES: dict[tuple[str, str], str] = {
    ("evaluation/harness.py", "wandb_mod"): "self._wandb_mod =",
    ("scripts/seed_dashboards.py", "run"): "run = wandb.init(",
    ("observability/cost.py", "wandb_run"): "def log_run_cost",
}


def _span(node: ast.AST) -> int:
    lo = getattr(node, "lineno", -1)
    hi = getattr(node, "end_lineno", lo)
    return hi - lo


def _enclosing_func(tree: ast.Module, node: ast.AST) -> ast.FunctionDef | None:
    """The innermost function containing *node* (the emission's scope)."""
    best: ast.FunctionDef | None = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, ast.FunctionDef):
            continue
        if getattr(candidate, "lineno", -1) <= getattr(node, "lineno", -1) <= getattr(
            candidate, "end_lineno", -1
        ) and (best is None or _span(candidate) < _span(best)):
            best = candidate
    return best


def _scope_stmt_nodes(func: ast.FunctionDef) -> list[ast.AST]:
    """Function-local statement descendants, excluding nested function bodies."""
    out: list[ast.AST] = []
    for node in ast.walk(func):
        if node is func:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        out.append(node)
    return out


def _joined_template(node: ast.JoinedStr) -> str:
    """An f-string key -> a ``{}``-marked template (the ``{key}`` placeholder
    may span several slash-segments, e.g. ``f"eval/{key}/latency_p50"``)."""
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:  # FormattedValue -> placeholder
            parts.append("{}")
    return "".join(parts)


def _key_exprs(expr: ast.AST) -> tuple[set[str], set[str]]:
    """Literal + template keys for one dict-KEY expression (f-strings/+ /format)."""
    keys: set[str] = set()
    templates: set[str] = set()
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        keys.add(expr.value)
    elif isinstance(expr, ast.JoinedStr):
        templates.add(_joined_template(expr))
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        for side in (expr.left, expr.right):
            k, t = _key_exprs(side)
            keys |= k
            templates |= t
    elif (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr == "format"
    ):
        # "eval/{...}/latency_p50".format(key) -> the base already carries {}.
        base_keys, _ = _key_exprs(expr.func.value)
        if len(base_keys) == 1:
            templates.add(next(iter(base_keys)))
    return keys, templates


def _dict_keys(node: ast.Dict) -> tuple[set[str], set[str]]:
    keys: set[str] = set()
    templates: set[str] = set()
    for key in node.keys:
        if key is None:  # **unpacking
            continue
        k, t = _key_exprs(key)
        keys |= k
        templates |= t
    return keys, templates


def _body_asserts_guard(scope: ast.FunctionDef, name: str) -> bool:
    """True when *scope* validates *name* at runtime via assert_registered."""
    for node in _scope_stmt_nodes(scope):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_registered"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id == name
        ):
            return True
    return False


def _resolve_call(call: ast.Call, module: ast.Module) -> tuple[set[str], set[str], bool]:
    """Keys of a module-level helper call; ``(set(), set(), True)`` when the
    helper asserts the dict itself (e.g. log_deploy.build_payload)."""
    if not isinstance(call.func, ast.Name):
        return set(), set(), False
    fn = next(
        (
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef) and node.name == call.func.id
        ),
        None,
    )
    if fn is None:
        return set(), set(), False
    for node in _scope_stmt_nodes(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "assert_registered"
        ):
            return set(), set(), True
    for node in fn.body:
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value.elts[0] if isinstance(node.value, ast.Tuple) else node.value
            return _resolve_arg(value, fn, module)
    return set(), set(), False


def _resolve_name(
    name: str, scope: ast.FunctionDef, module: ast.Module
) -> tuple[set[str], set[str], bool]:
    """Keys of a local Name: the last dict-literal/call assignment plus any
    literal ``name["k"] = v`` / ``name.update({k: v})`` in the same function."""
    keys: set[str] = set()
    templates: set[str] = set()
    base: ast.AST | None = None
    self_validated = False
    for node in _scope_stmt_nodes(scope):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    base = node.value
                elif isinstance(target, ast.Tuple) and any(
                    isinstance(elt, ast.Name) and elt.id == name for elt in target.elts
                ):
                    base = node.value  # metrics, reqs = _measure_endpoint(...)
                elif (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == name
                ):
                    # name["k"] = v — collect the key (also inside if/for bodies)
                    k, t = _key_exprs(target.slice)
                    keys |= k
                    templates |= t
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "update"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == name
            and node.value.args
        ):
            k, t, sv = _resolve_arg(node.value.args[0], scope, module)
            keys |= k
            templates |= t
            self_validated = self_validated or sv
    if isinstance(base, ast.Dict):
        k, t = _dict_keys(base)
        keys |= k
        templates |= t
    elif isinstance(base, ast.Call):
        k, t, sv = _resolve_call(base, module)
        keys |= k
        templates |= t
        self_validated = self_validated or sv
    if keys or templates or self_validated:
        return keys, templates, self_validated
    return set(), set(), _body_asserts_guard(scope, name)


def _resolve_arg(
    arg: ast.AST, scope: ast.FunctionDef | None, module: ast.Module
) -> tuple[set[str], set[str], bool]:
    if isinstance(arg, ast.Dict):
        k, t = _dict_keys(arg)
        return k, t, False
    if isinstance(arg, ast.Call):
        return _resolve_call(arg, module)
    if isinstance(arg, ast.Name) and scope is not None:
        return _resolve_name(arg.id, scope, module)
    return set(), set(), False


def _is_alias_site(relpath: str, receiver: str, source: str) -> bool:
    marker = _WANDB_ALIASES.get((relpath, receiver))
    return marker is not None and marker in source


def _emission_sites(tree: ast.Module, relpath: str, source: str) -> Iterator[tuple[str, ast.Call]]:
    """Yield ``(kind, call)`` for every wandb.log / log_metrics /
    assert_registered call (plus the marker-asserted alias receivers)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id in ("log_metrics", "assert_registered"):
            yield func.id, node
        elif isinstance(func, ast.Attribute) and func.attr == "log":
            receiver = func.value
            if isinstance(receiver, ast.Name) and (
                receiver.id == "wandb" or _is_alias_site(relpath, receiver.id, source)
            ):
                yield "wandb.log", node


def _concrete_key(template: str) -> str:
    """Fill every ``{}`` with a 3-segment eval shape for the hierarchy check."""
    places = template.count("{}")
    if not places:
        return template
    return template.format(*(["model/variant/template"] * places))


# ── test 1 — the registry AST walk ───────────────────────────────────────────


def test_all_wandb_log_keys_registered():  # noqa: PLR0912 — the registry walker is branchy by nature
    """Every ``wandb.log`` / ``log_metrics`` literal key ∈ METRIC_REGISTRY (or
    a hierarchical template); failures name file:line:key."""
    for (relpath, _receiver), marker in _WANDB_ALIASES.items():
        assert marker in (_REPO_ROOT / relpath).read_text(encoding="utf-8"), (
            f"wandb-alias marker missing from {relpath}: {marker!r}"
        )

    problems: list[str] = []
    checked = 0
    for dirpath in (_REPO_ROOT / d for d in _WALK_DIRS):
        if not dirpath.is_dir():
            continue
        for py in sorted(dirpath.rglob("*.py")):
            if py.name.startswith("__"):
                continue
            relpath = py.relative_to(_REPO_ROOT).as_posix()
            source = py.read_text(encoding="utf-8")
            tree = ast.parse(source)
            for kind, call in _emission_sites(tree, relpath, source):
                if kind == "assert_registered":
                    if call.args and isinstance(call.args[0], ast.Dict):
                        keys, templates = _dict_keys(call.args[0])
                    else:
                        continue  # Name arg: validated by assert_registered at runtime
                else:
                    keys, templates, self_validated = _resolve_arg(
                        call.args[0],
                        _enclosing_func(tree, call),
                        tree,
                    )
                    checked += 1
                    if not keys and not templates and not self_validated:
                        problems.append(
                            f"{relpath}:{call.lineno}: unresolvable emission keys — "
                            "extend the resolver or add a documented runtime probe"
                        )
                        continue
                for key in sorted(keys):
                    if not _is_registered(key):
                        problems.append(f"{relpath}:{call.lineno}: unregistered key {key!r}")
                for template in sorted(templates):
                    if not _is_registered(_concrete_key(template)):
                        problems.append(
                            f"{relpath}:{call.lineno}: unregistered template {template!r}"
                        )

    assert checked > 0, "registry walk found no emission sites — walker regression"
    assert not problems, "telemetry contract violations:\n  " + "\n  ".join(problems)


# ── tests 2-7 — behaviour, all offline ───────────────────────────────────────


def _reset_root_logger() -> tuple[list[logging.Handler], int]:
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    root.handlers.clear()
    return saved


def _restore_root_logger(saved: tuple[list[logging.Handler], int]) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.handlers.extend(saved[0])
    root.setLevel(saved[1])


def test_json_log_parse():
    """configure_logging() gives one JSON object per line; calling it twice
    keeps a single root handler (idempotent)."""
    saved = _reset_root_logger()
    try:
        stream = io.StringIO()
        logging.getLogger().addHandler(logging.StreamHandler(stream))
        configure_logging(level=logging.INFO)
        configure_logging(level=logging.INFO)  # idempotent: reuses the handler
        assert len(logging.getLogger().handlers) == 1

        logger = logging.getLogger("test.json.contract")
        logger.info("search started", extra={"event": "search", "run_id": "run-1"})
        logger.warning("timeout %s", "cold", extra={"event": "timeout", "run_id": "run-2"})

        lines = [line for line in stream.getvalue().splitlines() if line.strip()]
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert [p["level"] for p in parsed] == ["INFO", "WARNING"]
        assert all(p["logger"] == "test.json.contract" for p in parsed)
        assert all({"ts", "msg"} <= set(p) for p in parsed)
        assert parsed[0]["event"] == "search"
        assert parsed[0]["run_id"] == "run-1"
        assert parsed[1]["event"] == "timeout"
    finally:
        _restore_root_logger(saved)


def test_cost_formula():
    """estimate_cost_usd is gpu-seconds x rate; the eval cost-per-fix division
    guard (zero F2P passes) never divides by zero."""
    assert estimate_cost_usd(3600.0, 2.0) == 2.0
    assert estimate_cost_usd(1800.0, 3.0) == pytest.approx(1.5)
    # The zero-guard is hoisted into observability/cost.py (code-review N2) so
    # the behaviour is asserted directly, not source-grepped.
    assert cost_per_fix(90.0, 0) == 90.0
    assert cost_per_fix(90.0, 3) == 30.0
    assert_registered({"eval/cost_per_fix": 90.0})


def test_langfuse_noop_without_keys(mocker):
    """No keys -> every trace is a silent no-op; keys + patched client -> SDK
    calls recorded end-to-end."""
    from observability import langfuse

    env = os.environ
    saved = {
        key: env.get(key) for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")
    }
    records: dict[str, list[Any]] = {"started": [], "scores": []}

    class _FakeObservation:
        id = "obs-1"
        trace_id = "trace-1"

        def end(self) -> None:
            records["started"].append("end")

    class _FakeLangfuse:
        def __init__(self, **kwargs: object) -> None:
            records["started"].append(kwargs)

        def start_observation(self, **kwargs: object) -> _FakeObservation:
            records["started"].append(kwargs)
            return _FakeObservation()

        def create_score(self, **kwargs: object) -> None:
            records["scores"].append(kwargs)

    mocker.patch.dict(sys.modules, {"langfuse": types.SimpleNamespace(Langfuse=_FakeLangfuse)})
    mocker.patch.object(langfuse, "_client", None)
    try:
        for key in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"):
            env.pop(key, None)
        assert langfuse._enabled() is False
        assert langfuse._get_client() is None
        langfuse.trace_generation(
            name="n", model="m", prompt="p", completion="c", metadata={"run_id": "r"}
        )
        langfuse.trace_request(
            model="m", template_name=None, ttfbs_ms=None, latency_ms=1.0, output_tokens=5
        )
        assert not records["started"], "no-op without keys must not touch the SDK"

        env["LANGFUSE_PUBLIC_KEY"] = "pk"
        env["LANGFUSE_SECRET_KEY"] = "sk"
        env["LANGFUSE_HOST"] = "https://example.langfuse.com"
        assert langfuse._enabled() is True
        client = langfuse._get_client()
        assert client is langfuse._get_client(), "client is cached"
        assert records["started"][0]["public_key"] == "pk"
        assert records["started"][0]["host"] == "https://example.langfuse.com"

        langfuse.trace_generation(
            name="gen",
            model="qwen3-14b",
            prompt="fix bug",
            completion="done",
            metadata={"run_id": "run-x", "instance_id": "i-1"},
            scores={"f2p": 1.0},
        )
        assert records["scores"] == [
            {
                "name": "f2p",
                "value": 1.0,
                "trace_id": "trace-1",
                "observation_id": "obs-1",
            }
        ]
        langfuse.trace_generation(
            name="gen2", model="m", prompt="p", completion="c", metadata={"run_id": "r"}
        )
        langfuse.trace_request(
            model="m", template_name="tpl", ttfbs_ms=12.0, latency_ms=30.0, output_tokens=9
        )
        assert len(records["started"]) >= 3
    finally:
        for key, value in saved.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        mocker.patch.object(langfuse, "_client", None)


def test_slo_burn(mocker):
    """Attainment is per-key; burn escalates WARN -> ERROR; alerts fire only
    with a live run AND >= min_samples in the window."""
    assert attainment({"ttfb_p50_ms": 600, "cold_start_s": 8}) == {
        "ttfb_p50_ms": 0.0,
        "cold_start_s": 1.0,
    }
    assert attainment({"ttfb_p50_ms": 400}) == {"ttfb_p50_ms": 1.0}
    assert burn_rate([0.0] * 10, 60, 3600) >= 1.0
    assert burn_rate([], 60, 3600) == 0.0
    assert burn_level(2.0, 10) == "WARN"
    assert burn_level(5.0, 10) == "ERROR"
    assert burn_level(0.5, 10) is None
    assert burn_level(9.0, 9) is None  # min-sample guard

    alerts: list[tuple[str, str]] = []

    class _FakeWandb:
        run: object | None = None

        @staticmethod
        def alert(title: str, text: str, **_: object) -> None:
            alerts.append((title, text))

    mocker.patch.dict(sys.modules, {"wandb": _FakeWandb})
    assert maybe_alert_burn(0.5, 10) is False  # level None -> no import side effect
    assert maybe_alert_burn(2.0, 10) is False  # burn WARN but no active run
    _FakeWandb.run = object()
    assert maybe_alert_burn(2.0, 10) is True
    assert maybe_alert_burn(9.0, 9) is False  # sampled-out even with a live run
    assert len(alerts) == 1
    assert "WARN" in alerts[0][0]
    assert SLO_TARGETS == {"ttfb_p50_ms": 500.0, "cold_start_s": 10.0}


def test_deploy_payload(mocker):
    """build_payload emits exactly the registered deploy/* keys (pure, no W&B)."""
    assert build_payload(1, 90.0) == {"deploy/status": 1, "deploy/duration_s": 90.0}
    assert build_payload(0, None) == {"deploy/status": 0}
    with pytest.raises(ValueError, match="status must be 0 or 1"):
        build_payload(2)
    for payload in (build_payload(1, 90.0), build_payload(0, None)):
        assert_registered(payload)

    # deploy/sha is not a registered metric; it reaches the payload only if the
    # registry gains the key (poison-pill guard in build_payload).
    assert "sha" not in METRIC_REGISTRY["deploy"]
    original_deploy = dict(METRIC_REGISTRY["deploy"])
    try:
        METRIC_REGISTRY["deploy"] = dict(original_deploy, sha="commit hash")
        assert build_payload(1, None, "abc123")["deploy/sha"] == "abc123"
        assert_registered(build_payload(1, None, "abc123"))
    finally:
        METRIC_REGISTRY["deploy"] = original_deploy


def test_promote_keys_registered():
    """The six ``promote/*`` decision scalars are registered (spec 4.9)."""
    assert "promote" in METRIC_REGISTRY
    for key in (
        "promote/outcome",
        "promote/f2p_gain",
        "promote/p2p_delta",
        "promote/ci_lower",
        "promote/mcnemar_p",
        "promote/deploy_status",
    ):
        assert _is_registered(key)


def test_panels_spec_subset_of_registry(mocker):
    """Every PANELS metric (wildcards resolved to 'model') is registered; the
    dashboard module's own guard agrees and rejects an unregistered panel."""
    for dashboard, specs in PANELS.items():
        for spec in specs:
            concrete = spec.metric.replace("*", "model")
            assert _is_registered(concrete), (
                f"PANELS[{dashboard!r}] references unregistered metric {spec.metric!r}"
            )
    assert_panels_registered()

    fake_panels = {
        dashboard: [PanelSpec("unregistered/key", "line", "bad")] for dashboard in PANELS
    }
    mocker.patch("observability.dashboards.PANELS", fake_panels)
    with pytest.raises(KeyError, match="unregistered metric key"):
        assert_panels_registered()


# ── walker ground truth (keeps the resolver honest) ──────────────────────────


def test_registry_self_consistency():
    """Flat keys and the eval hierarchical fill round-trip through
    ``_is_registered`` — the exact predicate test 1 validates against."""
    for domain, metrics in METRIC_REGISTRY.items():
        for metric in metrics:
            if metric.startswith("{key}/"):
                # the hierarchical placeholder, filled with a 3-segment shape
                concrete = f"eval/{metric}".replace("{key}/", "m/v/t/")
                assert _is_registered(concrete)
            else:
                assert _is_registered(f"{domain}/{metric}")
    assert _is_registered("eval/qwen3-14b/baseline_14b/template_v1/latency_p50")
    assert _is_registered("eval/qwen3-14b/baseline_14b/template_v1/latency_p95")
    assert _is_registered("eval/f2p_rate")
    assert _is_registered("nope/missing") is False
    assert _is_registered("no/n/n") is False
    assert _is_registered("eval/only/two/segments_missing_suffix") is False
