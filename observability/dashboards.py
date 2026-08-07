"""PANELS spec: the versioned dashboard contract (plan §5.6, decision 5).

Single source of truth for the four W&B dashboards (Training, Evaluation,
Serving, Infrastructure/Cost). Every ``PanelSpec.metric`` must be a registered
key in ``observability.metrics.METRIC_REGISTRY`` — enforce with
``assert_panels_registered()``. Both build paths (``scripts/build_dashboards.py``
as-code, manual UI fallback) are mechanical from this spec.

Eval's hierarchical pattern is expressed as ``eval/*/latency_p50``; the ``*``
wildcard stands for the harness segment shape, so it renders as
``eval/{model}/{variant}/{template}/latency_p50`` (e.g.
``eval/qwen3-14b/baseline_14b/template_v1/latency_p50``).
"""

from dataclasses import dataclass
from typing import Literal

# The four panel types the plan allows. Only "line", "bar" and "custom" are in
# use; "run-table" is reserved (no spec currently needs it).
_PANEL_TYPES = ("line", "bar", "run-table", "custom")
_AGG_FUNCS = ("mean", "min", "max", "median", "sum", "samples")

PanelType = Literal["line", "bar", "run-table", "custom"]
AggFunc = Literal["mean", "min", "max", "median", "sum", "samples"]


@dataclass(frozen=True)
class PanelSpec:
    """One panel: a registered metric key + how to render it.

    ``target`` marks a horizontal target line (e.g. the S9 10s cold-start
    limit); ``aggregate`` overrides the cross-run aggregation for bar/custom
    panels (e.g. ``"sum"`` for cumulative cost across runs).
    """

    metric: str
    type: PanelType
    title: str
    target: float | None = None
    aggregate: AggFunc | None = None


PANELS: dict[str, list[PanelSpec]] = {
    "Training": [
        PanelSpec("train/loss", "line", "Training Loss"),
        PanelSpec("train/lr", "line", "Learning Rate"),
        PanelSpec("train/grad_norm", "line", "Gradient Norm"),
        PanelSpec("train/gpu_util", "line", "GPU Utilization"),
        PanelSpec("train/cost_usd", "line", "Training Cost (USD)"),
    ],
    "Evaluation": [
        PanelSpec("eval/f2p_rate", "line", "Fail-to-Pass Rate"),
        PanelSpec("eval/p2p_rate", "line", "Pass-to-Pass Rate"),
        # Wildcard = eval/{model}/{variant}/{template}/latency_p50 (harness L793).
        PanelSpec("eval/*/latency_p50", "line", "Eval Segment Latency p50 (ms)"),
        PanelSpec("eval/*/latency_p95", "line", "Eval Segment Latency p95 (ms)"),
        PanelSpec("eval/cost_per_fix", "bar", "Cost per Fix (USD)"),
        PanelSpec("eval/num_examples", "bar", "Examples Evaluated"),
    ],
    "Serving": [
        PanelSpec("serve/request_count", "line", "Request Count"),
        PanelSpec("serve/error_rate", "line", "Error Rate"),
        PanelSpec("serve/ttfb_p50_ms", "line", "TTFB p50 (ms)"),
        PanelSpec("serve/ttfb_p95_ms", "line", "TTFB p95 (ms)"),
        PanelSpec("serve/latency_p50_ms", "line", "Latency p50 (ms)"),
        PanelSpec("serve/tokens_per_sec", "line", "Tokens per Second"),
        PanelSpec("serve/gpu_util", "line", "GPU Utilization"),
        PanelSpec("serve/cost_usd", "line", "Serving Cost (USD)"),
    ],
    "Infrastructure/Cost": [
        PanelSpec(
            "cost/cost_usd",
            "bar",
            "Cost (USD, cumulative across runs)",
            aggregate="sum",
        ),
        PanelSpec(
            "cost/gpu_seconds",
            "bar",
            "GPU-Seconds (cumulative across runs)",
            aggregate="sum",
        ),
        PanelSpec("cost/rate_per_hour", "line", "GPU Rate (USD/hour)"),
        PanelSpec("eval/cost_per_fix", "bar", "Cost per Fix (USD)"),
        # S9 target: cold start < 10s (plan §5.6) — rendered as a target line.
        PanelSpec(
            "serve/cold_start_s",
            "line",
            "Cold Start vs S9 Target (10s)",
            target=10.0,
        ),
        PanelSpec("deploy/status", "custom", "Deploy Status"),
        PanelSpec("deploy/duration_s", "line", "Deploy Duration (s)"),
    ],
}


def assert_panels_registered() -> None:
    """Assert every PANELS metric key is covered by METRIC_REGISTRY.

    Flat keys must match exactly; eval hierarchical wildcards
    (``eval/*/latency_p50``) are resolved to a concrete segment and validated
    with the same pattern logic the registry uses (``_is_eval_hierarchical``).
    Raises ``KeyError`` naming the first offending key; returns ``None``.
    """
    from observability.metrics import _is_registered

    for dashboard, panels in PANELS.items():
        for spec in panels:
            # A wildcard stands for one segment (eval/{model}/...); substituting
            # a literal keeps the key on the registry's hierarchical shape.
            concrete = spec.metric.replace("*", "model")
            if not _is_registered(concrete):
                raise KeyError(
                    f"PANELS[{dashboard!r}] references unregistered metric key: {spec.metric!r}"
                )
