#!/usr/bin/env python3
"""Build the four W&B dashboards as code from the PANELS spec (plan §5.6, decision 5).

Consumes ``observability.dashboards.PANELS`` (single source of truth) and
creates one project **Workspace** per dashboard — Training, Evaluation,
Serving, Infrastructure/Cost — via the ``wandb-workspaces`` Public Preview
API (``ws.Workspace(...).save()``, which also creates the project workspace
itself). Panel types map mechanically: ``line`` → ``wr.LinePlot`` (with a
constant ``custom_expressions`` target line when the spec marks a ``target``,
e.g. the S9 10s cold-start limit), ``bar`` → ``wr.BarPlot``, ``custom`` →
``wr.ScalarChart``. No reports are created — workspaces only, per the plan.

Run::

    uv run python scripts/build_dashboards.py --project swe-qwen [--entity ...]

Requires ``wandb login`` (or ``WANDB_API_KEY``) and the optional dependency
``wandb-workspaces>=0.4.4`` (``uv sync --extra dashboards``); both are imported
lazily so the script fails with a clear message. Prints the URL of each saved
workspace. If the Public Preview API breaks, the documented fallback is to
build the same panels manually in the UI from PANELS.
"""

from __future__ import annotations

import argparse
import sys
from types import ModuleType

from observability.dashboards import PanelSpec


def _panel(wr: ModuleType, spec: PanelSpec) -> object:
    """Mechanical PANELS entry -> wandb-workspaces panel (plan §5.6)."""
    if spec.type == "line":
        kwargs: dict[str, object] = {}
        if spec.target is not None:
            # A constant expression renders as a horizontal target line
            # (e.g. the S9 10s cold-start limit).
            kwargs["custom_expressions"] = [str(spec.target)]
        return wr.LinePlot(x="Step", y=[spec.metric], title=spec.title, **kwargs)
    if spec.type == "bar":
        kwargs = {}
        if spec.aggregate is not None:
            kwargs["groupby_aggfunc"] = spec.aggregate
        return wr.BarPlot(metrics=[spec.metric], title=spec.title, **kwargs)
    if spec.type == "custom":
        kwargs = {}
        if spec.aggregate is not None:
            kwargs["groupby_aggfunc"] = spec.aggregate
        return wr.ScalarChart(metric=spec.metric, title=spec.title, **kwargs)
    raise ValueError(f"no as-code mapping for panel type {spec.type!r} ({spec.metric!r})")


def _default_entity() -> str | None:
    """Resolve the W&B entity from the logged-in API client."""
    try:
        import wandb

        return wandb.Api().default_entity
    except Exception:
        return None


def build(project: str, entity: str | None) -> int:
    """Create one workspace per dashboard from PANELS. Returns exit code."""
    from observability.dashboards import PANELS, assert_panels_registered

    assert_panels_registered()

    try:
        import wandb_workspaces.reports.v2 as wr
        import wandb_workspaces.workspaces as ws
    except ImportError:
        print(
            "ERROR: wandb-workspaces is not installed — run `uv sync --extra dashboards` "
            "(or `pip install wandb-workspaces>=0.4.4`).",
            file=sys.stderr,
        )
        return 1

    entity = entity or _default_entity()
    if entity is None:
        print(
            "ERROR: could not determine the W&B entity — pass --entity or run `wandb login`.",
            file=sys.stderr,
        )
        return 1

    for name, specs in PANELS.items():
        workspace_name = name.replace("/", "-")  # URL-safe workspace name
        panels = [_panel(wr, spec) for spec in specs]
        section = ws.Section(name=name, panels=panels, is_open=True)
        workspace = ws.Workspace(
            entity=entity,
            project=project,
            name=workspace_name,
            sections=[section],
        )
        print(
            f"Creating workspace {workspace_name!r} in {entity}/{project} "
            f"({len(panels)} panels: {', '.join(s.type for s in specs)})..."
        )
        try:
            workspace.save()
        except Exception as exc:
            print(f"ERROR: failed to save workspace {workspace_name!r}: {exc}", file=sys.stderr)
            return 1
        print(f"  {workspace_name}: {workspace.url}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the 4 SWE-Qwen dashboards as code from observability.dashboards.PANELS"
    )
    parser.add_argument("--project", default="swe-qwen", help="W&B project name")
    parser.add_argument("--entity", help="W&B entity (username or team; defaults to your account)")
    args = parser.parse_args()
    return build(args.project, args.entity)


if __name__ == "__main__":
    sys.exit(main())
