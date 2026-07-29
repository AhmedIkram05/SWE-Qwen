"""Prompt template loading, rendering, and W&B Artifact versioning.

Provides ``PromptLoader`` for loading Jinja2 templates from ``training/prompts/``,
rendering them with context, and versioning them via W&B Artifacts for
reproducibility.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import jinja2
import wandb

logger = logging.getLogger(__name__)

# Default template directory relative to this file
_DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompts"

# W&B Artifact type for prompt templates
_PROMPT_ARTIFACT_TYPE = "prompt_template"


class PromptLoader:
    """Load, render, and version prompt templates.

    Args:
        template_dir: Directory containing ``.j2`` template files.
            Defaults to ``training/prompts/``.

    Usage::

        loader = PromptLoader()
        prompt = loader.render("chat", system_prompt="...", user_prompt="...")
        loader.log_to_wandb_artifact(run_id="abc", version="1.0")
    """

    def __init__(self, template_dir: str | Path | None = None):
        self._template_dir = Path(template_dir or _DEFAULT_TEMPLATE_DIR)
        if not self._template_dir.is_dir():
            raise FileNotFoundError(f"Prompt template directory not found: {self._template_dir}")

        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self._template_dir)),
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=jinja2.Undefined,
        )

    # ── Template listing ──────────────────────────────────────────────────────

    @property
    def available_templates(self) -> list[str]:
        """Return list of available template names (without ``.j2`` suffix)."""
        return sorted(p.stem for p in self._template_dir.glob("*.j2"))

    def get_template_source(self, template_name: str) -> str:
        """Return raw source of a template."""
        path = self._template_dir / f"{template_name}.j2"
        if not path.exists():
            raise FileNotFoundError(
                f"Template {template_name!r} not found at {path}. "
                f"Available: {self.available_templates}"
            )
        return path.read_text(encoding="utf-8")

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render(self, template_name: str, **kwargs: Any) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of template (without ``.j2`` suffix).
            **kwargs: Variables to pass to the template.

        Returns:
            Rendered string.
        """
        template = self._env.get_template(f"{template_name}.j2")
        return template.render(**kwargs)

    def render_chat(
        self,
        system_prompt: str = "",
        messages: list[dict[str, str]] | None = None,
        user_prompt: str = "",
    ) -> str:
        """Render the ``chat`` template — the primary training format.

        Args:
            system_prompt: Optional system-level instruction.
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            user_prompt: Fallback raw user prompt (used when ``messages`` is empty).

        Returns:
            Rendered chat string.
        """
        return self.render(
            "chat",
            system_prompt=system_prompt,
            messages=messages or [],
            user_prompt=user_prompt,
        )

    # ── W&B Artifact versioning ──────────────────────────────────────────────

    def log_to_wandb_artifact(
        self,
        version: str = "1.0",
    ) -> wandb.Artifact | None:
        """Upload all prompt templates as a W&B Artifact.

        Args:
            version: Semantic version string for the artifact.

        Returns:
            The created W&B Artifact, or ``None`` if no active run.
        """
        if wandb.run is None:
            logger.warning("No active W&B run; skipping prompt artifact upload")
            return None

        artifact = wandb.Artifact(
            name="prompt-templates",
            type=_PROMPT_ARTIFACT_TYPE,
            metadata={"version": version, "template_count": len(self.available_templates)},
        )

        for template_name in self.available_templates:
            path = self._template_dir / f"{template_name}.j2"
            artifact.add_file(str(path), name=f"{template_name}.j2")

        wandb.log_artifact(artifact)
        artifact.wait()
        time.sleep(1)
        logger.info(
            "Logged prompt templates artifact v%s (%d templates)",
            version,
            len(self.available_templates),
        )
        return artifact

    @staticmethod
    def load_from_wandb_artifact(
        artifact_ref: str,
        download_dir: str | Path | None = None,
    ) -> PromptLoader:
        """Load prompt templates from a W&B Artifact.

        Args:
            artifact_ref: W&B artifact reference
                (e.g. ``"entity/project/prompt-templates:v1"``).
            download_dir: Local directory to download templates into.
                Defaults to a temp directory.

        Returns:
            ``PromptLoader`` initialized from the downloaded templates.
        """
        api = wandb.Api()
        artifact = api.artifact(artifact_ref)
        dl_dir = Path(download_dir or artifact.download())
        return PromptLoader(template_dir=dl_dir)
