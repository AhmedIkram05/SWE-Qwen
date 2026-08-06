"""Shared prompt construction for the eval harness and the Phase 6 serving API.

Extracted from ``evaluation/inference.py`` (Phase 6, Wave 1) so served
inference and eval F2P can never drift: one module, two consumers.  Pure
stdlib + lazy imports only — importable on macOS without modal, vllm, or GPU
libraries (``transformers``/``wandb``/``training.prompt_loader`` are imported
inside the functions that need them, exactly as in the source module).

Eval-side patch bridge: Phase 5 tests and the harness monkeypatch module state
on ``evaluation.inference`` (e.g. ``monkeypatch.setattr(inf, "_TOKENIZER_CACHE",
{...})``).  The moved functions resolve patchable state through ``_eval()`` so
those patches keep landing, while standalone serving (``evaluation.inference``
not imported) falls back to this module's own bindings.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evaluation.config import EvalConfig
    from evaluation.schema import EvalInput

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_PROMPTS_DIR = _REPO_ROOT / "training" / "prompts"

DEFAULT_TEMPLATE = "chat"
_DEFAULT_HF_ID = "Qwen/Qwen3-14B"

_DIFF_FILE_RE = re.compile(r"^diff --git a/\S+ b/(\S+)", re.MULTILINE)


# ── Eval-side patch bridge ─────────────────────────────────────────────────
# ``evaluation.inference`` re-exports every name below (X as X).  Phase 5 tests
# monkeypatch state on that module (``monkeypatch.setattr(inf, "_TOKENIZER_CACHE",
# {...})``) and then call the moved functions; reading patchable state through
# ``_eval()`` keeps those patches effective.  In production the re-exported
# attributes ARE this module's objects, so behavior is identical; when
# ``evaluation.inference`` is not loaded (plain Phase 6 serving) we read our
# own bindings.


def _eval() -> Any:
    """Return the module whose state the shared functions should read from.

    ``evaluation.inference`` when loaded (the patchable facade), else this
    module's own namespace.
    """
    return sys.modules.get("evaluation.inference") or sys.modules[__name__]


# ── Thinking-mode gate ─────────────────────────────────────────────────────
# Qwen3-14B thinks by default: it emits a long reasoning preamble before any
# patch, which eats the whole 2048-token cap (observed in a run checkpoint:
# 9851 chars of "Okay, let's see. The problem is that when using
# GaussianMixture..." and extract_patch got nothing).  The only reliable gate
# is the model's OWN chat template: with ``enable_thinking=False`` it
# pre-fills an empty ``<think>\n\n</think>`` block after
# ``<|im_start|>assistant``, which Qwen3 is trained to treat as "answer
# directly".  Raw strings passed to ``llm.generate`` bypass the chat
# template, so every rendered prompt is re-wrapped through
# ``tokenizer.apply_chat_template`` first.  (vLLM's ``LLM.generate`` has no
# chat_template_kwargs by design; ``LLM.chat`` does, but this keeps the
# existing LoRA path untouched.)
_TOKENIZER_CACHE: dict[str, Any] = {}


def no_think_wrap(hf_id: str, prompt: str) -> str:
    """Wrap a rendered prompt in the model's chat template, thinking OFF."""
    state = _eval()
    if hf_id not in state._TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        state._TOKENIZER_CACHE[hf_id] = AutoTokenizer.from_pretrained(hf_id)
    return state._TOKENIZER_CACHE[hf_id].apply_chat_template(  # type: ignore[no-any-return]
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


_no_think_wrap = no_think_wrap  # backward-compat alias


def _files_from_diff(patch: str) -> list[str]:
    """Extract the changed file paths (``b/`` side) from a unified diff."""
    if not patch:
        return []
    return [match.group(1) for match in _DIFF_FILE_RE.finditer(patch)]


# SWE-bench problem statements often name the files they touch (``django/
# db/models/fields/__init__.py``, ``path/to/file.py``).  Used to seed file
# snippets when no context_files are known.
_PATH_RE = re.compile(r"[A-Za-z0-9_./-]+\.py")


@lru_cache(maxsize=4096)
def fetch_raw_file(repo: str, base_sha: str, path: str) -> str | None:
    """Fetch one file at ``base_sha`` from GitHub raw.

    SWE-bench base commits are real GitHub objects, so the raw endpoint
    resolves them.  Returns None on any failure — snippets are best-effort.
    """
    url = f"https://raw.githubusercontent.com/{repo}/{base_sha}/{path}"
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return resp.read().decode("utf-8", errors="replace")  # type: ignore[no-any-return]
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError):
        return None


_fetch_raw_file = fetch_raw_file  # backward-compat alias


def file_snippets(
    repo: str,
    base_sha: str,
    paths: list[str],
    max_files: int = 10,
    max_lines: int = 500,
) -> list[dict[str, str]]:
    """Best-effort contents for the candidate files, deduped and capped.

    ponytail: 500 lines/file, 10 files — enough to see the code around a
    hunk without blowing the input budget on vendored giants.
    """
    state = _eval()
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for path in paths:
        if not path or path in seen or len(out) >= max_files:
            continue
        seen.add(path)
        content = state._fetch_raw_file(repo, base_sha, path)
        if content is None:
            continue
        lines = content.splitlines()
        if len(lines) > max_lines:
            content = "\n".join(lines[:max_lines]) + "\n# ... (file truncated)"
        out.append({"path": path, "content": content})
    return out


_file_snippets = file_snippets  # backward-compat alias


# ── Few-shot golden examples ──────────────────────────────────────────────
# Gold patch diffs for few-shot prompting come from the run's GCS golden
# (source of truth, see _ensure_golden).  A couple of same-repo examples teach
# the model the repo's diff shape without any training — Qwen3-14B zero-shot
# still fabricates placeholder headers (``revision 12345``) and guessed line
# numbers.
_GOLDEN_PATH = _REPO_ROOT / "data" / "golden.jsonl"
# Override set from GCS when a dataset run id is known; the image-baked
# data/golden.jsonl is stale once the pipeline re-runs.
_GOLDEN_SOURCE: Path | None = None
# {repo: [(instance_id, patch_diff), ...]} — built lazily, capped per repo so
# the 57 MB file doesn't become 200 MB of strings in RAM.
_GOLDEN_INDEX: dict[str, list[tuple[str, str]]] | None = None


def _ensure_golden(dataset_run_id: str) -> Path:
    """Return the golden file for a pipeline run, downloading from GCS.

    Source of truth: ``datasets/{dataset_run_id}/swebench/golden.jsonl`` on
    the public ``swe-qwen-datasets`` bucket (no creds needed — urllib only,
    this also runs inside the Modal container which lacks
    google-cloud-storage).  Cached at ``data/{dataset_run_id}/swebench/``
    relative to the repo root.  Falls back to the baked local file when GCS
    is unreachable (best-effort, matching ``_golden_patches`` semantics).
    """
    state = _eval()
    dst = state._REPO_ROOT / "data" / dataset_run_id / "swebench" / "golden.jsonl"
    if dst.is_file():
        return dst  # type: ignore[no-any-return]
    url = (
        "https://storage.googleapis.com/swe-qwen-datasets/datasets/"
        f"{dataset_run_id}/swebench/golden.jsonl"
    )
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dst)
        logger.info("downloaded golden from GCS: %s", url)
    except (urllib.error.URLError, OSError):
        logger.warning("could not fetch golden from %s — using local fallback", url)
        return state._GOLDEN_PATH  # type: ignore[no-any-return]
    return dst  # type: ignore[no-any-return]


def golden_patches(
    repo: str,
    exclude_instance_id: str | None = None,
    max_examples: int = 2,
    max_lines: int = 150,
) -> list[str]:
    """Gold patch diffs for ``repo``, for few-shot prompting.

    Best-effort: parses the golden file once per process (GCS-synced via
    ``_ensure_golden`` when a dataset run id is set, else the baked local
    file), skips the example's own instance (no leakage — golden ids match
    eval ids), and truncates each patch to ``max_lines``.  Missing file → [].
    """
    state = _eval()
    if state._GOLDEN_INDEX is None:
        index: dict[str, list[tuple[str, str]]] = {}
        try:
            with (state._GOLDEN_SOURCE or state._GOLDEN_PATH).open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    patch = rec.get("patch_diff")
                    if not patch:
                        continue
                    key = str(rec.get("repo") or "")
                    per_repo = index.setdefault(key, [])
                    if len(per_repo) < 4:  # noqa: PLR2004 — ponytail: cap index memory
                        per_repo.append((str(rec.get("instance_id") or ""), patch))
        except OSError:
            index = {}
        state._GOLDEN_INDEX = index
    out: list[str] = []
    for instance_id, patch in state._GOLDEN_INDEX.get(repo, []):
        if exclude_instance_id and instance_id == exclude_instance_id:
            continue
        lines = patch.splitlines()
        if len(lines) > max_lines:
            patch = "\n".join(lines[:max_lines]) + "\n# ... (diff truncated)"  # noqa: PLW2901
        out.append(patch)
        if len(out) >= max_examples:
            break
    return out


_golden_patches = golden_patches  # backward-compat alias


def render_patch_prompt(
    example: EvalInput,
    template_name: str = DEFAULT_TEMPLATE,
    template_dir: str | Path | None = None,
    include_file_contents: bool = False,
    example_patches: list[str] | None = None,
) -> str:
    """Render the inference prompt for one eval example.

    Args:
        example: Eval input (issue, repo, test names, test patch).
        template_name: One of ``chat`` (default), ``system``, ``user``,
            ``assistant`` — see ``training/prompts/*.j2``.
        template_dir: Prompt template directory (defaults to
            ``training/prompts/`` next to the repo checkout).
        include_file_contents: When True, fetch the candidate files'
            contents at ``base_sha`` from GitHub raw and embed them in the
            prompt (the model fabricates diffs without seeing real code).
        example_patches: Gold patch diffs to show as few-shot format
            examples.  Defaults to same-repo examples from the run's GCS
            golden (``_ensure_golden``; best-effort; [] when unavailable).

    Returns:
        The rendered prompt string.
    """
    from training.prompt_loader import PromptLoader

    loader = PromptLoader(template_dir=template_dir or _PROMPTS_DIR)
    metadata: dict[str, Any] = example.metadata or {}
    language = str(metadata.get("language") or "Python")
    task_description = str(
        metadata.get("task_description") or "Fix the bug described in the issue below."
    )
    style_guide = str(
        metadata.get("style_guide")
        or "Follow the repository's existing code style and test conventions."
    )
    context_files: list[str] = [str(f) for f in metadata.get("context_files", [])]
    test_files = _files_from_diff(example.test_patch)

    context_snippets: list[dict[str, str]] = []
    if include_file_contents and template_name in ("chat", "user"):
        # ponytail: context_files is never populated today, so seed candidates
        # from the test patch + file paths mentioned in the issue body.
        candidates = list(context_files)
        if not candidates:
            candidates = _PATH_RE.findall(example.issue_body or "")
        candidates += test_files
        context_snippets = _file_snippets(example.repo, example.base_sha, candidates)

    if example_patches is None and template_name in ("chat", "user"):
        state = _eval()
        example_patches = state._golden_patches(
            example.repo, exclude_instance_id=example.instance_id
        )

    if template_name == "chat":
        system_prompt = loader.render(
            "system",
            language=language,
            task_description=task_description,
            style_guide=style_guide,
        )
        user_prompt = loader.render(
            "user",
            issue_title=example.instance_id,
            issue_body=example.issue_body,
            repo_name=example.repo,
            repo_domain=example.repo_domain,
            context_files=context_files,
            test_files=test_files,
            context_snippets=context_snippets,
            example_patches=example_patches or [],
        )
        return loader.render_chat(
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

    if template_name == "system":
        return loader.render(
            "system",
            language=language,
            task_description=task_description,
            style_guide=style_guide,
        )

    if template_name == "user":
        return loader.render(
            "user",
            issue_title=example.instance_id,
            issue_body=example.issue_body,
            repo_name=example.repo,
            repo_domain=example.repo_domain,
            context_files=context_files,
            test_files=test_files,
            context_snippets=context_snippets,
            example_patches=example_patches or [],
        )

    if template_name == "assistant":
        logger.warning(
            "assistant template is a training-target template; rendering with empty sections"
        )
        return loader.render("assistant", analysis="", plan="", code_changes="")

    raise ValueError(
        f"unknown prompt template {template_name!r}; available: {loader.available_templates}"
    )


def resolve_hf_id(model_name: str = "qwen3-14b") -> str:
    """Resolve a ``config/models.yaml`` registry key to a Hugging Face model ID.

    Args:
        model_name: Key from ``models.yaml``, or a full ``owner/model`` ID.

    Returns:
        The Hugging Face model ID; a value containing ``/`` is returned as-is.
    """
    if "/" in model_name:
        return model_name
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not available — cannot resolve %s from models.yaml", model_name)
        return _DEFAULT_HF_ID
    state = _eval()
    path = state._REPO_ROOT / "config" / "models.yaml"
    if not path.is_file():
        logger.warning("model registry not found at %s — using default %s", path, _DEFAULT_HF_ID)
        return _DEFAULT_HF_ID
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models: dict[str, Any] = data.get("models", {}) or {}
    entry: dict[str, Any] = models.get(model_name, {}) or {}
    hf_id: str = entry.get("hf_id", _DEFAULT_HF_ID) or _DEFAULT_HF_ID
    return hf_id


def resolve_adapter_path(variant: str, config: EvalConfig | None = None) -> str | None:
    """Resolve the LoRA adapter directory for a variant.

    Checks ``models/checkpoints/{variant}`` locally first, then downloads the
    W&B artifact ``{entity}/{project}/model-qwen3-14b-{variant}:latest``.
    Any failure returns ``None`` so the baseline model path always works.

    Args:
        variant: Variant key (e.g. ``"baseline_14b"``).
        config: Eval config (defaults to env-driven ``EvalConfig()``).

    Returns:
        Absolute path to the adapter directory, or ``None`` if unavailable.
    """
    if config is None:
        # Lazy + branch-scoped: the serving image does not ship evaluation/,
        # and serve always passes a config, so this import never runs there.
        from evaluation.config import EvalConfig

        config = EvalConfig()
    state = _eval()
    local_dirs = (
        state._REPO_ROOT / "models" / "checkpoints" / variant,
        Path("models/checkpoints") / variant,
    )
    for candidate in local_dirs:
        if (candidate / "adapter_config.json").is_file():
            logger.info("found local LoRA adapter for %s at %s", variant, candidate)
            return str(candidate)
    try:
        import wandb

        api = wandb.Api()
        artifact_name = config.lora_artifact_pattern.format(variant=variant)
        artifact_ref = f"{config.wandb_entity}/{config.wandb_project}/{artifact_name}:latest"
        artifact = api.artifact(artifact_ref)
        local_dir = artifact.download()
        logger.info("downloaded LoRA adapter %s to %s", artifact_name, local_dir)
        return str(local_dir)
    except Exception:
        logger.warning(
            "no LoRA adapter found for variant %s — falling back to base model",
            variant,
            exc_info=True,
        )
        return None
