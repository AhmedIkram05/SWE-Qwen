"""Root conftest: early setup for all test suites."""

from __future__ import annotations

import contextlib
import os
import sys
import types
from typing import Any

# ── Fake Modal module ──────────────────────────────────────────────────────
#
# ``evaluation.test_runner`` and ``evaluation.inference`` import ``modal`` and
# call ``modal.Volume.from_name()``, ``modal.App(...)``, ``modal.Image.from_registry``,
# ``modal.Secret.from_name()`` *at module level* — each contacts Modal cloud.
# Replace ``sys.modules["modal"]`` with a no-op stand-in before any test file
# imports those modules.

# ponytail: global lock — hf_xet Rust library panics when another trace
# subscriber is already set. HuggingFace recommend HF_HUB_DISABLE_XET=1 for
# environments like test suites where tracing is not needed.
# Upgrade when hf_xet supports setting a subscriber without panic.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


class _FakeModalApp:
    """Stand-in for ``modal.App``: no cloud, stores name, passthrough
    decorator so ``@app.function`` wraps work locally."""

    def __init__(self, name: str | None = None, label: str | None = None) -> None:
        self._name = name
        self._label = label

    def function(self, *args: Any, **kwargs: Any) -> Any:
        """Passthrough decorator: decorated function runs directly.

        Returns a wrapper that has a .local() method for compatibility
        with Modal's Function.local() API.
        """
        _ = kwargs

        def _deco(fn: Any) -> Any:
            fn._modal_function = True

            # Create a wrapper with .local() method for compatibility
            class _ModalFunctionWrapper:
                def __init__(self, func):
                    self._func = func
                    # Copy all attributes from the original function
                    for attr in dir(func):
                        if not attr.startswith("_"):
                            with contextlib.suppress(AttributeError, TypeError):
                                setattr(self, attr, getattr(func, attr))

                def local(self, *a, **k):
                    """Call the function directly (no container)."""
                    return self._func(*a, **k)

                def remote(self, *a, **k):
                    """Alias for local - no container."""
                    return self._func(*a, **k)

            return _ModalFunctionWrapper(fn)

        # @app.function (no parens) → fn is first positional arg; return deco
        if args and callable(args[0]):
            return _deco(args[0])
        return _deco

    def cls(self, *args: Any, **kwargs: Any) -> Any:
        """Passthrough class decorator to mirror ``modal.App.cls``."""
        _ = kwargs

        def _deco(cls_: Any) -> Any:
            return cls_

        if args and isinstance(args[0], type):
            return _deco(args[0])
        return _deco

    def local_entrypoint(self, *args: Any, **kwargs: Any) -> Any:
        """Passthrough decorator for ``@app.local_entrypoint()``."""
        _ = kwargs

        def _deco(fn: Any) -> Any:
            fn._modal_local_entrypoint = True
            return fn

        if args and callable(args[0]):
            return _deco(args[0])
        return _deco

    def run(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs


class _FakeRetries:
    """Stand-in for ``modal.Retries`` config object."""

    def __init__(
        self, max_retries: int, backoff_coefficient: float = 2.0, initial_delay: float = 10.0
    ) -> None:
        self.max_retries = max_retries
        self.backoff_coefficient = backoff_coefficient
        self.initial_delay = initial_delay


class _FakeConcurrency:
    """Stand-in for ``modal.concurrent`` decorator config object."""

    def __init__(self, max_inputs: int = 16, **kwargs: Any) -> None:
        self.max_inputs = max_inputs
        self._kwargs = kwargs


def _passthrough_decorator(*args: Any, **kwargs: Any) -> Any:
    """Generic passthrough accepting both ``@x`` and ``@x(...)`` forms."""
    _ = kwargs

    if args and callable(args[0]):
        return args[0]
    if args and isinstance(args[0], type):
        return args[0]

    def _deco(target: Any) -> Any:
        return target

    return _deco


class _FakeVolume:
    @staticmethod
    def from_name(name: str, *, create_if_missing: bool = True) -> _FakeVolume:
        _ = name, create_if_missing
        return _FakeVolume()

    @staticmethod
    def persisted(name: str) -> _FakeVolume:
        _ = name
        return _FakeVolume()


class _FakeSecret:
    @staticmethod
    def from_name(name: str) -> _FakeSecret:
        _ = name
        return _FakeSecret()


class _FakeImage:
    """Minimal fake — never contacts container registry."""

    def __init__(self) -> None:
        self._dockerfile: str | None = None
        self._commands: list[str] = []

    @staticmethod
    def from_registry(
        base: str,
        **kwargs: Any,
    ) -> _FakeImage:
        _ = base, kwargs
        return _FakeImage()

    @staticmethod
    def debian_slim(python_version: str = "3.11") -> _FakeImage:
        _ = python_version
        return _FakeImage()

    def apt_install(self, *packages: str, **kwargs: Any) -> _FakeImage:
        _ = packages, kwargs
        return self

    def pip_install(self, *packages: str, **kwargs: Any) -> _FakeImage:
        _ = packages, kwargs
        return self

    def run_commands(self, *commands: str, **kwargs: Any) -> _FakeImage:
        _ = commands, kwargs
        return self

    def env(self, vars: dict[str, str] | None = None, **kwargs: str) -> _FakeImage:
        _ = vars, kwargs
        return self

    def entrypoint(self, *args: str) -> _FakeImage:
        _ = args
        return self

    def workdir(self, path: str) -> _FakeImage:
        _ = path
        return self

    def copy(self, local_path: str, remote_path: str) -> _FakeImage:
        _ = local_path, remote_path
        return self

    def add_local_dir(self, local_path: str, *, remote_path: str, copy: bool = True) -> _FakeImage:
        _ = local_path, remote_path, copy
        return self

    def add_local_file(self, local_path: str, *, remote_path: str, copy: bool = True) -> _FakeImage:
        _ = local_path, remote_path, copy
        return self

    def add_local_python_source(self, path: str) -> _FakeImage:
        _ = path
        return self

    def run_function(self, fn: Any, **kwargs: Any) -> _FakeImage:
        _ = fn, kwargs
        return self


class _FakeMount:
    def __init__(self, local_dir: str, remote_dir: str, **kwargs: Any) -> None:
        _ = local_dir, remote_dir, kwargs


class _FakeNetworkFileSystem:
    @staticmethod
    def from_name(name: str, *, create_if_missing: bool = True) -> _FakeNetworkFileSystem:
        _ = name, create_if_missing
        return _FakeNetworkFileSystem()


_fake_modal = types.ModuleType("modal")
_fake_modal.App = _FakeModalApp
_fake_modal.Volume = _FakeVolume
_fake_modal.Secret = _FakeSecret
_fake_modal.Image = _FakeImage
_fake_modal.Mount = _FakeMount
_fake_modal.NetworkFileSystem = _FakeNetworkFileSystem
_fake_modal.Retries = _FakeRetries
_fake_modal.concurrent = _passthrough_decorator
_fake_modal.enter = _passthrough_decorator
_fake_modal.asgi_app = _passthrough_decorator


class _FakeEnableOutput:
    def __enter__(self) -> _FakeEnableOutput:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


_fake_modal.enable_output = _FakeEnableOutput

# Prevent *any* cloud connection attempt at import.
sys.modules["modal"] = _fake_modal

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _disable_modal_batch() -> None:
    """Disable Modal execution paths → forces fallback routing.

    Existing integration tests monkeypatch ``_run_tests`` but not the Modal
    batch/swebench paths. This fixture ensures ``_run_tests_batch_modal`` and
    ``_run_tests_swebench`` always raise, so ``run_batch`` falls back to
    ``_run_tests_batch`` → ``_run_tests_batch_fallback`` (which calls the
    monkeypatched ``_run_tests`` per job).
    """
    import evaluation.harness

    evaluation.harness._run_tests_batch_modal = _raise_modal_disabled  # type: ignore[assignment]
    evaluation.harness._run_tests_swebench = _raise_modal_disabled  # type: ignore[assignment]


def _raise_modal_disabled(*_args: object, **_kwargs: object) -> None:  # type: ignore[empty-body]
    """Placeholder that always raises — Modal batch disabled in tests."""
    raise RuntimeError("Modal batch disabled in tests")
