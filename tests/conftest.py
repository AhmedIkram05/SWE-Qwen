"""Root conftest: early setup for all test suites."""

from __future__ import annotations

import os

# ponytail: global lock — hf_xet Rust library panics when another trace
# subscriber is already set. HuggingFace recommend HF_XET_DISABLE=1 for
# environments like test suites where tracing is not needed.
# Upgrade when hf_xet supports setting a subscriber without panic.
os.environ.setdefault("HF_XET_DISABLE", "1")
