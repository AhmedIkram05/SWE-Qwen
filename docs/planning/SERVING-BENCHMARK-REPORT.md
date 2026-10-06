
# Endpoint benchmark

- Date: 2026-10-06
- Requests: 250, errors: 0.0%
- Time-To-First-Byte p50: 14434.9 ms, p95: 25334.4 ms
- Latency p50: 14434.9 ms, p95: 25334.4 ms
- Tokens/s: 168.4, throughput: 0.7 req/s
- Cost per inference: $0.00042
- Note: the CLI aggregate above spans the whole ramp; `_measure_endpoint` keeps no per-level breakdown. The sections below come from the concurrency proof scripts (same endpoint, warm container).

## Per-level concurrency (2026-10-06)

- Method: ThreadPoolExecutor per level (1/8/16) × 10 non-stream requests, 256 tokens, model qwen3-14b, 80GB-GPU serving tier, warm container (45s keep-alive).
- Scaling (tok/s): level 1 → 50.0; level 8 → 227.2 (first run) / 168.6 (rerun); level 16 → 153.0 / 152.2. The pre-fix offline-`LLM` code was flat at ~50 tok/s at ANY concurrency (the throttling); the async-engine fix restores scaling to 8-way.
- Errors: 0/10 (lvl 1), 0/80 (lvl 8); 7/160 (4.4%) at level 16 in one rerun — client 120s SDK timeouts on the first wave, unrooted, needs investigation.
- Latency p50 (non-stream, warm): ~4.9s (lvl 1), ~8.4–11.3s (lvl 8), ~16.3–19.2s (lvl 16).
- Knee at 16: aggregate tok/s plateaus between 8 and 16 — decode/scheduler-bound; KV cache (373,744 tokens) is not the constraint.

## Response integrity (2026-10-06)

- 8-way concurrent requests, each with a unique marker word: 0/8 responses contained a foreign marker (no cross-request response bleed); 7/8 fully contained their own marker, 1 truncated at max_tokens mid-marker (model truncation, not bleed).

## Cold start (2026-10-06)

- Measured ~140–150s cold (scale-to-zero is deliberate; no `keep_warm`).
- Clients with timeout < ~150s fail on a cold endpoint; a client that gives up mid cold-start causes Modal to SIGTERM the booting container (observed `Server has lost track of input` 500 at ~73s). Mitigations: `keep_warm(1)` or client timeouts ≥ 200s.

## S3 gate

- TTFB p50 < 500ms FAILS on this non-stream benchmark by design — non-stream TTFB == full latency. Real first-token streaming TTFB (now supported by the async engine) is not yet measured.
