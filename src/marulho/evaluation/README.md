# Evaluation

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`evaluation` owns promotion gates, benchmarks, readiness checks, and validation
harnesses.

## Owns

- Evidence standards for speed, readiness, CUDA placement, liveness, and
  promotion.
- Same-checkpoint comparison runners and sustained stress checks.
- Slow-path environment snapshots that separate architecture changes from
  CPU/GPU contention.
- The maintained service endpoint benchmark over `/health` and `/brain/*`.

## Must Not Own

- Runtime mutation.
- Production status verdicts not backed by current reports.

## Runtime Rules

- Benchmark complete runtime behavior when making throughput claims. Separate
  setup/compile overhead from steady-state token throughput.
- `service_benchmark.py` measures the thin brain adapter only; it must not
  turn non-`/brain/*` HTTP surfaces into performance gates.
- Hot-window reports may profile trainer stages, but profiling is evidence and
  must not become ordinary runtime work.
- CUDA claims need observed backend/device evidence, fallback counts, and
  failure counts.
- Regressions should preserve exact failing repros and rejected reports rather
  than only reporting the winning run.

## Current Validation Snapshot

- `service_benchmark.py`, `continuous_runtime_quantum_benchmark.py`, and
  `continuous_runtime_stress_benchmark.py` now target `MarulhoBrain` or
  `/brain/*`.
- `source_tick_sleep_deferral_benchmark.py` builds `MarulhoBrain` directly. The
  report at
  `reports/brain_spine_20260630/source-tick-sleep-deferral-brain-spine.json`
  passed with brain tick sleep calls `0`, sequence sleep calls `0`, explicit
  slow-path sleep calls `1`, and `runtime_owner=MarulhoBrain`.
- The long sequence-input staging gate at
  `reports/brain_spine_20260630/sequence-input-staging-post-cleanup-long-default.json`
  reached `6601.19` sequence tokens/sec versus `6507.41` per-quantum
  tokens/sec. It used `cuda_graph_route_transition_burst`, backend
  `cuda_graph_conditional_while`, device `cuda:0`, and zero graph/native/burst
  failures.
- Continuous stress reports at `256`, `1024`, and `4096` tokens passed through
  the same conditional-WHILE CUDA backend with zero graph/native/burst
  failures. The `4096` token report reached `121.93` tokens/sec over `32`
  ticks.
- The `8192` and `131072` token continuous stress attempts did not produce a
  final JSON report before manual stop. Keep that boundary open until the
  runner is diagnosed.
