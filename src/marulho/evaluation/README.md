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

## Ported Guidance

- Benchmark complete runtime behavior when making throughput claims. Separate
  setup/compile overhead from steady-state token throughput.
- `service_benchmark.py` measures the thin brain adapter only; it must not
  revive `/terminus`, `/status`, root feed/query/respond, living-loop, or policy
  actuator endpoints as performance gates.
- Hot-window reports may profile trainer stages, but profiling is evidence and
  must not become ordinary runtime work.
- CUDA claims need observed backend/device evidence, fallback counts, and
  failure counts.
- Regressions should preserve exact failing repros and rejected reports rather
  than only reporting the winning run.
- Legacy service/status source-window runners are retired. New evaluation code
  must not import `MarulhoServiceManager`, `RuntimeFacade`, or
  `StatusReadModel` to prove the active brain spine.

## Current 2026-06-30 Spine Evidence

- `service_benchmark.py`, `continuous_runtime_quantum_benchmark.py`, and
  `continuous_runtime_stress_benchmark.py` now target `MarulhoBrain` or
  `/brain/*`.
- `source_tick_sleep_deferral_benchmark.py` now builds `MarulhoBrain`
  directly. The current brain-spine report at
  `reports/brain_spine_20260630/source-tick-sleep-deferral-brain-spine.json`
  passed with brain tick sleep calls `0`, sequence sleep calls `0`, explicit
  slow-path sleep calls `1`, and `runtime_owner=MarulhoBrain`.
- Tiny stress smoke completed after bounding source prefill:
  `reports/brain_spine_20260630/tiny-stress-smoke/stress.json`.
- Promoted-checkpoint brain-loop stress smoke completed through
  `cuda_graph_conditional_while` with zero graph/native failures:
  `reports/brain_spine_20260630/continuous-runtime-stress-brain-loop-smoke.json`.
- Short sequence-input staging smokes at
  `reports/brain_spine_20260630/sequence-input-staging-brain-spine-thin-service-rerun*.json`
  failed the speedup comparison while keeping the conditional-WHILE backend and
  zero failures. The longer default gate at
  `reports/brain_spine_20260630/sequence-input-staging-brain-spine-long-default.json`
  passed at `1.154x` speedup with zero graph/native failures. Use the longer
  gate as current comparable CUDA evidence; keep the short failures as variance
  evidence, not completion proof.
- Deleted legacy status/facade benchmark scripts are treated as retired
  historical evidence. Do not recreate them as active source unless a new
  package-local benchmark targets `MarulhoBrain` or the actual owner module.
