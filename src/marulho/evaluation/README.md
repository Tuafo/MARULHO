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
  retained 2026-06-30 local evidence passed with brain tick sleep calls `0`,
  sequence sleep calls `0`, explicit slow-path sleep calls `1`, and
  `runtime_owner=MarulhoBrain`.
- The retained 2026-06-30 long sequence-input staging gate reached `6601.19`
  sequence tokens/sec versus `6507.41` per-quantum tokens/sec. It used
  `cuda_graph_route_transition_burst`, backend `cuda_graph_conditional_while`,
  device `cuda:0`, and zero graph/native/burst failures.
- Continuous stress reports at `256`, `1024`, and `4096` tokens are smoke/debug
  history only. They passed through the same conditional-WHILE CUDA backend
  with zero graph/native/burst failures, but they are not promotion evidence.
- The sustained runtime evidence ladder is `8192` tokens for diagnostic
  evidence, `131072` tokens for the normal long-run promotion gate, and
  `524288` tokens for the preferred house-scale target when hardware/runtime
  budget allows. Promotion is not allowed from `256`, `1024`, or `4096`.
- `continuous_runtime_stress_benchmark.py` must write a final or partial JSON
  report for success, timeout, exception, interrupt, and manual stop. A report
  must include target tokens, token delta, elapsed time, tokens/sec when
  measurable, checkpoint, runtime owner, tick/quantum tokens, final/last
  `BrainTrace`, device report, CUDA/backend/executor evidence, graph/native/
  burst/sequence failure and fallback counters, event summary, and environment
  contention summary.
- Current 2026-07-03 evidence: `reports/runtime_evidence_20260703/diagnostic-8192.json`
  reached `8192/8192` tokens at `131.614 tokens/sec`; the normal long gate
  `reports/runtime_evidence_20260703/long-gate-131072.json` reached
  `131072/131072` tokens at `43.666 tokens/sec`, CUDA RTX 3060,
  `conditional_while`, zero CUDA graph/native/sequence failures, bounded
  `12/65536` route rows, no all-column state transition, `brain_feed_streaming_refill`
  with `16` feed calls and zero source drops, and contention `not_observed`.
  The house-scale gate `reports/runtime_evidence_20260703/house-scale-524288.json`
  reached `524288/524288` tokens at `44.834 tokens/sec`, with the same CUDA
  backend, zero CUDA graph/native/sequence failures, bounded `12/65536` route
  rows, no all-column state transition, `brain_feed_streaming_refill` with
  `64` feed calls and zero source drops, and contention `not_observed`.
  These reports close the normal long-run and house-scale artifact boundaries
  but are not speed promotions over the historical 6k-ish hot-path band because
  complete elapsed time includes bounded source refill/encoding.
- Preserved failure evidence: `reports/runtime_evidence_20260703/long-gate-131072-source-exhausted-before-refill.json`
  shows the old one-shot feed path exhausted the bounded `8192`-token source
  buffer after `8192` tokens. Keep long targets on bounded streaming refill.

## Sustained Runtime Commands

```bash
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/diagnostic-8192.json --target-tokens 8192 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 600
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/long-gate-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 7200
python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint checkpoints/marulho/model.pt --output reports/runtime_evidence_20260703/house-scale-524288.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 21600
```
