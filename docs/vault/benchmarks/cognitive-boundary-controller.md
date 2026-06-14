---
type: benchmark
status: current
related_code:
  - src/marulho/training/cognitive_boundary_controller.py
  - src/marulho/training/trainer.py
related_docs:
  - CONTEXT.md
  - docs/adr/0006-persistent-text-tick-executor.md
  - docs/research-living-brain.md
related_papers: []
related_benchmarks:
  - reports/cognitive_boundary_controller_20260614/stress-131072.json
---

# Cognitive Boundary Controller

The controller separates persistent CUDA cognition from observation and CPU
maintenance. Telemetry no longer interrupts execution. Drift refresh and floor
closure run after exact bounded event drains. Sleep/replay and structural
mutation boundaries remain fail-closed.

## Sustained Evidence

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/cognitive_boundary_controller_20260614/stress-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 8 --source-concept-observation-tick-interval 4 --timeout-seconds 600 --sample-interval-seconds 0.02`
- Throughput: `2126.013 tokens/sec` over `61.652 s`.
- CUDA: RTX 3060, `131072` graph transitions, zero graph/burst failures.
- Burst ownership: `116696` tokens.
- Maintenance: `2333` drift refreshes, `12` floor closures, `1790` deferred telemetry observations.
- Real fallbacks: two `sleep_boundary` events.

A `2027.181 tokens/sec` 32768-token sample is not promotion evidence because
measurement started while source prewarm was still running.

## Next Constraint

The long run measured `train_compute=0.281598 ms/token` and
`prepare_training=0.156315 ms/token`. Preparation/orchestration is now the next
large optimization boundary. This constraint was addressed by
[[prepared-source-tick-executor]].
