---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/runtime_evidence.py
  - ../../../src/marulho/evaluation/service_benchmark.py
related_benchmarks:
  - ../../../reports/bounded_replay_window_20260622/service-benchmark-replay-dataset-history-retired.json
  - ../../../reports/bounded_replay_window_20260622/replay-dataset-history-retired-source-window.json
  - ../../../reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-dataset-history-anchor-retired.json
---

# Replay Dataset History Wrapper

`GET /terminus/replay-dataset/history` is retired and removed. It returned
the same replay-sample records already owned by
`GET /terminus/replay-sample/history`, so it was a second public history path
without new dataset selection criteria.

Superseded replacement:

- The whole service advisory replay lane is now retired:
  `/terminus/replay-sample/history`, `/terminus/replay-plan`,
  `/terminus/replay-dataset/preview`, and `/terminus/replay-dataset/bundle`
  are deleted too.
- Replay/consolidation history that matters to cognition is checkpointed inside
  trainer/SNN slow-window reports and ReplayController artifacts, permits, and
  tickets.
- `/terminus/runtime-traces/export` remains trace-only.

The service benchmark no longer calls the wrapper and no longer writes
`replay_dataset_history_summary`. Current service benchmarks also omit the
former replay-plan, replay-sample history, dataset preview, and dataset bundle
surfaces.

Evidence:

- `service-benchmark-replay-dataset-history-retired.json`: endpoint absent,
  slow-path endpoint count `5`, hot-path budget passing.
- `replay-dataset-history-retired-source-window.json`: dataset preview still
  matches `50/50` selected target/link parity with CPU source windows.
- `hotpath-active-pressure-65536-524288-i32-replay-dataset-history-anchor-retired.json`:
  `6151.826 tokens/sec`, bounded `12/65536` route rows, and zero graph/native
  sequence failures.

Revisit only if a dataset-specific history contract adds unique bounded
selection evidence over trainer/SNN replay state.
