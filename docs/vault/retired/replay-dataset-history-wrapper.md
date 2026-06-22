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

Replacement:

- replay history: `/terminus/replay-sample/history`
- replay candidates: `/terminus/replay-plan`
- dataset preview/package: `/terminus/replay-dataset/preview` and
  `/terminus/replay-dataset/bundle`

The service benchmark no longer calls the wrapper and no longer writes
`replay_dataset_history_summary`. The current report keeps slow-path endpoints
to replay plan, replay-sample history, trace export, dataset preview, and
dataset bundle only.

Evidence:

- `service-benchmark-replay-dataset-history-retired.json`: endpoint absent,
  slow-path endpoint count `5`, hot-path budget passing.
- `replay-dataset-history-retired-source-window.json`: dataset preview still
  matches `50/50` selected target/link parity with CPU source windows.
- `hotpath-active-pressure-65536-524288-i32-replay-dataset-history-anchor-retired.json`:
  `6151.826 tokens/sec`, bounded `12/65536` route rows, and zero graph/native
  sequence failures.

Revisit only if a dataset-specific history contract adds unique bounded
selection evidence over replay-sample history.
