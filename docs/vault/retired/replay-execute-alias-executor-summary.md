---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/living_status.py
  - ../../../src/marulho/service/runtime_evidence.py
related_docs:
  - ../../retired-paths.md
  - ../../../CONTEXT.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/replay-sample-single-path-service-benchmark.json
  - reports/bounded_replay_window_20260620/replay-dataset-source-window-replay-sample-single-path.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-sample-single-path.json
---

# Replay Execute Alias And Executor Summary

`/terminus/replay-execute`, `/terminus/replay-execute/history`,
`mode="execute"`, `execution_id`, and `replay_executor_summary` are retired.
They made the audit-only replay sampler look like a second execution path even
though it performed no replay computation, training, sleep, memory promotion,
feedback posting, action execution, or external calls.

The maintained path is now only:

- `POST /terminus/replay-sample`
- `GET /terminus/replay-sample/history`
- `replay_sample_summary`

Old execute-shaped records normalize to `sample`; production summaries expose
only `dry_run` and `sample` modes. Runtime/export/benchmark evidence carries
one bounded summary surface:
`bounded_replay_sample_summary_source_window.v1`.
Offline comparison tooling also ignores the old `replay_executor_summary` key
and reads safety flags from replay dataset, bundle, Runtime Truth, or
`replay_sample_summary` evidence only. The separate replay-adapter experiment
stack is retired and removed, so it is no longer related code for this alias.

## Evidence

`reports/bounded_replay_window_20260620/replay-sample-single-path-service-benchmark.json`
passed with no `replay_executor_summary`, no executor key in trace-export
responses, replay-sample history latency `4.798 ms`, CPU summary placement, no
raw replay text, no hidden language reasoning, no live tick, and no every-token
work.

`reports/bounded_replay_window_20260620/replay-dataset-source-window-replay-sample-single-path.json`
passed with canonical `sample` history records, `50/50` target/link parity,
`64/256` bounded replay-sample summary records, `64/256` replay-sample link
records, CPU archival/source placement, no GPU-resident archival metadata, no
training/plasticity, and no live/every-token work.

`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-sample-single-path.json`
processed `524288` tokens at `5951.781 tokens/sec`, p95 `21.962 ms`,
`train_compute=0.136320 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, zero graph/native sequence failures, no observed
contention, CPU max `38%`, GPU max `14%`, and RTX 3060 memory `1894->1881 MiB`.

## Revisit Condition

Reintroduce an execution-named replay surface only through a new ADR and a real
bounded replay/consolidation executor with explicit selection criteria, quality
metric, memory budget, device placement, latency cost, safety gates, checkpoint
evidence, and repeated long-run proof that the live tick stays in the maintained
6k-ish throughput band.
