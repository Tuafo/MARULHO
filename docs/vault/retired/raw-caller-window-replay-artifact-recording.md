---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/evaluation/snn_replay_artifact_provenance_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json
---

# Raw Caller Window Replay Artifact Recording

## Status

Retired and removed from production. Replay artifacts are recorded only through
`record_evaluated_snn_transition_memory_replay_artifact(...)`.

## Why

The old raw caller-window recorder could persist replay artifacts built from a
caller-supplied replay window. Those artifacts were not permit-eligible, but the
recorder and load path still preserved a second production-shaped artifact
route beside the evaluated internal-ledger path. At LLM-size scale that would
let replay/consolidation evidence drift away from explicit source-window
budgets and server-held selection evidence.

## Replacement

The evaluated recorder requires a verified internal-ledger artifact proposal,
Replay Controller context, review ticket, known-readout source window,
replay-priority source window, and provenance source window. Controller load
drops raw caller-window artifacts, and permit verification recomputes the
persisted source-window hashes before accepting an artifact.

## Evidence

`reports/bounded_replay_window_20260620/snn-replay-artifact-raw-recorder-retired.json`
passed with `public_raw_recorder_callable=false`,
`raw_loaded_artifact_count=0`, `raw_artifact_index_hit=false`, `4` bounded
provenance lookups instead of `256` retained-record checks, mean verification
latency `0.538460 ms`, p95 `1.659900 ms`, traced Python peak `0.017773 MiB`,
and `0.0 MiB` CUDA allocation/reservation.

`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-raw-replay-artifact-recorder-retired.json`
processed `524288` tokens at `6004.719 tokens/sec`, with
`train_compute=0.135314 ms/token`, `prepare_training=0.007010 ms/token`,
`finalize_total=0.006575 ms/token`, `tick_duration_ms.p95=21.825`, bounded
`12/65536` route rows, `65526` cached transition rows, zero graph/native
sequence failures, and RTX 3060 memory `1863->1865 MiB`. Velocity observed GPU
contention, so this is same-band throughput protection rather than a clean
speed ceiling.

## Revisit Only If

Reintroduce caller-window artifact recording only through a new ADR and
benchmark-local diagnostic isolation. Production replay artifacts must remain
evaluated, internal-ledger-backed, source-window-hashed, CPU-archival, non-live,
non-every-token, no raw text, no hidden language reasoning, and protected by
repeated 6k-ish long-run evidence.
