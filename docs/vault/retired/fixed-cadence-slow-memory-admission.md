---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/strong_capture_admission_cadence_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/slow-memory-fixed-cadence-admission-retired.json
  - reports/bounded_replay_window_20260620/strong-capture-admission-cadence-after-fixed-cadence-retirement.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-slow-memory-fixed-cadence-retired-rerun.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/slow-memory-fixed-cadence-projection-removed.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/strong-capture-admission-projection-removed.json
  - ../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-slow-memory-admission-projections-removed-default-nosample.json
---

# Fixed-Cadence Slow-Memory Admission

Fixed-cadence slow-memory admission is retired as an executable archive-write
path. Retained `train_step` no longer archives ordinary
`slow_memory_archive_interval_tokens` hits. It records `cadence_deferred`
through the cognitive boundary controller instead.

The maintained live archive writes are:

- first-token retained/fallback admission
- selected strong-capture admission bounded by
  `slow_memory_archive_strong_capture_min_interval_tokens`

The retirement benchmark kept `1` first-token archive and removed `17`
projected fixed-cadence writes over `256` tokens. The refreshed strong-capture
benchmark still kept `17` selected strong archives versus `256` retired
every-strong writes, so useful STC-like selection remains without fixed cadence.

On 2026-06-24, the benchmark-side projection objects were removed too. The
maintained-only reports
`..\..\MARULHO_reports\bounded_replay_window_20260624\slow-memory-fixed-cadence-projection-removed.json`
and
`..\..\MARULHO_reports\bounded_replay_window_20260624\strong-capture-admission-projection-removed.json`
record `retired_fixed_cadence_admission_absence.implementation_present=false`
and `retired_every_strong_admission_absence.implementation_present=false`.
Fixed cadence keeps `1` first-token archive, removes `16` projected cadence
writes, defers `16` cadence hits, averages `1326.868180 ms`, and uses
`0.0 MiB` CUDA allocation/reservation. Strong capture archives `17` bounded CPU
records with `16` selected strong archives, skips `239` refractory writes,
projects `239` removed every-strong writes, averages `1335.328410 ms`, and
uses `0.0 MiB` CUDA allocation/reservation.

On 2026-06-30, the executable fixed-cadence retirement benchmark was deleted
from `src/marulho/evaluation`; the JSON reports above are historical evidence,
not runnable legacy code.

The accepted `524288`-token protection rerun stayed in the maintained band at
`6043.321 tokens/sec`, with bounded `12/65536` route rows, `65526` cached
transition rows, `2048` deferred cadence hits, zero graph/native sequence
failures, and flat RTX 3060 memory at `1958 MiB`. The first same-code run is
retained only as below-band variance evidence at `5758.051 tokens/sec`.

The current protection run after projection removal
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-slow-memory-admission-projections-removed-default-nosample.json`
processed `524288` tokens at `5957.637 tokens/sec`, with p95 `21.679 ms`,
`train_compute=0.135551 ms/token`, `prepare_training=0.006811 ms/token`,
`finalize_total=0.006772 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, `2048` deferred cadence hits, native sequence-loop and
burst-replay failure counts `0`, no observed before/after contention (`cpu
max=22%`, `gpu max=12%`), and RTX 3060 memory `2047->2046 MiB`.

Reopen this path only if a selected replay/admission design improves measured
recall, grounding, or reconstruction and repeated long-run evidence proves the
live tick remains protected without scans, every-token admission, GPU-resident
archival metadata, hidden language reasoning, or repo-local executable retired
projection objects.
