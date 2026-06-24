---
type: retired-path
status: retired and removed from benchmark code
related_code:
  - ../../../src/marulho/evaluation/sleep_repair_replay_bounded_benchmark.py
  - ../../../src/marulho/training/trainer.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../concepts/replay-window.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../../MARULHO_reports/bounded_replay_window_20260624/sleep-repair-replay-dense-prepare-comparator-removed.json
  - ../../../../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-sleep-repair-dense-prepare-comparator-removed-default-nosample.json
---

# Sleep Repair Dense Input Prepare Comparator

The production repair replay path already retired unconditional dense
`assembly_from_input(...)` preparation after a selected repair window has stored
routing keys. Keeping the dense prepare call as a repo-local benchmark
comparator preserved old implementation code beside the maintained replay path.

The maintained benchmark now runs only `prepare_input_for_candidate_routing(...)`
for selected entries and counts real repair replay dense-assembly calls by
instrumenting the runtime path. It does not execute the old dense prepare
comparator. Missing routing-key rows remain deferred; they are not repaired by
dense input reconstruction or stored-assembly projection.

Current evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260624\sleep-repair-replay-dense-prepare-comparator-removed.json`
  passes with `retired_dense_prepare_comparator_absence.implementation_present=false`.
- The report selects `32` anchored repair entries, has `16` stored routing keys
  and `16` missing keys before repair, applies `8` repair updates, defers `8`
  missing-key selected rows in the repair window, improves stored-key quality by
  `0.076463`, and keeps bounded prepare mean at `44.895575 ms` under a
  `100 ms` budget.
- Runtime Truth records `0` dense input-assembly calls, `0` dense fallback
  calls, no global candidate/score scan, no raw replay text, no hidden language
  reasoning, no live tick, no every-token cadence, CPU archival metadata, and
  CUDA active repair compute.
- The current `524288`-token hot-path run
  `..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-sleep-repair-dense-prepare-comparator-removed-default-nosample.json`
  stayed in band at `6410.861 tokens/sec`, p95 `20.195 ms`,
  `train_compute=0.126774 ms/token`, bounded `12/65536` route rows, no observed
  contention, CPU max `10%`, GPU max `15%`, and RTX memory `2190->2190 MiB`.

Reopen only through a new bounded repair policy with prediction, grounding, or
reconstruction evidence plus explicit source budget and repeated long-run
live-tick protection. Do not restore a repo-local executable dense-prepare
comparator.
