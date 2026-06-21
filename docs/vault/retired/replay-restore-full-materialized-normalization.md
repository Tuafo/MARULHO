---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/manager.py
  - ../../../src/marulho/service/persistence.py
  - ../../../src/marulho/evaluation/replay_restore_source_window_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/replay-restore-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json
---

# Replay Restore Full Materialized Normalization

Full-materialized replay restore normalization is retired. Checkpoint/reload
must not copy or normalize every persisted replay-controller history before
applying the active retention limits.

The maintained path is `bounded_replay_restore_source_window.v1`.
`ReplayController` reads only the newest source window for replay sample
history, regeneration permits, replay-evaluation contexts, review tickets,
scheduler installations, and transition-memory replay artifacts before
normalization, index rebuild, or evaluated-artifact validation. Service manager
construction and persistence restore pass those checkpoint fields directly into
the controller instead of adding extra full `list(...)` copies.

The 65536-record-per-field benchmark
`reports/bounded_replay_window_20260620/replay-restore-source-window.json`
matched the retired full-materialized diagnostic for the retained latest
window, restored `64` valid evaluated artifacts, inspected `656` records
instead of `524288`, and reduced mean latency from `6605.339529 ms` to
`15.600729 ms` (`423.399426x`). Archival/source metadata stayed on CPU, CUDA
allocation/reservation stayed `0.0 MiB`, Python traced peak was `0.581783 MiB`,
and the restore report states no live tick, no every-token work, no raw replay
text, no hidden language reasoning, no mutation/plasticity, and no
GPU-resident archival metadata.

The accepted `524288`-token protection rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-restore-source-window-rerun.json`
stayed in band at `5945.577 tokens/sec`, p95 tick `22.062 ms`,
`train_compute=0.136201 ms/token`, bounded route scoring at `12/65536`, cached
`65526` transition rows, no observed contention, CPU max `30%`, GPU max `13%`,
RTX memory `2061->2062 MiB`, and zero graph/native sequence failures.

Reopen only as benchmark-local diagnostics or if a new checkpoint format
provides an indexed restore contract that proves stronger replay quality
without archive-wide restore-time materialization and without weakening live
tick throughput.
