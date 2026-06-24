---
type: retired-path
status: retired and removed from benchmark code
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/evaluation/selected_replay_consolidation_cache_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/selected-replay-consolidation-cache.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-selected-replay-consolidation-cache-rerun.json
  - ../../../../../MARULHO_reports/bounded_replay_window_20260624/selected-replay-consolidation-cache-diagnostic-removed.json
  - ../../../../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-selected-replay-cache-diagnostic-removed-default-nosample.json
---

# Selected Replay Bucket-Consolidation Cache Rebuild

Full bucket-consolidation cache rebuild during selected replay is retired.
`DualMemoryStore.consolidate_replay(...)` must not recover a missing cache by
scanning every retained slow-memory entry before mutating a bounded replay
window.

The maintained path is `bounded_selected_replay_consolidation.v1`. Selected
replay mutates selected entries, updates replay counts, consolidation levels,
capture tags, and EMAs, and reports the selected indices and buckets. If the
bucket cache exists, it applies selected-bucket delta updates. If the cache is
missing, it records `cache_missing_deferred_no_full_rebuild`, performs no
cache-rebuild scan, and leaves exact cache reconstruction to checkpoint/load,
graph capture/prewarm, offline quality, or explicit tensor request.

The historical 65536-entry benchmark
`reports/bounded_replay_window_20260620/selected-replay-consolidation-cache.json`
matched selected-entry consolidation state, replay counts, capture tags,
consolidation events, and fast EMA against a benchmark-local retired diagnostic
that rebuilt the full cache first. The bounded path scanned `0` cache-rebuild
entries, averaged `2.291943 ms`, used `0.510799 MiB` traced Python peak, and
used no CUDA allocation. The retired diagnostic averaged `2979.156029 ms` for
the same selected window, so source work fell by `4096x`.

The current maintained-only report
`..\..\MARULHO_reports\bounded_replay_window_20260624\selected-replay-consolidation-cache-diagnostic-removed.json`
removes that executable diagnostic. It validates seeded selected replay counts,
consolidation-event increments, consolidation-level increase, capture-tag
decay, and fast-EMA centroid directly, reads `16/65536` selected CPU entries,
scans `0` cache-rebuild entries, projects `65536` removed full-cache rebuild
entries (`4096x` avoidance), averages `1.957360 ms`, uses `0.510799 MiB`
traced Python peak, and uses no CUDA allocation/reservation.

The current `524288`-token hot-path rerun
`..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-selected-replay-cache-diagnostic-removed-default-nosample.json`
stayed in the maintained band at `6209.501 tokens/sec`, p95 tick
`20.916 ms`, `train_compute=0.129968 ms/token`, bounded `12/65536` route rows,
`65526` cached transition rows, CPU max `22%`, GPU max `21%`, RTX 3060 memory
`2026->2026 MiB`, and zero native sequence-loop or burst-replay failures. The
before-sample GPU reading of `21%` makes this throughput protection under
borderline contention rather than a clean speed ceiling.

Reopen only if selected replay can prove that exact cache repair during replay
improves a measured prediction, grounding, or reconstruction target enough to
justify archive-wide selected-window work, and a repeated long-run check keeps
the live tick in the maintained 6k-ish band. Do not restore a repo-local
executable diagnostic baseline.
