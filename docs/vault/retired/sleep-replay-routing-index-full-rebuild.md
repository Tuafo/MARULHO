---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/retrieval/routing_index.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/sleep_replay_routing_index_refresh_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery.json
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery-sharded.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json
---

# Sleep Replay Routing-Index Full Rebuild

Full routing-index rebuild after selected deep/repair sleep replay is retired as normal replay maintenance. Selected replay already chose a bounded evidence window and updated a small set of prototype IDs; rebuilding all routing rows after that preserved an archive-scale maintenance path beside bounded replay.

The maintained path is `routing_index_existing_row_refresh.v1`. `MarulhoTrainer._refresh_sleep_replay_routing_index(...)` passes only replay-updated prototype IDs to the routing index. `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` update existing tensor-cache rows through CPU ID-to-row maps, report direct update count, row lookup mode, missing-ID count, cache dirty state, skipped-update count, and deferred recovery status, and leave active routing tensors on the configured index device.

Missing IDs, dirty caches, or missing row-update APIs now defer recovery and do not call `add()+rebuild()` inside selected replay. Full rebuild remains allowed only for checkpoint restore, bootstrap, or explicit offline repair. It must not become the normal selected sleep-replay path, selected-replay recovery path, benchmark-local diagnostic baseline, enter the live tick, scan all memory for replay selection, move archival metadata to CUDA, or reason through replay text.

The refreshed 65536-row benchmark `reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery.json` passed with exact top-1 recall for `16` updated rows, `1` missing row deferred and kept absent, no bounded-path rebuild, `row_lookup_mode=host_id_row_map`, and mean latency `4.171690 ms` versus `118.414640 ms` for the benchmark-local retired `add()+rebuild()` baseline. The sharded variant `reports/bounded_replay_window_20260620/sleep-replay-routing-index-deferred-recovery-sharded.json` passed with `16` direct shard and merged updates, `1` missing row deferred, no bounded-path rebuild, and mean latency `13.348040 ms` versus `140.566380 ms` for the sharded retired rebuild baseline.

The current maintained-only benchmark `..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-replay-routing-index-legacy-baseline-removed-rerun.json` removes that executable retired baseline and records `retired_full_rebuild_absence.implementation_present=false`, `diagnostic_callable=false`, and `active_report_field_present=false`. It passed with exact top-1 recall for `16` updated rows, `1` missing row deferred and kept absent, no bounded-path rebuild, `row_lookup_mode=host_id_row_map`, CPU archival and row metadata, CUDA routing tensors, and `9.892320 ms` mean bounded latency across `25` runs (`median=5.693200 ms`). The sharded maintained-only report `..\..\MARULHO_reports\bounded_replay_window_20260623\sleep-replay-routing-index-legacy-baseline-removed-sharded.json` passed with `16` direct shard and merged updates, `1` missing row deferred, no rebuild, CPU row metadata, CUDA routing tensors, and `19.582580 ms` mean latency. Python trace-memory peak stayed below `0.085 MiB`; CUDA allocation/reservation ended at `24.625/34.0 MiB` for single-index and `41.125/50.0 MiB` for sharded.

The current `524288`-token hot-path run `..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-routing-index-legacy-baseline-removed-default-nosample.json` stayed in band at `6097.811 tokens/sec`, `tick_duration_ms.p95=21.193`, `train_compute=0.132696 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, CPU max `32%`, GPU max `15%`, RTX 3060 memory `1825->1824 MiB`, and zero graph/native sequence failures. The earlier accepted `reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-routing-index-deferred-recovery-rerun.json` remains historical promotion evidence at `5943.512 tokens/sec`; an earlier same-slice run at `5688.783 tokens/sec` is rejected as primary protection evidence because velocity reported CPU contention at `99%`.
