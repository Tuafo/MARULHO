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
  - reports/bounded_replay_window_20260620/sleep-replay-routing-index-refresh.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-sleep-replay-routing-index-refresh.json
---

# Sleep Replay Routing-Index Full Rebuild

Full routing-index rebuild after selected deep/repair sleep replay is retired as normal replay maintenance. Selected replay already chose a bounded evidence window and updated a small set of prototype IDs; rebuilding all routing rows after that preserved an archive-scale maintenance path beside bounded replay.

The maintained path is `routing_index_existing_row_refresh.v1`. `MarulhoTrainer._refresh_sleep_replay_routing_index(...)` passes only replay-updated prototype IDs to the routing index. `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` update existing tensor-cache rows through CPU ID-to-row maps, report direct update count, row lookup mode, missing-ID count, cache dirty state, and full-rebuild fallback status, and leave active routing tensors on the configured index device.

Full rebuild remains allowed only for explicit missing-ID fallback, dirty-cache fallback, checkpoint restore, bootstrap, or benchmark-local diagnostics. It must not become the normal selected sleep-replay path, enter the live tick, scan all memory for replay selection, move archival metadata to CUDA, or reason through replay text.

The 65536-row benchmark `reports/bounded_replay_window_20260620/sleep-replay-routing-index-refresh.json` passed with exact top-1 recall for `16` updated rows, no bounded-path rebuild, `row_lookup_mode=host_id_row_map`, and mean latency `5.006260 ms` versus `133.747880 ms` for the retired `add()+rebuild()` baseline.

The paired `524288`-token hot-path run stayed in band at `6022.776 tokens/sec`, `tick_duration_ms.p95=21.373`, `train_compute=0.134715 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, flat RTX 3060 memory at `1967 MiB`, and zero graph/native sequence failures. Velocity reported GPU contention from a `25%` sample, so this is same-band protection evidence rather than a clean speed ceiling.
