---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/column_transition_runtime.py
  - ../../../src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py
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
  - reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/selected-replay-consolidation-cache.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-selected-replay-consolidation-cache-rerun.json
---

# Bucket Consolidation Level Full Scan

Scalar `DualMemoryStore.bucket_consolidation_level(...)` full-memory scans are retired from live metric lookup. A single winner/bucket consolidation metric must not iterate every retained slow-memory entry in retained or graph metric ticks.

The maintained path is `bucket_consolidation_level_cache_lookup.v1`. `DualMemoryStore` maintains CPU bucket consolidation metadata as entries are stored or selected replay updates consolidation. Scalar reads use that cache, report `full_memory_scan=false`, `scan_entry_count=0`, cache generation, cache hit/miss status, and return a no-scan miss if the cache is absent. Explicit `bucket_consolidation_tensor(...)` rebuilds remain checkpoint load, graph capture/prewarm, offline quality, or explicit tensor request. The benchmark-local full-scan comparator is now removed too.

Selected replay is not a cache-recovery context anymore. `bounded_selected_replay_consolidation.v1` mutates only selected replay entries; if cache metadata exists, it applies selected-bucket delta updates, and if cache metadata is missing, it reports `cache_missing_deferred_no_full_rebuild` rather than rebuilding over the whole archive. `reports/bounded_replay_window_20260620/selected-replay-consolidation-cache.json` matched selected-entry state against the benchmark-local retired full-cache rebuild diagnostic with `0` cache-rebuild scan entries, `2.291943 ms` bounded mean latency versus `2979.156029 ms`, and `4096x` less source work on `65536` retained entries.

The selected-replay cache-recovery rerun stayed same-band at `5973.047 tokens/sec` with no observed contention, bounded `12/65536` route rows, and zero graph/native sequence failures.

The historical 65536-entry benchmark `reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json` matched the retired scalar scan within `1e-6`, reported `cache_hit`, scanned `0` entries, and reduced mean lookup latency from `12.999192 ms` to `0.016260 ms`.

The current maintained-only benchmark `..\..\MARULHO_reports\bounded_replay_window_20260623\bucket-consolidation-cache-legacy-baseline-removed.json` records `retired_full_bucket_scan_absence.implementation_present=false`, `diagnostic_callable=false`, and `active_report_field_present=false`. It passed with exact seeded bucket-value parity (`value_delta=0.0`), `cache_hit`, `scan_entry_count=0`, `0.017516 ms` mean lookup latency over `25` runs, CPU archival/cache metadata, `0.002090 MiB` Python trace-memory peak, and `0.0 MiB` CUDA allocation.

The current `524288`-token hot-path run `..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-bucket-cache-legacy-baseline-removed-default-nosample.json` stayed in band at `6461.135 tokens/sec`, `tick_duration_ms.p95=19.207`, `train_compute=0.125097 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, CPU max `9%`, GPU max `18%`, RTX 3060 memory `1929->1928 MiB`, no observed contention, and zero graph/native sequence failures. The older paired run remains historical promotion evidence at `5967.267 tokens/sec`; velocity reported GPU contention from a `25%` sample there, so it is same-band protection evidence rather than a clean speed ceiling.

Reopen only as an external diagnostic script, an explicit offline quality check, or a stronger cache-rebuild policy that proves better complete-runtime evidence without live scalar scans. Do not reintroduce a repo-local executable full-scan benchmark baseline.
