---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/column_transition_runtime.py
  - ../../../src/marulho/evaluation/bucket_consolidation_cache_lookup_benchmark.py
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
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-bucket-consolidation-cache-lookup.json
---

# Bucket Consolidation Level Full Scan

Scalar `DualMemoryStore.bucket_consolidation_level(...)` full-memory scans are retired from live metric lookup. A single winner/bucket consolidation metric must not iterate every retained slow-memory entry in retained or graph metric ticks.

The maintained path is `bucket_consolidation_level_cache_lookup.v1`. `DualMemoryStore` maintains CPU bucket consolidation metadata as entries are stored or selected replay updates consolidation. Scalar reads use that cache, report `full_memory_scan=false`, `scan_entry_count=0`, cache generation, cache hit/miss status, and return a no-scan miss if the cache is absent. Explicit `bucket_consolidation_tensor(...)` rebuilds remain checkpoint load, graph capture/prewarm, offline quality, selected-replay recovery, or benchmark-local work.

The 65536-entry benchmark `reports/bounded_replay_window_20260620/bucket-consolidation-cache-lookup.json` matched the retired scalar scan within `1e-6`, reported `cache_hit`, scanned `0` entries, and reduced mean lookup latency from `12.999192 ms` to `0.016260 ms`.

The paired `524288`-token hot-path run stayed in band at `5967.267 tokens/sec`, `tick_duration_ms.p95=22.005`, `train_compute=0.135870 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, RTX 3060 memory `1963->1964 MiB`, and zero graph/native sequence failures. Velocity reported GPU contention from a `25%` sample, so this is same-band protection evidence rather than a clean speed ceiling.

Reopen only for explicit offline diagnostics or if a stronger cache-rebuild policy proves better complete-runtime evidence without live scalar scans.
