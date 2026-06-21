---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/service/brain_runtime.py
  - ../../../src/marulho/service/living_status.py
  - ../../../src/marulho/service/status_read_model.py
  - ../../../src/marulho/service/status_runtime.py
  - ../../../src/marulho/evaluation/live_memory_summary_projection_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/live-memory-summary-projection.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-live-memory-summary-projection.json
---

# Live Memory Summary Full Scan

Full `DualMemoryStore.summary_stats()` is retired from trainer telemetry, BrainRuntime summaries, living-loop status, and status Runtime Truth projections. That full path advances slow-memory state time and builds tensors over all retained slow entries to compute exact means and fragility, so it is a consolidation/quality-window operation rather than live display telemetry.

The maintained live projection is `DualMemoryStore.live_summary_stats()`. It reports `bounded_memory_summary_projection.v1`, scalar fill/counter aliases, last replay reports, `summary_full_memory_scan=false`, `summary_scan_entry_count=0`, and `summary_projection_read_only=true`. It does not advance STC tag/PRP decay, scan archival metadata, select replay, apply plasticity, move archive metadata to CUDA, or use replay text as hidden language reasoning.

The 65536-entry benchmark `reports/bounded_replay_window_20260620/live-memory-summary-projection.json` passed with scalar fill/report parity and read-only STC state. The retired full summary scanned `65536` entries at `658.789240 ms` mean, while the bounded projection averaged `0.149500 ms`.

The paired `524288`-token hot-path run stayed in band at `6024.783 tokens/sec`, `train_compute=0.135003 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, no observed contention, RTX 3060 memory `1959->1958 MiB`, and zero graph/native sequence failures.

Reopen this only for a specific operator workflow that requires exact current all-entry memory statistics, and only after repeated long-run evidence proves no live-tick tax. Replay selection, consolidation, and STC decay must remain explicit slow-window work.
