---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/brain_runtime.py
  - ../../../src/marulho/training/trainer.py
  - ../../../src/marulho/evaluation/source_tick_sleep_deferral_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-source-tick-sleep-replay-deferred.json
---

# Source Tick Sleep Replay Fallback

BrainRuntime source ticks used a fallback `train_step(...)` call when a tick needed per-token metrics or when burst execution was unavailable. That fallback inherited the trainer default for sleep maintenance, so a live service/source tick could run deep, micro, or repair sleep replay merely because replay was due. That path mixed replay/consolidation with the latency-sensitive source tick.

The maintained path passes `allow_sleep_maintenance=False` from source-tick fallback. Due sleep is recorded as deferred trainer maintenance, and explicit trainer sleep/replay windows remain the only execution path for replay computation. Archival storage stays CPU-resident; this retirement does not move metadata to CUDA, scan full memory, or use replay text as hidden language reasoning.

The CPU deferral benchmark `reports/bounded_replay_window_20260620/source-tick-sleep-replay-deferred.json` passed with service fallback sleep calls `0`, explicit allowed slow-path sleep calls `1`, visible `sleep_maintenance_deferred=1`, and a bounded source-tick memory budget. The paired `524288`-token CUDA protection run stayed in the same throughput band at `5993.959 tokens/sec`, `train_compute=0.135624 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, no observed contention, flat RTX 3060 memory at `1959 MiB`, and zero graph/native sequence failures.

Reopen this only if a selected slow-window scheduler proves replay, grounding, or reconstruction quality gains and repeated 6k-ish long-run evidence keeps source ticks protected without service-owned replay, live-tick scans, or hidden language reasoning.
