---
type: retired
status: retired
related_code:
  - ../../../src/marulho/training/query_runner.py
  - ../../../src/marulho/consolidation/memory_store.py
related_docs:
  - ../concepts/replay-window.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/query-memory-store-owned-row-access.json
  - reports/bounded_replay_window_20260622/synthetic-query-row-access.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-row-access-noprofile-rerun.json
---

# Query Runner Direct Slow Memory Row Reads

`query_runner.memory_matches_with_report(...)` used to select a bounded
candidate window, then read slow-memory arrays directly for row scoring,
returned payloads, and neighboring source text. That kept a second row-access
path beside the memory store and made it easier for future query work to drift
back toward archive-shaped recall.

The maintained path is store-owned:

- `DualMemoryStore.collect_query_memory_match_indices(...)` selects bounded
  candidate indices.
- `DualMemoryStore.query_match_row(...)` serves scoring rows and explicit text
  payload rows under `bounded_query_memory_match_row.v1`.
- `DualMemoryStore.query_neighbor_source_row(...)` serves source-neighbor text
  under `bounded_query_neighbor_source_row.v1`.
- `query_runner.py` consumes store summary stats and has no production
  references to slow-memory archive arrays.

Evidence:

- `reports/bounded_replay_window_20260622/query-memory-store-owned-row-access.json`
  preserved selected-index parity with the diagnostic eager payload path,
  reduced raw text payload reads from `192` to `5`, read `197` bounded rows,
  and improved mean latency from `42.525 ms` to `33.718 ms`.
- `reports/bounded_replay_window_20260622/synthetic-query-row-access.json`
  kept sleep associative recall bounded to `4` selected-window queries with
  mean best input-pattern distance `5.960464477539063e-08`.
- `reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-query-row-access-noprofile-rerun.json`
  processed `524288` tokens at `5935.802 tokens/sec`, kept route scoring at
  `12/65536`, cached `65526` transition rows, used CUDA on the RTX 3060, and
  recorded zero graph/native/sequence failures. Velocity observed GPU-side
  contention, so this is same-band protection evidence, not a speed ceiling.

Do not reintroduce direct query-runner archive reads. New query recall work
must add or reuse a bounded memory-store row surface with selection criteria,
row budget, device placement, quality evidence, latency cost, and long-run
hot-path protection.
