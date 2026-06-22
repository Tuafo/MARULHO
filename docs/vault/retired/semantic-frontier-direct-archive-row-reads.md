---
type: retired
status: retired
related_code:
  - ../../../src/marulho/semantics/frontier.py
  - ../../../src/marulho/gap_planner.py
  - ../../../src/marulho/training/autonomy_runner.py
  - ../../../src/marulho/consolidation/memory_store.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/source-bank-store-owned-row-reader.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/frontier-gap-store-owned-row-reader.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/concept-frontier-store-owned-row-reader.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/semantic-row-reader-replay-quality.json
---

# Semantic Frontier Direct Archive Row Reads

Semantic/source-frontier recall used to keep multiple row readers alive after
candidate selection was already bounded:

- `bank_memory_matches_with_report(...)` used `replay_entry(...)` to fetch
  returned source-bank text.
- `frontier_gap_plan(...)` scored frontier entries from direct `slow_*`
  archive arrays and `_effective_capture_strength(...)`.
- `concept_frontier_metrics_with_report(...)` used direct routing-key and
  capture/consolidation helpers beside the memory-store API.

That shape made read-only semantic planning depend on mutating replay/STC row
surfaces and preserved archive-array access outside `DualMemoryStore`.

The maintained path is one store-owned row surface:

- Candidate windows are still selected before row reads.
- `DualMemoryStore.query_match_row(...)` serves scoring and opt-in text rows
  under `bounded_query_memory_match_row.v1`.
- Production `_effective_capture_strength(...)` is removed.
- Semantic frontier reports state no direct slow-memory row reads, no
  `replay_entry(...)`, no STC advance, no live tick, and CPU archival/score
  placement.

Evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260622\source-bank-store-owned-row-reader.json`
  preserved selected-index parity at `1.0`, read `196` store-owned rows
  (`192` scoring rows plus `4` text rows), loaded raw text only for returned
  rows, and reduced mean latency from `958.681 ms` to `160.781 ms`.
- `..\..\MARULHO_reports\bounded_replay_window_20260622\frontier-gap-store-owned-row-reader.json`
  preserved term recall at `1.0`, read `192/65536` rows, and reduced mean
  latency from `229.118 ms` to `8.897 ms`.
- `..\..\MARULHO_reports\bounded_replay_window_20260622\concept-frontier-store-owned-row-reader.json`
  passed quality, bounded-scan, latency, and live-tick gates with `64` row
  reads at `8192` capacity, `top1_match=true`, no direct slow-memory row
  reads, and no capture helper.
- `..\..\MARULHO_reports\bounded_replay_window_20260622\semantic-row-reader-replay-quality.json`
  kept sleep recall passing with `1` query and best input-pattern distance
  `5.96046447753906e-08`.

Do not restore semantic/frontier direct archive reads, `replay_entry(...)` as a
read-only text reader, or `_effective_capture_strength(...)` in production.
Future semantic recall work must add or reuse a store-owned bounded row
surface with selection criteria, source budget, quality evidence, latency
cost, device placement, and long-run hot-path protection.
