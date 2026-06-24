---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/query_runner.py
  - ../../../src/marulho/evaluation/query_memory_payload_benchmark.py
  - ../../../src/marulho/evaluation/context_memory_match_benchmark.py
  - ../../../src/marulho/evaluation/sleep_repair_replay_bounded_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/query-memory-payload-query-row-no-replay-entry.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/context-memory-query-row-cache-no-replay-entry.json
  - ../../../../../MARULHO_reports/bounded_replay_window_20260624/sleep-repair-replay-dense-prepare-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-replay-entry-api-retired-noprofile.json
---

# Implicit Replay Entry Raw Text Payload Default

`DualMemoryStore.replay_entry(...)` is fully retired and removed. The first retirement made raw text opt-in, but keeping the generic mutating row reader still left a side-door API beside the promoted bounded row readers.

The maintained row APIs are named by purpose: `sleep_repair_replay_row(...)` is the mutating tensor-only slow repair row, `replay_recall_row(...)` is read-only sleep recall, and `query_match_row(...)` is query/source-bank/context recall. Text payloads are available only through explicit bounded query/source rows after selection.

Current external checks:

- `..\..\MARULHO_reports\bounded_replay_window_20260622\query-memory-payload-query-row-no-replay-entry.json`: selected indices matched the diagnostic eager payload path, bounded text payloads were `5` instead of `192`, and the report used `returned_similarity_matches_only`.
- `..\..\MARULHO_reports\bounded_replay_window_20260622\context-memory-query-row-cache-no-replay-entry.json`: two context reads preserved selected indices, loaded `8` payloads instead of `16`, and reused `8` query-row cache hits.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\sleep-repair-replay-dense-prepare-comparator-removed.json`: sleep repair used `sleep_repair_replay_row(...)`, improved mean anchor distance by `0.076463`, deferred `8` missing routing-key rows, made `0` dense input-assembly calls, removed the dense-prepare comparator, and kept raw text/language reasoning/global scan/live tick flags false.

Reopen only as a new named row API with explicit source budget, mutation/read-only semantics, CPU archival placement, no hidden language reasoning through replay text, and repeated long-run live-tick protection.
