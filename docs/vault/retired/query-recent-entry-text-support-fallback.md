---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/training/query_runner.py
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/evaluation/query_recent_fallback_retirement_benchmark.py
related_docs:
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../reports/bounded_replay_window_20260622/query-recent-fallback-retired-bucket-only.json
---

# Query Recent Entry Text Support Fallback

The query recent-entry text-support fallback is retired from production query
matching. The old branch widened `bounded_query_memory_match.v1` outside the
routing-owned candidate buckets by calling `collect_recent_entry_indices(...)`
with `require_bucket=false` when selected candidates did not cover the requested
query terms.

Replacement:

- `query_runner.memory_matches_with_report(...)` uses only the
  bucket-indexed candidate window selected from routed bucket IDs.
- Query reports no longer emit `recent_fallback_*` fields.
- Raw text payloads load only after bounded candidate selection and only for the
  selected/returned candidate records that need text support.

Evidence:

- `query-recent-fallback-retired-bucket-only.json`: `pass=true` on a
  `65536`-capacity store, recent collector not called, returned indices `[0]`,
  raw text loaded only for candidate `[0]`, no global candidate/score scan, no
  live tick, no hidden language reasoning, CPU archival placement, CUDA
  available but unused, and mean latency `62.085745 ms`.

Revisit only if a future query-readout design proves a bounded, routed,
source-windowed semantic-support operator that improves grounding quality
without widening into archive-recent scans, raw replay-text reasoning,
every-token work, or GPU-resident archival metadata.
