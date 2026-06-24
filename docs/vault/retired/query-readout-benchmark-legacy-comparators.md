---
type: retired
status: retired
related_code:
  - ../../../src/marulho/evaluation/context_memory_match_benchmark.py
  - ../../../src/marulho/evaluation/query_memory_payload_benchmark.py
  - ../../../src/marulho/evaluation/runtime_concept_memory_lookup_benchmark.py
  - ../../../src/marulho/evaluation/query_episode_readout_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
  - ../concepts/column-runtime.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/context-memory-match-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/query-memory-payload-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/runtime-concept-memory-lookup-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/query-episode-readout-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-query-readout-comparators-removed-default-nosample.json
---

# Query Readout Benchmark Legacy Comparators

Context comparison, query-memory payload materialization, runtime concept lookup, and query episode readout already had maintained bounded/reporting paths. Their benchmark files still kept executable old comparators: report-dropping context readout, eager candidate text payload loading, direct runtime concept archive lookup, and fragment-only episode readout.

The maintained benchmark shape now checks the active bounded readout reports directly. Reports include explicit absence sentinels, candidate or observation budgets, returned-payload budgets, CPU archival placement, no global scan, no live-tick work, and no hidden language reasoning.

Evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260624\context-memory-match-comparator-removed.json` passed with bounded context selection consistency `1.0`, `8` returned text payloads, `8` payload cache hits, `192` candidate limit, and `53.762454 ms` mean bounded latency.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\query-memory-payload-comparator-removed.json` passed with returned-only text payload policy, `5` payloads for `5` returned matches, selected indices `[0, 16, 32, 48, 64]`, and `25.570462 ms` mean bounded latency.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\runtime-concept-memory-lookup-comparator-removed.json` passed explicit memory-index evidence recall at `1.0`, `512` bounded matches, `64` unique payload reads, `448` payload cache hits, no archive iteration, and `7.919283 ms` mean bounded latency.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\query-episode-readout-comparator-removed.json` passed target phrase recovery for `a cat purrs when it feels safe.`, loaded `10` selected neighbor-window payloads across `4` bounded neighbor indices, and averaged `0.512807 ms`.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-query-readout-comparators-removed-default-nosample.json` processed `524288` tokens at `6586.097 tokens/sec`, p95 tick `18.732 ms`, `train_compute=0.123762 ms/token`, bounded route scoring `12/65536`, cached `65526` transition rows, no observed contention, RTX 3060 memory flat at `1886 MiB`, and zero graph/native sequence failures.

Do not restore these comparators in repo-local benchmark code. Query/readout evidence should stay on maintained bounded reports, with explicit payload and candidate budgets.
