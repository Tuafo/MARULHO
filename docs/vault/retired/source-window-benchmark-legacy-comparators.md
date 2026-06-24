---
type: retired
status: retired
related_code:
  - ../../../src/marulho/evaluation/bucket_candidate_source_window_benchmark.py
  - ../../../src/marulho/evaluation/sfa_sample_scope_benchmark.py
  - ../../../src/marulho/evaluation/awake_ripple_scope_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
  - ../concepts/column-runtime.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/bucket-candidate-source-window-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/sfa-sample-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/awake-ripple-comparator-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260624/hotpath-active-pressure-65536-524288-i32-source-window-comparators-removed-default-nosample.json
---

# Source-Window Benchmark Legacy Comparators

Hot-bucket candidate source construction, selected-window SFA sampling, and awake-ripple tagging already had maintained bounded production paths. Their benchmark files still kept executable retired comparators: full hot-bucket materialization, full-buffer SFA sampling, and scalar/vector full-memory awake-ripple scans.

The maintained benchmark shape now checks seeded bounded quality directly and reports retired-path absence. It keeps archival metadata on CPU, records candidate/sample budgets, and avoids global scan or hidden language reasoning report fields.

Evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260624\bucket-candidate-source-window-comparator-removed.json` passed with newest-candidate hit rate `1.0`, `32` bounded source reads from a `65536`-entry hot bucket, `0` materialized rows, no full-bucket scan, `0.036322 ms` mean latency, and `0.0 MiB` CUDA allocation.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\sfa-sample-comparator-removed.json` passed with selected-window sample purity `1.0`, `64` samples from `192` candidates, no global candidate scan, `0.534319 ms` mean latency, and `0.0 MiB` CUDA allocation.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\awake-ripple-comparator-removed.json` passed with `0` scalar scans, `0` vector scans, `256` wake-bucket scans, `10` last candidates within a `10`-candidate budget, `10` tagged traces, and `1.271697 ms` mean scoped latency.
- `..\..\MARULHO_reports\bounded_replay_window_20260624\hotpath-active-pressure-65536-524288-i32-source-window-comparators-removed-default-nosample.json` processed `524288` tokens at `6580.539 tokens/sec`, p95 tick `19.094 ms`, `train_compute=0.123491 ms/token`, bounded route scoring `12/65536`, cached `65526` transition rows, no observed contention, RTX 3060 memory flat at `1875 MiB`, and zero graph/native sequence failures.

Do not restore these comparators in repo-local benchmark code. If a paper needs a full-source diagnostic, keep it external, source-size-accounted, and unable to act as a production replay or live-tick path.
