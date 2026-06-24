---
type: retired
status: retired
related_code:
  - ../../../src/marulho/evaluation/concept_signature_lookup_benchmark.py
  - ../../../src/marulho/evaluation/concept_frontier_scope_benchmark.py
  - ../../../src/marulho/evaluation/frontier_gap_bounded_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
  - ../concepts/column-runtime.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - ../../../../MARULHO_reports/bounded_replay_window_20260623/concept-signature-legacy-baseline-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260623/concept-frontier-legacy-baseline-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260623/frontier-gap-legacy-baseline-removed.json
  - ../../../../MARULHO_reports/bounded_replay_window_20260623/hotpath-active-pressure-65536-524288-i32-semantic-frontier-legacy-baselines-removed-default-nosample.json
---

# Semantic Frontier Benchmark Legacy Comparators

ConceptStore signature lookup, concept-frontier metrics, and frontier-gap planning already had maintained bounded production paths. Their benchmark files still kept executable legacy comparators: archive-list materializing signature lookup, full slow-memory concept-frontier scan, and full raw-window frontier-gap scan.

The maintained benchmark shape now uses seeded expected quality instead of executing those old paths. Reports include explicit retired-path absence fields, source budgets, CPU archival placement, latency, Python trace-memory, CUDA allocation, and no global candidate/score scan.

Evidence:

- `..\..\MARULHO_reports\bounded_replay_window_20260623\concept-signature-legacy-baseline-removed.json` passed with seeded cosine minimum `0.9999998211860657`, `8` max indices per source, `65536` retired archive-materializing rows removed, `1.309419 ms` mean bounded lookup latency, `0.002109 MiB` Python peak, and `0.0 MiB` CUDA allocation.
- `..\..\MARULHO_reports\bounded_replay_window_20260623\concept-frontier-legacy-baseline-removed.json` passed with seeded target hit rate `1.0`, target indices `[7, 1031]`, `64/8192` scored entries, `16` source probes, `7.402244 ms` mean bounded latency, `0.089705 MiB` Python peak, and `0.0 MiB` CUDA allocation.
- `..\..\MARULHO_reports\bounded_replay_window_20260623\frontier-gap-legacy-baseline-removed.json` passed with expected term recall `1.0`, `192/65536` bounded candidates, `16.086825 ms` mean bounded latency, missing-collector fallback still empty/no-scan, `0.070801 MiB` Python peak, and `0.0 MiB` CUDA allocation.
- `..\..\MARULHO_reports\bounded_replay_window_20260623\hotpath-active-pressure-65536-524288-i32-semantic-frontier-legacy-baselines-removed-default-nosample.json` processed `524288` tokens at `6496.154 tokens/sec`, p95 tick `19.229 ms`, `train_compute=0.125202 ms/token`, bounded route scoring `12/65536`, cached `65526` transition rows, no observed contention, RTX 3060 memory `1866->1866 MiB`, and zero graph/native sequence failures.

Do not restore these comparators in repo-local benchmark code. If a full archive diagnostic is needed for a paper, keep it external to production and active benchmark modules, with explicit source-size accounting and no live-tick authority.
