---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/runtime_facade.py
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/snn_language_plasticity_executor.py
  - ../../../src/marulho/service/developmental_autonomy.py
related_docs:
  - ../../../docs/retired-paths.md
  - ../concepts/language-from-spikes.md
  - ../concepts/column-runtime.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/synthetic-readout-newborn-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-newborn-canonical.json
---

# SNN Language Thought-Newborn Developmental Production Naming

The `thought_newborn` production naming path is retired. It kept newborn
integration, critical-period learning, maturation review, and newborn-synapse
pruning behind thought-era ledger/executor/snapshot names after the upstream
readout path had become canonical.

The maintained path is `snn_language_readout_newborn_*` through API schema,
facade, ledger, executor, developmental autonomy, runtime snapshot fields, and
checkpoint persistence. Public routes use `snn-language-readout-newborn-*`, and
the API mapper no longer translates readout-newborn payloads back to
thought-era internals.

Focused verification kept the developmental path checkpoint-backed and bounded:
executor, ledger, API, service-manager, developmental-autonomy, persistence,
replay selection, and checkpoint/reload gates passed. The replay quality report
kept selection on `bucket_indexed_candidate_window`, blocked zero-pressure and
no-anchor controls, and used `0` global fallback cycles.

The hot-path protection report processed `524288` tokens at
`5783.832 tokens/sec`, p95 tick `23.205 ms`, bounded route scoring at
`12/65536`, `65526` cached transition rows,
`state_transition_runs_all_columns=false`, no observed contention, RTX memory
`1915->1913 MiB`, and zero graph/native/sequence failures.
