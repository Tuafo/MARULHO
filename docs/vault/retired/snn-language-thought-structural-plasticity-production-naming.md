---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/runtime_facade.py
  - ../../../src/marulho/service/api.py
related_docs:
  - ../../retired-paths.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-structural-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json
---

# SNN Language Thought Structural Plasticity Production Naming

The old thought-structural production names made bounded readout-driven
growth/prune evidence look like hidden thought. They are retired.

The maintained path is:

- `snn_language_readout_structural_plasticity_design`
- `snn_language_readout_structural_plasticity_preflight`
- `execute_snn_language_readout_structural_plasticity`
- `snn_language_readout_structural_plasticity_event_review`

Checkpoint load/save keeps only canonical readout-ledger fields such as
`snn_language_readout_structural_plasticity_events`,
`total_snn_language_readout_structural_plasticity_count`, and
`last_snn_language_readout_structural_plasticity_applied_at`. Noncanonical
readout-ledger state is dropped rather than migrated through compatibility
aliases.

Evidence:

- `snn-readout-ledger-normalization-readout-structural-canonical.json`:
  bounded mean `568.337767 ms`, retired diagnostic mean `7518.428000 ms`,
  `16x` source-work reduction, downstream autonomous-chain bounded mean
  `967.423500 ms` versus `21131.022800 ms`.
- `hotpath-active-pressure-65536-524288-i32-readout-structural-canonical.json`:
  `524288` measured tokens at `5885.572 tokens/sec`, p95 `22.879800 ms`,
  `train_compute=0.136747 ms/token`, bounded `12/65536` route rows,
  `state_transition_runs_all_columns=false`, no observed contention, GPU memory
  `1741->1970 MiB`, and zero graph/native sequence failures. Prewarm
  (`418.781 s`) completed before the measured throughput window.

Do not reintroduce thought-structural production APIs without a new ADR and
repeated quality, device, and long-run throughput evidence.
