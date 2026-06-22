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
  - reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-consolidation-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-consolidation-canonical.json
---

# SNN Language Thought Consolidation Production Naming

The old `autonomous_snn_language_thought_consolidation_*` production names made
bounded local readout-consolidation plasticity look like a hidden thought
stream. They are retired.

The maintained path is:

- `snn_language_readout_consolidation_design`
- `snn_language_readout_consolidation_preflight`
- `execute_snn_language_readout_consolidation`
- `snn_language_readout_consolidation_event_review`

Checkpoint load/save migrates old persisted consolidation fields once into
`snn_language_readout_consolidation_events`,
`total_snn_language_readout_consolidation_count`, and
`last_snn_language_readout_consolidated_at`. API/facade/ledger call aliases are
not retained.

Evidence:

- `snn-readout-ledger-normalization-readout-consolidation-canonical.json`:
  bounded mean `371.891600 ms`, retired diagnostic mean `5302.309467 ms`,
  `16x` source-work reduction, downstream autonomous-chain bounded mean
  `921.487733 ms` versus `19456.105067 ms`.
- `hotpath-active-pressure-65536-524288-i32-readout-consolidation-canonical.json`:
  `5938.794 tokens/sec`, p95 `22.043 ms`, `train_compute=0.135649 ms/token`,
  route scoring `12/65536`, no observed contention, GPU memory
  `1829->1828 MiB`, and zero graph/native/sequence failures.

Do not reintroduce thought-consolidation production APIs without a new ADR and
repeated quality, device, and long-run throughput evidence.
