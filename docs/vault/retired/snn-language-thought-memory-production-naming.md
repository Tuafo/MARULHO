---
type: retired
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/runtime_facade.py
  - ../../../src/marulho/service/api.py
  - ../../../src/marulho/service/persistence.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/language-from-spikes.md
  - ../modules/service.md
related_benchmarks:
  - reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-memory-canonical.json
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-memory-canonical-rerun.json
---

# SNN Language Thought-Memory Production Naming

## Status

Retired from production code on 2026-06-22.

## Why

The old `autonomous_snn_language_thought_memory_*` production names made a
bounded readout-memory trace look like hidden internal thought. That conflicted
with the MARULHO rule that language/readout memory is Runtime Evidence until
selected replay, consolidation quality, rollback evidence, and Runtime Truth
support stronger claims.

## Replacement

The maintained route is the canonical readout-memory chain:

- `snn_language_readout_memory_design`
- `snn_language_readout_memory_preflight`
- `execute_snn_language_readout_memory`
- `snn_language_readout_memory_event_review`

The ledger saves canonical fields such as `snn_language_readout_memory_events`,
`total_snn_language_readout_memory_count`, and
`last_snn_language_readout_memory_recorded_at`. Legacy persisted
`autonomous_snn_language_thought_memory_*` fields migrate once on load/save and
are not exposed as facade, API, or ledger call aliases.

## Evidence

Focused tests prove canonical memory route JSON contains no retired memory
vocabulary and that service checkpoint save migrates legacy persisted memory
fields to canonical readout-memory keys.

`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-memory-canonical.json`
passed with bounded mean `380.146067 ms` versus the benchmark-local legacy
diagnostic `5619.687233 ms` (`16x` work reduction), while the autonomous-chain
bounded mean stayed `944.358700 ms` versus `19539.615767 ms`.

The accepted paired long hot-path protection rerun
`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-memory-canonical-rerun.json`
processed `524288` tokens at `5987.142 tokens/sec`, p95 `21.546300 ms`,
`train_compute=0.134307 ms/token`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, CPU max `29%`, GPU max
`13%`, GPU memory `1827->1825 MiB`, and zero graph/native sequence failures.

## Revisit Only If

A future ADR explicitly promotes a MARULHO-owned cognition substrate beyond
bounded language readout evidence, with selected replay/consolidation quality,
checkpoint/rollback, device placement, and long-run throughput proof. Do not
restore thought-memory-named production APIs as compatibility aliases.
