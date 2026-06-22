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
  - reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-surface-canonical.json
---

# SNN Language Thought-Surface Production Naming

## Status

Retired from production code on 2026-06-22.

## Why

The old `autonomous_snn_language_thought_surface_*` production names made a
bounded SNN language readout look like a hidden internal thought stream. That
conflicted with the MARULHO rule that language surfaces are operator-facing
Runtime Evidence until grounding, training, promotion gates, rollback evidence,
and Runtime Truth support stronger claims.

## Replacement

The maintained route is the canonical readout-surface chain:

- `snn_language_readout_surface_design`
- `snn_language_readout_surface_preflight`
- `execute_snn_language_readout_surface`
- `snn_language_readout_surface_event_review`

The ledger saves canonical fields such as `snn_language_readout_surface_events`,
`total_snn_language_readout_surface_count`, and
`last_snn_language_readout_surface_recorded_at`. Legacy persisted
`autonomous_snn_language_thought_surface_*` fields migrate once on load/save and
are not exposed as facade, API, or ledger call aliases.

## Evidence

Focused tests prove canonical route JSON contains no retired surface vocabulary
and that service checkpoint save migrates legacy persisted surface fields to the
canonical readout-surface keys.

`reports/bounded_replay_window_20260622/snn-readout-ledger-normalization-readout-surface-canonical.json`
passed with bounded mean `408.799567 ms` versus the benchmark-local legacy
diagnostic `6288.893500 ms` (`16x` work reduction), while the autonomous-chain
bounded mean stayed `906.320467 ms` versus `19796.135733 ms`.

`reports/bounded_replay_window_20260622/hotpath-active-pressure-65536-524288-i32-readout-surface-canonical.json`
processed `524288` tokens at `6012.300 tokens/sec`, p95 `21.322 ms`,
`train_compute=0.134205 ms/token`, with bounded `12/65536` route rows,
`65526` cached transition rows, no observed contention, CPU max `59%`, GPU max
`13%`, RTX memory `1771->1772 MiB`, and zero graph/native sequence failures.

## Revisit Only If

A future ADR explicitly promotes a MARULHO-owned cognition substrate beyond
bounded language readout evidence, with grounding, quality, checkpoint/rollback,
device placement, and long-run throughput proof. Do not restore
thought-surface-named production APIs as compatibility aliases.
