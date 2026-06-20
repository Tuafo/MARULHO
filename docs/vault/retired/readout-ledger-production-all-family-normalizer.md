---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_snapshot_source_window_benchmark.py
  - ../../../tests/test_snn_language_readout_ledger.py
related_docs:
  - ../../retired-paths.md
  - ../concepts/column-runtime.md
  - ../benchmarks/hot-path-latency.md
  - ../benchmarks/replay-cost.md
  - ../papers/replay-consolidation.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/snn-readout-ledger-normalization-production-normalizer-retired.json
  - reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window-production-normalizer-retired-smoke.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-readout-ledger-production-normalizer-retired.json
---

# Readout Ledger Production All-Family Normalizer

## Status

Retired from production code on 2026-06-20.

## Why

After snapshots, record-family writes, dense-label calibration, provenance
lookups, emission history, autonomous output chains, and status projections had
all moved to named bounded source windows, the private production
`SNNLanguageReadoutEvidenceLedger._normalized_state()` callable remained as an
attractive all-family side path. Keeping it in production meant a future helper
could reopen every retained readout-ledger event family before doing one local
lookup, even if the live tick never called it.

The normalization source-window policy export was also production-owned even
though the only remaining all-family comparisons were benchmark diagnostics.

## Replacement

Production has no all-family normalizer callable. Ledger operations use one of
the explicit source-window operators instead:

- `bounded_snn_readout_ledger_snapshot_source_window.v1` for snapshot display.
- `bounded_snn_readout_ledger_record_family_source_window.v1` for one-family
  append/review paths.
- `bounded_snn_readout_known_evidence_hash_source_window.v1` and related
  dense-label/emission/readout windows for selected replay or review evidence.
- `_store_state(...)` for persistence copies, bounded by the event-field helper.

The all-family normalization and full-materialized legacy models remain only
inside `snn_readout_ledger_normalization_source_window_benchmark.py` and
`snn_readout_ledger_snapshot_source_window_benchmark.py` as benchmark-local
retired comparisons. Their reports mark `production_callable=false` and
`benchmark_local_only=true`.

## Evidence

Focused guard:
`PYTHONPATH=src python -m pytest tests\test_snn_language_readout_ledger.py::test_readout_ledger_does_not_expose_all_family_normalizer tests\test_snn_language_readout_ledger.py::test_readout_ledger_snapshot_normalizes_retained_histories_from_source_window tests\test_snn_language_readout_ledger.py::test_readout_ledger_snapshot_reads_only_requested_event_windows tests\test_snn_language_readout_ledger.py::test_readout_ledger_store_state_uses_bounded_event_field_windows -q`
passed `4` tests.

Checkpoint/reload guard:
`PYTHONPATH=src python -m pytest tests\test_service_api.py::ServiceApiTerminusRuntimeTests::test_snn_language_readout_draft_endpoint_generates_bounded_grounded_text -q`
passed and covered save/restore followed by restored readout-ledger and replay
priority reads.

`reports/bounded_replay_window_20260620/snn-readout-ledger-normalization-production-normalizer-retired.json`
passed all checks. The benchmark-local bounded all-family model read `2944`
rows instead of the full-materialized legacy model's `47104` rows (`16x` less
source work), preserved newest-first retention (`1.0` versus `0.0`), and
reduced mean normalization latency from `5807.281164 ms` to `379.736352 ms`.
The autonomous chain comparison preserved hash, review-match, total-count, and
current-pointer parity while checking `4352` target-family rows instead of
`100096` broad-normalized rows and reducing mean latency from `18942.731124 ms`
to `930.299448 ms` (`20.361972x`). CUDA was available on RTX 3060 but unused:
`memory_allocated_mib=0.0`, `memory_reserved_mib=0.0`; Python traced peak was
`430.391802 MiB`.

`reports/bounded_replay_window_20260620/snn-readout-ledger-snapshot-source-window-production-normalizer-retired-smoke.json`
preserved snapshot quality while reading `260` rows instead of `2944`
benchmark-local all-family rows (`11.323077x`) and reducing mean snapshot
latency from `409.182080 ms` to `71.702280 ms` (`5.706682x`), with CUDA
allocation/reservation at `0.0 MiB`.

The `524288`-token protection run
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-readout-ledger-production-normalizer-retired.json`
processed `524288` tokens at `6224.717 tokens/sec`,
`tick_duration_ms.p95=20.898`, `train_compute=0.130755 ms/token`,
`prepare_training=0.006849 ms/token`, and
`finalize_total=0.006249 ms/token`. Runtime Truth kept route scoring bounded at
`12/65536`, returned `10` output candidates, cached `65526` transition rows,
kept `state_transition_runs_all_columns=false`, and recorded zero graph/native
sequence failures. Velocity observed borderline GPU contention at `21%`, so this
is same-band throughput protection, not a clean ceiling claim. RTX 3060 memory
moved `1988->1987 MiB`.

## Revisit Only If

Do not restore a production all-family normalizer. If a future audit needs
all-family comparison, keep it benchmark-local with explicit source budgets,
CPU archival placement, no live tick, no every-token cadence, no hidden language
reasoning, and a fresh long-run proof that the live tick remains protected.
