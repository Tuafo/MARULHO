---
type: retired
status: active
related_code:
  - ../../../src/marulho/service/snn_language_readout_ledger.py
  - ../../../src/marulho/service/replay_runtime.py
  - ../../../src/marulho/service/runtime_facade.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json
  - reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json
---

# Readout Known-Hash Report-Dropping Path

The hash-only known readout-evidence verification helpers are retired and
removed. `SNNLanguageReadoutEvidenceLedger` now exposes one production path:
`known_readout_evidence_hashes_with_report()`, which returns the hash set plus
`bounded_snn_readout_known_evidence_hash_source_window.v1`.

Replay design, dry-run, plasticity preflight, bridge, and evaluated
replay-artifact recording must carry that source-window report. The replay
controller validates the report before recording an evaluated replay artifact
and persists `readout_evidence_source_window_hash` so permit/checkpoint
verification recomputes the same bounded provenance.

The focused benchmark
`reports/bounded_replay_window_20260620/snn-replay-artifact-known-readout-source-window.json`
passed with known-readout source window `1/8`, CPU archival placement, no global
candidate or score scan, no raw text payload, no language reasoning, no
live-tick/every-token work, `0.0 MiB` CUDA allocation/reservation, and
`0.014095 MiB` traced Python peak. Indexed replay provenance verification still
reduced worst-case retained lookup checks from `256` to `4` (`64x`).

The accepted long protection rerun
`reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-known-readout-source-window-rerun.json`
processed `524288` tokens at `6007.228 tokens/sec`, with
`train_compute=0.134831 ms/token`, bounded `12/65536` route rows, `65526`
cached transition rows, zero graph/native sequence failures, and flat RTX 3060
memory. Velocity still observed GPU contention, so this is same-band protection
evidence, not a new speed ceiling.
