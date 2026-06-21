---
type: retired-path
status: retired
related_code:
  - ../../../src/marulho/consolidation/memory_store.py
  - ../../../src/marulho/training/query_runner.py
  - ../../../src/marulho/evaluation/replay_entry_text_payload_opt_in_benchmark.py
related_docs:
  - ../../retired-paths.md
  - ../papers/replay-consolidation.md
  - ../concepts/column-runtime.md
  - ../benchmarks/replay-cost.md
  - ../benchmarks/hot-path-latency.md
related_papers:
  - ../papers/replay-consolidation.md
related_benchmarks:
  - reports/bounded_replay_window_20260620/replay-entry-text-payload-opt-in.json
---

# Implicit Replay Entry Raw Text Payload Default

`DualMemoryStore.replay_entry(...)` no longer returns raw replay text, expanded text, or metadata by default. The old default made text payload loading the easiest path for future replay/recall callers, even though sleep replay had already opted out explicitly.

The maintained API is tensor-first. Callers get assembly, input pattern, routing key, bucket, STC, PRP, consolidation, and replay-priority metadata by default. Raw text payloads require `include_text_payload=True`, and production query/source-bank/context readout uses that opt-in only after bounded candidate or returned-match selection.

The benchmark `reports/bounded_replay_window_20260620/replay-entry-text-payload-opt-in.json` passed on a `65536`-entry store. Default replay-entry reads loaded `0/192` raw text payloads; explicit opt-in loaded `192/192`; bounded query readout still loaded `5` returned-match payloads while reporting no global candidate/score scan, no live tick, no every-token cadence, CPU archival placement, and `language_reasoning=false`.

The paired hot-path report `reports/bounded_replay_window_20260620/hotpath-active-pressure-65536-524288-i32-replay-entry-text-payload-opt-in.json` processed `524288` tokens at `5993.863 tokens/sec`, `tick_duration_ms.p95=21.555`, `train_compute=0.135543 ms/token`, bounded route scoring at `12/65536`, cached `65526` transition rows, RTX 3060 memory `1878->1879 MiB`, no observed contention, and zero graph/native sequence failures.

Reopen only with a reviewed API contract that preserves explicit source-window budgets, no hidden language reasoning through replay text, and long-run live-tick protection.
