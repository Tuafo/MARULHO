---
type: benchmark
status: current
related_code:
  - src/marulho/service/runtime_sources.py
  - src/marulho/service/brain_runtime.py
  - src/marulho/training/trainer.py
related_docs:
  - CONTEXT.md
  - docs/adr/0006-persistent-text-tick-executor.md
  - docs/research-living-brain.md
related_papers: []
related_benchmarks:
  - reports/prepared_source_tick_executor_20260614/stress-32768.json
  - reports/prepared_source_tick_executor_20260614/stress-131072.json
  - reports/device_lightweight_metrics_20260614/stress-32768.json
  - reports/device_lightweight_metrics_20260614/stress-131072.json
---

# Prepared Source Tick Executor

Runtime Sources treats consumption-only prepared queues as immutable cache
generations, and training accepts one complete service text tick while
retaining ordered eight-token quanta and between-quantum stop checks.

## Sustained Evidence

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/prepared_source_tick_executor_20260614/stress-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 8 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.02`
- Throughput: `3359.378 tokens/sec` over `39.017 s`, versus the prior long baseline at `2126.013` (`1.580x`).
- Latency: mean tick `35.667 ms`, p95 `48.458 ms`.
- Preparation: `0.008415 ms/token`, down `94.6%`.
- CUDA: RTX 3060, all `131072` transitions, `116680` burst-owned tokens, zero graph/burst failures.
- Ownership: `1024` training-owned sequence calls, `16384` quanta, stop boundary between quanta.
- Cache: `1024` stable-generation skips, one cache write, zero failures.

The focused CUDA test compares the sequence result with 64 sequential
transitions and requires exact state tensors.

## Device-Burst Lightweight Metrics

Source concept sampling no longer asks training for full per-token metrics on
ordinary prepared source ticks. Training returns final lightweight metrics from
the CUDA burst result packet, while full metrics remain an explicit evaluator
request.

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/device_lightweight_metrics_20260614/stress-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 8 --timeout-seconds 900 --sample-interval-seconds 0.5`
- Throughput: `3565.968 tokens/sec` over `36.756 s`, versus the prior long prepared-source evidence at `3359.378` (`1.061x`).
- Latency: mean tick `33.367 ms`, p95 `43.292 ms`.
- Training: `train_compute=0.241568 ms/token`, down from `0.259876`.
- CUDA: RTX 3060, all `131072` transitions, `126952` burst-owned tokens, zero graph/burst failures.
- Runtime Truth: `1024` sequence calls, `16384` quanta, stop boundary between quanta, `1024` stable-generation skips, one cache write, zero cache failures.

## Remaining Cost

`train_compute=0.241568 ms/token` is still dominant. Source prewarm remains an
explicit startup slow path at `82.713 s` for 131072 prepared tokens. The long
run stop latency improved to `124.712 ms`.

## Rejected Continuation

`reports/device_sequence_burst_20260614/stress-32768.json` tested an
eight-token sequence CUDA Graph around the existing burst tick body. Focused
CUDA parity passed, but complete service throughput regressed to
`2249.691 tokens/sec` with `train_compute=0.372436 ms/token`. The retained
burst executor rerun at
`reports/device_sequence_burst_20260614/stress-32768-retained.json` restored
`3171.826 tokens/sec` and `train_compute=0.273302 ms/token`.

The next continuation is therefore not another nested CUDA Graph. It is either
a real fused/persistent device kernel for the recurrent tick cluster, or a
lower-cost Runtime Truth/event-drain boundary that preserves bounded evidence
without synchronizing as often.
