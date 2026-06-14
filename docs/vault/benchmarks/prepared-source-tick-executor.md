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
  - reports/hot_path_cadence_retired_20260614/stress-32768.json
  - reports/hot_path_cadence_retired_20260614/stress-131072.json
  - reports/sync_free_drift_20260614/stress-32768.json
  - reports/sync_free_drift_20260614/stress-131072.json
  - reports/wide_sequence_quantum_20260614/stress-32768-q8-current.json
  - reports/wide_sequence_quantum_20260614/stress-32768-q16.json
  - reports/wide_sequence_quantum_20260614/stress-32768-q16-repeat.json
  - reports/wide_sequence_quantum_20260614/stress-131072-q16.json
  - reports/host_truth_interval_sweep_20260614/stress-32768-i16-seq.json
  - reports/host_truth_interval_sweep_20260614/stress-32768-i32-seq.json
  - reports/host_truth_interval_sweep_20260614/stress-131072-i32.json
  - reports/native_graph_replay_20260614/stress-131072-parent-native.json
  - reports/native_graph_replay_20260614/stress-131072-parent-disabled.json
---

# Prepared Source Tick Executor

Runtime Sources treats consumption-only prepared queues as immutable cache
generations, and training accepts one complete service text tick while
retaining ordered eight-token device bursts and between-quantum stop checks.

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

## Deferred Slow Memory Cadence

Fixed-cadence slow-memory admission no longer forces ordinary burst quanta
through retained `train_step`. First-token retained/fallback admission and
strong-capture device-ring events remain; routine cadence is reported as
deferred maintenance.

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/hot_path_cadence_retired_20260614/stress-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 8 --timeout-seconds 900 --sample-interval-seconds 0.5`
- Throughput: `3901.906 tokens/sec` over `33.592 s`, versus `3565.968` for the prior long lightweight-metrics run (`1.094x`).
- Latency: mean tick `30.279 ms`, p95 `40.424 ms`.
- Training: `train_compute=0.219858 ms/token`, down from `0.241568`.
- CUDA: RTX 3060, all `131072` transitions, `131056` burst-owned tokens, zero graph/burst failures.
- Runtime Truth: `512` deferred slow-memory cadence events, only `runtime_not_fully_warm` and one `sleep_boundary` fallback, `1024` sequence calls, and `16384` quanta.
- Rejected variant: exact cadence archive payloads in the burst event ring removed slow-memory fallbacks but regressed 32768-token throughput to `3052.470` and `3145.964 tokens/sec`.

## Sync-Free Drift Maintenance

Drift refresh and drift-floor closure no longer force pending burst events to
the host. Burst ticks use winner-local drift only when the host winner mirror is
already fresh; otherwise they compute global drift and report that explicitly.

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/sync_free_drift_20260614/stress-131072.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 8 --timeout-seconds 900 --sample-interval-seconds 0.5`
- Throughput: `4045.419 tokens/sec` over `32.400 s`, versus `3901.906` for deferred slow-memory cadence (`1.037x`).
- Latency: mean tick `29.108 ms`, p95 `39.493 ms`.
- Training: `train_compute=0.211096 ms/token`, down from `0.219858`.
- CUDA: RTX 3060, all `131072` transitions, `131056` burst-owned tokens, zero graph/burst failures.
- Runtime Truth: host-truth syncs fell to `8193`, forced burst-event drains fell to `0`, `2620` drift refreshes were sync-free, and `1310` used global drift because the host winner mirror was stale.

## Wider Training-Owned Sequence Quantum

The maintained service execution quantum is now `16`, but training consumes
each wider quantum as exact ordered eight-token persistent CUDA bursts. This
keeps the neural transition and event-drain semantics unchanged while removing
the old wider-quantum path that fell through to retained per-token
`train_step`.

- Command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/wide_sequence_quantum_20260614/stress-131072-q16.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 900 --sample-interval-seconds 0.5`
- Throughput: `4247.306 tokens/sec` over `30.860 s`, versus `4045.419` for the previous q8 sync-free drift run (`1.050x`).
- Latency: mean tick `27.783 ms`, p95 `43.240 ms`.
- Training: `train_compute=0.200979 ms/token`, down from `0.211096`.
- CUDA: RTX 3060, all `131072` transitions, `131056` burst-owned tokens, `16382` eight-token burst replays, zero graph/burst failures.
- Runtime Truth: `quantum_tokens=16`, `8192` training-owned sequence quanta, `8192` event drains at the configured sixteen-token truth boundary, host-truth syncs `8193`, and forced burst-event drains `0`.
- Short-run guardrail: the first 32768-token q16 run was noisy/slower (`2741.238 tokens/sec`), but the repeat recovered to `3225.467`, slightly above the same-code q8 run at `3180.473`. The long run is the promotion evidence.
- Wider-quantum rejection: clean follow-up runs did not promote q32 or q64. q16/q32/q64 32768-token clean runs measured `3179.769`, `3295.352`, and `3178.163 tokens/sec`; the longer q32 131072-token run reached `3735.329 tokens/sec`, below the retained q16 long evidence at `4247.306`. The maintained default therefore stays q16 until a wider scheduler wins a long, uncontended complete-runtime run.

## Hot-Path-Safe Control Room

The control-room quick-start path now mirrors the promoted service runtime
instead of enabling old always-on higher layers. The curriculum preset uses a
128-token tick, 16-token execution quantum, explicit ingestion prewarm, and
keeps context and binding layers disabled until a conditional scheduler can
wake them without forcing `train_text_burst` into the retained per-token path.
It also promotes the 32-token host-truth cadence used by the current sustained
CUDA evidence.
This fixes the operator-facing path so dashboard runs are comparable to the
true sustained speed evidence rather than a slow compatibility shape.

## Thirty-Two Token Truth And Event Cadence

The device event queue now holds thirty-two tokens, matching the promoted
host-truth cadence. Training skips the Python event-apply call when the graph
reports that no truth packet was drained, so deferred CUDA evidence remains
device-owned until the actual boundary instead of paying an empty host path.

- Baseline command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/host_truth_interval_sweep_20260614/stress-32768-i16-seq.json --target-tokens 32768 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 300 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 16`
- Baseline throughput: `2768.913 tokens/sec`, `train_compute=0.293511 ms/token`, `2049` host-truth syncs, `2048` event drains, and `2046` skipped empty event-apply attempts.
- Promoted command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/host_truth_interval_sweep_20260614/stress-32768-i32-seq.json --target-tokens 32768 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 300 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Promoted throughput: `4237.534 tokens/sec`, `train_compute=0.197902 ms/token`, `1025` host-truth syncs, `1024` event drains, `31743` host-truth skips, and `3070` deferred event bursts.
- Long confirmation: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/host_truth_interval_sweep_20260614/stress-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --timeout-seconds 900 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`
- Long throughput: `4577.595 tokens/sec` over `28.633 s`, mean tick `25.576 ms`, p95 `38.320 ms`, and stop latency `147.713 ms`.
- CUDA evidence: RTX 3060, `resolved_mode=inplace_triton`, all `131072` transitions on CUDA, `4097` host-truth syncs, `126975` skips, `4096` event drains, `12286` deferred bursts, zero forced drains, zero graph/burst failures, and only `runtime_not_fully_warm` plus `sleep_boundary` fallbacks.
- Promotion rule: intervals `8` and `16` are migrated to `32` for legacy unstamped checkpoints. Exact interval `1` remains available for parity/evaluation runs.

## Native Repeated Child Graph Replay

The eight-token burst now keeps the proven one-tick graph body but composes it
as an eight-child parent CUDA graph through a small native extension. This
reduces the host launch boundary to one parent-graph launch per eligible burst
without changing sequential SNN order, source handling, host-truth cadence, or
event-drain semantics.

- Promoted command: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/native_graph_replay_20260614/stress-131072-parent-native.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 300`
- Promoted throughput: `4671.202 tokens/sec` over `28.060 s`, mean tick `25.253 ms`, p95 tick `31.694 ms`, and `train_compute=0.177193 ms/token`.
- CUDA evidence: RTX 3060, all `131072` transitions on CUDA, `16382` native parent-graph attempts/successes, `131056` native-covered burst tokens, `2` parent graphs, zero native fallbacks/failures, zero graph/burst failures, `4097` host-truth syncs, and `126975` skips.
- Runtime Truth: `native_burst_replay_backend=native_repeated_child_graph`, `native_burst_replay_enabled=true`, `native_burst_replay_parent_graph_count=2`, `native_burst_replay_compile_latency_ms=6202.4909`, and `capture_latency_ms=6790.4858`.
- Environment: `velocity_environment.v1` reported `contention.verdict=not_observed`, CPU max `83%`, GPU max `16%`.
- Disabled comparison: `python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/native_graph_replay_20260614/stress-131072-parent-disabled.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 300 --disable-native-burst-replay`
- Disabled throughput: `4340.160 tokens/sec`, `train_compute=0.192680 ms/token`, zero graph/burst failures, and `contention.verdict=not_observed`.
- Promotion delta: `1.076x` over the same-command disabled comparison and `1.020x` over the retained prior top `4577.595 tokens/sec`.
- Rejected diagnostic: the earlier native C++ loop over `cudaGraphLaunch(graph_exec)` was not promoted because it still launched once per token and lost the recorded long comparison (`4159.316` native-loop versus `4347.554` disabled Python replay).

## Remaining Cost

`train_compute=0.177193 ms/token` is still dominant. Source prewarm remains an
explicit startup slow path at about `91.051 s` for the parent-graph
confirmation. Native parent-graph build/capture adds startup cost
(`native_burst_replay_compile_latency_ms=6202.4909`,
`capture_latency_ms=6790.4858`) but is outside measured warm token throughput.
The long run stop latency was `205.315 ms`.

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
