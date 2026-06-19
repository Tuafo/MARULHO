---
type: benchmark
status: draft
related_code:
  - ../../../src/marulho/evaluation/service_benchmark.py
  - ../../../src/marulho/evaluation/binding_wake_benchmark.py
  - ../../../src/marulho/evaluation/cross_modal_wake_benchmark.py
  - ../../../src/marulho/evaluation/column_scheduler_benchmark.py
  - ../../../src/marulho/evaluation/hot_window_benchmark.py
  - ../../../src/marulho/evaluation/sequence_input_staging_benchmark.py
  - ../../../src/marulho/evaluation/compiled_hot_path_kernel_benchmark.py
  - ../../../src/marulho/evaluation/snn_emission_review_replay_policy_source_window_benchmark.py
  - ../../../src/marulho/evaluation/emission_replay_context_review_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_replay_evaluation_context_window_benchmark.py
  - ../../../src/marulho/evaluation/status_replay_path_source_window_benchmark.py
  - ../../../src/marulho/evaluation/snn_readout_ledger_normalization_source_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_replay_target_window_benchmark.py
  - ../../../src/marulho/evaluation/language_plasticity_replay_window_benchmark.py
  - ../../../src/marulho/evaluation/readout_ledger_rollout_candidate_window_benchmark.py
  - ../../../src/marulho/service/status_read_model.py
  - ../../../src/marulho/evaluation/promoted_scheduler_checkpoint.py
  - ../../../tests/test_service_benchmark.py
related_docs: []
related_papers: []
related_benchmarks:
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-capped-replay-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-collection.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-query-memory-match.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-524288-i32-query-memory-payload.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-profile-rerun.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/emission-replay-context-review-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json
  - reports/bounded_replay_window_20260618/status-replay-path-source-window.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-profile.json
  - reports/bounded_replay_window_20260618/hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-store-state-known-hash-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-update-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-dense-label-update-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-confidence-use-source-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-record-family-append.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-record-family-append.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-autonomous-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-autonomous-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-training-probe-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-output-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-output-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-text-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-text-surface-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-surface-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-surface-chain.json
  - reports/bounded_replay_window_20260619/snn-readout-ledger-normalization-language-generation-chain.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-generation-chain.json
  - reports/bounded_replay_window_20260619/readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json
  - reports/bounded_replay_window_20260619/language-plasticity-replay-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json
  - reports/bounded_replay_window_20260619/readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json
  - reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json
---

# Hot Path Latency

Latency-sensitive runtime surface checks.

## Sustained Velocity Recheck, 2026-06-14

The retained previous best long-run evidence was
`reports/host_truth_interval_sweep_20260614/stress-131072-i32.json`:
`4577.595 tokens/sec` over `131072` tokens, `train_compute=0.181855 ms/token`,
`prepare_training=0.008358 ms/token`, `finalize_total=0.007461 ms/token`,
`4097` host-truth syncs, `126975` host-truth skips, `16382` burst replays,
`131056` burst-owned tokens, zero forced drains, zero graph/burst failures, and
RTX 3060 CUDA execution.

The promoted native parent-graph evidence was
`reports/native_graph_replay_20260614/stress-131072-parent-native.json`:
`4671.202 tokens/sec` over `131072` tokens, `train_compute=0.177193 ms/token`,
`prepare_training=0.008906 ms/token`, `finalize_total=0.007671 ms/token`,
`4097` host-truth syncs, `126975` host-truth skips, `16382` burst replays,
`131056` burst-owned tokens, `16382` native repeated-child parent-graph
launches, zero native fallbacks/failures, zero forced drains, zero graph/burst
failures, and RTX 3060 CUDA execution. The same long command with native replay
disabled reached `4340.160 tokens/sec` at
`reports/native_graph_replay_20260614/stress-131072-parent-disabled.json`, so
the promoted parent graph measured `1.076x` over the disabled replay loop and
`1.020x` over the prior retained top. Both reports recorded
`velocity_environment.v1` with `contention.verdict=not_observed`.

The current refreshed base comparison is
`reports/base_comparison_20260615/current-native-131072-i32.json`:
`4992.049 tokens/sec` over `131072` tokens, `train_compute=0.166575 ms/token`,
`prepare_training=0.007805 ms/token`, `finalize_total=0.007046 ms/token`,
`4097` host-truth syncs, `126975` skips, `16382` native parent-graph
successes, `131056` native-covered burst tokens, zero native fallbacks/failures,
zero graph/burst failures, and `contention.verdict=not_observed`. The same
shape with native replay disabled at
`reports/base_comparison_20260615/disabled-native-131072-i32.json` reached
`4530.883 tokens/sec` with `train_compute=0.185263 ms/token` and no observed
contention, so the current native replay delta is `1.102x`.

The cost is startup, not measured warm throughput: the promoted parent-graph
run reported `capture_latency_ms=6790.4858` and
`native_burst_replay_compile_latency_ms=6202.4909`; the refreshed base run
reported `5961.3172` and `5452.4969`. Keep that visible in Runtime Truth and
avoid quoting the new top for cold start.

The CUDA conditional-WHILE q16 sequence executor is now promoted for eligible
text sequences. It keeps the proven one-tick graph body but moves burst-loop
control into a CUDA conditional node plus a device counter kernel. The first
clean probe at `reports/conditional_sequence_20260615/conditional-while16-131072-i32.json`
measured `5559.473 tokens/sec`, `train_compute=0.146978 ms/token`, `8190`
conditional parent launches, `131040` conditional-owned tokens, zero
sequence/native fallbacks, zero sequence/native failures, host-truth cadence
`4097/126975`, and `velocity_environment.v1` contention `not_observed`, beating
the same-session native8 rerun at `5035.537 tokens/sec`.

Repeated paired promotion gates kept that win in both orders:
`reports/conditional_sequence_promotion_clean_20260615/pair-a-native8-131072-i32.json`
reached `5485.105 tokens/sec`, while `pair-a-conditional16-131072-i32.json`
reached `5883.805`; then `pair-b-conditional16-131072-i32.json` reached
`6027.856`, while `pair-b-native8-131072-i32.json` reached `5816.477`. All four
reports had `velocity_environment.v1` contention `not_observed`, zero
sequence/native fallbacks, and zero sequence/native failures.

The post-promotion default run at
`reports/conditional_sequence_promotion_default_20260615/post-promotion-default-conditional16-rerun-131072-i32.json`
used no executor override and reached `6116.646 tokens/sec` with
`train_compute=0.134167 ms/token`, `8190` conditional launches covering
`131040` tokens, zero sequence/native fallbacks or failures, host-truth cadence
`4097/126975`, `capture_latency_ms=5482.6059`,
`native_sequence_loop_compile_latency_ms=4970.7865`, and
`velocity_environment.v1` contention `not_observed`. The former native8
comparison run at
`post-promotion-native8-131072-i32.json` stayed clean at
`5329.542 tokens/sec`.

The completion-audit rerun after the Runtime Truth gate-status patch used the
current default with no executor override at
`reports/conditional_sequence_completion_audit_20260615/current-default-conditional16-131072-i32.json`.
It reached `5955.123 tokens/sec`, `train_compute=0.138342 ms/token`, `8190`
conditional launches covering `131040` tokens, zero sequence/native fallbacks or
failures, host-truth cadence `4097/126975`, `capture_latency_ms=5164.9948`,
`native_sequence_loop_compile_latency_ms=4741.5911`, and
`velocity_environment.v1` contention `not_observed`. Runtime Truth reported
`native_sequence_executor_requested=cuda_graph_conditional_while`,
`persistent_executor_repeated_child_burst_tokens=8`,
`persistent_executor_sequence_loop_tokens=16`,
`native_sequence_loop_sequential_state_parity_gate_passed=true`, and
`native_sequence_loop_bounded_quality_gate_passed=true`.

The 2026-06-16 selector-cleanup run removed the old sequence-executor override
surface while keeping conditional-WHILE q16 fixed as the promoted path. The
131072-token CUDA stress gate at
`reports/column_scheduler_20260616/sequence-executor-selector-cleanup-8192-131072-i32.json`
reached `6133.925 tokens/sec`, `train_compute=0.129945 ms/token`,
`prepare_training=0.006425 ms/token`, `finalize_total=0.005955 ms/token`, and
`tick_p95=21.499 ms` with no observed contention. Runtime Truth reported
`cuda_graph_sequence_executor=conditional_while`,
`native_sequence_executor_requested=cuda_graph_conditional_while`,
`native_sequence_loop_success_count=8191`, `native_sequence_loop_token_count=131056`,
zero graph/native/sequence failures, `route_input_rows_scored=10/8192`,
`state_transition_cached_count=8182`, and
`state_transition_runs_all_columns=false`.

The route-vote selector cleanup then fixed `predictive_route_vote_mode` to the
promoted `cuda_graph_text` path, migrated retired `tensor`/`fused_triton_text`
checkpoint values forward, and moved old route-vote modes behind explicit
evaluation overrides. The first 131072-token CUDA gate at
`reports/column_scheduler_20260616/route-vote-selector-cleanup-8192-131072-i32.json`
reached `6011.457 tokens/sec`, `train_compute=0.130446 ms/token`, and
`tick_p95=23.112 ms`, but `velocity_environment.v1` reported GPU contention.
The requested rerun at
`reports/column_scheduler_20260616/route-vote-selector-cleanup-8192-131072-i32-rerun.json`
reached `6141.720 tokens/sec`, `train_compute=0.129445 ms/token`,
`prepare_training=0.006344 ms/token`, `finalize_total=0.005838 ms/token`, and
`tick_p95=21.018 ms`. Runtime Truth reported
`route_vote_requested_mode_source=promoted_config`,
`route_vote_config_mode=cuda_graph_text`, `route_input_rows_scored=10/8192`,
`route_rows_run_all_columns=false`, `state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, `native_sequence_loop_success_count=8191`,
`native_sequence_loop_token_count=131056`, and zero graph/native/sequence
failures. The rerun still observed GPU utilization at the configured contention
threshold, so treat it as in-band cleanup evidence, not a new top-speed claim.

The post column-scheduler long-run check used the same 131072-token default
conditional16 stress shape after promoting retained CPU candidate deep-sleep
filtering and predictive-location caching. The current working tree report at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-column-scheduler.json`
reached `5653.175 tokens/sec`, `train_compute=0.143769 ms/token`,
`prepare_training=0.007176 ms/token`, and `finalize_total=0.006404 ms/token`.
It preserved the promoted CUDA sequence counters: `8190` conditional launches
covering `131040` tokens, zero sequence/native failures or sequence/native
fallbacks, host-truth cadence `4097/126975`, only the expected text-burst
fallbacks (`runtime_not_fully_warm` and `sleep_boundary`), and
`velocity_environment.v1` contention `not_observed`. A same-host clean `HEAD`
control at
`reports/column_scheduler_20260615/head-current-default-conditional16-131072-i32-control.json`
reached `5807.210 tokens/sec` with the same logical counters, so the scheduler
slice measured `-2.65%` versus same-host `HEAD`, `-5.07%` versus the
completion-audit baseline, and `-7.58%` versus the post-promotion top run. This
keeps the long-run path in the same broad 6k-ish band, but exact same
throughput is not proven; the current run also showed a startup capture/compile
outlier (`capture_latency_ms=22008.1023`,
`native_sequence_loop_compile_latency_ms=21544.8843`) that is outside warm
throughput but should be rechecked before making a cold-start claim.

The follow-up lazy scheduler-materialization run used the same 131072-token
default conditional16 stress shape after fixing long-run cached-state parity.
The report at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-lazy-scheduler.json`
reached `5867.701 tokens/sec`, `train_compute=0.141240 ms/token`,
`prepare_training=0.006435 ms/token`, and `finalize_total=0.005910 ms/token`.
It preserved the promoted CUDA sequence counters: `8190` conditional launches
covering `131040` tokens, zero sequence/native failures or sequence/native
fallbacks, host-truth cadence `4097/126975`, only the expected text-burst
fallbacks (`runtime_not_fully_warm` and `sleep_boundary`), and
`velocity_environment.v1` contention `not_observed`. This is `+1.04%` versus
the same-host `5807.210` control, `+3.79%` versus the prior column-scheduler
run at `5653.175`, `-1.47%` versus the completion-audit `5955.123`, and
`-4.07%` versus the post-promotion top run at `6116.646`. The answer to the
throughput gate is therefore: same broad 6k-ish long-run band, not exact top-run
parity.

The explicit longer-run rerun requested for the scheduler audit used the same
131072-token stress shape at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-lazy-scheduler-long-rerun.json`.
It reached `5886.247 tokens/sec`, `train_compute=0.141223 ms/token`,
`prepare_training=0.006347 ms/token`, and `finalize_total=0.005770 ms/token`.
CUDA selected the RTX 3060, `velocity_environment.v1` again reported
`contention.verdict=not_observed`, and Runtime Truth preserved `8190`
conditional sequence-loop successes over `131040` tokens, zero sequence/native
fallbacks or failures, and host-truth cadence `4097/126975`. Relative to the
documented long baselines, this is `+1.36%` versus same-host `HEAD` at
`5807.210`, `+0.32%` versus the prior lazy-scheduler rerun, `-1.16%` versus the
completion-audit `5955.123`, and `-3.77%` versus the post-promotion top
`6116.646`. Treat the scheduler throughput answer as stable 6k-ish sustained
runtime, not proof that the exact top historical run is preserved.

The following CPU scaling audit added an exact no-relaxation/no-clamp
homeostasis wake fast path but kept predictive lazy wake on ordered replay after
a vectorized predictive catch-up broke long winner parity. The valid sweep at
`reports/column_scheduler_20260615/cpu-scaling-large-lazy-fast-homeostasis-final.json`
preserved winner sequences at `2048`, `8192`, and `16384` columns and kept all
scoped specialist work at `10` candidates with `runs_all_columns=false`.
However, `neutral_or_better_all_sizes=false`; scoped means were `11.3276475`,
`11.382465`, and `11.70441 ms`. The longer `8192` report at
`reports/column_scheduler_20260615/cpu-8192-lazy-fast-homeostasis-long.json`
preserved winner parity and improved median latency from `7.257` to
`6.4019 ms`, but mean latency regressed by `6.27%`. The scheduler now has
post-fix total-column correctness evidence, but the cost-neutral durable
scaling gate remains open.

The next scheduler cost audit tried replacing ordered per-step predictive wake
replay with a vectorized closed-form materializer for skipped non-winner
predictive updates. It remained candidate-bounded and improved one 8192-column,
400-sample CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-lazy-exact-predictive-fast-long.json`,
but the longer sweep at
`reports/column_scheduler_20260615/cpu-scaling-large-lazy-exact-predictive-fast-long-sweep.json`
failed the parity gate (`all_winner_sequences_equal=false`), and repeated
seed-20260616 reruns such as
`reports/column_scheduler_20260615/cpu-8192-lazy-exact-predictive-fast-seed20260616-long-rerun2.json`
showed one-tick winner shifts near fallback-threshold boundaries. The matching
131072-token CUDA stress report still reached `5907.750 tokens/sec`, selected
the RTX 3060, reported no observed contention, and preserved zero
sequence/native fallbacks or failures, but that report is not a scheduler
promotion because the CPU cached-state parity gate failed. Retain ordered
candidate-bounded predictive replay; treat the throughput answer as the prior
documented stable 6k-ish path, not a vectorized predictive-wake speed claim.

The following scheduler-ownership audit introduced a training-owned
`column_wake_plan` boundary so the retained route carries one awake mask and
wake/sleep/fallback reason through predictive vote, competition, predictive
update/location update, and homeostasis. The first implementation built the
legacy sleep-filter report dict every tick and measured
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-wake-plan-scheduler.json`
at `5808.990 tokens/sec`, `train_compute=0.142889 ms/token`,
`prepare_training=0.006470 ms/token`, and
`finalize_total=0.005898 ms/token`. After moving legacy report materialization
to Runtime Truth/report time and making the wake-plan object slotted, the same
131072-token CUDA shape at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-wake-plan-scheduler-slots.json`
reached `5822.624 tokens/sec`, `train_compute=0.142080 ms/token`,
`prepare_training=0.006576 ms/token`, and
`finalize_total=0.005888 ms/token`. Both runs preserved RTX 3060 CUDA execution,
`8190` conditional loop successes over `131040` tokens, zero sequence/native
fallbacks or failures, host-truth cadence `4097/126975`, and no observed
contention. The retained CPU 8192-column A/B
`reports/column_scheduler_20260615/cpu-8192-wake-plan-scheduler-slots.json`
preserved exact winners and bounded all reported specialist work at `10/8192`,
but scoped mean latency was `12.24272625 ms` versus `7.4809125 ms`, so this is
a scheduler ownership/Runtime Truth promotion rather than a speed promotion.
Treat `6k-ish` as a noisy sustained-runtime band and compare the stage
`ms/token` values before claiming improvement.

The follow-up Column Runtime Truth Projection audit changed the live
`column_runtime` projection so top-level awake counts, awake IDs, vote samples,
wake/sleep reasons, fallback reason, and service evidence come from the
training-owned `ColumnWakePlan` instead of a second report-local top-k mask.
Focused verification passed with `10 passed` across column runtime, status, and
scheduler benchmark tests, and `54 passed` in `tests\test_predictive_columns.py`.
The CPU reports prove bounded work but not speed neutrality: the 2048-column
A/B at
`reports/column_scheduler_20260615/cpu-2048-wake-plan-runtime-truth-projection.json`
preserved exact winners and `runs_all_columns=false`, but scoped mean complete
step was `14.2402075 ms` versus `6.7495825 ms`; the 512/2048/8192 scaling sweep
at
`reports/column_scheduler_20260615/cpu-scaling-wake-plan-runtime-truth-projection.json`
kept awake count at `10` while `neutral_or_better_all_sizes=false`. The latest
8192 rerun at
`reports/column_scheduler_20260615/cpu-8192-wake-plan-runtime-truth-projection.json`
kept bounded specialist work at `10/8192`, but `winner_sequence_equal=false`
and scoped mean was `10.63430875 ms` versus `8.33335625 ms`, so it is not a
promotion gate. The valid 131072-token CUDA comparison is
`reports/column_scheduler_20260615/current-conditional16-131072-i32-after-wake-plan-truth-projection.json`:
it reached `4975.507 tokens/sec`, `train_compute=0.159667 ms/token`,
`prepare_training=0.009256 ms/token`, and `finalize_total=0.008210 ms/token`,
with RTX 3060 CUDA selected, no observed contention, `8190` conditional loop
successes over `131040` tokens, and zero sequence/native failures or fallbacks.
This is below the recent `5.8k-6.1k` sustained band, so throughput parity is not
proven. The excluded
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-wake-plan-truth-projection.json`
run used the old sequence-selector surface, selected the native repeated-child
fallback instead of conditional-WHILE, and observed contention; do not compare
that run to the 6k-ish baseline. That selector surface has since been removed.

The fused retained CPU candidate-transition cleanup moved prediction-error,
location/velocity, prediction-weight, and cached predictive materialization
work behind one candidate-scoped core call instead of three split calls over the
same wake mask. The focused parity test matched the old split sequence state,
and the 8192-column CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-fused-candidate-transition.json`
preserved exact winners with bounded specialist work at `10/8192`.
Scoped mean complete-step latency improved against the previous scoped
wake-plan baseline (`12.24272625` to `10.637575 ms`), but same-run all-column
mean was still lower at `9.8858125 ms`; this is a CPU-path cleanup, not a
neutral-or-better scheduler speed promotion. The corresponding long CUDA run at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-fused-candidate-transition.json`
reached `5889.241 tokens/sec`, `train_compute=0.140840 ms/token`,
`prepare_training=0.006423 ms/token`, and
`finalize_total=0.005995 ms/token`, with RTX 3060 CUDA selected, no observed
contention, `8190` conditional loop successes over `131040` tokens, zero
sequence/native fallbacks or failures, and host-truth cadence `4097/126975`.
CUDA therefore remains in the documented broad 6k-ish band, while the retained
CPU scoped path is faster than its prior scoped implementation but still has an
open complete-step cost gate.

The next retained scheduler cleanup removed a bounded deep-sleep backfill that
could not filter anything before `dead_column_steps`, reused the predictive
vote materialization in the following candidate transition, exposed
`candidate_subset_completed_noop` for repeated same-tick candidate
materialization, and fixed checkpoint restore so cached predictive wake truth is
derived from saved step stamps. Focused verification passed:
`python -m pytest tests\test_predictive_columns.py tests\test_column_scheduler_benchmark.py -q`
reported `56 passed`, and
`python -m pytest tests\test_column_runtime.py tests\test_adr_runtime_state_ownership.py tests\test_column_scheduler_benchmark.py -q`
reported `17 passed`.

The CPU A/B at
`reports/column_scheduler_20260615/cpu-8192-age-gate-single-materialization-completed-cache.json`
preserved exact winners and kept scoped predictive vote, predictive update,
predictive location, candidate sleep filter, and wake-plan work bounded at
`10/8192` with `runs_all_columns=false`. It did not pass the cost gate:
same-run all-column mean/median were `6.57934625/6.1021 ms`, while scoped
mean/median were `11.2669375/8.43225 ms`. The scaling sweep at
`reports/column_scheduler_20260615/cpu-scaling-age-gate-single-materialization-completed-cache.json`
kept awake work bounded at `10` for `512`, `2048`, and `8192` columns and
reported `scoped_never_runs_all_columns=true`, but
`neutral_or_better_all_sizes=false`.

The CUDA A/B at
`reports/column_scheduler_20260615/cuda-8192-age-gate-single-materialization-completed-cache.json`
kept predictive vote bounded at `10/8192`, but Runtime Truth correctly reported
`runs_all_columns=true` because predictive update/location stayed dense with
fallback reason `cuda_sparse_prediction_update_launch_bound_dense_retained`.
Same-run all-column mean/median were `14.55886625/13.3442 ms`, while scoped
mean/median were `17.98001875/17.09485 ms`. This confirms the CUDA rule for
now: keep the dense GPU predictive transition until a fused or lower-launch
sparse CUDA path beats dense complete-runtime evidence.

The direct CUDA predictive-writeback experiments tested that lower-level
question in isolation on a synthetic 8192-column CUDA checkpoint. The first run
at
`reports/column_scheduler_20260615/cuda-8192-predictive-writeback-scope-experiment.json`
rejected eager candidate indexing: it updated only `10/8192` rows and matched
dense candidate rows exactly, but mean/median/p95 regressed from dense
`3.0670796875/2.7036/4.8754 ms` to `6.7971921875/6.40065/10.229 ms`.

The follow-up three-arm run at
`reports/column_scheduler_20260615/cuda-8192-predictive-writeback-scope-triton-experiment.json`
kept dense writeback at `3.080762890625/2.74815/4.378 ms`, eager candidate
indexing at `7.0080195312499995/6.5332/10.4585 ms`, and the new fused Triton
candidate writeback at `0.1748/0.1383/0.3041 ms`. The Triton candidate rows matched
dense with maximum absolute deltas of `2.384185791015625e-07` for location,
`1.4901161193847656e-08` for velocity, `2.2351741790771484e-08` for prediction
weights, `1.7881393432617188e-07` for prediction error, and
`5.960464477539063e-08` for confidence; prediction failure streaks matched
exactly. This promoted the candidate predictive writeback idea only when it is
inside the existing transition state boundary, not as a separate side launch:
the standalone helper and benchmark arm are removed from active code after
`ColumnTransitionRuntime` integrated candidate predictive updates into the
promoted fused in-place/graph transition and won complete `train_step` evidence.

A direct live-runtime integration attempt was rejected and removed. The valid
1024-column direct `inplace_triton` A/B at
`reports/column_scheduler_20260615/cuda-direct-inplace-predictive-scope-ab.json`
preserved winners and reduced predictive update scope to `10/1024` with
`runs_all_columns=false`, but complete step mean/median/p95 regressed from
`8.607221875/7.88135/14.0249 ms` to
`27.7788925/14.7495/118.536 ms`. The attempted 8192-column direct run is not
promotion evidence because all-column warmup hit Triton's tensor-size limit.
The live CUDA runtime therefore keeps dense predictive update/location until
candidate materialization and writeback are fused into a lower-overhead runtime
boundary with complete-step evidence.

That lower-overhead boundary is now the promoted fused in-place/graph candidate
predictive transition. Focused CUDA tests prove the direct kernel branch,
non-graph runtime path, CUDA graph path, checkpoint-load default behavior, and
retired config surface. The first opt-in 131072-token probe at
`reports/column_scheduler_20260615/fused-inplace-candidate-predictive-131072-i32.json`
reached `6083.586 tokens/sec` with `0.132856 ms/token` train compute, no
observed contention, and zero sequence/native failures or fallbacks. After
promotion and removal of the `cuda_candidate_predictive_transition_mode` switch,
the same longer gate at
`reports/column_scheduler_20260615/promoted-fused-candidate-predictive-131072-i32.json`
reached `6141.078 tokens/sec`, with `train_compute=0.126682 ms/token`,
`prepare_training=0.006629 ms/token`, `finalize_total=0.004710 ms/token`,
`candidate_predictive_transition_mode=fused_inplace`, `130816` candidate
predictive executions, `132647424` cumulative cached predictive row-skips, zero
graph/sequence fallbacks, and `contention.verdict=not_observed`. This restores the 6k-ish
sustained band while making candidate predictive update/location a real
scheduler execution effect rather than a report-only claim.

After removing the leftover standalone candidate predictive writeback helper
and benchmark arm, the cleanup verification run at
`reports/cleanup_candidate_predictive_writeback_20260616/current-131072-i32.json`
processed `131072` tokens at `6044.116 tokens/sec` with
`train_compute=0.135451 ms/token`, `prepare_training=0.006398 ms/token`, and
`finalize_total=0.006237 ms/token`. CUDA selected the RTX 3060, contention was
not observed, `candidate_predictive_transition_mode=fused_inplace` stayed
active with `130816` executions and `132647424` cumulative cached row-skips, and
graph/sequence failures remained zero. This is neutral cleanup evidence: the single promoted
fused transition path kept the sustained band after the retired side path was
deleted.

After removing configurable `compiled` predictive dense transition from runtime
config/core and deleting the isolated predictive transition benchmark module,
the same long cleanup gate at
`reports/compiled_predictive_retirement_20260616/current-131072-i32.json`
processed `131072` tokens at `5949.698 tokens/sec`, with
`train_compute=0.136408 ms/token`, `prepare_training=0.006460 ms/token`, and
`finalize_total=0.006228 ms/token`. CUDA stayed on the RTX 3060 with no
observed contention, `candidate_predictive_transition_mode=fused_inplace`
remained active, route/vote executed `131072` times, the conditional-WHILE
sequence loop succeeded `8190` times over `131040` burst tokens, and graph,
sequence, and native failures remained zero. This is cleanup evidence, not a
new speed ceiling: it is slightly below the previous `6044.116 tokens/sec`
candidate-writeback cleanup check but still within the maintained 6k-ish band
while removing the stale compiled runtime branch.

The longer CUDA gate the user requested stayed healthy. The 131072-token run at
`reports/column_scheduler_20260615/current-default-conditional16-131072-i32-after-age-gate-materialization-cache.json`
reached `5909.600 tokens/sec`, `train_compute=0.140558 ms/token`,
`prepare_training=0.006582 ms/token`, and
`finalize_total=0.005748 ms/token`, with RTX 3060 CUDA selected, no observed
contention, `8190` conditional loop successes over `131040` tokens, zero
sequence/native fallbacks or failures, and host-truth cadence `4097/126975`.
Compared with the previous fused-candidate run (`5889.241 tokens/sec`,
`train_compute=0.140840 ms/token`), this is neutral/stable long-run evidence,
not proof that the retained CPU scheduler A/B is solved.

The retained CPU lazy state-transition cleanup removed the last hidden
all-column state-transition tax from the retained candidate scheduler path. The
final 8192-column CPU A/B at
`reports/column_scheduler_20260616/cpu-8192-lazy-state-transition-final.json`
preserved exact winners and bounded predictive vote/update/location,
candidate-sleep filtering, wake-plan awake count, competitive scoring, and
state transition at `10/8192` with `scoped_runs_all_columns=false`. Scoped
complete-step cost improved from `7.8869` to `5.7537 ms` median and from
`9.15943125` to `8.3204845 ms` mean. The longer scaling sweep at
`reports/column_scheduler_20260616/cpu-scaling-lazy-state-transition-long.json`
kept awake work bounded at `10` and never ran all columns, but
`neutral_or_better_all_sizes=false`; the 2048 arm also diverged after the
deep-sleep filter changed the awake mask. Treat that sweep as boundedness
evidence, not durable total-column cost completion.

The matching 131072-token CUDA stress rerun at
`reports/column_scheduler_20260616/lazy-state-transition-current-131072-i32-after-marker.json`
processed `131072` tokens at `6068.986 tokens/sec`, about
`0.164772 ms/token`, with `train_compute=0.135598 ms/token`,
`prepare_training=0.006294 ms/token`, and `finalize_total=0.005997 ms/token`.
It preserved RTX 3060 CUDA execution, conditional-WHILE q16 coverage,
`8190` sequence-loop successes over `131040` tokens, zero sequence/native
fallbacks or failures, host-truth cadence `4097/126975`, and active
`route_vote_deep_sleep_filter`. This is within the maintained 6k-ish band and
slightly below the previous `6135.026`/`6141.078 tokens/sec` fused baselines.
CUDA state transition remains dense and truthfully reports
`state_transition_runs_all_columns=true`; the scalar materialized-step metadata
only avoids an extra hot-path CUDA bookkeeping fill.

After promoting the graph route/vote scheduler from checkpoint opt-in to the
default `predictive_route_vote_mode="cuda_graph_text"` and retiring the
remaining configurable `legacy` dense-transition bypass, the 131072-token CUDA
gate at
`reports/column_scheduler_20260616/default-route-vote-promotion-current-131072-i32.json`
processed `131072` tokens at `6014.550 tokens/sec`
(`0.166263 ms/token` wall-clock), with `train_compute=0.136192 ms/token`,
`prepare_training=0.006288 ms/token`, and
`finalize_total=0.006012 ms/token`. Runtime Truth reported
`route_vote_requested_mode=route_vote_resolved_mode=cuda_graph_text`,
`131072` route/vote executions, `route_vote_kernel_variant=two_stage_route_vote`,
active `route_vote_deep_sleep_filter`, `1024` route input rows selecting
`10` awake candidates, `8190` conditional-WHILE q16 sequence successes over
`131040` burst tokens, and zero graph, sequence, or native failures. This is a
real-path/default cleanup gate, not a new speed ceiling: it is slightly below
the previous `6068.986 tokens/sec` lazy-state-transition rerun and the
benchmark reported GPU contention at the configured threshold. The path remains
inside the maintained 6k-ish band while removing an opt-in-only scheduler
default and the stale `legacy` transition selector.

The column-metabolism cleanup promoted per-column cost and memory pressure from
placeholder Runtime Truth fields to checkpointed training-owned state. The
retained route can now apply a candidate-local memory-pressure filter from
cached `ColumnMetabolismState.memory_pressure` values without running an
all-column pressure census or calling the memory-store consolidation builder in
the tick. Focused tests covered high-pressure candidate skip behavior, service
projection without scheduler ownership, checkpoint restore of metabolism state,
and numeric cost/pressure projection in `column_runtime`.

CPU evidence is bounded but not a speed claim. The 8192-column A/B at
`reports/column_scheduler_20260616/cpu-8192-column-metabolism.json` preserved
the winner sequence and kept predictive vote/update/location, wake-plan count,
candidate filter output, and column-metabolism update count at `10/8192` with
`runs_all_columns=false`, but scoped mean complete `train_step` was
`13.74354 ms` versus `9.92042125 ms` for the all-column comparison. The scaling
run at `reports/column_scheduler_20260616/cpu-scaling-column-metabolism.json`
kept awake/metabolism work at `10` for `1024`, `8192`, and `32768` columns and
never ran all columns, but `neutral_or_better_all_sizes=false`. Treat these CPU
reports as boundedness evidence only.

The matching longer CUDA gate used the same 131072-token real path as the
current 6k-ish baseline:
`reports/column_scheduler_20260616/column-metabolism-current-131072-i32.json`.
It processed `131072` tokens at `5960.035 tokens/sec` with
`train_compute=0.135423 ms/token`, compared with `6014.550 tokens/sec` and
`train_compute=0.136192 ms/token` for
`default-route-vote-promotion-current-131072-i32.json`. Runtime Truth evidence
showed `route_vote_resolved_mode=cuda_graph_text`, `131072` route-vote
executions, route-vote deep-sleep filtering from `1024` rows to `10` eligible
route candidates, `8190` q16 sequence-loop successes over `131040` tokens, and
zero graph, sequence, or native failures. `velocity_environment.v1` reported
`contention_observed` with GPU busy, so this is stable same-band evidence, not a
new speed ceiling. The follow-up route-owner pressure slice no longer uses
post-selection pressure filtering: CUDA masks high-pressure route rows before
candidate and winner selection when cached pressure evidence exists, and keeps
the pressure gate disabled when evidence is absent.

The next route-owner usefulness slice keeps the same promoted path and adds one
cached scheduler signal to the fused route-vote mask. The 32768-column
131072-token gate at
`reports/column_scheduler_20260616/usefulness-scheduler-32768-131072-i32.json`
processed `131072` tokens at `5886.235 tokens/sec` with
`train_compute=0.134212 ms/token`, `tick_duration_ms.p95=22.614`,
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`, and
`candidate_predictive_transition_cached_count=32758`. Runtime Truth reported
`usefulness_enabled=true`, `usefulness_applied=true`,
`filtered_low_usefulness_count=0`, `usefulness_threshold=0.1`,
`usefulness_source=predictive_confidence_error_win_rate_and_cost_candidate_cache`,
zero route/state/predictive fallback reasons, zero graph/native/sequence
failures, and no observed contention. The same-shape previous fixed-count gate
was `6163.265 tokens/sec` with `train_compute=0.132789 ms/token` but observed
GPU contention, so this result is throughput-neutral enough for the scheduler
truth claim and remains inside the expected 6k-ish sustained band; it is not a
new speed ceiling.

ADR 0007 now records the promoted boundary and the next executor direction:
further work should move below local graph composition into C++/CUDA, Triton,
persistent-kernel, or hybrid sequence ownership only if it beats the promoted
conditional q16 default with the same parity and Runtime Truth gates.
The post-promotion Runtime Truth surface now carries explicit
`native_sequence_loop_sequential_state_parity_gate_*` and
`native_sequence_loop_bounded_quality_gate_*` fields so the long-run report
states whether the loaded sequence executor is covered by the focused parity
or bounded quality gate.

The first current rerun after the route/vote experiment measured only
`2784.022 tokens/sec` at
`reports/direct_route_vote_20260614/stress-131072-clean-retained-interval32-rerun.json`.
Runtime Truth showed the same logical path: `131072` executions, `16382` burst
replays, `131056` burst-owned tokens, `4097` host-truth syncs, zero graph/burst
failures, zero forced drains, and `route_vote_kernel_variant=two_stage_route_vote`.
The drop came from stage timing inflation, especially
`train_compute=0.310189 ms/token`, plus slower preparation/finalization.

After freeing some machine resources, the same current retained path reran at
`reports/perf_regression_probe_20260614/current-131072-i32-after-free.json` and
recovered to `3804.642 tokens/sec` with the same logical CUDA counters:
`131072` executions, `16382` burst replays, `131056` burst-owned tokens, zero
graph/burst failures, zero forced drains, and
`route_vote_kernel_variant=two_stage_route_vote`. `train_compute` improved to
`0.222647 ms/token`, but still trailed the retained best run's
`0.181855 ms/token`, so this remains a host-condition reproducibility issue
rather than a proved code regression.

The old good commit `f5d8ba2d` was rerun in a clean temporary worktree under the
same host conditions and measured `2413.061 tokens/sec` for `32768` tokens in
`reports/perf_regression_probe_20260614/f5d8-32768-i32-current-host.json`.
That falsifies a simple "later commit caused the 4.5k drop" diagnosis. Host
probes during the slow run showed all CPU cores at `100%`, the GPU in WDDM
`P3`, about `25%` graphics utilization before MARULHO compute, and active
League/Riot/browser/Codex processes. Treat the 4.5k number as the retained
uncontended best evidence, while current "right now" speed must be measured in
an uncontended profile.

`continuous_runtime_stress_benchmark` now writes `velocity_environment.v1`
next to every long-run report. The field is collected before and after the
measured window, outside the cognitive tick, using best-effort CPU counters and
`nvidia-smi` GPU state. Its `contention.verdict` is not a correctness verdict;
it is the comparability signal needed before treating a lower long-run speed as
an architecture regression.

The proof run at
`reports/velocity_environment_20260614/stress-4096-env.json` completed `4096`
sequential tokens at `3867.015 tokens/sec`, selected CUDA on the RTX 3060,
executed `4096` persistent graph and in-place Triton transitions, and reported
zero graph/burst failures. `velocity_environment.v1` reported
`contention.verdict=not_observed`, with CPU max `87%`, GPU max `3%`, and the
field marked `not_hot_path=true`. This proves the stress report now carries
run-condition evidence; it is not a new speed promotion over the retained
`4577.595 tokens/sec` long-run baseline.

The direct route/vote fusion candidate was rejected: the profiled 8192-token
direct run measured `2381.587 tokens/sec` versus `2408.630` for the same
profiled boundary before direct selection, and the 32768-token direct clean run
measured `2266.882` versus `2359.929` after reverting to the retained
two-stage route/vote path. This confirms that the next real speed target is a
lower-level device-owned multi-tick executor or persistent sequence kernel, not
a local one-block top-k fusion that still leaves one CUDA Graph replay per
token.

Boundary-aware source-sequence input staging is now promoted as the maintained
input-ring policy. The same-process reversed A/B at
`reports/sequence_input_staging_20260614/sequence-input-staging-ab-clean.json`
measured sequence staging at `3365.278 tokens/sec` versus per-quantum staging at
`3007.088 tokens/sec` (`1.119x`). The active sequence arms reduced measured
graph input-stage calls from `1536` to `1024` over `16384` tokens while staging
`12288` source-sequence tokens and preserving zero graph/burst failures. A
profiled short A/B was noisy and failed the speed gate (`0.967x`), but it proved
the Runtime Truth counters and showed the same stage-call reduction.

The complete warm runtime proof at
`reports/sequence_input_staging_20260614/stress-65536-segment-staging.json`
processed `65536` tokens at `3593.347 tokens/sec` with CUDA selected on the RTX
3060, `65536` graph-backed executions, zero graph/burst failures, and active
sequence staging for `65408` tokens. `velocity_environment.v1` reported GPU
contention, so this is not a new top-speed claim over the retained
`4577.595 tokens/sec` run. It does prove the promoted staging boundary executes
inside the service-style path and keeps the dominant bottleneck at
`train_compute=0.234968 ms/token`.

## Commands

- Search tests: `rg -n "hot|path|latency" tests src`
- Focused tests: `python -m pytest tests\test_service_benchmark.py`
- Local benchmark:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --checkpoint reports\service_benchmark_cycle_configured\tiny.pt --output reports\service_benchmark_cycle_configured\service-benchmark.json --trace-dir reports\service_benchmark_cycle_configured\traces --env-root reports\service_benchmark_cycle_configured --web-dist-dir MARULHO_UI\dist --create-synthetic-checkpoint --configure-local-source --local-source-tick-steps 1"`
- Regression gate:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-before reports\service_benchmark_cycle_configured\service-benchmark.json --compare-after reports\service_benchmark_cycle_configured\service-benchmark.json --output reports\service_benchmark_regression_gate\comparison.json"`
- Accept configured benchmark as a reviewed baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --accept-baseline-from reports\service_benchmark_cycle_configured\service-benchmark.json --accepted-by codex-local-cycle --baseline-label configured-source-cpu-2026-06-09 --baseline-note \"Accepted local configured-source CPU benchmark for regression-gate smoke comparison.\" --output reports\service_benchmark_baseline\accepted-baseline.json"`
- Compare a run against the accepted baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-baseline reports\service_benchmark_baseline\accepted-baseline.json --compare-after reports\service_benchmark_cycle_configured\service-benchmark.json --output reports\service_benchmark_baseline\comparison.json"`
- Run a fresh configured-source benchmark and compare it against the accepted baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --run-against-baseline reports\service_benchmark_baseline\accepted-baseline.json --checkpoint reports\service_benchmark_cycle_configured\tiny.pt --output reports\service_benchmark_baseline_fresh_cycle --trace-dir reports\service_benchmark_baseline_fresh_cycle\traces --web-dist-dir MARULHO_UI\dist --configure-local-source --local-source-tick-steps 1"`
- Paired CPU/CUDA device comparison:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-devices --checkpoint reports\service_benchmark_device_compare\tiny.pt --output reports\service_benchmark_device_compare --web-dist-dir MARULHO_UI\dist --configure-local-source --local-source-tick-steps 1"`

## Latest Known Result

Measured on 2026-06-09 with a tiny synthetic checkpoint, a generated local file source, and one manual Terminus tick on the local CPU test environment. Raw JSON lives under ignored `reports/service_benchmark_cycle_configured/service-benchmark.json`.

- Benchmark success: `true`
- Total endpoint sweep latency: `1466.864 ms`
- Setup endpoints: `terminus_configure`, `terminus_tick`
- Setup total latency: `452.745 ms`
- Setup evidence: `24` tick tokens processed from `benchmark_local_source`; setup is marked `not_hot_path=true`
- Hot-path endpoints: `feed`, `query`, `respond`
- Hot-path total latency: `818.798 ms`
- Hot-path p95 latency: `439.258 ms`
- Hot-path budget verdict: within `1000.0 ms` p95 and `3000.0 ms` total budgets
- Regression gate status: `passed`
- Regression tolerance: `25%`
- Regression gate allowed after p95: `549.072 ms`
- Regression gate allowed after total: `1023.497 ms`
- Regression gate grouping: setup and slow-path endpoints did not leak into hot path
- Accepted baseline status: `accepted`
- Accepted baseline id: `service-benchmark-baseline:cc257251119a5335`
- Accepted baseline label: `configured-source-cpu-2026-06-09`
- Accepted baseline report hash: `cc257251119a53356ead486f456bfce7cd950337877d12d2b2add7140dc97645`
- Baseline comparison status: `passed`; the compared after-report hash matched the accepted baseline hash in this smoke comparison
- Fresh baseline-run bundle status: `passed`
- Fresh baseline-run bundle path: `reports/service_benchmark_baseline_fresh_cycle/bundle-summary.json`
- Fresh run total endpoint sweep latency: `1368.191 ms`
- Fresh run hot-path p95 latency: `432.406 ms`
- Fresh run hot-path total latency: `739.666 ms`
- Fresh run comparison bounds: allowed p95 `549.072 ms`, allowed total `1023.497 ms`
- Fresh run comparison hashes: baseline `cc257251119a53356ead486f456bfce7cd950337877d12d2b2add7140dc97645`, after `9a15573b07e26727fa256d8e4841de791f73117ad49abc53852f3bb28c6bc45c`
- Validation discovery: `/terminus/validation/reports` recognizes `marulho_service_benchmark_regression_gate`, `marulho_service_benchmark_accepted_baseline`, and `marulho_service_benchmark_baseline_run_bundle`, exposing Runtime Truth, hot-path budget, grouping, configured-source, accepted-baseline identity, source report hash, baseline snapshot hash, operator acceptance hash, fresh-run hashes, integrity statuses, evidence freshness, failed-check summary fields, and inert operator action hints for review
- Dashboard visibility: the Validation UI renders the latest regression gate, fresh benchmark bundle, and accepted benchmark baseline as separate operator cards. The benchmark cards show freshness status and evidence age, alerting on stale or unknown timestamps. The bundle card shows status, Runtime Truth, fresh hot-path p95/total, allowed p95/total, configured-source ticks, baseline hash, fresh hash, bundle paths, and failed checks. The baseline card shows status, snapshot integrity, approval integrity, baseline id, label, reviewer, Runtime Truth, baseline hot-path p95/total, source report hash, snapshot hash, acceptance hash, action hint, command templates, and failed checks.
- UI metabolism evidence: the Neural Space route now opens through a lightweight telemetry shell and loads the WebGL canvas only after an explicit operator action. The 2026-06-10 `npm run build` output reports `NeuralSpace3D` at `4.44 kB` minified / `1.89 kB` gzip, with the visual slow path isolated as `NeuralSpaceCanvas` (`20.45 kB` / `6.58 kB` gzip), `r3f` (`487.78 kB` / `160.03 kB` gzip), and `three` (`724.38 kB` / `187.35 kB` gzip). The remaining Vite chunk warning is therefore retained as an explicit visual-tooling cost, not routine dashboard startup or runtime evidence-path cost.
- Status sidecar total latency: `92.49 ms`
- Slow-path replay/export/dataset total latency: `86.397 ms`
- Runtime Truth verdict: `alive`
- Runtime Truth recommended action: `continue_monitoring`
- Source configuration evidence: `configured=true`, `source_count=1`, `source_names=["benchmark_local_source"]`
- Feed evidence: `44` tokens processed, `lexical_rolling_segments`, `7` sampled concept observations
- Device evidence: `tensor_device=cpu`, `encoder_device=cpu`, `routing_search_device=cpu`, `cuda_available=false`
- Paired CPU/CUDA comparison on 2026-06-10: report status `failed`; CPU hot-path p95 `574.0 ms`, CPU total `1012.343 ms`; CUDA hot-path p95 `2216.148 ms`, CUDA total `3272.994 ms`; CPU budget passed, CUDA budget failed; endpoint success-name parity passed; CUDA observed execution was `true`. This is a blocker for any CUDA speedup claim.
- Column Runtime evidence benchmark on 2026-06-10: report success `true`, CUDA observed execution `true`, Runtime Truth exposed `total_columns=4`, `awake_budget=4`, `awake_count=4`, and `runs_all_columns=false` for the tiny synthetic checkpoint. The same run showed hot-path p95 `2367.951 ms`, hot-path total `4017.128 ms`, and `/status` latency `30161.935 ms`, so column-metabolism visibility is implemented but hot-path/status cost is not yet acceptable on CUDA.
- Runtime Scope projection-cache probe on 2026-06-11: eight live `/status` reads against the CUDA-selected tiny checkpoint measured `52.017, 97.8, 72.56, 123.053, 100.834, 78.905, 70.459, and 475.196 ms`; median was `97.8 ms`, minimum `52.017 ms`, and observed p95/max `475.196 ms`. Runtime Truth still reported `resolved_device=cuda`, `tensor_device=cuda`, and `observed_cuda_execution=true`. Cached Runtime Scope evidence reported `max_age_ms=500`, its measured age, and source/current token counts. This improves repeated polling and removes duplicate report construction, but the cold/refresh path remains variable and no CUDA speedup is claimed.
- Column report snapshot benchmark on 2026-06-11: synchronized RTX 3060 microbenchmarks reduced median report latency from `10.256 ms` to `1.915 ms` for 4 columns, from `14.246 ms` to `2.459 ms` for 1024 columns, and from `12.619 ms` to `3.495 ms` for 8192 columns. Runtime Truth records one bounded four-vector CUDA-to-CPU snapshot (`64`, `16,384`, and `131,072` bytes respectively), CPU report compute, and a claim boundary that live column execution remains on CUDA.
- Post-change service benchmark at `reports/service_benchmark_column_snapshot/service-benchmark.json`: success `true`, Runtime Truth `alive`, CUDA execution observed, `/status=126.554 ms`, `/terminus=25.08 ms`, status-sidecar p95 `108.951 ms`, and status-sidecar total `206.669 ms`. Hot-path budget still failed: p95 `2070.643 ms` and total `3169.906 ms`. The report optimization is proven locally; broad CUDA runtime acceleration remains unproven.
- Fused reconstruction CUDA graph evidence on 2026-06-13: `reports/fused_reconstruction_20260613/summary-1024.json` compares the current persistent text tick against clean `HEAD` with `PYTHONPATH` pinned to the baseline worktree. On the same RTX 3060 checkpoint and 1024 measured samples per arm, quantum-staged complete encoded ticks improved from `630.168` to `796.219 tokens/sec` (`1.264x`), while per-token-copy arms improved from `538.902` to `658.845 tokens/sec` (`1.223x`). Runtime Truth in the current quantum arm reported `1088` graph replays, `1088` fused reconstruction updates, zero graph failures, zero routing-cache generation mismatches, zero staged-input mismatches, and `cuda:0` tensor execution. This is not a service/source/tokenization claim.
- Zero-blend dead-work benchmark on 2026-06-11: when `input_weight_blend=0.0`, competitive assembly and candidate competition now skip input-weight matrix-vector work whose contribution was exactly zero. Synchronized RTX 3060 A/B assembly measurements fell from median `3.835 ms` to `2.011 ms` for 4 columns (`47.6%`) and from `2.371 ms` to `2.187 ms` for a synthetic 1024-column, 4096-input case (`7.8%`).
- Post-zero-blend service benchmark at `reports/service_benchmark_zero_blend/service-benchmark.json`: success `true`, Runtime Truth `alive`, CUDA execution observed, tick latency `3364.322 ms` versus `3592.686 ms`, hot-path total `2939.723 ms` versus `3169.906 ms`, and total sweep `6654.24 ms` versus `7123.571 ms` in the preceding same-shape run. Hot-path p95 remained over budget at `2039.671 ms`, so sequential binding/context/predictive launch pressure remains the dominant blocker.
- Rejected CUDA scalar-branch experiment on 2026-06-11: replacing zero-state Python branches in column fallback, context prediction, and binding modulation with unconditional tensor computation did not protect the hot path. Two uncontended runs produced hot-path p95/total values of `2134.881/3206.347 ms` and `2320.403/3411.759 ms`, both worse than the zero-blend run's `2039.671/2939.723 ms`. The experiment was rolled back; two additional concurrently launched runs were excluded because they contended for the same GPU.
- Candidate-first competitive scoring benchmark on 2026-06-11: synchronized RTX 3060 routing-plus-competition medians improved from `4.164` to `2.787 ms` for 256 columns, `4.038` to `3.457 ms` for 1024 columns, and `3.364` to `2.994 ms` for 8192 columns, all with `k=10`. The path is exact for learned chunking because the former dense assembly was discarded before routing.
- Post-change tiny service benchmark at `reports/service_benchmark_sparse_candidate/service-benchmark.json`: tick latency `2311.359 ms`, hot-path total `2880.399 ms`, hot-path p95 `2046.558 ms`, and total sweep `5517.871 ms`. Compared with the zero-blend run, tick and total sweep improved substantially, hot total improved slightly, and p95 remained effectively unchanged. Its four-column checkpoint correctly reported an all-column candidate set rather than sparse execution.
- Scale-specific service benchmark at `reports/service_benchmark_sparse_candidate_1024/service-benchmark.json`: Runtime Truth `alive`, CUDA observed, `10/1024` competitive columns scored, hot-path total `1919.574 ms`, hot-path p95 `1157.332 ms`, and total sweep `3956.112 ms`. The total hot-path budget passed while p95 remained above target.
- Rejected reconstruction-distance reuse experiment on 2026-06-11: reusing the exact routing-index distance with an exact overlay for up to eight pending prototype updates preserved the fresh dense distance within `1.2e-7`, but the full CUDA routing pipeline regressed. Across 160 synchronized 1024-column samples, median latency rose from `1.187` to `1.432 ms` (`20.6%` slower) and p95 rose from `3.047` to `3.725 ms` (`22.3%` slower). The experiment was removed; reconstruction retains its fresh dense scan until a fused implementation beats the complete pipeline.
- Hot-path dead-column auto-revival retirement on 2026-06-11: `CompetitiveColumnLayer.process()` no longer scans `steps_since_win` and revives dead columns during every tick. A synchronized RTX 3060 microbenchmark comparing current process against a legacy dead-scan wrapper with no dead columns showed a 1024-column median improvement of `0.355 ms` and an 8192-column median/p95 improvement of `1.203 / 2.561 ms`. The post-change 1024-column service benchmark at `reports/service_benchmark_no_hot_revive/service-benchmark.json` reported Runtime Truth `alive`, CUDA observed, `10/1024` competitive columns scored, tick latency `1406.161 ms`, hot-path total `2107.226 ms`, hot-path p95 `1208.661 ms`, and total sweep `3923.614 ms`. Hot total remains within budget; p95 remains above the 1000 ms target.
- Background tick heartbeat on 2026-06-11: `RuntimeControl` now uses one-token background sub-batches with a short yield and reports `terminus_runtime.execution` (`tick_in_progress`, `tick_phase`, source, target tokens, elapsed time, active requests). This protects status/UI observability while long CUDA ticks run. It does not improve tick throughput: one live 1024-column HF tick reported `73.3 s` for 64 tokens, and later UI-visible counters showed completed ticks with last-tick latency still in multi-second to multi-10-second range.
- Column registry/repeated-growth/recall microbenchmark on 2026-06-11: with `PYTHONPATH=src`, CUDA available, and a 1024-column report including optional prediction-failure streaks, report median/p95 was `5.31565/6.1083 ms` on CPU and `6.1076/6.6982 ms` on CUDA, with `5` snapshot tensors, `20,480` bytes, and `2` CUDA-to-CPU transfers for the CUDA case. The bounded single-column associative recall helper over `64x64` memory and `top_k=4` measured `0.552/0.6715 ms` on CPU and `2.2159/2.6899 ms` on CUDA while returning CUDA tensors for CUDA inputs. This proves the helper's device placement and bounded cost at tiny size; it is not a CUDA speedup claim and remains outside the always-on tick.
- Predictive-owned failure-streak growth evidence on 2026-06-11: with `PYTHONPATH=src`, CUDA available, and a synthetic 12-column model with `k_routing=3`, repeated predictive failures produced `prediction_error_max=0.737856`, `streak_max=6`, `snapshot_tensor_count=5`, `snapshot_bytes=240`, `repeated_surprise_count=11`, and growth gate `ready=true` on both CPU and CUDA. CUDA streak state stayed on `cuda:0` and the report recorded `2` device transfers. The growth gate still reported `mutates_runtime_state=false`, so this is live growth evidence, not structural mutation.
- Bounded column-society export rejection on 2026-06-14: a device-side scheduler export reduced a 1024-column five-tensor status materialization from `20480` bytes to `400`, but warmed CUDA latency was worse at current sizes (`15-20 ms`) than the retained latency-first full report (`6.3695/8.6799 ms` median/p95). MARULHO therefore keeps CUDA column Runtime Truth on one full snapshot for current status reads and records the bounded export as a future large-column option, not the default. This preserves speed over smaller reports and does not claim that cached/sleeping columns skip every execution path.
- Candidate-scoped competitive homeostasis microbenchmark on 2026-06-11: with `PYTHONPATH=src`, CUDA available, 8192 columns, 10 active candidates, and deep-sleep stale counters outside the candidate set, all-column versus scoped `CompetitiveColumnLayer.process()` medians were `1.28555` to `1.08545 ms` on CPU and `5.06735` to `4.93965 ms` on CUDA. P95 moved from `4.3084` to `2.3779 ms` on CPU and `10.6108` to `9.195 ms` on CUDA. Runtime evidence reports `homeostasis_update_count=10` and `homeostasis_update_fraction=0.001221` for the scoped path. This is a process-level improvement only; full endpoint/tick latency remains unproven for this slice.
- Candidate-scoped predictive update microbenchmark on 2026-06-11: with `PYTHONPATH=src`, CUDA available, 8192 columns, 10 active candidates, and synchronized timings around prediction error plus prediction-weight decay, all-column versus scoped medians were `2.91545` to `1.042 ms` on CPU and `4.86425` to `6.9281 ms` on CUDA. P95 moved from `55.6886` to `1.878 ms` on CPU and `11.2076` to `10.0935 ms` on CUDA. Larger checks at 65536 and 262144 columns kept the same pattern: CPU scoped updates were much faster, CUDA scoped updates were launch-bound and not a median speedup. The trainer therefore promoted candidate-scoped predictive updates on CPU first. CUDA sparse PyTorch indexing remains rejected, but the later fused in-place/graph transition supersedes the dense-only CUDA rule when Runtime Truth reports `candidate_predictive_transition_mode=fused_inplace` and active candidate execution.
- Cadenced adaptive-context plasticity benchmark on 2026-06-11: at 1024 columns, one adaptive-context observation measured update/state-only median and p95 of `7.7118/216.0963 ms` versus `1.2798/2.3037 ms` on CPU and `5.9417/14.9707 ms` versus `4.7235/10.8026 ms` on CUDA. Over 256 sparse observations, a four-token plasticity cadence reduced total sequence time from `1784.719` to `1044.086 ms` on CPU (`1.71x`) and `1781.95` to `1205.839 ms` on CUDA (`1.48x`) while final context-state and prediction cosine agreement remained `0.999999`. Weight relative L2 divergence was `0.1565`, so long-run learning parity is not proven. Scaling retained updates by `4x` worsened weight divergence and was rejected.
- Hot-path hypercube hub-refresh retirement benchmark on 2026-06-11: with 1024 columns and synchronized timings, current evidence-only `bind()` measured median/p95 `6.6278/9.0511 ms` on CUDA versus `11.65755/15.2508 ms` for a legacy-equivalent wrapper that called hub-profile refresh after every bind. Both accumulated 35 hub-evidence updates; the current path recorded zero topology refreshes and zero growth/prune events. CPU medians were `1.7777 ms` current versus `1.6213 ms` legacy-equivalent with noisy tails, so no CPU speedup is claimed. The primary result is removal of uncontrolled hot-path structural mutation.
- Live 1024-column curriculum verification after restart reported `resolved_device=cuda`, `observed_cuda_execution=true`, context `state_update_count=111` versus `plasticity_update_count=28`, binding `hub_evidence_update_count=110`, `hub_topology_refresh_count=0`, and zero binding growth/prune events. The latest full background tick still measured about `21.7 s`, so the local mechanism improvements do not establish full-runtime throughput improvement.
- Grounded in-place transition gate on 2026-06-12: the 1024-column synthetic visual/audio checkpoint measured retained versus in-place throughput of `22.9012` versus `27.4966 ticks/sec` (`1.2007x`). Median/p95 latency improved from `38.3108/49.6470 ms` to `30.8081/38.4805 ms`, with zero measured compile events after warmup, exact winner agreement, and bit-exact cross-modal weights/confidences. Peak allocated/reserved VRAM stayed effectively flat at about `11.21/30 MB`. The evidence is synthetic correlated sensory spikes, not camera, microphone, or real-world semantic grounding.
- Triton specialization diagnosis on 2026-06-12: an earlier run compiled twice for `223.93 s` total because the transient one-element winner tensor changed pointer divisibility at measured tick 93. `do_not_specialize_on_alignment=["winners"]` reduced the same workload to one warmup specialization and no mid-run specialization while retaining alignment specialization for persistent state tensors. Production promotion still requires an explicit cold-start/cache lifecycle and Runtime Truth fallback.
- Conditional Binding Wake benchmark on 2026-06-11: `python -m marulho.evaluation.binding_wake_benchmark` compared always-probe interval 1 with inactive-probe interval 4 on the same 1024-column checkpoint and synchronized RTX 3060 inputs. Across 120 samples per arm, median train-step latency fell from `32.2069` to `29.5535 ms` (`8.24%`), p95 from `46.3573` to `42.7967 ms` (`7.68%`), and mean from `32.6817` to `31.0250 ms` (`5.07%`). Both arms retained zero binding usage/state, `20.4585 MB` allocated VRAM, and `48.0 MB` reserved VRAM. Isolated profiling measured 70 CUDA kernels and 136 ATen ops for an idle probe versus zero CUDA kernels and zero ATen ops for a cached skip. A live post-restart tick reported Runtime Truth `alive`, `cuda:0` binding tensors, 3 probes, 11 cached skips, 14 processed tokens, and `2521.655 ms` tick duration. This proves a local metabolism improvement; it does not prove binding learning quality or production-level full-runtime throughput.
- Conditional Cross-Modal Text Wake benchmark on 2026-06-11: `python -m marulho.evaluation.cross_modal_wake_benchmark` compared always-update interval 1 with text-only interval 4 on a live 1024-column CUDA checkpoint. Across 120 text-only samples per arm, median latency fell from `57.59725` to `51.5753 ms` (`10.46%`) and mean from `59.4546` to `54.8716 ms` (`7.71%`); p95 regressed from `82.0887` to `83.8702 ms`, so no tail-latency claim is made. Isolated profiling measured 49 CUDA kernels and 108 ATen ops for one text update versus 2 CUDA kernels and 3 ATen ops for a cached skip. A live `/feed` text-only check reported Runtime Truth `alive`, device `cuda`, cross-modal tensors on `cuda:0`, 2 text updates, 8 cached skips, and final saved revision `960`.
- Conditional Cross-Modal Text Wake interval-16 promotion on 2026-06-12: `python -m marulho.evaluation.cross_modal_wake_benchmark --checkpoint reports/fused_vote_competition_20260612/fused-runtime.pt --output reports/cross_modal_wake_20260612/interval16.json --samples 120 --warmup-steps 20 --text-idle-probe-interval-tokens 16 --seed 20260612` measured text-only interval 16 on RTX 3060/PyTorch `2.11.0+cu128`. Against always-on text updates, median/mean/p95 latency improved `39.44%/35.54%/24.35%`, with text updates reduced from `140` to `9` and cached idle skips raised to `131`. The current interval-4 comparison run (`reports/cross_modal_wake_20260612/interval4.json`) measured conditional median/mean/p95 `21.48485/21.542455/37.3566 ms`, while interval 16 measured `16.91405/18.49991/32.0321 ms`. MARULHO therefore promotes interval 16 as the default text-only cross-modal metabolism cadence. Sensory-backed visual/audio text updates still run every tick; no grounding-quality improvement is claimed.
- Candidate Homeostasis Wake promotion on 2026-06-12: the trainer now separates threshold/win-rate homeostasis wakeup from the structural `dead_column_steps` threshold. `candidate_homeostasis_start_tokens` defaults to `512`; once reached, homeostasis updates only the routed candidate set while stale counters and spike windows still update and dead-column revival remains an explicit maintenance path. A same-code RTX 3060 hot-window A/B used `reports/fused_vote_competition_20260612/fused-runtime.pt`, interval-16 cross-modal wake, seed `20260613`, and the fused in-place transition. Forced all-column homeostasis (`candidate_homeostasis_start_tokens=1000000000`) reached `47.4682 ticks/sec`, median/p95 `18.54175/39.3832 ms`, and updated `1024/1024` columns. The default candidate gate reached `64.8535 ticks/sec`, median/p95 `12.405/29.5659 ms`, updated `10/1024` columns, and reported `candidate_homeostasis_start_tokens=512`. This is a hot-path metabolism win, not a pruning/growth claim.
- Predictive Vote Cache Wake promotion on 2026-06-15: `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 2048 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-2048.json` compared the old retained all-column predictive vote scan with the new routed awake-mask cached-vote path. The exact winner sequence was preserved. All-column vote updated `2048/2048` predictive votes and measured median/p95/mean `4.3396/7.6456/4.67866375 ms` (`213.736 tokens/sec`). Scoped cached vote updated `10/2048`, cached `2038`, reported `runs_all_columns=false`, and measured `3.8765/6.7643/4.34479625 ms` (`230.160 tokens/sec`). Mean complete `train_step` improved `7.14%`; the claim is CPU complete-step metabolism for retained predictive voting, not CUDA acceleration, endpoint throughput, sleep mutation, growth, or pruning.
- Predictive Update/Vote Wake promotion on 2026-06-15: `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 2048 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-2048-predictive-update-vote.json` compared all-column predictive update plus all-column vote against the scheduler-owned candidate-scoped update plus cached-vote path. The exact winner sequence was preserved. The retained all-column arm updated predictive state `2048/2048`, updated votes `2048/2048`, reported `runs_all_columns=true`, and measured median/p95/mean `3.8277/6.0854/4.0616075 ms` (`246.208 tokens/sec`). The scoped arm updated `10/2048`, cached `2038` prediction states and votes, reported `runs_all_columns=false`, and measured `3.5018/6.032/3.80747875 ms` (`262.641 tokens/sec`). Mean complete `train_step` improved `6.26%`; the claim is CPU retained-path scheduler metabolism, not CUDA acceleration, sleep/deep-sleep mutation, growth, or pruning.
- Candidate Deep-Sleep Filter and Predictive Location Cache promotion on 2026-06-15: the retained CPU route now asks for a bounded backfill pool, sorts by retrieval distance, filters only deep-sleep candidates, and passes the same awake mask into predictive location, predictive update, predictive vote, competition, and homeostasis. The final 8192-column gate `python -m marulho.evaluation.column_scheduler_benchmark --n-columns 8192 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-8192-deep-sleep-filter-location-update-vote.json` preserved exact winner sequence. The all-column arm updated predictive state/location/votes `8192/8192`, reported `runs_all_columns=true`, and measured median/p95/mean `5.41375/12.7247/6.06891375 ms` (`164.774 tokens/sec`). The scoped arm updated predictive state, predictive location, votes, and sleep-filter output at `10/8192`, cached `8182`, reported `runs_all_columns=false`, and measured `4.28095/12.8977/5.084465 ms` (`196.678 tokens/sec`). Mean complete `train_step` improved `16.22%`. The scaling sweep `python -m marulho.evaluation.column_scheduler_benchmark --sweep-columns 2048 8192 16384 --column-latent-dim 64 --k-routing 10 --samples 80 --warmup-steps 10 --seed 20260615 --device cpu --output reports/column_scheduler_20260615/cpu-scaling-large-deep-sleep-filter-location-update-vote-final.json` kept all four specialist counts bounded at `10` and neutral-or-better at all sizes, with exact winner parity at `2048` and `8192`; `16384` still diverged, so durable scaling correctness remains open.
- Hot-window throughput baseline on 2026-06-11: `python -m marulho.evaluation.hot_window_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/hot_window_cuda_1024/hot-window-benchmark.json --samples 256 --warmup-steps 32` measured already-encoded CUDA tensor throughput with service, source loading, tokenization, UI/status reads, checkpointing, replay, and sleep maintenance excluded. On RTX 3060 with torch `2.11.0+cu128`, the current eager core loop reached `24.6404 tokens/sec`, median step `35.18625 ms`, p95 `76.0758 ms`, mean `40.5594 ms`, allocated VRAM `20.6758 MB`, and reserved VRAM `50.0 MB`. The 1000 tokens/sec target was not met; the report estimates a `40.5838x` improvement is needed before endpoint overhead.
- Historical compiled column-kernel benchmark on 2026-06-11 measured fixed-shape, candidate-scoped competition only. After installing `triton-windows`, the runner auto-selected the bundled TinyCC compiler at `triton/runtime/tcc/tcc.exe`. On RTX 3060, eager batched execution reached `79651.68 isolated tokens/sec`; `torch_compile_default` reached `178071.38 isolated tokens/sec` with median/p95 `1.1134/2.7291 ms`; `torch_compile_reduce-overhead` reached `212698.98 isolated tokens/sec` with median/p95 `1.01745/2.0355 ms`; compile errors were absent. This proved the isolated awake-column competition kernel could exceed a low 1000 tokens/sec reference floor, but the runner is removed because wider hot-path and production route/vote/transition evidence superseded the competition-only probe.
- Compiled hot-path-kernel benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high` measured input projection, candidate-scoped competition, and candidate-local predictive-state math without runtime writeback. On RTX 3060, eager batched execution reached `45118.41 isolated tokens/sec`; `torch_compile_default` reached `242784.01 isolated tokens/sec` with median/p95 `0.7662/2.677 ms`; `torch_compile_reduce-overhead` reached `206492.85 isolated tokens/sec` with median/p95 `0.91425/2.8345 ms`; compile errors were absent. A default-precision comparison reached `203483.21 isolated tokens/sec`, so precision policy remains measured evidence rather than a static assumption. This benchmark still excludes retrieval, Python trainer orchestration, in-place plasticity, memory/replay, binding, cross-modal grounding, checkpointing, and service throughput.
- Real-candidate compiled hot-path benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-routing-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high --candidate-source routing_index` used live sharded `torch_topk` routing candidates. Candidate preparation through the legacy list-returning API measured `354.905 ms` for 256 tokens (`721.32 tokens/sec`) with zero fallback rows, while the compiled block reached `191913.60 isolated tokens/sec`. This exposed Python-list/CPU-numpy candidate extraction as a bottleneck and is not a production routing ceiling.
- Tensor-routing compiled hot-path benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-routing-tensor-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high --candidate-source routing_index_tensor` used retrieval-owned `search_tensors()` to keep candidate ids and distances on CUDA. Cold candidate prep after checkpoint/cache setup measured `212.7566 ms` for 256 tokens (`1203.25 tokens/sec`); warm cached prep measured `4.7468 ms` (`53931.07 tokens/sec`), both with zero fallback rows and sharded torch caches on `cuda:0`. The compiled block reached `234286.58 isolated tokens/sec` with `torch_compile_reduce-overhead`, median/p95 `0.959/2.0064 ms`, compile errors absent, and CUDA memory `13.266/56.0 MB` allocated/reserved. This is still an isolated benchmark: retrieval prep, kernel math, runtime mutation, trainer orchestration, service endpoints, and checkpointing remain separate evidence surfaces. The active benchmark name is now `candidate_source=routing_index`; the old `routing_index_tensor` spelling is historical only because tensor routing is no longer one option among two.
- Live tensor-routing hot-window A/B on 2026-06-11: the benchmark temporarily gained `--routing-candidate-mode list|tensor` as an evaluation-only switch, while the trainer default used retrieval-owned tensor candidates. Two sequential, reversed-order 256-sample pairs on the same revision-960 checkpoint and seed measured tensor/list throughput of `24.7167/23.9371` and `32.6721/32.3280 tokens/sec`. Tensor routing improved median latency in both pairs (`38.7952` versus `40.18805 ms`, then `26.68645` versus `27.5149 ms`); p95 was mixed (`62.7875` versus `60.0293 ms`, then `52.8824` versus `53.0974 ms`). Runtime counters proved `288` tensor searches and zero list searches in tensor arms, with the inverse in list arms; allocated/reserved VRAM remained `20.6758/50.0 MB`. The promotion was a modest roughly `2%` aggregate throughput improvement and did not close the gap between eager `train_step` and compiled-kernel capacity. On 2026-06-16, the list mode and list-returning routing API were removed so hot-window benchmarks exercise only the tensor-native route surface.
- Merged sharded-torch routing benchmark on 2026-06-11: `compiled_hot_path_kernel_benchmark --candidate-source routing_index_tensor` compared the default merged exact cache against the now-retired `--disable-merged-torch-shards` switch. Warm 256-token candidate preparation improved from `6.7612 ms` (`37863.10 tokens/sec`) to `2.0147 ms` (`127066.06 tokens/sec`), a `3.36x` routing improvement with zero fallback rows. The merged cache held all 1024 normalized vectors and ids on `cuda:0`, adding `532480` bytes. Two reversed-order hot-window pairs measured merged/per-shard throughput of `38.5582/37.0260` and `37.3134/36.8980 tokens/sec`; median latency improved from `26.363` to `25.01605 ms` and from `26.0075` to `25.2821 ms`. P95 was mixed, so no tail-latency claim is made. Shards still own add/remove/rebuild; every mutation invalidates the merged cache. On 2026-06-16, the disable switch and config field were removed because the promoted CUDA graph route/vote path requires the merged `routing_tensor_cache()`; benchmarks now report `routing_cache_boundary=merged_torch_route_cache_required`.
- Historical compiled dense predictive transition benchmark on 2026-06-11: `python -m marulho.evaluation.predictive_transition_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/predictive_transition_cuda_1024/predictive-transition-benchmark.json --iterations 512 --warmup-iterations 16` measured one 1024-column fixed-shape predictive state transition. Eager reached `204.35 transitions/sec`; compile default reached `497.90/sec`; compile reduce-overhead reached `1121.66/sec`, median/p95 `0.72745/1.8407 ms`, with no compile errors. This is now historical evidence only: the benchmark module and configurable `compiled` runtime mode were removed after the fused in-place/graph transition became the maintained scheduler boundary.
- Historical live compiled-predictive hot-window A/B on 2026-06-11: with merged tensor routing held constant, two reversed-order 256-token pairs measured compiled/legacy throughput of `47.8081/34.5995` and `42.0725/30.1816 tokens/sec`. Compiled median latency was `19.73835/22.0747 ms` versus legacy `27.00975/31.58265 ms`; compiled p95 was `30.3/34.9 ms` versus legacy `44.6731/49.9381 ms`. First-use warmup increased from about `1.3-1.6 s` to `8.7 s`, and reserved VRAM increased from `50` to `54 MB`. The active path now defaults and migrates checkpoints to `inplace_triton`; if in-place cannot start, dense eager fallback reports the concrete Runtime Truth fallback reason rather than reviving the compiled branch.
- Dormant lightweight input-plasticity retirement on 2026-06-12: the revision-960 checkpoint has `input_weight_blend=0.0`, so winner input-weight rows could not contribute to competition or assembly output. A direct same-seed pre-change/skip probe measured `40.3786` versus `43.9311 tokens/sec`, median `23.81175` versus `21.03955 ms`, and p95 `36.6842` versus `34.5157 ms`; a reversed repetition measured `41.6699` versus `45.6825 tokens/sec`, median `22.6771` versus `20.83225 ms`, and p95 `33.6637` versus `32.6971 ms`. Two uncontended post-change 256-token confirmations measured `42.8296` and `41.9188 tokens/sec`, median `21.9380` and `22.52325 ms`, p95 `36.1204` and `38.4941 ms`, with `21.1924 MB` allocated and `54.0 MB` reserved VRAM. Runtime evidence reported `input_plasticity_mode=skipped_zero_blend`, zero updates, 296 skips, CUDA tensor execution, and sparse `10/1024` competition. Local STDP is intentionally excluded from this skip.
- Rejected functional steady-state transition on 2026-06-12: a pure fixed-shape transition combined competition, winner selection, dense prediction, prediction-error modulation, prototype/velocity plasticity, stale counters, and homeostasis with exact eager-module parity. In isolation, including eleven stable state-buffer copies, eager reached `64.9176 transitions/sec` with median/p95 `14.3822/22.8777 ms`; `torch.compile(mode="reduce-overhead")` reached `271.3224/sec` with median/p95 `2.9067/7.7558 ms`, a `4.18x` isolated gain. Full configured hot-window evidence rejected promotion. Pair A measured compiled/retained throughput `37.5324/36.6468 tokens/sec`; reversed Pair B measured `31.9652/36.6624`. Compiled average throughput was about `34.75` versus `36.65 tokens/sec`, latency results were mixed, and first-use warmup remained about `14 s`. Earlier runs while another CUDA backend held the GPU were excluded as contaminated. The functional transition was later deleted from active `core`/`evaluation` after the in-place/graph transition became the maintained runtime path; CUDA parity tests now compare against retained module semantics instead of importing a rejected full-state implementation.
- Historical evaluation-oracle rerun at `reports/steady_state_column_transition_cuda_1024/steady-state-column-transition-benchmark.json`: on the idle RTX 3060, eager reached `55.6270 transitions/sec` with median/p95 `17.39165/24.5979 ms`; compiled reduce-overhead reached `414.0303/sec` with median/p95 `1.7943/6.9034 ms`, no compile errors, and `4.6431/14.0 MB` allocated/reserved VRAM. This remains historical evidence only; the standalone runner is removed.
- Historical in-place steady-state column kernel benchmark on 2026-06-12 measured candidate competition followed by one Triton launch mutating predictive state, winner prototype/velocity plasticity, candidate-scoped homeostasis, stale counters, spike history, and assembly. Functional eager stable-writeback reached `59.8240 transitions/sec`, median/p95 `16.3240/23.0268 ms`. Eager competition plus in-place Triton reached `314.5419/sec`, median/p95 `2.52885/6.8216 ms`, a `5.2578x` cluster speedup. Runtime state remained finite, all eleven state tensor addresses remained stable, and CUDA memory was `12.7666/28.0 MB` allocated/reserved. Warmup/compilation cost `10.3777 s`. The isolated comparison runner is removed because complete-runtime gates and `ColumnTransitionRuntime` now own promotion evidence.
- Complete hot-window in-place A/B on 2026-06-12: after replacing the Python spike-ring cursor specialization with one persistent CUDA scalar, same-checkpoint 128-token Pair A measured in-place/runtime throughput `55.2774/38.0221 ticks/sec`, median `17.4702/25.8010 ms`, and p95 `24.2847/34.1031 ms`. Reversed Pair B measured `105.3975/37.9536`, median `8.6513/25.8147 ms`, and p95 `15.0841/34.2618 ms`. In-place VRAM was `11.7241/28.0 MB` allocated/reserved versus runtime `11.7183/30.0 MB`. The disk-cached in-place warmups were `1.85` and `0.65 s`; a cold diagnostic compiled two expected variants at about `98` and `73 s`, so cold startup remains unacceptable.
- Production lifecycle promotion on 2026-06-12: `ColumnTransitionRuntime` moved the in-place executor into trainer ownership with checkpoint opt-in, persistent workspace, compile-only warmup for all-column and `k=10` shapes, pre-mutation fallback, post-launch fail-closed behavior, and Runtime Truth counters. Empty-cache compile-only startup took `80.746 s`; populated disk-cache startup took `0.348 s`. A CUDA lifecycle test proves warmup leaves every model tensor bit-exact before the first execution.
- Production-backed complete hot-window evidence on 2026-06-12: in-place runs reached `80.5912` and `110.3597 ticks/sec`, with median/p95 `12.299/17.81 ms` and `8.50705/14.8878 ms`. Retained compiled observations reached `70.7727` and `51.5562 ticks/sec`, with median/p95 `13.5353/20.7919 ms` and `18.74655/27.9535 ms`. The strongest measured complete encoded hot window is `110.36 ticks/sec`, leaving a `9.06x` gap to the low `1000 ticks/sec` reference floor.
- Production-backed grounded quality evidence on 2026-06-12: the synthetic visual/audio gate passed at `42.8081` versus `38.7751 ticks/sec` (`1.104x`) with exact winners, bit-exact cross-modal weights/confidences, and zero measured Triton compile events. P95 was noisy in this run (`36.1636` versus `33.3056 ms`), so no tail-latency claim is made.
- Live service execution evidence on 2026-06-12: one real source tick processed 12 tokens in `3481.4487 ms`, or `3.44684 tokens/sec`. A clean post-restart repetition processed 12 tokens in `8646.97 ms`, or `1.39 tokens/sec`. Both runs reported requested/resolved `inplace_triton`, CUDA observation, 12 executions, zero failures, successful warmup, and revision progress. This proves production execution but also locates the next bottleneck and substantial variance outside the transition kernel in source/tick orchestration and remaining per-token stages.
- Live stage-profile optimization on 2026-06-12: an instrumented 12-token tick measured `7912.83 ms` in training and only `255.17 ms` in source collection. Splitting training attributed `5490.65 ms` to service-side concept observation versus `1488.05 ms` to `trainer_step`. Direct CPU profiling of three restored 94-concept observations measured `850.23 ms/call`; normalized-centroid caching reduced the warm result to `117.80 ms/call`, about `7.2x`. Sampling background observation at tokens 1, 8, and the final pending token raised the first cold live result to `4.338 tokens/sec`; same-process ticks reached `7.841`, `8.075`, and `8.438 tokens/sec`.
- Remote-source overlap evidence on 2026-06-12: scheduling the existing refill worker immediately after a consumed chunk lets provider I/O overlap CUDA training. Once warm, a configured 64-token tick collected its source window in `0.04 ms`, processed at `7.965 tokens/sec`, left 76 buffered tokens, and recorded a queue hit. One shorter 18-token tick reached `9.880 tokens/sec`. Source orchestration is no longer the dominant warm-path cost; the next gate is a synchronized stage profile inside `MarulhoTrainer.train_step`, whose 64-token contribution was `6601.45 ms`.
- Trainer profiler evidence on 2026-06-12: an explicit 12-tick PyTorch profiler slow path observed `1338` CUDA launches, `475` async copies, and `186` stream synchronizations. Reported CUDA operator work was about `1.2 ms/tick` while CPU orchestration averaged about `42 ms/tick`, so the configured path is launch/synchronization bound rather than arithmetic bound. A separate cProfile run with the promoted in-place executor active measured about `21 ms/tick`; the in-place transition was about `2.2 ms/tick`, while candidate competition, predictive voting, routing-key projection, and tensor routing-index lookup together consumed roughly `11 ms/tick`. The next implementation gate is candidate-scoped voting and a broader device-resident routing/competition cluster, followed by repeated full-`train_step` quality and throughput evidence.
- Device-resident winner-selection promotion on 2026-06-12: the active in-place transition now precompiles a one-block Triton selector that writes winner, unit strength, and positive-activation evidence into persistent CUDA buffers. The transition kernel consumes the boolean and preserves all-silent fallback threshold decay without the previous Python `values.max()` branch. Two reversed 256-tick complete hot-window pairs measured device/retained selection at `56.5076/58.7954` and `68.1373/59.9590 ticks/sec`; median latency was `16.1034/15.8209` and `13.4684/15.5395 ms`, while p95 was `29.5302/26.3276` and `24.2590/27.5564 ms`. Average throughput improved about `4.9%`; average p95 was effectively neutral, so no universal tail claim is made. The grounded 128-tick gate passed with exact winners, exact cross-modal tensors, finite state, zero measured Triton compilation events, and `35.9594` versus `25.4370 ticks/sec` against the retained transition baseline. Runtime Truth recorded 288 selector and transition executions with zero failures. One empty-process Windows compile-only warmup took `111.423 s`; populated-cache warmup was `0.687 s`, making cache packaging or prewarming a production startup requirement.
- Fused predictive vote/competition promotion on 2026-06-12: for the checkpoint-proven learned-chunk, zero-blend, one-winner shape without context, abstraction, or binding gain, `ColumnTransitionRuntime` now replaces the dense 1024-column predictive vote plus PyTorch candidate scoring with one candidate-local Triton launch. The kernel keeps the previous winner on-device and fuses reference-frame agreement, ten-candidate prototype score, threshold inhibition, positive/silent fallback, winner output, and previous-winner writeback. A 96-tick recurrent clone comparison matched `96/96` winners and was bit-exact for prototypes, thresholds, predictive locations, prediction error, and confidence. Reversed 256-tick complete hot-window pairs measured fused/unfused `84.1064/66.4301` and `141.2969/65.7654 ticks/sec`; median latency improved `10.4984/13.8884` and `6.0542/14.1862 ms`, while p95 improved `21.2732/23.4537` and `13.6325/24.2594 ms`. Both arms reserved `26 MB`; fused allocated only about `0.0005 MB` more. A 12-tick profiler reduced launches from `902` to `670`, async copies `155→149`, stream synchronizations `117→109`, CPU self-time `236.2→154.0 ms`, and CUDA self-time `49.9→7.8 ms`. The grounded 128-tick gate passed at `35.8357` versus `31.6136 ticks/sec`, exact winners, exact cross-modal tensors, finite state, and zero measured compilation events. Empty-process compile-only warmup was about `110.526 s`; cached warmup ranged `0.258–0.964 s`.
- Predictive-vote experiment and dead-state retirement on 2026-06-12: candidate-scoped voting measured `47.2868 ticks/sec`, median/p95 `19.7735/31.4794 ms`, versus dense voting at `71.7411 ticks/sec`, `13.0741/21.874 ms`. A separately compiled dense vote reached `75.8192 ticks/sec` versus an uncontended dense repetition at `109.8457 ticks/sec`, while adding `9.77 s` benchmark warmup. Both variants were rejected. Removing the unconsumed checkpointed `hypothesis` tensor reduced a 12-tick profiler window from `1338` to `1319` CUDA launches and `475` to `462` async copies, with stream synchronizations unchanged at `186`. Two post-retirement runs reached `117.9459` and `113.0185 ticks/sec`; the matched pre-retirement run was `109.8457`, but median results were mixed, so the durable claim is dead-state/launch removal plus p95 improvement from `13.9687` to `12.9947 ms`, not a universal throughput percentage.
- Routing normalization rejection on 2026-06-12: passing an already-normalized key through projection, tensor retrieval, and competition removed `109` launches over a 12-tick profiler window, but a 64-step checkpoint-clone comparison matched only `30` winners and produced maximum prototype/location/prediction-error differences of `0.01254/0.54929/0.00381`. Complete throughput was within or below environmental variance. The change was reverted; future fusion must preserve the full numerical trajectory or pass an explicit cognitive-quality gate.
- routing-index winner-ID reuse on 2026-06-12: the in-place transition already materializes the winner ID on CPU, so the trainer now passes that list into the 16-token routing-index update buffer instead of copying the same winner tensor to CPU again. A 12-tick profiler comparison reduced async copies from `463` to `450` and stream synchronizations from `186` to `174`, exactly one avoided transfer/sync per tick. Launch count was noisy (`1319` versus `1327`) and desktop GPU load contaminated throughput runs (`52-54 ticks/sec`), so no throughput percentage is claimed. The buffered IDs, vectors, deduplication, flush cadence, and index mutation remain unchanged.
- Exact route/competition fusion gate on 2026-06-12: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/runtime.pt --output reports/compiled_hot_path_kernel_cuda_1024/exact-route-compete-runtime-single.json --batch-size 1 --iterations 512 --warmup-iterations 16 --matmul-precision high --candidate-source routing_index_tensor --exact-route-compete --route-compete-last-winner 0` ran on RTX 3060 with PyTorch `2.11.0+cu128`. Single-tick learned-chunk projection + tensor routing + candidate competition had exact candidate, candidate-set, winner, and strength parity and reached `978.34 ticks/sec` with `torch_compile_reduce-overhead` (median/p95 `0.8664/2.257 ms`). Starting after production `routing_key_from_pattern`, fused tensor routing + competition also had exact parity and reached `1252.86 ticks/sec` (median/p95 `0.67535/1.7508 ms`). A 256-token batch was rejected as a promotion proxy because tiny projection differences and sequential threshold state produced candidate-set and winner divergence. Next step: a production-owned single-tick route/competition executor with Runtime Truth, followed by full `train_step` A/B.
- Fused Text Route Vote promotion on 2026-06-12: a custom two-launch Triton probe over the exact 1024x64 torch routing cache and `k=10` candidates matched candidate order/set, winner, and positive/silent branch on `128/128` recurrent keys. Isolated production tensor routing plus fused vote reached `415.16 ticks/sec`; the probe reached `1716.56` (`4.135x`, median `2.1331` to `0.4881 ms`). After moving ownership into retrieval/core/training and adding checkpoint mode `fused_triton_text`, reversed 256-tick complete text/idle runs averaged `92.00` versus `66.80 ticks/sec` (`1.377x`). Runtime Truth reported `288` executions per fused arm, zero failures, and `17` cache refreshes. Cold route warmup was `2.577 s`; cached warmup was `0.206 ms`.
- Fused Text Route Vote sensory boundary on 2026-06-12: the global sensory variant was rejected after a reversed grounded result of `0.919x` despite exact cognition. The promoted lifecycle falls back before routing on visual/audio ticks. A later reversed 128-tick grounded gate reported zero fused route executions, `272/272` sensory fallbacks, exact winners and cross-modal tensors, finite state, zero measured compile events, and mean throughput `37.06` versus `32.56 ticks/sec` in that run. The sensory gate proves preserved fallback behavior, not fused sensory acceleration.
- Serialized checkpoint evidence on 2026-06-12: `reports/fused_route_vote_20260612/fused-text-runtime.pt` reloaded without benchmark injection and ran `144/144` fused text/idle ticks with zero failures, nine cache refreshes, CUDA tensors on `cuda:0`, and `82.40 ticks/sec` (median/p95 `10.7104/21.8857 ms`). Full encoded-tick throughput remains `12.14x` below the 1000 ticks/sec reference floor.
- Complete route/competition runtime rejection on 2026-06-12: a temporary trainer-owned compiled executor was tested with the promoted in-place transition over complete encoded `train_step`. Pair A measured retained/compiled `82.0011/71.0966 ticks/sec` with median/p95 `12.0586/14.5078 ms` versus `13.8128/19.7336 ms`. Reversed Pair B measured compiled/retained `88.1523/90.3215 ticks/sec` with median/p95 `11.1096/15.7194 ms` versus `10.6433/15.8042 ms`. Compiled reserved `50 MB` versus retained `28 MB` and added compile warmup. The runtime/config/status prototype was removed; only the evaluation gate remains. This rejects another standalone compiled launch and points toward a broader projection/vote/route/competition cluster or persistent host-orchestration removal.
- Archival reservoir rejection optimization on 2026-06-12: the CUDA memory-store rejection branch now draws the exact reservoir candidate before copying optional input-pattern and routing-key tensors to CPU. A forced-rejection RTX 3060 comparison over `2048` measured updates reached `706.91` versus `575.48 updates/sec` (`1.228x`), with median `0.8988` versus `1.3530 ms`; p95 was `2.8269` versus `3.2865 ms`. Two complete steady-state trainer pairs also favored deferred copies: `41.36` versus `35.98 ticks/sec` and `48.51` versus `46.98 ticks/sec`, with lower median latency in both but mixed p95. The claim is limited to rejected archival payload copies. Assembly CPU transfer, EMA/drift observation, and O(memory-size) STC decay remain live bottlenecks.
- Zero-copy STC numeric-buffer promotion on 2026-06-12: capture tags, local PRP, and strong-tag flags now remain in contiguous numeric buffers and decay through NumPy zero-copy views. At production memory capacity `1000`, two controlled retained-buffer/list-reference pairs measured `80.3014/64.4427` and `67.3642/43.9913 ticks/sec`; median latency was `11.9151/14.8021` and `14.1093/21.8126 ms`, with p95 `17.3493/22.8803` and `20.5032/38.1632 ms`. The average throughput improvement across those pairs was about `37.5%`. Capacity-64 pairs were one clear win and one neutral result, averaging about `14.1%` better for buffers. A post-change cProfile window attributed only `0.051 s` to `579` `_advance_state` calls; remaining costs were broader trainer orchestration, transition execution, memory update/ripple work, predictive vote, and routing search.
- Trainer telemetry cadence promotion on 2026-06-12: scalar metric extraction inside `train_step` now refreshes on `trainer_telemetry_interval_tokens` instead of the previous hardcoded ten-token cadence, while cached metrics fill intervening ticks. On the 1024-column RTX 3060 checkpoint, interval-64 versus interval-10 complete hot-window pairs measured `108.6422/95.1195` and `98.1217/90.2412 ticks/sec`; median latency improved from `10.27905` to `9.1036 ms` and from `10.89835` to `10.28885 ms`, and p95 improved from `14.6463` to `12.3683 ms` and from `15.6469` to `14.4946 ms`. Metrics now expose the configured interval and whether the tick refreshed telemetry. This is a display/Runtime Truth metabolism improvement, not skipped cognition.
- Stream text episode-term cache promotion on 2026-06-12: `_update_stream_text()` now caches the token set for the current cached episode text instead of rebuilding it every tick while checking refresh pressure. A direct 2000-window text-path microbenchmark improved median time from `152.1478 ms` to `125.131 ms`. Complete 1024-column CUDA hot-window A/B also favored the cached path in both orders: `99.0435` versus `55.1780 ticks/sec`, then `63.1403` versus `60.2135 ticks/sec`, with lower median and p95 latency in both pairs. This is a source-text/memory-context overhead reduction; it does not change raw archival windows, learned chunking, routing, prediction, replay, or Subcortex state.
- Rejected previous-routing-key buffer reuse on 2026-06-12: a persistent copied previous-key tensor preserved semantics but did not win repeated complete hot-window A/B. Buffer/legacy measured `61.0333/59.3997 ticks/sec` in one order but `71.7686/76.4724` in reversed order, with p95 worse in both buffer arms (`24.2077/28.47 ms`, then `24.5848/19.6343 ms`). The experiment was removed.
- Consolidation quality and checkpoint gate on 2026-06-12: `memory_consolidation_hf_smoke` passed after `384` replay updates. Task-A reconstruction moved from `0.085321` after A to `0.066652` after B and `0.048156` after consolidation; Task-B ended at `0.051666`, Task-A overlap was `0.969471`, and the gate passed. Reloading the resulting 256-entry checkpoint rebuilt all three numeric buffers, executed another real training tick, kept archival tensors on CPU, and reported `4352` STC scalar-state bytes.
- Rejected memory GPU variants on 2026-06-12: bulk staging was neutral in complete paired runs (`69.14/72.76` and `83.91/83.79 ticks/sec` retained/staged), while a CUDA-resident observation path reached `75.69` versus `97.18 ticks/sec` for retained CPU observation. Neither path was promoted. The production boundary keeps archival metadata on CPU and moves sampled replay tensors to CUDA only for active replay computation.
- Adaptive awake-ripple scan on 2026-06-12: `reports/adaptive_ripple_scan_20260612/ripple-scan-crossover.json` measured the retained scalar path for small ledgers and zero-copy NumPy vector mode for large ledgers. Capacity-64 stayed scalar-small with median `0.04735 ms`; capacity-256 stayed scalar-small with median `0.1081 ms`; capacity-1000 switched to vector-large and improved median from `1.247` to `0.15545 ms` and mean from `1.487` to `0.2993 ms`. Focused tests passed `225` cases. A later service run at `reports/adaptive_ripple_scan_20260612/service-benchmark-with-runtime-truth-memory-hot-path.json` succeeded, reported CUDA observed execution, `24` fused in-place transition executions with zero failures, trace-derived living-loop throughput `164.3319 tokens/sec`, and Runtime Truth `memory_hot_path` counters. That source tick did not exercise ripple tagging (`last_ripple_scan_mode=not_run`), so the live-service evidence proves projection visibility and no fallback, not a full-runtime ripple speedup.
- File-source ingestion cache on 2026-06-12: `reports/file_source_cache_20260612/cold.json` versus `warm.json` used the same 1024-column fused CUDA checkpoint and same local source path. The cold run spent `1385.2147 ms` in `collect_source_queue`, took `2459.6596 ms` for the tick, and reported `9.7574 tokens/sec`. The warm run restored the file-source cache, spent `0.0293 ms` in `collect_source_queue`, took `784.4856 ms` for the same 24-token tick, and reported `30.5933 tokens/sec`. Both runs reported CUDA observed execution, `24` fused in-place transition executions, zero transition failures, and `cuda:0` transition tensors. This proves source-ingestion orchestration improved; it is not a neural-kernel speedup and does not remove the remaining `train_compute` bottleneck.
- Runtime-cache unchanged-material skip on 2026-06-13: after sampled batched ConceptStore observation, the warmed 24-token CUDA service tick still spent `388.0982 ms` in `prepare_training` because Runtime Sources rewrote identical restored source-cache material during tick preparation. Runtime Sources now records a material hash for brain-source cache payloads, restores it with the ready queue, skips identical rewrites, and exposes `source_cache` counters in the configured-source benchmark summary. `reports/runtime_cache_skip_20260613/service-benchmark-final2.json` measured `461.6319 ms` for the same 24 source tokens, `51.9895 tokens/sec`, `0.0450 ms` source collection, `4.0725 ms` preparation, `422.6478 ms` trainer step, `31.6244 ms` concept observation, and `source_cache.cache_skip_count=1` with `last_cache_update_mode=skipped_unchanged_material`. Runtime Truth still reported CUDA observed execution, `24` graph replays, and zero graph failures. This is an ingestion/persistence hot-path cleanup, not a neural-kernel speedup; the remaining wall is host-orchestrated trainer execution.
- Cognitive-trajectory fidelity blocked promotion: two trainers loaded from the same checkpoint and processed the same 32 CUDA input tensors. Winner agreement was `46.875%`; maximum prototype and prediction-error differences were `0.02531` and `0.00314`, while final location and threshold maximum differences reached `0.59042` and `0.26067` after routing diverged. The default trainer remains unchanged. The next gate must compare predictive loss, grounding, spike health, memory usefulness, and trajectory stability over a bounded corpus, then either tighten kernel numerics or prove the faster trajectory is not cognitively worse.
- Bounded cognitive-quality A/B on 2026-06-12: `PYTHONPATH=src python -m marulho.evaluation.inplace_transition_quality_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/runtime.pt --output reports/inplace_transition_quality_1024/inplace-transition-quality-benchmark.json --samples 128 --warmup-steps 8 --seed 20260620` compared sequential clones over identical encoded tensors. Runtime/in-place throughput was `46.5678/68.0983 ticks/sec` (`1.4623x`); median latency was `20.8646/14.4042 ms`, p95 `27.6272/18.3551 ms`, and peak allocated/reserved VRAM `12.5972/30.0` versus `12.6040/34.0 MB`. The in-place arm lowered mean reconstruction error from `0.0049263` to `0.0038990`, slightly lowered mean prediction error, preserved confidence, increased normalized winner entropy from `0.4345` to `0.4816`, stayed finite and `sparse_responsive`, and passed every declared spike/memory/diversity quality gate. Winner agreement was only `31.25%`, now treated as diagnostic because aggregate quality held. Promotion remains blocked: the synthetic checkpoint had cross-modal grounding disabled and the cold Triton compile/cache lifecycle is not production-ready.
- Fused predictive-vote/competition promotion evidence on 2026-06-12: two reversed-order 256-tick RTX 3060 hot-window pairs averaged `112.7017` versus `66.0978 ticks/sec` for fused versus unfused device selection (`1.7051x`), with median `8.2763` versus `14.0373 ms` and p95 `17.45285` versus `23.85655 ms`. A 96-tick recurrent clone comparison preserved every winner and was bit-exact for prototypes, thresholds, predictive location, prediction error, and confidence. The grounded 128-tick gate passed with exact winners and cross-modal state, `35.8357` versus `31.6136 ticks/sec` (`1.13355x`), and zero measured compilation events. A 12-tick profiler reduced CUDA launches from `902` to `670`, CPU self time from `236.2` to `154.048 ms`, and CUDA self time from `49.907` to `7.769 ms`. Empty-process compilation remained about `110.5 s`; populated-cache warmups returned below one second.
- Live fused-service evidence on 2026-06-12: `python -m marulho.evaluation.service_benchmark --checkpoint reports/fused_vote_competition_20260612/fused-runtime.pt --output reports/fused_vote_competition_20260612/service-benchmark-fused.json --trace-dir reports/fused_vote_competition_20260612/traces-fused --env-root reports/fused_vote_competition_20260612/service-env-fused --web-dist-dir MARULHO_UI/dist --configure-local-source --local-source-tick-steps 1` processed one real local-source tick containing `24` tokens. Runtime Truth reported requested/resolved `inplace_triton`, `24` transition and fused-vote executions, zero failures or fallbacks, `cuda:0` tensors, observed CUDA execution, and candidate scoring mode `candidate_subset_fused_vote_competition` over `10/1024` columns. The source tick took `968.183 ms`; the trace-derived living-loop rate was `556.2678 tokens/sec`. This proves the promoted executor is active in the service but does not meet the maximum-throughput objective: all-column homeostasis remained `1024/1024`, and source/service orchestration plus remaining per-token stages still dominate.
- CUDA Graph text-tick island promotion on 2026-06-12: `predictive_route_vote_mode=cuda_graph_text` captures production input normalization/projection, exact fresh reconstruction distance, fused route/vote, and the in-place transition after checkpoint restore. Controlled 128-tick parity was exact across winners, reconstruction, competitive/predictive tensors, spike windows, and prepared input/projected state. Three fresh-process 512-tick arms averaged `264.46 ticks/sec` versus `176.24` for `fused_triton_text` (`1.501x`); graph median latency was `2.806-3.105 ms`, p95 `7.608-8.511 ms`, and VRAM was `18.88/50 MB` allocated/reserved versus `10.76/26 MB`. A native opt-in checkpoint ran 576 graph-backed ticks at `274.17 ticks/sec`, with zero failures and capture latency `138.75 ms`. The sensory gate bypassed graph pre-routing on all `72/72` warmup/measured ticks, executed zero graph transitions, and preserved winners and cross-modal tensors exactly. A configured service source tick replayed the graph `24` times with zero failures but took `1240.473 ms`, so graph replay is promoted while full-runtime production velocity remains unproven.
- Device-resident consolidation-vector foundation on 2026-06-12: the memory store now materializes the exact importance-weighted per-column consolidation level once and caches a tensor per compute device. Appends and replacements update the affected cached bucket; replay consolidation and restore invalidate the cache. This removes candidate-ID CUDA-to-CPU synchronization from future fused/in-place transition work without moving archival memory buffers onto GPU. Focused CPU/CUDA tests verify scalar parity, cache identity, invalidation, and CUDA placement.
- Explicit binding-hub topology maintenance benchmark on 2026-06-11: a fresh 32-column layer with ramped hub evidence produced a bounded `12`-edge growth delta. Core refresh median/p95 was `18.4834/31.6406 ms` on CPU and `127.3827/221.0722 ms` on CUDA. At 1024 columns the same synthetic evidence produced `572` edges and measured `517.0388/709.0404 ms` on CPU versus `4646.7792/5072.0355 ms` on CUDA; the transaction's default `16`-edge budget would reject and restore that mutation. This is an explicit slow-path result and a negative CUDA speed result: topology selection remains Python/control-bound even though binding tensors stay on CUDA. It adds no work to `bind()`.
- Post-restart live verification on 2026-06-11: the local curriculum runtime reached Runtime Truth `alive` with `resolved_device=cuda`, `observed_cuda_execution=true`, `10/1024` competitive columns scored, `145` hub-evidence updates, zero hub-topology refreshes, and zero binding growth/prune events. Four completed background ticks processed `89` tokens; the latest tick was `15177.6583 ms`, and a fifth tick was active. The design/preflight API produced a ready binding-hub transaction with target/method/reason binding, while an application with `confirmation=false` stayed blocked and left revision unchanged.
- Candidate binding-growth planner benchmark on 2026-06-11: eight repeated-failure candidate sources and a 16-edge budget measured median/p95 `0.868/1.0014 ms` on CPU and `1.3233/2.3053 ms` on CUDA at 32 columns. At 1024 columns it measured `5.0028/6.8094 ms` on CPU and `5.1334/5.8004 ms` on CUDA. The first CUDA implementation measured about `303.2645 ms` median because row-wise scalar reads synchronized the host; replacing them with two bounded adjacency snapshots and CPU vectorized membership checks removed that regression. This is an explicit endpoint cost, not tick cost or CUDA acceleration evidence.
- Live binding-growth trial verification after backend restart on 2026-06-11: Runtime Truth was `alive`, `resolved_device=cuda`, and observed CUDA execution was true. The endpoint returned `binding_growth_trial_design.v1` with status `blocked_missing_binding_growth_trial_evidence`, zero candidates, zero proposed edges, and no mutation because the live growth gate had no repeated failures. Binding reported `156` hub-evidence updates, zero topology refreshes, and zero growth/prune events. Candidate-subset execution remained `10/1024`; five ticks had processed `148` background tokens. One latest tick measured `787.3722 ms`, but this single sample is not a full-runtime speedup claim.
- Operator responsiveness verification on 2026-06-11: while the 1024-column CUDA curriculum runtime was active, checkpoint save failed closed with HTTP `409` in `343.68 ms` and no output file. Stop completed in `3967.76 ms`. Saving after stop took `14508.63 ms` and wrote an `11,667,609` byte checkpoint containing `18` slow-memory input patterns. The change removes external source shutdown from the manager-lock transition and prevents live checkpoint serialization from stalling Runtime Truth. Checkpoint serialization remains an explicit slow path, not a hot-path or CUDA speed claim.

## Interpretation

- Persistent text-tick executor promotion on 2026-06-12/13: cProfile over 256 graph-backed ticks identified the two-replay split around reconstruction readback and Python neuromodulation as the next fixed-shape barrier. The promoted executor now captures reconstruction-driven neuromodulation, fused route/vote, and in-place transition with projection/reconstruction in one replay. Sixteen-tick sequential parity preserved winners and reconstruction exactly; maximum model-state drift was `5.82e-11` in three prototype-velocity elements. A reversed same-process 256-tick A/B averaged `151.55` versus `97.15 ticks/sec` (`1.560x`) for persistent versus fused. Separate fresh processes measured `95.58` versus `63.49 ticks/sec` (`1.506x`), median `8.558` versus `13.726 ms`, p95 `21.445` versus `28.724 ms`, and allocated/reserved VRAM `27.04/76 MB` versus `18.91/52 MB`; capture startup was `68.17 ms`. The text quality gate preserved all winners and declared quality at `57.14` versus `47.52 ticks/sec` (`1.202x`). Live service processed 24 source tokens with 24 replays, 24 host truth synchronizations, zero failures, and `8.79 tokens/sec`, proving execution but not production velocity.
- Graph-owned competitive surprise on 2026-06-13: the persistent executor now computes the exact post-transition winner/prototype error inside its captured workflow and returns it in the existing host truth packet. A reversed 256-tick complete hot-window A/B averaged `86.1057` versus `70.2125 ticks/sec` (`1.2264x`), with variant/control arm medians `7.9721/9.1797` and `9.2160/13.6455 ms`. Sixteen sequential ticks preserved every winner and matched competitive surprise history within `1e-7`; text quality preserved every winner and declared quality at `34.3431` versus `24.3698 ticks/sec`. A fresh cProfile no longer contained `SurpriseMonitor.update` or its per-tick tensor norm; `record_error` consumed `0.040 s` across 256 ticks. Runtime Truth reported 320 graph-owned competitive-surprise updates, 320 packet syncs, and zero failures. A configured service tick processed 24 source tokens with 24 replays/updates and CUDA observed, but took `4763.93 ms`; status-sidecar column metabolism projection alone reported `3078.705 ms`, so neither service throughput nor thousands of complete ticks is claimed.
- Sampled batched ConceptStore observation on 2026-06-13: the next service bottleneck was not CUDA transition work but synchronous semantic maintenance. The prior 24-token CUDA source tick at `reports/graph_competitive_surprise_20260613/service-benchmark.json` spent `4763.9338 ms` total, `1289.8819 ms` in source collection, `825.6972 ms` in `trainer_step`, and `2391.1165 ms` in concept observation, for `5.0379 tokens/sec`. The promoted path keeps the same first/eighth/final sampling but sends sampled observations as one batch and runs structural growth/pruning once at the source-window boundary. `reports/concept_observation_batch_20260613/service-benchmark.json` kept CUDA observed execution and 24 graph replays with zero failures while reducing concept observation to `27.3519 ms`. After fixing the benchmark so unchanged local source fixtures do not invalidate the file-source cache, `reports/concept_observation_batch_20260613/service-benchmark-final.json` measured `1117.6784 ms` for 24 source tokens, `21.4731 tokens/sec`, `0.0458 ms` source collection, `0.1805 ms` concept observation, one concept-observation batch, four attempts, one observed concept update, and one structural-maintenance pass. The remaining live bottleneck is `trainer_step` at `723.0595 ms` for 24 tokens, so the next throughput target is host-orchestrated trainer stages rather than semantic observation.

- Slow Memory Archive Cadence on 2026-06-13: opt-in trainer-stage profiling first isolated the 24-token configured-source tick at `reports/trainer_stage_profile_20260613/service-benchmark-profile-final.json`. That run measured `543.5608 ms` for the tick (`44.1533 tokens/sec`), `534.9111 ms` in `trainer_step`, CUDA observed execution, 24 graph replays, and unchanged source-cache skip. The trainer profile counted 24 ticks at `45.1631 tokens/sec`; the largest substages were `memory_update=10.3515 ms/token`, `cross_modal=5.4062`, `routing_prepare=3.3326`, `column_transition=1.2587`, and `routing_index_buffer=1.1877`. The promoted path adds `slow_memory_archive_interval_tokens=8`, archives first/cadence/high-surprise tokens, skips stream-text episode reconstruction when no slow-memory record will be written, and exposes archive counters in Runtime Truth. A clean warm run before the Runtime Truth status-field addition, `reports/slow_memory_archive_cadence_20260613/service-benchmark-stream-gated-final2.json`, measured `371.8173 ms` for 24 tokens (`64.5478 tokens/sec`) and trainer-profile `86.5639 tokens/sec`; `memory_archive` fell to `0.2347 ms/token`, with remaining top costs `cross_modal=3.7565`, `routing_prepare=2.1664`, `stream_text_context=1.8036`, `routing_index_buffer=1.3103`, and `column_transition=1.1978`. Final Runtime Truth verification at `reports/slow_memory_archive_cadence_20260613/service-benchmark-final-runtime-truth.json` reported CUDA observed execution, 24 graph replays, zero graph failures, source-cache skip, `slow_memory_archive_count=3`, `slow_memory_archive_skip_count=21`, `slow_memory_last_archive_reason=cadence`, and tick throughput `55.0733 tokens/sec`. Its Runtime Truth verdict was `degraded` because memory pressure was high, so this is a trainer-metabolism speed win, not a completed production-velocity claim.

- Cross-modal trace-gated wake cleanup on 2026-06-13: text-only ticks no longer wake cross-modal grounding on a fixed cadence unless accepted visual/audio evidence has opened a bounded CPU-side residual-trace window. This removes ungrounded text-only cross-modal plasticity from the hot path and avoids a rejected implementation that read visual/audio trace magnitude with a CUDA scalar `.item()` each token. `reports/cuda_graph_prepare_profile_20260613/service-benchmark-profile.json` measured `cross_modal=3.3192 ms/token`, `routing_prepare=2.1148`, graph host truth sync `1.0241`, and 24 CUDA graph replays with zero failures at trainer-profile `90.5115 tokens/sec`. The rejected GPU-scalar trace gate at `reports/cross_modal_trace_gated_20260613/service-benchmark-profile.json` regressed total profile cost to `18.6317 ms/token`, proving that report/wake scalar reads can dominate. The promoted CPU-side wake-window run at `reports/cross_modal_cpu_trace_window_20260613/service-benchmark-final.json` kept CUDA graph execution active with 24 replays, zero failures, and `cuda:0` tensors; `cross_modal` fell to `1.0215 ms/token`, but total trainer-profile throughput was `76.0989 tokens/sec` due to higher routing/index and stream/context costs in that run. This is a grounded metabolism cleanup, not a complete throughput win. The next bottleneck is still host-controlled per-token orchestration, especially graph host-truth synchronization and copied/scalar control state.

- Cadenced CUDA graph host truth mirror on 2026-06-13: Runtime Truth scalars no longer force a full CUDA graph result `.tolist()` every text tick. The graph still performs 24 device-owned neuromodulator/surprise updates for a 24-token source tick, while `cuda_graph_host_truth_sync_interval_tokens=4` mirrors Python scalar truth on the first tick and then every fourth tick. `reports/cuda_graph_cadenced_truth_20260613/service-benchmark-final2.json` reported CUDA graph active, `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, and `surprise_update_count=24`. Compared with `reports/cuda_graph_prepare_profile_20260613/service-benchmark-profile.json`, `cuda_graph_prepare_host_truth_sync` fell from `1.0241` to `0.3349 ms/token`. After also cadencing host-only winner-consolidation readback, `column_transition_consolidation_readback` fell from `0.7558` to `0.0461 ms/token` and trainer-profile throughput measured `92.5874 tokens/sec`. This is a Runtime Truth metabolism improvement, not proof of thousands/sec: `stream_text_context=3.1755`, `routing_prepare=2.2285`, `routing_index_buffer=2.0445`, `column_transition_winner_readback=0.8222`, and input-copy/replay overhead still dominate the complete tick.

- Deferred winner host materialization on 2026-06-13: the graph/in-place transition now returns the device winner tensor without forcing an immediate Python list. The trainer materializes the host winner id only for the first CPU-owned consumer and `_buffer_routing_index_update()` can carry CUDA id tensors until the routing-index flush boundary. `reports/cuda_graph_deferred_winner_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, and `surprise_update_count=24`. Compared with `reports/cuda_graph_cadenced_truth_20260613/service-benchmark-final2.json`, trainer-profile throughput rose from `92.5874` to `156.2895 tokens/sec`, total profiled cost fell from `10.8006` to `6.3984 ms/token`, `column_transition` fell from `1.5285` to `0.2873 ms/token`, and `column_transition_winner_readback` fell from `0.8222` to `0.0014 ms/token`. The host sync still exists as `winner_host_materialize=0.2495 ms/token`; remaining bottlenecks are `stream_text_context=2.9539`, `routing_prepare=1.2102`, `routing_index_buffer=0.5546`, and graph input copy/replay/sync.

- Raw-window archive text on 2026-06-13: live `train_step` no longer rebuilds expanded stream episode text just to write a slow-memory archive record. It stores the bounded raw window as replay text and leaves `_update_stream_text()` to slow/evaluation/display helpers. Compared with `reports/cuda_graph_deferred_winner_20260613/service-benchmark-final.json`, `reports/raw_window_archive_text_20260613/service-benchmark-final.json` reduced `stream_text_context` from `2.9539` to `0.0009 ms/token` while keeping CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, and `surprise_update_count=24`. Total profiled cost improved only from `6.3984` to `6.2952 ms/token` and throughput from `156.2895` to `158.8525 tokens/sec` because routing, host-truth sync, input copy, replay, and memory archive timing varied upward in that run. The durable claim is deletion of the old stream-context hot-path cost, not a broad throughput step.

- Cadenced winner host mirror on 2026-06-13: graph-backed text ticks now reuse winner ids already present in synced graph truth packets and skip separate host winner materialization on graph ticks where no CPU-owned consumer needs an exact id. Slow-memory archive bucket ids remain exact. `reports/cadenced_winner_host_mirror_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, `surprise_update_count=24`, `graph_host_winner_reuse_count=7`, `winner_host_mirror_sync_count=7`, and `winner_host_mirror_skip_count=17`. Compared with `reports/raw_window_archive_text_20260613/service-benchmark-final.json`, `winner_host_materialize` disappeared from the per-tick profile, total profiled trainer cost fell from `6.2952` to `4.0150 ms/token`, and profiled trainer throughput rose from `158.8525` to `249.0665 tokens/sec`. The configured-source tick wall time remained noisy (`1873.75` versus `2656.94 ms`), so this is a trainer hot-path host-sync deletion, not an endpoint throughput claim.

- Host truth interval-8 default on 2026-06-13: after device-owned graph surprise and cadenced winner host mirroring, the production default for `cuda_graph_host_truth_sync_interval_tokens` moved from `4` to `8`. `reports/host_truth_interval_8_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_interval_tokens=8`, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, `winner_host_mirror_sync_count=4`, and `winner_host_mirror_skip_count=20`. Compared with `reports/cadenced_winner_host_mirror_20260613/service-benchmark-final.json`, `cuda_graph_prepare_host_truth_sync` fell from `0.4596` to `0.1055 ms/token`, `routing_prepare` fell from `1.8441` to `1.3480 ms/token`, total profiled trainer cost fell from `4.0150` to `3.8688 ms/token`, and profiled trainer throughput rose from `249.0665` to `258.4778 tokens/sec`. This is a Runtime Truth mirror metabolism improvement; exact per-token scalar parity remains an explicit interval-1 evaluation setting.

- Cadenced awake ripple tagging on 2026-06-13: high-dopamine awake ripple tagging now follows slow-memory archive cadence instead of attempting replay-priority memory scans on every warm-memory tick. `reports/awake_ripple_archive_cadence_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, `slow_memory_archive_count=3`, `slow_memory_archive_skip_count=21`, `awake_ripple_tag_count=3`, and `awake_ripple_tag_skip_count=21`. Compared with `reports/host_truth_interval_8_20260613/service-benchmark-final.json`, `post_surprise_replay_tag` fell from `0.4866` to `0.0685 ms/token`, total profiled trainer cost fell from `3.8688` to `3.1264 ms/token`, and profiled trainer throughput rose from `258.4778` to `319.8550 tokens/sec`. This is a replay-metabolism cleanup; it does not prove improved replay quality, and every-token ripple tagging remains retired unless consolidation evidence justifies the cost.

- Cross-modal fast idle skip on 2026-06-13: text-only ticks with no accepted sensory evidence, no residual trace, and no due self-criticism window now record cached idle state directly instead of walking the full cross-modal bookkeeping block. `reports/cross_modal_fast_idle_20260613/service-benchmark-final2.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, and Runtime Truth `cross_modal_hot_path` evidence reporting `fast_idle_skip_count=24`, `text_idle_skip_count=24`, and `text_update_count=0`. Compared with `reports/awake_ripple_archive_cadence_20260613/service-benchmark-final.json`, `cross_modal` fell from `0.6770` to `0.4933 ms/token`. Total profiled trainer cost regressed from `3.1264` to `3.7369 ms/token` because `routing_index_buffer` spiked from `0.4427` to `1.2420 ms/token` in that run, so the claim is limited to cross-modal idle bookkeeping reduction, not broad throughput.

- Batched routing-index winner-id flush on 2026-06-13: pending CUDA winner ids now materialize as one tensor batch during the explicit routing-index flush instead of one scalar read per buffered entry. The routing-index flush cadence is unchanged. A rejected snapshot-on-enqueue variant at `reports/hnsw_batched_id_flush_20260613/service-benchmark-final.json` regressed `routing_index_buffer` to `3.6198 ms/token` and profiled trainer throughput to `160.9413 tokens/sec`, so per-token CUDA id cloning was removed. The promoted batched-flush-only run at `reports/hnsw_batched_id_flush_20260613/service-benchmark-final2.json` kept CUDA graph execution active with `24` replays, zero failures, and `cuda:0` tensors. Compared with `reports/cross_modal_fast_idle_20260613/service-benchmark-final2.json`, `routing_index_buffer` fell from `1.2420` to `0.7771 ms/token`, total profiled trainer cost fell from `3.7369` to `2.5741 ms/token`, and profiled trainer throughput rose from `267.6018` to `388.4834 tokens/sec`. Treat this as routing-index maintenance evidence; complete endpoint timing remains environment-sensitive.

- Nonblocking graph input staging on 2026-06-13: the persistent text graph now stages the next input vector into the fixed graph input buffer with `non_blocking=True`, matching the existing parameter staging path without changing graph math or pointer ownership. `reports/graph_input_nonblocking_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, and `cuda:0` tensors. Compared with `reports/hnsw_batched_id_flush_20260613/service-benchmark-final2.json`, `cuda_graph_prepare_input_copy` fell from `0.4154` to `0.3203 ms/token`, `routing_prepare` fell from `0.9592` to `0.7870 ms/token`, total profiled trainer cost fell from `2.5741` to `1.6251 ms/token`, and profiled trainer throughput rose from `388.4834` to `615.3373 tokens/sec`. The configured-source tick wall time was noisy (`67.05` versus `672.14 ms`), so this is trainer graph-prep evidence, not endpoint throughput proof.

- Bounded batched live text ingestion on 2026-06-13: the live source path no longer mutates the learned-chunk codebook while collecting a cognitive tick. For the current empty-codebook, order-weighted RTF windows and deterministic chunk signatures are assembled in batches of at most 32 and emitted as one CUDA tensor batch. CPU scalar/batch parity differed by at most `1.19e-7`. Against `reports/device_owned_routing_cache_20260613/service-benchmark-runtime-truth.json`, the fresh-path run at `reports/batched_live_ingestion_cold_20260613/service-benchmark.json` reduced `collect_source_queue` from `505.6024` to `37.0373 ms`, reduced the complete 24-token tick from `547.4245` to `118.1937 ms`, and raised complete-runtime throughput from `43.8417` to `203.0565 tokens/sec` (`4.632x`). Runtime Truth still recorded `24` CUDA graph replays, zero graph failures, `cuda:0`, and observed CUDA execution. The cache-restored run at `reports/batched_live_ingestion_20260613/service-benchmark-batch32.json` reached `566.3356 tokens/sec`, while its trainer-only profile reached `889.0404 tokens/sec`. Learned-chunk plasticity quality under a future slow cadence remains unproven.

- Post-replay previous-flag cleanup on 2026-06-13: the persistent text graph no longer writes Python `host_parameters[4] = 1.0` after replay because the next preparation step recomputes that flag from `trainer._prev_routing_key` before staging device parameters. Focused CUDA graph parity and host-truth cadence tests passed. `reports/post_replay_previous_flag_cleanup_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, and host truth sync/skip counts of `4/20`; its specific `cuda_graph_prepare_bookkeeping` bucket fell from `0.0645` to `0.0139 ms/token`, but total profiled trainer cost regressed to `2.0787 ms/token` due to higher input/replay/routing-index stages. This is deletion of stale host bookkeeping, not a throughput promotion.

- Rejected CUDA graph parameter-stage skip on 2026-06-13: a signature-based attempt to skip the pinned host-parameter copy did not delete work because the competitive modulator changed every token. `reports/parameter_stage_skip_20260613/service-benchmark-final.json` reported `parameter_copy_count=25`, `parameter_skip_count=0`, `cuda_graph_prepare_parameter_stage=0.2998 ms/token`, and total profiled trainer cost `2.4091 ms/token` (`415.10 tokens/sec`), worse than the retained nonblocking-input baseline. The experiment was removed; the next credible path is a larger device-owned modulator/control executor, not more Python signature checks.

- Rejected existing-id torch routing-cache in-place update on 2026-06-13: a direct CUDA trainer smoke proved the mechanism could keep the `torch_topk` cache ready and preserve tensor data pointers after an existing-id routing-index update, but complete service profiles rejected promotion. `reports/routing_cache_inplace_update_20260613/service-benchmark-final.json` reduced `cuda_graph_prepare_eligible` to `0.0330 ms/token`, yet raised `routing_index_buffer` to `0.4161 ms/token` and total profiled cost to `1.9300 ms/token`. The optimized position-map version at `reports/routing_cache_positioned_update_20260613/service-benchmark-final.json` reduced `cuda_graph_prepare_eligible` further to `0.0317 ms/token`, but raised `routing_index_buffer` to `0.6745 ms/token` and total profiled cost to `3.1060 ms/token` (`321.96 tokens/sec`). The retained baseline remains `reports/graph_input_nonblocking_20260613/service-benchmark-final.json` at `1.6251 ms/token` (`615.34 tokens/sec`). The runtime experiment was removed; the next routing-index speed path must be a broader device-owned route/index executor, not per-update cache mutation.

- Rejected device-owned recent-spike-row graph cursor on 2026-06-13: removing the pre-replay host fill and letting the persistent CUDA graph advance the spike-row cursor looked aligned with device-owned control state. Focused CUDA graph parity still passed, and Runtime Truth showed `24` graph replays, zero failures, `cuda:0` tensors, and `4/20` host truth sync/skip counts. Complete configured-source profiles rejected promotion: `reports/recent_spike_row_device_owned_20260613/service-benchmark-final.json` measured `366.21 tokens/sec` and `2.7307 ms/token`; the warm repeat measured `342.98 tokens/sec` and `2.9156 ms/token`. The same-current retained host-fill run at `reports/recent_spike_row_host_fill_retained_20260613/service-benchmark-final.json` measured `487.93 tokens/sec` and `2.0495 ms/token` with the same CUDA replay/failure/device evidence. The runtime experiment was removed; keep the small pre-replay fill until a broader persistent executor wins complete ticks.

- Routing-cache clean reuse on 2026-06-13: `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` now expose a read-only dirty bit so `ColumnTransitionRuntime` and the persistent text graph can reuse already-bound routing-cache tensors on clean ticks instead of calling the rebuild-capable cache accessor every time. Dirty caches still rebuild through retrieval before graph replay. Focused routing/cache and CUDA graph tests passed. The first configured-source run at `reports/routing_cache_clean_reuse_20260613/service-benchmark-final.json` proved the counters (`route_cache_clean_fastpath_count=23`, `route_cache_rebuild_check_count=1`, `route_vote_clean_cache_reuse_count=24`) but regressed to `457.89 tokens/sec` and `2.1839 ms/token`, so it is not a universal win. The warm repeat at `reports/routing_cache_clean_reuse_20260613/service-benchmark-warm.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, and `4/20` host truth sync/skip counts while improving to `626.22 tokens/sec` and `1.5969 ms/token`, slightly above the prior strongest retained `graph_input_nonblocking` profile at `615.34 tokens/sec` and `1.6251 ms/token`. Treat this as a warmed trainer graph-prep improvement, not endpoint throughput proof.

- Rejected deferred routing-index winner-vector gather on 2026-06-13: a contained trainer experiment queued routing-index winner ids without prototype vectors, then gathered deduplicated prototype rows only at the routing-index flush boundary. Focused tests and CUDA graph checks passed, and the targeted `routing_index_buffer` bucket fell from the retained warm `0.5031 ms/token` to `0.2040`, `0.1748`, and `0.1808 ms/token` across `reports/hnsw_deferred_vector_flush_20260613/service-benchmark-final.json`, `service-benchmark-warm.json`, and `service-benchmark-repeat.json`. Complete configured-source throughput did not hold up: those same runs measured `435.55`, `327.18`, and `424.24 tokens/sec`, all below the retained warm routing-cache baseline at `626.22 tokens/sec`. CUDA graph execution stayed active with `24` replays, zero failures, `cuda:0` tensors, and `23` clean route-cache fast-path hits in the attempted runs. The runtime edit was removed; the result is retired-path evidence that the next useful speed slice must collapse a larger route/index/graph-prep cluster, not just move the routing-index vector gather.

- Profiled persistent text-tick A/B evidence on 2026-06-13: `hot_window_benchmark` can now opt into measured-step-only trainer-stage profiling, and `persistent_tick_hot_window_benchmark` reports reversed same-process stage deltas. Focused CPU contract tests passed. On the RTX 3060 with torch `2.11.0+cu128`, `reports/persistent_tick_profile_ab_20260613/profiled-ab.json` compared fused route/vote against the persistent CUDA text-tick executor over four 64-sample arms after 16 warmup steps per arm. Persistent averaged `360.0649 ticks/sec` versus fused `135.5841` (`2.6557x`). CUDA evidence reported active graph execution on persistent arms with `80` replays each, zero failures, tensor device `cuda:0`, and host truth sync/skip counts `11/69`. Stage deltas identify the next bottlenecks and wins: total measured cost fell from `7.1688` to `2.5739 ms/tick`; `routing_prepare` fell by `1.8044 ms/tick`, `column_transition` by `1.5452`, `candidate_winner` by `0.7734`, `post_surprise_replay_tag` by `0.6529`, and winner readback by `0.5513`. The persistent path still pays `cuda_graph_prepare_input_copy=0.5025`, `cuda_graph_prepare_eligible=0.2482`, `cuda_graph_prepare_replay=0.1562`, and `routing_index_buffer=0.4513 ms/tick`, so the next credible speed slice is a larger route/index/graph-prep executor or device-owned modulator/control state, not another isolated micro-optimization.

- Consolidation generation fast-path on 2026-06-13: the Persistent Text Tick Executor now checks the memory-store bucket-consolidation cache generation instead of calling `bucket_consolidation_tensor()` during every graph eligibility check. Focused CUDA tests passed, including fail-closed deactivation after cache invalidation. On the RTX 3060 with the same checkpoint, `reports/consolidation_generation_fastpath_20260613/profiled-ab.json` measured persistent throughput at `408.8154 ticks/sec` versus `360.0649` in the prior profiled persistent A/B, while fused arms in the same run averaged `152.5924 ticks/sec`. Persistent graph arms stayed active with `80` replays each, zero failures, tensor device `cuda:0`, host truth sync/skip counts `11/69`, `consolidation_cache_generation_fastpath_count=160` per arm, and zero generation or memory-warm-state mismatches. Persistent stage cost fell from `2.5739` to `2.2644 ms/tick`; `cuda_graph_prepare_eligible` fell from `0.2482` to `0.2041 ms/tick`, `routing_prepare` from `1.0502` to `0.9511`, and `cuda_graph_prepare_input_copy` from `0.5025` to `0.4476`. This promotes the pointer-generation guard as hot-path cleanup, not a new replay-quality claim.

- Prepared graph candidate reuse on 2026-06-13: `ColumnTransitionRuntime.route_candidates()` now reuses the same-token candidate buffer after `prepare_routing()` has already replayed the persistent text graph, instead of repeating routing-cache and graph-eligibility checks. Focused CUDA graph parity, host-truth cadence, consolidation fail-closed, sensory-bypass, and bootstrap-bypass tests passed. The first profiled A/B at `reports/prepared_graph_reuse_20260613/profiled-ab.json` was mixed: persistent throughput was `371.0043 ticks/sec`, below the prior `408.8154`, even though `route_vote_prepared_graph_reuse_count=80` per persistent arm and duplicate route-vote clean-cache reuse fell to `0`. The immediate repeat at `reports/prepared_graph_reuse_20260613/profiled-ab-repeat.json` was positive: persistent throughput `437.2580 ticks/sec`, fused `161.2748`, persistent total `2.0611 ms/tick`, `routing_prepare=0.8978`, `cuda_graph_prepare_eligible=0.1998`, active graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `route_cache_clean_fastpath_count=76`, `route_cache_rebuild_check_count=4`, and `consolidation_cache_generation_fastpath_count=80`. Treat this as a small duplicate-check deletion with noisy throughput evidence, not a standalone proof that the remaining graph-prep bottleneck is solved.

- Graph-prep substage profiling on 2026-06-13: the persistent graph profiler now splits the former `cuda_graph_prepare_input_copy` bucket into parameter staging, recent-spike-row fill, actual input staging, and the retained aggregate marker. This is measurement-only and runs only when trainer-stage profiling is enabled. `reports/graph_prep_substage_profile_20260613/profiled-ab.json` was intentionally not used as a throughput promotion because the extra profiler marks perturb measured timing; persistent throughput measured `260.9408 ticks/sec` while preserving active CUDA graph replay, `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, and `route_vote_prepared_graph_reuse_count=80`. The useful evidence is the split: persistent mean `cuda_graph_prepare_parameter_stage=0.3219 ms/tick`, `cuda_graph_prepare_recent_row_fill=0.1245`, `cuda_graph_prepare_input_stage=0.0933`, `cuda_graph_prepare_eligible=0.2854`, and `routing_index_buffer=0.5299`. The next speed architecture should therefore move host-owned modulator/control state into a larger device-owned persistent executor before chasing the input copy itself.

- Device-owned previous-routing flag on 2026-06-13: the persistent text graph now keeps the `has_previous_routing_key` flag in graph/device state instead of staging it from host on every replay. Focused CUDA graph parity, host-truth cadence, and consolidation fail-closed tests passed, including `previous_flag_device_owned_count`. `reports/device_owned_previous_flag_20260613/profiled-ab.json` showed active CUDA graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, prepared candidate reuse `80`, and `previous_flag_device_owned_count=80`; the measured parameter-stage bucket fell from the profiling baseline `0.3219` to `0.2462 ms/tick`. Unprofiled repeats were noisy rather than promotional: `reports/device_owned_previous_flag_20260613/ab.json` measured persistent `311.4105 ticks/sec`, and `reports/device_owned_previous_flag_20260613/ab-repeat.json` measured persistent `316.3692` versus fused `103.9618` (`3.043x`) with `144` graph replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `19/125`, and `previous_flag_device_owned_count=144`. Treat this as deletion of one host-staged control bit, not proof that graph prep is solved.

- Device-owned learning-rate counter on 2026-06-13: graph-backed text ticks now compute the competitive learning rate from a graph-owned update-count scalar and increment that scalar after replay. A new mixed-path CUDA test proves that a sensory fallback tick increments Python `update_count`, then the next graph tick resynchronizes device state and records `learning_rate_host_resync_count=1`. Focused CUDA graph tests passed. `reports/device_owned_learning_rate_20260613/profiled-ab.json` kept active graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `learning_rate_device_owned_count=80`, and zero host resyncs, but profiled throughput was only `297.9833 ticks/sec` and `cuda_graph_prepare_parameter_stage=0.2913 ms/tick`, so the profiled run is not a promotion. Unprofiled repeats were better but still not a headline win: `reports/device_owned_learning_rate_20260613/ab.json` measured persistent `336.8304` versus fused `98.7550` ticks/sec, and `reports/device_owned_learning_rate_20260613/ab-repeat.json` measured persistent `361.1783` versus fused `128.4246`, both with `144` graph replays per persistent arm, zero failures, `cuda:0`, `learning_rate_device_owned_count=144`, and zero host resyncs. Treat this as moving one exact control scalar toward device ownership; the remaining speed target is still a larger graph-prep/control executor.

- Revision-cached graph modulator staging on 2026-06-13: `SurpriseMonitor.modulator_revision` now invalidates the graph-staged competitive modulator after CPU-visible surprise changes, while graph prep reuses the already-staged device scalar between revisions. Focused surprise and CUDA graph tests passed, including exact interval-1 parity (`modulator_stage_copy_count=16`, skip `0`) and interval-4 cache behavior (`copy=3`, skip `5`). `reports/modulator_revision_cache_20260613/profiled-ab.json` kept active CUDA graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `modulator_stage_copy_count=11`, and `modulator_stage_skip_count=69`; `cuda_graph_prepare_parameter_stage` fell to `0.0317 ms/tick` from the previous learning-rate-control profile at `0.2913 ms/tick`, and persistent profiled throughput measured `329.7518 ticks/sec`. Unprofiled repeats were mixed but acceptable for a stage promotion: `reports/modulator_revision_cache_20260613/ab.json` measured persistent `296.7573` versus fused `113.0540`, while `reports/modulator_revision_cache_20260613/ab-repeat.json` measured persistent `354.1062` versus fused `102.8524`, both with `144` graph replays per persistent arm, zero failures, `cuda:0`, `modulator_stage_copy_count=19`, and `modulator_stage_skip_count=125`. This is a graph-prep host-copy reduction, not endpoint throughput proof.

- Routing-cache generation fast path on 2026-06-13: `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` now expose retrieval-owned routing-cache generation stamps, and the Persistent Text Tick Executor skips dirty-bit/pointer validation while the captured generation is unchanged. Focused routing and CUDA graph tests passed. `reports/routing_cache_generation_fastpath_20260613/profiled-ab.json` measured persistent `482.2661` versus fused `165.9433 ticks/sec`, `speedup=2.9062`, `cuda_graph_prepare_eligible=0.2017 ms/tick`, `routing_prepare=0.7375 ms/tick`, and total profiled persistent cost `1.8935 ms/tick`. Both persistent arms reported `80` graph replays, zero failures, `cuda:0`, `route_cache_generation_fastpath_count=76`, `route_cache_generation_mismatch_count=4`, `route_cache_rebuild_check_count=4`, and `route_cache_clean_fastpath_count=0`. Unprofiled repeats kept graph replay active with zero failures and `136` generation fast-path hits per persistent arm: `ab.json` measured persistent `360.1320` versus fused `98.8577`, while `ab-repeat.json` measured persistent `282.9019` versus fused `98.7611`. The second repeat was noisy (`persistent_b=214.0754`, p95 `13.4248 ms`), so this is a graph-eligibility cleanup with CUDA evidence, not a final endpoint-throughput claim.

- Device-owned routing-cache coherence on 2026-06-13: the persistent Triton transition now writes the normalized next winner prototype into the captured exact torch routing cache and the trainer skips the duplicate routing-index winner/vector enqueue on eligible graph ticks. CPU and CUDA contracts passed, including exact cache/prototype parity, exact 16-tick graph/fused sequential state, empty graph-tick routing-index buffers, and host-mirror synchronization without device-cache invalidation. The fresh pre-change profile measured persistent `500.3434 ticks/sec` and `routing_index_buffer=0.3263 ms/tick`. Two post-change profiles at `reports/device_owned_routing_cache_20260613/profiled-ab.json` and `profiled-ab-repeat.json` measured persistent `437.6423` and `484.5770 ticks/sec`; the targeted stage fell to `0.0111` and `0.0282 ms/tick`. Throughput remains variable, so the claim is deletion of the duplicate stage rather than a broad mean-throughput gain. The production-backed grounded gate passed with exact winners, bit-exact cross-modal tensors, zero measured Triton compilation events, and `36.1596` versus `25.8103 ticks/sec` (`1.4010x`). Final configured-source evidence at `reports/device_owned_routing_cache_20260613/service-benchmark-runtime-truth.json` processed 576 graph ticks with 576 device cache updates, 576 skipped buffer writes, zero failures, `cuda:0`, trainer-stage `516.3006 ticks/sec`, and `routing_index_buffer=0.0135 ms/tick`; Runtime Truth exposed `cpu_mirror_stale=true` and zero host-mirror syncs because no retained mutation boundary occurred. Full service throughput was `43.8417 tokens/sec`, leaving source/endpoint orchestration and graph preparation as the next bottlenecks.

The benchmark now emits `endpoint_metabolism_summary`, a non-runtime evidence field that separates setup work, hot-path service endpoints, status sidecars, and explicit replay/export/dataset slow paths. This protects the always-on runtime by measuring setup and slow tooling without making them part of the hot path.

The benchmark also emits `runtime_device_evidence`, a compact observed-device summary from `/status` and `/terminus`. This run proves local configured-source liveness for the benchmark harness, but it does not prove CUDA acceleration: the focused test environment observed CPU placement and CUDA was unavailable.

Column Runtime growth, pruning, retirement, and associative recall remain report-only or explicit reviewed helper surfaces. Competitive execution now has two live sparse boundaries: learned-chunk routing retrieves candidates before scoring and evaluates only that exact set, and delayed candidate-scoped homeostasis updates win-rate/threshold state only for active candidates once stale/deep-sleep evidence can exist. Retained CPU routing also has a bounded candidate deep-sleep filter: it inspects only the retrieved backfill pool, filters candidates at `dead_column_steps`, and falls back truthfully when CUDA graph/fused route-vote already owns selection. Predictive updates add a CPU-only sparse boundary under the same delayed gate, now including predictive location/velocity scope; CUDA retains dense predictive updates until a fused or lower-launch-overhead path beats the dense CUDA baseline. Retained predictive voting has a training-owned awake-mask cache boundary: after routing candidates are known, only those candidate votes recompute while non-awake columns keep cached consensus gain. Runtime Scope polling still uses a bounded 500 ms read-model projection cache and reuses the already-built scope when deriving structural-plasticity and SNN-language advisory surfaces. Explicit fresh status reads bypass the cache.

The Column Runtime report no longer performs its bounded scheduling summary through many tiny CUDA kernels and scalar synchronizations. It snapshots four source vectors once, computes report-only evidence on CPU, and exposes the transfer cost. This follows the CUDA-first boundary: tensor-heavy cognition stays on CUDA, while Python/JSON control-plane work uses the cheaper device when measured. The failed hot-path budget now points the next profiling cycle toward tick/feed execution rather than further status-sidecar tuning.

Competitive routing now also treats a zero input-weight blend as a real execution boundary. Prototype similarity remains active, while the disabled input-drive branch is not evaluated in either pre-routing assembly or candidate competition. This preserves outputs by construction and removes dead CUDA work. CPU predictive updates avoid dense per-column work after the stale horizon, adaptive context separates continuous state dynamics from four-token dense plasticity cadence, and hypercube binding no longer refreshes structural hubs during every bind. CUDA remains launch-bound across routing projection, routing search, predictive updates, binding tensor operations, and scalar control synchronization. The next optimization should target fused CUDA predictive/binding execution with quality evidence.

PyTorch guidance warns that CUDA scalar extraction can synchronize the host, but removing a synchronization is not automatically a runtime improvement when it causes otherwise skipped tensor work. MARULHO therefore keeps zero-state short-circuits until a narrower fused or event-driven replacement beats them under an uncontended endpoint benchmark.

Structural repair now follows the same hot/cold boundary. Stale counters are updated during the tick and reported through spike health, but column revival is no longer hidden inside routine competitive processing. Deep-sleep/maintenance remains the explicit place to call `force_revive_dead_columns`, preserving the possibility of checkpoint-backed review while cutting routine CUDA scans from the hot path.

The regression gate is a report-only evaluator. It compares benchmark JSON artifacts for Runtime Truth regression, configured-source liveness, absolute hot-path budgets, relative hot-path latency regression, and endpoint grouping boundaries. It does not mutate runtime state or claim speedup.

Benchmark evidence freshness is a validation projection only. The service API classifies benchmark reports from their `generated_at` timestamp as `fresh` through 24 hours, `aging` through 72 hours, `stale` after 72 hours, or `unknown_timestamp` when the timestamp is missing or unparsable. This status does not rerun benchmarks, alter Runtime Truth, or mutate baselines; it tells operators whether the saved hot-path evidence is current enough to rely on without a new slow-path run.

Runtime Truth now also exposes Benchmark Evidence Currency as advisory read-only evidence under `runtime_truth.evidence.benchmark_evidence_currency`. It scans the reports directory for the latest accepted baseline, fresh benchmark bundle, and regression gate report, then reports whether that saved evidence is `current`, `missing`, `stale`, or `failed`. This status never changes the Runtime Truth verdict and never runs benchmark work from the status path; it only prevents operators from mistaking absent or stale benchmark files for current hot-path evidence.

The accepted baseline manifest is also report-only. It records operator review metadata, a canonical JSON hash over the accepted benchmark snapshot, and a separate canonical JSON hash over the operator acceptance material. The acceptance hash binds reviewer id, note, acceptance time, baseline id/label, source report hash, Runtime Truth verdict, and hot-path p95/total into the manifest so later comparison and validation surfaces can detect review metadata drift. This is not a cryptographic identity signature; it is deterministic tamper evidence for the local slow-path report. Creating or comparing a baseline does not start the service, run replay, apply plasticity, write checkpoints, or claim CUDA acceleration.

The baseline-run bundle is the one-command slow path for this workflow. It runs a fresh configured-source service benchmark, writes `fresh-benchmark.json`, compares it against the accepted baseline, writes `comparison.json`, and records a compact `bundle-summary.json`. The validation API summarizes that bundle as read-only operator evidence, including fresh-run hot-path metrics, Runtime Truth, accepted-baseline identity, report hashes, configured-source ticks, paths, and failed checks. The bundle remains evaluation tooling; it is not an always-on runtime loop.

The service validation report endpoint now summarizes regression gate artifacts without moving the comparison logic into `service`. The API remains a read-only projection over reports generated by the explicit evaluation slow path.

The service validation report endpoint now also summarizes accepted baseline artifacts directly and recomputes the embedded snapshot hash and operator acceptance hash. A snapshot mismatch is surfaced as `baseline_integrity_status=failed` with `baseline_snapshot_hash_match` in failed checks. An approval-material mismatch is surfaced as `acceptance_integrity_status=failed` with `baseline_acceptance_hash_match` in failed checks. Older baselines without acceptance hashes are reported as `acceptance_integrity_status=legacy_unbound` and should be re-accepted before becoming durable comparison anchors. These are read-only operator evidence surfaces, not repair or mutation paths. The dashboard consumes the same report projection and does not call benchmark or replay work itself, so UI visibility adds no runtime hot-path cost.
## Deferred Source Cache Persistence, 2026-06-13

Runtime Sources now schedules changed source-cache material to one coalescing worker instead of calling `torch.save` from `prepare_training`. Focused concurrency tests prove the scheduling call returns while a blocked save remains pending, duplicate material produces one write, failed writes are visible and retryable, and close flushes the queue.

`reports/deferred_source_cache_20260613/steady-state-5-ticks.json` used the 1024-column CUDA graph checkpoint for five sequential source ticks. The final complete 24-token tick took `40.9787 ms` (`585.6701 tokens/sec`): collection `0.2144 ms`, preparation `3.1371 ms`, and train compute `36.5514 ms`. Runtime evidence recorded three scheduled/completed cache writes and zero pending writes at report time. This removes durability I/O from the cognitive tick; it does not raise the warm neural ceiling. The next large target is the remaining host-controlled trainer cluster.

## Bounded Source Concept Observation, 2026-06-13

The service benchmark now exposes the configured source tick width through `--local-source-tick-tokens` and `--local-source-queue-target-tokens` instead of hard-coding a 24-token source tick. The maintained quick-start preset uses a 128-token source tick so fixed service/source overhead can be amortized while Runtime Truth still reports the configured budget.

`reports/source_tick_window_20260613/tick-128-solo.json` measured an uncapped 128-token CUDA source tick at `281.7705 ms` (`454.270 tokens/sec` by wall time). Training took `211.4241 ms`, trainer-step time was `209.0310 ms`, and ConceptStore observation still cost `66.8507 ms` because the larger tick attempted 17 sampled source observations.

`reports/source_tick_window_20260613/tick-128-capped-concepts.json` capped source ConceptStore observation to four sampled windows per service tick. The same checkpoint processed 128 tokens in `179.0320 ms` (`714.957 tokens/sec` by wall time), with `train_compute=163.8751 ms`, `trainer_step=161.9080 ms`, `concept_observation=10.2716 ms`, CUDA observed execution, and `cuda:0` tensor evidence. The trainer-stage profile reported `823.5605 tokens/sec` over 128 train steps with active persistent graph replay and zero CUDA graph failures in Runtime Truth. This is a knowledge-layer metabolism cut, not skipped neural training; the next bottleneck is still per-token trainer orchestration around the persistent graph executor.

## Cadenced Source Concept Observation, 2026-06-14

Continuous runtime stress now isolates each report under its own run root, avoids Runtime Truth polling during measurement when the event history can retain all expected ticks, and records the configured source ConceptStore observation tick interval. `ServiceManager(trace_history_limit=...)` also controls the Runtime State event-history capacity, so long stress runs can retain all measured tick summaries without observer polling.

The isolated same-checkpoint A/B compares every-source-tick ConceptStore observation with the default interval-4 cadence. `reports/continuous_runtime_stress_20260614/stress-1024-isolated-concept-interval1-baseline.json` processed 1024 full-warm source tokens at `401.973 tokens/sec`, with 8 sampled ConceptStore ticks, 32 attempts, 7 observations, CUDA selected on RTX 3060, active `inplace_triton`, 1024 graph replays, and zero CUDA graph failures. `reports/continuous_runtime_stress_20260614/stress-1024-isolated-concept-interval4-candidate.json` processed the same 1024-token shape at `517.818 tokens/sec` (`1.264x`), with 3 sampled ticks, 5 cadenced skips, 12 attempts, 2 observations, active `inplace_triton`, 1024 graph replays, and zero failures.

The current-tree confirmation `reports/continuous_runtime_stress_20260614/stress-1024-cadence-current-tree.json` processed 1024 full-warm tokens at `513.695 tokens/sec`, retained all 8 expected tick events without measurement polling, reported 5 cadenced skips and 3 sampled ticks, and kept CUDA active on RTX 3060 with 1024 graph replays, 1024 fused reconstruction updates, and zero graph failures.

This promotes a knowledge-layer wake cadence, not skipped SNN learning. Neural `train_step` still executes for every emitted token; Runtime Truth now exposes `source_concept_observation_tick_interval` and each tick's concept-observation `mode`, `tick_due`, `tick_interval`, attempts, skipped attempts, and observations. A 2048-token full-history report (`reports/continuous_runtime_stress_20260614/stress-2048-fullwarm-concept-cadence-fullhistory.json`) retained all 16 expected tick events without measurement polling, observed 2048 tick tokens, and reported active CUDA graph replay with zero failures, but throughput was noisy at `654.266 tokens/sec`; use it as history-retention/runtime-evidence proof rather than the interval-speed promotion.

Larger capped source windows were rejected for the maintained default on the same checkpoint. `reports/source_tick_window_20260613/tick-256-capped-concepts.json` processed 256 tokens in `680.164 ms` (`376.380 tokens/sec`) with `256` graph replays and zero failures. `reports/source_tick_window_20260613/tick-512-capped-concepts.json` processed 512 tokens in `1110.996 ms` (`460.848 tokens/sec`) with `512` graph replays and zero failures. The default service/runtime source tick window is therefore aligned to `128` tokens; larger windows remain explicit benchmark/operator choices.

## Cross-Modal Trace Sleep, 2026-06-13

Text-only background ticks already kept Cross-Modal Grounding asleep when no visual/audio evidence or residual sensory trace was alive, but the cached-idle path still decayed cross-modal traces every token. The promoted path clears expired traces once, records `cross_modal_idle_trace_reset_count`, and then records cached-idle skips without per-token trace decay until sensory evidence wakes the specialist again.

`reports/cross_modal_idle_trace_sleep_20260613/tick-128-profile-repeat.json` processed 128 CUDA source tokens in `154.115 ms` (`830.548 tokens/sec`), with trainer-stage throughput `900.9459 tokens/sec`, `cross_modal=0.0127 ms/token`, `routing_prepare=0.5422`, `memory_archive=0.1137`, and active persistent graph replay. The retained comparison at `reports/source_tick_window_20260613/tick-128-capped-concepts.json` measured `714.957 tokens/sec`, trainer-stage throughput `823.5605`, and `cross_modal=0.2153 ms/token`. A first post-change profiled run was noisy (`343.823 tokens/sec`) because source collection took `131.243 ms`, so the claim is the repeated complete-tick win plus the direct cross-modal stage deletion, not universal endpoint stability.

`reports/cross_modal_idle_trace_sleep_20260613/tick-128-no-profile.json` provided an unprofiled sanity check: 128 tokens in `168.6541 ms` (`758.9498 tokens/sec`), `train_compute=148.0481 ms`, `trainer_step=145.5344 ms`, CUDA graph active, and zero graph failures.

## CPU Mirror Winner Consolidation Metric, 2026-06-13

Graph-backed text ticks were still synchronizing CUDA on telemetry refreshes to compute `winner_consolidation_level` from the device consolidation tensor. The metric does not drive transition math, slow-memory archive admission, or mutation. The promoted path refreshes it from `DualMemoryStore.bucket_consolidation_level()` when the cadenced graph truth packet already synchronized a host winner id; otherwise it reuses the cached metric and reports `winner_consolidation_cpu_metric_count` / `winner_consolidation_cached_metric_count`.

The immediate pre-change profile, `reports/current_speed_probe_20260613/tick-128-profile.json`, processed 128 tokens at `217.463 tokens/sec` with trainer-stage throughput `459.5667 tokens/sec`. Its profiler isolated `column_transition_consolidation_readback=0.7027 ms/token`, `column_transition=1.0251`, and noisy source collection at `264.8843 ms`.

Post-change profiles kept CUDA graph replay active with `128` replays, zero failures, `cuda:0`, host truth sync/skip `17/111`, and `route_vote_prepared_graph_reuse_count=128`. `reports/consolidation_metric_cpu_mirror_20260613/tick-128-profile.json` reduced `column_transition_consolidation_readback` to `0.0026 ms/token`, reached `252.184 tokens/sec` complete service throughput, and recorded one CPU metric refresh plus one cached telemetry refresh. The repeat at `reports/consolidation_metric_cpu_mirror_20260613/tick-128-profile-repeat.json` held `column_transition_consolidation_readback=0.0032 ms/token`, reached `484.245 tokens/sec` service and `569.0105 tokens/sec` trainer-stage throughput, with source collection back down to `0.0497 ms`.

This is a scalar Runtime Truth sync deletion, not a new best complete-tick claim: the older retained `reports/cross_modal_idle_trace_sleep_20260613/tick-128-profile-repeat.json` still measured `830.548 tokens/sec` service and `900.9459` trainer-stage throughput. The next bottleneck is therefore broader host-controlled route/source orchestration, especially routing preparation and fixed graph input/recent-row staging under stable source conditions.

## Continuous Hot-Window Sync Gate, 2026-06-13

The hot-window benchmark previously synchronized CUDA before and after every measured token. That is the right mode for p50/p95 per-token latency, but it adds an artificial host barrier when the question is continuous sequential tick throughput. The benchmark now exposes `sync_mode`: `step` keeps the old per-token synchronization behavior, while `window` synchronizes once around the measured arm and labels per-step samples as host-dispatch latency rather than exact CUDA latency.

Same-checkpoint 128-token no-profile A/B evidence:

- `reports/persistent_window_sync_gate_20260613/step-sync-no-profile-128-ab.json`: persistent CUDA graph mean `696.524 ticks/sec`, fused Triton route/vote mean `146.526`, speedup `4.754x`. Persistent arms replayed `144` graph ticks each with zero failures, `cuda:0`, host truth sync/skip `19/125`.
- `reports/persistent_window_sync_gate_20260613/window-sync-no-profile-128-ab.json`: persistent CUDA graph mean `1062.896 ticks/sec`, fused Triton route/vote mean `155.823`, speedup `6.821x`. Persistent arms replayed `144` graph ticks each with zero failures, `cuda:0`, host truth sync/skip `19/125`.

The profiled step-sync run `reports/persistent_window_sync_gate_20260613/step-sync-profiled-ab.json` gives stage attribution without service/source noise: persistent mean `954.967 ticks/sec`, trainer-stage `total=0.7804 ms/tick`, `routing_prepare=0.3968`, `cuda_graph_prepare_replay=0.0972`, `cuda_graph_prepare_recent_row_fill=0.0893`, `cuda_graph_prepare_host_truth_sync=0.0758`, and `cuda_graph_prepare_input_stage=0.0611`. The profiled window-sync run was slower (`857.013`) and should be treated as profiler perturbation evidence, not a promotion target.

This gate proves the current encoded text-tick CUDA graph path can exceed a low 1000-ticks/sec floor under continuous window timing, but it does not prove live service throughput. The next production speed target is to move source/service execution toward the same boundary: prefilled device-ready source windows, fewer per-token Python metric dictionaries, and a broader persistent tick executor that reduces `routing_prepare`, recent-row staging, input staging, and scalar truth sync together.

## Source-Window Prewarm Gate, 2026-06-13

The service benchmark now makes configured-source prewarm the default evidence path. It configures the local source with `ingestion.prewarm_on_startup=true`, polls `/terminus` until Runtime Truth reports `ingestion.full_warm_ready`, records that wait as `configured_source_summary.warmup.not_hot_path=true`, and then measures `/terminus/tick`. The CLI keeps `--disable-local-source-prewarm` for cold-source regressions.

Same-checkpoint CUDA evidence used `reports/host_truth_interval_16_20260613/runtime.pt`, `MARULHO_DEVICE=cuda`, one configured source tick, `--local-source-tick-tokens 128`, and `--local-source-queue-target-tokens 128`:

- `reports/source_prewarm_live_tick_20260613/cold-no-prewarm.json`: 128 tokens in `497.4618 ms` (`257.306 tokens/sec`), `collect_source_queue=147.5918 ms`, `prepare_training=28.5999 ms`, `train_compute=315.7685 ms`, observed CUDA execution.
- `reports/source_prewarm_live_tick_20260613/warm-prewarm.json`: Runtime Truth warmup reached `full_warm_ready=true`, `ready_source_count=1`, `full_queue_source_count=1`, and `total_buffered_tokens=128` before the tick. The measured tick processed 128 tokens in `218.1416 ms` (`586.775 tokens/sec`), with `collect_source_queue=0.0559 ms`, `prepare_training=1.2179 ms`, `train_compute=212.9062 ms`, and observed CUDA execution.

This is a large live-service metabolism win because it removes source-window construction from the cognitive tick. It is not the final thousands/sec path: the warm tick is still slower than the encoded hot-window upper bound because per-token trainer orchestration, graph prep, metrics construction, and live service finalization still run on the host.

Stale report tensor artifacts were pruned after this evidence: 129 old `.pt` files under `reports/` were removed, freeing about `223.4 MB`. The retained checkpoint is `reports/host_truth_interval_16_20260613/runtime.pt`, because it is the current comparable evidence path.

## Cadenced Slow Memory Archival, 2026-06-13

The production model default and Terminus curriculum preset now set `slow_memory_archive_interval_tokens=256`. This retires the earlier every-eight-token and sixty-four-token slow-memory write defaults from the hot path while preserving first-token, strong-capture, and cadenced archival evidence through Runtime Truth.

Same-checkpoint CUDA evidence cloned `reports/host_truth_interval_16_20260613/runtime.pt` to a temporary benchmark checkpoint and changed only `slow_memory_archive_interval_tokens` to `64`. The valid warm rerun reached Runtime Truth `full_warm_ready=true`, processed 128 source tokens in `223.755 ms` (`572.055 tokens/sec`), and observed CUDA execution on `cuda`. The full service tick did not beat the prior warm baseline at `586.775 tokens/sec`, so this is not an endpoint-throughput claim.

The longer 4096-token continuous stress surface showed the old checkpoint still carried the retired interval-8 value, which made the current main path slower than the documented production cadence. `reports/continuous_runtime_stress_20260614/stress-4096-cadence-profile-current.json` processed 4096 full-warm tokens at `668.976 tokens/sec` with active `inplace_triton`, 4096 graph replays, and zero graph failures. Cloned config-only comparisons measured interval-64 at `742.767 tokens/sec` and interval-256 at `832.915 tokens/sec`, with the same full-warm source shape, 32 retained tick events, CUDA selected on RTX 3060, and zero graph failures. Interval-512 reached `873.997 tokens/sec`, but it halves replay archival density again, so it remains unpromoted until replay/consolidation quality evidence supports it.

The implementation now migrates unstamped legacy checkpoints carrying retired interval values `8` or `64` to interval `256` on load and records the migration in checkpoint metadata. The same original checkpoint, without editing the `.pt` file, then reported interval-256 execution and migration evidence in `reports/continuous_runtime_stress_20260614/stress-32768-checkpoint-migrated-memory-256-clean-complete-final.json`: `1779.859 tokens/sec`, 32768 graph replays, 32768 fused reconstruction updates, zero graph failures, all 256 tick events captured, no measurement polling, `slow_memory_archive_interval_tokens=256`, and migration metadata from `8` to `256`. Shorter same-checkpoint runs were noisy (`4096` tokens at `886.115`, `518.713`, and `701.090`; `8192` tokens at `1279.174`), so the durable claim is that the clean longer full-warm path now crosses 1k tokens/sec while the next target remains repeated long-run stability and further reduction of `train_compute`/`trainer_step`.

The continuous stress benchmark now enlarges its local runtime event-history ring for long runs instead of polling Runtime Truth snapshots during the measured window. The contaminated 32768-token run with 652 measurement polls fell to `739.383 tokens/sec` and showed `train_lock_wait=0.4516 ms/token`; the clean rerun with `poll_snapshot_count=0` reached `1779.859 tokens/sec` and `train_lock_wait=0.000745 ms/token`. This is an evidence-tool correction, not a production runtime change.

The trainer hot path did improve: trainer-stage throughput moved from `638.6378` to `663.6743 tokens/sec`, trainer-step time fell from `209.6825` to `201.745 ms`, and `memory_archive` fell from `0.188347` to `0.018416 ms/token`. The next speed slice must therefore collapse the remaining route/graph-prep cluster rather than adding more memory/reporting micro-changes.

## Rejected Route-Prep Defaults, 2026-06-13

Two plausible route-prep default changes were tested after Cross-Modal Trace Sleep and rejected because the complete 128-token CUDA service tick regressed.

`reports/candidate_homeostasis_128_20260613/tick-128-profile.json` lowered `candidate_homeostasis_start_tokens` to `128`, causing the first source tick to use the `candidate_subset` graph. CUDA graph replay stayed active with `128` replays and zero failures, but wall throughput fell to `406.525 tokens/sec` and trainer-stage throughput to `621.0137 tokens/sec`; the retained `reports/cross_modal_idle_trace_sleep_20260613/tick-128-profile-repeat.json` measured `830.548` and `900.9459` respectively. The default stays `512`.

`reports/host_truth_interval_16_20260613/tick-128-profile-repeat.json` raised `cuda_graph_host_truth_sync_interval_tokens` to `16`. Host truth sync count fell from `17` to `9` and `cuda_graph_prepare_host_truth_sync` fell from `0.0875` to `0.0456 ms/token`, but complete throughput fell to `702.015 tokens/sec` and trainer-stage throughput to `778.0866`. The production default stays `8`; larger intervals remain benchmark-only.

An attempted Brain Runtime metrics-copy pruning after the bounded source-observation cap was also rejected and reverted. It did not change neural training, but repeated complete service evidence failed to beat the retained profile: `reports/source_metrics_copy_prune_20260613/tick-128-profile-repeat.json` measured `577.810 tokens/sec` and trainer-stage throughput `723.9786`, below the retained `830.548` / `900.9459`.

## Continuous Execution Quantum, 2026-06-13

Runtime Control previously forced every background source token through a separate execution-lock/mutation cycle followed by a `5 ms` sleep. The neural updates were sequential, but the host scheduler imposed a throughput ceiling before GPU work. The production default now runs up to eight sequential token updates per bounded execution quantum with no artificial yield, checks stop requests between quanta, and exposes the policy through Runtime Truth. API and persisted configuration cap the quantum at `128` tokens.

`reports/continuous_runtime_quantum_20260613/quantum-ab-stop-evidence.json` used the same 1024-column checkpoint, a fully prewarmed 128-token local source queue, and reversed arm order:

- Legacy `1 token + 5 ms`: `131.995` and `138.484 tokens/sec`, mean `135.240`.
- Quantum `8 tokens + 0 ms`: `572.055` and `753.546 tokens/sec`, mean `662.800`.
- Mean speedup: `4.901x`.
- Quantum stop latency: `13.896` and `11.883 ms`; neither shutdown timed out.
- Every arm observed the NVIDIA GeForce RTX 3060 on CUDA, requested/resolved `inplace_triton`, and executed 256 transitions with zero failures.

The quantum path removes host scheduling waste; it does not parallelize or skip SNN token updates. A separate full-warm manual service tick with the default quantum, `reports/continuous_runtime_quantum_20260613/manual-warm-quantum8-rerun.json`, reached `692.507 tokens/sec` versus the prior one-token warm baseline at `586.775`, while lock wait and mutation-mark cost fell. The remaining production gap to sustained thousands per second is inside per-token trainer orchestration and graph preparation, not source starvation or forced scheduler sleep.

## Continuous Quantum Size Recheck, 2026-06-14

The quantum-size sweep separated encoded hot-window evidence from complete live-runtime evidence. On the encoded CUDA hot-window surface, `reports/quantum_size_sweep_20260614/quantum-16.json` measured candidate quantum staging at `613.125 tokens/sec` mean versus `534.134` for per-token copy (`1.148x`) and beat the same-run quantum-8 mean (`580.795`) from `quantum-8.json`. That was not enough to promote the runtime default.

The first live comparison was noisy because it prewarmed only one 128-token tick while measuring 256 tokens, so later arms could pay source collection inside the timing window. The benchmark now prewarms at least the target-token count, uses a longer synthetic source, and gives each arm an isolated source file so stale source-cache state cannot leak across arms.

The fixed complete background runtime check `reports/quantum_size_sweep_20260614/continuous-8-vs-16-fullwarm.json` compared legacy one-token/yield, baseline quantum-8, and candidate quantum-16 with `queue_target_tokens=256`. Baseline quantum-8 measured `638.394 tokens/sec` mean; candidate quantum-16 measured `602.249 tokens/sec` mean (`0.943x` versus baseline) with CUDA selected on the NVIDIA GeForce RTX 3060, active persistent graph replay, zero graph failures, `256` staged-token reuses per candidate arm, and zero staged-input mismatches. The default remains `8`.

The benchmark now reports `baseline_quantum_*` and `candidate_quantum_*` fields so a future quantum-size change must beat the current service-shaped path, not only a narrower encoded loop. The next credible thousands/sec path is a broader persistent tick executor that removes more host-controlled per-token stages; increasing host quantum length alone is rejected.

## Continuous Runtime Stress Gate, 2026-06-14

`continuous_runtime_stress_benchmark` is now the maintained long-run complete-runtime gate for the promoted CUDA path. It waits for a fully prewarmed source queue, runs the background Terminus loop for a target token budget, retains enough tick events to avoid measurement-window Runtime Truth polling, and aggregates tick duration plus stage timings after the run. This is an explicit evaluation slow path; it does not add hot-path reporting work. `--profile-trainer-stages` can now opt into trainer substage evidence during the same sustained surface; those profiled runs are diagnostic because instrumentation perturbs throughput.

`reports/continuous_runtime_stress_20260614/stress-1024-fullwarm.json` measured the 1024-column CUDA checkpoint with 1024 fully prewarmed source tokens, `tick_tokens=128`, and the retained `execution_quantum_tokens=8`. It completed 1024 sequential SNN token updates in `1.4923 s` (`686.196 tokens/sec`), with eight 128-token tick events observed, CUDA selected on the NVIDIA GeForce RTX 3060, active `inplace_triton`, `1024` CUDA graph replays, zero graph failures, zero quantum-input mismatches, and stop latency `13.169 ms`.

That first long run exposed dead metabolism in source-cache persistence: while `torch.save` was already off the tick, the runtime still scheduled shrinking tail-cache rewrites as the fully warmed queue drained. The promoted source-cache rule now keeps the full cached material and skips partial-tail rewrites once a full cache exists. `reports/continuous_runtime_stress_20260614/stress-1024-fullwarm-cache-tail-skip.json` reran the same surface and reached `711.113 tokens/sec`; `prepare_training` fell from `5.6861 ms/tick` to `1.8056 ms/tick`, cache writes fell from `8` to `1`, `cache_partial_skip_count=7`, cache failures remained zero, and cache flush completed with `cache_pending_count=0`.

The remaining large bottleneck is still the trainer/graph-prep cluster, not source collection or disk persistence. In the post-change run, top complete-runtime stage costs were `train_compute=1.0669 ms/token`, `trainer_step=0.9952 ms/token`, `train_lock_wait=0.1757 ms/token`, and concept/finalize work below that. The next broad speed slice should collapse more per-token trainer orchestration into the persistent tick executor while preserving sequential spike updates, Runtime Truth sampling cadence, and fail-closed CUDA fallback.

## Burst Metadata Cleanup Rejection, 2026-06-14

The promoted thirty-two-token truth/event cadence still keeps a small CPU-side pending metadata window so rare strong events can attach raw text, input patterns, and metadata to device-captured assembly/routing payloads. A compact pending-window rewrite preserved focused strong-event tests but did not win complete runtime: clean 32768-token CUDA stress runs measured `3551.811` and `3334.081 tokens/sec`. A narrower shared-metadata-only version kept the proven tuple shape and removed per-token dict copies, but still measured `3543.149 tokens/sec`. These are below the retained same-surface interval-32 evidence (`4237.534 tokens/sec` at 32768 tokens and `4577.595` at 131072 tokens), so the runtime code was reverted and the cleanup was retired. The next speed work should target broader device-owned execution and routing/graph replay boundaries, not Python metadata shape churn.

## Promoted Burst-Path Profiling, 2026-06-14

`--profile-trainer-stages` no longer forces the training-owned text-burst executor to fall back to per-token `train_step`. The profiler now records burst-level buckets on the promoted CUDA path while keeping burst Runtime Truth active. `reports/burst_path_profile_20260614/stress-8192-burst-profile.json` processed `8192` sequential tokens with `8184` burst-owned tokens, `1023` burst executions, `257` host-truth syncs, `7935` host-truth skips, active `inplace_triton` on `cuda:0`, and zero graph/burst failures. The diagnostic profile measured `text_burst_graph_replay=0.372945 ms/token`, `text_burst_cpu_maintenance=0.010769`, `text_burst_event_and_idle=0.008031`, `text_burst_pending_metadata=0.005825`, and `text_burst_boundary_plan=0.00543`. Profiled throughput was perturbed (`1936.341 tokens/sec` complete, `2464.714` trainer-observed), so clean stress runs remain the promotion gate. The useful result is target selection: the next large speed slice should attack the true burst graph replay/device-owned execution boundary, not fallback-only routing or metadata buckets.

The runtime subprofile at `reports/burst_runtime_subprofile_20260614/stress-8192-burst-runtime-subprofile.json` split the prior broad graph-replay bucket while preserving the promoted executor: `8192` sequential tokens, `8184` burst-owned tokens, `1023` burst executions, `257` host-truth syncs, `7935` skips, active `inplace_triton` on `cuda:0`, and zero graph/burst failures. The dominant sub-buckets were `text_burst_runtime_event_drain=0.153854 ms/token`, `text_burst_runtime_replay_loop=0.115524`, and `text_burst_runtime_input_stage=0.037902`; control-state staging was `0.009396`, Python mirrors `0.003568`, and host-truth gate checks `0.002022`. A clean unprofiled 32768-token check after the instrumentation reached `2547.717 tokens/sec` with 32768 CUDA graph/in-place Triton executions and zero failures, below prior best evidence, so this is not a throughput promotion. It is architecture evidence: the next large speed slice should move the truth/event drain boundary and burst result publication farther out of the hot path or make it device-resident before attempting another replay-loop fusion.

## Slim Burst Event Packet, 2026-06-14

Current inspection found that ordinary cadence drains copied all pending result rows to Python even when no strong replay event existed. The trainer needs exact per-token rows only for strong captures; no-strong cadence drains need the final truth row, strong flags, and counters. The CUDA burst runtime now returns a slim packet: one final result row plus strong-result rows only for strong indices. Forced strong-event tests still preserve all raw windows, token markers, and slow-memory admissions.

The profiled retained interval-32 run at `reports/slim_burst_event_packet_20260614/stress-8192-profile-interval32.json` processed `8192` sequential tokens with `8184` burst-owned tokens, `257` host-truth syncs, `7935` skips, active `inplace_triton` on `cuda:0`, and zero graph/burst failures. `text_burst_runtime_event_drain` fell from the prior `0.153854` to `0.056869 ms/token`; `text_burst_graph_replay` fell from `0.328792` to `0.228635`; Runtime Truth reported `256` slim result packets and `0` strong result rows for the no-strong corpus.

The clean 131072-token run at `reports/slim_burst_event_packet_20260614/stress-131072-clean-interval32.json` reached `3656.459 tokens/sec`, below the retained best interval-32 evidence at `4577.595`, so this is a packet/metabolism cleanup and profiling-stage improvement rather than an endpoint-throughput promotion. It still removes dead host materialization from the maintained path and strengthens Runtime Truth with explicit packet counters. A wider 128-token truth/event candidate at `reports/event_queue_128_20260614/stress-131072-clean-interval128.json` reached only `3973.773 tokens/sec`; the next large speed slice should remove or make asynchronous the synchronization boundary itself, not merely widen or shrink host packets.

## Sparse Burst Event Payload Loads, 2026-06-14

The Slim Burst Event Packet removed ordinary no-strong result-row materialization
on the host, but code inspection found a remaining device-side mismatch: the
Triton packet publisher still loaded routing-key and assembly payload tensors for every
burst event before masking stores for no-strong rows. Those payloads are replay
archive data, not ordinary Runtime Truth. MARULHO now predicates both the loads
and stores on the strong-event condition. Result rows and strong flags still
write in every packet; exact routing/assembly payloads remain available for
real strong captures.

Focused CUDA tests prove both sides of the boundary:
`test_burst_event_snapshot_skips_payload_when_not_strong` leaves sentinel
routing/assembly rows untouched under a high threshold, while
`test_burst_event_snapshot_preserves_payload_when_strong` copies exact
routing/assembly payloads under a low threshold.

The diagnostic 8192-token profile at
`reports/sparse_burst_payload_20260614/stress-8192-profile.json` kept the
promoted path active: `8192` sequential tokens, `8184` burst-owned tokens,
`1023` burst executions, `257` host-truth syncs, `7935` skips, active
`inplace_triton` on the RTX 3060, zero graph/burst failures, and zero strong
event rows. It measured `3041.984 tokens/sec` complete and `3744.536`
trainer-observed, with `text_burst_runtime_event_drain=0.058908 ms/token`,
`text_burst_runtime_replay_loop=0.138568`, and
`text_burst_graph_replay=0.221720`. Compared with the previous slim-packet
profile's `0.056869 ms/token` event drain, this does not prove an event-drain
speed win; profiling overhead and host conditions dominate that small bucket.

The clean 32768-token stress run at
`reports/sparse_burst_payload_20260614/stress-32768-clean.json` is the
maintained evidence for correctness under the full service executor:
`3411.104 tokens/sec`, `32768` in-place CUDA/Triton executions, `4094` burst
replays, `32752` burst-owned tokens, `1025` host-truth syncs, `31743` skips,
`32768` staged/reused quantum-input tokens, zero mismatch/discard/fallback
copies, zero graph/burst failures, and `0` strong result rows. Runtime Truth
kept `resolved_device=cuda`, `cuda_selected=true`,
`route_vote_resolved_mode=cuda_graph_text`, and
`route_vote_kernel_variant=two_stage_route_vote`. CPU/GPU contention probes
were clean (`64%` CPU max, `0%` GPU max). The result is above recent contended
checks but below the retained top `4577.595 tokens/sec`, so this is accepted as
dead device-memory work deletion, not a new throughput ceiling. The next broad
target remains the actual replay loop and host-truth boundary: one graph replay
per token plus synchronous truth publication, not more no-strong payload
trimming.

The longer same-surface check at
`reports/sparse_burst_payload_20260614/stress-131072-clean.json` is the stronger
comparison because the retained top was also a 131072-token run. It processed
`131072` sequential tokens in `29.5296 s` (`4438.669 tokens/sec`), with
`131072` in-place CUDA/Triton executions, `16382` burst replays, `131056`
burst-owned tokens, `4097` host-truth syncs, `126975` skips, `8194` quantum
input stages for `131072` staged/reused tokens, zero graph/burst failures, zero
forced event drains, and zero strong result rows. Runtime Truth stayed on the
RTX 3060 `cuda_graph_text` path with `route_vote_kernel_variant=two_stage_route_vote`.
Velocity environment probes reported no contention (`82%` CPU max, `17%` GPU
max). This protects the high-throughput path but still does not beat the
retained top `4577.595 tokens/sec`; `train_compute=0.191043 ms/token` remains
the dominant stage. The next large implementation target is therefore a
lower-level device-owned multi-tick executor or persistent sequence kernel that
reduces actual per-token graph/kernel launch and truth-publication work while
preserving exact sequential SNN state.

## Fused Burst Event Packet Publication, 2026-06-14

The standalone `burst_event_cuda` snapshot kernel is retired. The in-place
transition kernel now publishes the Slim Burst Event Packet directly: final
result row, neuromodulator scalars, winner, effective modulator,
competitive-surprise, strong flag, and optional routing/assembly payloads for
real strong captures. Normal full-capacity truth drains also skip the redundant
device slot reset because the fused packet writer already wraps the slot at the
thirty-two-token event capacity.

Focused CUDA tests cover the fused packet in both no-strong and forced-strong
paths, partial forced drains, and full-capacity natural slot wrap. The promoted
runtime activation check reported `active=true`, `fallback_reason=None`,
captured `all_columns` and `candidate_subset` burst graphs, and kept
`route_vote_kernel_variant=two_stage_route_vote`.

The diagnostic 8192-token profiled run at
`reports/fused_burst_event_packet_20260614/stress-8192-profile-interval32.json`
processed `8192` tokens with `8184` burst-owned tokens, active
`inplace_triton`, zero graph/burst failures, and zero strong rows. Profiling
measured `text_burst_runtime_event_drain=0.036503 ms/token`,
`text_burst_runtime_replay_loop=0.137001`, and
`text_burst_graph_replay=0.195881`. The complete profiled throughput was
`3136.410 tokens/sec`, but the run reported CPU contention, so it is target
selection evidence rather than a promotion.

The long clean stress gate at
`reports/fused_burst_event_packet_20260614/stress-131072-clean-slot-skip-interval32.json`
processed `131072` sequential tokens at `3923.410 tokens/sec` with
`train_compute=0.210933 ms/token`, `131072` in-place CUDA/Triton transitions,
`16382` burst executions for `131056` burst-owned tokens, `4097` host-truth
syncs, `126975` skips, `4096` slim packet drains, `4094` full-capacity slot
reset skips, `2` actual slot resets, zero forced drains, zero strong rows, and
zero graph/burst failures. Runtime Truth selected the RTX 3060 CUDA path and
kept `burst_event_ring_device_owned=true`.

This is accepted as device-boundary cleanup and dead-path deletion, not a new
throughput ceiling. The run reported CPU contention (`96%` max CPU), and it
remained below the retained uncontended top
`reports/host_truth_interval_sweep_20260614/stress-131072-i32.json` at
`4577.595 tokens/sec`. The next large speed target is still a real lower-level
device-owned multi-tick/persistent sequence executor that reduces the one
CUDA Graph replay per token, not another post-transition packet trim.

## Preferred Burst Capacity Ownership And q16 Rejection, 2026-06-14

The service execution quantum is `16`, but the maintained CUDA burst capacity remains `8`. The trainer now asks `ColumnTransitionRuntime` for the runtime-owned burst capacity instead of carrying a separate hard-coded eight-token chunk size. This keeps the boundary in the transition runtime and prevents service/trainer drift if a future executor proves a different capacity.

A candidate that raised the preferred Python burst group to `16` passed focused CUDA parity and reduced host burst groups in `reports/wide_burst_20260614/stress-8192-profile-interval32.json`: `burst_replay_count=511` versus the prior eight-token shape's `1023`, `persistent_executor_burst_tokens=16`, zero graph/burst failures, and `text_burst_runtime_replay_loop=0.179855 ms/token`. That did not reduce the underlying per-token CUDA Graph replay launch count, because `replay_staged_text_burst()` still calls the one-tick graph once per token.

The clean sustained gate at `reports/wide_burst_20260614/stress-32768-clean-interval32.json` rejected the candidate: `2283.710 tokens/sec`, `train_compute=0.379761 ms/token`, `persistent_executor_burst_tokens=16`, `2046` burst groups, all `32768` transitions on CUDA, and zero graph/burst failures. The retained direction is therefore not wider Python bursts. The next credible launch-reduction work is a fused or persistent device-owned multi-tick executor that actually lowers graph/kernel launches while preserving exact sequential SNN state, host-truth freshness, stop responsiveness, and rollback evidence.

## Sustained Host-Truth Recheck and Abstraction Clone Retirement, 2026-06-14

Host-truth synchronization and slow-memory archival use separate cadences. The retained checkpoint carries `cuda_graph_host_truth_sync_interval_tokens=16`, while the promoted slow-memory archival cadence is `256`. A wider host-truth interval therefore cannot be inferred from the memory result.

Config-only 8192-token candidates measured interval `32` at `1530.180 tokens/sec`, interval `64` at `1267.956`, and interval `128` at `1098.737`. The promising interval-32 candidate then completed a clean 32768-token run at `1560.459 tokens/sec`, below the retained interval-16 clean reference at `1779.859 tokens/sec`. Larger host-truth intervals remain benchmark-only; no production or checkpoint default changed.

The sustained profiler exposed unconditional `assembly.clone()` work after the column transition even when `enable_abstraction_layer=false`. The current speed checkpoint has the abstraction layer disabled, so the trainer now clones only inside the enabled abstraction branch. Abstraction and column-transition tests passed. The post-change profiled 8192-token CUDA run at `reports/next_speed_cycle_20260614/stress-8192-abstraction-clone-skip-profiled.json` completed 8192 graph replays with zero failures on the RTX 3060, reported trainer-observed throughput `1984.683 tokens/sec`, and measured `column_transition=0.093578 ms/tick`. Two clean 32768-token complete-runtime repeats measured `1717.703` and `1716.632 tokens/sec`, so this is accepted as exact dead-work deletion and a local stage reduction, not an endpoint-throughput promotion.

The remaining profiled costs are broad rather than one scalar copy: `routing_prepare=0.309408 ms/tick`, `cuda_graph_prepare_replay=0.170041`, `column_transition_python_bookkeeping=0.054251`, host-truth sync `0.041206`, and parameter staging `0.018574`. The next credible architecture slice is persistent ownership across routing preparation, replay launch/post-transition bookkeeping, and compact truth publication, not further cadence widening.

## Allocation-Free Graph Bookkeeping, 2026-06-14

Current code inspection found that graph eligibility correctly guarded the captured consolidation tensor by generation, but `ColumnTransitionRuntime.apply()` still called `bucket_consolidation_tensor()` again after replay. The same graph path also created a fresh empty CUDA `last_revived_indices` tensor every token even though hot-path revival is retired and explicit maintenance owns real revival evidence.

The promoted path skips the redundant graph-only consolidation lookup and reuses one runtime-owned empty revival tensor. A 16-step CUDA parity test preserves the retained sequential state, proves zero downstream consolidation lookups, and verifies stable empty-tensor pointer identity. Runtime Truth exposes `graph_consolidation_lookup_skip_count` and `graph_empty_revival_tensor_reuse_count`.

The profiled 8192-token run at `reports/next_speed_cycle_20260614/stress-8192-graph-bookkeeping-reuse-profiled.json` reduced `column_transition_python_bookkeeping` from `0.054251` to `0.014375 ms/tick` and total `column_transition` from `0.093578` to `0.049348`. That process had unusually slow replay timing, so it is stage evidence rather than an endpoint result.

Two clean sustained runs provide the complete-runtime evidence:

- `stress-32768-graph-bookkeeping-reuse-clean.json`: `2169.815 tokens/sec`.
- `stress-32768-graph-bookkeeping-reuse-clean-repeat.json`: `2191.057 tokens/sec`.

Both processed 32768 sequential tokens with no measurement polling, selected CUDA on the RTX 3060, executed 32768 persistent graph and in-place Triton transitions with zero failures, and reported 32768 consolidation-lookup skips plus 32768 empty-tensor reuses. Against the retained clean reference at `1779.859 tokens/sec`, the repeated gains are `1.219x` and `1.231x`. No additional startup or persistent VRAM allocation was introduced beyond one zero-length tensor owned by the transition runtime.

NVIDIA CUDA Graph guidance supports paying repeated launch/setup cost once, while PyTorch/NVIDIA memory guidance warns that dynamic allocations and graph memory-pool behavior remain real overhead. This change applies that existing direction to MARULHO's downstream bookkeeping rather than widening the cognitive graph or changing SNN semantics. The next target remains fewer host replay launches and broader device-owned truth/bookkeeping across a sequential execution quantum.

## Persistent Quantum Input Ring, 2026-06-13

The Persistent Text Tick Executor now owns a fixed 128-row CUDA input ring and
the recent-spike-row cursor. Brain Runtime offers already encoded tensors in
bounded sequential quanta; training stages each quantum with one or two
contiguous device operations, verifies pointer order during consumption, and
falls back before mutation on mismatch or sensory boundaries. This removes the
per-token static-input copy and host cursor fill without batching or skipping
SNN state transitions.

Two same-checkpoint reversed-order, 256-sample continuous CUDA A/B runs at
`reports/quantum_input_staging_20260613/ab-256-run1.json` and
`ab-256-run2.json` measured:

- Per-token-copy means `758.571` and `746.187 ticks/sec`.
- Quantum-ring means `1026.381` and `877.533 ticks/sec`.
- Speedups `1.353x` and `1.176x`.
- Every arm replayed 288 persistent graph ticks with zero graph failures.
- Quantum arms staged 288 tokens, reused all 288, and recorded zero fallback
  copies, mismatches, or discards.

The synchronized-per-token diagnostic at `profile-step-128.json` regressed to
`0.926x` because the forced barrier removes the cross-token overlap that the
ring is designed to preserve. It nevertheless reduced measured
`cuda_graph_prepare_input_stage` from roughly `0.207-0.303 ms/tick` to
`0.004-0.005 ms/tick`. Treat window synchronization as the continuous
throughput gate and step synchronization as latency/profile evidence.

Live Runtime Truth at `service-enabled.json` proved 128 graph replays, 16
quantum stages, 128 staged reuses, CUDA `cuda:0`, and zero fallback copies,
mismatches, or failures. Its source prewarm reached only one buffered item, so
the resulting service throughput is not a warm service-speed claim. The next
large target is broader persistent multi-tick ownership across routing
preparation, graph replay/post-transition bookkeeping, compact metric packets,
and event-driven memory admission.

### Quantum-Boundary Input Staging, 2026-06-14

Warm text-sequence execution now pre-stages a full metric-free training quantum
when the persistent graph has already produced host truth. Consecutive
eight-token bursts consume pointer-checked slices from that staged q16 window
instead of staging a smaller window for each burst. This keeps the same exact
one-tick CUDA graph and does not batch neural time.

The before profile at
`reports/next_speed_cycle_20260614/stress-8192-profile-current-env.json`
measured `text_burst_runtime_input_stage=0.053073 ms/token` and
`train_compute=0.364045 ms/token`. After quantum-boundary staging,
`reports/next_speed_cycle_20260614/stress-8192-profile-quantum-input-stage.json`
measured `text_burst_runtime_input_stage=0.001323` plus
`text_sequence_quantum_input_stage=0.024848 ms/token`; the guarded 4096-token
profile measured `0.001447` plus `0.030428 ms/token` with exactly `4096`
staged tokens and `4096` reused tokens.

Clean stress evidence proved the maintained CUDA path but not a new throughput
ceiling because both fresh runs reported CPU contention:
`reports/next_speed_cycle_20260614/stress-32768-clean-quantum-input-stage.json`
reached `2872.995 tokens/sec` with `32768` CUDA transitions, zero graph/burst
failures, `32784` staged tokens, and `32768` reused tokens while CPU peaked at
`93%`; the repeat
`reports/next_speed_cycle_20260614/stress-4096-clean-quantum-input-stage-repeat.json`
reached `2361.073 tokens/sec` while CPU peaked at `100%`. The retained best
long-run velocity evidence remains
`reports/host_truth_interval_sweep_20260614/stress-131072-i32.json` at
`4577.595 tokens/sec`; this iteration improves staging correctness and
profiling visibility while leaving the next velocity gate to an uncontended
long run.

The follow-up preflight fix at
`reports/boundary_prestage_preflight_20260614/stress-32768-clean.json`
keeps quantum input staging from crossing a known burst fallback boundary.
The run processed `32768` sequential tokens at `2624.774 tokens/sec`, selected
CUDA on the NVIDIA GeForce RTX 3060, executed `32768` in-place Triton
transitions and persistent graph replays with zero failures, reported
`4094` burst executions for `32752` burst-owned tokens, and kept quantum
staging exact: `32768` staged tokens, `32768` reused tokens, and zero discards.
It is not a velocity promotion because `velocity_environment.v1` reported
CPU-busy and GPU-busy contention (`100%` CPU max, `33%` GPU max). The evidence
is narrower but important: training now previews q16 slices with a
non-mutating boundary classifier, so speculative full-quantum staging does not
dirty Runtime Truth counters or copy input tensors across a known sleep,
host-truth, metrics, or other fallback boundary.

## Boundary-Aware Text Burst, 2026-06-14

The first implementation captured eight complete tick bodies into one CUDA
Graph. It passed exact eight-tick state parity and a 4096-tick probe reached
`1.230x`, but the required 32768-tick run regressed to `0.951x`: `6748.34`
versus `7097.98 ticks/sec`. That graph was deleted rather than retained as an
unused experiment.

The promoted path keeps the faster one-tick graph and removes Python
orchestration between safe ticks. Training owns the eligibility gate and falls
back before mutation at drift, telemetry, sleep, slow-memory, strong-capture,
cross-modal wake, host-truth, routing-mode, and metrics boundaries. Every
neural transition remains sequential and executes on CUDA.

Repeated complete full-warm 32768-token runs:

- `reports/text_burst_executor_20260614/stress-32768-long.json`:
  `2387.898 tokens/sec`, 32768 graph replays, 18032 burst tokens, zero graph or
  burst failures.
- `reports/text_burst_executor_20260614/stress-32768-long-rerun.json`:
  `2607.316 tokens/sec`, 32768 graph replays, 18016 burst tokens, zero graph or
  burst failures.

The prior allocation-free bookkeeping repeats measured `2169.815` and
`2191.057 tokens/sec`. Mean complete throughput therefore rose from `2180.436`
to `2497.607 tokens/sec` (`1.145x`). The rerun measured
`train_compute=0.3012 ms/token` and `trainer_step=0.1657 ms/token`. The next
large bottleneck is source preparation plus the boundary ticks that still need
full Python orchestration, not the in-place transition kernel.

## Device Strong-Event Ring, 2026-06-14

The previous burst gate used the last mirrored reconstruction error and could
miss a threshold crossing inside a burst. The promoted burst graph now writes
eight result rows into a fixed CUDA ring and copies assembly/routing rows only
for strong events. A forced-threshold CUDA test preserved all eight events,
their exact token markers and raw windows, and CPU ownership of archival input
and routing tensors.

Final complete full-warm 32768-token runs measured `2648.747`, `2533.719`, and
`2599.013 tokens/sec`. Each ran all 32768 transitions on the RTX 3060 with zero
graph failures and exact one-time quantum staging. The final Runtime Truth run
used 9760 burst tokens, captured zero strong events for that corpus, and
reported fallback counts: host truth 1033, exploration 581, drift refresh 441,
telemetry 369, drift floor 1. Capture startup was `480.524 ms`. The next large
speed slice is a bounded truth-interval device event queue that can preserve
events across multiple host bursts, followed by measured treatment of
exploration/drift/telemetry boundaries.

### Device Strong-Count Drain And Range Boundary Classifier, 2026-06-14

The in-place transition kernel now maintains a device-owned cumulative
strong-event count. Host-truth drains read that scalar and skip the CPU
strong-flag vector when the count proves no strong events occurred in the
pending window; strong drains still materialize exact flags, result rows,
routing keys, and assemblies. The cognitive boundary controller also replaced
its per-token Python scan with range arithmetic while preserving the prior loop
semantics in focused tests.

The profiled 8192-token run at
`reports/boundary_controller_profile_20260614/stress-8192-profile-after-strong-count.json`
processed `8192` CUDA graph-backed executions with zero graph/burst failures,
`256` no-strong flag-scan skips, `0` strong flag scans, and
`classification_mode=range_arithmetic`. It measured `2875.310 tokens/sec`
complete and trainer-observed `3651.014 tokens/sec`; the targeted buckets were
`text_burst_runtime_event_drain=0.057988 ms/token` and
`text_burst_boundary_plan=0.004666 ms/token`. The run still reported GPU
contention, so it is diagnostic evidence, not a top-speed promotion.

The longer validation at
`reports/strong_event_count_20260614/stress-32768-clean.json` reached
`3908.062 tokens/sec` over `32768` tokens with CUDA selected on the RTX 3060,
zero graph/burst failures, `1024` no-strong flag-scan skips, and
`train_compute=0.215580 ms/token`. The 4x long run at
`reports/strong_event_count_20260614/stress-131072-clean.json` reached
`4122.568 tokens/sec`, executed all `131072` transitions on the persistent CUDA
path, skipped `4096` no-strong flag scans, preserved zero graph/burst failures,
and measured `train_compute=0.202385 ms/token`. `velocity_environment.v1`
reported CPU and GPU contention on the long run, so the retained uncontended
best remains `4577.595 tokens/sec` at
`reports/host_truth_interval_sweep_20260614/stress-131072-i32.json`.

## Native Parent-Graph Coverage Refresh, 2026-06-15

The maintained exact-burst path was refreshed with the same 131072-token
service stress shape used for current throughput claims. Under observed CPU/GPU
contention, `reports/base_comparison_20260615_partial_replay/current-native-131072-i32.json`
reached `4489.641 tokens/sec` with `train_compute=0.183604 ms/token`,
`16382` native parent-graph launches, `131056` native-covered burst tokens,
zero native fallbacks/failures, zero graph/burst failures, and parent graph
token-count coverage `[8]`. The same shape with native replay disabled reached
`4012.083 tokens/sec` and `train_compute=0.208029 ms/token` at
`reports/base_comparison_20260615_partial_replay/disabled-native-131072-i32.json`.
The earlier uncontended refreshed top remains
`reports/base_comparison_20260615/current-native-131072-i32.json` at
`4992.049 tokens/sec`.

A partial-tail experiment on `tick_tokens=130` intentionally exercised
non-eight-token burst tails. Opt-in native partial replay at
`reports/partial_native_burst_replay_20260615/native-partial-16640-t130.json`
proved parent graph token-count coverage `[2, 8]`, `lazy_compile_count=2`, and
zero native partial fallbacks, but complete runtime measured only
`2786.829 tokens/sec`. The default partial-disabled comparison at
`reports/partial_native_burst_replay_20260615/partial-disabled-16640-t130.json`
measured `2907.600 tokens/sec`, retained `[8]` parent graph coverage, and
reported `128` partial replay fallbacks covering `256` Python-loop tokens.
Both runs also exposed `384` `host_truth_boundary` burst fallbacks caused by
the unaligned 130-token tick width. Partial native parent graphs therefore stay
opt-in; the production path should preserve aligned 128-token source ticks
until a startup-warmed or lower-level device-owned multi-tick executor wins
long complete-runtime evidence.

### Startup-Warmed Native16 Parent Graph Prototype, 2026-06-15

The next capacity probe kept the exact fast shape: 1024 columns, 64 column dim,
`k=10`, text-only CUDA checkpoint, `tick_tokens=128`,
`execution_quantum_tokens=16`, host-truth cadence `32`, and a clean
`131072`-token sustained run. It changed only the startup-warmed native
repeated-child parent graph capacity from the maintained eight-token default
to `16` through the now-retired `--native-burst-tokens 16` probe.

The clean report at
`reports/native_burst_sequence_20260615/native16-131072-i32.json` processed
all `131072` tokens on the RTX 3060 with `velocity_environment.v1` reporting
`not_observed` contention. Runtime Truth exposed
`persistent_executor_burst_tokens=16`,
`persistent_executor_default_burst_tokens=8`, then-allowed capacities `[8, 16, 32]`,
parent graph token-count coverage `[16]`, `8190` native parent-graph launches,
`131040` native-covered tokens, `4097` host-truth syncs, `126975` host-truth
skips, zero native fallbacks/failures, zero graph/burst failures, and fallback
reasons `runtime_not_fully_warm=1` plus `sleep_boundary=1`. Startup cost
remained visible outside measured warm throughput:
`capture_latency_ms=6112.8292` and
`native_burst_replay_compile_latency_ms=5609.5473`.

The throughput result rejects native16 as a default promotion:
`4887.767 tokens/sec` and `train_compute=0.168278 ms/token`, below the refreshed
native8 base at `reports/base_comparison_20260615/current-native-131072-i32.json`
with `4992.049 tokens/sec` and `train_compute=0.166575 ms/token`. Because total
throughput varies with host availability, the promotion comparison uses the
clean `velocity_environment.v1=not_observed` long run; the contended profile
pair is only diagnostic.

The 2026-06-16 cleanup removed this repeated-child capacity as a live option:
`cuda_graph_native_burst_tokens` is fixed at `8`, checkpoint load migrates old
`16`/`32` values to `8`, and the stress benchmark no longer accepts native burst
capacity overrides. Native16 remains historical rejection evidence only.

The profile pair at
`reports/native_burst_sequence_20260615/profile-8192-native8.json` and
`reports/native_burst_sequence_20260615/profile-8192-native16.json` ran under
observed contention. It still explains the boundary: native16 reduced parent
launches from `1023` to `511`, but moved
`text_burst_runtime_replay_loop` only from `0.159096` to
`0.149719 ms/token` and `text_burst_graph_replay` from `0.302545` to
`0.293198 ms/token`, while event drain worsened from `0.101787` to
`0.122930 ms/token`. The next executor boundary should therefore move below
repeated child-graph wrapping into a C++/CUDA, Triton persistent-kernel, or
hybrid device-owned sequence executor rather than adding another local Python
wrapper.

### Native32 Parent Graph Under Q16, 2026-06-15

The former configured capacity, `--native-burst-tokens 32`, is not a valid
native parent-graph benchmark under the maintained exact fast shape because
`execution_quantum_tokens=16` caps the chunks that training offers to the burst
executor. The probe at
`reports/native_burst_sequence_20260615/native32-131072-i32.json` still ran the
full `131072` tokens and warmed `[32]` parent graphs, but Runtime Truth showed
zero native coverage: `native_burst_replay_attempt_count=8190`,
`native_burst_replay_success_count=0`,
`native_burst_replay_fallback_count=8190`,
`native_burst_replay_python_loop_token_count=131040`, and backend
`python_loop_partial_disabled`. Host-truth cadence and fail-closed behavior
remained intact (`4097` syncs, `126975` skips, zero graph/burst failures), but
the executor that actually ran for burst tokens was the retained Python replay
loop.

The same cleanup removed native32 from the live config/env/benchmark surfaces.
Reopening it now requires a new explicit execution-quantum decision or a
lower-level sequence executor, not a restored repeated-child capacity knob.
The follow-up q16 cleanup also removed conditional-WHILE q8/q32 capacity knobs:
`cuda_graph_sequence_loop_tokens` is fixed at `16`, old checkpoint values are
migrated back to `16`, and the stress benchmark no longer accepts
`--sequence-loop-tokens`.
The clean rerun at
`reports/column_scheduler_20260616/sequence-loop-capacity-cleanup-8192-131072-i32-rerun.json`
kept the real 8192-column path in band at `6145.401 tokens/sec` with
`train_compute=0.129706 ms/token`, `prepare_training=0.006353 ms/token`,
`finalize_total=0.005916 ms/token`, `tick_duration_ms.p95=21.270`,
`route_input_rows_scored=10/8192`, `state_transition_cached_count=8182`,
`persistent_executor_sequence_loop_capacity_fixed=true`, zero
graph/native/sequence failures, and no observed contention.

The native-burst capacity cleanup gate at
`reports/column_scheduler_20260616/native-burst-capacity-cleanup-8192-131072-i32.json`
ran the real 8192-column promoted checkpoint for `131072` tokens with
`tick_tokens=128`, `quantum_tokens=16`, and host-truth cadence `32`. It reached
`6276.616 tokens/sec` with `train_compute=0.129187 ms/token`,
`prepare_training=0.006331 ms/token`, `finalize_total=0.005816 ms/token`, and
`tick_duration_ms.p95=21.513`. Runtime Truth kept `route_input_rows_scored=10`
out of `8192`, `state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, `native_partial_burst_replay_enabled=false`,
and zero graph/native/sequence failures. `velocity_environment.v1` reported no
observed contention.

The run also reported `velocity_environment.v1=contention_observed` from GPU
activity, so its `4454.287 tokens/sec` is not a promotion comparison. The
evidence is executor coverage, not speed: a warmed parent graph token count is
not native token coverage unless the native success/token counters move. The
stress benchmark now rejects native burst capacities that exceed
`quantum_tokens` or fail to divide `quantum_tokens`, so future exact-capacity
probes fail before startup instead of producing misleading Python-loop reports.
Together with the native16 miss, nested graph rejection, native host-loop
rejection, partial-native rejection, and route/vote wrapper rejection, this is
the evidence basis for [ADR 0007](../../adr/0007-lower-level-text-sequence-executor-required.md):
the next promotable text executor must move below the current Python/CUDA Graph
replay boundary.

### Conditional-WHILE Sequence Executor Promotion, 2026-06-15/16

The first lower-level CUDA sequence executor is now the fixed eligible CUDA text
sequence path. It is not another repeated-child capacity wrapper: the native
extension builds a CUDA Graph conditional `WHILE` parent around the retained
one-tick child graph and uses a tiny device counter kernel to decide whether the
loop body runs again. Failed construction falls back before mutation to retained
repeated-child replay; launch failures remain fail-closed. The former
config/env/CLI selector was removed after promotion so native8 is fallback, not
a live alternate path.

The clean same-session comparison on the RTX 3060 used the same checkpoint,
`tick_tokens=128`, `quantum_tokens=16`, host-truth cadence `32`, and
`131072` target tokens:

- `reports/conditional_sequence_20260615/native8-rerun-131072-i32.json`:
  retained native8 repeated-child replay reached `5035.537 tokens/sec`,
  `train_compute=0.165231 ms/token`, zero native failures, parent token counts
  `[8]`, `16382` parent launches, `131056` native-owned tokens, and
  `velocity_environment.v1` contention `not_observed`.
- `reports/conditional_sequence_20260615/conditional-while-131072-i32.json`:
  conditional-WHILE q8 reached `5277.975 tokens/sec`,
  `train_compute=0.156673 ms/token`, parent token counts `[8]`, `16382`
  conditional parent launches, `131056` conditional-owned tokens, zero
  sequence/native fallbacks, zero sequence/native failures, and no observed
  contention.
- `reports/conditional_sequence_20260615/conditional-while16-131072-i32.json`:
  conditional-WHILE q16 reached `5559.473 tokens/sec`,
  `train_compute=0.146978 ms/token`, parent token counts `[16]`, `8190`
  conditional parent launches, `131040` conditional-owned tokens, zero
  sequence/native fallbacks, zero sequence/native failures, host-truth cadence
  `4097/126975`, and no observed contention.

Startup remains visible and outside warm throughput. The q16 conditional run
reported `capture_latency_ms=6864.9883` and
`native_sequence_loop_compile_latency_ms=6450.2083`. Runtime Truth exposes the
new `native_sequence_loop_*` fields separately from
`native_burst_replay_parent_graph_*`, so coverage claims must use
`native_sequence_loop_success_count`, `native_sequence_loop_token_count`,
fallback/failure counters, parent token counts, and host-truth cadence.

This first probe became promotion-candidate evidence. The follow-up promotion
gate supplied repeated paired clean long runs in both orders, fallback tests for
unavailable conditional construction, fail-closed launch-failure coverage, and
an ADR/config decision. Conditional-WHILE q16 is now the maintained eligible
default, while native8 repeated-child replay remains internal fallback only.

### Route-Vote Scheduler Filter In Fused CUDA Route, 2026-06-15/16

The scheduler slice moved deep-sleep filtering into the fused CUDA route-vote
owner instead of filtering candidates after graph/fused route-vote had already
selected a winner. The follow-up pressure slice uses the same owner for
memory-pressure filtering when cached pressure evidence exists.
`core.fused_route_vote_cuda` now reads the existing route-score rows plus
`steps_since_win` and, when enabled, `ColumnMetabolismState.memory_pressure`;
it masks ineligible rows before route top-k vote, writes a twelve-field
`route_vote_scheduler_filter.v1` device state packet, and lets training build
the `ColumnWakePlan`. There is no extra all-column sleep or pressure census;
fallback remains explicit when the route rows do not contain enough eligible
candidates.

The first clean long run at
`reports/column_scheduler_20260615/route-vote-sleep-filter-131072-i32.json`
proved the execution effect but was slower: `5930.322 tokens/sec` with
`train_compute=0.139577 ms/token`, no observed contention, and `4097` filter
state syncs. That sync cadence was too eager because the filter count is Runtime
Truth evidence, not per-token transition input.

The promoted cadence run at
`reports/column_scheduler_20260615/route-vote-sleep-filter-131072-i32-sync-cadence.json`
kept the same `131072`-token, `tick_tokens=128`, q16 conditional-WHILE, host
truth interval `32` shape and reached `6135.026 tokens/sec` with
`train_compute=0.133995 ms/token`, no observed CPU/GPU contention, `8190`
conditional sequence-loop launches over `131040` burst tokens, and zero
sequence/native fallbacks or failures. Runtime Truth reported
`route_vote_deep_sleep_filter.v1` enabled on `cuda:0`, input route rows `1024`,
output candidates `10`, filtered deep-sleep rows `1014`, eligible route rows
`10`, no fallback reason, one control update, `129` state syncs, and
`state_dirty=false`.

Compared with the fused candidate-predictive baseline at
`reports/column_scheduler_20260615/promoted-fused-candidate-predictive-131072-i32.json`
(`6141.078 tokens/sec`, `train_compute=0.126682 ms/token`), throughput is
effectively the same 6k-ish band while train compute still regresses by about
`0.0073 ms/token`. Keep the route-vote sleep filter as the real scheduler
boundary, but the next speed pass should reduce the remaining route/filter
bookkeeping rather than add a new all-column sleep decision.

Focused 2026-06-16 tests now prove the pressure half of the same route-owner
packet: high-pressure route rows are masked before CUDA candidate/winner
selection in both direct `fused_triton_text` and captured `cuda_graph_text`
modes, and the trainer wake plan reports
`candidate_memory_pressure_filter_route_vote`. The pressure gate is
evidence-aware: when `ColumnMetabolismState` has no cached pressure source, the
CUDA route owner leaves pressure filtering disabled so the default hot path
does not pay a no-op route-row pressure read.

The current 131072-token stress rerun at
`reports/column_scheduler_20260616/route-vote-pressure-filter-current-131072-i32-rerun.json`
processed `131072` tokens at `5947.863 tokens/sec` with
`train_compute=0.136041 ms/token`, `prepare_training=0.006541 ms/token`, and
`finalize_total=0.005931 ms/token`. It stayed on RTX 3060 CUDA with
`route_vote_resolved_mode=cuda_graph_text`, `131072` route/vote executions,
`8190` q16 conditional sequence-loop launches over `131040` tokens, zero
graph/sequence/native failures or fallbacks, and no observed CPU/GPU contention.
Runtime Truth showed the pressure gate had real cached evidence
(`memory_store_bucket_consolidation_gap`) but did not apply because only `6`
route rows remained below the pressure threshold for `k=10`; it reported
`memory_pressure_applied=false`, `memory_pressure_over_threshold_count=4`, and
fallback reason `insufficient_awake_route_scores_after_memory_pressure_filter`.
Compared with the previous default-route run (`6014.550 tokens/sec`,
`train_compute=0.136192 ms/token`), complete throughput is slightly lower while
train compute is effectively neutral/slightly lower. Treat this as stable
same-band Runtime Truth evidence, not a new speed ceiling.

### Sparse CUDA State Transition Promotion, 2026-06-16

The next cleanup moved stale-counter, recent-spike, assembly active-winner, and
candidate homeostasis bookkeeping onto the same bounded candidate-subset CUDA
transition path instead of leaving state transition as a dense fallback label.
The fused transition now receives device state-step counters, an
all-materialized-step scalar, per-row stale-age stamps, recent-spike active IDs,
and an assembly-active-winner scalar. In candidate-subset graph/burst mode it
updates only routed candidates plus the winner; non-awake rows remain logically
exact through cached age instead of being physically incremented every tick.
Route-vote deep-sleep masking reads the logical age, so cached sleep is not a
decorative status field.

Focused verification passed:

- `python -m pytest tests/test_column_transition_runtime.py tests/test_fused_route_vote_triton.py tests/test_inplace_column_cuda.py -q` -> `57 passed`
- `python -m pytest tests/test_predictive_columns.py tests/test_column_runtime.py tests/test_status_read_model.py tests/test_columns.py tests/test_column_scheduler_benchmark.py -q` -> `292 passed`

The longer 131072-token run used the same checkpoint, `tick_tokens=128`,
`quantum_tokens=16`, and host-truth interval `32` as the pressure-filter
baseline:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/column_scheduler_20260616/sparse-cuda-state-transition-candidate-homeostasis-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 300 --sample-interval-seconds 0.5`

It processed `131072` tokens at `6007.582 tokens/sec` with
`train_compute=0.132374 ms/token`, `prepare_training=0.006450 ms/token`,
`finalize_total=0.005983 ms/token`, `tick_duration_ms.mean=18.763324`,
`tick_duration_ms.p95=21.143100`, RTX 3060 CUDA selected, no observed
CPU/GPU contention, `8190` q16 conditional sequence-loop successes over
`131040` tokens, and zero graph/sequence/native failures or fallbacks. Runtime
Truth reported `state_transition_mode=candidate_subset_sparse_cuda_graph_route_transition_burst`,
`state_transition_column_count=10`, `state_transition_cached_count=1014`,
`state_transition_cached_fraction=0.990234375`,
`state_transition_runs_all_columns=false`, and no state-transition fallback.

Against the immediate baseline
`reports/column_scheduler_20260616/route-vote-pressure-filter-current-131072-i32-rerun.json`
(`5947.863 tokens/sec`, `train_compute=0.136041 ms/token`,
`prepare_training=0.006541 ms/token`, `finalize_total=0.005931 ms/token`,
`tick_duration_ms.p95=21.867800`), the sparse CUDA state-transition path is
faster in train compute and p95 tick while preserving the broad 6k-ish complete
runtime band. The remaining total-column scaling blocker is route-vote input
scoring, which still reads all route-cache rows before selecting the fixed
`k=10` awake mask.

### Real-Path Column Scaling Probe, 2026-06-15

The next probe tested the promoted path on power-of-two column growth rather
than a CPU synthetic scheduler sweep. The first 8192-column checkpoint attempt
failed before scheduler evidence because the in-place Triton warmup compiled an
all-columns candidate membership check as an `8192 x 8192` matrix. The live
kernel now treats all-columns candidates as a direct mask, and promoted
candidate-gated checkpoints skip unused dense startup graph and warmup shapes.

The matching promoted real-path control at
`reports/real_path_column_scaling_20260615/runtime-1024-promoted-131072-i32.json`
reached `6108.728 tokens/sec` with `train_compute=0.133438 ms/token`.
Runtime Truth stayed on `cuda_graph_text`, precompiled `[10]`, captured only
`candidate_subset`, selected `10/1024` route candidates, and reported no
sequence/native fallback.

The 8192-column real-path run at
`reports/real_path_column_scaling_20260615/runtime-8192-promoted-131072-i32.json`
also stayed on the promoted CUDA/text path: `cuda:0`,
`precompiled_candidate_counts=[10]`, graph names `["candidate_subset"]`,
`capture_graph_policy=candidate_subset_only_after_homeostasis_gate`,
`route_vote_deep_sleep_filter.output_candidate_count=10`, zero graph/native
failures or fallbacks, and `131072` fused route/transition executions. It
reached `3564.222 tokens/sec` with `train_compute=0.251487 ms/token`.

Conclusion: awake specialist work is real and bounded at 8192, but throughput
is not preserved. Runtime Truth exposes why: route-vote input rows rose from
`1024` to `8192`, so the current two-stage route-vote still pays a
total-route-cache scoring cost before selecting the fixed `k=10` awake mask.
Do not promote a same-throughput scaling claim from this result. The next speed
target is a sparse/GPU-owned route-candidate retrieval boundary, not another
sleep/status projection.

### Scheduler Truth Surface Long Run, 2026-06-15

The follow-up truth-surface run kept the same 131072-token, q16 real path and
made state-transition scope explicit in Runtime Truth. The 1024-column rerun at
`reports/scheduler_truth_surface_20260615/runtime-1024-truth-131072-i32.json`
stayed in the same 6k-ish band at `6152.495 tokens/sec` with
`train_compute=0.133843 ms/token`; route-vote still scored `1024` rows before
selecting `10`, and state transition reported `1024` columns with
`state_transition_runs_all_columns=true`.

The matching 8192-column rerun at
`reports/scheduler_truth_surface_20260615/runtime-8192-truth-131072-i32.json`
reached `3526.002 tokens/sec` with `train_compute=0.253295 ms/token`;
route-vote scored `8192` rows before selecting `10`, and state transition
reported `8192` columns with `state_transition_runs_all_columns=true`. The
explicit fallback reason is
`dense_state_transition_retained_until_lazy_column_state`.

| Columns | Tokens/sec | Train compute ms/token | Route rows scored | Awake output | State transition columns | `runs_all_columns` cause | Environment |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1024 | `6152.495` | `0.133843` | `1024` | `10` | `1024` | dense state transition | contention observed |
| 8192 | `3526.002` | `0.253295` | `8192` | `10` | `8192` | dense state transition | not observed |

This closed the 2026-06-15 evidence gap without promoting a fake scheduler
claim: candidate wake was bounded, but total-column scaling was still blocked by
route-score rows and dense column-state transition. The dense column-state half
is superseded by the 2026-06-16 sparse CUDA state-transition promotion above.
The remaining implementation target is a sparse/GPU-owned route-candidate
retrieval boundary so route-score input rows stop scaling with total columns.

### TurboQuant Routing Audit, 2026-06-15

TurboQuant was rechecked against Google Research's TurboQuant paper, current
vLLM/community implementation notes, and MARULHO's scheduler contract. The
paper and community code make TurboQuant useful as vector compression or
approximate inner-product scoring, but not as a wake/sleep scheduler by itself.
MARULHO's removed legacy `turboquant_plus` backend was also not paper-faithful:
it used per-prototype min/max uniform codes, kept FP32 prototypes, applied a
QJL-style correction, and scanned every route row before top-k.

The archived audit before removal was
`python -m marulho.evaluation.routing_backend_audit --n-columns 8192 --dim 64 --k 10 --samples 40 --warmup-steps 5 --seed 20260615 --device auto --output reports/routing_backend_audit_20260615/routing-8192-auto.json`.

| Columns | Backend | Mean route ms | p95 route ms | Top-1 recall vs exact | Exact top-1 in candidates | Route rows scored | Graph route/vote eligible |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | `torch_topk` | `1.034438` | `1.5262` | `1.000` | `1.000` | `1024` | yes |
| 1024 | `turboquant_plus` | `3.280950` | `5.0043` | `0.650` | `1.000` | `1024` | no |
| 8192 | `torch_topk` | `0.822393` | `1.1669` | `1.000` | `1.000` | `8192` | yes |
| 8192 | `turboquant_plus` | `5.502870` | `8.3901` | `0.750` | `1.000` | `8192` | no |

The retained path is still exact torch-cache routing for CUDA route/vote. The
TurboQuant backend was removed after rejection as a scheduler boundary because
it was slower on the measured GPU path, did not preserve exact winner sequence,
could not feed the promoted graph cache interface, kept a full FP32 copy, and
continued to score all columns. A future revisit must be a bounded GPU-owned
candidate router or a paper-faithful compressed scorer embedded under such a
router, with long complete-runtime evidence against the 131072-token 6k-ish
baseline.

After removal, the live 1024-column promoted checkpoint still stayed on the
real CUDA path. The long cleanup check at
`reports/turboquant_removal_20260615/runtime-1024-cleanup-131072-i32.json`
used the same `131072` target tokens, `tick_tokens=128`, q16
conditional-WHILE, and host-truth interval `32`. It completed successfully at
`5690.654 tokens/sec` with `train_compute=0.142132 ms/token`, no observed
contention, CUDA selected on the RTX 3060, `131072` route-vote executions,
`8190` conditional sequence-loop successes over `131040` burst tokens, and zero
sequence/native failures. This is below the best 6k-ish route-vote
sync-cadence run (`6135.026 tokens/sec`, `0.133995 ms/token`) but preserves the
promoted execution path and shows the deletion did not force a fallback or CPU
route.

### Routing Backend Consolidation, 2026-06-16

After the sparse state-transition promotion, cleanup removed the remaining
selectable CPU routing backends: `auto`, `faiss_hnsw`, and `exact_cosine`.
Those branches could produce candidate ids, but they could not expose
`routing_tensor_cache()` for the promoted CUDA route/vote graph, so keeping them
preserved a second non-promoted path. `routing_index_mode` was later removed as
a live config field; focused routing/config tests assert retired backend kwargs
fail fast, and checkpoint load drops old keys with migration evidence.

The follow-up long run used
`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/host_truth_interval_16_20260613/runtime.pt --output reports/column_scheduler_20260616/routing-backend-consolidation-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 300 --sample-interval-seconds 0.5`.
It reached `6152.191 tokens/sec` with `train_compute=0.132471 ms/token`,
`prepare_training=0.006478 ms/token`, `finalize_total=0.006010 ms/token`, and
`tick_duration_ms.p95=21.526`. CUDA selected the RTX 3060, contention was
`not_observed`, graph/sequence/native failures and fallbacks stayed zero, and
Runtime Truth kept `state_transition_mode=candidate_subset_sparse_cuda_graph_route_transition_burst`,
`state_transition_column_count=10`, `state_transition_cached_count=1014`, and
`state_transition_runs_all_columns=false`. Route-score input rows remain `1024`;
this cleanup removes dead selectable backends, not the next sparse route-candidate
retrieval boundary.

A follow-up live-vocabulary cleanup renamed active `hnsw_index` and `_hnsw_*`
surfaces to `routing_index` and `_routing_index_*` so the promoted path no
longer carries the retired backend name. The same long command shape wrote
`reports/column_scheduler_20260616/routing-index-vocabulary-cleanup-131072-i32.json`
and reached `6006.928 tokens/sec` with `train_compute=0.133305 ms/token`,
`prepare_training=0.006519 ms/token`, `finalize_total=0.006295 ms/token`, and
`tick_duration_ms.p95=21.661`. Compared with the backend-consolidation gate,
that is `-145.263 tokens/sec`, `+0.000833 ms/token` in train compute,
`+0.000041 ms/token` in preparation, and `+0.000285 ms/token` in finalize.
CUDA again selected the RTX 3060, contention was `not_observed`, graph,
sequence, and native failures/fallbacks stayed zero, and Runtime Truth kept
`state_transition_column_count=10`, `state_transition_cached_count=1014`, and
`state_transition_runs_all_columns=false`.

The sharded merged-cache cleanup then removed the live config and benchmark
switches that could disable the merged torch cache required by the promoted
CUDA route/vote graph. The same long command shape wrote
`reports/column_scheduler_20260616/sharded-merge-cache-cleanup-131072-i32.json`
and reached `6169.616 tokens/sec` with `train_compute=0.131525 ms/token`,
`prepare_training=0.006390 ms/token`, `finalize_total=0.005854 ms/token`, and
`tick_duration_ms.p95=21.197`. Checkpoint load recorded
`retired_non_promoted_sharded_route_cache_switch` for the old
`merge_torch_routing_shards` key, CUDA selected the RTX 3060, contention was
`not_observed`, graph/sequence/native failures and fallbacks stayed zero, and
Runtime Truth kept `state_transition_column_count=10`,
`state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`,
while route scoring still truthfully reported `route_input_rows_scored=1024`,
`route_rows_run_all_columns=true`, and `bounded_route_scoring=false`.

The routing list-surface cleanup removed the legacy list-returning
`routing_index.search()` API after tensor routing was the only promoted
candidate surface. Query detail projection now converts tensor candidates at
the display edge, hot-window benchmark arguments no longer allow
`routing_candidate_mode=list`, and compiled hot-path probes use
`candidate_source=routing_index` for tensor-native retrieval. This did not
change the promoted CUDA graph route/vote algorithm and does not claim bounded
route-row scoring; it removes a benchmark/control branch that could no longer
feed the real path. The 131072-token stress report at
`reports/column_scheduler_20260616/routing-list-surface-cleanup-131072-i32.json`
reached `6142.710 tokens/sec` with `train_compute=0.132356 ms/token`,
`prepare_training=0.006332 ms/token`, `finalize_total=0.005919 ms/token`, and
`tick_duration_ms.p95=21.142`. CUDA selected the RTX 3060, contention was
`not_observed`, graph/sequence/native failures and fallbacks stayed zero, and
Runtime Truth kept `state_transition_column_count=10`,
`state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`,
while route scoring still truthfully reported `route_input_rows_scored=1024`
and `bounded_route_scoring=false`.

The routing backend selector cleanup then removed the last retrieval-local
backend constructor argument and the private benchmark compatibility branch that
checked `_uses_merged_torch_search`. The follow-up removed `routing_index_mode`
from live config entirely; checkpoint load now drops old keys with
`retired_routing_backend_config_surface` migration evidence, and benchmark
helpers consume the public `routing_tensor_cache()` surface. This did not change
the promoted route-bank algorithm. The matching 8192-column 131072-token stress
report at
`reports/column_scheduler_20260616/routing-backend-selector-cleanup-8192-131072-i32.json`
reached `6134.242 tokens/sec`, `train_compute=0.131117 ms/token`,
`prepare_training=0.006441 ms/token`, `finalize_total=0.005995 ms/token`, and
`tick_duration_ms.p95=21.987`. Runtime Truth reported
`route_input_rows_scored=10/8192`, `state_transition_column_count=10`,
`state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`,
zero graph/sequence/native failures, and `velocity_environment=not_observed`.
The config-field removal gate at
`reports/column_scheduler_20260616/routing-index-mode-removal-8192-131072-i32.json`
reached `6126.128 tokens/sec`, `train_compute=0.129371 ms/token`,
`prepare_training=0.006485 ms/token`, `finalize_total=0.005973 ms/token`, and
`tick_duration_ms.p95=21.830` with the same `10/8192` route rows, `10` active
state-transition columns, `8182` cached columns, zero graph/sequence/native
failures, and checkpoint migrations for both `merge_torch_routing_shards` and
`routing_index_mode`.

The route-scoring truth cleanup adds `route_vote_scoring` to the training-owned
transition report and projects it through Runtime Truth without changing the
route/vote algorithm. Focused tests assert that the then-promoted exact-cache path
keeps bounded specialist execution while still reporting
`route_input_rows_scored=total_columns`, `route_rows_run_all_columns=true`, and
`bounded_route_scoring=false`. This prevents an exact `torch_topk`
`search_tensors()` pre-narrow from being promoted as sparse routing, because it
would already have scored every route row before fused route/vote. The matching
131072-token stress report at
`reports/column_scheduler_20260616/route-scoring-truth-131072-i32.json`
reached `5628.291 tokens/sec` with `train_compute=0.138304 ms/token`,
`prepare_training=0.007313 ms/token`, `finalize_total=0.006800 ms/token`, and
`tick_duration_ms.p95=24.122`. CUDA selected the RTX 3060, but
`velocity_environment.v1` reported `contention_observed` from GPU busy state, so
this is correctness/truth-surface evidence rather than a new speed ceiling.
Runtime Truth showed `route_vote_scoring.route_input_rows_scored=1024`,
`route_output_candidate_count=10`, `route_rows_run_all_columns=true`,
`bounded_route_scoring=false`, `state_transition_column_count=10`,
`state_transition_cached_count=1014`, `state_transition_runs_all_columns=false`,
and zero graph, sequence, or native failures/fallbacks.

### Route Candidate Bank Scheduler, 2026-06-16

The route-candidate-bank slice changes the CUDA route/vote contract from
full-cache route scoring on every eligible text tick to an explicit two-phase
scheduler boundary:

- first eligible tick: exact complete-cache seed,
  `candidate_boundary=exact_full_cache_score_seed_route_bank`,
  `route_scoring_unbounded_reason=route_candidate_bank_not_ready_exact_seed`
- steady fused/graph ticks: indexed bank scoring with
  `route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`
- steady graph/burst replays refresh the next bank/probe positions inside the
  fused route/vote select kernel; host refresh remains for exact seed/restore
- `service` only projects `route_candidate_bank` and `route_vote_scoring`; it
  does not choose the bank, wake columns, or decide sleep

The first complete run,
`reports/column_scheduler_20260616/route-candidate-bank-131072-i32.json`,
proved bounded route rows (`route_input_rows_scored=10`,
`bounded_route_scoring=true`) but reached only `5323.085 tokens/sec` and
`train_compute=0.158101 ms/token`. The outlier was visible in measured train
cost: `train_compute.max=3859.036 ms`, while p95 was `17.919 ms/tick`. That
exposed an implementation-path miss: the exact seed route was not warmed, so
the full-cache seed kernel could JIT compile inside the measured window.

After warming both the exact seed route and the indexed bank route, the 1024
column long promotion gate at
`reports/column_scheduler_20260616/route-candidate-bank-warmseed-131072-i32.json`
returned to the 6k-ish band:

| Metric | Prior list-surface cleanup | First route-bank run | Warm-seed route-bank |
| --- | ---: | ---: | ---: |
| tokens/sec | `6142.710` | `5323.085` | `6109.301` |
| train_compute ms/token | `0.132356` | `0.158101` | `0.130536` |
| prepare_training ms/token | `0.006332` | `0.006173` | `0.006706` |
| finalize_total ms/token | `0.005919` | `0.005740` | `0.005764` |
| tick_duration p95 ms | `21.142` | `20.070` | `21.299` |
| route rows scored | `1024` | `10` | `10` |
| bounded_route_scoring | `false` | `true` | `true` |
| graph/native/sequence failures | `0` | `0` | `0` |

The warm-seed report selected the RTX 3060, completed `131072` tokens, kept
`state_transition_column_count=10`, `state_transition_cached_count=1014`, and
`state_transition_runs_all_columns=false`, and exposed
`route_candidate_bank.enabled=true`, `ready=true`, `bank_size=10`,
`seed_count=1`, `refresh_count=8222`, `graph_bypass_count=1`, and
`fallback_count=1`. Runtime Truth also showed
`host_truth_cadence_tick_count=131072` with `tick_replay_count=131071`, so the
exact seed is counted as a truth cadence tick without pretending it was a graph
replay.

The larger-column scaling gate then reran the same 131072-token real path
against the promoted 8192-column checkpoint:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\route-candidate-bank-8192-warmseed-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

| Metric | Old 8192 exact-cache route-vote | 1024 route-bank warmseed | 8192 route-bank warmseed |
| --- | ---: | ---: | ---: |
| tokens/sec | `3564.222` | `6109.301` | `6110.715` |
| train_compute ms/token | `0.251487` | `0.130536` | `0.135007` |
| prepare_training ms/token | `0.006526` | `0.006706` | `0.006082` |
| finalize_total ms/token | `0.005773` | `0.005764` | `0.005988` |
| tick_duration p95 ms | `36.002` | `21.299` | `20.370` |
| route rows scored | `8192` | `10` | `10` |
| state transition columns | not separately surfaced | `10/1024` | `10/8192` |
| graph/native/sequence failures | `0` | `0` | `0` |
| environment contention | not observed | contention observed | not observed |

The 8192 route-bank report selected the RTX 3060, completed `131072` tokens,
kept the then-current `route_vote_kernel_variant=indexed_route_bank_vote`,
`route_candidate_bank.enabled=true`, `ready=true`, `seed_count=1`,
`fallback_count=1`, `graph_bypass_count=1`, and
`last_reason=bounded_route_bank_burst_refresh`. Runtime Truth reported
`route_input_rows_scored=10`, `route_input_fraction=0.001220703125`,
`route_rows_run_all_columns=false`, `bounded_route_scoring=true`,
`state_transition_column_count=10`, `state_transition_cached_count=8182`, and
`state_transition_runs_all_columns=false`.

This promotes the route-bank path as the current steady scheduler boundary for
the 1024 and 8192 real paths. It does not prove a general ANN router,
wider-bank quality, or growth/pruning autonomy. The remaining scaling work is a
quality/recall gate for larger banks or future GPU-owned routers, plus explicit
growth/pruning budget policy, not another per-tick all-column route scorer.

### Device-Owned Route-Bank Refresh, 2026-06-16

The follow-up scheduler slice moved steady route-bank refresh from Python-side
candidate-to-route-position indexing into the fused route/vote select kernel.
The long gate
`reports/column_scheduler_20260616/device-route-bank-refresh-32768-131072-i32.json`
used the same `32768`-column usefulness scheduler checkpoint as the previous
run and completed `131072` tokens on the RTX 3060. It reported
`route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`,
`refresh_owner=fused_route_vote_device`, `device_refresh_count=131072`,
`host_refresh_count=1`, `route_input_rows_scored=12/32768`,
`route_output_candidate_count=10`, `state_transition_cached_count=32758`,
`state_transition_runs_all_columns=false`, and zero graph/native/sequence
failures. Throughput was `6008.953 tokens/sec`; train compute was
`0.133379 ms/token`; tick p95 was `22.068 ms`. When this device-refresh path
is active, Runtime Truth reports effective `refresh_interval_tokens=1`; the
historical q16 probe cadence is retained only for quality-gate comparisons and
non-device-refresh fallback truth.

The immediately previous same-checkpoint usefulness-filter run,
`reports/column_scheduler_20260616/usefulness-scheduler-32768-131072-i32.json`,
reported `5886.235 tokens/sec`, `0.134212 ms/token`, `tick p95=22.614 ms`,
`route_vote_kernel_variant=indexed_route_bank_vote`, `route_input_rows_scored=12/32768`,
and zero graph/native/sequence failures. The device-refresh run therefore kept
the 6k-ish long path and slightly improved this local before/after despite
observed GPU contention, but it is not a new global speed ceiling.

Quality remains open. Same-checkpoint offline gates over `512` default-text
ticks compared the old q16 probe cadence against per-token device refresh:
q16 probe2 reached top-1-in-bank `0.271484375`, winner match `0.046875`, mean
overlap `0.2025390625`, and worst top-1 miss streak `19`; device refresh
reached top-1-in-bank `0.2734375`, winner match `0.046875`, mean overlap
`0.2078125`, and worst top-1 miss streak `19`. This promotes device ownership
of the existing bounded scheduler refresh, not a quality-complete discovery
router.

### Route Candidate Bank Quality Gate, 2026-06-16

The quality gate compares the training-owned route candidate bank against an
offline complete-cache exact top-k oracle. The oracle work is evaluation-only:
it does not add a hot-path all-column scan and the report keeps steady
`route_candidate_bank` scoring separate from `offline_oracle_score_rows_per_tick`.

| Report | Source | Columns | Bank rows | Exact reseed | Exact top-1 in bank | Winner match | Mean top-k overlap | Worst top-1 miss streak | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `route-candidate-bank-quality-8192-default-text-s512.json` | default text | `8192` | `10` | `0` | `0.009765625` | `0.001953125` | `0.02578125` | `134` | rejected |
| `route-candidate-bank-quality-8192-default-text-s512-reseed32.json` | default text | `8192` | `10` | `15` | `0.486328125` | `0.001953125` | `0.33046875` | `30` | rejected |
| `route-candidate-bank-quality-8192-random-s256.json` | random/stable control | `8192` | `10` | `0` | `1.0` | `1.0` | `0.86875` | `0` | passed control |
| `route-candidate-graph-quality-8192-default-text-s512-neighbor208-cap1536.json` | default text, offline graph-neighbor probe | `8192` | `1481.068` mean, max `1534` | `0` | `0.994140625` | `0.98828125` | `0.9748046875` | `1` | passed offline quality |

The control proves the gate can pass when the route distribution stays local.
The default-text k-only runs prove the current self-refreshing bank can trap the
scheduler in an old relevance neighborhood. The graph-neighbor probe proves a
bounded discovery policy can recover relevance drift in offline evaluation
without adding a runtime all-column oracle, but it is not a promotion by itself.

The 2026-06-17 wider-retained-bank diagnostic fixes the quality gate so the
retained bank can be wider than awake `k` without pretending the current runtime
already supports that refresh shape. These commands all used the restored
8192-column route-bank checkpoint and default text:

`python -m marulho.evaluation.route_candidate_bank_quality_gate --checkpoint reports\column_scheduler_20260616\checkpoints\route-bank-restored-8192-seeded.pt --output reports\column_scheduler_20260617\wider-bank256-probe64-quality-8192-default-text-s512.json --samples 512 --source-mode default_text --bank-size 256 --route-candidate-probe-rows 64 --route-candidate-bank-refresh-interval 1`

`python -m marulho.evaluation.route_candidate_bank_quality_gate --checkpoint reports\column_scheduler_20260616\checkpoints\route-bank-restored-8192-seeded.pt --output reports\column_scheduler_20260617\wider-bank512-probe128-quality-8192-default-text-s512.json --samples 512 --source-mode default_text --bank-size 512 --route-candidate-probe-rows 128 --route-candidate-bank-refresh-interval 1`

`python -m marulho.evaluation.route_candidate_bank_quality_gate --checkpoint reports\column_scheduler_20260616\checkpoints\route-bank-restored-8192-seeded.pt --output reports\column_scheduler_20260617\wider-bank1024-probe256-quality-8192-default-text-s512.json --samples 512 --source-mode default_text --bank-size 1024 --route-candidate-probe-rows 256 --route-candidate-bank-refresh-interval 1`

| Shape | Steady rows | Exact top-1 | Winner match | Mean top-k overlap | Worst miss | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bank256 probe64 | `320/8192` | `0.92578125` | `0.630859375` | `0.8824218749999989` | `8` | rejected |
| bank512 probe128 | `640/8192` | `1.0` | `0.826171875` | `0.9757812499999993` | `0` | rejected |
| bank1024 probe256 | `1280/8192` | `1.0` | `0.97265625` | `0.9949218749999997` | `0` | quality pass, runtime throughput rejected |

The pass is not a runtime promotion. An isolated synchronized route-row lower-bound at
`reports/column_scheduler_20260617/wider-bank-route-rows-fused-lower-bound-8192.json`
measured the current fused route/vote path at `0.236956 ms` mean for `12` rows,
`0.271268 ms` for `320`, `0.319130 ms` for `640`, and `0.476562 ms` for `1280`.
That lower bound predicted risk correctly. A live CUDA attempt added a
GPU-owned top-retained refresh for a `1024` row retained bank plus `256` probe
rows while still waking only `10` candidates:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260617\checkpoints\wide-bank-promoted-scheduler-65536-seeded.pt --report reports\column_scheduler_20260617\wide-bank-promoted-scheduler-65536-checkpoint.json --n-columns 65536 --column-latent-dim 64 --k-routing 10 --seed 20260617 --device cuda`

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\wide-bank-promoted-scheduler-65536-seeded.pt --output reports\column_scheduler_20260617\wide-bank-runtime-65536-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

The runtime truth was real but too expensive: `route_input_rows_scored=1280/65536`,
`route_output_candidate_count=10`, `refresh_owner=fused_route_vote_topk_device`,
`state_transition_cached_count=65526`, zero graph/native/sequence failures, and
no observed contention, but throughput fell to `4656.790 tokens/sec` with
`train_compute=0.186216 ms/token`. The same-session retained k+2 long run at
`reports/column_scheduler_20260617/wider-bank-eval-no-runtime-change-65536-131072-i32.json`
was `6156.500 tokens/sec` with `train_compute=0.130623 ms/token`. A cheap
`sorted=False` top-k variant did not recover the path: the 32768-token check at
`reports/column_scheduler_20260617/wide-bank-runtime-unsorted-topk-65536-32768-i32.json`
measured `4596.633 tokens/sec` and `train_compute=0.176464 ms/token`. The
runtime branch was removed after this gate; the promoted scheduler remains the
k+2 route-bank/probe lane until a different bounded discovery router passes both
winner quality and the 6k-ish long-run cost gate.

The attempted live implementation built a fixed neighbor graph and scored the
bank plus bounded neighbors on the 8192 real path:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\route-candidate-graph-neighbor208-cap1536-8192-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5 --route-candidate-graph-neighbor-count 208 --route-candidate-graph-capacity-rows 1536`

| Metric | Promoted 8192 k-only route bank | Runtime graph-neighbor probe |
| --- | ---: | ---: |
| tokens/sec | `6110.715` | `4682.167` |
| train_compute ms/token | `0.135007` | `0.183927` |
| prepare_training ms/token | `0.006082` | `0.006079` |
| finalize_total ms/token | `0.005988` | `0.005763` |
| tick_duration p95 ms | `20.370` | `30.584` |
| route rows scored | `10/8192` | `1509/8192` |
| graph/native/sequence failures | `0` | `0` |
| environment contention | not observed | not observed |

So the route-neighbor runtime shape is rejected and removed from the live path:
it improved quality, but lost too much of the 6k-ish long-run throughput. A
runtime exact reseed would be an explicit full-cache fallback or maintenance
cadence, not a hidden promotion. The next route scheduler needs a bounded
GPU-owned discovery mechanism, multi-bank exploration policy, or quality-aware
refresh that can find outside the current bank without making every tick score
all columns and without falling below the promoted ms/token baseline.

The post-cleanup longer rerun confirmed the live path still uses the promoted
k-only bank:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\route-candidate-bank-8192-warmseed-131072-i32-after-graph-reject-cleanup.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

That run reached `6150.296 tokens/sec`, `train_compute=0.130390 ms/token`,
`prepare_training=0.006365 ms/token`, `finalize_total=0.005874 ms/token`,
`tick_duration_ms.p95=21.097`, `route_input_rows_scored=10`,
`state_transition_column_count=10`, `state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and `velocity_environment.v1` contention `not_observed`.

The initial promoted follow-up added a fixed two-row route-bank probe lane and
kept refresh at the graph quantum boundary so fused sequential, graph sequential,
and burst execution preserved exact state parity:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\route-bank-probe-lane-8192-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

| Metric | k-only bank cleanup rerun | bounded probe lane |
| --- | ---: | ---: |
| tokens/sec | `6150.296` | `6141.234` |
| train_compute ms/token | `0.130390` | `0.130664` |
| prepare_training ms/token | `0.006365` | `0.006246` |
| finalize_total ms/token | `0.005874` | `0.005932` |
| tick_duration p95 ms | `21.097` | `21.976` |
| route rows scored | `10/8192` | `12/8192` |
| awake/state-transition columns | `10/8192` | `10/8192` |
| graph/native/sequence failures | `0` | `0` |
| environment contention | not observed | not observed |

Runtime Truth for that pre-device-refresh run reported `route_candidate_bank.probe_rows=2`,
`score_rows=12`, `refresh_interval_tokens=16`,
`probe_refresh_count=8192`, `route_input_rows_scored=12`,
`route_rows_run_all_columns=false`, `state_transition_cached_count=8182`, and
`state_transition_runs_all_columns=false`. This keeps the 6k-ish long path while
adding bounded outside-bank discovery; it is not evidence that the route bank is
quality-complete on changing text. The later device-refresh promotion moved the
steady refresh owner into the fused route/vote kernel; active device-refresh
reports use effective `refresh_interval_tokens=1`, while q16 remains the older
quality/evaluation cadence.

The follow-up quality gate updated
`route_candidate_bank_quality_gate` to simulate the promoted fixed probe lane
and the live 16-token bank refresh cadence:

`python -m marulho.evaluation.route_candidate_bank_quality_gate --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\route-bank-probe-lane-quality-8192-default-text-s512.json --samples 512 --source-mode default_text --route-candidate-probe-rows 2 --route-candidate-bank-refresh-interval 16`

| Quality shape | steady rows | exact top-1 in bank | winner match | worst miss streak | status |
| --- | ---: | ---: | ---: | ---: | --- |
| probe2 q16 | `12/8192` | `0.017578125` | `0.001953125` | `134` | bounded, quality incomplete |
| probe64 q16 | `74/8192` | `0.140625` | `0.0078125` | `52` | rejected as quality fix |
| probe256 q16 | `266/8192` | `0.34765625` | `0.044921875` | `63` | rejected as quality fix |
| graph32 cap256 q16 | `256/8192` | `0.640625` | `0.1484375` | `16` | quality incomplete |
| graph64 cap512 q16 | `511.906/8192` mean | `0.796875` | `0.26171875` | `8` | quality incomplete |
| graph128 cap1024 q16 | `985.961/8192` mean | `0.900390625` | `0.484375` | `5` | quality incomplete |
| graph208 cap1536 q16 | `1476.677/8192` mean | `0.9453125` | `0.74609375` | `4` | still below promotion gate |

The 32768-column same-checkpoint diagnostic then added
`exact_winner_in_bank_rate` and
`bank_candidates_with_exact_previous_winner_match_rate` to separate missing
candidate discovery from previous-winner/reference-frame drift:

| Quality shape | steady rows | exact top-1 in bank | exact winner in bank | winner match | exact-previous diagnostic | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| probe2 per-token | `12/32768` | `0.2734375` | `0.09375` | `0.046875` | `0.087890625` | quality incomplete |
| probe16 per-token | `26/32768` | `0.279296875` | `0.095703125` | `0.046875` | `0.08984375` | rejected as quality fix |
| probe64 per-token | `74/32768` | `0.591796875` | `0.486328125` | `0.046875` | `0.455078125` | rejected as quality fix |

Those reports are
`reports/column_scheduler_20260616/route-bank-probe2-device-diagnostic-quality-32768-default-text-s512.json`,
`reports/column_scheduler_20260616/route-bank-probe16-device-diagnostic-quality-32768-default-text-s512.json`,
and
`reports/column_scheduler_20260616/route-bank-probe64-device-diagnostic-quality-32768-default-text-s512.json`.
They reject blind probe widening more strongly than top-1 alone: even when
probe64 puts the exact winner into the bounded bank on nearly half the ticks,
the live previous-winner path still matches only `0.046875`, and the oracle
previous-winner diagnostic is still below `0.95`.

The same gate also tested an evaluation-only column-ID hypercube-neighbor lane:
score the current route bank, the fixed probe rows, and a bounded set of bit-flip
neighbors of the runtime previous winner. A focused toy fixture proves this can
recover a deliberately local bit-flip shift, but the real 32768-column checkpoint
showed that current column IDs are not routing locality. With
`--route-candidate-hypercube-neighbor-rows 16`, the default-text gate at
`reports/column_scheduler_20260616/route-bank-hypercube-neighbor-quality-32768-default-text-s512.json`
reported `25.215/32768` mean steady route rows, exact top-1 `0.2734375`,
exact-winner-in-bank `0.09375`, winner match `0.046875`, and oracle-previous
diagnostic `0.08984375`, with
`promotion_status=hypercube_neighbors_bounded_but_requires_stronger_discovery_router_before_quality_claim`.
This rejects column-ID bit flips as a scheduler promotion, not the bounded route
bank itself.

Because this diagnostic changed only the offline quality gate and docs, the
runtime path was rerun unchanged against the same 32768-column seeded
checkpoint:
`reports/column_scheduler_20260616/probe-diagnostic-no-runtime-change-32768-131072-i32.json`.
It completed `131072` tokens at `6148.022 tokens/sec` with
`train_compute=0.132288 ms/token`, `prepare_training=0.006419 ms/token`,
`finalize_total=0.005910 ms/token`, `tick_duration_ms.p95=21.421`,
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and no observed contention. This verifies
the diagnostic/rejection cycle did not move the real scheduler path or weaken
the 6k-ish baseline.

After adding the hypercube-neighbor rejection evidence, the same real path was
checked again at
`reports/column_scheduler_20260616/route-bank-hypercube-neighbor-rejection-32768-131072-i32.json`.
It processed `131072` tokens at `6149.283 tokens/sec` with
`train_compute=0.131785 ms/token`, `prepare_training=0.006355 ms/token`,
`finalize_total=0.005877 ms/token`, `tick_duration_ms.p95=21.274`,
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and `velocity_environment.v1` contention
observed. The promoted path still scores the route bank plus fixed probe lane;
the hypercube-neighbor probe is not used in runtime.

The next probe tested a CAGRA-style bounded graph walk in the same evaluation
gate. It scores a bounded frontier from the current bank plus probe lane, keeps
a fixed beam as graph-walk parents, and expands neighbor frontiers for fixed
rounds. This is still evaluation-only: the full routing cache is used for the
offline oracle and graph precompute, not as a hot-path scheduler.

| Graph-walk shape | steady rows | exact top-1 | winner match | mean top-k overlap | worst miss | status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| neighbor64 beam4 round4 cap512 | `512/8192` | `0.896484375` | `0.263671875` | `0.8371093749999988` | `5` | rejected |
| neighbor64 beam2 round6 cap512 | `512/8192` | `0.912109375` | `0.259765625` | `0.8609374999999994` | `3` | rejected |
| neighbor128 beam8 round4 cap1024 | `1024/8192` | `0.98046875` | `0.70703125` | `0.9400390624999977` | `2` | rejected |
| neighbor128 beam2 round6 cap1024 | `1024/8192` | `1.0` | `0.73046875` | `0.9660156249999977` | `0` | rejected |
| neighbor208 beam12 round4 cap1536 | `1536/8192` | `0.9609375` | `0.822265625` | `0.9486328124999986` | `4` | rejected |
| neighbor256 beam2 round6 cap1536 | `1536/8192` | `1.0` | `0.767578125` | `0.9912109374999994` | `0` | rejected |

The strict pass criteria remain exact top-1 and winner match at least `0.95`
with worst exact top-1 miss streak at most `2`. The earlier graph-neighbor
quality pass assumed per-tick refresh and did not survive the live q16 refresh
cadence, and graph walking shows why exact top-1 is not enough: the
previous-winner/location-sensitive vote needs the supporting candidate set. No
graph-walk runtime gate was run because every tested shape failed winner parity
before promotion. A same-session real-path recheck kept the promoted runtime
unchanged:
`reports/column_scheduler_20260616/graph-walk-eval-no-runtime-change-8192-131072-i32.json`
reached `6141.295 tokens/sec`, `train_compute=0.132462 ms/token`,
`prepare_training=0.006577 ms/token`, `finalize_total=0.006070 ms/token`,
`tick_duration_ms.p95=22.216`, `route_input_rows_scored=12/8192`,
`state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and `velocity_environment.v1` contention `contention_observed`, so it is
in-band evidence rather than a new top-speed claim. The next promotable router
needs a fused/GPU-owned discovery mechanism or graph-ordered device refresh
that can preserve burst parity, recover relevance, and keep the 131072-token
6k-ish path.

The retained fallback vote-scope cleanup prevents a context-gain selection
fallback from becoming a hidden all-column predictive vote. The focused CUDA
test forces the fallback and verifies `awake_mask_cached_vote`,
`updated_column_count=4`, `cached_vote_use_count=12`, and
`runs_all_columns=false`. The longer 8192-column real-path check at
`reports/column_scheduler_20260616/fallback-candidate-vote-scope-8192-131072-i32.json`
processed `131072` tokens at `6016.247 tokens/sec` with
`train_compute=0.1330716 ms/token`, `prepare_training=0.006573 ms/token`,
`finalize_total=0.006077 ms/token`, `tick_duration_ms.p95=22.786`,
`route_input_rows_scored=12/8192`, `route_output_candidate_count=10`,
`state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. Compared with the previous in-band
`graph-walk-eval-no-runtime-change-8192-131072-i32.json` result
(`6141.295 tokens/sec`, `0.132462 ms/token` train compute, GPU contention
observed), this is neutral-ish same-band evidence rather than a speed
promotion; the value is that fallback truth now preserves the bounded vote
contract.

The delayed predictive-gate graph split removes the old CUDA fallback that
disabled fused candidate predictive transition when
`candidate_predictive_update_start_tokens` was later than
`candidate_homeostasis_start_tokens`. The focused CUDA parity test captures
both `candidate_subset_dense_predictive` and `candidate_subset`: before the
predictive gate, prediction update reports all-column dense truth; after the
gate, it reports candidate-subset updates and matches the retained path's
winner and predictive tensors. The current promoted checkpoint has aligned
gates, so the 131072-token real-path run at
`reports/column_scheduler_20260616/predictive-gate-graph-split-8192-131072-i32.json`
still captured only `candidate_subset` and remained in-band at
`6135.996 tokens/sec`, `train_compute=0.131881 ms/token`,
`prepare_training=0.006492 ms/token`, `finalize_total=0.006019 ms/token`,
`tick_duration_ms.p95=21.566`, `route_input_rows_scored=12/8192`,
`route_output_candidate_count=10`, `state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention.

The predictive dense-transition selector cleanup removed `fused_eager` as a
production config value while keeping dense eager tensor semantics as the
explicit internal fallback when the promoted in-place runtime cannot start.
Focused tests now reject `fused_eager` in new config and migrate old
revision-stamped checkpoints to `inplace_triton`. The longer real-path check at
`reports/column_scheduler_20260616/predictive-dense-mode-selector-cleanup-8192-131072-i32.json`
processed `131072` tokens at `6174.224 tokens/sec`,
`train_compute=0.131907 ms/token`, `prepare_training=0.006647 ms/token`,
`finalize_total=0.006016 ms/token`, `tick_duration_ms.p95=22.558`,
`route_input_rows_scored=12/8192`, `route_output_candidate_count=10`,
`state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. This is accepted as one-path cleanup evidence, not a
new speed ceiling.

A follow-up cleanup removed the remaining internal
`PredictiveColumnState.apply_dense_transition(..., transition_mode=...)`
selector and renamed the fallback Runtime Truth mode from `fused_eager` to
`dense_eager_fallback`. Dense eager tensor semantics remain available only as
fallback/oracle behavior when the promoted in-place runtime cannot start, not as
a selectable runtime path. The 32768-column longer real-path check at
`reports/column_scheduler_20260616/dense-fallback-selector-cleanup-32768-131072-i32.json`
processed `131072` tokens at `6011.640 tokens/sec`,
`train_compute=0.132767 ms/token`, `prepare_training=0.006491 ms/token`,
`finalize_total=0.006139 ms/token`, and `tick_duration_ms.p95=22.446`.
Runtime Truth stayed on the promoted CUDA path:
`route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`,
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_column_count=10`, `state_transition_cached_count=32758`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention.

The route-bank checkpoint restore cleanup removed the first live exact-cache
seed after a checkpoint already has a ready route bank. A seeded checkpoint was
created from the promoted 8192-column scheduler checkpoint by paying the exact
seed before save; restore then loaded the saved bank IDs before CUDA graph
capture. The 131072-token run at
`reports/column_scheduler_20260616/route-bank-checkpoint-restore-8192-131072-i32.json`
reported `checkpoint_restore_count=1`, `seed_count=0`, `fallback_count=0`,
`graph_bypass_count=0`, `route_input_rows_scored=12/8192`,
`route_output_candidate_count=10`, `state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. Throughput stayed in band at
`6129.693 tokens/sec`, `train_compute=0.130218 ms/token`,
`prepare_training=0.006547 ms/token`, `finalize_total=0.005899 ms/token`, and
`tick_duration_ms.p95=21.637`. This is restore/startup scheduler evidence, not
a new route-quality claim.

The route-bank size selector cleanup removed `route_candidate_bank_size` from
production config so the promoted runtime has one live bank-capacity path:
`ColumnTransitionRuntime` derives the route bank from `k_routing`, old
checkpoints drop the retired key with `retired_route_candidate_bank_size_selector`
migration evidence, and wider-bank work remains explicit evaluation-only
input. The longer real-path check at
`reports/column_scheduler_20260616/route-bank-size-selector-removal-8192-131072-i32.json`
processed `131072` tokens at `6281.314 tokens/sec`,
`train_compute=0.129934 ms/token`, `prepare_training=0.006244 ms/token`,
`finalize_total=0.005841 ms/token`, and `tick_duration_ms.p95=21.686`, with
`route_input_rows_scored=12/8192`, `route_output_candidate_count=10`,
`state_transition_cached_count=8182`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
no observed contention, and checkpoint migration evidence for the retired
selector. This is one-path cleanup evidence, not a new quality or throughput
claim.

The promoted scheduler checkpoint builder makes larger-column scale gates
reproducible instead of relying on a hand-built checkpoint artifact:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-16384-seeded.pt --report reports\column_scheduler_20260616\promoted-scheduler-16384-checkpoint.json --n-columns 16384 --column-latent-dim 64 --k-routing 10 --seed 20260616 --device cuda`

The builder pays the explicit full-cache route-bank seed before save, verifies
restore, and disables micro/deep sleep maintenance for the scale gate so the
run measures the promoted scheduler path rather than sleep replay. Its builder
report showed the seed scored `16384/16384`; the restored first tick scored
`12/16384`, output `10` candidates, cached `16374` state-transition columns,
and kept `state_transition_runs_all_columns=false`.

The first 16384-column long-run attempt used an earlier synthetic checkpoint
with micro/deep sleep intervals left at defaults. It failed as scheduler
evidence before the corrected rerun overwrote the output path: the partial
report reached only `2499/131072` tokens before timeout, reported repeated
`sleep_boundary` text-burst fallbacks, and measured `6.935 tokens/sec` overall.
The bounded route/state truth was still intact (`route_input_rows_scored=12`,
`state_transition_cached_count=16374`), but the run measured sleep-maintenance
interruption, not the promoted scale path. The builder now guards against this
by setting `micro_sleep_interval_tokens=1e9` and
`deep_sleep_interval_tokens=1e9`, matching the 8192 promoted checkpoint shape.

The corrected 16384-column longer gate then used the same command and output
path after rebuilding the checkpoint:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-16384-seeded.pt --output reports\column_scheduler_20260616\promoted-scheduler-16384-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 360 --sample-interval-seconds 0.5`

It processed `131072` tokens at `6154.503 tokens/sec`,
`train_compute=0.130874 ms/token`, `prepare_training=0.006364 ms/token`,
`finalize_total=0.005948 ms/token`, and `tick_duration_ms.p95=21.419`, with
`route_input_rows_scored=12/16384`, `route_output_candidate_count=10`,
`state_transition_cached_count=16374`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. This is stronger scaling evidence for the promoted
route-bank/probe-lane scheduler: total columns doubled from 8192 to 16384
while scored route rows, awake/state-transition columns, and ms/token stayed in
the same 6k-ish band.

The same builder and long gate were then extended to `32768` columns:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-32768-seeded.pt --report reports\column_scheduler_20260616\promoted-scheduler-32768-checkpoint.json --n-columns 32768 --column-latent-dim 64 --k-routing 10 --seed 20260616 --device cuda`

The builder report kept the warm-up honest: the explicit seed tick scored
`32768/32768`, while the restored bounded tick scored `12/32768`, output `10`
candidates, cached `32758` state-transition columns, and reported
`state_transition_runs_all_columns=false`.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-32768-seeded.pt --output reports\column_scheduler_20260616\promoted-scheduler-32768-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 420 --sample-interval-seconds 0.5`

It processed `131072` tokens at `6298.380 tokens/sec`,
`train_compute=0.130618 ms/token`, `prepare_training=0.006178 ms/token`,
`finalize_total=0.005816 ms/token`, and `tick_duration_ms.p95=20.991`, with
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. This strengthens the scheduler-boundary claim:
total columns grew from 8192 to 32768, but steady route scoring stayed at
`12` rows, awake mutation stayed at `10` rows, and complete runtime stayed in
the same 6k-ish band.

The next power-of-two scale gate extended the same promoted
route-bank/probe-lane/device-refresh path to `65536` columns:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260617\checkpoints\promoted-scheduler-65536-seeded.pt --report reports\column_scheduler_20260617\promoted-scheduler-65536-checkpoint.json --n-columns 65536 --column-latent-dim 64 --k-routing 10 --seed 20260617 --device cuda`

The builder report again kept the seed visible: the pre-save exact seed scored
`65536/65536`, while the restored bounded tick scored `12/65536`, output `10`
candidates, cached `65526` state-transition columns, restored the bank from the
checkpoint, and reported `state_transition_runs_all_columns=false`.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\promoted-scheduler-65536-seeded.pt --output reports\column_scheduler_20260617\promoted-scheduler-65536-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `131072` tokens at `6154.501 tokens/sec`,
`train_compute=0.130339 ms/token`, `prepare_training=0.006100 ms/token`,
`finalize_total=0.006153 ms/token`, and `tick_duration_ms.p95=20.092`, with
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. Runtime Truth reported
`route_vote_kernel_variant=indexed_route_bank_vote_device_refresh`,
`refresh_owner=fused_route_vote_device`, `device_refresh_count=131072`,
`host_refresh_count=1`, `route_rows_run_all_columns=false`, and
`bounded_route_scoring=true`. This keeps complete-runtime throughput in the
same 6k-ish band while total columns double again from `32768` to `65536`; the
evidence is scheduler-scale evidence, not a relevance-quality pass.

The active pressure gate then proved the route-vote scheduler filter can change
execution on the same 65536-column path without widening route rows or adding a
service-owned decision:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --report reports\column_scheduler_20260617\active-pressure-scheduler-65536-checkpoint.json --n-columns 65536 --column-latent-dim 64 --k-routing 10 --seed 20260617 --device cuda --active-pressure-filter-count 2 --candidate-memory-pressure-filter-start-tokens 0`

The checkpoint fixture marked exactly two cached route-bank rows as high memory
pressure, matching the two probe rows so route-vote could still emit `k=10`
awake candidates without fallback. The restored tick scored `12/65536`, output
`10`, cached `65526`, observed `filtered_memory_pressure_count=2`,
`observed_filtered_memory_pressure_total=2`, and kept
`state_transition_runs_all_columns=false`.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\column_scheduler_20260617\active-pressure-scheduler-65536-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

The long run processed `131072` tokens at `6297.455 tokens/sec` with
`train_compute=0.130524 ms/token`, `prepare_training=0.006053 ms/token`,
`finalize_total=0.006145 ms/token`, and `tick_duration_ms.p95=20.115`.
Runtime Truth reported `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`,
`observed_filtered_memory_pressure_total=2`,
`observed_filtered_deep_sleep_total=254`, `observed_fallback_count=0`,
`route_rows_run_all_columns=false`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and `contention.verdict=not_observed`.
This is positive scheduler-boundary evidence: the pressure/sleep mask acted
inside the fused route-vote owner while complete-runtime cost stayed neutral to
slightly better than the same-session `6156.500` retained k+2 reference and the
plain 65536 scale gate.

The bounded replay-window slice reused that same active-pressure checkpoint to
prove live-tick protection after adding slow-path replay selection, bounded
stored-experience recall, reconstruction-gated candidate repair, retiring
zero-pressure replay, and blocking unanchored deep-sleep global replay mutation:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-candidate-repair.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.25 --host-truth-sync-interval-tokens 32`

The latest run processed `262144` tokens at `6306.507 tokens/sec` with
`train_compute=0.129511 ms/token`, `prepare_training=0.006408 ms/token`,
`finalize_total=0.006255 ms/token`, and `tick_duration_ms.p95=20.176`.
Runtime Truth stayed on CUDA RTX 3060 with `contention.verdict=not_observed`,
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, `state_transition_runs_all_columns=false`,
`observed_filtered_memory_pressure_total=2`, `observed_filtered_deep_sleep_total=510`,
zero graph/native/sequence failures, and `slow_memory_cadence_execution_gate=false`.
CPU max was `18%`, GPU utilization max `11%`, GPU memory utilization max `11%`,
and GPU memory stayed at `1871 MiB` before/after measurement. This is hot-path
protection evidence only: replay-window selection, recall, and candidate repair
remain inside explicit sleep/replay maintenance and do not run in the live tick.

The repair-scope cleanup then bounded emergency repair replay to the same
anchor-bucket window and blocked no-anchor repair mutation. The current-tree
hot-path rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-131072-i32-bounded-repair.json`
processed `131072` tokens at `6252.073 tokens/sec` with
`train_compute=0.129794 ms/token`, `prepare_training=0.006361 ms/token`,
`finalize_total=0.006213 ms/token`, `tick_duration_ms.p95=20.060`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, no observed contention, CPU max `47%`,
GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU memory
`1822 MiB` before/after measurement. This is not a new speed ceiling; it proves
the no-anchor repair retirement does not tax the live tick.

The micro-maintenance cleanup then bounded micro refresh to the same
anchor-bucket window, blocked no-anchor micro refresh, and removed the old
zero-LR `CompetitiveColumnLayer.process(...)` call from micro sleep. The longer
current-tree rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-bounded-micro.json`
processed `262144` tokens at `6332.439 tokens/sec` with
`train_compute=0.129001 ms/token`, `prepare_training=0.006330 ms/token`,
`finalize_total=0.006198 ms/token`, `tick_duration_ms.p95=20.048`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, no observed contention, CPU max `28%`,
GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU memory
`1881 MiB` before/after measurement. The live tick remains protected; micro
maintenance is now a bounded slow-window CPU metadata refresh.

The column structural-review queue first tried to capture candidate evidence on
every CUDA host-truth boundary. That was rejected by the longer real-path run at
`reports/column_scheduler_20260616/structural-review-queue-8192-131072-i32.json`:
it still reported bounded route/state truth, but throughput fell to
`4788.708 tokens/sec`, `train_compute=0.180271 ms/token`, and
`tick_duration_ms.p95=30.495` despite no observed contention. The retained
implementation cadences CUDA structural-review capture at `4096` tokens and
marks intervening bursts as deferred review capture. The rerun at
`reports/column_scheduler_20260616/structural-review-queue-cadenced-8192-131072-i32.json`
processed `131072` tokens at `6312.338 tokens/sec`,
`train_compute=0.129495 ms/token`, `prepare_training=0.006171 ms/token`,
`finalize_total=0.005776 ms/token`, and `tick_duration_ms.p95=21.155`, with
`route_input_rows_scored=12/8192`, `route_output_candidate_count=10`,
`state_transition_cached_count=8182`, `state_transition_runs_all_columns=false`,
zero graph/native/sequence failures, and no observed contention. This proves
the queue can remain checkpoint-backed and operator-reviewable without becoming
a hot-path structural-review CPU sync; it is not a growth/prune mutation claim
or a new top-speed ceiling.

A follow-up scheduler-benchmark audit made the structural-review queue fields
part of the retained CPU A/B report and added
`--force-structural-review-evidence` so the benchmark can prove bounded ticket
capture instead of only empty queue status. The current forced 8192-column report
`reports/column_scheduler_20260616/cpu-8192-structural-review-growth-prune-queue.json`
preserved winner parity, kept bounded specialist work true, and queued both
growth and prune/sleep review tickets from the wake plan. The capture scope is
`post_measurement_bounded_wake_plan_ticket_audit_not_timed`, so the forced audit
is coverage evidence, not measured live throughput. The scoped arm evaluated
`10/8192` columns, cached `8182`, queued `10` growth tickets and `11`
prune/sleep tickets, recorded `10` growth candidates and `10` prune/sleep
candidates, reported `checkpoint_backed=true`, `requires_operator_review=true`,
`mutates_runtime_state=false`, `runs_all_columns=false`, update mode
`benchmark_forced_bounded_structural_review_evidence`, and next gate
`operator_review_column_structural_ticket`. It still did not pass the CPU cost
gate: all-column mean/median/p95 complete `train_step` was
`7.11954625/6.6511/12.3495 ms`, while the scoped forced arm was
`12.20728125/9.28945/32.3889 ms`
(`neutral_or_better_complete_tick=false`). Treat this as truth/coverage evidence
for the scheduler boundary, not a CPU speed promotion.

The current longer real-path recheck on the 32768-column promoted checkpoint,
`reports/column_scheduler_20260616/structural-review-growth-prune-benchmark-only-audit-32768-131072-i32.json`,
processed `131072` tokens at `6314.030 tokens/sec` with
`train_compute=0.128313 ms/token`, `prepare_training=0.005927 ms/token`,
`finalize_total=0.005469 ms/token`, and `tick_duration_ms.p95=19.470 ms`.
It kept route scoring bounded at `12/32768`, output `10` candidates, cached
`32758` state-transition rows, reported `state_transition_runs_all_columns=false`,
used the promoted device-refresh route bank with effective
`refresh_interval_tokens=1`, and had zero graph/native/sequence failures with
no observed contention. This recheck keeps the runtime in the maintained
6k-ish band while the forced structural-review benchmark remains coverage-only.

The follow-up cheap-discovery probe measured fixed landmark and
random-projection buckets. Those probes were evaluation-only: offline precompute
could inspect the full routing cache, but the simulated hot path reported
bounded selector rows and route rows explicitly. After the rejection below, the
`marulho.evaluation.route_candidate_discovery_probe` implementation and tests
were deleted so the repo keeps one promoted scheduler path plus the retained
route-bank quality oracle.

| Probe shape | refresh | steady route rows | selector rows | exact top-1 | winner match | worst miss | status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| landmark256 top8 bucket128 | q16 | `926.544/8192` mean | `256` per refresh (`16.0` amortized/tick) | `0.77734375` | `0.4296875` | `7` | rejected |
| random-projection512 top32 bucket64 | q16 | `1790.795/8192` mean | `512` per refresh (`32.0` amortized/tick) | `0.255859375` | `0.23046875` | `26` | rejected |
| landmark256 top8 bucket128 | q1 | `924.920/8192` mean | `256` per tick | `0.884765625` | `0.525390625` | `3` | rejected |
| landmark512 top16 bucket128 | q1 | `1642.932/8192` mean | `512` per tick | `1.0` | `0.91015625` | `0` | rejected |

These shapes do not justify a runtime path. The best landmark probe recovers
exact top-1 but still fails winner parity while paying a larger row budget than
the rejected graph-neighbor runtime. The random-projection bucket is worse than
the route-bank probe lane despite far more rows. The next candidate must be a
fused/GPU-owned graph or ANN discovery boundary that beats this quality/cost
frontier and then survives the 131072-token real-path run.

The cleanup removed the rejected `route_candidate_discovery_probe` module and
its tests rather than keeping a non-promoted scheduler variant in source. The
post-cleanup 32768-column real-path validation:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-32768-seeded.pt --output reports\column_scheduler_20260616\promoted-scheduler-32768-131072-i32-after-discovery-cleanup.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 420 --sample-interval-seconds 0.5`

processed `131072` tokens at `6128.457 tokens/sec`,
`train_compute=0.132636 ms/token`, `prepare_training=0.006447 ms/token`,
`finalize_total=0.005978 ms/token`, and `tick_duration_ms.p95=22.109`, with
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence failures,
and no observed contention. This validates the deletion as one-path cleanup,
not a new relevance-quality claim.

The follow-up Runtime Truth counter fix separated last-transition cached rows
from cumulative row-skips across the same promoted burst/sequence path:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260616\checkpoints\promoted-scheduler-32768-seeded.pt --output reports\column_scheduler_20260616\promoted-scheduler-32768-131072-i32-truth-counts-fixed.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 420 --sample-interval-seconds 0.5`

The fixed run processed `131072` tokens at `6163.265 tokens/sec`,
`train_compute=0.132789 ms/token`, `prepare_training=0.006285 ms/token`,
`finalize_total=0.005966 ms/token`, and `tick_duration_ms.p95=21.827`, with
`route_input_rows_scored=12/32768`, `route_output_candidate_count=10`,
`state_transition_cached_count=32758`,
`candidate_predictive_transition_cached_count=32758`,
`candidate_predictive_transition_cached_count_scope=last_transition`,
`candidate_predictive_transition_cached_total_count=4293656576`,
`candidate_predictive_transition_cached_total_scope=cumulative_row_skips`, and
zero graph/native/sequence failures. The slow-path velocity surface reported
`contention.verdict=contention_observed` because GPU utilization crossed its
background threshold, but the complete-runtime throughput stayed in the same
6k-ish band.

### Wake-Plan-Scoped Awake Ripple Tagging, 2026-06-16

Awake-ripple tagging is a replay-priority metabolism path, not a CUDA route
kernel. The cleanup moved it from a global recent-memory scan on scheduler-owned
ticks to the same wake-plan bucket IDs already chosen by training. The memory
ledger remains CPU archival storage; the execution effect is that retained
`train_step` and CUDA text-burst flushing no longer scan unrelated sleeping
memory buckets when they already have an awake mask.

The isolated micro-benchmark command was:

`python -m marulho.evaluation.awake_ripple_scope_benchmark --output reports\column_scheduler_20260616\awake-ripple-scope-8192-i256.json --capacity 8192 --bucket-count 8192 --awake-bucket-count 10 --iterations 256 --dim 16`

| Metric | Global recent-memory scan | Wake-bucket scoped |
| --- | ---: | ---: |
| mean tag time | `1.1831957031063212 ms` | `0.9895980468854759 ms` |
| speedup | baseline | `1.1956326175361276x` |
| scalar/vector scans | global vector path | `0` |
| awake-bucket index scans | `0` | `256` |
| last candidate entries touched | all recent entries | `10` |

The scoped path passed all benchmark gates:
`scoped_avoids_global_memory_scan=true`,
`scoped_candidate_count_bounded=true`, and
`scoped_not_slower_than_global=true`.

The longer real-path check reused the 8192-column promoted scheduler
checkpoint and the same 131072-token shape as the 6k-ish baseline:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\real_path_column_scaling_20260615\checkpoints\runtime-8192-promoted-scheduler.pt --output reports\column_scheduler_20260616\awake-ripple-scope-8192-131072-i32.json --target-tokens 131072 --tick-tokens 128 --quantum-tokens 16 --host-truth-sync-interval-tokens 32 --timeout-seconds 900 --sample-interval-seconds 0.5`

| Metric | Post graph-reject cleanup | Wake-ripple scoped cleanup |
| --- | ---: | ---: |
| tokens/sec | `6150.296` | `6286.386` |
| train_compute ms/token | `0.130390` | `0.130081` |
| prepare_training ms/token | `0.006365` | `0.006330` |
| finalize_total ms/token | `0.005874` | `0.005830` |
| tick_duration p95 ms | `21.097` | `21.009` |
| route rows scored | `10/8192` | `10/8192` |
| state transition columns | `10/8192` | `10/8192` |
| graph/native/sequence failures | `0` | `0` |
| environment contention | not observed | not observed |

This is accepted as a true scheduler-consumer cleanup: the memory/replay helper
uses the training-owned wake mask and the long CUDA path stayed neutral or
better. It is not a new route-discovery solution and does not move archival
memory storage to GPU.

### Bounded Awake Ripple Candidate Window, 2026-06-17

The follow-up retires the production unscoped awake-ripple scan instead of
leaving it as a fallback. `ripple_tag_awake(...)` now returns an empty
`bounded_awake_ripple_tag.v1` report when awake bucket scope is absent, and the
old scalar/vector global scan has since been moved out of `DualMemoryStore`
into benchmark-local retired baseline code. When scope exists, the store
collects a recent round-robin candidate window from awake buckets before
mutating ripple/capture tags.

The isolated benchmark command was:

`python -m marulho.evaluation.awake_ripple_scope_benchmark --output reports\bounded_replay_window_20260617\awake-ripple-bounded-scope-8192-i256.json --capacity 8192 --bucket-count 8192 --awake-bucket-count 10 --iterations 256 --dim 16`

| Metric | Diagnostic global scan | Wake-bucket candidate window |
| --- | ---: | ---: |
| mean tag time | `1.433332 ms` | `1.091997 ms` |
| speedup | baseline | `1.312579x` |
| scalar/vector scans | `256` vector scans | `0` |
| awake-bucket index scans | `0` | `256` |
| last candidate entries touched | all recent entries | `10` |

The scoped path passed all benchmark gates:
`scoped_avoids_global_memory_scan=true`,
`scoped_candidate_count_bounded=true`, and
`scoped_not_slower_than_global=true`.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-524288-i32-awake-ripple-bounded-scope.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6152.328 tokens/sec`, with
`train_compute=0.131727 ms/token`, `prepare_training=0.006691 ms/token`,
`finalize_total=0.006526 ms/token`, and `tick_duration_ms.p95=20.949`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `50%`, GPU utilization max `16%`, GPU memory
utilization max `13%`, and GPU memory stayed flat at `2013 MiB` before and
after measurement.

### Reconstruction-Guarded Replay Consolidation, 2026-06-17

The replay/consolidation slice added a slow-window quality guard, not live-tick
work. The guarded HF consolidation report at
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation/summary.json`
kept bounded stored-experience recall passing over `3` Task-A anchor-window
queries with `mean_input_pattern_distance=0.0`, but rejected all `9` attempted
candidate repair updates across `3` post-Task-B replay cycles because Task-A
`mean_reconstruction_error` would regress. The guard restored model and memory
state after each rejected cycle, left effective replay updates at `0`, and
recorded `runs_live_tick=false`.

The matching current-tree hot-path check reused the 65536-column
active-pressure checkpoint:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-guarded-consolidation.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.25 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6606.251 tokens/sec`, with
`train_compute=0.123393 ms/token`, `prepare_training=0.005897 ms/token`,
`finalize_total=0.005707 ms/token`, and `tick_duration_ms.p95=18.562`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`, with zero graph/native/sequence
failures. The run reported no observed contention: CPU max `6%`, GPU utilization
max `10%`, GPU memory utilization max `10%`, and GPU memory flat at `1539 MiB`
before/after. This protects the live tick while replay acceptance remains an
explicit slow-path window.

The follow-up cadence cleanup skips repeated identical rejected replay
selections inside the same slow window. The HF report
`reports/bounded_replay_window_20260617/hf-recall-guarded-consolidation-cadenced/summary.json`
reduced attempted post-Task-B replay repairs from `9` to `3`, skipped `2`
repeated rejected cycles, kept effective updates at `0`, and kept both bounded
recall and the memory-consolidation gate passing. Guard latency fell from
`1003.442 ms` in the previous guarded report to `559.694 ms`.

The first matching 262144-token live-path run,
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced.json`,
completed at only `5388.450 tokens/sec` with `train_compute=0.152640 ms/token`
and `tick_duration_ms.p95=43.215`; it is not accepted as throughput evidence.
The immediate rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-guarded-consolidation-cadenced-rerun.json`
stayed in the maintained band at `6199.988 tokens/sec`, with
`train_compute=0.130574 ms/token`, `prepare_training=0.006633 ms/token`,
`finalize_total=0.006399 ms/token`, `tick_duration_ms.p95=20.215`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`,
zero graph/native/sequence failures, and no observed contention. GPU memory was
`1688 MiB` before and `1689 MiB` after the accepted rerun.

The target-aware replay-strength follow-up does not add live-tick replay work:
strength trials run only inside `reconstruction_guarded_replay_consolidation.v1`
slow windows. The matching promoted hot-path check first reached
`6205.925 tokens/sec` at
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-target-strength-guard-promoted.json`,
but that run recorded a pre-run GPU utilization sample above the contention
threshold. The clean rerun
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-target-strength-guard-promoted-rerun.json`
processed `262144` tokens at `6073.263 tokens/sec`, with
`train_compute=0.133405 ms/token`, `prepare_training=0.006616 ms/token`,
`finalize_total=0.006505 ms/token`, `tick_duration_ms.p95=21.118`,
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`,
`state_transition_runs_all_columns=false`, zero graph/native/sequence
failures, and no observed contention. CPU max was `20%`, GPU utilization max
`10%`, GPU memory utilization max `10%`, and GPU memory moved from `1708 MiB`
to `1709 MiB`. Treat this as same-band live-tick protection, not a new speed
ceiling.

The trial-budget cleanup keeps that boundary after retiring the old low-tail
strength defaults. The current 65536-column 262144-token check
`reports/bounded_replay_window_20260617/hotpath-active-pressure-65536-262144-i32-target-strength-budget-compact.json`
processed `262144` tokens at `6232.282 tokens/sec`, with
`train_compute=0.130988 ms/token`, `prepare_training=0.006491 ms/token`,
`finalize_total=0.006431 ms/token`, and `tick_duration_ms.p95=20.659`.
Route scoring stayed bounded at `12/65536` rows with `10` output candidates,
state transition cached `65526` rows, `state_transition_runs_all_columns=false`,
and graph/native/sequence failures remained `0`. The run reported no observed
contention: CPU max `33%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory stayed flat at `1715 MiB` before and after
measurement. This is the accepted live-tick protection evidence for the
target-specific replay-strength budget defaults.

### Replay Tensor Payload and Bounded SFA, 2026-06-17

The replay text/SFA cleanup changes only slow-window replay and memory-store
reporting, but it still gets a current long-run gate because the goal requires
throughput to stay in the same band. Sleep replay now loads tensor payloads only
from `DualMemoryStore.replay_entry(..., include_text_payload=False)` and reports
`sleep_replay_text_payload_loaded=false`,
`sleep_replay_language_reasoning=false`, and
`sleep_replay_text_payload_policy=sleep_replay_uses_tensor_payloads_only`.
Deep replay SFA correction now samples from selected replay indices and reports
`sleep_replay_sfa_full_memory_sample_retired=true`.

The final 65536-column 262144-token check reused the active-pressure checkpoint:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-replay-tensor-payload-boundary.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6237.420 tokens/sec`, with
`train_compute=0.130490 ms/token`, `prepare_training=0.006495 ms/token`,
`finalize_total=0.006446 ms/token`, and `tick_duration_ms.p95=20.383`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `28%`, GPU utilization max `18%`, GPU memory
utilization max `12%`, and GPU memory flat at `1719 MiB` before and after
measurement.

### Unscoped Replay Helper Retirement, 2026-06-17

The helper-retirement slice removed attractive full-buffer defaults from replay
and SFA helpers. At that point `DualMemoryStore.sample_replay_indices(...)`
required bucket ids unless a caller explicitly opted into a diagnostic global
scorer, and `sample_for_sfa(...)` returned no samples without selected
candidate indices unless an explicit diagnostic flag marked the call as
diagnostic. Those list-only helpers were later removed entirely in favor of
`select_replay_window(...)` and `sample_for_sfa_with_report(...)`, and the
2026-06-18 runtime hook cleanup removed the remaining full-scan diagnostic
flags from the runtime store.

The first 65536-column run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6068.338 tokens/sec`, with
`train_compute=0.135063 ms/token`, `prepare_training=0.006458 ms/token`,
`finalize_total=0.006376 ms/token`, `tick_duration_ms.p95=21.140`,
`route_input_rows_scored=12/65536`, `state_transition_cached_count=65526`, zero
graph/native/sequence failures, and flat `1856 MiB` GPU memory. It is secondary
evidence because the environment reported `contention_observed` with GPU
utilization max `21%`.

The accepted clean rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-unscoped-replay-helper-retired-rerun.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `5668.688 tokens/sec`, with
`train_compute=0.141909 ms/token`, `prepare_training=0.007435 ms/token`,
`finalize_total=0.006774 ms/token`, and `tick_duration_ms.p95=25.429`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `71%`, GPU utilization max `11%`, GPU memory
utilization max `11%`, and GPU memory moved from `1877 MiB` before measurement
to `1844 MiB` after measurement. Treat this as same-band live-tick protection
for removing unscoped helper defaults, not as a new speed ceiling.

### Capped Replay Candidate Windows, 2026-06-17

The capped replay-candidate slice is also slow-window work, but it changes the
selection cost contract. Bucket-scoped replay now collects recent entries
round-robin across anchor buckets up to `candidate_window_limit` before scoring,
then reports available versus scored entries. Unscoped random replay is retired
by default unless an explicit diagnostic global candidate scan is requested.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-capped-replay-window.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6148.125 tokens/sec`, with
`train_compute=0.132113 ms/token`, `prepare_training=0.006656 ms/token`,
`finalize_total=0.006548 ms/token`, and `tick_duration_ms.p95=21.137`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `29%`, GPU utilization max `15%`, GPU memory
utilization max `13%`, and GPU memory stayed flat at `1848 MiB` before and after
measurement. This preserves same-band live-tick throughput while the replay
selector gains a stronger pre-score memory budget.

### Capped Replay Query Collection, 2026-06-17

The capped query-collection slice retires the HF recall runner's linear
`slow_bucket_ids` scan for Task-A anchor queries. It is slow-window work, but
the live tick still gets a long protection run because the change touches
checkpointed replay reports and consolidation evidence.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-collection.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6221.949 tokens/sec`, with
`train_compute=0.131162 ms/token`, `prepare_training=0.006563 ms/token`,
`finalize_total=0.006444 ms/token`, and `tick_duration_ms.p95=20.657`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `28%`, GPU utilization max `12%`, GPU memory
utilization max `11%`, and GPU memory stayed flat at `1848 MiB` before and
after measurement. This keeps replay query collection off the live tick and in
the same sustained throughput band.

### Bounded Query-Memory Match, 2026-06-17

The query-memory match slice retires the explicit query readout's full
slow-memory scan. Query matching now derives routing bucket candidates, collects
a capped bucket-indexed memory window, and computes similarity/replay-priority
scores only for those candidate indices. It is query/readout work, not live
training, but it still received the same long hot-path protection gate.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-query-memory-match.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6137.185 tokens/sec`, with
`train_compute=0.131555 ms/token`, `prepare_training=0.006550 ms/token`,
`finalize_total=0.006528 ms/token`, and `tick_duration_ms.p95=20.711`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `14%`, GPU utilization max `10%`, GPU memory
utilization max `10%`, and GPU memory stayed flat at `1848 MiB` before and
after measurement.

### Returned-Only Query Memory Payload, 2026-06-18

The query-memory payload slice keeps explicit readout text payloads behind the
bounded return set when no query/focus terms require candidate-window text
ranking. Similarity-only query readout still scores the selected candidate
window, but `bounded_query_memory_match.v1` now reports
`raw_text_payload_policy=returned_similarity_matches_only`,
`raw_text_payload_count`, and `language_reasoning=false`. The benchmark
`reports/bounded_replay_window_20260617/query-memory-payload-returned-only.json`
preserved selected indices against the retired eager candidate-payload shape,
loaded raw text for `5` returned matches instead of `192` candidates, and
reduced mean latency from `33.612 ms` to `25.881 ms`.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-524288-i32-query-memory-payload.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6152.079 tokens/sec`, with
`train_compute=0.132199 ms/token`, `prepare_training=0.006607 ms/token`,
`finalize_total=0.006542 ms/token`, and `tick_duration_ms.p95=21.044`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `47%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory moved only from `1874 MiB` to `1878 MiB`.

### Bounded Concept-Frontier Memory Metrics, 2026-06-17

The concept-frontier metric slice retires the autonomy acquisition planner's
full `slow_routing_keys` scan. Frontier novelty, uncertainty, and support now
derive routing candidate buckets from the probe-bank signature, collect a
capped bucket-indexed memory window, and score only those selected entries. It
is slow-path planning work, but it received the same long live-tick protection
gate because it changes source-acquisition recall reports.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-concept-frontier-bounded-scope.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6148.846 tokens/sec`, with
`train_compute=0.131437 ms/token`, `prepare_training=0.006683 ms/token`,
`finalize_total=0.006436 ms/token`, and `tick_duration_ms.p95=20.683`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `22%`, GPU utilization max `17%`, GPU memory
utilization max `14%`, and GPU memory stayed flat at `1805 MiB` before and
after measurement.

### Bounded Source-Bank Memory Match, 2026-06-18

The source-bank memory-match slice makes source acquisition recall auditable at
the bank-planning layer. `bank_memory_matches_with_report(...)` samples capped
source-bank probes, runs each probe through `bounded_query_memory_match.v1`,
shares returned replay-entry payloads across probes, and records
`bounded_source_bank_memory_match.v1` with candidate totals, unique match count,
payload cache hits, CPU archival/score placement, no global scans,
`runs_live_tick=false`, and `language_reasoning=false`. The paired benchmark
`reports/bounded_replay_window_20260618/source-bank-memory-match-bounded.json`
preserved selected indices against the diagnostic legacy path, reduced raw text
payload loads from `32` to `4`, and reduced mean latency from `194.259 ms` to
`179.366 ms` over a `65536`-entry archive.

The clean 65536-column 524288-token protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-source-bank-memory-match-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 720 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6524.395 tokens/sec`, with
`train_compute=0.124824 ms/token`, `prepare_training=0.006321 ms/token`,
`finalize_total=0.005793 ms/token`, and `tick_duration_ms.p95=18.989`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`, with `32767` conditional sequence-loop successes
covering `524272` tokens. The velocity surface reported no observed
contention: CPU max `12%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory moved from `1833 MiB` before measurement to
`1798 MiB` after measurement. This proves source-bank recall reporting did not
add a live-tick tax; it does not make source-bank recall every-token work.

### Bounded ConceptStore Signature Lookup, 2026-06-18

The ConceptStore signature lookup slice retires an old archive-shaped helper
inside semantic observation. `ConceptStore._memory_signature(...)` used to
materialize `slow_routing_keys`, `slow_input_patterns`, or `slow_buffer` with
`list(...)` before reading one evidence-provided memory index. The replacement
uses direct indexing only, caps each evidence source at `8` unique memory
indices with a `32`-reference scan budget, and reports
`bounded_concept_memory_signature_lookup.v1` in the concept summary. This is
cadenced semantic-observation work, not replay mutation; archival storage stays
CPU-resident and no hidden text reasoning or global memory scan is introduced.

The clean 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-concept-signature-lookup-clean-gate.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6143.768 tokens/sec`, with
`train_compute=0.131822 ms/token`, `prepare_training=0.006630 ms/token`,
`finalize_total=0.006432 ms/token`, and `tick_duration_ms.p95=20.691`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `35%`, GPU utilization max `18%`, GPU memory
utilization max `12%`, and GPU memory stayed flat at `1746 MiB` before and
after measurement. The longer 524288-token same-code runs reached `6183.670`
and `6196.447 tokens/sec`, but both are secondary evidence because the
benchmark condition report observed pre-measurement GPU contention after
prewarm.

### Bounded Runtime Concept Memory Lookup, 2026-06-18

The runtime concept memory lookup slice retires service-owned direct archive
reads during cadenced source/feed concept observation. `OperatorInteractionRuntime`
now asks `DualMemoryStore.resolve_runtime_concept_memory_matches(...)` to
resolve only trainer-provided `memory_index` evidence, cap the observation
batch, cache duplicate payloads, and report
`bounded_runtime_concept_memory_lookup.v1`. This path can run inside live
runtime observation cadence, so the report says `runs_live_tick=true`; it also
reports `runs_every_token=false`, `global_candidate_scan=false`,
`global_score_scan=false`, CPU archival/score placement, and
`language_reasoning=false`.

The bounded lookup benchmark was:

`python -m marulho.evaluation.runtime_concept_memory_lookup_benchmark --capacity 65536 --observation-count 512 --unique-indices 64 --max-observations 512 --text-repeats 64 --iterations 24 --min-speedup 1.0 --output reports\bounded_replay_window_20260618\runtime-concept-memory-lookup-bounded.json`

It preserved selected-index parity (`quality.min=1.0`), reduced raw text
payload reads from `512` to `64` with `448` cache hits, and reduced mean lookup
latency from `47.156 ms` to `6.380 ms` (`7.391x`) while reporting no archive
iteration, no global score/candidate scan, CPU archival placement, and
`runs_every_token=false`.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-runtime-concept-memory-lookup.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6237.075 tokens/sec`, with
`train_compute=0.131104 ms/token`, `prepare_training=0.006390 ms/token`,
`finalize_total=0.006141 ms/token`,
`concept_observation=0.000474 ms/token`, and `tick_duration_ms.p95=20.906`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Native sequence and burst failures
were `0`. The velocity surface reported no observed contention: CPU max `21%`,
GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU memory
changed from `1809 MiB` to `1861 MiB`.

### Bounded Context-Comparison Memory Recall, 2026-06-18

The context-comparison recall slice removes the old report-dropping
`query_runner.memory_matches(...)` wrapper. `build_context_comparison(...)`
now uses `memory_matches_with_report(...)` for each context, shares one
returned replay-entry payload cache across the comparison, returns each
per-context report, and emits the aggregate
`bounded_context_comparison_memory_match.v1`. This path is explicit slow
readout: `runs_live_tick=false`, `runs_every_token=false`,
`global_candidate_scan=false`, `global_score_scan=false`,
`language_reasoning=false`, and CPU archival/score placement.

The bounded recall benchmark was:

`python -m marulho.evaluation.context_memory_match_benchmark --capacity 65536 --candidate-limit 192 --top-k 8 --context-count 2 --text-repeats 256 --iterations 24 --min-speedup 1.0 --output reports\bounded_replay_window_20260618\context-memory-match-bounded.json`

It preserved selected-index parity for both contexts (`quality.min=1.0`),
reduced raw text payload reads from `16` to `8` with `8` cache hits, and
reduced mean readout latency from `71.927 ms` to `70.550 ms` (`1.020x`). The
benchmark reports `candidate_index_count=384`, `unique_candidate_index_count=192`,
`candidate_window_limit=192`, and no global score or candidate scan.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-context-memory-match.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6065.987 tokens/sec`, with
`train_compute=0.135179 ms/token`, `prepare_training=0.006512 ms/token`,
`finalize_total=0.006339 ms/token`,
`concept_observation=0.000474 ms/token`, and `tick_duration_ms.p95=24.179`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Native sequence and burst failures
were `0`. The velocity surface reported no observed contention: CPU max `32%`,
GPU utilization max `10%`, GPU memory utilization max `10%`, and GPU memory
changed from `1839 MiB` to `1845 MiB`. This is lower than the strongest clean
source-bank rerun but still inside the maintained 6k-ish sustained band.

### Bounded Semantic Frontier-Gap Planner, 2026-06-18

The semantic frontier-gap planner slice retires the old archive-shaped term
planner. `frontier_gap_plan(...)` now collects a capped CPU recency or bucket
candidate window through `DualMemoryStore.collect_frontier_gap_indices(...)`
before scoring terms, then reports `bounded_frontier_gap_selection.v1` with no
global candidate/score scan, raw text loaded only for selected candidates, and
`language_reasoning=false`. The paired quality benchmark
`reports/bounded_replay_window_20260617/frontier-gap-bounded.json` preserved
expected and diagnostic legacy terms with `quality.min=1.0` while reducing mean
latency from `217.530 ms` to `9.073 ms` over a `65536`-entry archive. It also
passes a missing-collector retirement gate: a store without
`collect_frontier_gap_indices(...)` returns zero candidates, zero text
payloads, and no global scans instead of using a compatibility prefix read.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-frontier-gap-collector-required.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6233.085 tokens/sec`, with
`train_compute=0.131067 ms/token`, `prepare_training=0.006384 ms/token`,
`finalize_total=0.006244 ms/token`,
`concept_observation=0.000470 ms/token`, and `tick_duration_ms.p95=20.524`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `32%`, GPU utilization max `14%`, GPU memory utilization
max `13%`, and GPU memory changed from `1844 MiB` to `1840 MiB`.

### Reported SFA Sampler, 2026-06-18

The reported SFA sampler changes only selected slow-window abstraction
correction and report/checkpoint surfaces. The hot path does not call
`sample_for_sfa_with_report(...)`, and the list-only
`sample_replay_indices(...)`, `sample_for_sfa(...)`, and
`bank_memory_matches(...)` wrappers are removed so callers cannot drop bounded
reports.

The accepted 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-reported-sfa-sampler-noprofile-rerun2.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6127.490 tokens/sec`, with
`last_tick_duration_ms=20.317`. The no-profile report records last-tick stage
timings: `train_compute=17.738 ms` (`0.138579 ms/token` over the 128-token
tick), `prepare_training=1.225 ms`, `finalize_total=0.935 ms`, and
`concept_observation=0.074 ms`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `15%`, GPU utilization max `11%`, GPU memory utilization
max `11%`, and GPU memory changed from `1840 MiB` to `1861 MiB`.

A clean detached `HEAD` baseline immediately before the reported-SFA edits
processed the same `524288` tokens at `6188.048 tokens/sec` with
`last_tick_duration_ms=18.079` and last-tick `train_compute=16.287 ms`
(`0.127241 ms/token` over the 128-token tick), so the accepted current run is
`99.0%` of the clean baseline throughput and remains in the same sustained
band. Earlier same-code current reruns at `5382.323`, `5639.785`, and
`5462.307 tokens/sec` are kept as secondary noisy evidence and are not used for
promotion.

### Bounded Sleep Repair Input Prep, 2026-06-18

The sleep repair replay slice changes only selected slow-window repair replay.
Normal replay entries already carry stored routing keys, so repair no longer
builds dense input assemblies after the replay window has selected anchored
entries. The focused benchmark
`reports/bounded_replay_window_20260618/sleep-repair-replay-bounded-input-prepare.json`
selected and repaired `32/32` anchored entries at `65536` columns, improved
mean anchor distance from `0.508855` to `0.360171`, reduced selected input-prep
latency from `61.351 ms` to `32.613 ms` (`1.881x`), and reported `0` dense
assembly calls during repair.

The accepted 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-sleep-repair-bounded-input-prepare.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6302.207 tokens/sec`, with
`tick_duration_ms.p95=20.440`, `train_compute=0.129260 ms/token`,
`prepare_training=0.006437 ms/token`, `finalize_total=0.006086 ms/token`, and
`concept_observation=0.000463 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Runtime failure count was `0`, no
route-vote fallback was reported, and the velocity surface reported no observed
contention: CPU max `30%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory changed from `1805 MiB` to `1806 MiB`.

### Query Episode Readout, 2026-06-18

The query episode readout slice changes explicit query slow-path output only.
The hot path does not call `build_memory_episodes_with_report(...)`.
`build_memory_episodes(...)` is removed so callers cannot build replay-text
episodes while dropping the bounded readout report.

The accepted 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-query-episode-readout.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6219.926 tokens/sec`, with
`tick_duration_ms.p95=20.421`, `train_compute=0.130647 ms/token`,
`prepare_training=0.006404 ms/token`, `finalize_total=0.006373 ms/token`, and
`concept_observation=0.000467 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `26%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory changed from `1810 MiB` to `1811 MiB`.

### Bounded Source-Episode Admission, 2026-06-18

The source-episode admission slice changes explicit query-runner feed and
query readout only. The hot path does not call
`bounded_feed_source_episode_admission.v1`. Admission is capped to `32`
deduplicated source episodes and `240` chars per payload, stores archival
tensors/text on CPU, and reports no live tick, no every-token work, no global
candidate/score scan, and no language reasoning. Readout also stops stitching
raw cadence fragments across source-admission boundaries unless neighboring
windows prove character overlap.

The accepted 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-source-episode-admission.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6702.362 tokens/sec`, with
`tick_duration_ms.p95=18.491`, `train_compute=0.121727 ms/token`,
`prepare_training=0.005927 ms/token`, `finalize_total=0.005560 ms/token`, and
`concept_observation=0.000417 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `11%`, GPU utilization max `13%`, GPU memory utilization
max `12%`, and GPU memory stayed flat at `1808 MiB` before and after
measurement.

The v2 rerun after retiring the dense source-admission assembly call was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-source-episode-admission-v2.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6412.209 tokens/sec`, with
`tick_duration_ms.p95=19.973`, `train_compute=0.126270 ms/token`,
`prepare_training=0.006236 ms/token`, `finalize_total=0.005747 ms/token`, and
`concept_observation=0.000437 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Runtime failure count was `0`, no
route-vote fallback was reported, CPU max was `11%`, and GPU memory changed
from `1812 MiB` to `1866 MiB`. The sampler reported GPU-side contention, so
the run is accepted as hot-path protection evidence but not as a throughput
improvement claim.

### Bounded Replay-Plan Source Window, 2026-06-18

The service replay-plan slice changes replay planning and sample revalidation,
not neural training. `build_replay_plan(...)` no longer materializes every
runtime episode, action, prediction, uncertain domain, or recent-feedback list
before returning a capped plan. It now reports
`bounded_replay_plan_source_window.v1`, selects `64` recent items per source
stream by timestamp orientation, indexes `128` recent feedback entries, and adds up to `32`
feedback-target stubs ranked by contradiction/correction signal before recency.
Archival/status metadata and active ranking stay on CPU; the report states
`runs_live_tick=false` and `gpu_used=false`.

The planner benchmark was:

`python -m marulho.evaluation.replay_plan_source_window_benchmark --output reports/bounded_replay_window_20260618/replay-plan-source-window-bounded.json --source-size 20000 --feedback-size 128 --domain-size 2000 --limit 10 --runs 7 --baseline-unbounded-mean-ms 6860.919`

It used `20000` episodes, `20000` actions, `20000` predictions, `2000`
uncertain domains, and `128` recent feedback rows. The bounded planner returned
`ep-42` as the top contradicted feedback target while considering only `96`
episode rows including stubs, `64` actions, `64` predictions, `64` domains, and
`128` feedback rows. Mean latency was `14.684 ms` versus a pre-change
unbounded mean of `6860.919 ms` (`467.225x`), with traced Python peak allocation
`0.519 MiB`. CUDA was available on the RTX 3060 but unused, with `0.0 MiB`
allocated/reserved VRAM.

The paired 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-replay-plan-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6344.404 tokens/sec`, with
`tick_duration_ms.p95=20.160`, `train_compute=0.128679 ms/token`,
`prepare_training=0.006359 ms/token`, `finalize_total=0.006097 ms/token`, and
`concept_observation=0.000463 ms/token`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`. The velocity surface reported no observed
contention: CPU max `24%`, GPU utilization max `10%`, GPU memory utilization
max `10%`, and GPU memory stayed flat at `1799 MiB` before and after
measurement.

### Bounded Recent Replay Setup, 2026-06-17

The recent tag/anchor setup slice changes slow-window replay setup only:
`tag_recent_entries(...)` and `capture_recent_memory_anchors(...)` now use a
CPU recency index capped by `max_recent_entries` instead of walking archival
timestamps or bucket ids. It still received the full 65536-column live-tick gate
because replay setup state is checkpointed and consumed by consolidation.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-recent-anchor-window.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6228.243 tokens/sec`, with
`train_compute=0.131307 ms/token`, `prepare_training=0.006430 ms/token`,
`finalize_total=0.006432 ms/token`, and `tick_duration_ms.p95=20.538`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `25%`, GPU utilization max `13%`, GPU memory
utilization max `11%`, and GPU memory stayed flat at `1846 MiB` before and
after measurement.

### Full-Buffer Replay Score Helper Retirement, 2026-06-17

The replay-score helper cleanup removes a test-only full-buffer slow-memory
scorer and leaves replay priority available only through explicit candidate
indices. It does not add live-tick work, but it still received the same
65536-column protection run because it changes the replay helper API surface.

The 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-replay-score-helper-retired.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6211.859 tokens/sec`, with
`train_compute=0.131468 ms/token`, `prepare_training=0.006475 ms/token`,
`finalize_total=0.006438 ms/token`, and `tick_duration_ms.p95=20.679`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `38%`, GPU utilization max `16%`, GPU memory
utilization max `14%`, and GPU memory stayed flat at `1852 MiB` before and
after measurement.

### Score Tensor Helper Family Retirement, 2026-06-17

The score tensor helper cleanup removes the remaining public full-buffer
slow-memory score tensor family after the priority helper retirement. Production
bounded replay selection still scores only selected candidate indices; the
2026-06-18 runtime hook cleanup removes the remaining diagnostic global scoring
branch from `select_replay_window(...)`.

The accepted 65536-column 262144-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260617\hotpath-active-pressure-65536-262144-i32-score-tensor-helpers-retired-rerun3.json --target-tokens 262144 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 480 --sample-interval-seconds 0.5 --host-truth-sync-interval-tokens 32`

It processed `262144` tokens at `6151.952 tokens/sec`, with
`train_compute=0.132119 ms/token`, `prepare_training=0.006688 ms/token`,
`finalize_total=0.006420 ms/token`, and `tick_duration_ms.p95=20.697`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `32%`, GPU utilization max `18%`, GPU memory
utilization max `14%`, and GPU memory stayed flat at `1805 MiB` before and
after measurement. Earlier same-code reruns were kept secondary because one
observed GPU contention and one clean run fell below the maintained throughput
band, so this accepted rerun is the primary hot-path evidence.

### SNN Replay-Priority Source Window, 2026-06-18

The SNN replay-priority queue cleanup changes a service/control-plane replay
review surface, not the neural live tick. The old queue returned a bounded
candidate count but verified every retained replay-evaluation context before
ranking. The replacement emits `bounded_snn_replay_priority_source_window.v1`:
`16` recent contexts plus up to `16` explicit readout-target context IDs are
verified through a controller-owned ID index, with CPU archival/score placement,
no global scan, no raw replay text, no hidden language reasoning,
`runs_live_tick=false`, and `gpu_used=false`.

The focused benchmark
`reports/bounded_replay_window_20260618/snn-replay-priority-source-window.json`
kept an old readout-targeted high-signal context selectable outside the recent
window while verifying `17` contexts instead of all `64` retained contexts. It
averaged `1.825268 ms`, reported a `3.764706x` verification-work reduction, used
`0.050346 MiB` traced peak Python allocation, and allocated `0.0 MiB` CUDA/VRAM.

The matching long protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-replay-priority-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6298.310 tokens/sec`, with
`train_compute=0.129349 ms/token`, `prepare_training=0.006320 ms/token`,
`finalize_total=0.006134 ms/token`, and `tick_duration_ms.p95=20.369`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `31%`, GPU utilization max `11%`, GPU memory
utilization max `11%`, and GPU memory moved only from `1799 MiB` to `1800 MiB`.

### SNN Replay Artifact Provenance Source Window, 2026-06-18

The replay-artifact provenance cleanup changes service/control-plane lookup
boundaries, not neural live-tick execution. The old shape verified replay
artifact review tickets, transition-memory replay artifacts, regeneration
permits, and related sleep/scheduler tickets by walking retained deques. The
replacement keeps controller-owned ID indexes and emits
`bounded_snn_replay_artifact_provenance_source_window.v1` for evaluated
artifacts and permits, capped to context/ticket/artifact/permit IDs with CPU
archival placement, no global scan, no raw replay text, no hidden language
reasoning, `runs_live_tick=false`, and `gpu_used=false`.

The focused benchmark was:

`python -m marulho.evaluation.snn_replay_artifact_provenance_source_window_benchmark --retention-count 64 --runs 25 --output reports\bounded_replay_window_20260618\snn-replay-artifact-provenance-source-window.json`

It kept the oldest retained context/ticket/artifact/permit chain verifiable at
the retention tail, used `4` indexed lookups instead of `256` worst-case
retained-record checks (`64x` less lookup work), averaged `0.348376 ms`, used
`0.012636 MiB` traced peak Python allocation, and allocated `0.0 MiB` CUDA/VRAM.

The first matching long protection run completed but was rejected as throughput
evidence because it reached only `5849.047 tokens/sec`, below the maintained
6k-ish band, despite no observed contention. The accepted rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-replay-artifact-provenance-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6286.248 tokens/sec`, with
`train_compute=0.129585 ms/token`, `prepare_training=0.006458 ms/token`,
`finalize_total=0.006156 ms/token`, and `tick_duration_ms.p95=20.417`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `20%`, GPU utilization max `10%`, GPU memory
utilization max `10%`, and GPU memory moved from `1817 MiB` to `1799 MiB`.

### Runtime Global Scan Hook Retirement, 2026-06-18

This slice removes the remaining runtime full-scan hooks from `DualMemoryStore`.
Awake ripple now requires awake bucket ids, replay-window selection requires
bucket ids, and SFA sampling requires selected replay indices. Missing scope
returns a bounded empty report; retired full-buffer comparisons are isolated in
benchmark modules and cannot be requested through the runtime store.

The direct evidence reports were:

`python -m marulho.evaluation.awake_ripple_scope_benchmark --output reports\bounded_replay_window_20260618\awake-ripple-runtime-global-hooks-retired.json --capacity 8192 --bucket-count 8192 --awake-bucket-count 10 --iterations 256`

`python -m marulho.evaluation.sfa_sample_scope_benchmark --output reports\bounded_replay_window_20260618\sfa-runtime-global-hooks-retired.json --capacity 65536 --candidate-count 192 --sample-count 64 --iterations 32`

The awake-ripple benchmark measured the benchmark-local retired full scan at
`1.285064 ms` mean and the wake-bucket scoped path at `1.082768 ms`
(`1.186832x`). The scoped path used `10` candidates, zero runtime global
scans, and passed the bounded-candidate gate. The SFA benchmark passed with
selected-window purity `1.0` versus retired full-buffer purity `0.00439453125`;
mean latency improved from `1.740475 ms` to `0.622956 ms` (`2.793896x`) with
CPU archival/sample placement.

The 65536-column 524288-token protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-runtime-global-scan-hooks-retired.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6342.218 tokens/sec`, with
`train_compute=0.128534 ms/token`, `prepare_training=0.006349 ms/token`,
`finalize_total=0.006160 ms/token`, and `tick_duration_ms.p95=20.119`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. GPU memory moved from `1801 MiB` to
`1802 MiB`; the velocity surface observed brief GPU-side contention
(`23%` max utilization), so this is accepted as sustained-throughput protection
evidence rather than a contention-free hardware run.

### Source-Bank Frontier Probe Signature Window, 2026-06-18

This slice bounds the source-bank probe signature used by autonomy
concept-frontier planning. The old implementation averaged every source-bank
probe before asking the routing index for candidate buckets. That was slow-path
planning, not neural live-tick work, but it preserved an input-unbounded recall
shape that would scale poorly for future LLM-size source banks. The maintained
path samples an evenly spaced `16`-probe source window, reports the source-probe
budget and selected indices, then scores only the capped bucket-indexed memory
candidate window. Archival memory stays CPU-resident; active routing signature
computation uses the existing trainer path.

The focused benchmark was:

`python -m marulho.evaluation.concept_frontier_scope_benchmark --output reports\bounded_replay_window_20260618\concept-frontier-source-probe-window-bounded.json --capacity 16384 --bucket-count 2048 --candidate-bucket-count 8 --probe-count 64 --dim 32 --iterations 32`

It sampled `16/64` source probes, scored `64/16384` memory entries, preserved
the diagnostic full-scan top-1, kept `novelty_delta=0.0000706911`,
`uncertainty_delta=0.0`, and `support_delta=0.0219844`, and reduced mean
latency from `1556.602 ms` to `7.637 ms` (`203.829x`). The report kept
`global_candidate_scan=false`, `global_score_scan=false`,
`archival_storage_device=cpu`, and `runs_live_tick=false`.

The first same-code long hot-path runs reached `5935.082`, `5753.405`, and
`5798.030 tokens/sec`, so they were rejected as primary throughput evidence.
The accepted comparison used a paired committed-baseline/current-code run on the
same 524288-token shape:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-baseline-29a1ffe-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-concept-frontier-source-probe-window-paired-current.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

The baseline at `29a1ffe` processed `524288` tokens at `6307.437 tokens/sec`.
The source-probe current tree processed `524288` tokens at
`6303.548 tokens/sec`, with `train_compute=0.129019 ms/token`,
`prepare_training=0.006522 ms/token`, `finalize_total=0.006066 ms/token`, and
`tick_duration_ms.p95=19.815`. Runtime Truth stayed bounded at
`route_input_rows_scored=12/65536`, `route_output_candidate_count=10`,
`state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `29%`, GPU utilization max `10%`, GPU memory
utilization max `10%`, and GPU memory stayed flat at `1789 MiB`.

### Repair Replay Dense Legacy Fallback Retirement, 2026-06-18

This slice removes the remaining dense fallback from selected repair replay.
The previous repair path used stored routing keys when present, but entries
without stored routing keys rebuilt a routing key from the input pattern, which
could call dense all-column input assembly in legacy checkpoint shapes. The
maintained path uses the stored routing key when present and otherwise projects
the already-selected stored assembly trace. The report records
`sleep_replay_stored_assembly_projection_fallback_count` and keeps
`sleep_replay_dense_input_assembly_fallback_count=0`.

The focused mixed-key benchmark was:

`python -m marulho.evaluation.sleep_repair_replay_bounded_benchmark --output reports\bounded_replay_window_20260618\sleep-repair-replay-no-dense-legacy-fallback.json --n-columns 65536 --column-latent-dim 64 --entry-count 32 --candidate-pool 64 --prepare-iterations 8 --drop-routing-key-every 2 --min-prepare-speedup 1.0`

It selected `32` anchored repair entries, dropped routing keys for `16`, used
`16` stored-assembly projection fallbacks, made `0` dense input-assembly calls,
improved mean repair quality by `0.171254`, and kept selected input-prep
speedup at `1.990857x`. Archival repair traces stayed CPU-resident, and active
repair computation ran on CUDA.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-repair-no-dense-legacy-fallback.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.5`

It processed `524288` tokens at `6298.782 tokens/sec`, with
`train_compute=0.129392 ms/token`, `prepare_training=0.006428 ms/token`,
`finalize_total=0.006182 ms/token`, and `tick_duration_ms.p95=20.403`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`. The velocity surface reported no
observed contention: CPU max `20%`, GPU utilization max `10%`, GPU memory
utilization max `10%`, and GPU memory moved only from `1790 MiB` to `1791 MiB`.

### Replay Query Anchor Source Window, 2026-06-18

This slice bounds the HF replay-query anchor source before the memory store
collects query indices. The replay-query benchmark is CPU-only slow-path
evidence, but the live tick still needs a long protection run because the
runner/checkpoint changes touch anchor metadata and checkpoint restore.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-replay-query-anchor-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6376.873 tokens/sec`, with
`train_compute=0.128288 ms/token`, `prepare_training=0.006247 ms/token`,
`finalize_total=0.005964 ms/token`, and `tick_duration_ms.p95=20.160`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, selection, native sequence,
and native burst failures were all `0`; conditional-WHILE q16 remained the
active sequence executor with zero native sequence fallback/failure. Runtime
device was RTX 3060 CUDA, and GPU memory stayed flat at `1787 MiB`. The
velocity sampler observed borderline GPU contention at `20%` max utilization
before/around measurement, while CPU stayed at `15%` max and GPU memory
utilization at `16%`; because throughput and stage timings stayed in the
maintained band, this is accepted as hot-path protection evidence but not as a
clean contention-free throughput ceiling.

### Bucket Candidate Source Window, 2026-06-18

This slice retires hot-bucket source materialization inside the shared memory
candidate collector. The old implementation returned a bounded candidate list
but first built `list(reversed(...))` over each selected bucket, so a large
bucket could still create source-size work before replay selection, replay-query
collection, query readout, frontier planning, or awake ripple tagging.

Focused source benchmark:

`python -m marulho.evaluation.bucket_candidate_source_window_benchmark --output reports\bounded_replay_window_20260618\bucket-candidate-source-window-bounded.json --archive-size 65536 --candidate-limit 32 --iterations 64`

It passed newest-candidate parity on a `65536`-entry hot bucket. The legacy
materialized source path averaged `0.416944 ms`; the bounded cursor path
averaged `0.060931 ms` (`6.843x`). The bounded report read `32` source
entries within a `32`-entry source-read budget, materialized `0`, set
`candidate_source_full_bucket_scan=false`, kept archival/source placement on
CPU, and allocated `0.0 MiB` CUDA memory.

Replay quality benchmark:

`python -m marulho.evaluation.bounded_replay_window_benchmark --output reports\bounded_replay_window_20260618\synthetic-bucket-source-window.json`

The positive-pressure arm kept the memory-consolidation gate and bounded recall
gate passing, applied `2` accepted updates, measured
`mean_input_pattern_distance=5.96046447753906e-08`, read `14` source entries in
the selected replay cycle, materialized `0`, and reported no global candidate
scan. Zero-pressure and no-anchor controls remained blocked as expected.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-bucket-candidate-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6290.744 tokens/sec`, with
`train_compute=0.129997 ms/token`, `prepare_training=0.006358 ms/token`,
`finalize_total=0.006150 ms/token`, and `tick_duration_ms.p95=20.763`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention: CPU max `33%`, GPU max
`10%`, GPU memory-util max `10%`, and RTX 3060 memory stayed flat at
`1788 MiB`. This is accepted as same-band hot-path protection evidence; the
change is a slow/source-window replay boundary and does not add live-tick work.

## SNN Readout Replay-Priority Source Window

`snn_language_readout_replay_priority.v1` now bounds source scoring before
ranking readout replay candidates. The previous implementation capped returned
candidates but scored every retained readout ledger event and called the full
ledger snapshot summary. The replacement reads a recent `32`-event CPU source
window, reports `bounded_snn_readout_replay_priority_source_window.v1`, and sets
`global_candidate_scan=false`, `global_score_scan=false`,
`raw_text_payload_loaded=false`, `language_reasoning=false`,
`runs_live_tick=false`, `runs_every_token=false`, and `gpu_used=false`.

Focused source benchmark:

`python -m marulho.evaluation.snn_readout_replay_priority_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-readout-replay-priority-source-window.json`

It matched the diagnostic full-retained scorer's top high-signal readout and
selected `8` candidates. The bounded path scored `32` of `2048` retained events
(`64x` less scoring work), averaged `1.424948 ms` versus `51.002932 ms` for the
benchmark-local retired scorer (`35.792837x`), used `0.065639 MiB` traced peak
Python allocation, kept archival/source/score placement on CPU, and allocated
`0.0 MiB` CUDA memory on RTX 3060.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-readout-replay-priority-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6284.379 tokens/sec`, with
`train_compute=0.129905 ms/token`, `prepare_training=0.006326 ms/token`,
`finalize_total=0.006227 ms/token`, and `tick_duration_ms.p95=20.623`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention: CPU max `23%`, GPU max
`11%`, GPU memory-util max `11%`, and RTX 3060 memory moved from `1852` to
`1858 MiB`. This is accepted as same-band hot-path protection evidence; readout
priority remains a slow/control-plane replay review surface, not live-tick or
every-token replay work.

## SNN Emission-Review Replay-Policy Source Window

`snn_language_readout_emission_replay_evaluation_policy.v1` now bounds source
matching before reviewed emissions can become replay-context seeds. The previous
policy capped returned candidates but matched reviewed emissions against every
retained internal readout event, and the design step verified selected seeds by
reopening every retained readout. The replacement reads a recent `16`-event CPU
review source window and a recent `16`-event CPU readout source window, reports
`bounded_snn_emission_review_replay_policy_source_window.v1`, and sets
`global_candidate_scan=false`, `global_score_scan=false`,
`raw_text_payload_loaded=false`, `language_reasoning=false`,
`runs_live_tick=false`, `runs_every_token=false`, and `gpu_used=false`.

Focused source benchmark:

`python -m marulho.evaluation.snn_emission_review_replay_policy_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-emission-review-replay-policy-source-window.json`

It matched the diagnostic full-retained policy/design top candidate and selected
`8` hash-only seeds. The bounded path checked `32` source events instead of
`4096` retained review/readout records (`128x` less match work), averaged
`2.476164 ms` versus `166.924984 ms` for the benchmark-local retired policy and
design scan (`67.412734x`), used `0.046277 MiB` traced peak Python allocation,
kept archival/source/score placement on CPU, and allocated `0.0 MiB` CUDA memory
on RTX 3060.

The replacement checkpoint was regenerated after the local reports directory was
deleted:

`python -m marulho.evaluation.promoted_scheduler_checkpoint --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --report reports\column_scheduler_20260618\active-pressure-scheduler-65536-checkpoint.json --n-columns 65536 --column-latent-dim 64 --k-routing 10 --seed 20260617 --device cuda --active-pressure-filter-count 2 --candidate-memory-pressure-filter-start-tokens 0`

The clean 65536-column profiled protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-emission-review-replay-policy-source-window-profile-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6376.714 tokens/sec`, with
`train_compute=0.128297 ms/token`, `prepare_training=0.006487 ms/token`,
`finalize_total=0.005965 ms/token`, and `tick_duration_ms.p95=20.0283`. Runtime
Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention: CPU max `12%`, GPU max `13%`,
GPU memory-util max `18%`, and RTX 3060 memory moved from `2122` to `2123 MiB`.
The no-profile rerun reached `6392.672 tokens/sec` with no observed contention.
An earlier same-code profiled run at `5618.255 tokens/sec` is rejected because
external GPU load was present during measurement.

## SNN Emission Replay-Context Review Windows

`snn_language_readout_emission_replay_context_review(...)` now bounds both
caller-supplied replay-context seeds and observed sparse slots before it can
recompute mismatch/pressure or record a Replay Controller context. The bridge
uses the shared `SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32` readout replay
source-window budget and emits
`bounded_snn_emission_replay_context_review_seed_window.v1` plus
`bounded_snn_emission_replay_context_review_observed_slot_window.v1`. Both
windows must be bounded, untruncated, and well formed before context recording.

Focused quality benchmark:

`python -m marulho.evaluation.emission_replay_context_review_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\emission-replay-context-review-window.json`

Result: `pass=true`, exact seed and observed-slot windows recorded one Replay
Controller context at `32/32`; oversized seeds and observed slots both blocked
at `32/2048`; blocked payloads made no mismatch, pressure, or Replay
Controller calls; projected source work fell `64x`; traced Python peak
allocation was `1.832774 MiB`; CUDA allocation/reservation stayed `0.0 MiB`;
archival storage, source-window selection, and review gates stayed
CPU-resident; and the reports state no global candidate/score scan, no raw text
payload, no hidden language reasoning, no live tick, and no every-token cadence.

Hot-path protection rerun:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-emission-replay-context-review-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

Result: `success=true`, `524288` tokens in `87.513941 s`,
`5990.908 tokens/sec`, `train_compute=0.135901 ms/token`,
`prepare_training=0.007159 ms/token`, and `finalize_total=0.006388 ms/token`.
Runtime Truth kept route scoring bounded at `12/65536` input rows and `10`
output candidates, with `65526` cached transition rows,
`state_transition_runs_all_columns=false`, `route_vote_rows_run_all_columns=false`,
and zero route-vote/native sequence failures. Contention was not observed
(`cpu max=37%`, `gpu max=14%`); the live runtime used CUDA on the RTX 3060,
while the context-review benchmark kept archival/source/review work on CPU.
GPU memory moved from `2032 MiB` to `2031 MiB`. The first clean same-code run
at `5877.891 tokens/sec` is retained as below-band variance evidence.

## SNN Replay Evaluation Context Observed-Slot Windows

`snn_replay_evaluation_context(...)` now uses the same
`SNN_READOUT_REPLAY_TARGET_WINDOW_LIMIT=32` source-window operator before it can
recompute mismatch/pressure or record a Replay Controller context. The generic
context route emits `bounded_snn_replay_evaluation_context_observed_slot_window.v1`,
requires a bounded, untruncated, well-formed observed-slot window, and stores
that source-window evidence in the recorded context metadata. This keeps the
direct context endpoint as a bounded server-recomputed evidence gate rather
than a caller-sized side route.

Focused quality benchmark:

`python -m marulho.evaluation.snn_replay_evaluation_context_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\snn-replay-evaluation-context-window.json`

Result: `pass=true`, an exact observed-slot window recorded one Replay
Controller context at `32/32`, oversized observed slots blocked at `32/2048`,
blocked payloads made no mismatch, pressure, or Replay Controller calls, and
the recorded context carried the observed-slot source-window metadata. The
exact path averaged `1.276744 ms`; the oversized block averaged `8.440372 ms`;
projected source work fell `64x`; traced Python peak allocation was
`0.656714 MiB`; CUDA allocation/reservation stayed `0.0 MiB`; archival
storage, source-window selection, and review gates stayed CPU-resident; and the
reports state no global candidate/score scan, no raw text payload, no hidden
language reasoning, no live tick, and no every-token cadence.

Hot-path protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-snn-replay-evaluation-context-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

Result: `success=true`, `524288` tokens in `87.236934 s`,
`6009.932 tokens/sec`, `train_compute=0.135671 ms/token`,
`prepare_training=0.006985 ms/token`, `finalize_total=0.006359 ms/token`, and
profiled runtime `total=0.130420 ms/token`. Runtime Truth kept route scoring
bounded at `12/65536` input rows and `10` output candidates, with `65526`
cached transition rows, `state_transition_runs_all_columns=false`,
`route_vote_rows_run_all_columns=false`, and zero graph/native sequence
failures. The live runtime used CUDA on the RTX 3060; GPU memory moved from
`2031 MiB` to `2045 MiB`. The velocity sampler observed GPU-side contention
(`gpu max=22%`, memory-util max `23%`, CPU max `68%`), so the claim is
throughput protection in the maintained band, not contention-free hardware.

## SNN Rollout Rehearsal Source Window

`snn_language_readout_rollout_rehearsal_promotion_policy.v1` now bounds source
scoring before ranking rollout rehearsal candidates. The previous policy capped
returned candidates but first normalized and scored every retained rollout event
and called the full ledger snapshot summary. The replacement reads a recent
`16`-event CPU source window, caps replay targets to `32` per event, reports
`bounded_snn_readout_rollout_rehearsal_source_window.v1`, and sets
`global_candidate_scan=false`, `global_score_scan=false`,
`raw_text_payload_loaded=false`, `language_reasoning=false`,
`runs_live_tick=false`, `runs_every_token=false`, and `gpu_used=false`.

Focused source benchmark:

`python -m marulho.evaluation.snn_rollout_rehearsal_source_window_benchmark --retention-count 2048 --limit 8 --runs 25 --output reports\bounded_replay_window_20260618\snn-rollout-rehearsal-source-window.json`

It matched the diagnostic full-retained scorer's top high-signal rollout and
selected `8` candidates. The bounded path scored `16` of `2048` retained events
(`128x` less scoring work), averaged `2.090592 ms` versus `309.922768 ms` for
the benchmark-local retired scorer (`148.246414x`), used `0.066692 MiB` traced
peak Python allocation, kept archival/source/score placement on CPU, and
allocated `0.0 MiB` CUDA memory on RTX 3060.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260617\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-snn-rollout-rehearsal-source-window.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6339.682 tokens/sec`, with
`train_compute=0.129022 ms/token`, `prepare_training=0.006321 ms/token`,
`finalize_total=0.006030 ms/token`, and `tick_duration_ms.p95=20.305`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported CPU max `34%`, GPU max `22%`, GPU memory-util max
`17%`, and RTX 3060 memory moved from `1867` to `1865 MiB`. Because the sampler
reported `contention_observed`, this is accepted as same-band throughput
protection evidence, not contention-free hardware evidence. The rollout
rehearsal policy remains a slow/control-plane replay review surface, not
live-tick or every-token replay work.

## SNN Status Replay-Path Source Windows

`StatusReadModel` now bounds replay-path Runtime Truth projection before
exporting emission review-history, emission replay-design, and rollout
consolidation readiness. The old status projection materialized all retained
readout, emission-review, and rollout events before reporting capped readiness
fields. The replacement reads `16` recent emission reviews for
`bounded_snn_status_emission_review_history_source_window.v1`, `16` recent
emission reviews plus `16` recent internal readout events for
`bounded_snn_status_emission_replay_design_path_source_window.v1`, and `16`
recent rollout events plus `16` recent readout events for
`bounded_snn_status_rollout_consolidation_path_source_window.v1`. All three
surfaces report retained counts, source counts, truncation, CPU archival/score
placement, no global candidate or score scan, no raw text payload, no hidden
language reasoning, `runs_live_tick=false`, `runs_every_token=false`, and
`gpu_used=false`.

Focused status source-window benchmark:

`python -m marulho.evaluation.status_replay_path_source_window_benchmark --retention-count 2048 --runs 25 --output reports\bounded_replay_window_20260618\status-replay-path-source-window.json`

It matched the diagnostic full-retained latest history, emission, and rollout
evidence while checking `80` source records instead of `10240` retained records
(`128x` less projection work). The bounded combined mean was `1.309999 ms`
versus `102.831789 ms` for the benchmark-local retired full-scan projection
(`78.497629x`). Emission design projection averaged `2.925552 ms` versus
`276.103140 ms`; rollout consolidation projection averaged `0.550184 ms`
versus `21.263004 ms`; emission review-history projection averaged
`0.454260 ms` versus `11.129224 ms`. The report kept Python traced peak
allocation at `2.121575 MiB`, used CPU archival/score placement, and
allocated/reserved `0.0 MiB` CUDA memory on RTX 3060.

The 65536-column protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-profile.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --profile-trainer-stages`

It processed `524288` tokens at `6081.034 tokens/sec`, with
`train_compute=0.134328 ms/token`, `prepare_training=0.006811 ms/token`,
`finalize_total=0.006355 ms/token`, and `tick_duration_ms.p95=21.243`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native sequence, and native
burst failures were all `0`; the velocity sampler reported no observed
contention, CPU max `49%`, GPU max `13%`, GPU memory-util max `18%`, and RTX
3060 memory moved from `2164` to `2159 MiB`.

The no-profile same-code rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-status-replay-path-source-window-noprofile-rerun.json --target-tokens 524288 --tick-tokens 128 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05`

It processed `524288` tokens at `6408.252 tokens/sec`, with
`train_compute=0.127614 ms/token`, `prepare_training=0.006245 ms/token`,
`finalize_total=0.005869 ms/token`, `tick_duration_ms.p95=19.945`, bounded
`12/65536` route rows, `65526` cached transition rows, and zero
graph/native/sequence failures. The sampler observed GPU-side contention before
measurement (`27%` max GPU utilization, `26%` memory utilization), so this is
same-band throughput protection evidence rather than contention-free hardware
evidence.

The SNN readout-ledger normalization/store-state no-profile protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-ledger-store-state-window-noprofile-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6044.412 tokens/sec`, with
`train_compute=0.134651 ms/token`, `prepare_training=0.007100 ms/token`,
`finalize_total=0.006343 ms/token`, and `tick_duration_ms.p95=21.680`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `25%`, GPU max `13%`,
GPU memory-util max `18%`, and RTX 3060 memory moved `2029->2032 MiB`. The
profiled companion run succeeded but reached `5953.828 tokens/sec`, so it is
stage-profile evidence rather than the primary throughput gate. This is
same-band throughput protection for the ledger normalization/store cleanup, not
a new top-speed claim.

The known-readout-evidence hash source-window protection rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-known-readout-hash-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `5938.461 tokens/sec`, with
`train_compute=0.136714 ms/token`, `prepare_training=0.007355 ms/token`,
`finalize_total=0.006530 ms/token`, and `tick_duration_ms.p95=22.329`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `51%`, GPU max `13%`,
GPU memory-util max `18%`, and RTX 3060 memory moved `2032->2031 MiB`. The
first same-code run reached `5871.364 tokens/sec`, so this is same-band
throughput protection for a slow replay/readout helper, not a speed promotion or
new ceiling.

The dense-label calibration source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-calibration-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6018.915 tokens/sec`, with
`train_compute=0.135317 ms/token`, `prepare_training=0.006959 ms/token`,
`finalize_total=0.006455 ms/token`, and `tick_duration_ms.p95=21.517`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `64%`, GPU max `16%`,
GPU memory-util max `20%`, and RTX 3060 memory moved `2030->2029 MiB`. This is
same-band throughput protection for a slow dense-label calibration/readout
helper, not a speed promotion.

The dense-label calibration evaluation source-window first run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It succeeded at `5906.886 tokens/sec`; this is retained as below-band variance
evidence, not the protection claim. The accepted rerun was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-evaluation-source-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6116.710 tokens/sec`, with
`train_compute=0.133135 ms/token`, `prepare_training=0.006912 ms/token`,
`finalize_total=0.006197 ms/token`, and `tick_duration_ms.p95=21.705`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `41%`, GPU max `13%`,
GPU memory-util max `18%`, and RTX 3060 memory moved `2030->2030 MiB`. This is
same-band throughput protection for the slow dense-label calibration evaluation
gate, not a speed promotion.

The dense-label calibration update source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-label-update-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

It processed `524288` tokens at `6009.497 tokens/sec`, with
`train_compute=0.134959 ms/token`, `prepare_training=0.007078 ms/token`,
`finalize_total=0.006414 ms/token`, and `tick_duration_ms.p95=22.051`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported CPU max `34%`, GPU max `21%`, GPU memory-util max
`23%`, and RTX 3060 memory moved `2045->2046 MiB`; the `21%` GPU sample marks
this as in-band protection evidence under observed contention, not a new speed
claim.

The autonomous confidence-use source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-confidence-use-source-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

It processed `524288` tokens at `5965.377 tokens/sec`, with
`train_compute=0.136087 ms/token`, `prepare_training=0.007205 ms/token`,
`finalize_total=0.006409 ms/token`, and `tick_duration_ms.p95=22.526`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `29%`, GPU max `15%`,
GPU memory-util max `19%`, and RTX 3060 memory moved `2045->2047 MiB`. This is
same-band throughput protection for a slow confidence-use ledger helper, not a
speed promotion.

The readout-ledger record-family append source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-record-family-append.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

It processed `524288` tokens at `5966.765 tokens/sec`, with
`train_compute=0.136141 ms/token`, `prepare_training=0.007153 ms/token`,
`finalize_total=0.006368 ms/token`, and `tick_duration_ms.p95=22.420`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `22%`, GPU max `13%`,
GPU memory-util max `18%`, and RTX 3060 memory moved `2046->2043 MiB`. This is
same-band throughput protection for the slow readout-ledger record append
cleanup, not a speed promotion.

The autonomous hash-readout binding/observation source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-autonomous-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

It processed `524288` tokens at `6272.156 tokens/sec`, with
`train_compute=0.130202 ms/token`, `prepare_training=0.006310 ms/token`,
`finalize_total=0.005864 ms/token`, and `tick_duration_ms.p95=21.100`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported no observed contention, CPU max `7%`, GPU max `17%`,
GPU memory-util max `20%`, and RTX 3060 memory moved `2044->2045 MiB`. This is
same-band throughput protection for the slow autonomous hash-readout ledger
cleanup, not a speed promotion.

The autonomous training-window/decoder-probe source-window protection run was:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports/column_scheduler_20260618/checkpoints/active-pressure-scheduler-65536-seeded.pt --output reports/bounded_replay_window_20260619/hotpath-active-pressure-65536-524288-i32-training-probe-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

It processed `524288` tokens at `6057.953 tokens/sec`, with
`train_compute=0.134322 ms/token`, `prepare_training=0.006849 ms/token`,
`finalize_total=0.006282 ms/token`, and `tick_duration_ms.p95=21.917`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`. Graph, native burst, and native
sequence failures were all `0`; conditional-WHILE q16 remained active. The
velocity sampler reported GPU-side contention (`24%` max GPU utilization), CPU
max `46%`, GPU memory-util max `20%`, and RTX 3060 memory moved
`2046->2064 MiB`. This is same-band throughput protection for the slow
autonomous training/probe ledger cleanup under contention, not a speed
promotion or clean ceiling.

## Strong-Capture Admission Cadence

Strong-capture slow-memory admission now has its own refractory cadence, so a
low threshold cannot turn every threshold crossing into a `DualMemoryStore`
archive write. The device strong-event ring still records strong-event evidence,
but archival admission is selected by
`slow_memory_archive_strong_capture_min_interval_tokens` and stays CPU-resident.
Production config rejects values `<=1`; the default is `16`.

Focused quality benchmark:

`python -m marulho.evaluation.strong_capture_admission_cadence_benchmark --tokens 256 --min-interval-tokens 16 --runs 10 --output reports\bounded_replay_window_20260618\strong-capture-admission-cadence.json`

Result: `pass=true`, bounded archive writes `17`, strong captures archived
`16`, refractory skips `239`, max selected gap `16`, final gap `14`, and
bounded mean latency `1172.027720 ms` over `10` runs. The retired every-strong
admission shape is projected from the forced-strong candidate count, not
executed as a side path: `256` projected writes, `15.058824x` write reduction.
The report states archival storage `cpu`, active replay computation `none`, no
GPU use, `0.0 MiB` CUDA allocation/reservation, no global candidate or score
scan, and no hidden language reasoning.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-strong-capture-admission-cadence.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6100.415 tokens/sec`, with
`train_compute=0.133405 ms/token`, `prepare_training=0.007070 ms/token`,
`finalize_total=0.006437 ms/token`, and `tick_duration_ms.p95=21.328`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native/sequence failures were
all `0`. The ordinary live tick reported
`slow_memory_strong_capture_archive_count=0`,
`slow_memory_strong_capture_refractory_skip_count=0`, and
`slow_memory_last_strong_capture_token=-1`, proving this slice did not add a
new always-on archive workload. RTX 3060 memory stayed flat at `2390 MiB`.
The environment sampler observed GPU-side contention, so this is accepted as
same-band hot-path protection evidence rather than a clean top-speed ceiling.

Same-code rerun:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260618\hotpath-active-pressure-65536-524288-i32-strong-capture-admission-cadence-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

The rerun succeeded at `5326.602 tokens/sec`, with
`train_compute=0.157298 ms/token`, bounded `12/65536` route rows, zero
runtime failures, and flat `2390 MiB` GPU memory, but a `12435 ms` max tick
outlier and observed GPU contention make it variance evidence rather than a
promotion run.

## SNN Readout Replay Target Windows

`SNNLanguageReadoutEvidenceLedger.replay_dry_run(...)`,
`plasticity_preflight(...)`, and `plasticity_replay_bridge(...)` now cap
caller-supplied replay payload windows to `32` records before sparse tensor
materialization, plasticity preflight, or bridge canonicalization. The old
full-payload shape is not kept as an executable comparison path; the focused
benchmark projects the retired work from source counts only.

Focused quality benchmark:

`python -m marulho.evaluation.readout_replay_target_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\readout-replay-target-window.json`

Result: `pass=true`, dry-run `32/2048`, bridge `32/2048`, `64x` work
reduction on both surfaces, mean dry-run latency `6.061784 ms`, mean bridge
latency `1.328924 ms`, CPU archival/source/replay placement, `0.0 MiB` CUDA
allocation/reservation, no global candidate/score scan, no raw replay text, no
hidden language reasoning, no live tick, and no every-token cadence.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-readout-replay-target-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6109.000 tokens/sec`, with
`train_compute=0.133186 ms/token`, `prepare_training=0.006965 ms/token`,
`finalize_total=0.006289 ms/token`, and `tick_duration_ms.p95=21.677`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. The environment sampler reported no observed contention, CPU max
`31%`, GPU max `13%`, GPU memory-util max `18%`, and RTX 3060 memory
`2020->2018 MiB`. This remains same-band throughput protection for a
slow/control-plane readout replay cleanup, not a new live replay path.

## SNN Language Plasticity Replay Windows

The semantics-level language plasticity replay path now owns the same replay
budget as the API schema. `evaluate_spike_language_plasticity_replay(...)` and
`run_spike_language_plasticity_replay_experiment(...)` inspect at most `32`
caller-supplied replay records. `build_spike_language_plasticity_shadow_delta(...)`
also caps per-record sparse sides to `16` indices before pair scoring, so
shadow-delta work is bounded by selected records and selected sparse indices
rather than by caller payload size.

Focused quality benchmark:

`python -m marulho.evaluation.language_plasticity_replay_window_benchmark --payload-count 2048 --index-count 256 --runs 25 --output reports\bounded_replay_window_20260619\language-plasticity-replay-window.json`

Result: `pass=true`, replay evaluation `32/2048`, replay experiment `32/2048`,
shadow-delta pair checks `8192/134217728`, `64x` record-work reduction,
`16384x` pair-work reduction, mean latencies `11.024580 ms`, `8.622980 ms`, and
`297.890092 ms`, CPU archival/source/replay placement, traced Python peak
allocation `14.474813 MiB`, `0.0 MiB` CUDA allocation/reservation, no global
candidate/score scan, no raw replay text, no hidden language reasoning, no live
tick, and no every-token cadence.

Long protection rerun:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-plasticity-replay-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `5999.398 tokens/sec`, with
`train_compute=0.135445 ms/token`, `prepare_training=0.007140 ms/token`,
`finalize_total=0.006422 ms/token`, and `tick_duration_ms.p95=22.016`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. `velocity_environment.v1` observed GPU-side contention at `22%`, with
CPU max `38%`, GPU memory-util max `23%`, and RTX 3060 memory `2018->2023 MiB`.
The first same-command run reached `6025.620 tokens/sec` but was also marked
GPU-contended. These runs prove same-band protection for a slow/control-plane
cleanup, not a clean speed ceiling.

## SNN Language Application Synapse Windows

The checkpointed SNN language application boundary now refuses caller-sized
application payloads before checkpoint writes. `apply_live_application(...)`
and `regenerate_transition_memory(...)` inspect at most `32` synapse candidates
and require the source payload to be untruncated before mutation. The old
full-payload mutation-side behavior is recorded as a retired projection, not an
executable comparison path.

Focused quality benchmark:

`python -m marulho.evaluation.language_application_synapse_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\language-application-synapse-window.json`

Result: `pass=true`, oversized live application `32/2048` blocked, oversized
regeneration `32/2048` blocked, zero checkpoint calls for oversized payloads,
zero state mutation, exact-window live application applied `32` synapses, exact
regeneration added `32` synapses, `64x` projected source-work reduction, CPU
archival/source/application placement, traced Python peak allocation
`1.982166 MiB`, `0.0 MiB` CUDA allocation/reservation, no global candidate/score
scan, no raw text payload, no hidden language reasoning, no live tick, and no
every-token cadence.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-application-synapse-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32`

It processed `524288` tokens at `6039.734 tokens/sec`, with
`train_compute=0.134728 ms/token`, `prepare_training=0.006949 ms/token`,
`finalize_total=0.006436 ms/token`, and `tick_duration_ms.p95=21.511`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. The environment sampler reported no observed contention, CPU max
`29%`, GPU max `16%`, GPU memory-util max `19%`, and RTX 3060 memory
`2020->2034 MiB`. This keeps the checkpointed application cleanup out of the
live tick while preserving the maintained 6k-ish throughput band.

## Rollout Regeneration Facade Candidate Windows

The rollout-regeneration facade now refuses caller-sized candidate payloads
before permit issuance, application preflight, or checkpoint-backed application.
Permit, preflight, and application all use the executor-owned
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32` source-window operator and
require the source payload to be untruncated before calling the replay
controller or executor. The old full-payload facade behavior is retired, not
kept as a compatibility route.

Focused quality benchmark:

`python -m marulho.evaluation.rollout_regeneration_facade_candidate_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\rollout-regeneration-facade-candidate-window.json`

Result: `pass=true`, oversized permit `32/2048` blocked before the replay
controller, oversized preflight `32/2048` blocked before proposal readiness,
oversized application `32/2048` blocked before the executor, zero checkpoint
writes for oversized applications, exact-window flow still advanced through one
permit and one executor call with `32` candidates, `64x` projected source-work
reduction, CPU archival, source-window, facade-gate, and active-application placement, traced Python peak
allocation `1.852119 MiB`, `0.0 MiB` CUDA allocation/reservation, no global
candidate/score scan, no raw text payload, no hidden language reasoning, no
live tick, and no every-token cadence.

Long protection runs:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-rollout-regeneration-facade-candidate-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

The first same-code run succeeded but is below-band variance evidence:
`5938.820 tokens/sec`, `train_compute=0.137347 ms/token`,
`prepare_training=0.007065 ms/token`, `finalize_total=0.006444 ms/token`,
bounded `12/65536` route rows, `65526` cached rows, no observed contention, GPU
memory `2033->2031 MiB`, and zero graph/native sequence failures.

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-rollout-regeneration-facade-candidate-window-rerun.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

The accepted rerun processed `524288` tokens at `6121.143 tokens/sec`, with
`train_compute=0.133293 ms/token`, `prepare_training=0.006856 ms/token`,
`finalize_total=0.006270 ms/token`, and `tick_duration_ms.p95=21.611`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. The environment sampler observed light GPU contention at `23%`, CPU
max `36%`, GPU memory-util max `23%`, and flat RTX 3060 memory at `2031 MiB`.
This is accepted hot-path protection evidence under contention, not a clean
speed-ceiling claim.

## Readout Ledger Rollout Candidate Windows

The upstream readout-ledger rollout chain now refuses caller-sized structural
candidate payloads before permit preview or Replay Controller permit hashing.
Consolidation design, shadow delta, developmental plasticity review,
regeneration adapter, regeneration replay-artifact review, and direct
Replay Controller normalization all use the shared
`SNN_LANGUAGE_APPLICATION_SYNAPSE_WINDOW_LIMIT=32` source-window operator and
require untruncated source payloads.

Focused quality benchmark:

`python -m marulho.evaluation.readout_ledger_rollout_candidate_window_benchmark --payload-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\readout-ledger-rollout-candidate-window.json`

Result: `pass=true`, exact rollout evidence reached permit preview with
`32/32` candidates; oversized design, shadow, developmental, adapter,
replay-artifact review, and direct Replay Controller normalization all blocked
at `32/2048`; projected source work fell `64x`; traced Python peak allocation
was `9.073439 MiB`; CUDA allocation/reservation stayed `0.0 MiB`; archival
storage, source-window selection, and review gates stayed CPU-resident; and
the reports state no global candidate/score scan, no raw text payload, no
hidden language reasoning, no live tick, and no every-token cadence.

Hot-path protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-readout-ledger-rollout-candidate-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

Result: `success=true`, `524288` tokens in `86.298389 s`,
`6075.293 tokens/sec`, `train_compute=0.134312 ms/token`,
`prepare_training=0.006819 ms/token`, and `finalize_total=0.006250 ms/token`.
Runtime Truth kept route scoring bounded at `12/65536` input rows and `10`
output candidates, with `65526` cached transition rows,
`state_transition_runs_all_columns=false`, `route_vote_rows_run_all_columns=false`,
and zero route-vote/native sequence failures. Contention was not observed
(`cpu max=35%`, `gpu max=15%`); the live runtime used CUDA on the RTX 3060,
while the candidate-window benchmark kept archival/source/review work on CPU.
GPU memory moved from `2031 MiB` to `2043 MiB`.

## Dense Readout Training Transition Windows

The checkpointed dense-readout training boundary now refuses caller-sized
transition and sparse-index payloads before checkpoint writes. The schema,
read-model design, preflight, and executor all share a `32` transition window
and `32` pre/post index windows, and oversized payloads must fail closed rather
than silently training on a prefix.

Focused quality benchmark:

`python -m marulho.evaluation.dense_readout_training_transition_window_benchmark --payload-count 2048 --index-count 2048 --runs 25 --output reports\bounded_replay_window_20260619\dense-readout-training-transition-window.json`

Result: `pass=true`, oversized transition payload `32/2048` blocked, oversized
index payload `32/2048` blocked, zero checkpoint calls for oversized payloads,
zero state mutation, exact-window training applied `32` dense/sparse updates,
`64x` projected transition-record reduction, CPU archival/source/training
placement, traced Python peak allocation `5.696876 MiB`, `0.0 MiB` CUDA
allocation/reservation, no global candidate/score scan, no raw text payload, no
hidden language reasoning, no live tick, and no every-token cadence.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-dense-readout-training-transition-window.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 16 --source-concept-observation-tick-interval 4 --timeout-seconds 900 --sample-interval-seconds 0.05 --host-truth-sync-interval-tokens 32 --profile-trainer-stages`

It processed `524288` tokens at `6028.820 tokens/sec`, with
`train_compute=0.135088 ms/token`, `prepare_training=0.007078 ms/token`,
`finalize_total=0.006280 ms/token`, and `tick_duration_ms.p95=21.702`.
Runtime Truth stayed bounded at `route_input_rows_scored=12/65536`,
`route_output_candidate_count=10`, `state_transition_cached_count=65526`, and
`state_transition_runs_all_columns=false`; graph/native sequence failures were
all `0`. The environment sampler reported no observed contention, CPU max
`54%`, GPU max `15%`, GPU memory-util max `18%`, and RTX 3060 memory
`2029->2028 MiB`. This keeps checkpointed transition training outside the live
tick while preserving the maintained 6k-ish throughput band.

## Autonomous Output-Chain Ledger Windows

Language-output and decoded-output execution/review no longer normalize every
readout-ledger event family to append or review one hash-only output event. The
production path now uses `bounded_snn_readout_ledger_record_family_source_window.v1`
for `autonomous_language_output_events` and
`autonomous_decoded_output_events`, matching the preceding binding,
observation, training-window, and decoder-probe event families.

Focused quality benchmark:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 3 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-output-chain.json`

Result: `pass=true`; the bounded chain preserved hash, review-match, and
total-count parity across binding, observation, training, decoder probe,
language output, and decoded output. Checked source rows fell from `35328` to
`1536` (`23x`), and mean chain latency fell from `6778.768800 ms` to
`321.988933 ms` (`21.052801x`). Traced Python peak allocation was
`447.571689 MiB`; CUDA was available on the RTX 3060 but the benchmark used no
GPU allocation/reservation for archival ledger metadata.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-output-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

Result: `success=true`, `524288` tokens in `86.678694 s`,
`6048.638 tokens/sec`, `tick_duration_ms.p95=21.307`,
`train_compute=0.134492 ms/token`, `prepare_training=0.006912 ms/token`, and
`finalize_total=0.006334 ms/token`. Runtime Truth kept route scoring bounded
at `12/65536` input rows and `10` output candidates, with `65526` cached
transition rows, `state_transition_runs_all_columns=false`, and zero
graph/native sequence failures. The environment sampler observed GPU
contention (`cpu max=23%`, `gpu max=36%`, `gpu memory util max=26%`), so this
is same-band protection evidence rather than a clean speed ceiling. Runtime
CUDA memory moved `2046->2047 MiB`; the replay/ledger benchmark itself kept
archival/source/review metadata on CPU.

## Autonomous Text-Surface Ledger Windows

Bounded text-emission execution/review and text-surface commit execution/review
no longer normalize every readout-ledger event family. The production path now
uses `bounded_snn_readout_ledger_record_family_source_window.v1` for
`autonomous_bounded_text_emission_events` and
`autonomous_text_surface_commit_events`, while preserving
`current_text_surface_commit` as the single current pointer.

Focused quality benchmark:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 3 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-text-surface-chain.json`

Result: `pass=true`; the bounded chain preserved hash, review-match,
total-count, and current-commit parity across binding, observation, training,
decoder probe, language output, decoded output, bounded text emission, and
text-surface commit. Checked source rows fell from `47104` to `2048` (`23x`),
and mean chain latency fell from `9289.008333 ms` to `429.436800 ms`
(`21.630676x`). Traced Python peak allocation was `442.869928 MiB`; CUDA was
available on the RTX 3060 but the benchmark used no GPU allocation/reservation
for archival ledger metadata.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-text-surface-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

Result: `success=true`, `524288` tokens in `87.663099 s`,
`5980.715 tokens/sec`, `tick_duration_ms.p95=22.136`,
`train_compute=0.135992 ms/token`, `prepare_training=0.007115 ms/token`, and
`finalize_total=0.006345 ms/token`. Runtime Truth kept route scoring bounded
at `12/65536` input rows and `10` output candidates, with `65526` cached
transition rows, `state_transition_runs_all_columns=false`, and zero
graph/native sequence failures. The environment sampler reported no observed
contention (`cpu max=65%`, `gpu max=14%`, `gpu memory util max=18%`). Runtime
CUDA memory moved `2045->2047 MiB`; the replay/ledger benchmark itself kept
archival/source/review metadata on CPU.

## Autonomous Language-Surface Ledger Windows

Text-surface materialization and bounded language-surface commit no longer
normalize every readout-ledger event family. The production path now uses
`bounded_snn_readout_ledger_record_family_source_window.v1` for
`autonomous_text_surface_materialization_events` and
`autonomous_bounded_language_surface_commit_events`, while preserving
`current_text_surface_materialization` and
`current_bounded_language_surface_commit` as the single current pointers.

Focused quality benchmark:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 3 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-language-surface-chain.json`

Result: `pass=true`; the bounded chain preserved hash, review-match,
total-count, and current-pointer parity across binding, observation, training,
decoder probe, language output, decoded output, bounded text emission,
text-surface commit, text-surface materialization, and bounded language-surface
commit. Checked source rows fell from `58880` to `2560` (`23x`), and mean chain
latency fell from `11175.229267 ms` to `525.534133 ms` (`21.264517x`). CUDA was
available on the RTX 3060 but the benchmark used no GPU execution for archival
ledger metadata; archival/source/review placement stayed CPU-only with
`runs_live_tick=false`, `runs_every_token=false`, and `language_reasoning=false`.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-surface-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

Result: `success=true`, `524288` tokens in `87.467930 s`,
`5994.060 tokens/sec`, `tick_duration_ms.p95=21.991`,
`train_compute=0.135570 ms/token`, `prepare_training=0.007057 ms/token`, and
`finalize_total=0.006414 ms/token`. Runtime Truth kept route scoring bounded
at `12/65536` input rows and `10` output candidates, with `65526` cached
transition rows, `state_transition_runs_all_columns=false`, and zero
graph/native sequence failures. The environment sampler observed GPU-side
contention at the threshold (`cpu max=20%`, `gpu max=21%`, `gpu memory util
max=23%`), so this is same-band protection evidence rather than a clean
speed-ceiling claim. Runtime CUDA memory moved `2044->2059 MiB`; the
replay/ledger benchmark itself kept archival/source/review metadata on CPU.

## Autonomous Language-Generation Ledger Windows

Bounded language-surface use and SNN language-generation now use the same
record-family source-window path instead of normalizing every retained
readout-ledger event family. The production path uses
`bounded_snn_readout_ledger_record_family_source_window.v1` for
`autonomous_bounded_language_surface_use_events` and
`autonomous_snn_language_generation_events`.

Focused quality benchmark:

`python -m marulho.evaluation.snn_readout_ledger_normalization_source_window_benchmark --retention-count 2048 --ledger-limit 128 --runs 3 --output reports\bounded_replay_window_20260619\snn-readout-ledger-normalization-language-generation-chain.json`

Result: `pass=true`; the bounded chain preserved hash, review-match,
total-count, and current-pointer parity across binding, observation, training,
decoder probe, language output, decoded output, bounded text emission,
text-surface commit, text-surface materialization, bounded language-surface
commit, bounded language-surface use, and SNN language-generation. Checked
source rows fell from `70656` to `3072` (`23x`), and mean chain latency fell
from `13505.919533 ms` to `631.221 ms` (`21.396499x`). CUDA was available on
the RTX 3060 but the benchmark used no GPU execution for archival ledger
metadata; archival/source/review placement stayed CPU-only with
`runs_live_tick=false`, `runs_every_token=false`, and `language_reasoning=false`.

Long protection run:

`python -m marulho.evaluation.continuous_runtime_stress_benchmark --checkpoint reports\column_scheduler_20260618\checkpoints\active-pressure-scheduler-65536-seeded.pt --output reports\bounded_replay_window_20260619\hotpath-active-pressure-65536-524288-i32-language-generation-chain.json --target-tokens 524288 --tick-tokens 128 --quantum-tokens 32 --source-concept-observation-tick-interval 4 --timeout-seconds 300 --sample-interval-seconds 0.02`

Result: `success=true`, `524288` tokens in `86.310830 s`,
`6074.417 tokens/sec`, `tick_duration_ms.p95=21.376`,
`train_compute=0.133727 ms/token`, `prepare_training=0.007038 ms/token`, and
`finalize_total=0.006252 ms/token`. Runtime Truth kept route scoring bounded
at `12/65536` input rows and `10` output candidates, with `65526` cached
transition rows, `state_transition_runs_all_columns=false`, and zero
graph/native sequence failures. The environment sampler reported no observed
contention (`cpu max=50%`, `gpu max=13%`, `gpu memory util max=18%`). Runtime
CUDA memory moved `2044->2047 MiB`; the replay/ledger benchmark itself kept
archival/source/review metadata on CPU.
