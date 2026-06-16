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
  - ../../../src/marulho/evaluation/compiled_column_kernel_benchmark.py
  - ../../../src/marulho/evaluation/compiled_hot_path_kernel_benchmark.py
  - ../../../src/marulho/evaluation/predictive_transition_benchmark.py
  - ../../../tests/test_service_benchmark.py
related_docs: []
related_papers: []
related_benchmarks: []
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
`velocity_environment.v1` contention `not_observed`. The explicit opt-out
native8 run at
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
run used `--sequence-executor default`, selected the native repeated-child
fallback instead of conditional-WHILE, and observed contention; do not compare
that run to the 6k-ish baseline.

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
exactly. This promotes the fused Triton candidate writeback to the next runtime
integration candidate, not to live scheduler truth: `ColumnTransitionRuntime`
still reports dense predictive update/location until the kernel is integrated
with the transition state boundary and wins complete `train_step` evidence.

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
predictive executions, `132647424` cached predictive rows, zero graph/sequence
fallbacks, and `contention.verdict=not_observed`. This restores the 6k-ish
sustained band while making candidate predictive update/location a real
scheduler execution effect rather than a report-only claim.

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
- Compiled column-kernel benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_column_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_column_kernel_cuda_1024/compiled-column-kernel-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8` measured fixed-shape, candidate-scoped competition only. After installing `triton-windows`, the runner auto-selected the bundled TinyCC compiler at `triton/runtime/tcc/tcc.exe`. On RTX 3060, eager batched execution reached `79651.68 isolated tokens/sec`; `torch_compile_default` reached `178071.38 isolated tokens/sec` with median/p95 `1.1134/2.7291 ms`; `torch_compile_reduce-overhead` reached `212698.98 isolated tokens/sec` with median/p95 `1.01745/2.0355 ms`; compile errors were absent. This proves the isolated awake-column competition kernel can exceed a low 1000 tokens/sec reference floor by `212.70x`, but does not include retrieval, trainer orchestration, plasticity, memory, binding, cross-modal grounding, checkpointing, or service throughput. The actual goal remains maximum sustainable local throughput, not stopping at 1000 tokens/sec.
- Compiled hot-path-kernel benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high` measured input projection, candidate-scoped competition, and candidate-local predictive-state math without runtime writeback. On RTX 3060, eager batched execution reached `45118.41 isolated tokens/sec`; `torch_compile_default` reached `242784.01 isolated tokens/sec` with median/p95 `0.7662/2.677 ms`; `torch_compile_reduce-overhead` reached `206492.85 isolated tokens/sec` with median/p95 `0.91425/2.8345 ms`; compile errors were absent. A default-precision comparison reached `203483.21 isolated tokens/sec`, so precision policy remains measured evidence rather than a static assumption. This benchmark still excludes retrieval, Python trainer orchestration, in-place plasticity, memory/replay, binding, cross-modal grounding, checkpointing, and service throughput.
- Real-candidate compiled hot-path benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-routing-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high --candidate-source routing_index` used live sharded `torch_topk` routing candidates. Candidate preparation through the legacy list-returning API measured `354.905 ms` for 256 tokens (`721.32 tokens/sec`) with zero fallback rows, while the compiled block reached `191913.60 isolated tokens/sec`. This exposed Python-list/CPU-numpy candidate extraction as a bottleneck and is not a production routing ceiling.
- Tensor-routing compiled hot-path benchmark on 2026-06-11: `python -m marulho.evaluation.compiled_hot_path_kernel_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/compiled_hot_path_kernel_cuda_1024/compiled-hot-path-kernel-routing-tensor-benchmark.json --batch-size 256 --iterations 128 --warmup-iterations 8 --matmul-precision high --candidate-source routing_index_tensor` used retrieval-owned `search_tensors()` to keep candidate ids and distances on CUDA. Cold candidate prep after checkpoint/cache setup measured `212.7566 ms` for 256 tokens (`1203.25 tokens/sec`); warm cached prep measured `4.7468 ms` (`53931.07 tokens/sec`), both with zero fallback rows and sharded torch caches on `cuda:0`. The compiled block reached `234286.58 isolated tokens/sec` with `torch_compile_reduce-overhead`, median/p95 `0.959/2.0064 ms`, compile errors absent, and CUDA memory `13.266/56.0 MB` allocated/reserved. This is still an isolated benchmark: retrieval prep, kernel math, runtime mutation, trainer orchestration, service endpoints, and checkpointing remain separate evidence surfaces.
- Live tensor-routing hot-window A/B on 2026-06-11: the benchmark gained `--routing-candidate-mode list|tensor` as an evaluation-only switch, while the trainer default uses retrieval-owned tensor candidates. Two sequential, reversed-order 256-sample pairs on the same revision-960 checkpoint and seed measured tensor/list throughput of `24.7167/23.9371` and `32.6721/32.3280 tokens/sec`. Tensor routing improved median latency in both pairs (`38.7952` versus `40.18805 ms`, then `26.68645` versus `27.5149 ms`); p95 was mixed (`62.7875` versus `60.0293 ms`, then `52.8824` versus `53.0974 ms`). Runtime counters proved `288` tensor searches and zero list searches in tensor arms, with the inverse in list arms; allocated/reserved VRAM remained `20.6758/50.0 MB`. The promotion is a modest roughly `2%` aggregate throughput improvement and does not close the gap between eager `train_step` and compiled-kernel capacity.
- Merged sharded-torch routing benchmark on 2026-06-11: `compiled_hot_path_kernel_benchmark --candidate-source routing_index_tensor` compared the default merged exact cache against `--disable-merged-torch-shards`. Warm 256-token candidate preparation improved from `6.7612 ms` (`37863.10 tokens/sec`) to `2.0147 ms` (`127066.06 tokens/sec`), a `3.36x` routing improvement with zero fallback rows. The merged cache held all 1024 normalized vectors and ids on `cuda:0`, adding `532480` bytes. Two reversed-order hot-window pairs measured merged/per-shard throughput of `38.5582/37.0260` and `37.3134/36.8980 tokens/sec`; median latency improved from `26.363` to `25.01605 ms` and from `26.0075` to `25.2821 ms`. P95 was mixed, so no tail-latency claim is made. Shards still own add/remove/rebuild; every mutation invalidates the merged cache.
- Compiled dense predictive transition benchmark on 2026-06-11: `python -m marulho.evaluation.predictive_transition_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/objects/revision-960-manual_save-ba1eff946f2949f1857139a8f57b70ec.pt --output reports/predictive_transition_cuda_1024/predictive-transition-benchmark.json --iterations 512 --warmup-iterations 16` measured one 1024-column fixed-shape predictive state transition. Eager reached `204.35 transitions/sec`; compile default reached `497.90/sec`; compile reduce-overhead reached `1121.66/sec`, median/p95 `0.72745/1.8407 ms`, with no compile errors. Initial live integration failed closed because CUDA Graph-owned outputs were reused as persistent next-step state and were overwritten. The promoted path copies compiled outputs into stable runtime buffers outside the graph and passes repeated CUDA-step parity tests.
- Live compiled-predictive hot-window A/B on 2026-06-11: with merged tensor routing held constant, two reversed-order 256-token pairs measured compiled/legacy throughput of `47.8081/34.5995` and `42.0725/30.1816 tokens/sec`. Compiled median latency was `19.73835/22.0747 ms` versus legacy `27.00975/31.58265 ms`; compiled p95 was `30.3/34.9 ms` versus legacy `44.6731/49.9381 ms`. First-use warmup increased from about `1.3-1.6 s` to `8.7 s`, and reserved VRAM increased from `50` to `54 MB`. A later single steady-graph run measured `38.7050 tokens/sec`, median/p95 `24.4877/39.7961 ms`, compile count `1`, and no fallback, so the speed direction is proven but exact throughput remains environmentally variable.
- Dormant lightweight input-plasticity retirement on 2026-06-12: the revision-960 checkpoint has `input_weight_blend=0.0`, so winner input-weight rows could not contribute to competition or assembly output. A direct same-seed pre-change/skip probe measured `40.3786` versus `43.9311 tokens/sec`, median `23.81175` versus `21.03955 ms`, and p95 `36.6842` versus `34.5157 ms`; a reversed repetition measured `41.6699` versus `45.6825 tokens/sec`, median `22.6771` versus `20.83225 ms`, and p95 `33.6637` versus `32.6971 ms`. Two uncontended post-change 256-token confirmations measured `42.8296` and `41.9188 tokens/sec`, median `21.9380` and `22.52325 ms`, p95 `36.1204` and `38.4941 ms`, with `21.1924 MB` allocated and `54.0 MB` reserved VRAM. Runtime evidence reported `input_plasticity_mode=skipped_zero_blend`, zero updates, 296 skips, CUDA tensor execution, and sparse `10/1024` competition. Local STDP is intentionally excluded from this skip.
- Rejected functional steady-state transition on 2026-06-12: a pure fixed-shape transition combined competition, winner selection, dense prediction, prediction-error modulation, prototype/velocity plasticity, stale counters, and homeostasis with exact eager-module parity. In isolation, including eleven stable state-buffer copies, eager reached `64.9176 transitions/sec` with median/p95 `14.3822/22.8777 ms`; `torch.compile(mode="reduce-overhead")` reached `271.3224/sec` with median/p95 `2.9067/7.7558 ms`, a `4.18x` isolated gain. Full configured hot-window evidence rejected promotion. Pair A measured compiled/retained throughput `37.5324/36.6468 tokens/sec`; reversed Pair B measured `31.9652/36.6624`. Compiled average throughput was about `34.75` versus `36.65 tokens/sec`, latency results were mixed, and first-use warmup remained about `14 s`. Earlier runs while another CUDA backend held the GPU were excluded as contaminated. The functional transition is not imported by the always-on trainer and remains an evaluation oracle for a future in-place kernel.
- Canonical evaluation-oracle rerun at `reports/steady_state_column_transition_cuda_1024/steady-state-column-transition-benchmark.json`: on the idle RTX 3060, eager reached `55.6270 transitions/sec` with median/p95 `17.39165/24.5979 ms`; compiled reduce-overhead reached `414.0303/sec` with median/p95 `1.7943/6.9034 ms`, no compile errors, and `4.6431/14.0 MB` allocated/reserved VRAM. This reinforces the available isolated device capacity but does not reverse the full-tick rejection.
- In-place steady-state column kernel benchmark on 2026-06-12: `python -m marulho.evaluation.inplace_column_cuda_benchmark --checkpoint reports/service_benchmark_sparse_candidate_1024/runtime.pt --output reports/inplace_column_cuda_1024/inplace-column-cuda-benchmark.json --iterations 512 --warmup-iterations 32` measured candidate competition followed by one Triton launch mutating predictive state, winner prototype/velocity plasticity, candidate-scoped homeostasis, stale counters, spike history, and assembly. Functional eager stable-writeback reached `59.8240 transitions/sec`, median/p95 `16.3240/23.0268 ms`. Eager competition plus in-place Triton reached `314.5419/sec`, median/p95 `2.52885/6.8216 ms`, a `5.2578x` cluster speedup. Runtime state remained finite, all eleven state tensor addresses remained stable, and CUDA memory was `12.7666/28.0 MB` allocated/reserved. Warmup/compilation cost `10.3777 s`. This excludes encoder, routing search, context, binding, cross-modal grounding, memory writes, replay, service, and checkpointing, so promotion remains blocked on repeated complete-`train_step` A/B.
- Complete hot-window in-place A/B on 2026-06-12: after replacing the Python spike-ring cursor specialization with one persistent CUDA scalar, same-checkpoint 128-token Pair A measured in-place/runtime throughput `55.2774/38.0221 ticks/sec`, median `17.4702/25.8010 ms`, and p95 `24.2847/34.1031 ms`. Reversed Pair B measured `105.3975/37.9536`, median `8.6513/25.8147 ms`, and p95 `15.0841/34.2618 ms`. In-place VRAM was `11.7241/28.0 MB` allocated/reserved versus runtime `11.7183/30.0 MB`. The disk-cached in-place warmups were `1.85` and `0.65 s`; a cold diagnostic compiled two expected variants at about `98` and `73 s`, so cold startup remains unacceptable.
- Production lifecycle promotion on 2026-06-12: `ColumnTransitionRuntime` moved the in-place executor into trainer ownership with checkpoint opt-in, persistent workspace, compile-only warmup for all-column and `k=10` shapes, pre-mutation fallback, post-launch fail-closed behavior, and Runtime Truth counters. Empty-cache compile-only startup took `80.746 s`; populated disk-cache startup took `0.348 s`. A CUDA lifecycle test proves warmup leaves every model tensor bit-exact before the first execution.
- Production-backed complete hot-window evidence on 2026-06-12: in-place runs reached `80.5912` and `110.3597 ticks/sec`, with median/p95 `12.299/17.81 ms` and `8.50705/14.8878 ms`. Retained compiled observations reached `70.7727` and `51.5562 ticks/sec`, with median/p95 `13.5353/20.7919 ms` and `18.74655/27.9535 ms`. The strongest measured complete encoded hot window is `110.36 ticks/sec`, leaving a `9.06x` gap to the low `1000 ticks/sec` reference floor.
- Production-backed grounded quality evidence on 2026-06-12: the synthetic visual/audio gate passed at `42.8081` versus `38.7751 ticks/sec` (`1.104x`) with exact winners, bit-exact cross-modal weights/confidences, and zero measured Triton compile events. P95 was noisy in this run (`36.1636` versus `33.3056 ms`), so no tail-latency claim is made.
- Live service execution evidence on 2026-06-12: one real source tick processed 12 tokens in `3481.4487 ms`, or `3.44684 tokens/sec`. A clean post-restart repetition processed 12 tokens in `8646.97 ms`, or `1.39 tokens/sec`. Both runs reported requested/resolved `inplace_triton`, CUDA observation, 12 executions, zero failures, successful warmup, and revision progress. This proves production execution but also locates the next bottleneck and substantial variance outside the transition kernel in source/tick orchestration and remaining per-token stages.
- Live stage-profile optimization on 2026-06-12: an instrumented 12-token tick measured `7912.83 ms` in training and only `255.17 ms` in source collection. Splitting training attributed `5490.65 ms` to service-side concept observation versus `1488.05 ms` to `trainer_step`. Direct CPU profiling of three restored 94-concept observations measured `850.23 ms/call`; normalized-centroid caching reduced the warm result to `117.80 ms/call`, about `7.2x`. Sampling background observation at tokens 1, 8, and the final pending token raised the first cold live result to `4.338 tokens/sec`; same-process ticks reached `7.841`, `8.075`, and `8.438 tokens/sec`.
- Remote-source overlap evidence on 2026-06-12: scheduling the existing refill worker immediately after a consumed chunk lets provider I/O overlap CUDA training. Once warm, a configured 64-token tick collected its source window in `0.04 ms`, processed at `7.965 tokens/sec`, left 76 buffered tokens, and recorded a queue hit. One shorter 18-token tick reached `9.880 tokens/sec`. Source orchestration is no longer the dominant warm-path cost; the next gate is a synchronized stage profile inside `MarulhoTrainer.train_step`, whose 64-token contribution was `6601.45 ms`.
- Trainer profiler evidence on 2026-06-12: an explicit 12-tick PyTorch profiler slow path observed `1338` CUDA launches, `475` async copies, and `186` stream synchronizations. Reported CUDA operator work was about `1.2 ms/tick` while CPU orchestration averaged about `42 ms/tick`, so the configured path is launch/synchronization bound rather than arithmetic bound. A separate cProfile run with the promoted in-place executor active measured about `21 ms/tick`; the in-place transition was about `2.2 ms/tick`, while candidate competition, predictive voting, routing-key projection, and tensor HNSW lookup together consumed roughly `11 ms/tick`. The next implementation gate is candidate-scoped voting and a broader device-resident routing/competition cluster, followed by repeated full-`train_step` quality and throughput evidence.
- Device-resident winner-selection promotion on 2026-06-12: the active in-place transition now precompiles a one-block Triton selector that writes winner, unit strength, and positive-activation evidence into persistent CUDA buffers. The transition kernel consumes the boolean and preserves all-silent fallback threshold decay without the previous Python `values.max()` branch. Two reversed 256-tick complete hot-window pairs measured device/retained selection at `56.5076/58.7954` and `68.1373/59.9590 ticks/sec`; median latency was `16.1034/15.8209` and `13.4684/15.5395 ms`, while p95 was `29.5302/26.3276` and `24.2590/27.5564 ms`. Average throughput improved about `4.9%`; average p95 was effectively neutral, so no universal tail claim is made. The grounded 128-tick gate passed with exact winners, exact cross-modal tensors, finite state, zero measured Triton compilation events, and `35.9594` versus `25.4370 ticks/sec` against the retained transition baseline. Runtime Truth recorded 288 selector and transition executions with zero failures. One empty-process Windows compile-only warmup took `111.423 s`; populated-cache warmup was `0.687 s`, making cache packaging or prewarming a production startup requirement.
- Fused predictive vote/competition promotion on 2026-06-12: for the checkpoint-proven learned-chunk, zero-blend, one-winner shape without context, abstraction, or binding gain, `ColumnTransitionRuntime` now replaces the dense 1024-column predictive vote plus PyTorch candidate scoring with one candidate-local Triton launch. The kernel keeps the previous winner on-device and fuses reference-frame agreement, ten-candidate prototype score, threshold inhibition, positive/silent fallback, winner output, and previous-winner writeback. A 96-tick recurrent clone comparison matched `96/96` winners and was bit-exact for prototypes, thresholds, predictive locations, prediction error, and confidence. Reversed 256-tick complete hot-window pairs measured fused/unfused `84.1064/66.4301` and `141.2969/65.7654 ticks/sec`; median latency improved `10.4984/13.8884` and `6.0542/14.1862 ms`, while p95 improved `21.2732/23.4537` and `13.6325/24.2594 ms`. Both arms reserved `26 MB`; fused allocated only about `0.0005 MB` more. A 12-tick profiler reduced launches from `902` to `670`, async copies `155→149`, stream synchronizations `117→109`, CPU self-time `236.2→154.0 ms`, and CUDA self-time `49.9→7.8 ms`. The grounded 128-tick gate passed at `35.8357` versus `31.6136 ticks/sec`, exact winners, exact cross-modal tensors, finite state, and zero measured compilation events. Empty-process compile-only warmup was about `110.526 s`; cached warmup ranged `0.258–0.964 s`.
- Predictive-vote experiment and dead-state retirement on 2026-06-12: candidate-scoped voting measured `47.2868 ticks/sec`, median/p95 `19.7735/31.4794 ms`, versus dense voting at `71.7411 ticks/sec`, `13.0741/21.874 ms`. A separately compiled dense vote reached `75.8192 ticks/sec` versus an uncontended dense repetition at `109.8457 ticks/sec`, while adding `9.77 s` benchmark warmup. Both variants were rejected. Removing the unconsumed checkpointed `hypothesis` tensor reduced a 12-tick profiler window from `1338` to `1319` CUDA launches and `475` to `462` async copies, with stream synchronizations unchanged at `186`. Two post-retirement runs reached `117.9459` and `113.0185 ticks/sec`; the matched pre-retirement run was `109.8457`, but median results were mixed, so the durable claim is dead-state/launch removal plus p95 improvement from `13.9687` to `12.9947 ms`, not a universal throughput percentage.
- Routing normalization rejection on 2026-06-12: passing an already-normalized key through projection, tensor retrieval, and competition removed `109` launches over a 12-tick profiler window, but a 64-step checkpoint-clone comparison matched only `30` winners and produced maximum prototype/location/prediction-error differences of `0.01254/0.54929/0.00381`. Complete throughput was within or below environmental variance. The change was reverted; future fusion must preserve the full numerical trajectory or pass an explicit cognitive-quality gate.
- HNSW winner-ID reuse on 2026-06-12: the in-place transition already materializes the winner ID on CPU, so the trainer now passes that list into the 16-token HNSW update buffer instead of copying the same winner tensor to CPU again. A 12-tick profiler comparison reduced async copies from `463` to `450` and stream synchronizations from `186` to `174`, exactly one avoided transfer/sync per tick. Launch count was noisy (`1319` versus `1327`) and desktop GPU load contaminated throughput runs (`52-54 ticks/sec`), so no throughput percentage is claimed. The buffered IDs, vectors, deduplication, flush cadence, and index mutation remain unchanged.
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

- Deferred winner host materialization on 2026-06-13: the graph/in-place transition now returns the device winner tensor without forcing an immediate Python list. The trainer materializes the host winner id only for the first CPU-owned consumer and `_buffer_hnsw_update()` can carry CUDA id tensors until the HNSW flush boundary. `reports/cuda_graph_deferred_winner_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, and `surprise_update_count=24`. Compared with `reports/cuda_graph_cadenced_truth_20260613/service-benchmark-final2.json`, trainer-profile throughput rose from `92.5874` to `156.2895 tokens/sec`, total profiled cost fell from `10.8006` to `6.3984 ms/token`, `column_transition` fell from `1.5285` to `0.2873 ms/token`, and `column_transition_winner_readback` fell from `0.8222` to `0.0014 ms/token`. The host sync still exists as `winner_host_materialize=0.2495 ms/token`; remaining bottlenecks are `stream_text_context=2.9539`, `routing_prepare=1.2102`, `routing_index_buffer=0.5546`, and graph input copy/replay/sync.

- Raw-window archive text on 2026-06-13: live `train_step` no longer rebuilds expanded stream episode text just to write a slow-memory archive record. It stores the bounded raw window as replay text and leaves `_update_stream_text()` to slow/evaluation/display helpers. Compared with `reports/cuda_graph_deferred_winner_20260613/service-benchmark-final.json`, `reports/raw_window_archive_text_20260613/service-benchmark-final.json` reduced `stream_text_context` from `2.9539` to `0.0009 ms/token` while keeping CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, and `surprise_update_count=24`. Total profiled cost improved only from `6.3984` to `6.2952 ms/token` and throughput from `156.2895` to `158.8525 tokens/sec` because routing, host-truth sync, input copy, replay, and memory archive timing varied upward in that run. The durable claim is deletion of the old stream-context hot-path cost, not a broad throughput step.

- Cadenced winner host mirror on 2026-06-13: graph-backed text ticks now reuse winner ids already present in synced graph truth packets and skip separate host winner materialization on graph ticks where no CPU-owned consumer needs an exact id. Slow-memory archive bucket ids remain exact. `reports/cadenced_winner_host_mirror_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=7`, `host_truth_skip_count=17`, `surprise_update_count=24`, `graph_host_winner_reuse_count=7`, `winner_host_mirror_sync_count=7`, and `winner_host_mirror_skip_count=17`. Compared with `reports/raw_window_archive_text_20260613/service-benchmark-final.json`, `winner_host_materialize` disappeared from the per-tick profile, total profiled trainer cost fell from `6.2952` to `4.0150 ms/token`, and profiled trainer throughput rose from `158.8525` to `249.0665 tokens/sec`. The configured-source tick wall time remained noisy (`1873.75` versus `2656.94 ms`), so this is a trainer hot-path host-sync deletion, not an endpoint throughput claim.

- Host truth interval-8 default on 2026-06-13: after device-owned graph surprise and cadenced winner host mirroring, the production default for `cuda_graph_host_truth_sync_interval_tokens` moved from `4` to `8`. `reports/host_truth_interval_8_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_interval_tokens=8`, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, `winner_host_mirror_sync_count=4`, and `winner_host_mirror_skip_count=20`. Compared with `reports/cadenced_winner_host_mirror_20260613/service-benchmark-final.json`, `cuda_graph_prepare_host_truth_sync` fell from `0.4596` to `0.1055 ms/token`, `routing_prepare` fell from `1.8441` to `1.3480 ms/token`, total profiled trainer cost fell from `4.0150` to `3.8688 ms/token`, and profiled trainer throughput rose from `249.0665` to `258.4778 tokens/sec`. This is a Runtime Truth mirror metabolism improvement; exact per-token scalar parity remains an explicit interval-1 evaluation setting.

- Cadenced awake ripple tagging on 2026-06-13: high-dopamine awake ripple tagging now follows slow-memory archive cadence instead of attempting replay-priority memory scans on every warm-memory tick. `reports/awake_ripple_archive_cadence_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, `slow_memory_archive_count=3`, `slow_memory_archive_skip_count=21`, `awake_ripple_tag_count=3`, and `awake_ripple_tag_skip_count=21`. Compared with `reports/host_truth_interval_8_20260613/service-benchmark-final.json`, `post_surprise_replay_tag` fell from `0.4866` to `0.0685 ms/token`, total profiled trainer cost fell from `3.8688` to `3.1264 ms/token`, and profiled trainer throughput rose from `258.4778` to `319.8550 tokens/sec`. This is a replay-metabolism cleanup; it does not prove improved replay quality, and every-token ripple tagging remains retired unless consolidation evidence justifies the cost.

- Cross-modal fast idle skip on 2026-06-13: text-only ticks with no accepted sensory evidence, no residual trace, and no due self-criticism window now record cached idle state directly instead of walking the full cross-modal bookkeeping block. `reports/cross_modal_fast_idle_20260613/service-benchmark-final2.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, `host_truth_sync_count=4`, `host_truth_skip_count=20`, `surprise_update_count=24`, and Runtime Truth `cross_modal_hot_path` evidence reporting `fast_idle_skip_count=24`, `text_idle_skip_count=24`, and `text_update_count=0`. Compared with `reports/awake_ripple_archive_cadence_20260613/service-benchmark-final.json`, `cross_modal` fell from `0.6770` to `0.4933 ms/token`. Total profiled trainer cost regressed from `3.1264` to `3.7369 ms/token` because `routing_index_buffer` spiked from `0.4427` to `1.2420 ms/token` in that run, so the claim is limited to cross-modal idle bookkeeping reduction, not broad throughput.

- Batched HNSW winner-id flush on 2026-06-13: pending CUDA winner ids now materialize as one tensor batch during the explicit HNSW flush instead of one scalar read per buffered entry. The HNSW flush cadence is unchanged. A rejected snapshot-on-enqueue variant at `reports/hnsw_batched_id_flush_20260613/service-benchmark-final.json` regressed `routing_index_buffer` to `3.6198 ms/token` and profiled trainer throughput to `160.9413 tokens/sec`, so per-token CUDA id cloning was removed. The promoted batched-flush-only run at `reports/hnsw_batched_id_flush_20260613/service-benchmark-final2.json` kept CUDA graph execution active with `24` replays, zero failures, and `cuda:0` tensors. Compared with `reports/cross_modal_fast_idle_20260613/service-benchmark-final2.json`, `routing_index_buffer` fell from `1.2420` to `0.7771 ms/token`, total profiled trainer cost fell from `3.7369` to `2.5741 ms/token`, and profiled trainer throughput rose from `267.6018` to `388.4834 tokens/sec`. Treat this as routing-index maintenance evidence; complete endpoint timing remains environment-sensitive.

- Nonblocking graph input staging on 2026-06-13: the persistent text graph now stages the next input vector into the fixed graph input buffer with `non_blocking=True`, matching the existing parameter staging path without changing graph math or pointer ownership. `reports/graph_input_nonblocking_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, and `cuda:0` tensors. Compared with `reports/hnsw_batched_id_flush_20260613/service-benchmark-final2.json`, `cuda_graph_prepare_input_copy` fell from `0.4154` to `0.3203 ms/token`, `routing_prepare` fell from `0.9592` to `0.7870 ms/token`, total profiled trainer cost fell from `2.5741` to `1.6251 ms/token`, and profiled trainer throughput rose from `388.4834` to `615.3373 tokens/sec`. The configured-source tick wall time was noisy (`67.05` versus `672.14 ms`), so this is trainer graph-prep evidence, not endpoint throughput proof.

- Bounded batched live text ingestion on 2026-06-13: the live source path no longer mutates the learned-chunk codebook while collecting a cognitive tick. For the current empty-codebook, order-weighted RTF windows and deterministic chunk signatures are assembled in batches of at most 32 and emitted as one CUDA tensor batch. CPU scalar/batch parity differed by at most `1.19e-7`. Against `reports/device_owned_routing_cache_20260613/service-benchmark-runtime-truth.json`, the fresh-path run at `reports/batched_live_ingestion_cold_20260613/service-benchmark.json` reduced `collect_source_queue` from `505.6024` to `37.0373 ms`, reduced the complete 24-token tick from `547.4245` to `118.1937 ms`, and raised complete-runtime throughput from `43.8417` to `203.0565 tokens/sec` (`4.632x`). Runtime Truth still recorded `24` CUDA graph replays, zero graph failures, `cuda:0`, and observed CUDA execution. The cache-restored run at `reports/batched_live_ingestion_20260613/service-benchmark-batch32.json` reached `566.3356 tokens/sec`, while its trainer-only profile reached `889.0404 tokens/sec`. Learned-chunk plasticity quality under a future slow cadence remains unproven.

- Post-replay previous-flag cleanup on 2026-06-13: the persistent text graph no longer writes Python `host_parameters[4] = 1.0` after replay because the next preparation step recomputes that flag from `trainer._prev_routing_key` before staging device parameters. Focused CUDA graph parity and host-truth cadence tests passed. `reports/post_replay_previous_flag_cleanup_20260613/service-benchmark-final.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, and host truth sync/skip counts of `4/20`; its specific `cuda_graph_prepare_bookkeeping` bucket fell from `0.0645` to `0.0139 ms/token`, but total profiled trainer cost regressed to `2.0787 ms/token` due to higher input/replay/routing-index stages. This is deletion of stale host bookkeeping, not a throughput promotion.

- Rejected CUDA graph parameter-stage skip on 2026-06-13: a signature-based attempt to skip the pinned host-parameter copy did not delete work because the competitive modulator changed every token. `reports/parameter_stage_skip_20260613/service-benchmark-final.json` reported `parameter_copy_count=25`, `parameter_skip_count=0`, `cuda_graph_prepare_parameter_stage=0.2998 ms/token`, and total profiled trainer cost `2.4091 ms/token` (`415.10 tokens/sec`), worse than the retained nonblocking-input baseline. The experiment was removed; the next credible path is a larger device-owned modulator/control executor, not more Python signature checks.

- Rejected existing-id torch routing-cache in-place update on 2026-06-13: a direct CUDA trainer smoke proved the mechanism could keep the `torch_topk` cache ready and preserve tensor data pointers after an existing-id HNSW update, but complete service profiles rejected promotion. `reports/routing_cache_inplace_update_20260613/service-benchmark-final.json` reduced `cuda_graph_prepare_eligible` to `0.0330 ms/token`, yet raised `routing_index_buffer` to `0.4161 ms/token` and total profiled cost to `1.9300 ms/token`. The optimized position-map version at `reports/routing_cache_positioned_update_20260613/service-benchmark-final.json` reduced `cuda_graph_prepare_eligible` further to `0.0317 ms/token`, but raised `routing_index_buffer` to `0.6745 ms/token` and total profiled cost to `3.1060 ms/token` (`321.96 tokens/sec`). The retained baseline remains `reports/graph_input_nonblocking_20260613/service-benchmark-final.json` at `1.6251 ms/token` (`615.34 tokens/sec`). The runtime experiment was removed; the next routing-index speed path must be a broader device-owned route/index executor, not per-update cache mutation.

- Rejected device-owned recent-spike-row graph cursor on 2026-06-13: removing the pre-replay host fill and letting the persistent CUDA graph advance the spike-row cursor looked aligned with device-owned control state. Focused CUDA graph parity still passed, and Runtime Truth showed `24` graph replays, zero failures, `cuda:0` tensors, and `4/20` host truth sync/skip counts. Complete configured-source profiles rejected promotion: `reports/recent_spike_row_device_owned_20260613/service-benchmark-final.json` measured `366.21 tokens/sec` and `2.7307 ms/token`; the warm repeat measured `342.98 tokens/sec` and `2.9156 ms/token`. The same-current retained host-fill run at `reports/recent_spike_row_host_fill_retained_20260613/service-benchmark-final.json` measured `487.93 tokens/sec` and `2.0495 ms/token` with the same CUDA replay/failure/device evidence. The runtime experiment was removed; keep the small pre-replay fill until a broader persistent executor wins complete ticks.

- Routing-cache clean reuse on 2026-06-13: `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` now expose a read-only dirty bit so `ColumnTransitionRuntime` and the persistent text graph can reuse already-bound routing-cache tensors on clean ticks instead of calling the rebuild-capable cache accessor every time. Dirty caches still rebuild through retrieval before graph replay. Focused routing/cache and CUDA graph tests passed. The first configured-source run at `reports/routing_cache_clean_reuse_20260613/service-benchmark-final.json` proved the counters (`route_cache_clean_fastpath_count=23`, `route_cache_rebuild_check_count=1`, `route_vote_clean_cache_reuse_count=24`) but regressed to `457.89 tokens/sec` and `2.1839 ms/token`, so it is not a universal win. The warm repeat at `reports/routing_cache_clean_reuse_20260613/service-benchmark-warm.json` kept CUDA graph execution active with `24` replays, zero failures, `cuda:0` tensors, and `4/20` host truth sync/skip counts while improving to `626.22 tokens/sec` and `1.5969 ms/token`, slightly above the prior strongest retained `graph_input_nonblocking` profile at `615.34 tokens/sec` and `1.6251 ms/token`. Treat this as a warmed trainer graph-prep improvement, not endpoint throughput proof.

- Rejected deferred HNSW winner-vector gather on 2026-06-13: a contained trainer experiment queued HNSW winner ids without prototype vectors, then gathered deduplicated prototype rows only at the HNSW flush boundary. Focused tests and CUDA graph checks passed, and the targeted `routing_index_buffer` bucket fell from the retained warm `0.5031 ms/token` to `0.2040`, `0.1748`, and `0.1808 ms/token` across `reports/hnsw_deferred_vector_flush_20260613/service-benchmark-final.json`, `service-benchmark-warm.json`, and `service-benchmark-repeat.json`. Complete configured-source throughput did not hold up: those same runs measured `435.55`, `327.18`, and `424.24 tokens/sec`, all below the retained warm routing-cache baseline at `626.22 tokens/sec`. CUDA graph execution stayed active with `24` replays, zero failures, `cuda:0` tensors, and `23` clean route-cache fast-path hits in the attempted runs. The runtime edit was removed; the result is retired-path evidence that the next useful speed slice must collapse a larger route/index/graph-prep cluster, not just move the HNSW vector gather.

- Profiled persistent text-tick A/B evidence on 2026-06-13: `hot_window_benchmark` can now opt into measured-step-only trainer-stage profiling, and `persistent_tick_hot_window_benchmark` reports reversed same-process stage deltas. Focused CPU contract tests passed. On the RTX 3060 with torch `2.11.0+cu128`, `reports/persistent_tick_profile_ab_20260613/profiled-ab.json` compared fused route/vote against the persistent CUDA text-tick executor over four 64-sample arms after 16 warmup steps per arm. Persistent averaged `360.0649 ticks/sec` versus fused `135.5841` (`2.6557x`). CUDA evidence reported active graph execution on persistent arms with `80` replays each, zero failures, tensor device `cuda:0`, and host truth sync/skip counts `11/69`. Stage deltas identify the next bottlenecks and wins: total measured cost fell from `7.1688` to `2.5739 ms/tick`; `routing_prepare` fell by `1.8044 ms/tick`, `column_transition` by `1.5452`, `candidate_winner` by `0.7734`, `post_surprise_replay_tag` by `0.6529`, and winner readback by `0.5513`. The persistent path still pays `cuda_graph_prepare_input_copy=0.5025`, `cuda_graph_prepare_eligible=0.2482`, `cuda_graph_prepare_replay=0.1562`, and `routing_index_buffer=0.4513 ms/tick`, so the next credible speed slice is a larger route/index/graph-prep executor or device-owned modulator/control state, not another isolated micro-optimization.

- Consolidation generation fast-path on 2026-06-13: the Persistent Text Tick Executor now checks the memory-store bucket-consolidation cache generation instead of calling `bucket_consolidation_tensor()` during every graph eligibility check. Focused CUDA tests passed, including fail-closed deactivation after cache invalidation. On the RTX 3060 with the same checkpoint, `reports/consolidation_generation_fastpath_20260613/profiled-ab.json` measured persistent throughput at `408.8154 ticks/sec` versus `360.0649` in the prior profiled persistent A/B, while fused arms in the same run averaged `152.5924 ticks/sec`. Persistent graph arms stayed active with `80` replays each, zero failures, tensor device `cuda:0`, host truth sync/skip counts `11/69`, `consolidation_cache_generation_fastpath_count=160` per arm, and zero generation or memory-warm-state mismatches. Persistent stage cost fell from `2.5739` to `2.2644 ms/tick`; `cuda_graph_prepare_eligible` fell from `0.2482` to `0.2041 ms/tick`, `routing_prepare` from `1.0502` to `0.9511`, and `cuda_graph_prepare_input_copy` from `0.5025` to `0.4476`. This promotes the pointer-generation guard as hot-path cleanup, not a new replay-quality claim.

- Prepared graph candidate reuse on 2026-06-13: `ColumnTransitionRuntime.route_candidates()` now reuses the same-token candidate buffer after `prepare_routing()` has already replayed the persistent text graph, instead of repeating routing-cache and graph-eligibility checks. Focused CUDA graph parity, host-truth cadence, consolidation fail-closed, sensory-bypass, and bootstrap-bypass tests passed. The first profiled A/B at `reports/prepared_graph_reuse_20260613/profiled-ab.json` was mixed: persistent throughput was `371.0043 ticks/sec`, below the prior `408.8154`, even though `route_vote_prepared_graph_reuse_count=80` per persistent arm and duplicate route-vote clean-cache reuse fell to `0`. The immediate repeat at `reports/prepared_graph_reuse_20260613/profiled-ab-repeat.json` was positive: persistent throughput `437.2580 ticks/sec`, fused `161.2748`, persistent total `2.0611 ms/tick`, `routing_prepare=0.8978`, `cuda_graph_prepare_eligible=0.1998`, active graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `route_cache_clean_fastpath_count=76`, `route_cache_rebuild_check_count=4`, and `consolidation_cache_generation_fastpath_count=80`. Treat this as a small duplicate-check deletion with noisy throughput evidence, not a standalone proof that the remaining graph-prep bottleneck is solved.

- Graph-prep substage profiling on 2026-06-13: the persistent graph profiler now splits the former `cuda_graph_prepare_input_copy` bucket into parameter staging, recent-spike-row fill, actual input staging, and the retained aggregate marker. This is measurement-only and runs only when trainer-stage profiling is enabled. `reports/graph_prep_substage_profile_20260613/profiled-ab.json` was intentionally not used as a throughput promotion because the extra profiler marks perturb measured timing; persistent throughput measured `260.9408 ticks/sec` while preserving active CUDA graph replay, `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, and `route_vote_prepared_graph_reuse_count=80`. The useful evidence is the split: persistent mean `cuda_graph_prepare_parameter_stage=0.3219 ms/tick`, `cuda_graph_prepare_recent_row_fill=0.1245`, `cuda_graph_prepare_input_stage=0.0933`, `cuda_graph_prepare_eligible=0.2854`, and `routing_index_buffer=0.5299`. The next speed architecture should therefore move host-owned modulator/control state into a larger device-owned persistent executor before chasing the input copy itself.

- Device-owned previous-routing flag on 2026-06-13: the persistent text graph now keeps the `has_previous_routing_key` flag in graph/device state instead of staging it from host on every replay. Focused CUDA graph parity, host-truth cadence, and consolidation fail-closed tests passed, including `previous_flag_device_owned_count`. `reports/device_owned_previous_flag_20260613/profiled-ab.json` showed active CUDA graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, prepared candidate reuse `80`, and `previous_flag_device_owned_count=80`; the measured parameter-stage bucket fell from the profiling baseline `0.3219` to `0.2462 ms/tick`. Unprofiled repeats were noisy rather than promotional: `reports/device_owned_previous_flag_20260613/ab.json` measured persistent `311.4105 ticks/sec`, and `reports/device_owned_previous_flag_20260613/ab-repeat.json` measured persistent `316.3692` versus fused `103.9618` (`3.043x`) with `144` graph replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `19/125`, and `previous_flag_device_owned_count=144`. Treat this as deletion of one host-staged control bit, not proof that graph prep is solved.

- Device-owned learning-rate counter on 2026-06-13: graph-backed text ticks now compute the competitive learning rate from a graph-owned update-count scalar and increment that scalar after replay. A new mixed-path CUDA test proves that a sensory fallback tick increments Python `update_count`, then the next graph tick resynchronizes device state and records `learning_rate_host_resync_count=1`. Focused CUDA graph tests passed. `reports/device_owned_learning_rate_20260613/profiled-ab.json` kept active graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `learning_rate_device_owned_count=80`, and zero host resyncs, but profiled throughput was only `297.9833 ticks/sec` and `cuda_graph_prepare_parameter_stage=0.2913 ms/tick`, so the profiled run is not a promotion. Unprofiled repeats were better but still not a headline win: `reports/device_owned_learning_rate_20260613/ab.json` measured persistent `336.8304` versus fused `98.7550` ticks/sec, and `reports/device_owned_learning_rate_20260613/ab-repeat.json` measured persistent `361.1783` versus fused `128.4246`, both with `144` graph replays per persistent arm, zero failures, `cuda:0`, `learning_rate_device_owned_count=144`, and zero host resyncs. Treat this as moving one exact control scalar toward device ownership; the remaining speed target is still a larger graph-prep/control executor.

- Revision-cached graph modulator staging on 2026-06-13: `SurpriseMonitor.modulator_revision` now invalidates the graph-staged competitive modulator after CPU-visible surprise changes, while graph prep reuses the already-staged device scalar between revisions. Focused surprise and CUDA graph tests passed, including exact interval-1 parity (`modulator_stage_copy_count=16`, skip `0`) and interval-4 cache behavior (`copy=3`, skip `5`). `reports/modulator_revision_cache_20260613/profiled-ab.json` kept active CUDA graph replay with `80` replays per persistent arm, zero failures, `cuda:0`, host truth sync/skip `11/69`, `modulator_stage_copy_count=11`, and `modulator_stage_skip_count=69`; `cuda_graph_prepare_parameter_stage` fell to `0.0317 ms/tick` from the previous learning-rate-control profile at `0.2913 ms/tick`, and persistent profiled throughput measured `329.7518 ticks/sec`. Unprofiled repeats were mixed but acceptable for a stage promotion: `reports/modulator_revision_cache_20260613/ab.json` measured persistent `296.7573` versus fused `113.0540`, while `reports/modulator_revision_cache_20260613/ab-repeat.json` measured persistent `354.1062` versus fused `102.8524`, both with `144` graph replays per persistent arm, zero failures, `cuda:0`, `modulator_stage_copy_count=19`, and `modulator_stage_skip_count=125`. This is a graph-prep host-copy reduction, not endpoint throughput proof.

- Routing-cache generation fast path on 2026-06-13: `HierarchicalAssemblyIndex` and `ShardedHierarchicalAssemblyIndex` now expose retrieval-owned routing-cache generation stamps, and the Persistent Text Tick Executor skips dirty-bit/pointer validation while the captured generation is unchanged. Focused routing and CUDA graph tests passed. `reports/routing_cache_generation_fastpath_20260613/profiled-ab.json` measured persistent `482.2661` versus fused `165.9433 ticks/sec`, `speedup=2.9062`, `cuda_graph_prepare_eligible=0.2017 ms/tick`, `routing_prepare=0.7375 ms/tick`, and total profiled persistent cost `1.8935 ms/tick`. Both persistent arms reported `80` graph replays, zero failures, `cuda:0`, `route_cache_generation_fastpath_count=76`, `route_cache_generation_mismatch_count=4`, `route_cache_rebuild_check_count=4`, and `route_cache_clean_fastpath_count=0`. Unprofiled repeats kept graph replay active with zero failures and `136` generation fast-path hits per persistent arm: `ab.json` measured persistent `360.1320` versus fused `98.8577`, while `ab-repeat.json` measured persistent `282.9019` versus fused `98.7611`. The second repeat was noisy (`persistent_b=214.0754`, p95 `13.4248 ms`), so this is a graph-eligibility cleanup with CUDA evidence, not a final endpoint-throughput claim.

- Device-owned routing-cache coherence on 2026-06-13: the persistent Triton transition now writes the normalized next winner prototype into the captured exact torch routing cache and the trainer skips the duplicate HNSW winner/vector enqueue on eligible graph ticks. CPU and CUDA contracts passed, including exact cache/prototype parity, exact 16-tick graph/fused sequential state, empty graph-tick HNSW buffers, and host-mirror synchronization without device-cache invalidation. The fresh pre-change profile measured persistent `500.3434 ticks/sec` and `routing_index_buffer=0.3263 ms/tick`. Two post-change profiles at `reports/device_owned_routing_cache_20260613/profiled-ab.json` and `profiled-ab-repeat.json` measured persistent `437.6423` and `484.5770 ticks/sec`; the targeted stage fell to `0.0111` and `0.0282 ms/tick`. Throughput remains variable, so the claim is deletion of the duplicate stage rather than a broad mean-throughput gain. The production-backed grounded gate passed with exact winners, bit-exact cross-modal tensors, zero measured Triton compilation events, and `36.1596` versus `25.8103 ticks/sec` (`1.4010x`). Final configured-source evidence at `reports/device_owned_routing_cache_20260613/service-benchmark-runtime-truth.json` processed 576 graph ticks with 576 device cache updates, 576 skipped buffer writes, zero failures, `cuda:0`, trainer-stage `516.3006 ticks/sec`, and `routing_index_buffer=0.0135 ms/tick`; Runtime Truth exposed `cpu_mirror_stale=true` and zero host-mirror syncs because no retained mutation boundary occurred. Full service throughput was `43.8417 tokens/sec`, leaving source/endpoint orchestration and graph preparation as the next bottlenecks.

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
to `16` through `--native-burst-tokens 16`.

The clean report at
`reports/native_burst_sequence_20260615/native16-131072-i32.json` processed
all `131072` tokens on the RTX 3060 with `velocity_environment.v1` reporting
`not_observed` contention. Runtime Truth exposed
`persistent_executor_burst_tokens=16`,
`persistent_executor_default_burst_tokens=8`, allowed capacities `[8, 16, 32]`,
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

The remaining configured capacity, `--native-burst-tokens 32`, is not a valid
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

### Conditional-WHILE Sequence Executor Prototype, 2026-06-15

The first lower-level CUDA sequence executor prototype now exists behind
`cuda_graph_sequence_executor=conditional_while`,
`MARULHO_CUDA_GRAPH_SEQUENCE_EXECUTOR=conditional_while`, and
`continuous_runtime_stress_benchmark --sequence-executor conditional_while`.
It is not another repeated-child capacity wrapper: the native extension builds a
CUDA Graph conditional `WHILE` parent around the retained one-tick child graph
and uses a tiny device counter kernel to decide whether the loop body runs
again. Failed construction falls back before mutation to retained repeated-child
replay; launch failures remain fail-closed.

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
default, while native8 repeated-child replay remains fallback and opt-out.

### Route-Vote Deep-Sleep Filter In Fused CUDA Route, 2026-06-15

The scheduler slice moved deep-sleep filtering into the fused CUDA route-vote
owner instead of filtering candidates after graph/fused route-vote had already
selected a winner. `core.fused_route_vote_cuda` now reads the existing
route-score rows plus `steps_since_win`, masks deep-sleep rows before route
top-k vote, writes an eight-field `route_vote_deep_sleep_filter.v1` device state
packet, and lets training build the `ColumnWakePlan`. There is no extra
all-column sleep scan; fallback remains explicit when the route rows do not
contain enough awake candidates.

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

This closes the evidence gap without promoting a fake scheduler claim:
candidate wake is bounded, but total-column scaling is still blocked by
route-score rows and dense column-state transition. The next implementation
target is a fused or lazy transition contract that replaces dense
`steps_since_win`, spike-window, assembly, threshold/win-rate, and predictive
state touches with candidate-owned or lazily materialized state while preserving
checkpoint correctness.

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
