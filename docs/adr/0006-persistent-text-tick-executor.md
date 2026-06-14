# ADR 0006: Persistent Text Tick Executor

## Status

Accepted

## Context

The promoted `cuda_graph_text` path still used two graph replays separated by
a reconstruction scalar readback and Python neuromodulator update. Profiling
at 1024 columns attributed material time to graph replay, scalar readback,
surprise updates, and routing preparation. The split prevented CUDA from seeing
the fixed-shape text transition as one workflow.

The runtime must preserve sequential cognition, checkpoint restoration,
pre-mutation fallback, sensory grounding, and operator-visible truth. Archival
memory, source handling, replay, checkpointing, and cross-modal sensory work
must remain outside this fixed-shape graph.

## Decision

For checkpoint-opted-in eligible text ticks, `ColumnTransitionRuntime` owns one
persistent CUDA Graph replay spanning input normalization and projection,
reconstruction error, reconstruction-driven neuromodulator update on
persistent CUDA state, sparse route/vote, and the in-place competitive and
predictive transition. Reconstruction error is derived from the fused exact
route-score maximum inside the captured route/vote phase, removing the earlier
separate dense prototype scan for this graph-owned path.

The executor also owns a bounded fixed-address CUDA input ring. Brain Runtime
offers already encoded tensors in sequential execution quanta; training stages
each quantum with one or two contiguous device operations, verifies tensor
pointer order as tokens are consumed, and lets the captured graph advance the
input slot and recent-spike-row cursor. A mismatch or sensory boundary discards
unconsumed staged entries and falls back before mutation. Checkpoint restore
seeds the device cursor from the restored competitive state.

The same transition owns exact device routing-cache coherence for eligible
graph ticks. It writes the normalized next winner prototype into the captured
retrieval cache through a prevalidated column-id-to-cache-row tensor. The
trainer skips the duplicate per-token HNSW enqueue and treats the CPU index
store as a slow mirror. Before a retained slow-path index mutation, training
synchronizes that host mirror from live prototypes without invalidating the
captured device cache.

After graph eligibility validates the captured consolidation tensor generation,
post-transition bookkeeping does not reacquire that tensor from the memory
store. It also reuses one persistent empty revival-id tensor instead of
allocating a new CUDA tensor on every token. Retained and sensory paths keep
their normal lookup behavior, and explicit structural maintenance may still
replace `last_revived_indices` with real revival evidence.

The executor copies one bounded nine-scalar result packet to the host on the
configured truth cadence
to mirror reconstruction, neuromodulator, winner, effective-plasticity, and
post-transition competitive-surprise truth. The CPU surprise history records
that last scalar from the existing readback rather than launching a separate
CUDA norm and scalar synchronization. Sensory ticks and graph-ineligible states
use the retained path. Pointer changes and graph failures fail before further
mutation.

Training may use a Boundary-Aware Text Burst for exactly eight eligible
text-only ticks. The burst replays the existing one-tick graph eight times in
causal order, while collapsing repeated Python bookkeeping. It cannot cross
sleep/replay, slow-memory admission, cross-modal wake, routing-index mutation,
host-truth, or routing-mode boundaries. Trainer telemetry is observation-only.
Drift refresh and drift-floor closure are CPU archival maintenance performed
after a bounded event drain without replaying the quantum token-by-token.
Strong capture is data-dependent inside the burst, so a separate one-tick
burst graph snapshots the result packet into a bounded device ring every tick
and copies assembly/routing evidence only for threshold crossings. Training
drains the ring once at the host-truth or maintenance boundary and stores all
archival payloads on CPU. Brain Runtime may request the burst, but training
owns eligibility, event admission, maintenance, and neural/bookkeeping
semantics.

Brain Runtime submits one complete prepared text tick through the
Training-Owned Text Sequence API. Training retains the ordered eight-token
quantum boundary, checks cancellation between quanta, chooses burst versus
retained per-token execution, and drains bounded device evidence before
returning. Service supplies source metadata and requested metric positions,
then performs concept observation and Runtime Truth projection outside neural
execution. Runtime Sources separately treats a fully prepared cache generation
as immutable under consumption, so ordinary ticks do not rebuild or hash the
remaining source queue.

Device float neuromodulation and Triton reductions may differ from Python
double scalar arithmetic by floating-point noise. Promotion requires exact
winners, bounded reconstruction tolerance, bounded sequential tensor tolerance,
cognitive-quality evidence, and grounded fallback gates.

## Consequences

- The mid-tick reconstruction synchronization and second replay are removed
  from eligible text ticks.
- Runtime Truth reports persistent replay, host truth synchronization,
  neuromodulator and competitive-surprise update, capture, failure, and
  fallback evidence.
- Runtime Truth reports the reconstruction source as `fused_route_score_max`,
  whether fused reconstruction is active, and the update count; these are
  scalar ownership fields, not additional per-token readbacks.
- Runtime Truth also reports device-owned routing-cache updates, skipped
  per-token index buffering, host-mirror synchronization, and mirror freshness.
- Runtime Truth reports quantum-input stage/reuse/fallback/mismatch/discard
  counters and device-owned recent-spike-row updates.
- Runtime Truth reports graph consolidation-lookup skips and persistent empty
  revival-tensor reuses, proving the allocation-free bookkeeping path executed.
- Runtime Truth reports device strong-event ring ownership, strong-event count,
  and a bounded burst fallback-reason histogram without adding per-token host
  synchronization.
- Current evidence shows a fresh-process `1.506x` complete hot-window gain and
  a text quality-gate `1.202x` gain with exact winners.
- Fresh-process graph memory increased by about `8.13 MB` allocated and `24 MB`
  reserved on the tested RTX 3060.
- End-to-end service velocity remains dominated by stages outside this graph.
- A reversed 256-tick same-process A/B measured `86.11` versus `70.21
  ticks/sec` (`1.226x`) after folding competitive surprise into the result
  packet. Sixteen-tick parity preserved winners, surprise history, and
  precision within `1e-7`.
- Two reversed 256-tick continuous CUDA A/B runs measured quantum-ring means
  of `1026.38` and `877.53 ticks/sec` versus per-token-copy means of `758.57`
  and `746.19`, for `1.353x` and `1.176x` gains. Both runs reused every staged
  token with zero fallback copies, mismatches, or graph failures.
- A 1024-sample current-over-clean-HEAD comparison with `PYTHONPATH` pinned to
  the baseline worktree measured fused reconstruction plus quantum staging at
  `796.22 tokens/sec` versus `630.17 tokens/sec` (`1.264x`) for the quantum
  arms, with `1088` graph replays, `1088` fused reconstruction updates, zero
  graph failures, and zero staged-input mismatches.
- Removing the duplicate post-replay consolidation lookup and empty CUDA
  allocation reduced profiled transition bookkeeping from `0.054251` to
  `0.014375 ms/tick`. Two clean 32768-token continuous-runtime runs reached
  `2169.815` and `2191.057 tokens/sec`, versus the retained `1779.859`
  reference, with zero graph failures and exact skip/reuse counts for every
  token.
- A larger graph containing eight copied tick bodies was rejected after the
  32768-tick device-only run measured `0.951x`; the larger graph added device
  scheduling and evidence-copy work to an already efficient one-tick replay.
- The accepted host-burst path kept all 32768 CUDA transitions and measured
  `2387.898` and `2607.316 tokens/sec` in repeated complete-runtime runs.
  Runtime Truth counted `18032` and `18016` burst-owned tokens with zero burst
  replay failures.
- After preserving data-dependent strong events, final repeated 32768-token
  runs measured `2648.747`, `2533.719`, and `2599.013 tokens/sec`. The final
  run executed all `32768` transitions on `cuda:0`, used `9760` burst tokens,
  reported zero graph/burst failures, and exposed fallback counts:
  host truth `1033`, exploration `581`, drift refresh `441`, telemetry `369`,
  and drift floor `1`. Graph capture startup was `480.524 ms`.
- The Device-Owned Cognitive Boundary Controller later removed exploration,
  telemetry, drift-refresh, and drift-floor closure as execution gates.
  Inspection proved the former exploration noise scalar was never consumed by
  routing, plasticity, curiosity, or action, so it and its checkpoint field
  were deleted while device-owned norepinephrine/surprise dynamics remain.
  A clean 32768-token run reached `2745.790 tokens/sec` with `29136`
  burst-owned tokens and no old boundary fallbacks. A second sample at
  `2027.181` was rejected as promotion evidence because measurement began with
  source prewarm still running. The longer clean 131072-token gate reached
  `2126.013 tokens/sec` over `61.652 s`, executed all `131072` transitions on
  the RTX 3060, used `116696` burst-owned tokens, reported zero graph/burst
  failures, and retained only two real `sleep_boundary` fallbacks.
- The Prepared Source Generation and Training-Owned Text Sequence follow-up
  removed the largest remaining host preparation tax. The clean 131072-token
  CUDA run reached `3359.378 tokens/sec` over `39.017 s`, up `58.0%` from the
  prior `2126.013` long baseline. Mean tick latency fell from `57.714` to
  `35.667 ms`, p95 from `87.300` to `48.458 ms`, and preparation fell from
  `0.156315` to `0.008415 ms/token`. Runtime Truth reported `1024` complete
  training-owned sequences, `16384` quanta, `1024` stable cache-generation
  skips, all `131072` transitions on the RTX 3060, `116680` burst-owned
  tokens, and zero graph/burst failures.
- Device-Burst Lightweight Metrics keep ordinary prepared source ticks on the
  burst path instead of requesting full per-token `train_step` metrics for
  source concept samples. Full metrics remain available for explicit evaluator
  evidence positions, but normal service ticks use the final CUDA result packet
  as bounded Runtime Truth. The 131072-token CUDA service run improved to
  `3565.968 tokens/sec`, reduced `train_compute` to `0.241568 ms/token`, and
  raised burst-owned tokens to `126952` with zero graph/burst failures.

## Reversal

Set `predictive_route_vote_mode` to `fused_triton_text` or `tensor`, or let
eligibility fail closed. Set `cuda_graph_quantum_input_staging=false` to retain
the persistent graph while restoring exact per-token input copies. Checkpoints
preserve the retained execution paths.
