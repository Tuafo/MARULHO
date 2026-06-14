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

The executor copies one bounded nine-scalar result packet to the host per tick
to mirror reconstruction, neuromodulator, winner, effective-plasticity, and
post-transition competitive-surprise truth. The CPU surprise history records
that last scalar from the existing readback rather than launching a separate
CUDA norm and scalar synchronization. Sensory ticks and graph-ineligible states
use the retained path. Pointer changes and graph failures fail before further
mutation.

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

## Reversal

Set `predictive_route_vote_mode` to `fused_triton_text` or `tensor`, or let
eligibility fail closed. Set `cuda_graph_quantum_input_staging=false` to retain
the persistent graph while restoring exact per-token input copies. Checkpoints
preserve the retained execution paths.
