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
predictive transition.

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

Device float neuromodulation may differ from Python double scalar arithmetic by
floating-point noise. Promotion requires exact winners and reconstruction plus
a bounded sequential tensor tolerance, cognitive-quality evidence, and
grounded fallback gates.

## Consequences

- The mid-tick reconstruction synchronization and second replay are removed
  from eligible text ticks.
- Runtime Truth reports persistent replay, host truth synchronization,
  neuromodulator and competitive-surprise update, capture, failure, and
  fallback evidence.
- Runtime Truth also reports device-owned routing-cache updates, skipped
  per-token index buffering, host-mirror synchronization, and mirror freshness.
- Current evidence shows a fresh-process `1.506x` complete hot-window gain and
  a text quality-gate `1.202x` gain with exact winners.
- Fresh-process graph memory increased by about `8.13 MB` allocated and `24 MB`
  reserved on the tested RTX 3060.
- End-to-end service velocity remains dominated by stages outside this graph.
- A reversed 256-tick same-process A/B measured `86.11` versus `70.21
  ticks/sec` (`1.226x`) after folding competitive surprise into the result
  packet. Sixteen-tick parity preserved winners, surprise history, and
  precision within `1e-7`.

## Reversal

Set `predictive_route_vote_mode` to `fused_triton_text` or `tensor`, or let
eligibility fail closed. Checkpoints preserve the retained execution paths.
