# Service

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`service` is a thin HTTP/UI adapter over `MarulhoBrain`.

## Owns

- FastAPI route registration.
- UI-facing status stream, start/stop, feed, tick, generate, replay,
  grow/prune, checkpoints, traces, and saved report inventory through
  `/brain/*`.
- Runtime composition for the active `MarulhoBrain` checkpoint.

## Must Not Own

- Neural algorithms, CUDA execution policy, or readout learning.
- Hidden replay, consolidation, or delayed-consequence work inside status reads.
- Compatibility aliases that make HTTP routing look like a second runtime
  owner.

## Active Contract

The active API surface is `/health`, `/`, and `/brain/*`. `create_app()` builds
`MarulhoBrainServiceManager`, which loads a checkpoint-backed `MarulhoBrain`
and adapts its feed, tick, generate, replay, grow/prune, trace, lifecycle, and
checkpoint methods.

`/brain/start` and `/brain/stop` call the brain-owned lifecycle loop. Status
and status-stream responses should come from compact brain state and
`BrainTrace`, not from broad schema expansion or hidden work.

`api_schemas.py` contains the active request/response models for the maintained
contract. New route models should stay small and should be backed by an owner
module in `brain`, `training`, `core`, `data`, `semantics`, or another
machinery package.

`/brain/evidence/reports` is a read-only projection over saved JSON artifacts in
`reports/`. It summarizes report metadata and promotion gates for the UI but
must not run benchmarks, mutate `MarulhoBrain`, or turn report presence into a
capability claim.

`/brain/evidence/language` is the current saved-language-evidence projection
for UI/status display. It condenses the latest benchmark suite, installed-brain
generation, installed-brain repair sweep, installed-brain continual-learning
update throughput, forgetting/replay metrics, installed-brain structural
plasticity, checkpoint lineage, 524288-token house-scale sustained runs, active
compute, GPU-kernel, backend rejection, and selected checkpoint references from
JSON artifacts only. It is read-only, service-owned-cognition stays false, and
it does not promote runtime or generation-quality claims without the source
report gates doing so.

Some machinery modules for replay, SNN readout ledger, action execution,
runtime sources, and plasticity live under `service` because they currently
serve the HTTP/UI boundary. They must name their real owner clearly and should
not turn service into the algorithm owner.

## Runtime Rules

- Status and Runtime Truth reads must remain read-only.
- Checkpoint writes/restores and trace persistence are explicit slow paths.
- Runtime scope/status caches may keep the control room responsive, but they
  are not model memory, mutation state, scheduler state, or CUDA speed evidence.
- Service may offer encoded quanta and lifecycle controls; trainer owns burst
  algorithms and mutation semantics.
