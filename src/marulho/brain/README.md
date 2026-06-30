# Brain

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`MarulhoBrain` is the main runtime spine. The intended path is checkpoint
load/restore, feed/source buffering, tick/learn, local generate/readout, replay,
grow/prune, compact trace, and save.

## Owns

- Source buffering for brain ticks.
- Tick orchestration through `MarulhoTrainer.train_text_sequence`.
- Early local readout through sparse MARULHO-owned transition state.
- `BrainTrace` telemetry for status, generation before/after evidence, replay,
  growth/prune, checkpoint path, executor/device names, and negative external
  cognition flags.
- Brain-local checkpoint metadata continuity.
- Brain-owned background start/stop loop for queued source ticks.

## Must Not Own

- CUDA/Triton/native graph algorithms. Those stay in `training`.
- HTTP route shape or UI layout. Those stay in `service` and `MARULHO_UI`.
- Service-owned lifecycle or scheduler policy. `/brain/start` and
  `/brain/stop` must call `MarulhoBrain.start()` and `MarulhoBrain.stop()`.
- Hidden external LLM, Cortex, or ThoughtLoop cognition.
- Broad Runtime Truth schema expansion. Use compact `BrainTrace` as the default
  spine status and keep deeper gates explicit.

## Validation Snapshot

Current brain validation covers load, feed, tick, local generate/readout,
replay/growth hooks, save/restore continuity, and service adapter use through
focused tests plus the full repository test suite.

The current CUDA sequence-input gate uses the active checkpoint and preserves
`cuda_graph_route_transition_burst` with backend
`cuda_graph_conditional_while`, device `cuda:0`, and zero graph/native/burst
failures. The latest measured long sequence gate reached `6601.19` sequence
tokens/sec versus `6507.41` per-quantum tokens/sec.
