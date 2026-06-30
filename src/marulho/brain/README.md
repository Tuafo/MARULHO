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
  growth/prune, checkpoint path, executor/device names, and retired-surface
  booleans.
- Brain-local checkpoint metadata continuity.
- Brain-owned background start/stop loop for queued source ticks.

## Must Not Own

- CUDA/Triton/native graph algorithms. Those stay in `training`.
- HTTP route shape or UI layout. Those stay in `service` and `MARULHO_UI`.
- Legacy Terminus runtime-control lifecycle. `/brain/start` and `/brain/stop`
  must call `MarulhoBrain.start()` and `MarulhoBrain.stop()`.
- Hidden external LLM, Cortex, or ThoughtLoop cognition.
- Broad Runtime Truth schema expansion. Use compact `BrainTrace` as the default
  spine status and keep deeper gates explicit.

## Current Evidence

The pre-refactor archive branch is
`origin/archive/pre-refactor-2026-06-30` at
`1b2861c19d5f48cb9895f6fe600e21f36b9cc714`.

The first brain-spine CUDA stress check used the existing active-pressure
checkpoint and reached `6658.764` sequence tokens/sec with backend
`cuda_graph_conditional_while`, speedup `1.304x`, and zero graph, burst, or text
fallback failures.
