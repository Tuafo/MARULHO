# Brain

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`MarulhoBrain` is the main runtime owner. The intended path is checkpoint
load/restore, feed/source buffering, tick/learn, local generate/readout, replay,
grow/prune, compact trace, and save.

## Owns

- Source buffering for brain ticks.
- Tick orchestration through `MarulhoTrainer.train_text_sequence`.
- The active language path selection. When checkpointed language components are
  installed, `generate()` uses the training-owned `marulho_lm_head` adapter;
  otherwise it falls back to the sparse MARULHO-owned transition readout.
- `BrainTrace` telemetry for status, generation before/after evidence, replay,
  growth/prune/merge, checkpoint path, executor/device names, and negative external
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
  runtime status and keep deeper gates explicit.

## Validation Snapshot

Current brain validation covers load, feed, tick, local generate/readout,
replay/growth hooks, save/restore continuity, and service adapter use through
focused tests plus the full repository test suite.

The Iteration 2 LM-head path is now brain-selectable through
`BrainLanguageModelRuntime`. It is checkpointed inside `brain_state` and reports
`active_language_path=marulho_lm_head` plus `external_llm_used=false`. It is not
yet promoted as the live long-run language capability until online learning,
rollback, throughput, and sustained Runtime Evidence gates pass.

`MarulhoBrain.learn_language_window()` delegates to the training-owned continual
LM update window. It can mutate checkpointed language-model weights, records
old/new loss, replay retention, rollback evidence, and a `language_learn` trace,
but service/status endpoints still do not own or trigger cognition.

`MarulhoBrain.propose_language_structure()` and
`MarulhoBrain.apply_language_structure()` expose the training-owned LM expert
growth/prune/merge transaction. Proposal is read-only; application requires explicit
operator approval, writes a baseline checkpoint, verifies heldout
non-regression, records rollback evidence, and emits a `language_structure`
trace. These methods are runtime-owned helpers, not service/status mutation.

`MarulhoBrain.evolve_language_checkpoint()` exposes the training-owned controlled
checkpoint-evolution evaluator. It records parent/child checkpoint lineage,
child-only learning/replay/growth evidence, parent rollback verification, and a
`language_checkpoint_evolution` trace, but it does not replace the installed
parent language model. Promotion remains a separate operator-reviewed gate.

The current CUDA sequence-input gate uses the active checkpoint and preserves
`cuda_graph_route_transition_burst` with backend
`cuda_graph_conditional_while`, device `cuda:0`, and zero graph/native/burst
failures. The latest measured long sequence gate reached `6601.19` sequence
tokens/sec versus `6507.41` per-quantum tokens/sec.
On PyTorch builds that do not expose `torch.cuda.CUDAGraph.raw_cuda_graph()`,
current validation may instead use `torch_sequence_graph_*` q16 evidence; that
must remain distinct from native conditional-WHILE evidence.

Hot-path feed/tick invariants:

- `feed(..., learn=False)` queues source patterns without mutating learned
  chunk state.
- `tick()` records readout transitions from trainer winner evidence and must
  not recompute offline winners per token after the CUDA trainer step.
