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
  growth/prune/merge/deep-sleep, checkpoint path, executor/device names, and negative external
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
`generate()` can forward bounded MARULHO-owned decode controls
(`generation_repetition_penalty` and `generation_no_repeat_ngram_size`) into the
training-owned LM head and returns its `generation_decode` evidence. The service
adapter may pass these values through, but it must not implement decode logic or
use repetition control as a language-quality promotion.

`MarulhoBrain.learn_language_window()` delegates to the training-owned continual
LM update window. It can mutate checkpointed language-model weights, records
old/new loss, replay retention, rollback evidence, and a `language_learn` trace,
but service/status endpoints still do not own or trigger cognition.

`MarulhoBrain.propose_language_structure()` and
`MarulhoBrain.apply_language_structure()` expose the training-owned LM expert
growth/prune/merge/deep-sleep transaction. Proposal is read-only; application requires explicit
operator approval, writes a baseline checkpoint, verifies heldout
non-regression, records rollback evidence, and emits a `language_structure`
trace. These methods are runtime-owned helpers, not service/status mutation.

`MarulhoBrain.evolve_language_checkpoint()` exposes the training-owned controlled
checkpoint-evolution evaluator. It records parent/child checkpoint lineage,
child-only learning/replay/growth evidence, parent rollback verification, and a
`language_checkpoint_evolution` trace, but it does not replace the installed
parent language model. Promotion remains a separate operator-reviewed gate.
`language_checkpoint_promotion_review.py` now writes that separate evaluation
packet from saved reports. It can mark a selected child checkpoint
`ready_for_operator_parent_promotion_review`, but it does not call
`MarulhoBrain.install_language_model_runtime()`, does not write a live brain
checkpoint, and does not mutate runtime state from a status/read surface.
`MarulhoBrain.install_language_checkpoint_from_promotion_review()` is the
brain-owned follow-up mutation point. It requires an explicit operator approval
record, re-hashes the selected child checkpoint, verifies the review lineage and
rollback fields, loads the MARULHO language checkpoint through the training
loader, installs it as `active_language_path=marulho_lm_head`, and records a
`language_checkpoint_install` trace plus a checkpointed installation report.
The brain LM adapter accepts padded model-vocab checkpoints such as the
`524288` house-scale shape only when their generation policy is capped to the
tokenizer rows, preserving the no-full-padded-vocab decode boundary.
Blocked approval or hash evidence leaves the active runtime unchanged. The
method does not run from status/read surfaces, does not use a service-owned
cognition path, and still keeps `promotes_runtime_claim=false`.
`MarulhoBrain.install_language_checkpoint_from_direct_review()` is the fast
experiment sibling for freshly generated local checkpoints when older reviewed
payloads have been cleaned up. It requires explicit operator approval and an
exact checkpoint SHA before loading through the same MARULHO language checkpoint
loader, records `language_checkpoint_direct_install`, and keeps live checkpoint
writes, service-owned cognition, and runtime promotion false.
`language_brain_checkpoint_runtime_evidence.py` now verifies the next restored
brain step: install the reviewed child, save a brain checkpoint, restore it, and
generate from `MarulhoBrain`. The current CUDA report
`reports/language_brain_runtime/evo-child-quality-repair-installed-brain-runtime-fast-surface-8192-20260705.json`
restores the installed `524288` model-vocab parent as `marulho_lm_head` and
reaches the 8192 diagnostic boundary through the comparison
`MarulhoBrain.generate()` loop at `112.177` tokens/sec.
`MarulhoBrain.generate_sustained_language()` is the brain-owned fast surface:
it records a `language_generate_sustained` trace while delegating CUDA graph
execution to the training-owned LM sustained runner. The same restored language
runtime reaches the 524288 house-scale target at `8064.765` tokens/sec through
`torch_cuda_graph_burst_decode_controls` in
`reports/language_brain_runtime/evo-child-quality-repair-installed-brain-public-sustained-524288-20260705.json`.
This keeps the algorithm out of service/status code and remains evidence, not a
broad runtime promotion.
`language_brain_continual_learning_evidence.py` verifies the next installed
parent step through `MarulhoBrain.learn_language_window()`: install the reviewed
parent, save and restore the brain, run a brain-owned continual-learning window
with replay/forgetting metrics, save and restore the learned brain checkpoint,
then optionally run post-learning sustained generation. The current CUDA report
`reports/language_brain_continual_learning/evo-child-quality-repair-installed-parent-continual-update524288-20260705.json`
accepts a `524288` update-token window at `3079.877` update tokens/sec and
`2810.819` total-window tokens/sec, improves new-domain heldout loss by
`4.7118`, improves old-domain loss instead of forgetting (`-4.0211`), improves
replay loss (`-4.0014`), restores the learned brain checkpoint, and then reaches
`524288/524288` post-learning sustained tokens at `8132.276` tokens/sec through
`torch_cuda_graph_burst_decode_controls`. This is installed-parent learning and
speed evidence, still with `promotes_runtime_claim=false`.
The runner can apply a recurrent-gradient horizon override through
`MarulhoBrain.set_language_recurrent_gradient_horizon()` before saving the
pre-learning brain; the report gates that the configured model horizon and live
state-block horizon survive both pre-learning and learned-brain restore.
The direct-reviewed 2026-07-06 CUDA report
`reports/language_brain_continual_learning/direct-reviewed-horizon2-fresh-installed-continual-update524288-20260706.json`
installs a fresh `524288` model-vocab, horizon-2 checkpoint, accepts `524288`
update tokens at `5938.519` update tokens/sec and `4913.377` total-window
tokens/sec, restores the learned brain checkpoint, and reaches `524288/524288`
post-learning sustained tokens at `8341.802` tokens/sec with zero tracked
Triton failures. It is direct installed-brain speed/learning evidence, not a
runtime or quality promotion.
`language_brain_structural_plasticity_evidence.py` verifies installed-brain
structural mutation after learning. It loads the learned brain checkpoint, saves
and restores the pre-structure brain, proposes and applies structure through
`MarulhoBrain`, saves and restores the post-structure brain, then can run
post-mutation sustained generation. The current direct-reviewed CUDA report
`reports/language_brain_structural_plasticity/direct-reviewed-horizon2-fresh-learned-route-bank-structure-524288-20260706.json`
starts from the fresh horizon-2 learned brain checkpoint, applies route-bank
expansion from `4` to `7` bounded candidates on the learned `524288`
model-vocab brain, verifies checkpoint restore and rollback evidence, records
no status-read mutation, restores the post-structure brain checkpoint, and
sustains `524288/524288` post-structure tokens at `8157.211` tokens/sec through
`torch_cuda_graph_burst_decode_controls` with zero tracked Triton failures.
This is installed-brain structural evidence, not a runtime or language-quality
promotion.
`language_brain_generation_evidence.py` verifies installed-brain generation
after structural mutation. It restores the post-structure `MarulhoBrain`
checkpoint, checks that status reads do not mutate runtime state, verifies the
installed tokenizer, and scores grounded prompt continuations through public
`MarulhoBrain.generate()`. The current direct-reviewed CUDA report
`reports/language_brain_generation/direct-reviewed-horizon2-fresh-post-structure-generation-20260706.json`
records `generation_runs_through_marulho_brain=true`,
`status_read_mutation_absent=true`, no external/service-owned cognition, and
`0/4` grounded prompt cases passed. This proves the fresh post-structure path is
owned by the brain surface, but it also proves quality is not available without
the installed repair lane.
`language_brain_generation_repair_evidence.py` verifies the next brain-owned
quality repair step. It restores an installed brain checkpoint, builds
hard-prompt replay batches, calls `MarulhoBrain.learn_language_window()`,
saves/restores a repaired brain checkpoint, and re-scores public
`MarulhoBrain.generate()`. It can run multiple installed-brain repair passes
and candidate sweeps while recording per-candidate generation deltas. The
current fresh CUDA report
`reports/language_brain_generation_repair/direct-reviewed-horizon2-fresh-repair2-sweep-sustained524288-20260706.json`
selects `candidate-01`, reaches `4/4` grounded prompt cases with zero prompt
regressions, records `393216` update tokens at `5943.300` update tokens/sec and
`5365.354` total-window tokens/sec, improves mean prefix match by `4.5` chars,
restores the repaired checkpoint, and sustains `524288/524288` selected
post-repair controlled tokens at `8070.687` tokens/sec with zero tracked Triton
failures. This is
installed-brain learning and repair evidence, not a broad generation-quality or
runtime promotion.

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
