# Living Brain Cleanup Map

This file separates the code that moves Terminus toward the living-brain goal from older or supporting experiments. It is intentionally conservative: code is labeled before it is removed.

## Current State

Terminus has a safe, auditable loop:

- runtime traces are recorded
- operator feedback can verify or contradict outputs
- replay planning can prioritize useful past episodes
- replay sampling is operator-gated
- replay dataset preview and bundle exports are preview-only artifacts
- replay exports do not train adapters, mutate memory, promote facts, execute actions, call tools, or start sleep

The main cleanup problem is that live runtime code, learning evidence code, benchmarks, and older training experiments are still too close together.

## Proposed State

Keep Terminus focused on the architecture in `GPCSN.md`: a grounded SNN-style subcortex that gates and pressures an LLM cortex.

The project should be read in four lanes:

| Lane | Purpose | Keep Moving Forward |
| --- | --- | --- |
| Live runtime | The system an operator uses | service API, manager facade, cortex loop, sensory runtime, action audit loop |
| Learning evidence | Safe artifacts for future learning | runtime traces, feedback, replay plan/sample, replay dataset preview/bundle |
| Evaluation | Ways to measure progress | service benchmark, acceptance probes, ARC, grounding probes |
| Research-only | Useful experiments, not production paths | developmental, autonomy, memory consolidation, and older training runners |

## Keep As Live Support

- `src/hecsn/service/api.py`
- `src/hecsn/service/manager.py`
- `src/hecsn/service/runtime_evidence.py`
- `src/hecsn/service/runtime_feedback.py`
- `src/hecsn/service/action_assist.py`
- `src/hecsn/service/action_runtime.py`
- `src/hecsn/service/brain_runtime.py`
- `src/hecsn/service/delayed_consequence.py`
- `src/hecsn/service/persistence.py`
- `src/hecsn/service/cortex_runtime.py`
- `src/hecsn/service/reporting.py`
- `src/hecsn/service/replay_runtime.py`
- `src/hecsn/service/interaction_runtime.py`
- `src/hecsn/service/living_status.py`
- `src/hecsn/service/runtime_config.py`
- `src/hecsn/service/runtime_control.py`
- `src/hecsn/service/runtime_prewarm.py`
- `src/hecsn/service/runtime_sources.py`
- `src/hecsn/service/sensory_runtime.py`
- `src/hecsn/service/source_focus.py`
- `src/hecsn/service/status_runtime.py`
- `src/hecsn/service/sensory_preview.py`
- `src/hecsn/service/living_loop.py`
- `src/hecsn/service/action_loop.py`
- `src/hecsn/service/trace_export_runner.py`
- `src/hecsn/service/replay_dataset_runner.py`
- `src/hecsn/service/replay_dataset_bundle_runner.py`
- `src/hecsn/evaluation/service_benchmark.py`
- `src/hecsn/training/query_runner.py`

## Keep As Evaluation Or Research

- `src/hecsn/evaluation/arc_agi.py`
- `src/hecsn/training/developmental_runner.py`
- `src/hecsn/training/autonomy_runner.py`
- `src/hecsn/training/autonomy_acquisition_runner.py`
- `src/hecsn/training/memory_consolidation_runner.py`
- `src/hecsn/training/meaning_grounding_runner.py`
- `src/hecsn/training/long_test_runner.py`

These are useful for measurement and research, but they should not be treated as proof that Terminus is a living brain.

## Cleanup Candidates

Do not delete these yet. First check imports, tests, and operator value.

- generic training paths that do not feed Terminus runtime, replay, grounding, or evaluation
- duplicate benchmark helpers
- older scripts that bypass the safe replay and feedback boundary
- adapter-training experiments until dataset bundles have stronger versioning, deduplication, decontamination, and approval checks

## Safety Boundary

Replay datasets remain export artifacts only. A future adapter experiment must require a separate operator approval gate and must not reuse this preview/export path as implicit training permission.
