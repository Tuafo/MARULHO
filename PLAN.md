# HECSN / Terminus Living Brain Roadmap

## Summary
Terminus is currently a grounded, auditable cortex-subcortex runtime, not yet a living autonomous brain. The system has a predictive SNN-style substrate, a strict NVIDIA NIM cortex path, runtime truth reporting, replay evidence export, action audit surfaces, and an offline replay training gate. The immediate direction is to turn replay evidence into a controlled learning ladder without breaking safety: evidence export, offline gate, explicit operator approval, dry-run training plan, isolated training artifact, benchmark comparison, then only later controlled autonomy.

## Current Validated State
- Current commit inspected: `ef68bd3 Add offline replay training gate and fixes`.
- Maintained runtime validation rerun:
  - `python -m pytest tests/test_replay_training_gate.py tests/test_service_api.py tests/test_service_benchmark.py -q` -> `50 passed`
  - `python -m pytest tests/test_service_manager.py tests/test_long_test_runner.py -q` -> `134 passed`
- Saved replay gate truth:
  - `reports/replay_training_gate_report.json`
  - status: `passed_pending_operator_training_approval`
  - checks: `7/7 passed`
  - `eligible_for_training=false`
- Saved benchmark truth:
  - `reports/runtime_truth_benchmark_validation.json`
  - 15 endpoint timings
  - `/status` and `/terminus` runtime truth verdict: `partial`
  - recommended action: `configure_terminus_sources`
  - replay bundle gate: `blocked_preview_only`
- Refactor stabilization is considered done. Do not reopen it unless a current test or runtime result proves a regression.

## System DNA
- Grounding first, language second: the SNN/subcortex owns prediction error, replay pressure, novelty, salience, local grounding, and curiosity.
- Cortex as expressive layer: NIM cortex owns language, answers, reasoning, working memory, narrative self, and dream-style hypothesis formation.
- Runtime truth over optimism: every alive/degraded/partial/failed claim must include evidence, safety flags, latency, and recommended action.
- Replay is evidence, not permission: replay bundles must not train adapters, mutate memory, promote facts, execute actions, call tools, post feedback, or start sleep.
- One maintained path: remove old compatibility scaffolding, stale docs, unused tests, and experiments that do not support live runtime, learning evidence, evaluation, or research-only measurement.
- Every PR must leave the project easier to reason about than before.

## Maintained Lanes
- Live runtime: service API, manager facade, cortex runtime, sensory runtime, action runtime, status/runtime truth, persistence.
- Learning evidence: runtime traces, feedback, replay plan/sample, replay dataset preview/bundle, offline replay training gate.
- Evaluation: service benchmark, long-test runner, replay gate reports, grounding probes, ARC-style probes when useful.
- Research-only: developmental, autonomy, acquisition, meaning grounding, and memory consolidation runners. These may inform the roadmap but must not be treated as production proof.

## Active Roadmap

### Phase 1: Canonical Roadmap And Cleanup
Goal: make the repo’s direction unambiguous.
- Keep `PLAN.md` as the only active roadmap.
- Merge the useful current truth from `SYSTEM_IMPLEMENTATION_PLAN.md` and `LIVING_BRAIN_CLEANUP_MAP.md` into `PLAN.md`.
- Remove those two plan-like files after consolidation.
- Keep `GPCSN.md`, `Terminus_Cortex_Paper.md`, and `TERMINUS_Tutorial.md` only as architecture/tutorial/paper docs.
- Remove stale operator guidance for Ollama, Gemma, FakeCortex, removed presets, or old compatibility paths.

Exit criteria:
- One roadmap file remains.
- No active doc says old local-LLM paths are production paths.
- `git diff --check` passes.
- No code behavior changes in this phase.

### Phase 2: Operator Approval Artifact
Goal: allow a human to approve a specific replay bundle and gate report for dry-run planning only.
- Add a read-only approval artifact format bound to:
  - `bundle_hash`
  - `gate_report_hash`
  - operator id
  - creation time
  - expiry
  - approval scope
  - intended target
  - safety acknowledgements
  - rollback note
- Initial allowed scope: `dry_run_training_plan_only`.
- Any missing, expired, mismatched, tampered, or broader approval fails closed.
- Passing approval still does not train.

Public interface:
- `python -m hecsn.evaluation.replay_training_approval --bundle <bundle.json> --gate-report <report.json> --operator-id <id> --scope dry_run_training_plan_only --output <approval.json>`

Tests:
- valid approval writes deterministic artifact
- wrong bundle hash fails
- wrong gate report hash fails
- expired approval fails
- missing operator id fails
- unsafe scope fails
- no memory, adapter, feedback, action, sleep, or external call side effects

### Phase 3: Dry-Run Replay Training Planner
Goal: convert approved evidence into a training proposal without training.
- Consume bundle, replay gate report, and approval artifact.
- Write a deterministic plan with:
  - dataset identity
  - split counts
  - contamination result
  - target adapter name/path
  - proposed train/eval command
  - expected cost/time
  - benchmark suite to run before and after
  - rollback path
  - unresolved risks
- The planner must not create model weights or mutate production state.

Public interface:
- `python -m hecsn.evaluation.replay_training_plan --bundle <bundle.json> --gate-report <report.json> --approval <approval.json> --output <training_plan.json>`

Tests:
- refuses missing approval
- refuses mismatched hashes
- refuses approval scope other than `dry_run_training_plan_only`
- writes stable plan output
- confirms no adapter, memory, feedback, action, sleep, or network side effects

### Phase 4: Isolated Adapter Training Experiment
Goal: run the first learning experiment only as an isolated artifact.
- Require a second explicit approval scope: `isolated_adapter_training`.
- Train only into a non-production artifact directory.
- Record command, git commit, environment, source hashes, hyperparameters, and wall time.
- Never switch production runtime to the trained adapter automatically.
- Compare before/after on replay holdout, service benchmark, long-test health, runtime truth, and safety flags.

Exit criteria:
- Training artifact exists outside production runtime.
- Before/after report shows no safety regression.
- Any claimed improvement has saved evidence.
- Rollback is trivial because production runtime was not changed.

### Phase 5: Promotion Gate
Goal: decide whether an isolated learning artifact is worth using.
- Add an explicit promotion evaluation, not automatic deployment.
- Require:
  - benchmark improvement or clearly documented useful behavior
  - no regression on runtime truth
  - no contamination failure
  - no increased unsafe action/replay behavior
  - operator approval
- Promotion may only change a configured non-default experimental path first.

Tests:
- promotion refuses missing reports
- promotion refuses worse safety verdict
- promotion refuses missing operator approval
- promotion records rollback metadata

### Phase 6: Autonomy Ladder
Goal: grow autonomy through measured levels, never by jumping straight to self-modification.
- Level 0: observe only
- Level 1: propose actions only
- Level 2: execute approved actions
- Level 3: constrained recurring actions with limits
- Level 4: adaptive policy updates after evaluation
- Level 5: bounded self-improvement loop with approval, benchmark, rollback, and audit trail

Required for every autonomy level:
- permission model
- expected outcome
- rollback plan
- delayed consequence tracking
- trace replay
- operator-visible report
- failure mode tests

### Phase 7: Live Long-Run Validation
Goal: validate living behavior only when the maintained gates are strong.
- Run configured-source NIM long tests with saved traces.
- Capture:
  - runtime truth verdict
  - liveness verdict
  - latency and cost
  - cortex availability
  - embedding health
  - replay safety status
  - action audit status
  - memory pressure
  - recommended operator action
- A live run is not proof unless the saved report supports the claim.

## Benchmarks Worth Having
- Required now:
  - service manager/API tests
  - replay training gate tests
  - service benchmark tests
  - long-test runner tests
  - replay bundle safety tests
  - trace export tests
  - action audit tests when action code changes
- Required before learning claims:
  - replay holdout/eval split report
  - contamination/decontamination report
  - before/after service benchmark
  - before/after long-test health
  - no-mutation safety assertions
- Required before autonomy claims:
  - action proposal/execution audit
  - permission denial tests
  - rollback tests
  - delayed consequence tests
  - trace replay tests
- Deferred external pressure tests:
  - GAIA, AgentBench, WebArena-style tasks, ARC-style reasoning probes. These are useful later, not the next engineering path.

## PR Protocol
Every PR must include:
- hypothesis: what should improve and why
- focused tests for touched modules
- saved validation command/result
- performance note: cost, latency, memory, or simpler alternative considered
- docs update when behavior, API, safety policy, or benchmark meaning changes
- cleanup note: what unused code/docs/tests were removed or intentionally kept
- no compatibility layer kept only for old behavior unless it directly supports the roadmap

## Research Anchors
Use research to guide design, not to inflate the docs.
- Predictive coding: prediction, error, update, and attention pressure.
- Free-energy / active inference: action should reduce expected uncertainty, not become random tool use.
- Global workspace / working memory: active context must be selected and bounded.
- Hippocampal replay and continual learning: replay must be gated, evaluated, and protected from contamination.
- Reference-frame / Thousand Brains ideas: grounded concepts need structured local models, not only text embeddings.
- World models: imagination/planning is useful later, after evidence gates are reliable.
- LLM-agent benchmarks: external pressure tests, not the identity of the project.

## Near-Term Next Slice
The next implementation slice is Phase 1 plus Phase 2:
- consolidate roadmap docs into `PLAN.md`
- remove duplicated plan files
- add the operator approval artifact for a specific replay bundle and gate report
- keep the approval artifact read-only and limited to dry-run planning
- test the approval artifact thoroughly
- validate that replay remains non-training and non-mutating

## Non-Negotiable Safety Boundary
Until all gates exist and pass, Terminus must remain:
- no autonomous adapter training
- no memory promotion from replay
- no fact promotion from dreamed/synthetic content
- no action execution from replay artifacts
- no sleep/action/tool side effects from learning gates
- no production model switch without benchmark evidence and explicit operator approval
