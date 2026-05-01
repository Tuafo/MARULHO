# HECSN / Terminus Living Brain Roadmap 2

## Summary
Terminus has completed the first safety and evaluation ladder: canonical roadmap cleanup, replay training approval, dry-run planning, isolated adapter experiment artifacts, promotion gates, autonomy ladder evaluation, and live long-run validation. The system is now a grounded, auditable cortex-subcortex runtime with strict safety gates around learning and autonomy.

The system is still not a proven living autonomous brain. The first 30-minute configured-source NIM run produced valid saved evidence, but behavioral liveness was degraded: runtime processing advanced, NIM embedding worked, and safety held, yet the cortex produced zero thoughts and memory saturated. The next direction is to repair live liveness before expanding learning, autonomy, or run duration.

## Current Validated State
- Latest completed roadmap commits:
  - `7c54390 remove old plans`
  - `e1a26e0 Add replay training approval, plan, experiment`
  - `2bf490e Add autonomy ladder and live long-run validators`
- Worktree state at roadmap handoff: clean.
- Implemented gates:
  - replay training approval artifacts
  - dry-run replay training plans
  - isolated adapter experiment artifacts
  - experimental promotion gate
  - autonomy ladder evaluator for levels 0-5
  - live long-run validation report checker
- Saved Phase 7 evidence:
  - `reports/phase7_live_validation/long_test_20260501_001650.json`
  - `reports/phase7_live_validation/service_benchmark_20260501_001650.json`
  - `reports/phase7_live_validation/live_long_run_validation_20260501_001650.json`
- 30-minute configured-source NIM long run:
  - `health_verdict=degraded`
  - `acceptance_verdict=partial`
  - `samples_collected=30`
  - `total_thoughts=0`
  - `final_token_count=11852`
  - `max_background_tokens_processed=6686`
  - `final_tick_count=107`
  - `cortex_available=true`
  - `terminus_configured=true`
  - `terminus_running=true`
  - `final_memory_fill=1.0`
  - final runtime truth: `alive`
  - recommended action: `continue_monitoring`
- NIM embedder health:
  - `available=true`
  - `degraded=false`
  - `nim_calls=119`
  - `fallback_calls=0`
  - `error_calls=0`
  - `rate_limit_hits=0`
- Service benchmark:
  - `success=true`
  - `total_latency_ms=661.667`
  - `/status` and `/terminus` runtime truth: `partial`
  - recommended action: `configure_terminus_sources`
  - replay bundle gate: `blocked_preview_only`
- Live long-run validator:
  - `status=evidence_supported`
  - `passed=true`
  - runtime truth verdict: `alive`
  - liveness verdict: `degraded`
- Safety status from saved reports:
  - no autonomous adapter training
  - no replay memory mutation
  - no feedback posting
  - no digital action execution
  - no external replay calls
  - no sleep side effects
  - no production model switch

## System DNA
- Grounding first, language second: subcortex/runtime evidence must drive memory pressure, novelty, salience, replay, and action pressure before language claims.
- Runtime truth over optimism: an `alive` runtime truth verdict is not enough if long-run behavioral health is degraded.
- Liveness is behavioral: a living run must show thoughts, bounded memory, source-grounded context, and stable runtime progress.
- Replay remains evidence, not permission: replay artifacts do not train, mutate memory, promote facts, execute actions, call tools, post feedback, or start sleep.
- Autonomy grows by measured levels: observe, propose, execute approved actions, recurring constrained actions, evaluated policy updates, then bounded self-improvement.
- Every claim needs saved evidence: no “living brain,” “learning,” “improvement,” or “autonomy” claim is accepted without reports, safety flags, rollback, and operator-visible state.

## Maintained Lanes
- Live runtime: service API, manager facade, cortex runtime, sensory runtime, action runtime, status/runtime truth, persistence.
- Liveness evidence: long-test runner, acceptance harness, runtime truth, thought traces, source configuration, memory pressure reports.
- Learning evidence: replay dataset preview/bundle, replay training gate, approvals, dry-run plans, isolated adapter artifacts, promotion evaluation.
- Autonomy evidence: autonomy ladder reports, action audit, delayed consequence tracking, trace replay, permission denial tests.
- Evaluation: service benchmark, long-test runner, live long-run validator, replay gate reports, trace export, grounding probes.
- Research-only: developmental, acquisition, meaning grounding, memory consolidation, and autonomy runners may inform direction but are not production proof.

## Active Roadmap

### Phase 8: Live Liveness Repair
Goal: turn the degraded 30-minute run into an alive 30-minute run.

- Diagnose why runtime progressed but cortex produced `0` thoughts.
- Diagnose why acceptance harness was partial.
- Determine whether the blocker is thought-loop wake policy, source ingestion, cortex scheduling, prompt/cortex response path, long-test measurement, or memory saturation.
- Add explicit saved evidence for thought generation attempts, rejected thoughts, wake triggers, and cortex response outcomes.
- Keep all replay, training, action, and sleep safety boundaries unchanged.

Exit criteria:
- 30-minute configured-source NIM run reaches `health_verdict=alive`.
- `total_thoughts > 0`.
- acceptance harness passes.
- runtime truth and long-test health agree, or disagreement is explicitly explained in the report.
- no replay/training/action safety violation.

### Phase 9: Memory Pressure And Working Set Control
Goal: prevent live runs from filling memory without useful cognition.

- Treat `final_memory_fill=1.0` as a blocker for longer runs.
- Add a bounded working-set policy for long runs.
- Record memory pressure transitions, eviction/consolidation decisions, and recovery behavior.
- Ensure consolidation or eviction does not promote replay facts, dreamed content, or synthetic hypotheses.
- Add memory-pressure regression tests and saved operator report fields.

Exit criteria:
- 30-minute run avoids unrecovered high memory pressure or records a justified recovery action.
- memory pressure appears in runtime truth and long-test reports.
- no replay memory mutation or fact promotion occurs.
- longer-run readiness requires memory pressure below the configured high threshold.

### Phase 10: Thought Trace And Global Workspace Evidence
Goal: make cognition inspectable, selected, and grounded.

- Add thought lifecycle evidence:
  - wake trigger
  - selected working context
  - source evidence
  - cortex request status
  - generated thought or rejection reason
  - latency
  - safety label
- Add a bounded global-workspace-style active context report.
- Separate grounded thoughts from hypotheses.
- Prevent dreamed, synthetic, or unsupported content from entering verified memory.

Exit criteria:
- each generated thought has source evidence or an explicit hypothesis-only label.
- rejected thoughts are counted with reasons.
- long-test report includes thought count, thought latency, topic diversity, selected context, and rejected-thought summary.
- no unsupported thought is promoted as fact.

### Phase 11: Source Configuration And Acceptance Harness Hardening
Goal: make configured-source NIM runs reproducible and acceptance results actionable.

- Record exact source-bank configuration used by long runs.
- Add an acceptance failure breakdown to the long-test report.
- Ensure benchmark and long-test source configuration semantics match or explain why they differ.
- Reduce ambiguity around `configure_terminus_sources` when long-test runtime is configured but benchmark runtime truth is partial.

Exit criteria:
- acceptance harness failures name the failed check and required operator action.
- service benchmark and long-test reports include source configuration evidence.
- `/status` and `/terminus` recommended actions are actionable and not contradictory.
- a fresh 30-minute run saves enough evidence to reproduce the configuration.

### Phase 12: Replay-To-Adaptation Experiment 1
Goal: run the first learning experiment only after live liveness and memory pressure are stable.

- Use existing approval, dry-run plan, isolated adapter experiment, benchmark comparison, and promotion gate.
- Train only into a non-production isolated artifact path.
- Compare before/after on replay holdout, service benchmark, long-test health, runtime truth, and safety flags.
- Never switch production runtime automatically.

Exit criteria:
- isolated artifact exists outside production runtime.
- before/after reports show no safety regression.
- any claimed improvement has saved benchmark and holdout evidence.
- rollback is trivial because production runtime was not changed.

### Phase 13: Approved Action Loop Level 2
Goal: move from observe/propose toward approved action execution only after liveness is stable.

- Start with workspace-only approved actions.
- Require autonomy ladder level 2 approval.
- Record permission model, expected outcome, rollback plan, action audit, delayed consequence tracking, trace replay, and operator-visible report.
- Keep web/API actions behind separate approval until workspace actions are stable.

Exit criteria:
- approved workspace action execution passes audit tests.
- denied actions are recorded and non-mutating.
- delayed consequences are tracked across later queries and runs.
- action replay can reproduce the audit trail without re-executing unsafe effects.

### Phase 14: Multi-Hour Live Validation
Goal: validate living behavior over longer windows only after 30-minute health is alive.

- Run 2-4 hour configured-source NIM validation.
- Capture runtime truth, liveness, latency/cost, cortex availability, embedding health, replay safety, action audit, memory pressure, traces, and recommended operator action.
- Compare behavior against the successful 30-minute baseline.
- Do not run overnight until 2-4 hour validation is alive and stable.

Exit criteria:
- live validation report is evidence-supported.
- long-test health is `alive`, not merely validator-passed.
- memory pressure remains bounded or recovers.
- no replay/training/action safety boundary is violated.
- operator report identifies remaining bottlenecks.

### Phase 15: Bounded Self-Improvement Readiness
Goal: decide whether the system is ready for autonomy level 5 experiments.

- Require passing Phase 12-14 evidence.
- Require promotion gate success for any learning artifact considered.
- Require benchmark improvement or clearly documented useful behavior.
- Require rollback metadata, approval, and no safety regression.
- Keep production runtime unchanged by default.

Exit criteria:
- autonomy ladder level 5 report passes.
- benchmark and long-run reports support the claim.
- rollback is tested.
- production model switch remains blocked unless a later explicit roadmap phase authorizes it.

## Benchmarks Worth Having
- Required now:
  - `tests/test_service_api.py`
  - `tests/test_service_manager.py`
  - `tests/test_service_benchmark.py`
  - `tests/test_long_test_runner.py`
  - `tests/test_trace_export_runner.py`
  - `tests/test_action_loop.py`
  - `tests/test_replay_training_gate.py`
  - `tests/test_replay_training_approval.py`
  - `tests/test_replay_training_plan.py`
  - `tests/test_replay_adapter_experiment.py`
  - `tests/test_replay_adapter_promotion_gate.py`
  - `tests/test_autonomy_ladder.py`
  - `tests/test_live_long_run_validation.py`
- Required before another live claim:
  - 30-minute configured-source NIM long run
  - service benchmark from the same validation slice
  - live long-run validation report
  - acceptance harness breakdown
  - memory pressure report
  - thought lifecycle report
- Required before learning claims:
  - replay holdout/eval split report
  - contamination/decontamination report
  - before/after service benchmark
  - before/after long-test health
  - no-mutation safety assertions
- Required before autonomy claims:
  - permission denial tests
  - approved action audit
  - rollback tests
  - delayed consequence tests
  - trace replay tests
- Deferred external pressure tests:
  - AgentBench-style interactive environments
  - Agent-SafetyBench-style tool safety checks
  - WebArena/GAIA-style tasks
  - ARC-style reasoning probes

## PR Protocol
Every PR must include:
- hypothesis: what should improve and why
- exact saved validation command/result
- runtime truth impact
- liveness impact
- memory pressure impact when live runtime changes
- safety flags for replay, action, sleep, feedback, external calls, and training
- performance note: latency, cost, memory, or simpler alternative considered
- docs update when behavior, API, safety policy, or benchmark meaning changes
- cleanup note: unused code/docs/tests removed or intentionally kept
- no compatibility layer kept only for old behavior unless it directly supports the roadmap

## Research Anchors
Use research to guide design, not to inflate claims.

- Agent evaluation must be interactive and multi-step, not only text-output scoring: [AgentBench](https://arxiv.org/abs/2308.03688).
- Tool-using agents introduce safety risks beyond the base model, so tool boundaries and audit trails must be first-class: [Agent-SafetyBench](https://arxiv.org/abs/2412.14470).
- Replay can help continual learning but must be controlled to avoid catastrophic forgetting and contamination: [Brain-inspired replay for continual learning](https://www.nature.com/articles/s41467-020-17866-2).
- Active inference frames action as uncertainty reduction under stable preferred states, not random tool use: [Free Energy Principle for Perception and Action](https://www.mdpi.com/1099-4300/24/2/301).
- Global workspace ideas support bounded selected active context as a requirement for coherent cognition: [Global Workspace Theory review](https://www.sciencedirect.com/science/article/pii/S0079612305500049).
- Reference-frame and Thousand Brains ideas remain useful for grounded local models and cortical-column-style representation, but they are research guidance, not production proof.

## Near-Term Next Slice
The next implementation slice is Phase 8 plus the minimum Phase 9 instrumentation:
- explain why the 30-minute run produced zero thoughts
- explain the partial acceptance harness result
- add thought lifecycle evidence
- add memory pressure evidence and recovery policy
- rerun a 30-minute configured-source NIM validation
- require `health_verdict=alive` before any longer run, adapter experiment, or autonomy expansion

## Non-Negotiable Safety Boundary
Until liveness is repaired and validated, Terminus must remain:
- no autonomous adapter training
- no production model switch
- no memory promotion from replay
- no fact promotion from dreamed or synthetic content
- no action execution without explicit approval
- no recurring actions without limits and rollback
- no sleep/action/tool side effects from replay artifacts
- no “living brain” claim unless saved long-run evidence supports behavioral liveness
