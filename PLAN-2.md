# HECSN / Terminus Living Brain Roadmap 2

## Summary
Roadmap 2 is complete. Terminus repaired the degraded 30-minute liveness state from Roadmap 1, added memory-pressure control, thought lifecycle/global workspace evidence, source-configuration reproducibility, isolated replay-to-adaptation evidence, approved level-2 action execution, multi-hour live validation, and bounded level-5 self-improvement readiness gates.

The system now has saved evidence for a configured-source NIM run with behavioral liveness over 120 minutes: runtime truth `alive`, long-test health `alive`, acceptance `passed`, thoughts generated, bounded memory pressure, healthy NIM embedding, no replay/training/action safety boundary violation, and operator-readable reports. This supports the claim that Terminus has a working live cortex-subcortex runtime under the tested conditions.

This still does not authorize an unbounded autonomous “living brain” claim. Production model switching remains blocked, self-improvement is readiness-only for bounded experiments, and web/API actions remain behind later approval gates. The next roadmap should start from this completed evidence base and focus on UI/operator usability, longer-duration validation, stronger external benchmarks, and carefully bounded autonomy expansion.

## Current Validated State
- Roadmap 2 final state: Phases 8-15 implemented and validated.
- UI overhaul attempt was intentionally rolled back. The UI remains on the previous shadcn/sidebar shell; future UI work should wait for the shadcn MCP workflow and start from the restored UI baseline.
- Implemented gates:
  - replay training approval artifacts
  - dry-run replay training plans
  - isolated adapter experiment artifacts
  - experimental promotion gate
  - autonomy ladder evaluator for levels 0-5
  - live long-run validation report checker
  - multi-hour live validation evaluator
  - bounded self-improvement readiness evaluator
  - validation report API for browsing saved JSON/README evidence from the UI/service
- Completed Phase 8-10 outcome:
  - 30-minute configured-source NIM run repaired from degraded to `health_verdict=alive`.
  - thoughts generated above zero.
  - acceptance harness passed.
  - memory pressure stayed bounded/recovered below high threshold.
  - thought lifecycle, rejection reasons, global workspace, source evidence, and memory pressure were added to long-test reports.
- Completed Phase 11-13 outcome:
  - source configuration is recorded in long-test and benchmark reports.
  - acceptance failures are actionable when present.
  - isolated adapter experiment evidence exists outside production runtime.
  - approved level-2 workspace action audit passed.
- Completed Phase 14-15 outcome:
  - 120-minute configured-source NIM validation passed.
  - Phase 14 multi-hour evaluator passed.
  - Phase 15 bounded self-improvement readiness passed.
  - autonomy ladder level 5 report passed for bounded experiments.
  - production model switch remains blocked.
- Saved Phase 7 evidence:
  - `reports/phase7_live_validation/long_test_20260501_001650.json`
  - `reports/phase7_live_validation/service_benchmark_20260501_001650.json`
  - `reports/phase7_live_validation/live_long_run_validation_20260501_001650.json`
- Original pre-repair 30-minute configured-source NIM long run:
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

## Final Roadmap 2 Evidence
- Phase 8-10 repaired 30-minute run:
  - `reports/phase8_10_validation_30m_rerun/long_test_20260501_031350.json`
- Phase 11-13 30-minute configured-source NIM run:
  - `reports/phase11_13_validation_30m/long_test_20260501_135625.json`
  - `health_verdict=alive`
  - `acceptance_verdict=passed`
  - `total_thoughts=14`
  - `unique_topics=36`
  - `final_memory_fill=0.26861572265625`
  - runtime truth verdict: `alive`
  - recommended action: `continue_monitoring`
  - NIM embedder: `available=true`, `degraded=false`, `nim_calls=66`, `fallback_calls=0`, `error_calls=0`, `rate_limit_hits=0`
  - source configuration hash: `84ae88966dc56e664a769717df471c227080fc62b3543ff8ee0089016fead7ce`
- Phase 11-13 benchmark and live validation:
  - `reports/phase11_13_validation_30m/service_benchmark/service_benchmark_phase11_13.json`
  - `reports/phase11_13_validation_30m/live_validation/live_long_run_validation_phase11_13.json`
  - live validator: `status=evidence_supported`, `passed=true`
- Phase 12 isolated learning evidence:
  - `reports/phase11_13_validation/phase12/phase12_replay_adaptation_experiment_1.json`
  - `status=passed_isolated_adaptation_evidence`
  - production runtime unchanged
- Phase 13 approved action evidence:
  - `reports/phase11_13_validation/phase13/phase13_approved_action_level2.json`
  - `status=executed_approved_workspace_action`
  - denied actions remain non-mutating
- Phase 14 multi-hour configured-source NIM run:
  - `reports/phase14_15_validation/phase14_multi_hour/long_test_20260501_170346.json`
  - `reports/phase14_15_validation/phase14_multi_hour/README.md`
  - `duration_minutes=120.0`
  - `health_verdict=alive`
  - `acceptance_verdict=passed`
  - `total_thoughts=42`
  - `unique_topics=85`
  - `avg_latency_ms=23369.01090909091`
  - `p95_latency_ms=64066.8`
  - `final_memory_fill=0.36016845703125`
  - memory pressure: low, unrecovered high pressure false
  - NIM embedder: `available=true`, `degraded=false`, `nim_calls=118`, `fallback_calls=0`, `error_calls=0`, `rate_limit_hits=0`
- Phase 14 benchmark and live validation:
  - `reports/phase14_15_validation/service_benchmark/service_benchmark_phase14_15.json`
  - `reports/phase14_15_validation/live_validation/live_long_run_validation_phase14_15.json`
  - live validator: `status=evidence_supported`, `passed=true`
  - replay safety: no training, memory mutation, feedback posting, digital action execution, external calls, or sleep side effects
  - action audit: advisory policy actuator, executable false
- Phase 14 final gate:
  - `reports/phase14_15_validation/phase14/phase14_multi_hour_live_validation.json`
  - `status=passed_multi_hour_living_evidence`
  - checks all true
  - remaining bottleneck field: `no_blocking_bottleneck_detected`
- Phase 15 promotion/readiness evidence:
  - `reports/phase14_15_validation/phase15/phase15_experimental_promotion_approval.json`
  - `reports/phase14_15_validation/phase15/phase15_experimental_promotion_gate.json`
  - `reports/phase14_15_validation/phase15/phase15_self_improvement_readiness.json`
  - readiness status: `ready_for_bounded_level_5_experiment`
  - autonomy ladder level 5: `approved_for_level`
  - production model switch allowed: `false`
  - production runtime changed: `false`
  - rollback metadata recorded and tested by isolation/production-unchanged evidence

## Final Validation Commands
- Focused Phase 14/15/API/service-manager suite:
  - `python -m pytest tests/test_multi_hour_live_validation.py tests/test_self_improvement_readiness.py tests/test_live_long_run_validation.py tests/test_service_api.py::ServiceApiTerminusRuntimeTests::test_validation_report_endpoints_list_and_read_reports tests/test_service_manager.py::ServiceManagerTerminusRuntimeTests::test_provider_consequence_family_divergence_split_separates_mixed_query_branches`
  - result: `14 passed in 30.36s`
- Roadmap-critical suite:
  - `python -m pytest tests/test_multi_hour_live_validation.py tests/test_self_improvement_readiness.py tests/test_live_long_run_validation.py tests/test_service_api.py::ServiceApiTerminusRuntimeTests::test_validation_report_endpoints_list_and_read_reports tests/test_service_benchmark.py tests/test_long_test_runner.py tests/test_autonomy_ladder.py tests/test_replay_adapter_promotion_gate.py tests/test_replay_adaptation_experiment_1.py tests/test_approved_action_level2.py`
  - result: `43 passed in 10.98s`
- Full suite attempt after Phase 14/15:
  - `python -m pytest`
  - result: `1218 passed, 3 skipped, 1 warning, 1 failed in 1115.48s`
  - failed test: `tests/test_service_manager.py::ServiceManagerTerminusRuntimeTests::test_provider_consequence_family_divergence_split_separates_mixed_query_branches`
  - immediate isolated rerun passed: `1 passed in 26.31s`
  - interpretation: the full-suite single-shot was not clean because of one nondeterministic delayed-consequence assertion; the failed test passed in isolation and is not attributed to Phase 14/15 changes.
- UI build during attempted UI work:
  - `npm run build` in `HECSN_UI`
  - result: build passed
  - UI code changes were rolled back at operator request.

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

## Completed Roadmap

### Phase 8: Live Liveness Repair - Complete
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

Result: complete. The repaired configured-source NIM run reached `health_verdict=alive`, thoughts were generated, acceptance passed, and safety boundaries held.

### Phase 9: Memory Pressure And Working Set Control - Complete
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

Result: complete. Memory pressure is recorded in runtime truth and long-test reports; the 120-minute run ended at `final_memory_fill=0.36016845703125` with no unrecovered high pressure.

### Phase 10: Thought Trace And Global Workspace Evidence - Complete
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

Result: complete. Thought lifecycle, global workspace, selected context, topic diversity, and hypothesis/fact boundaries are saved in long-test reports.

### Phase 11: Source Configuration And Acceptance Harness Hardening - Complete
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

Result: complete. Long-test and benchmark reports now include source configuration evidence and actionable acceptance failure details.

### Phase 12: Replay-To-Adaptation Experiment 1 - Complete
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

Result: complete. Isolated adapter evidence exists outside production runtime; no production switch occurred.

### Phase 13: Approved Action Loop Level 2 - Complete
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

Result: complete. Approved workspace action execution passed audit tests and remains bounded.

### Phase 14: Multi-Hour Live Validation - Complete
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

Result: complete. The 120-minute configured-source NIM run passed with `health_verdict=alive`, validator `passed=true`, bounded memory pressure, healthy NIM embedding, and no safety boundary violation.

### Phase 15: Bounded Self-Improvement Readiness - Complete
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

Result: complete. The level-5 readiness report passed for bounded experiments, rollback metadata is recorded, and production model switching remains blocked.

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
Roadmap 2 is closed. The next roadmap should start from the completed Phase 14/15 evidence and focus on:
- UI/operator overhaul using the shadcn MCP workflow, starting from the restored pre-overhaul UI baseline.
- Longer validation: 4-hour and then overnight configured-source NIM runs only if 2-4 hour health remains alive and stable.
- External pressure tests: AgentBench-style interactive tasks, Agent-SafetyBench-style tool safety, WebArena/GAIA-style tasks, and ARC-style reasoning probes.
- Autonomy expansion beyond level 2 only with explicit approval, recurring limits, rollback, delayed consequence tracking, and trace replay.
- Bounded level-5 experiment design that remains isolated from production runtime.
- Latency investigation: Phase 14 passed, but long-test average latency was about 23.37 seconds and p95 was about 64.07 seconds.

## Non-Negotiable Safety Boundary
After Roadmap 2, Terminus must remain:
- no unbounded autonomous adapter training
- no production model switch
- no memory promotion from replay
- no fact promotion from dreamed or synthetic content
- no action execution without explicit approval
- no recurring actions without limits and rollback
- no sleep/action/tool side effects from replay artifacts
- no “living brain” claim beyond the tested configured-source NIM conditions unless saved long-run evidence supports behavioral liveness

## UI Operator Overhaul - 2026-05-01
- Reworked the restored `HECSN_UI` shadcn shell into an operator dashboard organized by Monitor, Control, Evidence, and Model lanes.
- Added a persistent shadcn tab strip across the top-level app so operators can move directly between Overview, Mind, Sensory, Dynamics, Neural Space, Learning, Workspace, Grounding, Validation, Systems, Growth, Checkpoints, and Traces without relying only on the sidebar.
- Promoted runtime controls to the global header: live/offline status, token count, dirty-state warning, Start, Tick, Stop, and API/context configuration are now visible from every screen.
- Added a new Validation Evidence screen backed by `/terminus/validation/reports` and `/terminus/validation/report`, including Phase 14/15 status cards, a report index table, selected README/JSON artifact viewing, and extracted safety-field inspection.
- Tightened the visual system away from the prior purple/blue gradient shell toward a quieter shadcn operator-console palette with neutral structure, teal primary actions, clearer active states, and less decorative styling.
- Validation completed: `npm run build` in `HECSN_UI` passed. The build still reports the existing large Three.js/Recharts chunk warning, which is expected for the current lazy-loaded visualization stack.
