## Problem Statement

The Living Loop module (`living_loop.py`) is a 3195-line monolith containing all runtime record types, policy scoring, replay planning, and the Operational Self-Model. It has no internal seams — every consumer (LivingStatusMixin, ReplayRuntimeMixin, RuntimeEvidenceMixin, ServiceManager, and the test suite) imports from a single file, forcing any change to any layer to reparse and re-validate the entire monolith. The file is difficult to navigate, impossible to review in a single pass, and resists isolated testing of individual layers.

## Solution

Split the Living Loop monolith into four depth-aligned modules that mirror its existing internal dependency stack: Runtime Records → Policy Scoring → Replay Planning → Operational Self-Model. Each module encapsulates one depth layer with a clear interface. The original `living_loop.py` becomes a backward-compatible re-export shim so that no consumer imports break. This preserves all existing behaviour while giving each layer testability, locality, and independent change velocity.

## User Stories

1. As a Terminus developer, I want runtime record types in their own module, so that I can add a new record type without touching policy or replay code.
2. As a Terminus developer, I want policy scoring logic in its own module, so that I can modify the policy actuator without parsing 3000 lines of unrelated dataclasses.
3. As a Terminus developer, I want replay planning in its own module, so that I can reason about replay candidate ranking in isolation from the self-model.
4. As a Terminus developer, I want the Operational Self-Model in its own module, so that I can evolve self-model surface methods without risking side effects in record parsing.
5. As a Terminus developer, I want benchmark telemetry in its own module, so that I can add new telemetry dimensions without touching the self-model.
6. As a Terminus developer, I want a re-export shim in the original location, so that existing consumer imports continue to work without modification.
7. As a Terminus developer, I want each depth layer to be under 800 lines, so that I can review any module in a single pass.
8. As a Terminus developer, I want private helpers shared across layers in their own module, so that they have a single source of truth rather than being duplicated or awkwardly public.
9. As a Terminus developer, I want the dependency direction between new modules to be strictly unidirectional (lower layers never import from higher layers), so that the depth stack remains acyclic.
10. As a Terminus developer, I want the existing test suite to pass without modification after the split, so that the refactoring is verified as behaviour-preserving.
11. As a Terminus developer, I want new per-module test files, so that each layer can be tested in isolation.
12. As a Terminus operator, I want the Living Loop status API to return identical payloads after the split, so that no downstream consumer breaks.
13. As a Terminus operator, I want replay plan payloads to be identical after the split, so that replay sampling workflows are unaffected.
14. As a Terminus operator, I want policy actuator recommendations to be identical after the split, so that advisory actions remain consistent.
15. As a Terminus developer, I want the frozen dataclass contracts (from_payload / to_payload round-trips) preserved exactly, so that serialisation behaviour is unchanged.
16. As a Terminus developer, I want enum types (PredictionStatus, ActionExecutionStatus, VerificationStatus, ConsolidationStatus) to remain importable from their new module, so that type annotations across the codebase stay valid.
17. As a Terminus developer, I want the re-export shim to re-export every public symbol currently available from living_loop, so that no import breaks silently.
18. As a Terminus developer, I want the safety boundary constants (REPLAY_SAMPLE_SAFETY_BOUNDARIES) to live in the replay module, so that they are co-located with the logic that uses them.
19. As a Terminus developer, I want an ADR recording this split, so that future contributors understand why the modules are organised this way.
20. As a Terminus developer, I want CONTEXT.md updated with the new module names, so that the domain vocabulary reflects the actual code structure.
21. As a Terminus developer, I want the helper module to export only the functions that are genuinely shared, so that each module's private logic stays private.
22. As a Terminus developer, I want WorldModelLiteSummary.from_records to remain callable from the policy module, so that the Operational Self-Model can still delegate world-model construction.
23. As a Terminus developer, I want build_replay_plan to remain callable from both the replay module and the status mixin, so that the existing call graph is preserved.
24. As a Terminus developer, I want the test_living_loop_primitives test suite to continue passing when importing from the re-export shim, so that backward compatibility is verified.
25. As a Terminus developer, I want each new module to have a module docstring explaining its depth layer and dependencies, so that future readers understand the architecture without reading the ADR.

## Implementation Decisions

- **Four depth-aligned modules** will be extracted from the monolith, corresponding to the four naturally acyclic layers already present:
  1. **Runtime Records module** — Layer A: all enums, frozen dataclasses (PredictionRecord, ActionVerificationRecord, ActionExecutionRecord, RuntimeEpisodeTrace, SkillMemoryRecord, ProvenanceState, ConsolidationRecord), and the private helpers they depend on.
  2. **Policy Scoring module** — Layer B: PolicyScore, WorldModelLiteSummary (including from_records), PolicyActuatorRecommendation, build_policy_actuator_status, and the ~10 policy-specific private helpers.
  3. **Replay Planning module** — Layer C: ReplayCandidate, ReplayPlan, build_replay_plan, replay_candidate_safety_flags, REPLAY_SAMPLE_SAFETY_BOUNDARIES, REPLAY_PLAN_PRIORITY_WEIGHTS, REPLAY_REASON_PRECEDENCE, and the ~20 replay-specific private helpers.
  4. **Operational Self-Model module** — Layer D: OperationalSelfModel (build, from_payload, to_payload, all _surface_* methods) and build_runtime_benchmark_telemetry with its ~8 telemetry-specific helpers.

- **A shared helpers module** will house the 12 truly cross-layer private functions (_stable_id, _clean_text, _clamp01, _safe_ratio, _limited_unique_clean_text, _latest_text, _as_mapping, _enum_value, _provenance_value, _verification_status_from_payload, _safe_float, _coerce_world_model_lite). These are used by all four layers and must have a single source of truth.

- **Dependency direction is strictly unidirectional**: Helpers → Records → Policy → Replay → Self-Model. No upward import from a higher layer to a lower one. The helpers module has no dependencies on any other new module.

- **Backward compatibility via re-export shim**: The original living_loop.py file will become a thin re-export module that imports and re-exports every public symbol from the four new modules and the helpers module. This ensures zero consumer breakage — all five current import sites (LivingStatusMixin, ReplayRuntimeMixin, RuntimeEvidenceMixin, ServiceManager, test_living_loop_primitives) continue to work without modification.

- **No API contract changes**: All frozen dataclass shapes, from_payload/to_payload round-trips, enum values, build function signatures, and safety boundary constants remain identical. The split is purely structural — no behavioural change.

- **Module naming convention** follows the existing service package pattern: each module gets a descriptive name within the service package, co-located with the consumers.

- **An ADR will be written** in docs/adr/ recording the decision, the four-layer depth stack, the dependency direction constraint, and the backward-compatibility strategy.

- **CONTEXT.md will be updated** to add module-level entries for each new module, mapping them to the existing domain vocabulary (Runtime Records → Living Loop records, Policy Scoring → Policy Actuator, Replay Planning → Replay Pipeline planning stage, Operational Self-Model → Runtime Truth / Living Loop self-model).

## Testing Decisions

- **A good test** verifies external behaviour (from_payload/to_payload round-trips, build function outputs given known inputs, enum membership) without asserting internal private helper structure or module layout.
- **All four new modules will be tested** with dedicated per-module test files, in addition to the existing test_living_loop_primitives.py which continues to test via the re-export shim.
- **Runtime Records module**: Test every dataclass from_payload → to_payload round-trip, enum value membership, and edge cases (empty payloads, missing keys, invalid enum values). Prior art: the existing test_living_loop_primitives.py already tests these round-trips.
- **Policy Scoring module**: Test WorldModelLiteSummary.from_records with known prediction/action/consolidation counts, PolicyActuatorRecommendation output from build_policy_actuator_status with known living-loop payloads, and policy action selection logic. Prior art: existing tests for build_policy_actuator_status.
- **Replay Planning module**: Test build_replay_plan with known payloads, replay candidate ranking order, safety flag computation, and priority score calculation. Prior art: existing tests for build_replay_plan and replay_candidate_safety_flags.
- **Operational Self-Model module**: Test OperationalSelfModel.build with known inputs, to_payload structure, surface method outputs (uncertain domains, recent failures, memory health, grounding health, budgets, capabilities). Prior art: existing tests for OperationalSelfModel and build_runtime_benchmark_telemetry.
- **Re-export shim**: Test that every public symbol importable from the original location resolves correctly. This is a single integration test.
- **Regression**: The full existing test_living_loop_primitives.py suite must pass unchanged after the split.

## Out of Scope

- Updating consumer imports to import directly from the new modules (deferred to a follow-up; the re-export shim handles compatibility).
- Refactoring the Service Manager God class or any of the mixin classes (LivingStatusMixin, ReplayRuntimeMixin, RuntimeEvidenceMixin) — those are separate deepening opportunities.
- Adding new dataclasses, enums, or build functions — this PRD is purely about extracting existing code into modules.
- Changing any API endpoint payloads or schemas.
- Modifying the Cortex, Subcortex, or ThoughtLoop packages.
- Introducing Protocol/ABC interfaces for the new modules — the current concrete dataclass interfaces are sufficient.
- Performance optimisation of any build function or telemetry computation.

## Further Notes

- The four-layer dependency stack was identified by tracing all internal imports and call references within the monolith. Layer A (Records) uses no other layer. Layer B (Policy) imports only Layer A enums. Layer C (Replay) imports Layer B types. Layer D (Self-Model) imports all three. This acyclic structure makes the split mechanically clean — no circular dependencies to resolve.
- The 12 shared helpers were identified by tracing which private functions are called from more than one layer. Functions used by only one layer (e.g., _policy_latency_pressure, _replay_rank_candidates) stay co-located with their consumer.
- The existing test file (test_living_loop_primitives.py, ~1068 lines) already provides strong regression coverage. The new per-module test files should extract and expand the tests relevant to each module rather than duplicating the entire suite.
- This deepening is the first of eight identified opportunities in the service package. The other seven (Service Manager God class, Delayed Consequence coupling, Action Loop extraction, etc.) are deferred and should each get their own PRD.
