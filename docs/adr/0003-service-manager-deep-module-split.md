# ADR 0003: Dissolve Service Manager God class into deep modules with real seams

## Status

Accepted

## Context

Before this ADR, the Service Manager (`src/hecsn/service/manager.py`) was a God class composed of 21 mixin classes that shared 50+ mutable state attributes via `self._*`. It was the single seam between the FastAPI layer and the runtime — but it had no internal seams. Every mixin read and wrote every other mixin's state. The test file (`tests/test_service_manager.py`) had to construct the entire Service Manager with all its real dependencies to test any one mixin-shaped behavior.

Analysis of the mixin call graph reveals:

1. **No independent interfaces.** Mixins don't have their own test surfaces — they can only be tested as part of the full Service Manager.
2. **Writer ambiguity.** State like `_brain_source_utility` is initialized in `manager.__init__`, mutated by DelayedConsequenceMixin, and read by TerminusAutonomyCore and SourceFocusMixin. No single module owns it.
3. **Cross-cutting infrastructure in the wrong place.** `_mark_mutated()` (called by 8+ mixins) lived in the former interaction mixin. `_record_brain_event_locked()` (called by 8+ mixins) lived in PersistenceMixin. `_normalize_action_text()` (called by 8+ mixins) lived in RuntimeFeedbackMixin.
4. **Shallow mixins that fail the deletion test.** LivingStatusCore (238L), SensoryPreviewMixin (64L), and the former shallow reporting half are pure read-side projections — they add no leverage, just file-splitting.

The tightest bidirectional coupling pairs are:
- BrainRuntimeMixin ↔ DelayedConsequenceMixin (12 cross-reads)
- BrainRuntimeMixin ↔ TerminusAutonomyCore (8 cross-reads)
- RuntimeStatusCore → BrainRuntime (40+ one-directional reads)

## Decision

Dissolve the mixin inheritance into 15 independent deep modules, each with its own interface, its own state, and constructor-injected dependencies. The Service Manager is a thin composition root that wires modules together and exposes the FastAPI facade.

The accepted implementation goes further than simple mixin extraction:

- `HECSNServiceManager` does not use catch-all `__getattr__`, `__setattr__`, or legacy unbound mixin fallback routing.
- Manager-owned ADR runtime state is moved behind the owning modules. Brain Runtime owns source runtimes, source utility, tick counters, stream epochs, and sensory episode/preview counters. Interaction Pipeline owns query gap history and runtime episode traces. RuntimeState owns mutation truth and brain event history.
- Public runtime behavior is reached through `RuntimeFacade`; any remaining manager methods are explicit internal dependency callbacks and state bridges, not manager-bound attribute magic, import-time dynamic delegate installation, generic mixin delegate trampolines, or manager-private wrappers around former interaction helper names.
- Deep modules no longer inherit from `ManagerBoundModule`; the owner-forwarder helper has been removed. RuntimeController and AutonomyPlanner use explicit dependency adapters, and the remaining service modules that previously used owner forwarders now receive explicit dependency objects or constructor callbacks. Module-level `*Mixin = ...` compatibility aliases have been removed. StatusReadModel owns the sensory preview projection directly.

### Module inventory

| Module | Absorbs | Lines | Owns (state) |
|---|---|---|---|
| RuntimeState | new | ~50 | `dirty_state`, `state_revision`, `last_event`, `recent_events`, brain event log |
| DelayedConsequenceTracker | DelayedConsequenceMixin | 2644 | consequence records, cooled/retired/compacted/split/remerged totals |
| AutonomyPlanner | TerminusAutonomyCore behavior | 1771 | (reads shared state through an explicit dependency adapter) |
| BrainRuntime | BrainRuntimeMixin | 1148 | source runtimes, source utility, tick counters, stream epochs, sensory episode/preview counters |
| InteractionPipeline + OperatorInteractionRuntime + RuntimeEvidenceReporter | former InteractionRuntimeMixin + runtime evidence behavior | ~2000 | query gap history, runtime episode traces, operator acquisition flow, replay/export evidence |
| FeedbackApplier | former RuntimeFeedbackMixin | 244 | (applies to targets, no persistent own state) |
| SourceFocusScorer | SourceFocusMixin | 373 | (scoring is stateless per call) |
| RuntimeController | RuntimeControlMixin + RuntimePrewarmer behavior | ~1600 | brain thread, stop event, active execution counters, prewarm thread lifecycle |
| StatusReadModel | RuntimeStatusCore + LivingStatusCore + sensory preview + shallow reporting behavior | ~1000 | cached status/telemetry/terminus snapshots |
| ActionExecutor | former ActionRuntimeMixin and deleted action-assist mixin module | ~750 | action history, audited action reuse |
| RuntimePersistence | PersistenceMixin | 233 | trace history and checkpoint save/restore orchestration |
| RuntimeConfig | RuntimeConfigMixin | 513 | (stateless — normalization only) |
| RuntimeSources | RuntimeSourcesMixin | 390 | source runtime dataclasses |
| ReplayController | ReplayRuntimeMixin + ReplayDatasetPackager | ~950 | replay sample history, replay planning and operator-gated sampling |

### Design constraints

1. **Constructor injection and explicit dependencies.** Each module declares its production dependencies through constructor wiring, explicit dependency adapters, or narrow owner interfaces. No module may rely on a catch-all manager-bound base class.
2. **Writer owns state.** The module that mutates a piece of state owns it. Other modules access it through the owner's read methods. Key ownership assignments:
 - `brain_source_utility` → BrainRuntime (mutated by DelayedConsequence via `brain_runtime.update_source_utility()`)
 - `brain_recent_query_gaps` → InteractionPipeline (read by Autonomy and SourceFocus via `interaction.recent_query_gaps()`)
 - `action_history` → ActionExecutor (read by 8+ modules via `action_executor.action_history()` and `action_executor.recent_relevant_actions()`)
3. **Shared RLock.** Each module receives the RLock as a constructor parameter. No module owns the lock. This avoids deadlock while preserving the current thread-safety model (brain thread + FastAPI handlers).
4. **Direct references between modules.** Modules hold references to each other through injected dependencies or explicit dependency adapters. No event bus, no manager orchestration. The dependency graph must stay visible in code.
5. **RuntimeState for cross-cutting concerns.** A tiny module owns `dirty_state`, `state_revision`, and the brain event log. Modules call `runtime_state.mark_mutated()` and `runtime_state.record_event()` instead of depending on the Service Manager's internals.
6. **Shallow modules collapsed.** LivingStatusCore, SensoryPreviewMixin, and the shallow half of reporting behavior collapse into StatusReadModel. StatusReadModel owns sensory preview serialization directly; the old mixin does not sit on the read-model path.
7. **InteractionRuntime + RuntimeEvidence merged.** Every interaction (query/feed/respond) produces an episode trace — they're tightly coupled (10 cross-reads) and form a single interaction pipeline.

### Relationship to ADR 0001 and ADR 0002

This ADR does not reopen or contradict ADR 0001 or ADR 0002. The Living Loop depth-aligned module split and its unidirectional dependency chain remain intact (ADR 0001). RuntimeState remains the single owner of mutation truth, brain event history, `dirty_state`, `state_revision`, `last_event`, and `recent_events` (ADR 0002). This ADR extends the module inventory around the existing RuntimeState and Living Loop seams without redefining their ownership boundaries.

### Deletion policy

The Service Manager no longer owns the operator-facing runtime method surface. FastAPI routes call `RuntimeFacade`, and manager methods that remain are internal dependency callbacks for explicitly wired deep modules. Compatibility is a short-lived migration state, not an architecture goal: once a deep owner exists and callers are migrated, the old module, alias, or wrapper is deleted and guarded by absence tests.

`HECSNServiceManager` must not inherit from legacy aliases or recover behavior through an unbound mixin fallback. Interaction pipeline collaborators are wired as constructor callbacks to `OperatorInteractionRuntime`, not exposed as manager `_build_query_locked`, `_plan_gaps_locked`, `_record_recent_query_gap_locked`, or `interaction_state_snapshot` methods. The former `interaction_runtime.py` compatibility module is deleted. Action execution and audited action reuse are owned directly by `ActionExecutor`; the stale `action_runtime.py` and `action_assist.py` mixin-shaped modules are deleted. Operator feedback is owned directly by `FeedbackApplier`; the stale `runtime_feedback.py` mixin module is deleted after its remaining assertions move to the real owner tests, and the Service Manager no longer duplicates feedback normalization/application logic. Runtime restore and delayed-consequence restore now target their owner modules directly instead of preserving manager-level `restore_runtime_state` or `restore_state` wrappers. Delayed-consequence and interaction callbacks call `ActionExecutor` directly instead of preserving manager-private action wrappers. The generic `_call_mixin_delegate` trampoline is deleted; remaining manager callbacks must name their owner explicitly while the next modules are extracted or renamed.

Runtime prewarm behavior is active Terminus queue-warming behavior, not a mixin. `RuntimePrewarmMixin` is renamed to `RuntimePrewarmer`; `RuntimeControl` may still compose it while runtime-control and prewarm lifecycle ownership is being separated, but the old mixin identity is gone from active source.

Sensory preview projection is owned by `StatusReadModel`; the stale `sensory_preview.py` mixin module is deleted rather than preserved as an alternate API path.

Multimodal sensory execution is active Terminus runtime behavior, not a mixin. `SensoryRuntimeMixin` is renamed to `SensoryRuntimeCore`; the remaining manager callbacks are explicit transition seams while this behavior is moved further behind domain-owned modules.

Runtime status evidence is active observability behavior for Runtime Truth, not a mixin. `StatusRuntimeMixin` is renamed to `RuntimeStatusCore`; its remaining helpers are used as explicit status/evidence callbacks while StatusReadModel continues to own the public read surface.

Living-loop and policy-actuator status are active observability behavior, not a mixin. `LivingStatusMixin` is renamed to `LivingStatusCore`; the read model remains the public facade for those snapshots.

Runtime source stream construction is owned by `RuntimeSources`. The Service Manager no longer exposes `_build_source_stream_from_spec`, `_build_brain_source_stream_locked`, `_build_sensory_stream_locked`, or `_build_sensory_stream_from_spec`; BrainRuntime receives RuntimeSources constructor callbacks directly, and RuntimePrewarmer/SensoryRuntimeCore rebuild detached streams through RuntimeSources static builders. Tests patch RuntimeSources or the owner runtime module constants, not manager compatibility names.

Interaction and persistence stores are owned by `InteractionPipeline` and `RuntimePersistence`, not the Service Manager. The manager no longer exposes `persist_trace`, `load_persisted_traces`, `load_interaction_state`, query-gap store helpers, runtime-episode store helpers, or `cognitive_signal_state`; callers use the owner module or `RuntimeFacade` for the operator-facing signal surface.

Owner callbacks are not compatibility wrappers. When the composition root already has the owner module, callbacks must point directly at that owner. Delayed-consequence, source-focus, runtime-evidence, and autonomy-calibration callbacks therefore do not survive as Service Manager private methods.

### Migration strategy

This migration is implemented far enough that the Service Manager is no longer a legacy mixin inheritance surface and no longer has catch-all attribute routing. RuntimeController and AutonomyPlanner moved off manager-backed owner forwarders, the remaining owner-forwarded modules have been converted to explicit dependency objects or constructor callbacks, and broad manager facade delegates are now explicit named methods rather than import-time installed wrappers. Remaining compatibility work should continue by moving mixin-named implementation modules to domain-named modules where import compatibility allows.

1. Keep the ADR guard in `tests/test_adr_service_manager_composition.py` as the first regression target for future service-manager architecture work.
2. When touching transitional facade delegates, prefer replacing them with a concrete constructor dependency or an interface method on the true owner.
3. Keep service-level tests focused on public behavior. Module tests should use fakes/adapters at the same interfaces production uses.
4. Delete compatibility aliases once the active owner exists; tests should assert absence rather than import the old name.

## Rationale

### Why not keep the mixin architecture?

Mixins have no seams. They share all state through `self`, so there's no way to test one mixin in isolation. The mixin pattern is file-splitting, not module-splitting — it creates the illusion of separation without the substance. The deletion test confirms: if you deleted all the mixin indirection and put the code back in one class, you'd have the same class with the same methods. The mixins add no leverage.

### Why not event bus / command queue?

An event bus maximizes decoupling but makes call flows invisible — you can't trace which module will respond to an event without reading every subscriber. This hurts AI-navigability. A command queue (brain thread owns all mutation) is a large refactor that changes every mutation from sync to async. Direct references are the simplest model that gives us real seams and testable interfaces.

### Why shared RLock instead of per-module locks?

Per-module locks create deadlock risk (A holds lock_A, waits for B which holds lock_B). The current codebase is entirely sync and uses a single RLock. Keeping a shared RLock preserves the current thread-safety model while giving each module an independently-mockable lock parameter. Tests can pass a no-op lock or run single-threaded.

### Why RuntimeState as a separate module?

`_mark_mutated()` and `_record_brain_event_locked()` are called by 8+ modules. If they lived on any one deep module, that module would become an implicit base class — every other module would depend on it. A tiny RuntimeState module with a 2-method interface avoids this. Tests provide a recording fake that asserts on call counts.

## Consequences

### Positive

- Each module has an independent interface and can be tested with its own test doubles
- Two adapters per seam (production wiring + test fakes) makes every seam real
- The Service Manager becomes a thin composition root instead of a mixin inheritance surface
- State ownership is explicit: the module that mutates owns, others read through the interface
- Legacy manager-bound attribute magic is removed from the Service Manager and deep modules
- Cross-cutting concerns (`mark_mutated`, `record_event`) have a dedicated home
- Shallow modules are collapsed — no more pass-throughs
- Future changes to one module don't require re-validating unrelated modules
- Retired runtime path state is named as cleanup scaffolding, not an active runtime module
- The 9,277-line test file can decompose into 15 per-module test files

### Negative

- Constructor signatures get larger — the BrainRuntime module, as the central hub, will have many dependencies
- Wiring code in the Service Manager's composition root must be maintained
- Some mixin-named implementation modules remain while older tests/imports are retired, and each should move toward deletion or a domain-owned rename
- Some cross-module calls that were previously `self.method()` become `other_module.method()`, which is a shallow syntax change with no semantic difference — but it adds a reference that must be managed

### Neutral

- The total line count stays roughly the same — this is a reorganization, not a reduction
- The Service Manager's public API surface doesn't change — FastAPI routes are untouched

## References

- PRD #50: Close ADR 0003 Service Manager deep module split
- Issue #61: Accept ADR 0003 and align domain docs
- ADR 0001: Split Living Loop monolith into depth-aligned modules
- ADR 0002: Runtime State owns mutation truth and brain event history
