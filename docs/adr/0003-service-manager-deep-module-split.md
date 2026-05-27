# ADR 0003: Dissolve Service Manager God class into deep modules with real seams

## Status

Accepted

## Context

Before this ADR, the Service Manager (`src/hecsn/service/manager.py`) was a God class composed of 21 mixin classes that shared 50+ mutable state attributes via `self._*`. It was the single seam between the FastAPI layer and the runtime — but it had no internal seams. Every mixin read and wrote every other mixin's state. The test file (`tests/test_service_manager.py`) had to construct the entire Service Manager with all its real dependencies to test any one mixin-shaped behavior.

Analysis of the mixin call graph reveals:

1. **No independent interfaces.** Mixins don't have their own test surfaces — they can only be tested as part of the full Service Manager.
2. **Writer ambiguity.** State like `_brain_source_utility` is initialized in `manager.__init__`, mutated by DelayedConsequenceMixin, and read by TerminusAutonomyMixin and SourceFocusMixin. No single module owns it.
3. **Cross-cutting infrastructure in the wrong place.** `_mark_mutated()` (called by 8+ mixins) lives in InteractionRuntimeMixin. `_record_brain_event_locked()` (called by 8+ mixins) lives in PersistenceMixin. `_normalize_action_text()` (called by 8+ mixins) lives in RuntimeFeedbackMixin.
4. **Shallow mixins that fail the deletion test.** LivingStatusMixin (238L), SensoryPreviewMixin (64L), and the shallow half of ReportingMixin are pure read-side projections — they add no leverage, just file-splitting.

The tightest bidirectional coupling pairs are:
- BrainRuntimeMixin ↔ DelayedConsequenceMixin (12 cross-reads)
- BrainRuntimeMixin ↔ TerminusAutonomyMixin (8 cross-reads)
- StatusRuntimeMixin → BrainRuntimeMixin (40+ one-directional reads)

## Decision

Dissolve the mixin inheritance into 15 independent deep modules, each with its own interface, its own state, and constructor-injected dependencies. The Service Manager is a thin composition root that wires modules together and exposes the FastAPI facade.

The accepted implementation goes further than simple mixin extraction:

- `HECSNServiceManager` does not use catch-all `__getattr__`, `__setattr__`, or legacy unbound mixin fallback routing.
- Manager-owned ADR runtime state is moved behind the owning modules. Brain Runtime owns source runtimes, source utility, tick counters, stream epochs, and sensory episode/preview counters. Interaction Pipeline owns query gap history and runtime episode traces. RuntimeState owns mutation truth and brain event history.
- Public Service Manager methods remain stable, but internal compatibility is expressed as explicit facade methods and explicit state properties, not manager-bound attribute magic or import-time dynamic delegate installation.
- Deep modules no longer inherit from `ManagerBoundModule`; the owner-forwarder helper has been removed. RuntimeController and AutonomyPlanner use explicit dependency adapters, and the remaining service modules that previously used owner forwarders now receive explicit dependency objects or constructor callbacks. Module-level `*Mixin = ...` compatibility aliases have been removed. StatusReadModel owns the sensory preview projection directly.

### Module inventory

| Module | Absorbs | Lines | Owns (state) |
|---|---|---|---|
| RuntimeState | new | ~50 | `dirty_state`, `state_revision`, `last_event`, `recent_events`, brain event log |
| DelayedConsequenceTracker | DelayedConsequenceMixin | 2644 | consequence records, cooled/retired/compacted/split/remerged totals |
| AutonomyPlanner | TerminusAutonomyMixin behavior | 1771 | (reads shared state through an explicit dependency adapter) |
| BrainRuntime | BrainRuntimeMixin | 1148 | source runtimes, source utility, tick counters, stream epochs, sensory episode/preview counters |
| InteractionPipeline | InteractionRuntimeMixin + RuntimeEvidenceMixin | ~2000 | query gap history, runtime episode traces |
| FeedbackApplier | RuntimeFeedbackMixin | 244 | (applies to targets, no persistent own state) |
| SourceFocusScorer | SourceFocusMixin | 373 | (scoring is stateless per call) |
| RuntimeController | RuntimeControlMixin + RuntimePrewarmMixin behavior | ~1600 | brain thread, stop event, active execution counters, prewarm thread lifecycle |
| StatusReadModel | StatusRuntimeMixin + LivingStatusMixin + sensory preview + shallow ReportingMixin behavior | ~1000 | cached status/telemetry/terminus snapshots |
| ActionExecutor | ActionRuntimeMixin + ActionAssistMixin | ~750 | action history |
| RetiredRuntimePathState | retired_runtime_path.py | minimal | retired-path status snapshot only; no ask/sleep/thought/action hooks |
| RuntimePersistence | PersistenceMixin | 233 | trace history and checkpoint save/restore orchestration |
| RuntimeConfig | RuntimeConfigMixin | 513 | (stateless — normalization only) |
| RuntimeSources | RuntimeSourcesMixin | 390 | source runtime dataclasses |
| ReplayController | ReplayRuntimeMixin + ReplayDatasetBundleMixin | ~950 | replay sample history, replay planning and operator-gated sampling |

### Design constraints

1. **Constructor injection and explicit dependencies.** Each module declares its production dependencies through constructor wiring, explicit dependency adapters, or narrow owner interfaces. No module may rely on a catch-all manager-bound base class.
2. **Writer owns state.** The module that mutates a piece of state owns it. Other modules access it through the owner's read methods. Key ownership assignments:
 - `brain_source_utility` → BrainRuntime (mutated by DelayedConsequence via `brain_runtime.update_source_utility()`)
 - `brain_recent_query_gaps` → InteractionPipeline (read by Autonomy and SourceFocus via `interaction.recent_query_gaps()`)
 - `action_history` → ActionExecutor (read by 8+ modules via `action_executor.action_history()` and `action_executor.recent_relevant_actions()`)
3. **Shared RLock.** Each module receives the RLock as a constructor parameter. No module owns the lock. This avoids deadlock while preserving the current thread-safety model (brain thread + FastAPI handlers).
4. **Direct references between modules.** Modules hold references to each other through injected dependencies or explicit dependency adapters. No event bus, no manager orchestration. The dependency graph must stay visible in code.
5. **RuntimeState for cross-cutting concerns.** A tiny module owns `dirty_state`, `state_revision`, and the brain event log. Modules call `runtime_state.mark_mutated()` and `runtime_state.record_event()` instead of depending on the Service Manager's internals.
6. **Shallow modules collapsed.** LivingStatusMixin, SensoryPreviewMixin, and the shallow half of ReportingMixin collapse into StatusReadModel. StatusReadModel owns sensory preview serialization directly; the old mixin does not sit on the read-model path.
7. **InteractionRuntime + RuntimeEvidence merged.** Every interaction (query/feed/respond) produces an episode trace — they're tightly coupled (10 cross-reads) and form a single interaction pipeline.

### Relationship to ADR 0001 and ADR 0002

This ADR does not reopen or contradict ADR 0001 or ADR 0002. The Living Loop depth-aligned module split and its unidirectional dependency chain remain intact (ADR 0001). RuntimeState remains the single owner of mutation truth, brain event history, `dirty_state`, `state_revision`, `last_event`, and `recent_events` (ADR 0002). This ADR extends the module inventory around the existing RuntimeState and Living Loop seams without redefining their ownership boundaries.

### Backward compatibility

The Service Manager no longer owns the operator-facing runtime method surface. FastAPI routes call `RuntimeFacade`, and manager methods that remain are internal dependency callbacks for explicitly wired deep modules.

Compatibility imports such as `RuntimeControlMixin = RuntimeControl` may remain for tests and older imports, but `HECSNServiceManager` must not inherit from those aliases or recover behavior through an unbound mixin fallback.

### Migration strategy

This migration is implemented far enough that the Service Manager is no longer a legacy mixin inheritance surface and no longer has catch-all attribute routing. RuntimeController and AutonomyPlanner moved off manager-backed owner forwarders, the remaining owner-forwarded modules have been converted to explicit dependency objects or constructor callbacks, and broad manager facade delegates are now explicit named methods rather than import-time installed wrappers. Remaining compatibility work should continue by moving mixin-named implementation modules to domain-named modules where import compatibility allows.

1. Keep the ADR guard in `tests/test_adr_service_manager_composition.py` as the first regression target for future service-manager architecture work.
2. When touching transitional facade delegates, prefer replacing them with a concrete constructor dependency or an interface method on the true owner.
3. Keep service-level tests focused on public behavior. Module tests should use fakes/adapters at the same interfaces production uses.
4. Delete compatibility aliases only when import compatibility no longer requires them.

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
- Some mixin-named implementation modules remain while older tests/imports are retired
- Some cross-module calls that were previously `self.method()` become `other_module.method()`, which is a shallow syntax change with no semantic difference — but it adds a reference that must be managed

### Neutral

- The total line count stays roughly the same — this is a reorganization, not a reduction
- The Service Manager's public API surface doesn't change — FastAPI routes are untouched

## References

- PRD #50: Close ADR 0003 Service Manager deep module split
- Issue #61: Accept ADR 0003 and align domain docs
- ADR 0001: Split Living Loop monolith into depth-aligned modules
- ADR 0002: Runtime State owns mutation truth and brain event history
