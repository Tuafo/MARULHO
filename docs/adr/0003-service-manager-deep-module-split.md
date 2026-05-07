# ADR 0003: Dissolve Service Manager God class into deep modules with real seams

## Status

Proposed

## Context

The Service Manager (`src/hecsn/service/manager.py`) is a God class composed of 21 mixin classes that share 50+ mutable state attributes via `self._*`. It is the single seam between the FastAPI layer and the runtime — but it has no internal seams. Every mixin reads and writes every other mixin's state. The test file (`tests/test_service_manager.py`) is 9,277 lines with 128 tests and 222 mock/patch calls, because testing any one mixin requires constructing the entire Service Manager with all its real dependencies.

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

Dissolve the mixin inheritance into 15 independent deep modules, each with its own interface, its own state, and constructor-injected dependencies. The Service Manager becomes a thin composition root that wires modules together and exposes the FastAPI facade.

### Module inventory

| Module | Absorbs | Lines | Owns (state) |
|---|---|---|---|
| RuntimeState | new | ~50 | `dirty_state`, `state_revision`, brain event log |
| DelayedConsequenceTracker | DelayedConsequenceMixin | 2644 | consequence records, cooled/retired/compacted/split/remerged totals |
| AutonomyPlanner | TerminusAutonomyMixin | 1771 | (reads shared state via interfaces) |
| BrainRuntime | BrainRuntimeMixin | 1148 | source runtimes, source utility, tick counters, ingestion/sensory thread state |
| InteractionPipeline | InteractionRuntimeMixin + RuntimeEvidenceMixin | ~2000 | query gap history, runtime episode traces |
| FeedbackApplier | RuntimeFeedbackMixin | 244 | (applies to targets, no persistent own state) |
| SourceFocusScorer | SourceFocusMixin | 373 | (scoring is stateless per call) |
| RuntimeController | RuntimeControlMixin + RuntimePrewarmMixin | ~1600 | brain thread, stop event, prewarm threads |
| StatusReadModel | StatusRuntimeMixin + LivingStatusMixin + SensoryPreviewMixin + shallow ReportingMixin | ~1000 | cached status/telemetry/terminus snapshots |
| ActionExecutor | ActionRuntimeMixin + ActionAssistMixin | ~750 | action history |
| CortexController | CortexRuntimeMixin | 350 | cortex query hint text/timestamp |
| RuntimePersistence | PersistenceMixin | 233 | trace history, metadata, brain event recording |
| RuntimeConfig | RuntimeConfigMixin | 513 | (stateless — normalization only) |
| RuntimeSources | RuntimeSourcesMixin | 390 | source runtime dataclasses |
| ReplayController | ReplayRuntimeMixin + ReplayDatasetBundleMixin | ~950 | replay sample history |

### Design constraints

1. **Constructor injection.** Each module declares its dependencies in its `__init__` signature. The Service Manager wires them. Tests provide fakes.
2. **Writer owns state.** The module that mutates a piece of state owns it. Other modules access it through the owner's read methods. Key ownership assignments:
   - `brain_source_utility` → BrainRuntime (mutated by DelayedConsequence via `brain_runtime.update_source_utility()`)
   - `brain_recent_query_gaps` → InteractionPipeline (read by Autonomy and SourceFocus via `interaction.recent_query_gaps()`)
   - `action_history` → ActionExecutor (read by 8+ modules via `action_executor.action_history()` and `action_executor.recent_relevant_actions()`)
3. **Shared RLock.** Each module receives the RLock as a constructor parameter. No module owns the lock. This avoids deadlock while preserving the current thread-safety model (brain thread + FastAPI handlers).
4. **Direct references between modules.** Modules hold references to each other (injected at construction). No event bus, no manager orchestration. The dependency graph is explicit in each module's constructor.
5. **RuntimeState for cross-cutting concerns.** A tiny module owns `dirty_state`, `state_revision`, and the brain event log. Modules call `runtime_state.mark_mutated()` and `runtime_state.record_event()` instead of depending on the Service Manager's internals.
6. **Shallow modules collapsed.** LivingStatusMixin, SensoryPreviewMixin, and the shallow half of ReportingMixin collapse into StatusReadModel. They don't earn their own modules.
7. **InteractionRuntime + RuntimeEvidence merged.** Every interaction (query/feed/respond) produces an episode trace — they're tightly coupled (10 cross-reads) and form a single interaction pipeline.

### Backward compatibility

The Service Manager retains its current public method signatures (`query()`, `feed()`, `respond()`, `acquire()`, `status()`, `terminus_status()`, etc.). Internally, each method delegates to the corresponding deep module. FastAPI routes require no changes.

### Migration strategy

1. Extract one module at a time, starting with the most independent (RuntimeConfig, RuntimeState, SourceFocusScorer).
2. For each extraction: create the module class, move methods and state, add constructor-injected dependencies, update the Service Manager to construct and delegate to the module.
3. Port tests: existing Service Manager tests that test the extracted module's behaviour move to a new per-module test file. Add interface-level tests that use module fakes.
4. Repeat until all 15 modules are extracted.
5. Delete the mixin files once all logic has moved.

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
- The Service Manager becomes a thin composition root (~200 lines) instead of a 17,500-line mixin soup
- State ownership is explicit: the module that mutates owns, others read through the interface
- Cross-cutting concerns (`mark_mutated`, `record_event`) have a dedicated home
- Shallow modules are collapsed — no more pass-throughs
- Future changes to one module don't require re-validating unrelated modules
- The 9,277-line test file can decompose into 15 per-module test files

### Negative

- Constructor signatures get larger — the BrainRuntime module, as the central hub, will have many dependencies
- Wiring code in the Service Manager's composition root must be maintained
- Migration is incremental — during the transition, some modules will be extracted while others are still mixins, creating a hybrid state
- Some cross-module calls that were previously `self.method()` become `other_module.method()`, which is a shallow syntax change with no semantic difference — but it adds a reference that must be managed

### Neutral

- The total line count stays roughly the same — this is a reorganization, not a reduction
- The Service Manager's public API surface doesn't change — FastAPI routes are untouched
