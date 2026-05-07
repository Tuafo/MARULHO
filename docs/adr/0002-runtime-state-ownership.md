# ADR 0002: Runtime State owns mutation truth and brain event history

## Status

Accepted

## Context

The Service Manager previously held runtime truth through shallow, implicit private fields: `dirty_state`, `state_revision`, `last_event`, and `recent_events`. Writers across checkpoint restore, status snapshots, event recording, and replay sampling reached into the same composition-root internals, so there was no single owner for mutation truth. That made the rules for "dirty", "clean", and revisioned state hard to reason about and easy to scatter across modules.

Replay sampling is the one intentional exception. Audit-only sampling must leave the runtime dirty without advancing the revision counter, so the replay safety contract can record the sample without pretending it was a normal mutation.

The broader Service Manager deep-module split is recorded separately in ADR 0003. This ADR captures the narrower ownership decision for runtime truth itself.

## Decision

Create a private `RuntimeState` module as the single owner of mutation truth and brain event history.

| Field | Owner | Notes |
|---|---|---|
| `dirty_state` | `RuntimeState` | Shared mutation flag |
| `state_revision` | `RuntimeState` | Revision counter for normal mutations and clean checkpoint restore |
| `last_event` | `RuntimeState` | Newest brain event payload |
| `recent_events` | `RuntimeState` | Bounded newest-first event history |

`RuntimeState` exposes the internal mutation API used by the rest of the service runtime:

- `mark_mutated()`
- `mark_clean()`
- `restore_clean()`
- `mark_dirty_without_revision()`
- `record_event()`
- `restore_event_history()`

Other service modules receive `RuntimeState` as a dependency and stop reading or writing those fields directly on `ServiceManager`.

### External payload compatibility

This ownership change is internal only. The external payload contract stays the same: `dirty_state`, `state_revision`, `last_event`, and `recent_events` remain the public field names in status, checkpoint, and tracing responses. JSON-safe normalization and defensive-copy behavior for event payloads also remain unchanged.

### Replay sampling exception

Replay sampling intentionally calls `mark_dirty_without_revision()` instead of `mark_mutated()`. That keeps audit-only replay samples dirty while preserving the current revision number. This is a deliberate exception, not an inconsistency, and it must remain documented and tested.

### Relationship to ADR 0001

This ADR does not reopen or contradict ADR 0001. The Living Loop split and its unidirectional dependency chain remain intact; Runtime State is a separate ownership seam in the service runtime layer.

## Rationale

### Why not keep runtime truth on `ServiceManager`?

The Service Manager is a composition root, not a domain owner. Leaving `dirty_state`, `state_revision`, `last_event`, and `recent_events` on the manager keeps mutation truth hidden behind a God object and forces unrelated modules to depend on private internals.

### Why keep event history with the mutation flag?

`last_event` and `recent_events` are not independent from the dirty/revision rules. Checkpoint restore, replay sampling, and status snapshots need to move those values together, so a single owner keeps the lifecycle atomic and testable.

### Why preserve dirty-without-revision replay semantics?

Replay sampling is audit-only. It must preserve the existing safety semantics by keeping the runtime dirty without creating a revision bump that would look like a normal mutation.

## Consequences

### Positive

- Runtime truth has a single explicit owner
- The public payload contract stays stable for API callers
- Event history restore and replay sampling semantics are easier to test
- Modules stop reaching into `ServiceManager` internals directly

### Negative

- More dependencies are injected into modules that need runtime truth
- The runtime-state boundary adds one more internal seam to understand

### Neutral

- The broader service-manager module split remains documented in ADR 0003
- This ADR records an ownership decision, not a public API change

## References

- PRD #27: Deepen Runtime State module
- Issue #32: Add ADR for Runtime State ownership
- ADR 0001: Split Living Loop monolith into depth-aligned modules
- ADR 0003: Dissolve Service Manager God class into deep modules with real seams
