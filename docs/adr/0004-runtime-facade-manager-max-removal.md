# ADR 0004: Runtime Facade owns the operator-facing runtime interface

## Status

Accepted

## Context

ADR 0003 dissolved the Service Manager God class into deep modules, but the
manager still exposed a broad pass-through surface. FastAPI routes, runners,
and tests could call `HECSNServiceManager` as though it still owned runtime
behaviour. That made the manager interface shallow: many methods only forwarded
to the module that actually owned the behaviour.

The remaining manager methods fall into two groups:

1. **Operator-facing runtime calls** such as status, feed/query/respond,
   checkpointing, replay, retired-runtime status, action, feedback, and Terminus control.
2. **Internal dependency callbacks** used by existing deep modules while they
   are still being converted away from manager-shaped dependencies.

Deleting both groups at once would conflate HTTP contract preservation,
runner migration, and deep-module dependency cleanup.

## Decision

Introduce `RuntimeFacade` as the single operator-facing runtime interface.

- `HECSNServiceManager` remains the composition root that constructs modules,
  wires dependencies, owns lifecycle cleanup, and exposes `runtime_facade`.
- FastAPI routes and export runners call `RuntimeFacade`, not manager runtime
  pass-through methods.
- `RuntimeFacade` delegates to the owning deep module where one exists.
- Replay-dataset and trace-export behaviour route through domain-named helper
  modules. Operator interaction and acquisition now route through
  `OperatorInteractionRuntime`, not the former interaction mixin module.
- Internal manager callback hooks may remain only when they are consumed by
  explicit constructor dependencies or state-property bridges inside the current
  deep modules. They are not the operator-facing runtime interface, and they are
  removed once the consuming module can call its real owner directly. Runtime
  stream construction is no longer one of those hooks; it belongs to
  `RuntimeSources`.

## Consequences

### Positive

- The operator-facing interface has a real module seam.
- FastAPI no longer depends on the manager's broad runtime method surface.
- The Service Manager is closer to a pure composition root.
- Future cleanup can target internal callback dependencies without changing
  the HTTP contract.

### Negative

- `RuntimeFacade` still knows about trace export and replay dataset packaging
  helpers until those helpers receive narrower constructor dependencies.
- Some internal manager callback hooks remain until RuntimeControl,
  BrainRuntime, the retired runtime path state holder, RuntimePersistence, and related modules move
  to narrower dependency objects.

### Neutral

- External HTTP routes and response schemas do not change.
- Existing checkpoints and runtime payload field names do not change.

## References

- ADR 0003: Dissolve Service Manager God class into deep modules with real seams
