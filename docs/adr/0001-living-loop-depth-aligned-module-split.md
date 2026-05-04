# ADR 0001: Split Living Loop monolith into depth-aligned modules

## Status

Accepted

## Context

The Living Loop module (`living_loop.py`) was a 3195-line monolith containing all runtime record types, policy scoring, replay planning, and the Operational Self-Model. It had no internal seams — every consumer (LivingStatusMixin, ReplayRuntimeMixin, RuntimeEvidenceMixin, ServiceManager, and the test suite) imported from a single file. This forced any change to any layer to reparse and re-validate the entire monolith. The file was difficult to navigate, impossible to review in a single pass, and resisted isolated testing of individual layers.

Analysis of the internal call graph revealed a naturally acyclic four-layer dependency stack within the monolith:

1. **Runtime Records** — enums and frozen dataclasses used by all other layers; no inward dependencies
2. **Policy Scoring** — policy actuator decision logic; imports only Records
3. **Replay Planning** — replay candidate ranking and plan building; imports Policy and Records
4. **Operational Self-Model** — self-model surface methods and telemetry; imports all three above

Additionally, 12 private helper functions were called from more than one layer, requiring a shared source of truth.

## Decision

We split the Living Loop monolith into five modules that mirror the existing internal dependency stack:

| Module | File | Depth Layer | Depends On |
|---|---|---|---|
| Shared Helpers | `living_loop_helpers.py` | Foundation | External packages only |
| Runtime Records | `living_loop_records.py` | Layer A | Helpers |
| Policy Scoring | `living_loop_policy.py` | Layer B | Helpers, Records |
| Replay Planning | `living_loop_replay.py` | Layer C | Helpers, Records, Policy |
| Operational Self-Model | `living_loop_self_model.py` | Layer D | Helpers, Records, Policy, Replay |

The original `living_loop.py` becomes a backward-compatible re-export shim that imports and re-exports every public (and cross-module private) symbol from the five new modules.

### Unidirectional dependency constraint

Dependency direction is strictly unidirectional: **Helpers → Records → Policy → Replay → Self-Model**. No module may import from a higher layer. This constraint is enforced by convention and verified by the existing module docstrings, which explicitly state the dependency direction and the "never imports from" constraint.

Violations would create circular dependencies and destroy the acyclic structure that makes the split mechanically clean.

### Backward-compatible re-export shim strategy

The original `living_loop.py` is replaced with a thin shim containing only `import` statements, re-exports, and an `__all__` list. No implementation code remains. This ensures:

- All five existing consumer import sites continue to work without modification
- No silent breakage from removed symbols (every previously available symbol is re-exported)
- The shim can be validated via AST analysis to confirm it contains no implementation code

Private symbols (`_coerce_feedback_telemetry`, `_policy_count`, etc.) are included in `__all__` and re-exported because they are used across module boundaries and were previously available from the monolith.

## Rationale

### Why five modules (not two or three)?

A two-module split (e.g., "data" vs "logic") would have left one module still over 1500 lines and created circular dependencies between the logic halves. A three-module split (e.g., "records", "policy", "everything else") would have left the "everything else" module at ~1800 lines — still unreviewable in a single pass and conflating replay, self-model, and telemetry concerns.

The four depth-aligned modules plus helpers is the natural articulation point: each module corresponds to a naturally acyclic layer already present in the codebase, each stays under 1000 lines, and each has a clear single responsibility.

### Why a re-export shim (not a migration)?

Migrating consumer imports to the new modules in the same PR as the split would conflate two changes: the structural reorganisation and the import site updates. The re-export shim decouples them, allowing the split to be verified as behaviour-preserving before any consumer changes. Consumer import migration is deferred to a follow-up.

### Why a shared helpers module?

The 12 private helper functions (`_stable_id`, `_clean_text`, `_clamp01`, `_safe_ratio`, `_limited_unique_clean_text`, `_latest_text`, `_as_mapping`, `_enum_value`, `_provenance_value`, `_verification_status_from_payload`, `_safe_float`, `_coerce_world_model_lite`) are used by multiple depth layers. Duplicating them across modules would violate single source of truth. Making them public would pollute the package API. Placing them in a dedicated helpers module preserves privacy (underscore prefix) while giving each layer a single import source.

Helpers that are used by only one layer (e.g., `_policy_latency_pressure`, `_replay_rank_candidates`) remain co-located with their consumer module.

## Consequences

### Positive

- Each module is under 1000 lines and reviewable in a single pass
- Each layer can be tested in isolation with dedicated per-module test files
- Dependency direction is explicit and acyclic by construction
- The re-export shim ensures zero consumer breakage
- Future changes to one layer do not require re-validating unrelated layers
- Each module docstring documents its depth layer and dependency constraints

### Negative

- The re-export shim adds one level of indirection for existing consumers until they are migrated
- Private symbols are visible in `__all__` (necessary for cross-module re-export correctness)
- The five-file module set is more files to navigate than the single monolith (offset by each file being focused and independently understandable)

### Neutral

- Consumer import migration to direct module imports is a follow-up task, not part of this decision
- The helpers module is a "Layer 0" foundation — it may grow if more cross-layer helpers are identified

## References

- PRD #1: Deepen Living Loop — split monolith into four depth-aligned modules
- Issue #2: Extract shared helpers module
- Issue #3: Extract runtime records module
- Issue #4: Extract policy scoring module
- Issue #5: Extract replay planning module
- Issue #6: Extract operational self-model and telemetry module
- Issue #7: Replace living_loop.py with backward-compatible re-export shim
