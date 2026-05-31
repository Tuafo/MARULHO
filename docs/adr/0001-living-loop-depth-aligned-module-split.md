# ADR 0001: Split Living Loop monolith into depth-aligned modules

## Status

Accepted

## Context

The Living Loop module (`living_loop.py`) was a 3195-line monolith containing all runtime record types, policy scoring, replay planning, and the Operational Self-Model. It had no internal seams — every consumer (LivingStatusCore, ReplayRuntimeMixin, the runtime evidence reporter, ServiceManager, and the test suite) imported from a single file. This forced any change to any layer to reparse and re-validate the entire monolith. The file was difficult to navigate, impossible to review in a single pass, and resisted isolated testing of individual layers.

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

The original `living_loop.py` compatibility shim is deleted after consumer imports are migrated. Active code imports directly from the five owning modules, so the split is enforced by the package boundary instead of hidden behind an aggregator namespace.

### Unidirectional dependency constraint

Dependency direction is strictly unidirectional: **Helpers → Records → Policy → Replay → Self-Model**. No module may import from a higher layer. This constraint is enforced by convention and verified by the existing module docstrings, which explicitly state the dependency direction and the "never imports from" constraint.

Violations would create circular dependencies and destroy the acyclic structure that makes the split mechanically clean.

### Direct-import enforcement strategy

The original `living_loop.py` monolith was split into the owning modules above. Its temporary import bridge was removed once active consumers migrated to direct imports. This ensures:

- Active consumers name the module that owns the concept they use
- No private helper symbols are published through an aggregator `__all__`
- Import errors expose accidental dependence on the old monolith or shim immediately

Private symbols (`_coerce_feedback_telemetry`, `_policy_count`, etc.) remain importable only from their owning modules where cross-layer use is intentional.

## Rationale

### Why five modules (not two or three)?

A two-module split (e.g., "data" vs "logic") would have left one module still over 1500 lines and created circular dependencies between the logic halves. A three-module split (e.g., "records", "policy", "everything else") would have left the "everything else" module at ~1800 lines — still unreviewable in a single pass and conflating replay, self-model, and telemetry concerns.

The four depth-aligned modules plus helpers is the natural articulation point: each module corresponds to a naturally acyclic layer already present in the codebase, each stays under 1000 lines, and each has a clear single responsibility.

### Why direct imports (not a permanent shim)?

The temporary shim decoupled the initial split from consumer migration. Keeping it permanently would preserve the monolith's public gravity and make the depth stack look optional. Direct imports make the architectural boundary concrete: Records users import Records, policy users import Policy, replay users import Replay, and self-model users import Self-Model.

### Why a shared helpers module?

The 12 private helper functions (`_stable_id`, `_clean_text`, `_clamp01`, `_safe_ratio`, `_limited_unique_clean_text`, `_latest_text`, `_as_mapping`, `_enum_value`, `_provenance_value`, `_verification_status_from_payload`, `_safe_float`, `_coerce_world_model_lite`) are used by multiple depth layers. Duplicating them across modules would violate single source of truth. Making them public would pollute the package API. Placing them in a dedicated helpers module preserves privacy (underscore prefix) while giving each layer a single import source.

Helpers that are used by only one layer (e.g., `_policy_latency_pressure`, `_replay_rank_candidates`) remain co-located with their consumer module.

## Consequences

### Positive

- Each module is under 1000 lines and reviewable in a single pass
- Each layer can be tested in isolation with dedicated per-module test files
- Dependency direction is explicit and acyclic by construction
- Direct imports make ownership visible at each consumer
- Future changes to one layer do not require re-validating unrelated layers
- Each module docstring documents its depth layer and dependency constraints

### Negative

- Consumers must update imports when symbols move between depth modules
- Private helper imports remain internal and should be used only by the depth modules that need them
- The five-file module set is more files to navigate than the single monolith (offset by each file being focused and independently understandable)

### Neutral

- The former shim is intentionally absent; compatibility is not part of the active runtime surface
- The helpers module is a "Layer 0" foundation — it may grow if more cross-layer helpers are identified

## References

- PRD #1: Deepen Living Loop — split monolith into four depth-aligned modules
- Issue #2: Extract shared helpers module
- Issue #3: Extract runtime records module
- Issue #4: Extract policy scoring module
- Issue #5: Extract replay planning module
- Issue #6: Extract operational self-model and telemetry module
- Issue #7: Replace living_loop.py with a temporary shim, then delete it after direct import migration
