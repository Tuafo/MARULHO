"""Tests for ADR 0003 Service Manager composition root and architecture guards (Issue #60).

These tests verify that HECSNServiceManager is a thin composition root per ADR 0003:
  - No legacy mixin classes in the manager inheritance
  - ADR-owned state is not directly defined on the manager
  - RuntimeState private fields do not leak outside RuntimeState
  - The manager constructs and wires deep modules explicitly
"""

from __future__ import annotations

import ast
import inspect
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVICE_SRC_ROOT = _REPO_ROOT / "src" / "hecsn" / "service"
_ADR_PATH = _REPO_ROOT / "docs" / "adr" / "0003-service-manager-deep-module-split.md"

# Legacy mixin classes that must not appear in HECSNServiceManager.__bases__
_LEGACY_MIXIN_CLASSES = frozenset({
    "ReplayDatasetBundleMixin",
    "RuntimeEvidenceMixin",
    "DelayedConsequenceMixin",
    "DelayedConsequenceTracker",
    "ServiceReportingMixin",
    "ReplayController",
    "InteractionRuntimeMixin",
    "LivingStatusMixin",
    "RuntimeConfigMixin",
    "RuntimePrewarmMixin",
    "RuntimeSourcesMixin",
    "SensoryRuntimeMixin",
    "SourceFocusMixin",
    "StatusRuntimeMixin",
    "SensoryPreviewMixin",
    "TerminusAutonomyMixin",
    "RuntimePersistence",
    "CortexController",
    "ManagerBoundModule",
})

# State fields that ADR 0003 assigns to owning modules, not the manager.
# DelayedConsequenceTracker owns consequence records and totals.
_ADR_DELAYED_CONSEQUENCE_FIELDS = frozenset({
    "_delayed_consequence_records",
    "_delayed_consequence_cooled_total",
    "_delayed_consequence_retired_total",
    "_delayed_consequence_compacted_total",
    "_delayed_consequence_split_total",
    "_delayed_consequence_remerged_total",
})

# BrainRuntime owns source runtimes, source utility, tick counters, etc.
_ADR_BRAIN_RUNTIME_FIELDS = frozenset({
    "_brain_source_runtimes",
    "_sensory_source_runtimes",
    "_brain_source_index",
    "_sensory_source_index",
    "_brain_tick_count",
    "_brain_background_tokens",
    "_brain_autonomy_tokens",
    "_brain_source_utility",
    "_brain_last_error",
    "_brain_last_acquisition_summary",
    "_brain_last_acquisition_token_count",
    "_brain_last_tick_completed_at",
    "_brain_last_tick_duration_ms",
    "_brain_last_tick_token_delta",
    "_brain_last_work_at",
    "_brain_stream_epoch",
    "_sensory_stream_epoch",
})

# SensoryRuntime owns sensory episode state
_ADR_SENSORY_RUNTIME_FIELDS = frozenset({
    "_last_real_sensory_episode_time",
    "_last_real_sensory_episode_token_count",
    "_real_sensory_last_error",
    "_last_sensory_focus_terms",
    "_sensory_preview_history",
    "_real_sensory_episodes_completed",
    "_real_visual_accepted",
    "_real_audio_accepted",
    "_brain_skip_next_autonomy_for_grounded_query",
})

# InteractionPipeline owns query gaps and episode traces (aliases on manager are ok
# but the source of truth must be the pipeline)
_ADR_INTERACTION_PIPELINE_FIELDS = frozenset({
    "_brain_recent_query_gaps",
    "_runtime_episode_traces",
})

# All ADR-owned fields that must not be newly defined on the manager __init__
_ADR_OWNED_FIELDS = (
    _ADR_DELAYED_CONSEQUENCE_FIELDS
    | _ADR_BRAIN_RUNTIME_FIELDS
    | _ADR_SENSORY_RUNTIME_FIELDS
    | _ADR_INTERACTION_PIPELINE_FIELDS
)

# RuntimeState private fields that must not appear outside RuntimeState
_RUNTIME_STATE_PRIVATE_FIELDS = frozenset({
    "_dirty_state",
    "_state_revision",
    "_brain_last_event",
    "_brain_event_history",
})


def _parse_manager_init_self_assigns() -> set[str]:
    """Parse manager.py to find all self._X = assignments in __init__."""
    manager_path = _SERVICE_SRC_ROOT / "manager.py"
    tree = ast.parse(manager_path.read_text(encoding="utf-8"), filename=str(manager_path))

    assigned_fields: set[str] = set()
    in_init = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            in_init = True
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.Assign)
                    and len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Attribute)
                    and isinstance(child.targets[0].value, ast.Name)
                    and child.targets[0].value.id == "self"
                ):
                    assigned_fields.add(child.targets[0].attr)
            in_init = False
            break

    return assigned_fields


class TestADR0003ManagerCompositionRoot(unittest.TestCase):
    """HECSNServiceManager must be a thin composition root per ADR 0003."""

    def test_manager_has_no_legacy_mixin_bases(self) -> None:
        """No legacy mixin class may appear in HECSNServiceManager.__bases__."""
        from hecsn.service.manager import HECSNServiceManager

        base_names = {cls.__name__ for cls in HECSNServiceManager.__bases__}
        legacy_in_bases = base_names & _LEGACY_MIXIN_CLASSES
        self.assertFalse(
            legacy_in_bases,
            f"Legacy mixin classes still in HECSNServiceManager.__bases__: {sorted(legacy_in_bases)}",
        )

    def test_manager_has_no_legacy_mixin_in_mro(self) -> None:
        """No legacy mixin class may appear in the full MRO (except object)."""
        from hecsn.service.manager import HECSNServiceManager

        mro_names = {cls.__name__ for cls in HECSNServiceManager.__mro__}
        legacy_in_mro = mro_names & _LEGACY_MIXIN_CLASSES
        self.assertFalse(
            legacy_in_mro,
            f"Legacy mixin classes still in HECSNServiceManager.__mro__: {sorted(legacy_in_mro)}",
        )

    def test_manager_constructs_deep_modules(self) -> None:
        """Manager must construct key ADR 0003 deep modules explicitly."""
        from hecsn.service.manager import HECSNServiceManager

        # Check for the key deep module attributes that the manager wires
        init_source = inspect.getsource(HECSNServiceManager.__init__)
        required_constructions = [
            "_runtime_state",
            "_brain_runtime",
            "_runtime_control",
            "_cortex_controller",
            "_interaction_pipeline",
            "_action_executor",
            "_feedback_applier",
            "_status_read_model",
            "_runtime_persistence",
            "_autonomy_planner",
            "_replay_controller",
        ]
        for module_attr in required_constructions:
            self.assertIn(
                module_attr,
                init_source,
                f"Manager __init__ must construct deep module attribute '{module_attr}'",
            )

    def test_manager_init_does_not_define_adr_owned_state(self) -> None:
        """ADR-owned state fields must not be newly assigned in manager __init__."""
        init_fields = _parse_manager_init_self_assigns()
        violations = init_fields & _ADR_OWNED_FIELDS
        self.assertFalse(
            violations,
            f"ADR-owned state fields still defined on manager __init__: {sorted(violations)}",
        )

    def test_runtime_state_private_fields_not_outside_runtime_state(self) -> None:
        """RuntimeState private fields must not appear outside runtime_state.py."""
        violations: list[str] = []
        for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
            if path.name == "runtime_state.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr in _RUNTIME_STATE_PRIVATE_FIELDS:
                    if isinstance(node.value, ast.Name) and node.value.id == "self":
                        violations.append(
                            f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{node.col_offset + 1} self.{node.attr}"
                        )
        self.assertFalse(
            violations,
            "RuntimeState private fields found outside runtime_state.py:\n"
            + "\n".join(violations),
        )


class TestADR0003DocumentStatus(unittest.TestCase):
    """ADR 0003 must exist and describe the composition root decision."""

    def test_adr_file_exists(self) -> None:
        self.assertTrue(_ADR_PATH.exists(), f"ADR file not found at {_ADR_PATH}")

    def test_adr_has_status_proposed_or_accepted(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("## Status", text)
        # After all module guards pass, this should be Accepted
        self.assertRegex(text, r"## Status\s*\n\s*Accepted")

    def test_adr_mentions_composition_root(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("composition root", text.lower())

    def test_adr_mentions_deep_modules(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        for module_name in ("RuntimeState", "InteractionPipeline", "ActionExecutor", "CortexController"):
            self.assertIn(module_name, text)


if __name__ == "__main__":
    unittest.main()
