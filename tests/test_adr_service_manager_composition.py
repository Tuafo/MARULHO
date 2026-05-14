"""Tests for ADR 0003 Service Manager composition root and architecture guards.

Issue #60: Manager composition root architecture guards.
Issue #61: Accept ADR 0003 and align domain docs.

These tests verify that HECSNServiceManager is a thin composition root per ADR 0003:

- No legacy mixin classes in the manager inheritance
- ADR-owned state is not directly defined on the manager
- RuntimeState private fields do not leak outside RuntimeState
- The manager constructs and wires deep modules explicitly
- ADR 0003 is Accepted and does not contradict ADR 0001 or ADR 0002
- ADR 0003 has a References section linking to PRD and prior ADRs
- CONTEXT.md matches the final module inventory and ownership language
"""

from __future__ import annotations

import ast
import inspect
import re
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SERVICE_SRC_ROOT = _REPO_ROOT / "src" / "hecsn" / "service"
_ADR_PATH = _REPO_ROOT / "docs" / "adr" / "0003-service-manager-deep-module-split.md"
_ADR_0004_PATH = _REPO_ROOT / "docs" / "adr" / "0004-runtime-facade-manager-max-removal.md"
_CONTEXT_PATH = _REPO_ROOT / "CONTEXT.md"

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

# Deep module names from ADR 0003 that CONTEXT.md must list
_ADR_0003_DEEP_MODULES = (
    "Runtime State",
    "Delayed Consequence Tracker",
    "Autonomy Planner",
    "Brain Runtime",
    "Interaction Pipeline",
    "Feedback Applier",
    "Source Focus Scorer",
    "Runtime Controller",
    "Status Read Model",
    "Action Executor",
    "Cortex Controller",
    "Runtime Persistence",
    "Runtime Config",
    "Runtime Sources",
    "Replay Controller",
)


def _parse_manager_init_self_assigns() -> set[str]:
    """Parse manager.py to find all self._X = assignments in __init__."""
    manager_path = _SERVICE_SRC_ROOT / "manager.py"
    tree = ast.parse(manager_path.read_text(encoding="utf-8"), filename=str(manager_path))

    assigned_fields: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "HECSNServiceManager":
            continue
        for class_child in node.body:
            if not isinstance(class_child, ast.FunctionDef) or class_child.name != "__init__":
                continue
            for child in ast.walk(class_child):
                if (
                    isinstance(child, ast.Assign)
                    and len(child.targets) == 1
                    and isinstance(child.targets[0], ast.Attribute)
                    and isinstance(child.targets[0].value, ast.Name)
                    and child.targets[0].value.id == "self"
                ):
                    assigned_fields.add(child.targets[0].attr)
            return assigned_fields

    return assigned_fields


def _all_composition_root_guard_conditions_pass() -> bool:
    """Return True when all ADR 0003 composition root guard conditions pass.

    This is the programmatic check that the implementation matches the
    architectural decision: no legacy mixins in MRO, no ADR-owned state
    on the manager __init__, and RuntimeState fields stay in RuntimeState.
    """
    from hecsn.service.manager import HECSNServiceManager

    # Guard 1: No legacy mixin in MRO
    mro_names = {cls.__name__ for cls in HECSNServiceManager.__mro__}
    if mro_names & _LEGACY_MIXIN_CLASSES:
        return False

    # Guard 2: No ADR-owned state on manager __init__
    init_fields = _parse_manager_init_self_assigns()
    if init_fields & _ADR_OWNED_FIELDS:
        return False

    # Guard 3: RuntimeState private fields stay in runtime_state.py
    for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
        if path.name == "runtime_state.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _RUNTIME_STATE_PRIVATE_FIELDS:
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    return False

    return True


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

    def test_manager_has_no_getattr_router(self) -> None:
        """Manager must not use catch-all attribute routing for module behavior."""
        from hecsn.service.manager import HECSNServiceManager

        self.assertNotIn("__getattr__", HECSNServiceManager.__dict__)
        self.assertNotIn("__setattr__", HECSNServiceManager.__dict__)
        manager_text = (_SERVICE_SRC_ROOT / "manager.py").read_text(encoding="utf-8")
        self.assertNotIn("_unbound_mixin_fallback", manager_text)

    def test_no_manager_bound_module_base_remains(self) -> None:
        """Deep modules must declare owner dependencies explicitly."""
        violations: list[str] = []
        for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "ManagerBoundModule":
                    violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
                if (
                    isinstance(node, ast.Name)
                    and node.id == "ManagerBoundModule"
                    and path.name != "manager.py"
                ):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}")
        self.assertFalse(
            violations,
            "ManagerBoundModule remains in service modules:\n" + "\n".join(violations),
        )

    def test_owner_forwarder_helper_removed(self) -> None:
        """The transition owner-forwarder helper must not remain in service code."""
        helper_path = _SERVICE_SRC_ROOT / "manager_bound_module.py"
        self.assertFalse(helper_path.exists(), "manager_bound_module.py must be removed")

        violations: list[str] = []
        for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for marker in ("ExplicitOwnerModule", "install_owner_forwarders", "manager_bound_module"):
                if marker in text:
                    violations.append(f"{path.relative_to(_REPO_ROOT)} contains {marker}")
        self.assertFalse(
            violations,
            "Owner-forwarder transition code remains:\n" + "\n".join(violations),
        )

    def test_dynamic_delegate_installers_removed(self) -> None:
        """Manager facade methods must be explicit, not installed by import-time loops."""
        manager_text = (_SERVICE_SRC_ROOT / "manager.py").read_text(encoding="utf-8")
        forbidden = (
            "_install_module_delegate",
            "_install_mixin_delegate",
            "for _module_attr, _module_cls in",
            "for _mixin_cls in",
        )
        violations = [marker for marker in forbidden if marker in manager_text]
        self.assertFalse(
            violations,
            "Dynamic manager delegate installation remains: " + ", ".join(violations),
        )

    def test_no_module_level_mixin_aliases_remain(self) -> None:
        """Compatibility aliases like FooMixin = Foo must be retired."""
        violations: list[str] = []
        for path in sorted(_SERVICE_SRC_ROOT.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"(?m)^\s*[A-Za-z_][A-Za-z0-9_]*Mixin\s*=", text):
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{text[:match.start()].count(chr(10)) + 1}")
        self.assertFalse(
            violations,
            "Module-level mixin compatibility aliases remain:\n" + "\n".join(violations),
        )

    def test_brain_runtime_has_explicit_dependencies(self) -> None:
        """BrainRuntime must not recover owner access through transition helpers."""
        brain_runtime_path = _SERVICE_SRC_ROOT / "brain_runtime.py"
        text = brain_runtime_path.read_text(encoding="utf-8")
        forbidden = (
            "ExplicitOwnerModule",
            "install_owner_forwarders(BrainRuntime",
            "_bound_module(",
            '"_manager"',
            "'_manager'",
        )
        violations = [value for value in forbidden if value in text]
        self.assertFalse(
            violations,
            "BrainRuntime still uses owner-backed transition helpers: "
            + ", ".join(violations),
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

    def test_manager_constructs_runtime_facade(self) -> None:
        """Manager must expose the ADR 0004 runtime facade instead of being the runtime interface."""
        from hecsn.service.manager import HECSNServiceManager
        from hecsn.service.runtime_facade import RuntimeFacade

        init_source = inspect.getsource(HECSNServiceManager.__init__)
        self.assertIn("_runtime_facade", init_source)
        self.assertIsInstance(HECSNServiceManager.runtime_facade, property)
        self.assertTrue(issubclass(RuntimeFacade, object))

    def test_manager_does_not_expose_operator_runtime_methods(self) -> None:
        """Operator-facing runtime methods belong on RuntimeFacade, not the manager class."""
        from hecsn.service.manager import HECSNServiceManager

        removed_runtime_methods = (
            "status",
            "terminus_status",
            "sensory_previews",
            "architecture_summary",
            "telemetry_snapshot",
            "living_loop_status",
            "policy_actuator_status",
            "checkpoint_list",
            "recent_traces",
            "save_checkpoint",
            "restore_checkpoint",
            "feed",
            "query",
            "respond",
            "acquire",
            "configure_terminus",
            "start_terminus",
            "stop_terminus",
            "quick_start_terminus",
            "terminus_tick",
            "replay_plan_status",
            "replay_sample",
            "replay_sample_history",
            "cortex_ask",
            "cortex_sleep",
            "cortex_thoughts",
            "cortex_snapshot",
            "action_history",
            "execute_digital_action",
            "record_runtime_feedback",
            "export_runtime_trace_examples",
            "replay_dataset_preview",
            "replay_dataset_candidates",
            "replay_dataset_history",
            "replay_dataset_bundle",
            "run_grounding_probe",
            "quick_start_presets",
        )
        leaked = [name for name in removed_runtime_methods if hasattr(HECSNServiceManager, name)]
        self.assertFalse(leaked, "Manager still exposes runtime methods: " + ", ".join(leaked))

    def test_fastapi_routes_use_runtime_facade(self) -> None:
        """FastAPI must call RuntimeFacade for runtime behaviour, not manager pass-through methods."""
        api_text = (_SERVICE_SRC_ROOT / "api.py").read_text(encoding="utf-8")
        self.assertIn("runtime = manager.runtime_facade", api_text)
        forbidden_calls = (
            "manager.status(",
            "manager.feed(",
            "manager.query(",
            "manager.respond(",
            "manager.terminus_status(",
            "manager.replay_sample(",
            "manager.configure_terminus(",
            "manager.terminus_tick(",
            "manager.cortex_snapshot(",
            "manager.execute_digital_action(",
        )
        violations = [call for call in forbidden_calls if call in api_text]
        self.assertFalse(violations, "FastAPI still calls manager runtime methods: " + ", ".join(violations))

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

    def test_adr_has_status_accepted(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("## Status", text)
        self.assertRegex(text, r"## Status\s*\n\s*Accepted")

    def test_adr_mentions_composition_root(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        self.assertIn("composition root", text.lower())

    def test_adr_mentions_deep_modules(self) -> None:
        text = _ADR_PATH.read_text(encoding="utf-8")
        for module_name in ("RuntimeState", "InteractionPipeline", "ActionExecutor", "CortexController"):
            self.assertIn(module_name, text)

    def test_adr_status_must_not_be_proposed_when_guards_pass(self) -> None:
        """ADR 0003 must not remain Proposed after all guard conditions pass.

        Architecture guards verify that the manager is a thin composition root
        with no legacy mixins in MRO and no ADR-owned state on manager __init__.
        When those conditions all pass, the ADR must reflect that the
        implementation is complete by being Accepted, not Proposed.
        """
        guards_pass = _all_composition_root_guard_conditions_pass()
        text = _ADR_PATH.read_text(encoding="utf-8")
        is_proposed = bool(re.search(r"## Status\s*\n\s*Proposed", text))
        if guards_pass:
            self.assertFalse(
                is_proposed,
                "ADR 0003 must not remain Proposed when all composition root "
                "guard conditions pass (no legacy mixins in MRO, no ADR-owned "
                "state on manager __init__, RuntimeState fields isolated). "
                "Change ADR 0003 status to Accepted.",
            )


class TestADR0003ConsistencyWithPriorADRs(unittest.TestCase):
    """ADR 0003 must not contradict ADR 0001 or ADR 0002."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.adr3_text = _ADR_PATH.read_text(encoding="utf-8")

    def test_adr3_has_references_section(self) -> None:
        """ADR 0003 must have a ## References section."""
        self.assertIn("## References", self.adr3_text)

    def test_adr3_references_prior_adrs_and_prd(self) -> None:
        """ADR 0003 must reference ADR 0001, ADR 0002, and PRD #50.

        ADR 0002 records that RuntimeState owns mutation truth. ADR 0003
        must acknowledge this ownership and confirm it is preserved.
        ADR 0003 must also reference ADR 0001 rather than redefine its layers.
        """
        for reference in ("ADR 0001", "ADR 0002", "PRD #50"):
            self.assertIn(reference, self.adr3_text)
        self.assertIn("RuntimeState", self.adr3_text)

    def test_adr3_preserves_runtime_state_ownership_from_adr2(self) -> None:
        """ADR 0003 must not assign RuntimeState fields to other modules."""
        for field in ("dirty_state", "state_revision", "last_event", "recent_events"):
            self.assertIn(field, self.adr3_text)


class TestADR0004RuntimeFacade(unittest.TestCase):
    """ADR 0004 records the runtime facade max-removal decision."""

    def test_adr4_file_exists_and_is_accepted(self) -> None:
        self.assertTrue(_ADR_0004_PATH.exists(), f"ADR file not found at {_ADR_0004_PATH}")
        text = _ADR_0004_PATH.read_text(encoding="utf-8")
        self.assertRegex(text, r"## Status\s*\n\s*Accepted")

    def test_adr4_names_runtime_facade_and_composition_root(self) -> None:
        text = _ADR_0004_PATH.read_text(encoding="utf-8")
        self.assertIn("RuntimeFacade", text)
        self.assertIn("composition root", text.lower())
        self.assertIn("operator-facing runtime interface", text)


class TestContextMdADR0003Alignment(unittest.TestCase):
    """CONTEXT.md must match the ADR 0003 Service Manager module inventory."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.context_text = _CONTEXT_PATH.read_text(encoding="utf-8")

    def test_context_mentions_adr_0003(self) -> None:
        """CONTEXT.md must reference ADR 0003."""
        self.assertIn("ADR 0003", self.context_text)

    def test_context_mentions_adr_0004_runtime_facade(self) -> None:
        """CONTEXT.md must describe the ADR 0004 Runtime Facade."""
        self.assertIn("ADR 0004", self.context_text)
        self.assertIn("**Runtime Facade**", self.context_text)

    def test_context_lists_all_15_deep_modules(self) -> None:
        """CONTEXT.md must list all 15 deep modules from ADR 0003."""
        for module_name in _ADR_0003_DEEP_MODULES:
            self.assertIn(
                module_name,
                self.context_text,
                f"CONTEXT.md missing ADR 0003 deep module: {module_name}",
            )

    def test_context_describes_service_manager_as_composition_root(self) -> None:
        """CONTEXT.md must describe Service Manager as a composition root."""
        self.assertIn("composition root", self.context_text.lower())

    def test_context_says_manager_owns_no_business_logic(self) -> None:
        """CONTEXT.md must state the manager owns no business logic."""
        # The Service Manager entry must say it wires/exposes but owns no logic
        self.assertIn("owns no business logic", self.context_text.lower())

    def test_context_runtime_state_description_matches_adr2(self) -> None:
        """CONTEXT.md Runtime State description must align with ADR 0002."""
        self.assertIn("dirty_state", self.context_text)
        self.assertIn("state_revision", self.context_text)

    def test_context_mentions_dirty_without_revision_for_replay(self) -> None:
        """CONTEXT.md must document the replay dirty-without-revision path."""
        self.assertIn("dirty-without-revision", self.context_text)

    def test_context_does_not_use_legacy_mixin_language(self) -> None:
        """CONTEXT.md Service Manager entry must not use legacy mixin language.

        The module list should describe deep modules, not mixin inheritance.
        """
        # Find the Service Manager section
        sm_idx = self.context_text.find("**Service Manager**")
        if sm_idx == -1:
            self.fail("CONTEXT.md missing Service Manager entry")
        # Get text from Service Manager to the next top-level entry
        sm_section = self.context_text[sm_idx:]
        # Should not describe inheritance-based architecture
        self.assertNotIn(
            "inherits from",
            sm_section.lower(),
        )
        self.assertNotIn(
            "mixin inheritance",
            sm_section.lower(),
        )


if __name__ == "__main__":
    unittest.main()
