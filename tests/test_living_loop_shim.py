"""Integration test for the living_loop.py backward-compatible re-export shim.

Verifies that:
- living_loop.py contains only import/re-export statements (no implementation code)
- Every public symbol previously available from living_loop resolves correctly
- All five consumer import sites work without modification
- The re-exported symbols are the exact same objects as in their source modules
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

# ── Source modules ──────────────────────────────────────────────────────────
from hecsn.service.living_loop_helpers import (
    _as_mapping,
    _clean_text,
    _clamp01,
    _coerce_world_model_lite,
    _enum_value,
    _latest_text,
    _limited_unique_clean_text,
    _provenance_value,
    _safe_float,
    _safe_ratio,
    _stable_id,
    _verification_status_from_payload,
)
from hecsn.service.living_loop_records import (
    ActionExecutionRecord,
    ActionExecutionStatus,
    ActionVerificationRecord,
    ConsolidationRecord,
    ConsolidationStatus,
    PredictionRecord,
    PredictionStatus,
    ProvenanceState,
    RuntimeEpisodeTrace,
    SkillMemoryRecord,
    VerificationStatus,
)
from hecsn.service.living_loop_policy import (
    POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS,
    POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS,
    POLICY_ACTUATOR_SCHEMA_VERSION,
    PolicyActuatorRecommendation,
    PolicyScore,
    WorldModelLiteSummary,
    build_policy_actuator_status,
    _coerce_feedback_telemetry,
    _policy_count,
    _policy_float,
    _policy_mapping,
)
from hecsn.service.living_loop_replay import (
    REPLAY_PLAN_DEFAULT_LIMIT,
    REPLAY_PLAN_MAX_LIMIT,
    REPLAY_PLAN_PRIORITY_RULES_VERSION,
    REPLAY_PLAN_PRIORITY_WEIGHTS,
    REPLAY_PLAN_SCHEMA_VERSION,
    REPLAY_REASON_PRECEDENCE,
    REPLAY_SAMPLE_SAFETY_BOUNDARIES,
    ReplayCandidate,
    ReplayPlan,
    build_replay_plan,
    replay_candidate_safety_flags,
    _coerce_replay_sample_summary,
    _default_replay_sample_safety_flags,
)
from hecsn.service.living_loop_self_model import (
    OperationalSelfModel,
    build_runtime_benchmark_telemetry,
    _endpoint_bucket_name,
    _endpoint_latency_empty,
    _extract_cache_summary,
    _extract_nim_summary,
    _latency_summary,
    _memory_counter_summary,
)

import hecsn.service.living_loop as ll

# ── Path to living_loop.py for AST inspection ───────────────────────────────
_LIVING_LOOP_PATH = Path(__file__).resolve().parent.parent / "src" / "hecsn" / "service" / "living_loop.py"


class TestShimIsReExportOnly(unittest.TestCase):
    """Verify that living_loop.py contains only import/re-export statements."""

    def test_no_function_or_class_definitions(self) -> None:
        """The shim must not define any functions, classes, or non-__all__ assignments."""
        source = _LIVING_LOOP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.fail(
                    f"living_loop.py contains a {node.__class__.__name__} "
                    f"({node.name!r}) — shim should be import/re-export only"
                )
            if isinstance(node, ast.Assign):
                # __all__ is the only allowed assignment in a re-export shim
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if names != ["__all__"]:
                    self.fail(
                        f"living_loop.py contains a non-__all__ assignment "
                        f"({names!r}) — shim should be import/re-export only"
                    )

    def test_only_imports_and_future(self) -> None:
        """Every top-level statement must be an import, __all__, or docstring."""
        source = _LIVING_LOOP_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.ImportFrom, ast.Import)):
                continue
            if isinstance(node, ast.Expr) and isinstance(
                node.value, ast.Constant
            ):
                # module-level string expressions (docstrings) are ok
                continue
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if names == ["__all__"]:
                    continue
            self.fail(
                f"Unexpected top-level {node.__class__.__name__} in living_loop.py"
            )


class TestPublicSymbolReExports(unittest.TestCase):
    """Verify every public symbol is re-exported and resolves to the source module object."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _assert_same(self, shim_name: str, source_obj: object) -> None:
        shim_obj = getattr(ll, shim_name)
        self.assertIs(
            shim_obj,
            source_obj,
            f"ll.{shim_name} is not the same object as the source module version",
        )

    # ── Records (public) ────────────────────────────────────────────────────

    def test_action_execution_record(self) -> None:
        self._assert_same("ActionExecutionRecord", ActionExecutionRecord)

    def test_action_execution_status(self) -> None:
        self._assert_same("ActionExecutionStatus", ActionExecutionStatus)

    def test_action_verification_record(self) -> None:
        self._assert_same("ActionVerificationRecord", ActionVerificationRecord)

    def test_consolidation_record(self) -> None:
        self._assert_same("ConsolidationRecord", ConsolidationRecord)

    def test_consolidation_status(self) -> None:
        self._assert_same("ConsolidationStatus", ConsolidationStatus)

    def test_prediction_record(self) -> None:
        self._assert_same("PredictionRecord", PredictionRecord)

    def test_prediction_status(self) -> None:
        self._assert_same("PredictionStatus", PredictionStatus)

    def test_provenance_state(self) -> None:
        self._assert_same("ProvenanceState", ProvenanceState)

    def test_runtime_episode_trace(self) -> None:
        self._assert_same("RuntimeEpisodeTrace", RuntimeEpisodeTrace)

    def test_skill_memory_record(self) -> None:
        self._assert_same("SkillMemoryRecord", SkillMemoryRecord)

    def test_verification_status(self) -> None:
        self._assert_same("VerificationStatus", VerificationStatus)

    # ── Policy (public) ─────────────────────────────────────────────────────

    def test_policy_actuator_schema_version(self) -> None:
        self._assert_same("POLICY_ACTUATOR_SCHEMA_VERSION", POLICY_ACTUATOR_SCHEMA_VERSION)

    def test_policy_actuator_high_latency_avg_ms(self) -> None:
        self._assert_same("POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS", POLICY_ACTUATOR_HIGH_LATENCY_AVG_MS)

    def test_policy_actuator_high_latency_max_ms(self) -> None:
        self._assert_same("POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS", POLICY_ACTUATOR_HIGH_LATENCY_MAX_MS)

    def test_policy_actuator_recommendation(self) -> None:
        self._assert_same("PolicyActuatorRecommendation", PolicyActuatorRecommendation)

    def test_policy_score(self) -> None:
        self._assert_same("PolicyScore", PolicyScore)

    def test_world_model_lite_summary(self) -> None:
        self._assert_same("WorldModelLiteSummary", WorldModelLiteSummary)

    def test_build_policy_actuator_status(self) -> None:
        self._assert_same("build_policy_actuator_status", build_policy_actuator_status)

    # ── Replay (public) ─────────────────────────────────────────────────────

    def test_replay_candidate(self) -> None:
        self._assert_same("ReplayCandidate", ReplayCandidate)

    def test_replay_plan(self) -> None:
        self._assert_same("ReplayPlan", ReplayPlan)

    def test_build_replay_plan(self) -> None:
        self._assert_same("build_replay_plan", build_replay_plan)

    def test_replay_candidate_safety_flags(self) -> None:
        self._assert_same("replay_candidate_safety_flags", replay_candidate_safety_flags)

    def test_replay_plan_default_limit(self) -> None:
        self._assert_same("REPLAY_PLAN_DEFAULT_LIMIT", REPLAY_PLAN_DEFAULT_LIMIT)

    def test_replay_plan_max_limit(self) -> None:
        self._assert_same("REPLAY_PLAN_MAX_LIMIT", REPLAY_PLAN_MAX_LIMIT)

    def test_replay_plan_priority_rules_version(self) -> None:
        self._assert_same("REPLAY_PLAN_PRIORITY_RULES_VERSION", REPLAY_PLAN_PRIORITY_RULES_VERSION)

    def test_replay_plan_priority_weights(self) -> None:
        self._assert_same("REPLAY_PLAN_PRIORITY_WEIGHTS", REPLAY_PLAN_PRIORITY_WEIGHTS)

    def test_replay_plan_schema_version(self) -> None:
        self._assert_same("REPLAY_PLAN_SCHEMA_VERSION", REPLAY_PLAN_SCHEMA_VERSION)

    def test_replay_reason_precedence(self) -> None:
        self._assert_same("REPLAY_REASON_PRECEDENCE", REPLAY_REASON_PRECEDENCE)

    def test_replay_sample_safety_boundaries(self) -> None:
        self._assert_same("REPLAY_SAMPLE_SAFETY_BOUNDARIES", REPLAY_SAMPLE_SAFETY_BOUNDARIES)

    # ── Self-Model (public) ─────────────────────────────────────────────────

    def test_operational_self_model(self) -> None:
        self._assert_same("OperationalSelfModel", OperationalSelfModel)

    def test_build_runtime_benchmark_telemetry(self) -> None:
        self._assert_same("build_runtime_benchmark_telemetry", build_runtime_benchmark_telemetry)

    # ── Private helpers re-exported for cross-module use ─────────────────────

    def test_coerce_feedback_telemetry(self) -> None:
        self._assert_same("_coerce_feedback_telemetry", _coerce_feedback_telemetry)

    def test_policy_count(self) -> None:
        self._assert_same("_policy_count", _policy_count)

    def test_policy_float(self) -> None:
        self._assert_same("_policy_float", _policy_float)

    def test_policy_mapping(self) -> None:
        self._assert_same("_policy_mapping", _policy_mapping)

    def test_coerce_replay_sample_summary(self) -> None:
        self._assert_same("_coerce_replay_sample_summary", _coerce_replay_sample_summary)

    def test_default_replay_sample_safety_flags(self) -> None:
        self._assert_same("_default_replay_sample_safety_flags", _default_replay_sample_safety_flags)

    def test_as_mapping(self) -> None:
        self._assert_same("_as_mapping", _as_mapping)

    def test_clean_text(self) -> None:
        self._assert_same("_clean_text", _clean_text)

    def test_clamp01(self) -> None:
        self._assert_same("_clamp01", _clamp01)

    def test_coerce_world_model_lite(self) -> None:
        self._assert_same("_coerce_world_model_lite", _coerce_world_model_lite)

    def test_enum_value(self) -> None:
        self._assert_same("_enum_value", _enum_value)

    def test_latest_text(self) -> None:
        self._assert_same("_latest_text", _latest_text)

    def test_limited_unique_clean_text(self) -> None:
        self._assert_same("_limited_unique_clean_text", _limited_unique_clean_text)

    def test_provenance_value(self) -> None:
        self._assert_same("_provenance_value", _provenance_value)

    def test_safe_float(self) -> None:
        self._assert_same("_safe_float", _safe_float)

    def test_safe_ratio(self) -> None:
        self._assert_same("_safe_ratio", _safe_ratio)

    def test_stable_id(self) -> None:
        self._assert_same("_stable_id", _stable_id)

    def test_verification_status_from_payload(self) -> None:
        self._assert_same("_verification_status_from_payload", _verification_status_from_payload)

    def test_endpoint_bucket_name(self) -> None:
        self._assert_same("_endpoint_bucket_name", _endpoint_bucket_name)

    def test_endpoint_latency_empty(self) -> None:
        self._assert_same("_endpoint_latency_empty", _endpoint_latency_empty)

    def test_extract_cache_summary(self) -> None:
        self._assert_same("_extract_cache_summary", _extract_cache_summary)

    def test_extract_nim_summary(self) -> None:
        self._assert_same("_extract_nim_summary", _extract_nim_summary)

    def test_latency_summary(self) -> None:
        self._assert_same("_latency_summary", _latency_summary)

    def test_memory_counter_summary(self) -> None:
        self._assert_same("_memory_counter_summary", _memory_counter_summary)


class TestAllDunderExportsComplete(unittest.TestCase):
    """Verify that __all__ in living_loop.py covers every re-exported symbol."""

    def test_all_covers_every_reexported_symbol(self) -> None:
        """Every name listed in the shim's imports must appear in __all__."""
        all_names = set(ll.__all__)
        # Collect every name that living_loop.py makes available that isn't
        # a dunder attribute or the 'annotations' future import
        imported_names = {
            name
            for name in dir(ll)
            if not name.startswith("__") and name != "annotations"
        }
        missing = imported_names - all_names
        extra = all_names - imported_names
        self.assertEqual(
            missing,
            set(),
            f"Symbols available from living_loop but missing from __all__: {missing}",
        )
        self.assertEqual(
            extra,
            set(),
            f"Symbols in __all__ but not actually available from living_loop: {extra}",
        )


class TestConsumerImportSites(unittest.TestCase):
    """Verify all five consumer import sites still work without modification."""

    def test_living_status_mixin_imports(self) -> None:
        """LivingStatusMixin imports from living_loop."""
        from hecsn.service.living_loop import (
            ActionExecutionRecord,
            ConsolidationRecord,
            OperationalSelfModel,
            ProvenanceState,
            RuntimeEpisodeTrace,
            build_policy_actuator_status,
            build_replay_plan,
            build_runtime_benchmark_telemetry,
        )
        # Verify they resolved (would raise ImportError if not)
        self.assertTrue(callable(build_policy_actuator_status))
        self.assertTrue(callable(build_replay_plan))
        self.assertTrue(callable(build_runtime_benchmark_telemetry))

    def test_replay_runtime_mixin_imports(self) -> None:
        """ReplayRuntimeMixin imports from living_loop."""
        from hecsn.service.living_loop import (
            REPLAY_SAMPLE_SAFETY_BOUNDARIES,
            build_replay_plan,
            replay_candidate_safety_flags,
        )
        self.assertIsNotNone(REPLAY_SAMPLE_SAFETY_BOUNDARIES)
        self.assertTrue(callable(build_replay_plan))
        self.assertTrue(callable(replay_candidate_safety_flags))

    def test_runtime_evidence_mixin_imports(self) -> None:
        """RuntimeEvidenceMixin imports from living_loop."""
        from hecsn.service.living_loop import RuntimeEpisodeTrace, build_replay_plan
        self.assertIsNotNone(RuntimeEpisodeTrace)
        self.assertTrue(callable(build_replay_plan))

    def test_service_manager_imports(self) -> None:
        """ServiceManager imports from living_loop."""
        from hecsn.service.living_loop import (
            ActionExecutionRecord,
            ConsolidationRecord,
            OperationalSelfModel,
            ProvenanceState,
            RuntimeEpisodeTrace,
            REPLAY_SAMPLE_SAFETY_BOUNDARIES,
            build_policy_actuator_status,
            build_replay_plan,
            build_runtime_benchmark_telemetry,
            replay_candidate_safety_flags,
        )
        self.assertIsNotNone(REPLAY_SAMPLE_SAFETY_BOUNDARIES)

    def test_living_loop_primitives_imports(self) -> None:
        """test_living_loop_primitives imports from living_loop."""
        from hecsn.service.living_loop import (
            ActionExecutionRecord,
            ConsolidationRecord,
            OperationalSelfModel,
            PredictionStatus,
            ProvenanceState,
            RuntimeEpisodeTrace,
            SkillMemoryRecord,
            VerificationStatus,
            WorldModelLiteSummary,
            build_policy_actuator_status,
            build_replay_plan,
            build_runtime_benchmark_telemetry,
            replay_candidate_safety_flags,
        )
        self.assertIsNotNone(WorldModelLiteSummary)


if __name__ == "__main__":
    unittest.main()
