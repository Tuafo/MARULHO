from __future__ import annotations

from collections import deque
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
import unittest

from hecsn.service.brain_runtime import BrainRuntime, BrainRuntimeMixin
from hecsn.service.manager import HECSNServiceManager
from hecsn.service.runtime_sources import _BrainSourceRuntime

_SNAPSHOT_TIMESTAMP = "2026-05-10T00:00:00+00:00"
_DEFAULT_BRAIN_CONFIG = {
    "tick_tokens": 8,
    "sleep_interval_seconds": 0.01,
    "repeat_sources": True,
    "autonomy": None,
    "sensory": None,
}


def _source_spec(name: str, source: str, *, topic_terms: list[str] | None = None) -> dict[str, object]:
    spec: dict[str, object] = {
        "name": name,
        "source": source,
        "source_type": "file",
    }
    if topic_terms is not None:
        spec["topic_terms"] = list(topic_terms)
    return spec


def _tick_event(timestamp: str = _SNAPSHOT_TIMESTAMP) -> dict[str, str]:
    return {
        "type": "tick",
        "timestamp": timestamp,
    }


class _FakeRuntimeState:
    def __init__(self, snapshot: dict[str, object] | None = None) -> None:
        self.mutated = 0
        self._snapshot = snapshot or {
            "last_event": None,
            "recent_events": [],
        }

    def mark_mutated(self) -> None:
        self.mutated += 1

    def snapshot(self) -> dict[str, object]:
        return deepcopy(self._snapshot)


class _FakeThoughtLoop:
    def __init__(self) -> None:
        self.observations: list[dict[str, object]] = []
        self.surprises: list[dict[str, float]] = []

    def inject_observation(self, **kwargs: object) -> None:
        self.observations.append(deepcopy(kwargs))

    def inject_surprise(self, **kwargs: float) -> None:
        self.surprises.append({str(key): float(value) for key, value in kwargs.items()})

    def snapshot(self) -> dict[str, object]:
        return {"enabled": True}


class _FakeConceptStore:
    def snapshot(self, limit: int = 5) -> dict[str, object]:
        return {
            "top_concepts": [
                {"label": "cats"},
                {"label": "mice"},
            ][: max(1, int(limit))],
        }


class _FakeAutonomyPlanner:
    def _autonomy_focus_plan_locked(self) -> dict[str, object]:
        return {
            "query_terms": ["cats", "mice"],
            "unsupported_terms": ["rabbits"],
        }

    def _autonomy_focus_pressure_locked(self, focus_plan: dict[str, object]) -> tuple[float, dict[str, float]]:
        return 0.25, {"focus_pressure": 0.25}

    def _provider_curriculum_snapshot_locked(
        self,
        autonomy: object,
        focus_plan: dict[str, object],
    ) -> dict[str, object]:
        return {"providers": []}

    def _adaptive_autonomy_settings_locked(
        self,
        autonomy: object,
        focus_plan: dict[str, object],
    ) -> dict[str, int]:
        return {
            "effective_trigger_interval_tokens": 4096,
            "effective_acquisition_tokens": 512,
            "effective_acquisition_slots": 1,
        }

    def _update_provider_curriculum_locked(
        self,
        *,
        autonomy: object,
        result: object,
        candidate_specs: object,
        focus_plan: dict[str, object],
    ) -> None:
        return None


class _BrainRuntimeFixtureBase:
    def __init__(self) -> None:
        self._brain_config = dict(_DEFAULT_BRAIN_CONFIG)
        self._brain_source_utility: dict[str, dict[str, object]] = {}
        self._brain_source_runtimes: list[_BrainSourceRuntime] = []
        self._initialize_runtime_state()
        self._thought_loop_actual = _FakeThoughtLoop()
        self._concept_store = _FakeConceptStore()
        self._autonomy_planner = _FakeAutonomyPlanner()
        self._interaction_pipeline = SimpleNamespace(
            recent_query_gaps=lambda: [],
            runtime_episode_traces=lambda: [],
        )
        self._trainer = SimpleNamespace(
            token_count=108,
            config=SimpleNamespace(window_size=32),
            model=SimpleNamespace(
                surprise=SimpleNamespace(
                    dopamine=0.1,
                    serotonin=0.2,
                    norepinephrine=0.3,
                    acetylcholine=0.4,
                )
            ),
        )
        self._runtime_state = _FakeRuntimeState()
        self._brain_events: list[dict[str, object]] = []

    def _initialize_runtime_state(self) -> None:
        self._brain_source_index = 0
        self._brain_tick_count = 0
        self._brain_background_tokens = 0
        self._brain_autonomy_tokens = 0
        self._brain_last_acquisition_summary = None
        self._brain_last_acquisition_token_count = 0
        self._brain_last_error = None
        self._brain_running_since = None
        self._brain_last_tick_completed_at = None
        self._brain_last_tick_duration_ms = None
        self._brain_last_tick_token_delta = 0
        self._brain_last_work_at = None
        self._brain_stop_requested_at = None
        self._brain_stop_requested_reason = None
        self._brain_stop_requested_perf = None
        self._brain_stop_timed_out = False
        self._brain_last_stop_duration_ms = None
        self._last_real_sensory_episode_time = 0.0
        self._last_real_sensory_episode_token_count = 0
        self._real_sensory_last_error = None
        self._last_sensory_focus_terms = ()
        self._real_sensory_episodes_completed = 0
        self._real_visual_accepted = 0
        self._real_audio_accepted = 0
        self._ingestion_configured_at = None
        self._ingestion_configured_perf = None
        self._ingestion_prewarm_started_at = None
        self._ingestion_prewarm_started_perf = None
        self._ingestion_prewarm_completed_at = None
        self._ingestion_prewarm_last_duration_ms = None
        self._ingestion_prewarm_last_error = None
        self._ingestion_prewarm_run_count = 0
        self._ingestion_prewarm_last_trigger = None
        self._ingestion_prewarm_budget_exhausted = False
        self._ingestion_prewarm_running = False
        self._ingestion_prewarm_thread = None
        self._ingestion_prewarm_stop_event = None
        self._ingestion_warm_ready_at = None
        self._ingestion_startup_warm_latency_ms = None
        self._remote_warm_promotion_thread = None
        self._remote_warm_promotion_stop_event = None
        self._remote_warm_promotion_running = False
        self._remote_warm_promotion_last_trigger = None
        self._sensory_configured_at = None
        self._sensory_configured_perf = None
        self._sensory_prewarm_budget_exhausted = False
        self._sensory_warm_ready_at = None
        self._sensory_startup_warm_latency_ms = None
        self._brain_stream_epoch = 0
        self._sensory_stream_epoch = 0

    def _background_focus_terms_locked(self, *, focus_plan: Any = None) -> list[str]:
        return ["cats", "mice"]

    def _background_focus_overlap_locked(
        self,
        focus_terms: list[str],
        grounded_observation: dict[str, object],
    ) -> float:
        return 0.5

    def _record_brain_event_locked(self, event: dict[str, object]) -> None:
        self._brain_events.append(deepcopy(event))

    def _run_brain_autonomy_locked(self) -> None:
        return None

    def _run_real_sensory_episode_locked(self) -> None:
        return None

    def _multimodal_runtime_summary_locked(self) -> dict[str, bool]:
        return {"configured": False}

    def _runtime_environment_summary(self) -> dict[str, str]:
        return {"mode": "test"}

    def _action_loop_summary_locked(self) -> dict[str, str]:
        return {"mode": "idle"}

    def _huggingface_runtime_summary_locked(self) -> dict[str, int]:
        return {"source_count": 0}

    def _ingestion_runtime_summary_locked(self) -> dict[str, bool]:
        return {"configured": False}

    def _delayed_consequence_summary_locked(self, limit: int = 4) -> dict[str, int]:
        return {"record_count": 0, "limit": int(limit)}

    def _living_loop_snapshot_locked(
        self,
        *,
        cortex_snapshot: dict[str, object],
        include_replay_dataset_summary: bool = True,
    ) -> dict[str, object]:
        return {
            "cortex_snapshot": deepcopy(cortex_snapshot),
            "include_replay_dataset_summary": bool(include_replay_dataset_summary),
        }

    def _remerge_converged_delayed_consequence_families_locked(self) -> None:
        return None

    def _split_divergent_delayed_consequence_families_locked(self) -> None:
        return None

    def _compact_delayed_consequence_records_locked(self) -> None:
        return None

    def _cool_delayed_consequence_records_locked(self) -> None:
        return None

    def _brain_runtime_active_locked(self) -> bool:
        return False

    def _cortex_unavailable_snapshot(self) -> dict[str, bool]:
        return {"enabled": False}


class _FinalizeTickManager(_BrainRuntimeFixtureBase):
    pass


class _SnapshotManager(_BrainRuntimeFixtureBase):
    def __init__(self) -> None:
        super().__init__()
        self._brain_config = {
            **dict(_DEFAULT_BRAIN_CONFIG),
            "source_bank": [_source_spec("source_a", "source-a.txt")],
        }
        runtime = _BrainSourceRuntime(
            spec=_source_spec("source_a", "source-a.txt"),
            stream=iter(()),
            tokens_processed=8,
            tick_visits=1,
            last_tokens_trained=8,
            last_activity_at=_SNAPSHOT_TIMESTAMP,
        )
        self._brain_source_runtimes = [runtime]
        self._brain_background_tokens = 8
        self._brain_autonomy_tokens = 0
        self._brain_last_acquisition_token_count = 108
        self._brain_last_acquisition_summary = {"tokens_trained_total": 8}
        self._brain_last_tick_completed_at = _SNAPSHOT_TIMESTAMP
        self._brain_last_tick_duration_ms = 12.5
        self._brain_last_tick_token_delta = 8
        self._brain_last_work_at = _SNAPSHOT_TIMESTAMP
        self._runtime_state = _FakeRuntimeState(
            snapshot={
                "last_event": _tick_event(),
                "recent_events": [_tick_event()],
            }
        )


class BrainRuntimeSeamTests(unittest.TestCase):
    def test_alias_points_to_constructed_module(self) -> None:
        self.assertIs(BrainRuntimeMixin, BrainRuntime)

    def test_manager_uses_explicit_brain_runtime_seam(self) -> None:
        self.assertNotIn(BrainRuntimeMixin, HECSNServiceManager.__mro__)

    def test_finalize_tick_updates_source_runtime_and_injects_grounded_observation(self) -> None:
        manager = _FinalizeTickManager()
        module = BrainRuntime(manager)
        runtime = _BrainSourceRuntime(
            spec=_source_spec("science_source", "science.txt", topic_terms=["cats", "mice"]),
            stream=iter(()),
        )

        summary = module._finalize_tick_locked(
            123.0,
            {
                "runtime": runtime,
                "idx": 0,
                "source_count": 2,
            },
            total_trained=8,
            last_metrics={
                "pred_error": 0.6,
                "surprise": 0.3,
            },
            evidence_windows=[
                "Cats rest indoors and chase mice at night.",
                "They hunt quietly from the shadows.",
            ],
        )

        self.assertTrue(summary["did_work"])
        self.assertEqual(summary["source"]["source_name"], "science_source")
        self.assertIn("grounded_observation", summary["source"])
        grounded = summary["source"]["grounded_observation"]
        self.assertIn("cats", str(grounded["content"]).lower())
        self.assertIn("mice", str(grounded["content"]).lower())
        self.assertGreater(len(grounded["topics"]), 0)
        self.assertEqual(grounded["metadata"]["source_name"], "science_source")
        self.assertTrue(grounded["metadata"]["grounded"])
        self.assertEqual(runtime.tokens_processed, 8)
        self.assertEqual(runtime.tick_visits, 1)
        self.assertEqual(runtime.last_tokens_trained, 8)
        self.assertEqual(module._brain_background_tokens, 8)
        self.assertEqual(module._brain_tick_count, 1)
        self.assertEqual(module._brain_source_index, 1)
        self.assertEqual(manager._runtime_state.mutated, 2)
        self.assertEqual(len(manager._thought_loop_actual.observations), 1)
        self.assertEqual(len(manager._thought_loop_actual.surprises), 1)
        self.assertEqual(manager._brain_events[-1]["type"], "tick")
        self.assertGreater(module._background_source_utility_entry_locked(runtime)["utility_ema"], 0.0)

    def test_brain_runtime_snapshot_exposes_source_progress_and_status_state(self) -> None:
        manager = _SnapshotManager()
        module = BrainRuntime(manager)

        snapshot = module._brain_runtime_snapshot_locked(include_replay_dataset_summary=False)

        self.assertTrue(snapshot["configured"])
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["source_count"], 1)
        self.assertEqual(snapshot["next_source_name"], "source_a")
        self.assertEqual(snapshot["background_tokens_processed"], 8)
        self.assertEqual(snapshot["tick_count"], 0)
        self.assertEqual(snapshot["last_event"]["type"], "tick")
        self.assertEqual(snapshot["source_progress"][0]["name"], "source_a")
        self.assertEqual(snapshot["source_progress"][0]["tokens_processed"], 8)
        self.assertEqual(snapshot["source_progress"][0]["tick_visits"], 1)
        self.assertEqual(snapshot["source_progress"][0]["last_tokens_trained"], 8)
        self.assertEqual(snapshot["source_progress"][0]["share_of_background_tokens"], 1.0)
        self.assertEqual(snapshot["background_source_routing"]["selection_order"], ["source_a"])
        self.assertEqual(snapshot["background_source_routing"]["delayed_consequence_tracking"]["record_count"], 0)
        self.assertEqual(snapshot["text_learning_balance"]["background_tokens_processed"], 8)
        self.assertEqual(snapshot["living_loop"]["cortex_snapshot"]["enabled"], True)


if __name__ == "__main__":
    unittest.main()
