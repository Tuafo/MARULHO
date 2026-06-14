from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
from types import SimpleNamespace
from typing import Any
import unittest

from marulho.service.brain_runtime import BRAIN_RUNTIME_STATE_FIELDS, BrainRuntime, BrainRuntimeDependencies
from marulho.service.manager import MarulhoServiceManager
from marulho.service.runtime_sources import _BrainSourceRuntime

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
        self._concept_store = _FakeConceptStore()
        self._geometric_curiosity = SimpleNamespace(summary=lambda: {}, state_dict=lambda: {})
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
        self._action_history: deque[dict[str, object]] = deque()
        self._replay_sample_history: deque[dict[str, object]] = deque()
        self._replay_regeneration_permits: deque[dict[str, object]] = deque()
        self._snn_replay_evaluation_contexts: deque[dict[str, object]] = deque()
        self._snn_replay_artifact_recording_review_tickets: deque[dict[str, object]] = deque()
        self._snn_transition_memory_replay_artifacts: deque[dict[str, object]] = deque()
        self._brain_events: list[dict[str, object]] = []
        self._delayed_maintenance_calls: dict[str, int] = {
            "remerge": 0,
            "split": 0,
            "compact": 0,
            "cool": 0,
        }

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
        self._brain_last_tick_stage_timings_ms = {}
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

    def _ingestion_runtime_summary_locked(self) -> dict[str, object]:
        return {
            "configured": False,
            "encoder_execution_mode": "bounded_batched_inference",
            "hot_path_chunk_plasticity": False,
            "chunk_plasticity_path": "explicit_training_or_remote_bootstrap",
        }

    def _delayed_consequence_summary_locked(self, limit: int = 4) -> dict[str, int]:
        return {"record_count": 0, "limit": int(limit)}

    def _living_loop_snapshot_locked(
        self,
        *,
        include_replay_dataset_summary: bool = True,
    ) -> dict[str, object]:
        return {
            "subcortex_sleep_pressure": {"fatigue": 0.2},
            "include_replay_dataset_summary": bool(include_replay_dataset_summary),
        }

    def snn_sleep_plasticity_review_scheduler_runtime(self) -> dict[str, bool]:
        return {"enabled": False}

    def _remerge_converged_delayed_consequence_families_locked(self) -> None:
        self._delayed_maintenance_calls["remerge"] += 1
        return None

    def _split_divergent_delayed_consequence_families_locked(self) -> None:
        self._delayed_maintenance_calls["split"] += 1
        return None

    def _compact_delayed_consequence_records_locked(self) -> None:
        self._delayed_maintenance_calls["compact"] += 1
        return None

    def _cool_delayed_consequence_records_locked(self) -> None:
        self._delayed_maintenance_calls["cool"] += 1
        return None

    def _brain_runtime_active_locked(self) -> bool:
        return False

    def _brain_execution_snapshot_locked(self) -> dict[str, object]:
        return {
            "active_execution_requests": 0,
            "idle": True,
            "tick_in_progress": False,
            "tick_started_at": None,
            "tick_elapsed_ms": None,
            "tick_phase": None,
            "tick_source_name": None,
            "tick_target_tokens": None,
        }


def _brain_runtime_dependencies(fixture: _BrainRuntimeFixtureBase) -> BrainRuntimeDependencies:
    return BrainRuntimeDependencies(
        lock=RLock(),
        trainer=fixture._trainer,
        encoder=getattr(fixture, "_encoder", None),
        runtime_state=fixture._runtime_state,
        brain_config=lambda: fixture._brain_config,
        runtime_control=lambda: fixture,
        runtime_sources=lambda: fixture,
        delayed_consequence=lambda: fixture,
        autonomy_planner=lambda: fixture._autonomy_planner,
        source_focus=lambda: fixture,
        interaction_pipeline=lambda: fixture._interaction_pipeline,
        action_executor=lambda: fixture,
        replay_controller=lambda: fixture,
        concept_store=lambda: fixture._concept_store,
        geometric_curiosity=lambda: fixture._geometric_curiosity,
        runtime_environment_summary=fixture._runtime_environment_summary,
        huggingface_runtime_summary_locked=fixture._huggingface_runtime_summary_locked,
        ingestion_runtime_summary_locked=fixture._ingestion_runtime_summary_locked,
        multimodal_runtime_summary_locked=fixture._multimodal_runtime_summary_locked,
        sensory_runtime_summary_locked=lambda sensory: {},
        living_loop_snapshot_locked=fixture._living_loop_snapshot_locked,
        maybe_mark_ingestion_warm_locked=lambda **kwargs: None,
        maybe_mark_sensory_warm_locked=lambda **kwargs: None,
        observe_runtime_concepts_locked=getattr(
            fixture,
            "_observe_runtime_concepts_locked",
            lambda **kwargs: None,
        ),
        observe_runtime_concept_batch_locked=getattr(
            fixture,
            "_observe_runtime_concept_batch_locked",
            lambda **kwargs: [],
        ),
        runtime_concept_callback_locked=lambda **kwargs: None,
        run_real_sensory_episode_locked=fixture._run_real_sensory_episode_locked,
        record_brain_event_locked=fixture._record_brain_event_locked,
        build_brain_source_stream_locked=lambda spec: iter(()),
        build_sensory_stream_locked=lambda spec: iter(()),
    )


def _brain_runtime_from_fixture(fixture: _BrainRuntimeFixtureBase) -> BrainRuntime:
    module = BrainRuntime(_brain_runtime_dependencies(fixture))
    for field_name in BRAIN_RUNTIME_STATE_FIELDS:
        if hasattr(fixture, field_name):
            setattr(module, field_name, deepcopy(getattr(fixture, field_name)))
    return module


class _FinalizeTickManager(_BrainRuntimeFixtureBase):
    pass


class _ConceptSamplingManager(_BrainRuntimeFixtureBase):
    def __init__(self) -> None:
        super().__init__()
        self.concept_observation_windows: list[str] = []
        self.train_step_return_metrics_requests: list[bool] = []
        self.staged_input_quantum_sizes: list[int] = []
        self._trainer = SimpleNamespace(
            token_count=0,
            config=SimpleNamespace(window_size=32),
            train_step=self._train_step,
            stage_text_input_quantum=self.stage_text_input_quantum,
            model=self._trainer.model,
        )

    def _train_step(self, pattern: object, **kwargs: object) -> dict[str, object]:
        self._trainer.token_count += 1
        return_metrics = bool(kwargs.get("return_metrics", True))
        self.train_step_return_metrics_requests.append(return_metrics)
        if not return_metrics:
            return {}
        return {
            "memory_index": self._trainer.token_count - 1,
            "winner": 0,
            "train_step_metrics_mode": "full",
        }

    def stage_text_input_quantum(self, patterns: list[object]) -> bool:
        self.staged_input_quantum_sizes.append(len(patterns))
        return True

    def _observe_runtime_concepts_locked(
        self,
        *,
        raw_window: str | None,
        metrics: dict[str, Any] | None,
    ) -> dict[str, object]:
        self.concept_observation_windows.append(str(raw_window))
        return {"observed": True}

    def _observe_runtime_concept_batch_locked(
        self,
        *,
        observations: list[tuple[str | None, dict[str, Any] | None]],
    ) -> list[dict[str, object]]:
        return [
            self._observe_runtime_concepts_locked(
                raw_window=raw_window,
                metrics=metrics,
            )
            for raw_window, metrics in observations
        ]


class _BurstSamplingManager(_ConceptSamplingManager):
    def __init__(self) -> None:
        super().__init__()
        self.burst_sizes: list[int] = []
        self.burst_flush_reasons: list[str] = []
        self._trainer.train_text_burst = self.train_text_burst
        self._trainer.flush_text_burst_events = self.flush_text_burst_events

    def train_text_burst(
        self,
        patterns: list[object],
        **_kwargs: object,
    ) -> bool:
        self.burst_sizes.append(len(patterns))
        self._trainer.token_count += len(patterns)
        return True

    def flush_text_burst_events(self, *, reason: str) -> bool:
        self.burst_flush_reasons.append(str(reason))
        return True


class _SequenceSamplingManager(_ConceptSamplingManager):
    def __init__(self) -> None:
        super().__init__()
        self.sequence_calls: list[dict[str, object]] = []
        self._trainer.train_text_sequence = self.train_text_sequence

    def train_text_sequence(
        self,
        patterns: list[object],
        *,
        raw_windows: list[str],
        quantum_tokens: int,
        metric_indices: set[int],
        **_kwargs: object,
    ) -> dict[str, object]:
        self.sequence_calls.append(
            {
                "pattern_count": len(patterns),
                "raw_windows": list(raw_windows),
                "quantum_tokens": int(quantum_tokens),
                "metric_indices": set(metric_indices),
            }
        )
        start = self._trainer.token_count
        self._trainer.token_count += len(patterns)
        return {
            "trained": len(patterns),
            "metrics_by_index": {
                index: {
                    "memory_index": start + index,
                    "winner": 0,
                    "train_step_metrics_mode": "full",
                }
                for index in metric_indices
            },
            "quantum_count": len(patterns) // max(1, int(quantum_tokens)),
            "stopped": False,
        }


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
            cache_write_count=1,
            cache_schedule_count=3,
            cache_skip_count=2,
            cache_failure_count=0,
            cache_pending=True,
            last_cache_update_mode="skipped_unchanged_material",
        )
        self._brain_source_runtimes = [runtime]
        self._brain_background_tokens = 8
        self._brain_autonomy_tokens = 0
        self._brain_last_acquisition_token_count = 108
        self._brain_last_acquisition_summary = {"tokens_trained_total": 8}
        self._brain_last_tick_completed_at = _SNAPSHOT_TIMESTAMP
        self._brain_last_tick_duration_ms = 12.5
        self._brain_last_tick_stage_timings_ms = {
            "collect_source_queue": 1.5,
            "train_compute": 9.0,
            "finalize_total": 2.0,
        }
        self._brain_last_tick_token_delta = 8
        self._brain_last_work_at = _SNAPSHOT_TIMESTAMP
        self._runtime_state = _FakeRuntimeState(
            snapshot={
                "last_event": _tick_event(),
                "recent_events": [_tick_event()],
            }
        )


class _ActiveExecutionSnapshotManager(_SnapshotManager):
    def _brain_execution_snapshot_locked(self) -> dict[str, object]:
        return {
            "active_execution_requests": 1,
            "idle": False,
            "tick_in_progress": True,
            "tick_started_at": "2026-05-10T00:00:01+00:00",
            "tick_elapsed_ms": 125.0,
            "tick_phase": "train_sub_batches",
            "tick_source_name": "source_a",
            "tick_target_tokens": 8,
        }


class BrainRuntimeSeamTests(unittest.TestCase):
    def test_manager_uses_explicit_brain_runtime_seam(self) -> None:
        self.assertNotIn(BrainRuntime, MarulhoServiceManager.__mro__)

    def test_finalize_tick_updates_source_runtime_and_injects_grounded_observation(self) -> None:
        manager = _FinalizeTickManager()
        module = _brain_runtime_from_fixture(manager)
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
            stage_timings_ms={
                "collect_source_queue": 1.25,
                "train_lock_wait": 0.5,
                "train_compute": 8.0,
            },
            concept_observation_summary={
                "mode": "sampled",
                "interval_tokens": 8,
                "attempts": 2,
                "observations": 2,
            },
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
        self.assertEqual(
            summary["source"]["concept_observation"],
            {
                "mode": "sampled",
                "interval_tokens": 8,
                "attempts": 2,
                "observations": 2,
            },
        )
        self.assertEqual(summary["stage_timings_ms"]["collect_source_queue"], 1.25)
        self.assertEqual(summary["stage_timings_ms"]["train_lock_wait"], 0.5)
        self.assertEqual(summary["stage_timings_ms"]["train_compute"], 8.0)
        self.assertGreaterEqual(summary["stage_timings_ms"]["finalize_total"], 0.0)
        self.assertEqual(module._brain_background_tokens, 8)
        self.assertEqual(module._brain_tick_count, 1)
        self.assertEqual(module._brain_source_index, 1)
        self.assertEqual(manager._runtime_state.mutated, 2)
        self.assertEqual(
            summary["delayed_consequence_maintenance"],
            {
                "remerged_records": 0,
                "split_records": 0,
                "compacted_records": 0,
                "cooled_records": 0,
                "retired_records": 0,
            },
        )
        self.assertEqual(
            manager._delayed_maintenance_calls,
            {
                "remerge": 1,
                "split": 1,
                "compact": 1,
                "cool": 1,
            },
        )
        self.assertEqual(grounded["observation_sink"], "subcortex_grounded_source_observation")
        self.assertNotIn("retired_loop_mirrored", grounded)
        self.assertEqual(grounded["metadata"]["observation_sink"], "subcortex_grounded_source_observation")
        self.assertNotIn("retired_loop_mirrored", grounded["metadata"])
        self.assertFalse(hasattr(manager, "_thought_loop_actual"))
        self.assertEqual(manager._brain_events[-1]["type"], "tick")
        self.assertGreater(module._background_source_utility_entry_locked(runtime)["utility_ema"], 0.0)

    def test_background_training_samples_concept_observation(self) -> None:
        manager = _ConceptSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        stage_timings: dict[str, float] = {}
        chunk = [(f"window-{index}", object()) for index in range(1, 13)]

        trained, metrics, windows, observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=1,
            yield_seconds=0.0,
            stage_timings_ms=stage_timings,
        )

        self.assertEqual(trained, 12)
        self.assertEqual(metrics["memory_index"], 11)
        self.assertEqual(len(windows), 12)
        self.assertEqual(
            manager.concept_observation_windows,
            ["window-1", "window-8", "window-12"],
        )
        self.assertEqual(
            manager.train_step_return_metrics_requests,
            [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
            ],
        )
        self.assertEqual(
            observation,
            {
                "mode": "sampled_batched",
                "interval_tokens": 8,
                "tick_interval": 4,
                "tick_due": True,
                "max_per_tick": 4,
                "attempts": 3,
                "skipped_attempts": 0,
                "observations": 3,
                "batches": 1,
                "structural_maintenance_passes": 1,
            },
        )
        self.assertIn("trainer_step", stage_timings)
        self.assertIn("concept_observation", stage_timings)

    def test_background_training_can_cadence_source_concept_observation(self) -> None:
        manager = _ConceptSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        stage_timings: dict[str, float] = {}
        chunk = [(f"window-{index}", object()) for index in range(1, 13)]

        trained, metrics, windows, observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=8,
            yield_seconds=0.0,
            stage_timings_ms=stage_timings,
            concept_observation_due=False,
            concept_observation_tick_interval=4,
        )

        self.assertEqual(trained, 12)
        self.assertEqual(metrics["memory_index"], 11)
        self.assertEqual(len(windows), 12)
        self.assertEqual(manager.concept_observation_windows, [])
        self.assertEqual(observation["mode"], "cadenced_tick_skip")
        self.assertFalse(observation["tick_due"])
        self.assertEqual(observation["tick_interval"], 4)
        self.assertEqual(observation["attempts"], 0)
        self.assertEqual(observation["observations"], 0)
        self.assertNotIn("concept_observation", stage_timings)

    def test_background_training_mutates_runtime_once_per_execution_quantum(self) -> None:
        manager = _ConceptSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        chunk = [(f"window-{index}", object()) for index in range(1, 13)]

        trained, metrics, _windows, _observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=8,
            yield_seconds=0.0,
        )

        self.assertEqual(trained, 12)
        self.assertEqual(metrics["memory_index"], 11)
        self.assertEqual(manager._runtime_state.mutated, 2)
        self.assertEqual(manager.staged_input_quantum_sizes, [8, 4])

    def test_background_training_uses_burst_for_metric_free_quanta(self) -> None:
        manager = _BurstSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        chunk = [(f"window-{index}", object()) for index in range(1, 25)]

        trained, metrics, windows, observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=8,
            yield_seconds=0.0,
            concept_observation_due=False,
        )

        self.assertEqual(trained, 24)
        self.assertEqual(metrics["memory_index"], 23)
        self.assertEqual(len(windows), 24)
        self.assertEqual(observation["mode"], "cadenced_tick_skip")
        self.assertEqual(manager.burst_sizes, [8, 8])
        self.assertEqual(manager.staged_input_quantum_sizes, [8])
        self.assertEqual(manager.train_step_return_metrics_requests, [False] * 7 + [True])
        self.assertEqual(
            manager.burst_flush_reasons,
            ["service_per_token_boundary", "service_tick_complete"],
        )

    def test_background_training_delegates_complete_tick_to_training(self) -> None:
        manager = _SequenceSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        chunk = [(f"window-{index}", object()) for index in range(1, 129)]

        trained, metrics, windows, observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=8,
            yield_seconds=0.0,
        )

        self.assertEqual(trained, 128)
        self.assertEqual(metrics["memory_index"], 127)
        self.assertEqual(len(windows), 128)
        self.assertEqual(len(manager.sequence_calls), 1)
        self.assertEqual(manager.sequence_calls[0]["pattern_count"], 128)
        self.assertEqual(manager.sequence_calls[0]["quantum_tokens"], 8)
        self.assertEqual(
            manager.sequence_calls[0]["metric_indices"],
            {0, 7, 15, 23, 127},
        )
        self.assertEqual(manager._runtime_state.mutated, 1)
        self.assertEqual(observation["execution_owner"], "training_text_sequence")
        self.assertEqual(
            manager.concept_observation_windows,
            ["window-1", "window-8", "window-16", "window-24"],
        )

    def test_background_training_caps_concept_observation_per_tick(self) -> None:
        manager = _ConceptSamplingManager()
        module = _brain_runtime_from_fixture(manager)
        stage_timings: dict[str, float] = {}
        chunk = [(f"window-{index}", object()) for index in range(1, 129)]

        trained, metrics, windows, observation = module._train_chunk_in_sub_batches(
            chunk,
            stop_event=None,
            sub_batch_size=1,
            yield_seconds=0.0,
            stage_timings_ms=stage_timings,
        )

        self.assertEqual(trained, 128)
        self.assertEqual(metrics["memory_index"], 127)
        self.assertEqual(len(windows), 128)
        self.assertEqual(
            manager.concept_observation_windows,
            ["window-1", "window-8", "window-16", "window-24"],
        )
        self.assertEqual(observation["max_per_tick"], 4)
        self.assertTrue(observation["tick_due"])
        self.assertEqual(observation["tick_interval"], 4)
        self.assertEqual(observation["attempts"], 4)
        self.assertEqual(observation["skipped_attempts"], 13)
        self.assertEqual(observation["observations"], 4)
        self.assertEqual(observation["structural_maintenance_passes"], 1)

    def test_brain_runtime_snapshot_exposes_source_progress_and_status_state(self) -> None:
        manager = _SnapshotManager()
        module = _brain_runtime_from_fixture(manager)

        snapshot = module._brain_runtime_snapshot_locked(include_replay_dataset_summary=False)

        self.assertTrue(snapshot["configured"])
        self.assertFalse(snapshot["running"])
        self.assertEqual(snapshot["source_count"], 1)
        self.assertEqual(snapshot["next_source_name"], "source_a")
        self.assertEqual(snapshot["background_tokens_processed"], 8)
        self.assertEqual(snapshot["tick_count"], 0)
        self.assertEqual(
            snapshot["last_tick_stage_timings_ms"],
            {
                "collect_source_queue": 1.5,
                "train_compute": 9.0,
                "finalize_total": 2.0,
            },
        )
        self.assertEqual(snapshot["last_event"]["type"], "tick")
        self.assertEqual(snapshot["source_progress"][0]["name"], "source_a")
        self.assertEqual(snapshot["source_progress"][0]["tokens_processed"], 8)
        self.assertEqual(snapshot["source_progress"][0]["tick_visits"], 1)
        self.assertEqual(snapshot["source_progress"][0]["last_tokens_trained"], 8)
        self.assertEqual(snapshot["source_progress"][0]["share_of_background_tokens"], 1.0)
        self.assertEqual(snapshot["source_progress"][0]["cache_write_count"], 1)
        self.assertEqual(snapshot["source_progress"][0]["cache_schedule_count"], 3)
        self.assertEqual(snapshot["source_progress"][0]["cache_skip_count"], 2)
        self.assertEqual(snapshot["source_progress"][0]["cache_failure_count"], 0)
        self.assertTrue(snapshot["source_progress"][0]["cache_pending"])
        self.assertEqual(snapshot["source_progress"][0]["last_cache_update_mode"], "skipped_unchanged_material")
        self.assertEqual(snapshot["ingestion"]["encoder_execution_mode"], "bounded_batched_inference")
        self.assertFalse(snapshot["ingestion"]["hot_path_chunk_plasticity"])
        self.assertEqual(
            snapshot["ingestion"]["chunk_plasticity_path"],
            "explicit_training_or_remote_bootstrap",
        )
        self.assertEqual(snapshot["background_source_routing"]["selection_order"], ["source_a"])
        self.assertEqual(snapshot["background_source_routing"]["delayed_consequence_tracking"]["record_count"], 0)
        self.assertEqual(snapshot["text_learning_balance"]["background_tokens_processed"], 8)
        self.assertNotIn("retired_runtime_dependency", snapshot["living_loop"]["subcortex_sleep_pressure"])
        self.assertNotIn("retired_runtime_path", snapshot)
        self.assertEqual(
            manager._delayed_maintenance_calls,
            {
                "remerge": 0,
                "split": 0,
                "compact": 0,
                "cool": 0,
            },
        )

    def test_brain_runtime_snapshot_exposes_active_tick_execution_evidence(self) -> None:
        manager = _ActiveExecutionSnapshotManager()
        module = _brain_runtime_from_fixture(manager)

        snapshot = module._brain_runtime_snapshot_locked(include_replay_dataset_summary=False)

        self.assertEqual(snapshot["execution"]["active_execution_requests"], 1)
        self.assertFalse(snapshot["execution"]["idle"])
        self.assertTrue(snapshot["execution"]["tick_in_progress"])
        self.assertEqual(snapshot["execution"]["tick_phase"], "train_sub_batches")
        self.assertEqual(
            snapshot["execution_schedule"],
            {
                "quantum_tokens": 8,
                "yield_seconds": 0.0,
                "stop_check_boundary": "between_quanta",
                "sequential_token_training": True,
                "execution_owner": "training_text_sequence",
                "service_dispatches_per_tick": 1,
            },
        )
        self.assertEqual(snapshot["execution"]["tick_source_name"], "source_a")
        self.assertEqual(snapshot["execution"]["tick_target_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
