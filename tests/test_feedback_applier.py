from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
import unittest
from typing import Any
from threading import RLock

from marulho.service.action_executor import ActionExecutor
from marulho.service.feedback_applier import FeedbackApplier


class _FeedbackRuntimeState:
    def __init__(self, state_revision: int = 7) -> None:
        self.state_revision = state_revision
        self.dirty_state = False
        self.mark_mutated_calls = 0

    def mark_mutated(self) -> None:
        self.mark_mutated_calls += 1
        self.dirty_state = True
        self.state_revision += 1

    def mutation_summary(self) -> dict[str, Any]:
        return {
            "dirty_state": self.dirty_state,
            "state_revision": self.state_revision,
        }


class _FeedbackInteractionPipeline:
    def __init__(self, episodes: list[dict[str, Any]]) -> None:
        self.episodes = episodes
        self.read_calls: list[str] = []
        self.replace_calls: list[dict[str, Any]] = []

    def runtime_episode_trace(self, episode_id: str) -> dict[str, Any] | None:
        self.read_calls.append(str(episode_id))
        for episode in self.episodes:
            if str(episode.get("episode_id", "")) == str(episode_id):
                return deepcopy(episode)
        return None

    def replace_runtime_episode_trace(self, episode_id: str, episode: Any) -> dict[str, Any] | None:
        self.replace_calls.append({"episode_id": str(episode_id), "episode": deepcopy(dict(episode))})
        for index, current in enumerate(self.episodes):
            if str(current.get("episode_id", "")) == str(episode_id):
                stored = deepcopy(dict(episode))
                self.episodes[index] = stored
                return deepcopy(stored)
        return None


class _FeedbackEventSink:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: Any) -> None:
        self.events.append(deepcopy(dict(event)))


def _build_action_store(root: Path, action_history: list[dict[str, Any]]) -> ActionExecutor:
    runtime_state = _FeedbackRuntimeState()
    return ActionExecutor(
        lock=RLock(),
        action_root=root,
        action_history=action_history,
        runtime_state_mark_mutated_fn=runtime_state.mark_mutated,
        runtime_state_mutation_summary_fn=runtime_state.mutation_summary,
        record_brain_event_fn=lambda _event: None,
        brain_runtime_snapshot_fn=lambda: {"runtime": "snapshot", "state_revision": runtime_state.state_revision},
        runtime_trace_export_safe_value_fn=deepcopy,
        apply_provider_outcome_calibration_fn=lambda **_kwargs: True,
    )


class FeedbackApplierTests(unittest.TestCase):
    def test_runtime_episode_feedback_updates_verification_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runtime_state = _FeedbackRuntimeState()
            event_sink = _FeedbackEventSink()
            runtime_episode_store = _FeedbackInteractionPipeline(
                [
                    {
                        "episode_id": "episode-1",
                        "trace_id": "trace-1",
                        "operation": "query",
                        "status": "succeeded",
                        "created_at": "2026-05-09T00:00:00+00:00",
                        "request": {"query_text": "cats chase mice"},
                        "prediction": {"predicted_output": "query"},
                        "action": {"action_type": "query"},
                        "actual_output": {"summary": "Retrieved memory evidence."},
                        "verification": {
                            "status": "verified",
                            "success": True,
                            "confidence": 0.61,
                            "contradiction": False,
                            "summary": "Original runtime episode verdict.",
                            "provenance": "observed",
                            "last_feedback_id": "",
                            "last_feedback_at": "",
                            "feedback_count": 0,
                        },
                        "feedback": [],
                        "provenance": "observed",
                    }
                ]
            )
            applier = FeedbackApplier(
                lock=RLock(),
                runtime_episode_store=runtime_episode_store,
                action_store=_build_action_store(root, []),
                runtime_state_mark_mutated_fn=runtime_state.mark_mutated,
                runtime_state_mutation_summary_fn=runtime_state.mutation_summary,
                record_brain_event_fn=event_sink.record,
                brain_runtime_snapshot_fn=lambda: {"runtime": "snapshot", "state_revision": runtime_state.state_revision},
                runtime_trace_export_safe_value_fn=deepcopy,
            )

            feedback = applier.record_runtime_feedback(
                {
                    "target_type": "runtime_episode",
                    "target_id": "episode-1",
                    "verdict": "contradicted",
                    "confidence": 0.82,
                    "summary": "Operator corrected the feed trace outcome.",
                    "corrected_output": {"summary": "The feed text mentioned cats and mice."},
                    "evidence": [{"note": "manual review"}],
                    "tags": ["Manual", "manual", "Runtime"],
                    "evaluator_id": " operator-1 ",
                }
            )

            self.assertTrue(feedback["accepted"])
            self.assertTrue(feedback["dirty_state"])
            self.assertEqual(feedback["state_revision"], 8)
            self.assertEqual(runtime_state.mark_mutated_calls, 1)
            self.assertEqual(runtime_episode_store.read_calls, ["episode-1"])
            self.assertEqual(runtime_episode_store.replace_calls[0]["episode_id"], "episode-1")
            self.assertEqual(feedback["feedback"]["applied_status"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["status"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["provenance"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["feedback_count"], 1)
            self.assertEqual(feedback["target"]["feedback_status"], "contradicted")
            self.assertEqual(feedback["target"]["feedback_provenance"], "contradicted")
            self.assertEqual(feedback["target"]["feedback"][0]["tags"], ["manual", "runtime"])
            self.assertEqual(feedback["target"]["feedback"][0]["evaluator_id"], "operator-1")
            self.assertEqual(
                feedback["target"]["corrected_output"]["summary"],
                "The feed text mentioned cats and mice.",
            )
            self.assertEqual(
                runtime_episode_store.episodes[0]["feedback"][0]["summary"],
                "Operator corrected the feed trace outcome.",
            )
            self.assertEqual(event_sink.events[0]["type"], "runtime_feedback_recorded")
            self.assertEqual(feedback["terminus_runtime"]["runtime"], "snapshot")

    def test_action_feedback_with_corrected_output_forces_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            action_store = _build_action_store(
                root,
                [
                    {
                        "action_id": "action-1",
                        "action_type": "workspace_search",
                        "inputs": {"query_text": "cats chase mice"},
                        "predicted_outcome": "I expect to find cats chasing mice.",
                        "actual_outcome": "Workspace search found matching file hits.",
                        "verification": {
                            "status": "verified",
                            "success": True,
                            "confidence": 0.74,
                            "contradiction": False,
                            "summary": "Original action verdict.",
                            "evidence": [],
                            "provenance": "verified",
                            "last_feedback_id": "",
                            "last_feedback_at": "",
                            "feedback_count": 0,
                        },
                        "feedback": [],
                        "feedback_status": "verified",
                        "feedback_provenance": "verified",
                        "provenance": "verified",
                        "recorded_at": "2026-05-09T00:00:00+00:00",
                        "episode_text": "workspace search episode",
                        "trigger_reason": "operator",
                        "trigger_query_text": "cats chase mice",
                    }
                ],
            )
            runtime_state = _FeedbackRuntimeState()
            event_sink = _FeedbackEventSink()
            applier = FeedbackApplier(
                lock=RLock(),
                runtime_episode_store=_FeedbackInteractionPipeline([]),
                action_store=action_store,
                runtime_state_mark_mutated_fn=runtime_state.mark_mutated,
                runtime_state_mutation_summary_fn=runtime_state.mutation_summary,
                record_brain_event_fn=event_sink.record,
                brain_runtime_snapshot_fn=lambda: {"runtime": "snapshot", "state_revision": runtime_state.state_revision},
                runtime_trace_export_safe_value_fn=deepcopy,
            )

            feedback = applier.record_runtime_feedback(
                {
                    "target_type": "action",
                    "target_id": "action-1",
                    "verdict": "verified",
                    "confidence": 0.91,
                    "summary": "Manual evaluator corrected the action result.",
                    "corrected_output": {"summary": "The action returned the wrong value."},
                    "evidence": [{"source": "review"}],
                    "tags": ["reviewed"],
                    "evaluator_id": "qa-bot",
                }
            )

            self.assertTrue(feedback["accepted"])
            self.assertEqual(feedback["feedback"]["applied_status"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["status"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["provenance"], "contradicted")
            self.assertEqual(feedback["target"]["verification"]["feedback_count"], 1)
            self.assertEqual(feedback["target"]["feedback"][0]["corrected_output"]["summary"], "The action returned the wrong value.")
            self.assertEqual(feedback["target"]["feedback_status"], "contradicted")
            self.assertEqual(feedback["target"]["provenance"], "contradicted")
            self.assertEqual(action_store.action_record("action-1")["verification"]["status"], "contradicted")
            self.assertEqual(event_sink.events[0]["target_type"], "action")

