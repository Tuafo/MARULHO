from __future__ import annotations

from collections import deque
from copy import deepcopy
from threading import RLock
import unittest
from typing import Any, Mapping, cast

from hecsn.service.history_store import read_history_record, replace_history_record
from hecsn.service.runtime_feedback import RuntimeFeedbackMixin


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

    def replace_runtime_episode_trace(self, episode_id: str, episode: Mapping[str, Any]) -> dict[str, Any] | None:
        self.replace_calls.append(
            {
                "episode_id": str(episode_id),
                "episode": deepcopy(dict(episode)),
            }
        )
        for index, current in enumerate(self.episodes):
            if str(current.get("episode_id", "")) == str(episode_id):
                stored = deepcopy(dict(episode))
                self.episodes[index] = stored
                return deepcopy(stored)
        return None


class _RuntimeFeedbackHarness(RuntimeFeedbackMixin):
    def __init__(self) -> None:
        self._lock = RLock()
        self._runtime_state = _FeedbackRuntimeState()
        self._brain_events: list[dict[str, Any]] = []
        self._interaction_pipeline = _FeedbackInteractionPipeline(
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
        self._action_history = deque(
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
            maxlen=24,
        )
        self.action_record_calls: list[str] = []
        self.replace_action_record_calls: list[dict[str, Any]] = []

    def _runtime_trace_export_safe_value(self, value: Any) -> Any:
        return deepcopy(value)

    def _record_brain_event_locked(self, event: Mapping[str, Any]) -> None:
        self._brain_events.append(deepcopy(dict(event)))

    def _brain_runtime_snapshot_locked(self) -> dict[str, Any]:
        return {
            "runtime": "snapshot",
            "state_revision": self._runtime_state.state_revision,
        }

    def action_record(self, action_id: str) -> dict[str, Any] | None:
        self.action_record_calls.append(str(action_id))
        with self._lock:
            return read_history_record(self._action_history, record_id=action_id, id_field="action_id")

    def replace_action_record(self, action_id: str, record: Mapping[str, Any]) -> dict[str, Any] | None:
        self.replace_action_record_calls.append(
            {
                "action_id": str(action_id),
                "record": deepcopy(dict(record)),
            }
        )
        with self._lock:
            self._action_history, replaced = replace_history_record(
                self._action_history,
                record_id=action_id,
                replacement=record,
                id_field="action_id",
            )
            return replaced


class ActionRuntimeSeamTests(unittest.TestCase):
    def test_action_record_read_and_replace_preserve_history_order(self) -> None:
        harness = _RuntimeFeedbackHarness()

        action = harness.action_record("action-1")
        self.assertIsNotNone(action)
        assert action is not None
        self.assertEqual(action["action_id"], "action-1")

        action["feedback"].append(
            {
                "feedback_id": "fb-1",
                "verdict": "verified",
                "applied_status": "verified",
            }
        )
        action["verification"]["status"] = "contradicted"

        replaced = harness.replace_action_record("action-1", action)
        self.assertIsNotNone(replaced)
        assert replaced is not None
        self.assertEqual(replaced["verification"]["status"], "contradicted")
        stored_action = cast(dict[str, Any], harness._action_history[0])
        self.assertEqual(stored_action["verification"]["status"], "contradicted")
        self.assertEqual(stored_action["feedback"][0]["feedback_id"], "fb-1")
        self.assertEqual(harness.action_record_calls, ["action-1"])
        self.assertEqual(harness.replace_action_record_calls[0]["action_id"], "action-1")


class RuntimeFeedbackRoutingTests(unittest.TestCase):
    def test_runtime_episode_feedback_routes_through_interaction_pipeline_seam(self) -> None:
        harness = _RuntimeFeedbackHarness()

        feedback = harness.record_runtime_feedback(
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
        self.assertEqual(harness._runtime_state.mark_mutated_calls, 1)
        self.assertEqual(harness._interaction_pipeline.read_calls, ["episode-1"])
        self.assertEqual(harness._interaction_pipeline.replace_calls[0]["episode_id"], "episode-1")
        self.assertEqual(feedback["feedback"]["applied_status"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["status"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["provenance"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["feedback_count"], 1)
        self.assertEqual(feedback["target"]["feedback_status"], "contradicted")
        self.assertEqual(feedback["target"]["feedback_provenance"], "contradicted")
        self.assertEqual(feedback["target"]["feedback"][0]["tags"], ["manual", "runtime"])
        self.assertEqual(feedback["target"]["feedback"][0]["evaluator_id"], "operator-1")
        self.assertEqual(feedback["target"]["corrected_output"]["summary"], "The feed text mentioned cats and mice.")
        self.assertEqual(harness._interaction_pipeline.episodes[0]["feedback"][0]["summary"], "Operator corrected the feed trace outcome.")
        self.assertEqual(harness._brain_events[0]["type"], "runtime_feedback_recorded")
        self.assertEqual(feedback["terminus_runtime"]["runtime"], "snapshot")

    def test_action_feedback_routes_through_action_seam_and_corrected_output_forces_contradiction(self) -> None:
        harness = _RuntimeFeedbackHarness()

        feedback = harness.record_runtime_feedback(
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
        self.assertEqual(harness.action_record_calls, ["action-1"])
        self.assertEqual(harness.replace_action_record_calls[0]["action_id"], "action-1")
        self.assertEqual(feedback["feedback"]["applied_status"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["status"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["provenance"], "contradicted")
        self.assertEqual(feedback["target"]["verification"]["feedback_count"], 1)
        self.assertEqual(feedback["target"]["feedback"][0]["corrected_output"]["summary"], "The action returned the wrong value.")
        self.assertEqual(feedback["target"]["feedback_status"], "contradicted")
        self.assertEqual(feedback["target"]["provenance"], "contradicted")
        stored_action = cast(dict[str, Any], harness._action_history[0])
        self.assertEqual(stored_action["verification"]["status"], "contradicted")
        self.assertEqual(stored_action["feedback"][0]["evaluator_id"], "qa-bot")
        self.assertEqual(harness._brain_events[0]["target_type"], "action")
