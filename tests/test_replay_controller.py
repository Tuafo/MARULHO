from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
import unittest
from unittest.mock import patch

from hecsn.service.replay_runtime import ReplayController, ReplayRuntimeMixin


@dataclass
class _FakeRuntimeState:
    state_revision: int = 7

    def __post_init__(self) -> None:
        self.dirty_without_revision_calls = 0

    def mark_dirty_without_revision(self) -> None:
        self.dirty_without_revision_calls += 1

    def mutation_summary(self) -> dict[str, int]:
        return {"dirty_state": True, "state_revision": self.state_revision}


class _FakeReplayManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._runtime_state = _FakeRuntimeState()
        self._thought_loop_actual = None
        self._replay_sample_history = deque(maxlen=256)
        self._trainer = type("_Trainer", (), {"token_count": 13})()
        self._action_history = deque(maxlen=24)

    @staticmethod
    def _normalize_action_text(value: object) -> str:
        return " ".join(str(value).split()).strip()

    @classmethod
    def _normalize_feedback_text(cls, value: object, *, max_chars: int = 2000) -> str:
        text = cls._normalize_action_text(value)
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + "…"
        return text

    def _cortex_unavailable_snapshot(self) -> dict[str, object]:
        return {"enabled": False, "initialization": {"started": False, "finished": True, "timed_out": False, "error": None}}

    def _living_loop_snapshot_locked(self, *, cortex_snapshot: dict[str, object]) -> dict[str, object]:
        return {
            "state_revision": self._runtime_state.state_revision,
            "token_count": 13,
            "runtime_episode_count": 0,
            "action_count": 0,
            "prediction_count": 0,
            "runtime_episodes": [],
            "actions": [],
            "predictions": [],
            "uncertain_domains": [],
            "feedback_summary": {},
            "benchmark_telemetry": {},
            "memory_health": {"status": "available", "fill_ratio": 0.2},
            "policy_decision": {"action": "continue_current_policy"},
            "world_model_lite": {"uncertainty": 0.0},
            "cortex": cortex_snapshot,
        }

    @staticmethod
    def _replay_plan_summary(plan: dict[str, object] | None) -> dict[str, object]:
        payload = dict(plan or {})
        return {"endpoint": payload.get("endpoint", "/terminus/replay-plan"), "count": len(payload.get("candidates", []))}

    @staticmethod
    def _runtime_trace_export_safe_value(value: object) -> object:
        return value

    def _replay_sample_state_counts_locked(self) -> dict[str, int]:
        return {
            "token_count": 13,
            "state_revision": self._runtime_state.state_revision,
            "action_history_count": 0,
            "feedback_count": 0,
        }

    @staticmethod
    def _runtime_feedback_summary_locked() -> dict[str, int]:
        return {"feedback_count": 0}


class ReplayControllerTests(unittest.TestCase):
    def test_alias_points_to_constructed_module(self) -> None:
        self.assertIs(ReplayRuntimeMixin, ReplayController)

    def test_replay_sample_history_is_controller_owned(self) -> None:
        manager = _FakeReplayManager()
        controller = ReplayController(manager)
        controller.history.appendleft(
            {
                "schema_version": 1,
                "replay_sample_id": "replay-sample-1",
                "mode": "sample",
                "status": "recorded",
                "selected_candidate_ids": ["candidate-1"],
                "selected_candidates": [],
                "safety_flags": {"audit_only": True, "operator_confirmed": True},
            }
        )

        history = controller.replay_sample_history(limit=1)

        self.assertEqual(history["count"], 1)
        self.assertEqual(history["history"][0]["replay_sample_id"], "replay-sample-1")
        self.assertEqual(controller.history[0]["replay_sample_id"], "replay-sample-1")

    def test_replay_sample_marks_dirty_without_revision(self) -> None:
        manager = _FakeReplayManager()
        controller = ReplayController(manager)

        plan_payload = {
            "endpoint": "/terminus/replay-plan",
            "candidates": [
                {
                    "candidate_id": "candidate-1",
                    "target_type": "runtime_episode",
                    "target_id": "episode-1",
                    "priority_score": 8.0,
                }
            ],
        }

        class _FakePlan:
            def to_payload(self) -> dict[str, object]:
                return plan_payload

        with patch("hecsn.service.replay_runtime.build_replay_plan", return_value=_FakePlan()):
            record = controller.replay_sample(
                mode="sample",
                candidate_id="candidate-1",
                operator_id="operator-1",
                confirmation=True,
            )

        self.assertEqual(manager._runtime_state.dirty_without_revision_calls, 1)
        self.assertEqual(record["selected_candidate_ids"], ["candidate-1"])
        self.assertFalse(record["safety_flags"]["state_revision_mutated"])
        self.assertEqual(record["before"]["state_revision"], record["after"]["state_revision"])
        self.assertEqual(controller.history[0]["replay_sample_id"], record["replay_sample_id"])


if __name__ == "__main__":
    unittest.main()
