from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
import unittest
from threading import RLock
from typing import Any

from hecsn.service.action_executor import ActionExecutor


class _ActionExecutorRuntimeState:
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


def _build_action_executor(
    root: Path,
    *,
    action_history: list[dict[str, Any]] | None = None,
) -> tuple[ActionExecutor, list[dict[str, Any]], _ActionExecutorRuntimeState]:
    runtime_state = _ActionExecutorRuntimeState()
    brain_events: list[dict[str, Any]] = []
    executor = ActionExecutor(
        lock=RLock(),
        action_root=root,
        action_history=action_history or [],
        runtime_state_mark_mutated_fn=runtime_state.mark_mutated,
        runtime_state_mutation_summary_fn=runtime_state.mutation_summary,
        record_brain_event_fn=lambda event: brain_events.append(deepcopy(dict(event))),
        brain_runtime_snapshot_fn=lambda: {"runtime": "snapshot", "state_revision": runtime_state.state_revision},
        runtime_trace_export_safe_value_fn=deepcopy,
        apply_provider_outcome_calibration_fn=lambda **_kwargs: True,
    )
    return executor, brain_events, runtime_state


class ActionExecutorTests(unittest.TestCase):
    def test_execute_digital_action_records_verified_history_entry(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "notes.md").write_text(
                "Cats rest indoors during the day.\nCats chase mice at night.\n",
                encoding="utf-8",
            )
            executor, brain_events, runtime_state = _build_action_executor(root)

            result = executor.execute_digital_action(
                {
                    "action_type": "workspace_search",
                    "query_text": "cats chase mice",
                    "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                }
            )

            self.assertTrue(result["accepted"])
            self.assertEqual(result["result"]["verification"]["status"], "verified")
            self.assertEqual(result["state_revision"], 8)
            self.assertTrue(runtime_state.dirty_state)
            self.assertEqual(runtime_state.mark_mutated_calls, 1)
            self.assertEqual(brain_events[0]["type"], "digital_action_executed")
            history = executor.action_history(limit=4)
            self.assertEqual(history["count"], 1)
            self.assertEqual(history["actions"][0]["verification"]["status"], "verified")
            self.assertEqual(executor.action_record(result["result"]["action_id"])["action_id"], result["result"]["action_id"])

    def test_recent_verified_action_reuse_augments_query_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            preloaded_history = [
                {
                    "action_id": "action-1",
                    "action_type": "workspace_search",
                    "inputs": {"query_text": "cats chase mice"},
                    "predicted_outcome": "I expect to find evidence about cats chasing mice.",
                    "actual_outcome": "Workspace search found matching file hits.",
                    "verification": {
                        "status": "verified",
                        "success": True,
                        "confidence": 0.74,
                        "contradiction": False,
                        "summary": "Verified workspace evidence.",
                        "evidence": [
                            {
                                "snippet": "Cats chase mice at night.",
                                "path": "notes.md",
                            }
                        ],
                        "provenance": "verified",
                        "last_feedback_id": "",
                        "last_feedback_at": "",
                        "feedback_count": 0,
                    },
                    "feedback": [],
                    "feedback_status": "verified",
                    "feedback_provenance": "verified",
                    "provenance": "verified",
                    "recorded_at": "2026-05-10T00:00:00+00:00",
                    "episode_text": "workspace search episode",
                    "trigger_reason": "operator",
                    "trigger_query_text": "cats chase mice",
                }
            ]
            executor, _, _runtime_state = _build_action_executor(root, action_history=preloaded_history)
            query_result = {
                "query_summary": {"memory_episodes": []},
                "gap_plan": {"unsupported_terms": ["mice"], "grounded_fraction": 0.4},
            }
            response = {
                "response_mode": "insufficient_evidence",
                "unsupported_terms": ["mice"],
                "evidence_coverage": 0.2,
            }

            assist = executor.maybe_auto_action_assist(
                query_text="cats chase mice",
                query_result=query_result,
                response=response,
            )

            self.assertIsNotNone(assist)
            assert assist is not None
            self.assertTrue(assist["reused_recent_action"])
            self.assertFalse(assist["executed"])
            self.assertEqual(assist["reason"], "recent_verified_action")
            self.assertTrue(assist["used_in_response"])
            self.assertGreaterEqual(assist["response_episode_count"], 1)
            self.assertGreaterEqual(len(query_result["query_summary"]["memory_episodes"]), 1)
            self.assertEqual(query_result["query_summary"]["memory_episodes"][0]["action_origin"], "action-1")
