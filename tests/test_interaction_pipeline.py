"""Direct test surface for the Interaction Pipeline seam."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from typing import Any

from hecsn.service.interaction_pipeline import InteractionPipeline


def _build_pipeline(
    trace_dir: Path,
    *,
    build_query_raises: BaseException | None = None,
) -> tuple[InteractionPipeline, dict[str, Any]]:
    calls: dict[str, Any] = {
        "build_query_result": [],
        "observe_concepts": [],
        "plan_gaps": [],
        "apply_delayed": [],
        "record_recent_query_gap": [],
        "runtime_episode_payload": [],
        "persist_trace": [],
        "append_runtime_episode_trace": [],
        "service_state_snapshot": [],
    }
    episodes: list[dict[str, Any]] = []

    def build_query_result_fn(**kwargs: Any) -> dict[str, Any]:
        calls["build_query_result"].append(kwargs)
        if build_query_raises is not None:
            raise build_query_raises
        return {
            "query_summary": {
                "query_text": kwargs["query_text"],
                "top_candidates": [{"candidate_id": "cand-1"}],
                "memory_matches": [
                    {"text": "cats chase mice", "similarity": 0.91},
                    {"text": "cats rest indoors", "similarity": 0.77},
                ],
                "memory_episodes": [{"episode_id": "episode-a"}],
                "winner_column": 2,
                "reconstruction_error": 0.125,
            },
            "checkpoint": "test://checkpoint",
            "checkpoint_metadata": {},
            "config": {},
            "feed_summary": None,
            "context_summary": None,
            "context_comparison": None,
        }

    def observe_concepts_fn(**kwargs: Any) -> dict[str, Any]:
        calls["observe_concepts"].append(kwargs)
        return {
            "concept_count": 3,
            "observations": 5,
            "top_concepts": [{"concept_id": "cats", "label": "cats"}],
        }

    def plan_gaps_fn(**kwargs: Any) -> dict[str, Any]:
        calls["plan_gaps"].append(kwargs)
        return {
            "planner_mode": "semantic_gap_planner",
            "grounded_fraction": 0.6,
            "unsupported_terms": ["mice"],
            "retrieval_queries": ["cats chase"],
            "follow_up_questions": ["What do cats chase?"],
            "weak_concepts": [],
        }

    def apply_delayed_query_consequence_fn(**kwargs: Any) -> dict[str, Any]:
        calls["apply_delayed"].append(kwargs)
        return {
            "enabled": True,
            "matched_records": 1,
            "credited_records": 0,
            "penalized_records": 0,
            "forgiven_records": 0,
        }

    def record_recent_query_gap_fn(**kwargs: Any) -> None:
        calls["record_recent_query_gap"].append(kwargs)

    def runtime_episode_payload_fn(**kwargs: Any) -> dict[str, Any]:
        calls["runtime_episode_payload"].append(kwargs)
        payload = {
            "episode_id": "episode-1",
            "trace_id": kwargs["trace_id"],
            "operation": kwargs["operation"],
            "request": kwargs["request"],
            "prediction": kwargs["prediction"],
            "action": kwargs["action"],
            "actual_output": kwargs["actual_output"],
            "verification": kwargs["verification"],
            "status": "failed" if kwargs.get("error") is not None else "succeeded",
            "created_at": kwargs["created_at"],
            "trace_path": kwargs.get("trace_path", ""),
        }
        if kwargs.get("error") is not None:
            error = kwargs["error"]
            payload["failure"] = {
                "error_type": type(error).__name__,
                "message": str(error),
            }
        return payload

    def persist_trace_fn(trace: dict[str, Any]) -> Path:
        calls["persist_trace"].append(trace)
        trace_path = trace_dir / f"{trace['trace_id']}.json"
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return trace_path

    def append_runtime_episode_trace_fn(episode: dict[str, Any]) -> dict[str, Any]:
        calls["append_runtime_episode_trace"].append(episode)
        episodes.append(dict(episode))
        return dict(episode)

    def service_state_snapshot_fn(*, include_replay_dataset_summary: bool = True) -> dict[str, Any]:
        calls["service_state_snapshot"].append(include_replay_dataset_summary)
        return {
            "checkpoint_path": "/tmp/test.pt",
            "dirty_state": False,
            "state_revision": 7,
            "token_count": 12,
            "concept_count": 3,
            "terminus_runtime": {
                "configured": True,
                "running": False,
            },
        }

    pipeline = InteractionPipeline(
        lock=threading.RLock(),
        build_query_result_fn=build_query_result_fn,
        observe_concepts_fn=observe_concepts_fn,
        plan_gaps_fn=plan_gaps_fn,
        apply_delayed_query_consequence_fn=apply_delayed_query_consequence_fn,
        record_recent_query_gap_fn=record_recent_query_gap_fn,
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        append_runtime_episode_trace_fn=append_runtime_episode_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
    )
    calls["episodes"] = episodes
    return pipeline, calls


class InteractionPipelineQueryTests(unittest.TestCase):
    def test_query_orchestrates_query_turn_and_persists_runtime_episode_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, calls = _build_pipeline(trace_dir)

            result = pipeline.query(
                query_text="cats chase mice",
                context_text="shared context",
                top_k_candidates=3,
                top_k_memories=4,
                top_chars=5,
            )

            self.assertEqual(len(calls["build_query_result"]), 1)
            build_kwargs = calls["build_query_result"][0]
            self.assertEqual(build_kwargs["query_text"], "cats chase mice")
            self.assertEqual(build_kwargs["context_text"], "shared context")
            self.assertEqual(build_kwargs["top_k_candidates"], 3)
            self.assertEqual(build_kwargs["top_k_memories"], 4)
            self.assertEqual(build_kwargs["top_chars"], 5)
            self.assertEqual(len(calls["observe_concepts"]), 1)
            self.assertEqual(len(calls["plan_gaps"]), 1)
            self.assertEqual(len(calls["apply_delayed"]), 1)
            self.assertEqual(len(calls["record_recent_query_gap"]), 1)
            self.assertEqual(calls["record_recent_query_gap"][0]["source"], "query")
            self.assertFalse(calls["service_state_snapshot"][0])

            self.assertIn("concept_summary", result)
            self.assertIn("gap_plan", result)
            self.assertIn("delayed_consequence", result)
            self.assertEqual(result["service_state"]["state_revision"], 7)
            self.assertEqual(result["runtime_episode"]["operation"], "query")
            self.assertTrue(str(result["runtime_episode"]["trace_path"]).endswith(".json"))
            self.assertEqual(result["runtime_episode"]["actual_output"]["summary"], "Retrieved 2 memory matches and 1 memory episodes.")
            self.assertEqual(result["runtime_episode"]["verification"]["status"], "verified")
            self.assertAlmostEqual(float(result["runtime_episode"]["verification"]["confidence"]), 0.6)
            self.assertTrue((trace_dir / f"{result['runtime_episode']['trace_id']}.json").exists())

            stored_trace = json.loads((trace_dir / f"{result['runtime_episode']['trace_id']}.json").read_text(encoding="utf-8"))
            self.assertEqual(stored_trace["operation"], "query")
            self.assertEqual(stored_trace["runtime_episode"]["episode_id"], "episode-1")
            self.assertEqual(stored_trace["state_after"]["state_revision"], 7)
            self.assertEqual(len(calls["episodes"]), 1)

    def test_query_runtime_actual_output_and_verification_reflect_query_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, _calls = _build_pipeline(trace_dir)
            payload = {
                "query_summary": {
                    "query_text": "cats chase mice",
                    "top_candidates": [{"candidate_id": "cand-1"}],
                    "memory_matches": [{"text": "cats chase mice"}],
                    "memory_episodes": [],
                    "winner_column": 4,
                    "reconstruction_error": 0.25,
                },
                "gap_plan": {
                    "planner_mode": "semantic_gap_planner",
                    "grounded_fraction": 0.1,
                    "unsupported_terms": ["mice"],
                    "retrieval_queries": ["cats chase", "mice chase"],
                },
                "concept_summary": {
                    "concept_count": 9,
                    "observations": 12,
                    "top_concepts": [{"label": "cats"}, {"label": "mice"}],
                },
            }

            actual = pipeline._query_runtime_actual_output(payload)
            verification = pipeline._query_runtime_verification(payload)

            self.assertEqual(actual["summary"], "Retrieved 1 memory matches and 0 memory episodes.")
            self.assertEqual(actual["top_candidate_count"], 1)
            self.assertEqual(actual["memory_match_count"], 1)
            self.assertEqual(actual["gap_plan"]["retrieval_queries"], ["cats chase", "mice chase"])
            self.assertEqual(actual["concept_summary"]["top_concepts"][0]["label"], "cats")
            self.assertEqual(verification["status"], "verified")
            self.assertTrue(verification["success"])
            self.assertAlmostEqual(float(verification["confidence"]), 0.2)

    def test_query_failure_still_persists_error_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, calls = _build_pipeline(trace_dir, build_query_raises=RuntimeError("boom"))

            with self.assertRaises(RuntimeError):
                pipeline.query(query_text="cats chase mice")

            self.assertEqual(len(calls["persist_trace"]), 1)
            self.assertEqual(len(calls["append_runtime_episode_trace"]), 1)
            stored_trace = calls["persist_trace"][0]
            self.assertEqual(stored_trace["operation"], "query")
            self.assertEqual(stored_trace["error"]["type"], "RuntimeError")
            self.assertEqual(stored_trace["error"]["message"], "boom")
            self.assertEqual(stored_trace["runtime_episode"]["status"], "failed")
            self.assertEqual(stored_trace["runtime_episode"]["failure"]["error_type"], "RuntimeError")
            self.assertFalse(calls["service_state_snapshot"][0])
