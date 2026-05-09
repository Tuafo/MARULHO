"""Direct test surface for the Interaction Pipeline seam."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest
from typing import Any, Mapping

from hecsn.service.interaction_pipeline import (
    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    REQUEST_FEED_ENCODING_MODE,
    InteractionPipeline,
    build_feed_runtime_actual_output,
    build_feed_runtime_verification,
)


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

    def append_runtime_episode_trace_fn(episode: Mapping[str, Any]) -> dict[str, Any]:
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
        trainer=_FeedTestTrainer(),
        encoder=_FeedTestEncoder([]),
        build_query_result_fn=build_query_result_fn,
        observe_concepts_fn=observe_concepts_fn,
        plan_gaps_fn=plan_gaps_fn,
        apply_delayed_query_consequence_fn=apply_delayed_query_consequence_fn,
        record_recent_query_gap_fn=record_recent_query_gap_fn,
        observe_runtime_concepts_fn=lambda **kwargs: None,
        runtime_state_mark_mutated_fn=lambda: None,
        runtime_state_mutation_summary_fn=lambda: {"dirty_state": False, "state_revision": 0},
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        append_runtime_episode_trace_fn=append_runtime_episode_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
    )
    calls["episodes"] = episodes
    return pipeline, calls


class _FeedTestConfig:
    def __init__(self, window_size: int = 4) -> None:
        self.window_size = window_size


class _FeedTestMemoryStore:
    def __init__(self) -> None:
        self.slow_buffer: list[Any] = []


class _FeedTestModel:
    def __init__(self) -> None:
        self.memory_store = _FeedTestMemoryStore()


class _FeedTestTrainer:
    def __init__(self, *, window_size: int = 4, train_step_raises: BaseException | None = None) -> None:
        self.config = _FeedTestConfig(window_size)
        self.model = _FeedTestModel()
        self.token_count = 0
        self.encoder: Any = None
        self.train_step_calls: list[dict[str, Any]] = []
        self._train_step_raises = train_step_raises

    def train_step(
        self,
        pattern: Any,
        *,
        raw_window: str,
        allow_sleep_maintenance: bool = False,
    ) -> dict[str, Any]:
        call_index = len(self.train_step_calls) + 1
        self.train_step_calls.append(
            {
                "pattern": pattern,
                "raw_window": raw_window,
                "allow_sleep_maintenance": allow_sleep_maintenance,
            }
        )
        if self._train_step_raises is not None:
            raise self._train_step_raises
        self.token_count += 1
        self.model.memory_store.slow_buffer.append(pattern)
        return {
            "winner": call_index,
            "recon_error": 0.125 * call_index,
            "memory_index": call_index - 1,
            "sleep_maintenance_deferred": 1,
        }


class _FeedTestEncoder:
    def __init__(self, windows: list[tuple[str, Any]]) -> None:
        self.windows = windows
        self.iter_calls: list[dict[str, Any]] = []

    def iter_segment_patterns(
        self,
        text: str,
        window_size: int,
        *,
        learn: bool,
        use_learned_boundaries: bool,
    ):
        self.iter_calls.append(
            {
                "text": text,
                "window_size": window_size,
                "learn": learn,
                "use_learned_boundaries": use_learned_boundaries,
            }
        )
        yield from self.windows


def _build_feed_pipeline(
    trace_dir: Path,
    *,
    train_step_raises: BaseException | None = None,
    window_count: int = 9,
) -> tuple[InteractionPipeline, dict[str, Any], _FeedTestTrainer, _FeedTestEncoder]:
    calls: dict[str, Any] = {
        "build_query_result": [],
        "observe_concepts": [],
        "plan_gaps": [],
        "apply_delayed": [],
        "record_recent_query_gap": [],
        "observe_runtime_concepts": [],
        "runtime_state_mark_mutated": 0,
        "runtime_state_mutation_summary": 0,
        "runtime_episode_payload": [],
        "persist_trace": [],
        "append_runtime_episode_trace": [],
        "service_state_snapshot": [],
    }
    episodes: list[dict[str, Any]] = []
    trainer = _FeedTestTrainer(train_step_raises=train_step_raises)
    encoder = _FeedTestEncoder(
        [
            (f"feed-window-{index}", f"feed-pattern-{index}")
            for index in range(1, max(1, int(window_count)) + 1)
        ]
    )

    def build_query_result_fn(**kwargs: Any) -> dict[str, Any]:
        calls["build_query_result"].append(kwargs)
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

    def observe_runtime_concepts_fn(**kwargs: Any) -> dict[str, Any] | None:
        calls["observe_runtime_concepts"].append(kwargs)
        return {
            "concept_count": 3,
            "observations": len(calls["observe_runtime_concepts"]),
            "top_concepts": [{"label": "cats"}],
        }

    def runtime_state_mark_mutated_fn() -> None:
        calls["runtime_state_mark_mutated"] += 1

    def runtime_state_mutation_summary_fn() -> dict[str, Any]:
        calls["runtime_state_mutation_summary"] += 1
        return {
            "dirty_state": True,
            "state_revision": 8,
        }

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

    def append_runtime_episode_trace_fn(episode: Mapping[str, Any]) -> dict[str, Any]:
        calls["append_runtime_episode_trace"].append(episode)
        episodes.append(dict(episode))
        return dict(episode)

    def service_state_snapshot_fn(*, include_replay_dataset_summary: bool = True) -> dict[str, Any]:
        calls["service_state_snapshot"].append(include_replay_dataset_summary)
        return {
            "checkpoint_path": "/tmp/test.pt",
            "dirty_state": bool(calls["runtime_state_mark_mutated"]),
            "state_revision": 8,
            "token_count": trainer.token_count,
            "concept_count": 3,
            "terminus_runtime": {
                "configured": True,
                "running": False,
            },
        }

    pipeline = InteractionPipeline(
        lock=threading.RLock(),
        trainer=trainer,
        encoder=encoder,
        build_query_result_fn=build_query_result_fn,
        observe_concepts_fn=observe_concepts_fn,
        plan_gaps_fn=plan_gaps_fn,
        apply_delayed_query_consequence_fn=apply_delayed_query_consequence_fn,
        record_recent_query_gap_fn=record_recent_query_gap_fn,
        observe_runtime_concepts_fn=observe_runtime_concepts_fn,
        runtime_state_mark_mutated_fn=runtime_state_mark_mutated_fn,
        runtime_state_mutation_summary_fn=runtime_state_mutation_summary_fn,
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        append_runtime_episode_trace_fn=append_runtime_episode_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
    )
    calls["episodes"] = episodes
    calls["trainer"] = trainer
    calls["encoder"] = encoder
    return pipeline, calls, trainer, encoder


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


class InteractionPipelineFeedTests(unittest.TestCase):
    def test_feed_orchestrates_feed_turn_and_persists_runtime_episode_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, calls, trainer, encoder = _build_feed_pipeline(trace_dir)

            result = pipeline.feed(text="cats chase mice and rest indoors.")

            self.assertIs(trainer.encoder, encoder)
            self.assertEqual(len(trainer.train_step_calls), 9)
            self.assertTrue(all(call["allow_sleep_maintenance"] is False for call in trainer.train_step_calls))
            self.assertEqual(len(calls["observe_runtime_concepts"]), 3)
            self.assertEqual(calls["observe_runtime_concepts"][0]["raw_window"], "feed-window-1")
            self.assertEqual(calls["observe_runtime_concepts"][1]["raw_window"], "feed-window-8")
            self.assertEqual(calls["observe_runtime_concepts"][2]["raw_window"], "feed-window-9")
            self.assertEqual(len(calls["build_query_result"]), 0)
            self.assertEqual(len(calls["observe_concepts"]), 0)
            self.assertEqual(len(calls["plan_gaps"]), 0)
            self.assertEqual(len(calls["apply_delayed"]), 0)
            self.assertEqual(len(calls["record_recent_query_gap"]), 0)
            self.assertEqual(calls["runtime_state_mark_mutated"], 1)
            self.assertEqual(calls["runtime_state_mutation_summary"], 1)
            self.assertFalse(calls["service_state_snapshot"][0])

            self.assertEqual(result["feed_summary"]["tokens_processed"], 9)
            self.assertEqual(result["feed_summary"]["token_count"], 9)
            self.assertEqual(result["feed_summary"]["memory_buffer_size"], 9)
            self.assertEqual(result["feed_summary"]["feed_encoding_mode"], REQUEST_FEED_ENCODING_MODE)
            self.assertEqual(result["feed_summary"]["concept_observation_mode"], "sampled")
            self.assertEqual(
                result["feed_summary"]["concept_observation_interval"],
                DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
            )
            self.assertEqual(result["feed_summary"]["concept_observations"], 3)
            self.assertEqual(result["feed_summary"]["sleep_maintenance_deferred"], 9)
            self.assertEqual(result["runtime_episode"]["operation"], "feed")
            self.assertEqual(result["runtime_episode"]["actual_output"]["summary"], "Processed 9 feed tokens.")
            self.assertEqual(result["runtime_episode"]["actual_output"]["memory_buffer_size"], 9)
            self.assertEqual(result["runtime_episode"]["actual_output"]["feed_encoding_mode"], REQUEST_FEED_ENCODING_MODE)
            self.assertEqual(result["runtime_episode"]["verification"]["status"], "verified")
            self.assertTrue(result["runtime_episode"]["verification"]["success"])
            self.assertAlmostEqual(float(result["runtime_episode"]["verification"]["confidence"]), 1.0)
            self.assertTrue(result["dirty_state"])
            self.assertEqual(result["state_revision"], 8)
            self.assertTrue(str(result["runtime_episode"]["trace_path"]).endswith(".json"))

            trace_path = trace_dir / f"{result['runtime_episode']['trace_id']}.json"
            self.assertTrue(trace_path.exists())
            stored_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_trace["operation"], "feed")
            self.assertEqual(stored_trace["state_after"]["state_revision"], 8)
            self.assertTrue(stored_trace["state_after"]["dirty_state"])
            self.assertEqual(len(calls["persist_trace"]), 1)
            self.assertEqual(len(calls["append_runtime_episode_trace"]), 1)
            self.assertEqual(len(calls["runtime_episode_payload"]), 1)

    def test_feed_runtime_actual_output_and_verification_reflect_feed_payload(self) -> None:
        payload = {
            "tokens_processed": 0,
            "token_count": 17,
            "last_winner": 2,
            "last_recon_error": 0.75,
            "memory_buffer_size": 4,
            "feed_encoding_mode": REQUEST_FEED_ENCODING_MODE,
            "concept_observation_mode": "sampled",
            "concept_observations": 0,
        }

        actual = build_feed_runtime_actual_output(payload)
        verification = build_feed_runtime_verification(payload)

        self.assertEqual(actual["summary"], "Processed 0 feed tokens.")
        self.assertEqual(actual["token_count"], 17)
        self.assertEqual(actual["memory_buffer_size"], 4)
        self.assertEqual(actual["feed_encoding_mode"], REQUEST_FEED_ENCODING_MODE)
        self.assertEqual(actual["concept_observation_mode"], "sampled")
        self.assertEqual(verification["status"], "unverified")
        self.assertFalse(verification["success"])
        self.assertAlmostEqual(float(verification["confidence"]), 0.0)

    def test_feed_failure_still_persists_error_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, calls, trainer, encoder = _build_feed_pipeline(
                trace_dir,
                train_step_raises=RuntimeError("boom"),
                window_count=1,
            )

            with self.assertRaises(RuntimeError):
                pipeline.feed(text="cats chase mice")

            self.assertIs(trainer.encoder, encoder)
            self.assertEqual(len(trainer.train_step_calls), 1)
            self.assertEqual(len(calls["observe_runtime_concepts"]), 0)
            self.assertEqual(len(calls["build_query_result"]), 0)
            self.assertEqual(calls["runtime_state_mark_mutated"], 0)
            self.assertEqual(calls["runtime_state_mutation_summary"], 0)
            self.assertEqual(len(calls["persist_trace"]), 1)
            self.assertEqual(len(calls["append_runtime_episode_trace"]), 1)
            self.assertFalse(calls["service_state_snapshot"][0])

            stored_trace = calls["persist_trace"][0]
            self.assertEqual(stored_trace["operation"], "feed")
            self.assertEqual(stored_trace["error"]["type"], "RuntimeError")
            self.assertEqual(stored_trace["error"]["message"], "boom")
            self.assertEqual(stored_trace["runtime_episode"]["status"], "failed")
            self.assertEqual(stored_trace["runtime_episode"]["failure"]["error_type"], "RuntimeError")
            self.assertEqual(stored_trace["runtime_episode"]["failure"]["message"], "boom")
            self.assertIsNone(stored_trace["runtime_episode"]["actual_output"])
            self.assertIsNone(stored_trace["runtime_episode"]["verification"])
            trace_path = trace_dir / f"{stored_trace['trace_id']}.json"
            self.assertTrue(trace_path.exists())
