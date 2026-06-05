"""Direct test surface for the Interaction Pipeline seam."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import unittest
from typing import Any, Mapping

from marulho.service.interaction_pipeline import (
    DEFAULT_FEED_CONCEPT_OBSERVATION_INTERVAL,
    REQUEST_FEED_ENCODING_MODE,
    InteractionPipeline,
    build_feed_runtime_actual_output,
    build_feed_runtime_verification,
    build_respond_runtime_actual_output,
    build_respond_runtime_verification,
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
        "runtime_episode_payload": [],
        "persist_trace": [],
        "service_state_snapshot": [],
    }

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
        observe_runtime_concepts_fn=lambda **kwargs: None,
        runtime_state_mark_mutated_fn=lambda: None,
        runtime_state_mutation_summary_fn=lambda: {"dirty_state": False, "state_revision": 0},
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
    )
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
        "observe_runtime_concepts": [],
        "runtime_state_mark_mutated": 0,
        "runtime_state_mutation_summary": 0,
        "runtime_episode_payload": [],
        "persist_trace": [],
        "service_state_snapshot": [],
    }
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
        observe_runtime_concepts_fn=observe_runtime_concepts_fn,
        runtime_state_mark_mutated_fn=runtime_state_mark_mutated_fn,
        runtime_state_mutation_summary_fn=runtime_state_mutation_summary_fn,
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
    )
    calls["trainer"] = trainer
    calls["encoder"] = encoder
    return pipeline, calls, trainer, encoder


def _build_respond_pipeline(
    trace_dir: Path,
    *,
    build_query_raises: BaseException | None = None,
    build_response_outputs: list[dict[str, Any]] | None = None,
    action_assist_result: dict[str, Any] | None = None,
    response_outcome_score: float = 0.84,
    background_provenance_applied: bool = False,
    delayed_candidate: dict[str, Any] | None = None,
    learn_mode_raises: BaseException | None = None,
) -> tuple[InteractionPipeline, dict[str, Any], _FeedTestTrainer, _FeedTestEncoder]:
    calls: dict[str, Any] = {
        "build_query_result": [],
        "observe_concepts": [],
        "plan_gaps": [],
        "apply_delayed": [],
        "build_response": [],
        "maybe_auto_action_assist": [],
        "response_grounded_outcome_score": [],
        "apply_background_source_response_provenance": [],
        "apply_background_source_outcome_calibration": [],
        "apply_provider_response_outcome_calibration": [],
        "learn_from_turn": [],
        "record_response_consequence_candidate": [],
        "runtime_state_mark_mutated": 0,
        "runtime_state_mutation_summary": 0,
        "runtime_episode_payload": [],
        "persist_trace": [],
        "service_state_snapshot": [],
    }
    trainer = _FeedTestTrainer()
    encoder = _FeedTestEncoder([])
    response_outputs = list(build_response_outputs or [
        {
            "response_mode": "grounded_synthesis",
            "response_text": "Grounded answer: cats chase mice at night.",
            "support_score": 0.84,
            "evidence_coverage": 0.72,
            "unsupported_terms": [],
            "selected_evidence": [
                {
                    "text": "cats chase mice at night",
                    "source_name": "notes.md",
                    "source_names": ["notes.md"],
                    "provider": "web",
                    "providers": ["web"],
                }
            ],
            "concept_grounding": {"top_concepts": [{"label": "cats"}]},
            "native_decode": None,
        }
    ])
    response_call_count = {"count": 0}

    def build_query_result_fn(**kwargs: Any) -> dict[str, Any]:
        calls["build_query_result"].append(kwargs)
        if build_query_raises is not None:
            raise build_query_raises
        return {
            "query_summary": {
                "query_text": kwargs["query_text"],
                "top_candidates": [{"candidate_id": "cand-1"}],
                "memory_matches": [
                    {"text": "cats chase mice at night", "similarity": 0.93},
                    {"text": "cats rest indoors", "similarity": 0.76},
                ],
                "memory_episodes": [{"episode_id": "episode-a", "text": "cats chase mice at night"}],
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
            "grounded_fraction": 0.58,
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

    def build_response_fn(**kwargs: Any) -> dict[str, Any]:
        calls["build_response"].append(kwargs)
        index = min(response_call_count["count"], len(response_outputs) - 1)
        response_call_count["count"] += 1
        return deepcopy(response_outputs[index])

    def maybe_auto_action_assist_fn(**kwargs: Any) -> dict[str, Any] | None:
        calls["maybe_auto_action_assist"].append(kwargs)
        if action_assist_result is None:
            return None
        return deepcopy(action_assist_result)

    def response_grounded_outcome_score_fn(**kwargs: Any) -> float:
        calls["response_grounded_outcome_score"].append(kwargs)
        return response_outcome_score

    def apply_background_source_response_provenance_fn(**kwargs: Any) -> bool:
        calls["apply_background_source_response_provenance"].append(kwargs)
        return background_provenance_applied

    def apply_background_source_outcome_calibration_fn(**kwargs: Any) -> None:
        calls["apply_background_source_outcome_calibration"].append(kwargs)

    def apply_provider_response_outcome_calibration_fn(**kwargs: Any) -> bool:
        calls["apply_provider_response_outcome_calibration"].append(kwargs)
        return False

    def runtime_state_mark_mutated_fn() -> None:
        calls["runtime_state_mark_mutated"] += 1

    def runtime_state_mutation_summary_fn() -> dict[str, Any]:
        calls["runtime_state_mutation_summary"] += 1
        mutated = bool(calls["runtime_state_mark_mutated"])
        return {
            "dirty_state": mutated,
            "state_revision": 8 if mutated else 7,
        }

    def learn_from_turn_fn(**kwargs: Any) -> dict[str, Any] | None:
        calls["learn_from_turn"].append(kwargs)
        if learn_mode_raises is not None:
            raise learn_mode_raises
        if kwargs["learn_mode"] == "none":
            return None
        runtime_state_mark_mutated_fn()
        selected_evidence = list(kwargs.get("response", {}).get("selected_evidence") or [])
        return {
            "learn_mode": kwargs["learn_mode"],
            "user_feed": {
                "text": kwargs["query_text"],
                "tokens_processed": 1,
            },
            "evidence_feed": None,
            "selected_evidence_count": len(selected_evidence),
        }

    def record_response_consequence_candidate_fn(**kwargs: Any) -> dict[str, Any] | None:
        calls["record_response_consequence_candidate"].append(kwargs)
        if delayed_candidate is None:
            return None
        return deepcopy(delayed_candidate)

    def runtime_episode_payload_fn(**kwargs: Any) -> dict[str, Any]:
        calls["runtime_episode_payload"].append(kwargs)
        payload = {
            "episode_id": "episode-respond-1",
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

    def service_state_snapshot_fn(*, include_replay_dataset_summary: bool = True) -> dict[str, Any]:
        calls["service_state_snapshot"].append(include_replay_dataset_summary)
        mutated = bool(calls["runtime_state_mark_mutated"])
        return {
            "checkpoint_path": "/tmp/test.pt",
            "dirty_state": mutated,
            "state_revision": 8 if mutated else 7,
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
        observe_runtime_concepts_fn=lambda **kwargs: None,
        runtime_state_mark_mutated_fn=runtime_state_mark_mutated_fn,
        runtime_state_mutation_summary_fn=runtime_state_mutation_summary_fn,
        runtime_episode_payload_fn=runtime_episode_payload_fn,
        persist_trace_fn=persist_trace_fn,
        service_state_snapshot_fn=service_state_snapshot_fn,
        build_response_fn=build_response_fn,
        maybe_auto_action_assist_fn=maybe_auto_action_assist_fn,
        response_grounded_outcome_score_fn=response_grounded_outcome_score_fn,
        apply_background_source_response_provenance_fn=apply_background_source_response_provenance_fn,
        apply_background_source_outcome_calibration_fn=apply_background_source_outcome_calibration_fn,
        apply_provider_response_outcome_calibration_fn=apply_provider_response_outcome_calibration_fn,
        learn_from_turn_fn=learn_from_turn_fn,
        record_response_consequence_candidate_fn=record_response_consequence_candidate_fn,
    )
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
            recent_query_gaps = pipeline.recent_query_gaps()
            self.assertEqual(len(recent_query_gaps), 1)
            self.assertEqual(recent_query_gaps[0]["source"], "query")
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
            self.assertEqual(len(pipeline.runtime_episode_traces()), 1)

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
            self.assertEqual(len(pipeline.runtime_episode_traces()), 1)
            stored_trace = calls["persist_trace"][0]
            self.assertEqual(stored_trace["operation"], "query")
            self.assertEqual(stored_trace["error"]["type"], "RuntimeError")
            self.assertEqual(stored_trace["error"]["message"], "boom")
            self.assertEqual(stored_trace["runtime_episode"]["status"], "failed")
            self.assertEqual(stored_trace["runtime_episode"]["failure"]["error_type"], "RuntimeError")
            self.assertFalse(calls["service_state_snapshot"][0])


class InteractionPipelineTraceSeamTests(unittest.TestCase):
    def test_runtime_episode_trace_read_and_replace_use_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, _calls = _build_pipeline(trace_dir)
            pipeline.load_interaction_state(
                runtime_episode_traces=[
                    {
                        "episode_id": "episode-1",
                        "trace_id": "trace-1",
                        "operation": "query",
                        "feedback": [],
                        "verification": {"status": "verified"},
                    }
                ]
            )

            episode = pipeline.runtime_episode_trace("episode-1")
            self.assertIsNotNone(episode)
            assert episode is not None
            self.assertEqual(episode["episode_id"], "episode-1")
            episode["feedback"].append({"feedback_id": "fb-1"})
            self.assertEqual(pipeline.runtime_episode_traces()[0]["feedback"], [])

            replaced = pipeline.replace_runtime_episode_trace("episode-1", episode)
            self.assertIsNotNone(replaced)
            assert replaced is not None
            self.assertEqual(replaced["feedback"][0]["feedback_id"], "fb-1")
            self.assertEqual(pipeline.runtime_episode_trace("episode-1")["feedback"][0]["feedback_id"], "fb-1")
            self.assertEqual(pipeline.runtime_episode_traces()[0]["feedback"][0]["feedback_id"], "fb-1")


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
            self.assertEqual(len(pipeline.runtime_episode_traces()), 1)
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
            self.assertEqual(len(pipeline.runtime_episode_traces()), 1)
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


class InteractionPipelineRespondTests(unittest.TestCase):
    def test_respond_runtime_actual_output_and_verification_reflect_response_payload(self) -> None:
        response = {
            "response_mode": "grounded_synthesis",
            "response_text": "Grounded answer: cats chase mice at night.",
            "support_score": 0.84,
            "evidence_coverage": 0.72,
            "unsupported_terms": ["mice"],
            "selected_evidence": [
                {
                    "text": "cats chase mice at night",
                    "source_name": "notes.md",
                    "provider": "web",
                }
            ],
        }
        action_assist = {
            "triggered": True,
            "executed": False,
            "reused_recent_action": True,
            "reason": "recent_verified_action",
            "used_in_response": True,
            "result": {
                "action_type": "workspace_read",
                "action_id": "action-1",
                "verification": {
                    "success": True,
                    "contradiction": False,
                    "confidence": 0.91,
                    "summary": "Action verification supplied grounded evidence.",
                },
            },
        }

        actual = build_respond_runtime_actual_output(
            response=response,
            action_assist=action_assist,
            outcome_score=0.73,
        )
        verification = build_respond_runtime_verification(
            response=response,
            action_assist=action_assist,
            outcome_score=0.73,
        )

        self.assertEqual(actual["summary"], "Grounded answer: cats chase mice at night.")
        self.assertEqual(actual["selected_evidence_count"], 1)
        self.assertEqual(actual["action_assist"]["reason"], "recent_verified_action")
        self.assertTrue(actual["action_assist"]["reused_recent_action"])
        self.assertEqual(actual["action_assist"]["action_type"], "workspace_read")
        self.assertEqual(actual["action_assist"]["action_id"], "action-1")
        self.assertEqual(verification["status"], "verified")
        self.assertTrue(verification["success"])
        self.assertAlmostEqual(float(verification["confidence"]), 0.91)

    def test_respond_orchestrates_response_turn_and_persists_runtime_episode_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            delayed_candidate = {
                "candidate_id": "response-1",
                "summary": "Delayed consequence candidate",
            }
            pipeline, calls, _trainer, _encoder = _build_respond_pipeline(
                trace_dir,
                delayed_candidate=delayed_candidate,
            )

            result = pipeline.respond(
                query_text="What do cats chase at night?",
                context_text="shared context",
                top_k_candidates=4,
                top_k_memories=2,
                top_chars=9,
                max_evidence_items=2,
                learn_mode="user_only",
            )

            self.assertEqual(len(calls["build_query_result"]), 1)
            self.assertEqual(len(calls["build_response"]), 1)
            self.assertEqual(len(calls["maybe_auto_action_assist"]), 1)
            self.assertEqual(len(calls["response_grounded_outcome_score"]), 1)
            self.assertEqual(len(calls["apply_background_source_response_provenance"]), 1)
            self.assertEqual(len(calls["apply_background_source_outcome_calibration"]), 1)
            self.assertEqual(len(calls["apply_provider_response_outcome_calibration"]), 1)
            self.assertEqual(len(calls["learn_from_turn"]), 1)
            self.assertEqual(len(calls["record_response_consequence_candidate"]), 1)
            self.assertEqual(pipeline.recent_query_gaps()[0]["source"], "respond")
            self.assertFalse(calls["service_state_snapshot"][0])
            build_kwargs = calls["build_query_result"][0]
            self.assertEqual(build_kwargs["query_text"], "What do cats chase at night?")
            self.assertEqual(build_kwargs["context_text"], "shared context")
            self.assertEqual(build_kwargs["top_k_candidates"], 4)
            self.assertEqual(build_kwargs["top_k_memories"], 2)
            self.assertEqual(build_kwargs["top_chars"], 9)

            self.assertEqual(result["response"]["response_text"], "Grounded answer: cats chase mice at night.")
            self.assertEqual(result["response"]["delayed_consequence_candidate"]["candidate_id"], "response-1")
            self.assertEqual(result["learning"]["learn_mode"], "user_only")
            self.assertTrue(result["dirty_state"])
            self.assertEqual(result["state_revision"], 8)
            self.assertEqual(result["runtime_episode"]["operation"], "respond")
            self.assertEqual(result["runtime_episode"]["actual_output"]["summary"], "Grounded answer: cats chase mice at night.")
            self.assertEqual(result["runtime_episode"]["verification"]["status"], "verified")
            self.assertAlmostEqual(float(result["runtime_episode"]["verification"]["confidence"]), 0.84)
            self.assertTrue(str(result["runtime_episode"]["trace_path"]).endswith(".json"))

            trace_path = trace_dir / f"{result['runtime_episode']['trace_id']}.json"
            self.assertTrue(trace_path.exists())
            stored_trace = json.loads(trace_path.read_text(encoding="utf-8"))
            self.assertEqual(stored_trace["operation"], "respond")
            self.assertEqual(stored_trace["query_result"]["delayed_consequence"]["matched_records"], 1)
            self.assertEqual(stored_trace["response"]["delayed_consequence_candidate"]["candidate_id"], "response-1")
            self.assertEqual(len(pipeline.runtime_episode_traces()), 1)

    def test_respond_reuses_recent_verified_action_and_appends_response_note(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            pipeline, calls, _trainer, _encoder = _build_respond_pipeline(
                trace_dir,
                build_response_outputs=[
                    {
                        "response_mode": "grounded_synthesis",
                        "response_text": "Grounded answer: cats chase mice at night.",
                        "support_score": 0.84,
                        "evidence_coverage": 0.72,
                        "unsupported_terms": [],
                        "selected_evidence": [{"text": "cats chase mice at night"}],
                    },
                    {
                        "response_mode": "grounded_synthesis",
                        "response_text": "Rebuilt grounded answer: cats chase mice at night.",
                        "support_score": 0.84,
                        "evidence_coverage": 0.72,
                        "unsupported_terms": [],
                        "selected_evidence": [{"text": "cats chase mice at night"}],
                    },
                ],
                action_assist_result={
                    "triggered": True,
                    "executed": False,
                    "reused_recent_action": True,
                    "reason": "recent_verified_action",
                    "used_in_response": False,
                    "response_episode_count": 1,
                    "response_note": " I verified the workspace evidence.",
                    "result": {
                        "action_type": "workspace_read",
                        "action_id": "action-1",
                        "predicted_outcome": "read notes.md",
                        "verification": {
                            "success": True,
                            "contradiction": False,
                            "confidence": 0.91,
                            "summary": "Action verification supplied grounded evidence.",
                        },
                    },
                },
            )

            result = pipeline.respond(
                query_text="What do cats chase at night?",
                learn_mode="none",
            )

            self.assertEqual(len(calls["build_response"]), 2)
            self.assertEqual(result["query_result"]["action_assist"]["reason"], "recent_verified_action")
            self.assertTrue(result["query_result"]["action_assist"]["reused_recent_action"])
            self.assertTrue(result["response"]["action_assist"]["used_in_response"])
            self.assertEqual(result["response"]["action_assist"]["result"]["action_type"], "workspace_read")
            self.assertEqual(result["runtime_episode"]["action"]["action_assist"]["action_type"], "workspace_read")
            self.assertEqual(result["runtime_episode"]["action"]["proposed_action"], "read notes.md")
            self.assertIn("I verified the workspace evidence.", result["response"]["response_text"])
            self.assertFalse(result["dirty_state"])
            self.assertEqual(result["state_revision"], 7)
            self.assertEqual(result["runtime_episode"]["verification"]["status"], "verified")
            self.assertEqual(result["runtime_episode"]["action"]["action_assist"]["reason"], "recent_verified_action")

    def test_respond_auto_executes_workspace_web_and_api_action_assist(self) -> None:
        cases = [
            {
                "reason": "query_gap_auto_read",
                "action_type": "workspace_read",
                "action_id": "workspace-1",
                "predicted_outcome": "read notes.md",
            },
            {
                "reason": "query_gap_auto_fetch",
                "action_type": "web_fetch",
                "action_id": "web-1",
                "predicted_outcome": "fetch page.html",
            },
            {
                "reason": "query_gap_auto_api_request",
                "action_type": "api_request",
                "action_id": "api-1",
                "predicted_outcome": "request data.json",
            },
        ]
        for case in cases:
            with self.subTest(action_type=case["action_type"]):
                with tempfile.TemporaryDirectory() as tmpdir:
                    trace_dir = Path(tmpdir) / "traces"
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    pipeline, calls, _trainer, _encoder = _build_respond_pipeline(
                        trace_dir,
                        build_response_outputs=[
                            {
                                "response_mode": "grounded_synthesis",
                                "response_text": "Initial grounded answer: cats chase mice at night.",
                                "support_score": 0.84,
                                "evidence_coverage": 0.72,
                                "unsupported_terms": [],
                                "selected_evidence": [{"text": "cats chase mice at night"}],
                            },
                            {
                                "response_mode": "grounded_synthesis",
                                "response_text": "Rebuilt grounded answer: cats chase mice at night.",
                                "support_score": 0.84,
                                "evidence_coverage": 0.72,
                                "unsupported_terms": [],
                                "selected_evidence": [{"text": "cats chase mice at night"}],
                            },
                        ],
                        action_assist_result={
                            "triggered": True,
                            "executed": True,
                            "reused_recent_action": False,
                            "reason": case["reason"],
                            "used_in_response": False,
                            "response_episode_count": 1,
                            "result": {
                                "action_type": case["action_type"],
                                "action_id": case["action_id"],
                                "predicted_outcome": case["predicted_outcome"],
                                "verification": {
                                    "success": True,
                                    "contradiction": False,
                                    "confidence": 0.88,
                                    "summary": "Action verification supplied grounded evidence.",
                                },
                            },
                        },
                    )

                    result = pipeline.respond(
                        query_text="What do cats chase at night?",
                        learn_mode="none",
                    )

                    self.assertEqual(len(calls["build_response"]), 2)
                    self.assertEqual(result["query_result"]["action_assist"]["reason"], case["reason"])
                    self.assertTrue(result["query_result"]["action_assist"]["executed"])
                    self.assertTrue(result["response"]["action_assist"]["used_in_response"])
                    self.assertEqual(result["response"]["action_assist"]["result"]["action_type"], case["action_type"])
                    self.assertEqual(result["runtime_episode"]["action"]["action_assist"]["action_type"], case["action_type"])
                    self.assertEqual(result["runtime_episode"]["action"]["proposed_action"], case["predicted_outcome"])
                    self.assertEqual(result["runtime_episode"]["verification"]["status"], "verified")
                    self.assertEqual(result["response"]["response_text"], "Rebuilt grounded answer: cats chase mice at night.")
