from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest

from hecsn.config.model_config import HECSNConfig
from hecsn.service.replay_dataset_runner import (
    build_arg_parser as build_replay_dataset_arg_parser,
    export_replay_dataset_preview,
    main as replay_dataset_main,
)
from hecsn.service.trace_export_runner import build_arg_parser, export_runtime_trace_dataset, main
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.trainer import HECSNModel, HECSNTrainer


def _build_checkpoint(root: Path, *, metadata: dict | None = None) -> Path:
    cfg = HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=64,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )
    model = HECSNModel(cfg)
    trainer = HECSNTrainer(model, cfg)
    checkpoint_metadata = {"test_case": "trace_export_runner"}
    if metadata is not None:
        checkpoint_metadata.update(metadata)
    return save_trainer_checkpoint(root / "initial.pt", trainer, metadata=checkpoint_metadata)


def _runtime_trace_payload() -> dict:
    return {
        "episode_id": "episode-respond",
        "trace_id": "trace-respond",
        "trace_path": "C:\\private\\trace.json",
        "operation": "respond",
        "status": "succeeded",
        "created_at": "2025-01-01T00:00:00+00:00",
        "completed_at": "2025-01-01T00:00:01+00:00",
        "latency_ms": 1.25,
        "request": {
            "query_text": "cats chase mice",
            "top_k_memories": 4,
            "raw_environment": {"NVIDIA_API_KEY": "secret-value"},
        },
        "prediction": {
            "proposed_answer": "Cats chase mice.",
            "api_key": "secret-value",
        },
        "action": {"action_type": "respond"},
        "actual_output": {
            "response_text": "Cats chase mice at night.",
            "dotenv_path": "C:\\private\\.env",
        },
        "verification": {
            "status": "verified",
            "success": True,
            "password": "secret-value",
        },
        "feedback": [
            {
                "feedback_id": "fb-1",
                "created_at": "2025-01-01T00:00:02+00:00",
                "target_type": "runtime_episode",
                "target_id": "episode-respond",
                "verdict": "verified",
                "applied_status": "verified",
                "confidence": 0.9,
                "summary": "Manual evaluator verified the trace.",
                "evidence": [{"note": "reviewed", "api_key": "secret-value"}],
                "tags": ["manual"],
                "evaluator_id": "qa-1",
            }
        ],
        "corrected_output": {"response_text": "Cats chase mice.", "password": "secret-value"},
        "provenance": "verified",
    }


class TraceExportRunnerTests(unittest.TestCase):
    def test_arg_parser_accepts_checkpoint_output_limit_and_endpoint(self) -> None:
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "checkpoints\\terminus\\model.pt",
                "--output",
                "reports\\runtime_trace_examples.json",
                "--limit",
                "3",
                "--endpoint",
                "respond",
            ]
        )

        self.assertEqual(args.checkpoint, Path("checkpoints\\terminus\\model.pt"))
        self.assertEqual(args.output, Path("reports\\runtime_trace_examples.json"))
        self.assertEqual(args.limit, 3)
        self.assertEqual(args.endpoint, "respond")

    def test_export_runtime_trace_dataset_sanitizes_persisted_checkpoint_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(
                root,
                metadata={
                    "service_state": {
                        "terminus_runtime": {
                            "runtime_episode_traces": [_runtime_trace_payload()],
                            "replay_sample_history": [
                                {
                                    "schema_version": 1,
                                    "replay_sample_id": "replay-execute-1",
                                    "execution_id": "replay-execute-1",
                                    "created_at": "2025-01-01T00:00:03+00:00",
                                    "mode": "execute",
                                    "status": "recorded",
                                    "reason": "operator-gated audit execution recorded without side effects",
                                    "endpoint": "/terminus/replay-sample",
                                    "operator_id": "qa-1",
                                    "requested_count": 1,
                                    "selected_candidate_ids": ["candidate-1"],
                                    "selected_candidates": [
                                        {
                                            "candidate_id": "candidate-1",
                                            "target_type": "runtime_episode",
                                            "target_id": "episode-respond",
                                            "safety": {"audit_only": True, "not_promoted": True},
                                        }
                                    ],
                                    "safety_checks": {"passed": True},
                                    "safety_flags": {
                                        "audit_only": True,
                                        "operator_confirmed": True,
                                        "training_started": False,
                                        "sleep_started": False,
                                        "memory_verification_promoted": False,
                                        "feedback_posted": False,
                                        "digital_action_executed": False,
                                        "external_calls_made": False,
                                        "memory_mutated": False,
                                        "state_revision_mutated": False,
                                        "token_count_mutated": False,
                                        "action_history_mutated": False,
                                        "feedback_mutated": False,
                                        "not_promoted": True,
                                    },
                                    "before": {"token_count": 0, "state_revision": 0, "action_history_count": 0, "feedback_count": 0},
                                    "after": {"token_count": 0, "state_revision": 0, "action_history_count": 0, "feedback_count": 0},
                                }
                            ],
                        }
                    }
                },
            )

            dataset = export_runtime_trace_dataset(
                checkpoint,
                limit=5,
                endpoint="respond",
                trace_dir=root / "traces",
            )

        self.assertEqual(dataset["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertEqual(dataset["count"], 1)
        self.assertEqual(dataset["metadata"]["source"], "checkpoint_runtime_episode_traces")
        self.assertTrue(dataset["metadata"]["contains_examples"])
        self.assertEqual(dataset["policy_decision"]["schema_version"], 1)
        self.assertIsInstance(dataset["policy_decision"]["action"], str)
        self.assertEqual(dataset["metadata"]["policy_decision"]["action"], dataset["policy_decision"]["action"])
        self.assertEqual(dataset["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(dataset["metadata"]["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(dataset["replay_sample_summary"]["count"], 1)
        self.assertEqual(dataset["replay_sample_summary"]["mode_counts"]["execute"], 1)
        self.assertTrue(dataset["replay_sample_summary"]["safety_flags"]["audit_only"])
        self.assertFalse(dataset["replay_sample_summary"]["safety_flags"]["external_calls_made"])
        self.assertEqual(dataset["metadata"]["replay_sample_summary"]["count"], 1)
        self.assertEqual(dataset["replay_dataset_summary"]["endpoint"], "/terminus/replay-dataset/preview")
        self.assertEqual(dataset["metadata"]["replay_dataset_summary"]["endpoint"], "/terminus/replay-dataset/preview")
        self.assertEqual(dataset["replay_dataset_summary"]["latest_history_timestamp"], "2025-01-01T00:00:03+00:00")
        example = dataset["examples"][0]
        self.assertEqual(example["endpoint"], "/respond")
        self.assertEqual(example["type"], "respond")
        self.assertIn("context", example)
        self.assertIn("prediction", example)
        self.assertIn("actual_output", example)
        self.assertIn("verification", example)
        self.assertIn("feedback", example)
        self.assertIn("feedback_summary", example)
        self.assertEqual(example["feedback_summary"]["feedback_count"], 1)
        self.assertEqual(example["feedback_summary"]["verified_count"], 1)
        self.assertEqual(example["feedback"][0]["evidence"][0]["note"], "reviewed")
        self.assertEqual(example["policy_decision"]["action"], dataset["policy_decision"]["action"])
        self.assertIn("reason_codes", example["policy_decision"])
        self.assertEqual(example["replay_plan_summary"]["endpoint"], "/terminus/replay-plan")
        self.assertEqual(example["replay_sample_summary"]["count"], 1)
        self.assertTrue(example["replay_sample_summary"]["safety_flags"]["audit_only"])
        self.assertIn("provenance", example)
        exported_json = json.dumps(dataset["examples"], sort_keys=True)
        self.assertNotIn("secret-value", exported_json)
        self.assertNotIn("raw_environment", exported_json)
        self.assertNotIn("NVIDIA_API_KEY", exported_json)
        self.assertNotIn("dotenv_path", exported_json)
        self.assertNotIn("api_key", exported_json)
        self.assertNotIn("password", exported_json)
        self.assertNotIn("trace_path", exported_json)

    def test_main_writes_empty_valid_dataset_when_checkpoint_has_no_runtime_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(root)
            output = root / "exports" / "runtime_traces.json"

            exit_code = main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(output),
                    "--trace-dir",
                    str(root / "traces"),
                ]
            )
            dataset = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(dataset["export_kind"], "terminus_runtime_trace_dataset_preview")
        self.assertEqual(dataset["count"], 0)
        self.assertEqual(dataset["examples"], [])
        self.assertFalse(dataset["metadata"]["contains_examples"])
        self.assertEqual(dataset["metadata"]["empty_reason"], "checkpoint_contains_no_persisted_runtime_episode_traces")

    def test_main_can_write_json_to_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(root)
            stdout = io.StringIO()

            exit_code = main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    "-",
                    "--trace-dir",
                    str(root / "traces"),
                ],
                stdout=stdout,
            )
            dataset = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(dataset["count"], 0)

    def test_replay_dataset_runner_exports_preview_only_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(
                root,
                metadata={
                    "service_state": {
                        "terminus_runtime": {
                            "runtime_episode_traces": [_runtime_trace_payload()],
                            "replay_sample_history": [
                                {
                                    "schema_version": 1,
                                    "replay_sample_id": "replay-execute-1",
                                    "execution_id": "replay-execute-1",
                                    "created_at": "2025-01-01T00:00:03+00:00",
                                    "mode": "execute",
                                    "status": "recorded",
                                    "reason": "operator-gated audit execution recorded without side effects",
                                    "endpoint": "/terminus/replay-sample",
                                    "operator_id": "qa-1",
                                    "requested_count": 1,
                                    "selected_candidate_ids": ["candidate-1"],
                                    "selected_candidates": [
                                        {
                                            "candidate_id": "candidate-1",
                                            "target_type": "runtime_episode",
                                            "target_id": "episode-respond",
                                            "safety": {"audit_only": True, "not_promoted": True},
                                        }
                                    ],
                                    "safety_checks": {"passed": True},
                                    "safety_flags": {
                                        "audit_only": True,
                                        "operator_confirmed": True,
                                        "training_started": False,
                                        "sleep_started": False,
                                        "memory_verification_promoted": False,
                                        "feedback_posted": False,
                                        "digital_action_executed": False,
                                        "external_calls_made": False,
                                        "memory_mutated": False,
                                        "state_revision_mutated": False,
                                        "token_count_mutated": False,
                                        "action_history_mutated": False,
                                        "feedback_mutated": False,
                                        "not_promoted": True,
                                    },
                                    "before": {"token_count": 0, "state_revision": 0, "action_history_count": 0, "feedback_count": 0},
                                    "after": {"token_count": 0, "state_revision": 0, "action_history_count": 0, "feedback_count": 0},
                                }
                            ],
                        }
                    }
                },
            )

            dataset = export_replay_dataset_preview(
                checkpoint,
                limit=5,
                endpoint="respond",
                trace_dir=root / "traces",
            )

        self.assertEqual(dataset["export_kind"], "terminus_replay_dataset_preview")
        self.assertEqual(dataset["training_role"], "replay_dataset_preview_only_not_training_no_mutation")
        self.assertEqual(dataset["count"], 1)
        self.assertEqual(dataset["positive_count"], 1)
        self.assertEqual(dataset["negative_count"], 0)
        self.assertEqual(dataset["endpoint"], "/terminus/replay-dataset/preview")
        self.assertIsNotNone(dataset["latest_export_timestamp"])
        self.assertEqual(dataset["latest_history_timestamp"], "2025-01-01T00:00:03+00:00")
        self.assertEqual(dataset["metadata"]["source"], "checkpoint_runtime_episode_traces_with_replay_context")
        self.assertTrue(dataset["metadata"]["contains_items"])
        self.assertFalse(dataset["safety_flags"]["training_started"])
        self.assertFalse(dataset["safety_flags"]["memory_mutated"])
        item = dataset["items"][0]
        self.assertEqual(item["example_type"], "sft_example_preview")
        self.assertEqual(item["verification_label"], "verified")
        self.assertTrue(item["is_verified_fact"])
        self.assertEqual(item["sft_example"]["output_source"], "corrected_output")
        self.assertIsNone(item["preference_pair"])
        self.assertTrue(item["replay_sample_linkage"]["selected"])
        exported_json = json.dumps(dataset, sort_keys=True)
        self.assertNotIn("secret-value", exported_json)
        self.assertNotIn("api_key", exported_json)
        self.assertNotIn("password", exported_json)

    def test_replay_dataset_runner_main_writes_empty_valid_dataset(self) -> None:
        parser = build_replay_dataset_arg_parser()
        args = parser.parse_args(
            [
                "--checkpoint",
                "checkpoints\\terminus\\model.pt",
                "--output",
                "reports\\replay_dataset.json",
                "--limit",
                "3",
            ]
        )
        self.assertEqual(args.checkpoint, Path("checkpoints\\terminus\\model.pt"))
        self.assertEqual(args.output, Path("reports\\replay_dataset.json"))
        self.assertEqual(args.limit, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint = _build_checkpoint(root)
            stdout = io.StringIO()

            exit_code = replay_dataset_main(
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    "-",
                    "--trace-dir",
                    str(root / "traces"),
                ],
                stdout=stdout,
            )
            dataset = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(dataset["export_kind"], "terminus_replay_dataset_preview")
        self.assertEqual(dataset["count"], 0)
        self.assertEqual(dataset["items"], [])
        self.assertFalse(dataset["metadata"]["contains_items"])
        self.assertEqual(dataset["metadata"]["empty_reason"], "checkpoint_contains_no_eligible_sanitized_runtime_traces")


if __name__ == "__main__":
    unittest.main()
