from __future__ import annotations

import json
from pathlib import Path
import tempfile

from hecsn.evaluation.replay_training_gate import (
    PASSING_STATUS,
    evaluate_replay_training_gate,
    evaluate_replay_training_gate_file,
)


def _bundle() -> dict[str, object]:
    item = {
        "package_item_id": "bundle-item-a",
        "dedupe_fingerprint": "hash-a",
        "split": "train",
        "target_id": "episode-a",
        "example_type": "positive_preference",
        "safety_flags": {
            "preview_only": True,
            "training_started": False,
            "eligible_for_training": False,
        },
    }
    return {
        "schema_version": 1,
        "export_kind": "terminus_replay_dataset_bundle_preview",
        "bundle_id": "terminus-replay-dataset-bundle-v1-test",
        "bundle_version": "v1.test",
        "bundle_hash": "bundle-hash",
        "source_preview_hash": "source-hash",
        "count": 1,
        "split_counts": {"train": 1, "holdout": 0, "eval": 0},
        "packaging_policy": {
            "decontamination": {
                "enabled": True,
                "blocked_terms": ["arc_agi", "benchmark", "heldout"],
            }
        },
        "manifest": {
            "schema_version": 1,
            "bundle_hash": "bundle-hash",
            "source_preview_hash": "source-hash",
            "item_hashes": ["hash-a"],
            "excluded_hashes": [],
            "artifact_role": "preview_export_only_not_training",
        },
        "training_gate": {
            "schema_version": 1,
            "gate_name": "offline_replay_to_learning_gate",
            "status": "blocked_preview_only",
            "eligible_for_training": False,
        },
        "splits": {"train": [item], "holdout": [], "eval": []},
        "excluded_items": [],
        "safety_flags": {
            "training_started": False,
            "memory_mutated": False,
            "feedback_posted": False,
            "digital_action_executed": False,
            "external_calls_made": False,
            "sleep_started": False,
            "eligible_for_training": False,
        },
    }


def test_replay_training_gate_passes_but_remains_not_trainable() -> None:
    report = evaluate_replay_training_gate(_bundle())

    assert report["status"] == PASSING_STATUS
    assert report["passed"] is True
    assert report["eligible_for_training"] is False
    assert report["training_started"] is False
    assert report["satisfied_conditions"]["offline_regression_benchmark"] is True
    assert report["satisfied_conditions"]["explicit_operator_training_approval"] is False
    assert report["unsatisfied_conditions"] == ["explicit_operator_training_approval"]
    assert report["next_action"] == "request_explicit_operator_training_approval"


def test_replay_training_gate_fails_decontamination_match() -> None:
    bundle = _bundle()
    train = bundle["splits"]["train"]  # type: ignore[index]
    train[0]["source_text"] = "This item leaks arc_agi benchmark content."  # type: ignore[index]

    report = evaluate_replay_training_gate(bundle)

    assert report["passed"] is False
    assert report["eligible_for_training"] is False
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert "decontamination" in failed
    assert "offline_regression_benchmark" in report["unsatisfied_conditions"]


def test_replay_training_gate_file_writes_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "bundle.json"
        output_path = Path(tmpdir) / "gate.json"
        input_path.write_text(json.dumps(_bundle()), encoding="utf-8")

        report = evaluate_replay_training_gate_file(input_path, output_path=output_path)
        loaded = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["passed"] is True
    assert loaded["status"] == PASSING_STATUS
    assert loaded["input_path"].endswith("bundle.json")
