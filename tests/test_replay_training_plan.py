from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import pytest

from hecsn.evaluation.replay_training_approval import (
    ALLOWED_APPROVAL_SCOPE,
    build_replay_training_approval,
)
from hecsn.evaluation.replay_training_gate import evaluate_replay_training_gate
from hecsn.evaluation.replay_training_plan import (
    build_replay_training_plan,
    create_replay_training_plan_file,
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


def _gate_report(bundle: dict[str, object]) -> dict[str, object]:
    report = evaluate_replay_training_gate(bundle)
    report["generated_at"] = "2026-04-30T00:00:00+00:00"
    return report


def _approval(bundle: dict[str, object], report: dict[str, object]) -> dict[str, object]:
    created = datetime(2099, 4, 30, 0, 0, tzinfo=timezone.utc)
    return build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ALLOWED_APPROVAL_SCOPE,
        created_at=created,
        expires_at=created + timedelta(days=1),
    )


def test_refuses_missing_approval() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundle = _bundle()
        report = _gate_report(bundle)
        bundle_path = root / "bundle.json"
        report_path = root / "gate.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")

        with pytest.raises(FileNotFoundError):
            create_replay_training_plan_file(
                bundle_path,
                report_path,
                root / "missing-approval.json",
                output_path=root / "plan.json",
            )


def test_refuses_mismatched_hashes() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    approval = _approval(bundle, report)
    tampered_bundle = _bundle()
    tampered_bundle["count"] = 2

    with pytest.raises(ValueError, match="bundle hash"):
        build_replay_training_plan(tampered_bundle, report, approval)


def test_refuses_approval_scope_other_than_dry_run() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    approval = _approval(bundle, report)
    approval["scope"] = "isolated_adapter_training"

    with pytest.raises(ValueError, match="approval scope"):
        build_replay_training_plan(bundle, report, approval)


def test_writes_stable_plan_output() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    approval = _approval(bundle, report)

    first = build_replay_training_plan(bundle, report, approval)
    second = build_replay_training_plan(bundle, report, approval)

    assert first == second
    assert first["status"] == "dry_run_plan_only_not_executable"
    assert first["split_counts"] == {"train": 1, "holdout": 0, "eval": 0}
    assert first["target_adapter"]["create_weights_in_this_phase"] is False
    assert first["side_effects"]["adapter_created"] is False
    assert first["executable"] is False


def test_plan_file_has_no_runtime_side_effects() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundle = _bundle()
        report = _gate_report(bundle)
        approval = _approval(bundle, report)
        bundle_path = root / "bundle.json"
        report_path = root / "gate.json"
        approval_path = root / "approval.json"
        plan_path = root / "training_plan.json"
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")
        approval_path.write_text(json.dumps(approval), encoding="utf-8")

        before = sorted(path.name for path in root.iterdir())
        plan = create_replay_training_plan_file(bundle_path, report_path, approval_path, output_path=plan_path)
        after = sorted(path.name for path in root.iterdir())
        readme_text = (root / "README.md").read_text(encoding="utf-8")

    assert before == ["approval.json", "bundle.json", "gate.json"]
    assert after == ["README.md", "approval.json", "bundle.json", "gate.json", "training_plan.json"]
    assert "Replay Training Dry-Run Plan" in readme_text
    assert plan["side_effects"]["training_started"] is False
    assert plan["side_effects"]["memory_mutated"] is False
    assert plan["side_effects"]["feedback_posted"] is False
    assert plan["side_effects"]["digital_action_executed"] is False
    assert plan["side_effects"]["sleep_started"] is False
