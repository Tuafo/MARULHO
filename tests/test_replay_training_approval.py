from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import pytest

from hecsn.evaluation.replay_training_approval import (
    ALLOWED_APPROVAL_SCOPE,
    build_replay_training_approval,
    create_replay_training_approval_file,
    validate_replay_training_approval,
)
from hecsn.evaluation.replay_training_gate import evaluate_replay_training_gate


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


def _approval(
    bundle: dict[str, object],
    report: dict[str, object],
    *,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> dict[str, object]:
    created = created_at or datetime(2099, 4, 30, 0, 0, tzinfo=timezone.utc)
    return build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ALLOWED_APPROVAL_SCOPE,
        created_at=created,
        expires_at=expires_at or created + timedelta(hours=2),
    )


def test_valid_approval_writes_deterministic_artifact() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    created = datetime(2026, 4, 30, 1, 2, 3, tzinfo=timezone.utc)

    first = build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ALLOWED_APPROVAL_SCOPE,
        created_at=created,
        expires_at=created + timedelta(hours=1),
    )
    second = build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ALLOWED_APPROVAL_SCOPE,
        created_at=created,
        expires_at=created + timedelta(hours=1),
    )

    assert first == second
    assert first["scope"] == ALLOWED_APPROVAL_SCOPE
    assert first["read_only"] is True
    assert first["training_started"] is False
    assert first["memory_mutated"] is False
    assert first["adapter_created"] is False


def test_wrong_bundle_hash_fails() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    approval = _approval(bundle, report)
    tampered_bundle = _bundle()
    tampered_bundle["count"] = 2

    with pytest.raises(ValueError, match="bundle hash"):
        validate_replay_training_approval(approval, tampered_bundle, report)


def test_wrong_gate_report_hash_fails() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    approval = _approval(bundle, report)
    tampered_report = dict(report)
    tampered_report["passed_checks"] = 0

    with pytest.raises(ValueError, match="gate report hash"):
        validate_replay_training_approval(approval, bundle, tampered_report)


def test_expired_approval_fails() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    created = datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc)
    approval = _approval(bundle, report, created_at=created, expires_at=created + timedelta(minutes=5))

    with pytest.raises(ValueError, match="expired"):
        validate_replay_training_approval(
            approval,
            bundle,
            report,
            now=created + timedelta(minutes=6),
        )


def test_missing_operator_id_fails() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)

    with pytest.raises(ValueError, match="operator_id"):
        build_replay_training_approval(
            bundle,
            report,
            operator_id=" ",
            scope=ALLOWED_APPROVAL_SCOPE,
            created_at=datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc),
        )


def test_unsafe_scope_fails() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)

    with pytest.raises(ValueError, match="approval scope"):
        build_replay_training_approval(
            bundle,
            report,
            operator_id="operator-a",
            scope="production_model_switch",
            created_at=datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc),
        )


def test_approval_file_has_no_runtime_side_effects() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        bundle_path = root / "bundle.json"
        report_path = root / "gate.json"
        approval_path = root / "approval.json"
        bundle = _bundle()
        report = _gate_report(bundle)
        bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
        report_path.write_text(json.dumps(report), encoding="utf-8")

        before = sorted(path.name for path in root.iterdir())
        approval = create_replay_training_approval_file(
            bundle_path,
            report_path,
            operator_id="operator-a",
            scope=ALLOWED_APPROVAL_SCOPE,
            output_path=approval_path,
            created_at=datetime(2026, 4, 30, 0, 0, tzinfo=timezone.utc),
        )
        after = sorted(path.name for path in root.iterdir())
        readme_text = (root / "README.md").read_text(encoding="utf-8")

    assert before == ["bundle.json", "gate.json"]
    assert after == ["README.md", "approval.json", "bundle.json", "gate.json"]
    assert "Replay Training Operator Approval" in readme_text
    assert approval["feedback_posted"] is False
    assert approval["digital_action_executed"] is False
    assert approval["external_calls_made"] is False
    assert approval["sleep_started"] is False
