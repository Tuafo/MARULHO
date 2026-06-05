from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import pytest

from marulho.evaluation.replay_training_approval import (
    ALLOWED_APPROVAL_SCOPE,
    ISOLATED_ADAPTER_TRAINING_SCOPE,
    build_replay_training_approval,
)
from marulho.evaluation.replay_training_gate import evaluate_replay_training_gate
from marulho.evaluation.replay_training_plan import build_replay_training_plan
from marulho.training.replay_adapter_experiment import (
    ADAPTER_DELTA_NAME,
    ADAPTER_MANIFEST_NAME,
    COMPARISON_REPORT_NAME,
    run_isolated_replay_adapter_experiment,
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
        "packaging_policy": {"decontamination": {"enabled": True, "blocked_terms": ["arc_agi"]}},
        "manifest": {
            "schema_version": 1,
            "bundle_hash": "bundle-hash",
            "source_preview_hash": "source-hash",
            "item_hashes": ["hash-a"],
            "excluded_hashes": [],
            "artifact_role": "preview_export_only_not_training",
        },
        "training_gate": {"status": "blocked_preview_only", "eligible_for_training": False},
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


def _approvals(bundle: dict[str, object], report: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    created = datetime(2099, 4, 30, 0, 0, tzinfo=timezone.utc)
    dry_run = build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ALLOWED_APPROVAL_SCOPE,
        created_at=created,
        expires_at=created + timedelta(days=1),
    )
    isolated = build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=ISOLATED_ADAPTER_TRAINING_SCOPE,
        created_at=created,
        expires_at=created + timedelta(days=1),
    )
    return dry_run, isolated


def _benchmark(verdict: str = "alive", *, unsafe: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "success": True,
        "endpoint_timings": [{"name": "status", "success": True, "latency_ms": 1.0}],
        "status_runtime_truth_summary": {
            "schema_version": 1,
            "verdict": verdict,
            "recommended_action": "continue_monitoring",
            "safety_flags": {
                "replay_safety": {
                    "training_started": False,
                    "memory_mutated": False,
                    "feedback_posted": False,
                    "digital_action_executed": unsafe,
                    "external_calls_made": False,
                    "sleep_started": False,
                    "eligible_for_training": False,
                }
            },
        },
    }


def _long_test(verdict: str = "alive") -> dict[str, object]:
    return {"health_verdict": verdict, "total_errors": 0, "total_readouts": 1}


def test_isolated_experiment_writes_artifact_and_comparison() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    dry_run_approval, isolated_approval = _approvals(bundle, report)
    plan = build_replay_training_plan(bundle, report, dry_run_approval)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "artifacts" / "replay_training" / "isolated" / "adapter-a"
        manifest = run_isolated_replay_adapter_experiment(
            bundle=bundle,
            gate_report=report,
            training_plan=plan,
            approval=isolated_approval,
            before_benchmark=_benchmark("partial"),
            after_benchmark=_benchmark("alive"),
            before_long_test=_long_test("alive"),
            after_long_test=_long_test("alive"),
            output_dir=output_dir,
            repo_root=Path(tmpdir),
        )
        written = sorted(path.name for path in output_dir.iterdir())
        comparison = json.loads((output_dir / COMPARISON_REPORT_NAME).read_text(encoding="utf-8"))

    assert written == [ADAPTER_DELTA_NAME, ADAPTER_MANIFEST_NAME, COMPARISON_REPORT_NAME]
    assert manifest["status"] == "completed_isolated_artifact_no_production_switch"
    assert manifest["adapter"]["production_runtime_switched"] is False
    assert manifest["side_effects"]["training_started"] is True
    assert manifest["side_effects"]["memory_mutated"] is False
    assert manifest["side_effects"]["digital_action_executed"] is False
    assert comparison["status"] == "passed"
    assert comparison["claimed_improvement"]["claimed"] is False


def test_isolated_experiment_refuses_dry_run_approval_scope() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    dry_run_approval, _isolated_approval = _approvals(bundle, report)
    plan = build_replay_training_plan(bundle, report, dry_run_approval)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="approval scope"):
            run_isolated_replay_adapter_experiment(
                bundle=bundle,
                gate_report=report,
                training_plan=plan,
                approval=dry_run_approval,
                before_benchmark=_benchmark(),
                after_benchmark=_benchmark(),
                before_long_test=_long_test(),
                after_long_test=_long_test(),
                output_dir=Path(tmpdir) / "isolated" / "adapter-a",
                repo_root=Path(tmpdir),
            )


def test_isolated_experiment_refuses_non_isolated_output_dir() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    dry_run_approval, isolated_approval = _approvals(bundle, report)
    plan = build_replay_training_plan(bundle, report, dry_run_approval)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="isolated"):
            run_isolated_replay_adapter_experiment(
                bundle=bundle,
                gate_report=report,
                training_plan=plan,
                approval=isolated_approval,
                before_benchmark=_benchmark(),
                after_benchmark=_benchmark(),
                before_long_test=_long_test(),
                after_long_test=_long_test(),
                output_dir=Path(tmpdir) / "production" / "adapter-a",
                repo_root=Path(tmpdir),
            )


def test_isolated_experiment_records_failed_safety_comparison() -> None:
    bundle = _bundle()
    report = _gate_report(bundle)
    dry_run_approval, isolated_approval = _approvals(bundle, report)
    plan = build_replay_training_plan(bundle, report, dry_run_approval)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "isolated" / "adapter-a"
        run_isolated_replay_adapter_experiment(
            bundle=bundle,
            gate_report=report,
            training_plan=plan,
            approval=isolated_approval,
            before_benchmark=_benchmark("alive"),
            after_benchmark=_benchmark("degraded", unsafe=True),
            before_long_test=_long_test("alive"),
            after_long_test=_long_test("failed"),
            output_dir=output_dir,
            repo_root=Path(tmpdir),
        )
        comparison = json.loads((output_dir / COMPARISON_REPORT_NAME).read_text(encoding="utf-8"))

    assert comparison["status"] == "failed"
    assert comparison["runtime_truth"]["regressed"] is True
    assert comparison["unsafe_flag_increases"]["digital_action_executed"]["after"] is True
