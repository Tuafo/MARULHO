from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile

import pytest

from marulho.evaluation.replay_adapter_promotion_gate import (
    evaluate_replay_adapter_promotion_gate,
    evaluate_replay_adapter_promotion_gate_files,
)
from marulho.evaluation.replay_training_approval import (
    ALLOWED_APPROVAL_SCOPE,
    EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
    ISOLATED_ADAPTER_TRAINING_SCOPE,
    build_replay_training_approval,
)
from marulho.evaluation.replay_training_gate import evaluate_replay_training_gate
from marulho.evaluation.replay_training_plan import build_replay_training_plan
from marulho.training.replay_adapter_experiment import (
    ADAPTER_MANIFEST_NAME,
    COMPARISON_REPORT_NAME,
    run_isolated_replay_adapter_experiment,
)

from tests.test_replay_adapter_experiment import _benchmark, _bundle, _gate_report, _long_test


def _artifacts(tmpdir: str) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    bundle = _bundle()
    report = _gate_report(bundle)
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
    promotion = build_replay_training_approval(
        bundle,
        report,
        operator_id="operator-a",
        scope=EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
        created_at=created,
        expires_at=created + timedelta(days=1),
    )
    plan = build_replay_training_plan(bundle, report, dry_run)
    output_dir = Path(tmpdir) / "isolated" / "adapter-a"
    manifest = run_isolated_replay_adapter_experiment(
        bundle=bundle,
        gate_report=report,
        training_plan=plan,
        approval=isolated,
        before_benchmark=_benchmark("partial"),
        after_benchmark=_benchmark("alive"),
        before_long_test=_long_test("alive"),
        after_long_test=_long_test("alive"),
        output_dir=output_dir,
        repo_root=Path(tmpdir),
    )
    comparison = json.loads((output_dir / COMPARISON_REPORT_NAME).read_text(encoding="utf-8"))
    return bundle, report, manifest, comparison, promotion


def test_promotion_records_rollback_metadata_for_experimental_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _bundle_value, report, manifest, comparison, promotion = _artifacts(tmpdir)
        gate = evaluate_replay_adapter_promotion_gate(
            adapter_manifest=manifest,
            comparison_report=comparison,
            before_benchmark=_benchmark("partial"),
            after_benchmark=_benchmark("alive"),
            gate_report=report,
            approval=promotion,
            useful_behavior_note="Holdout responses became more consistent in isolated evaluation.",
        )

    assert gate["eligible_for_experimental_promotion"] is True
    assert gate["eligible_for_production_promotion"] is False
    assert gate["rollback_metadata"]["production_runtime_changed"] is False
    assert gate["rollback_metadata"]["configured_path_kind"] == "non_default_experimental_path_only"


def test_promotion_refuses_missing_reports() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _bundle_value, report, manifest, comparison, promotion = _artifacts(tmpdir)
        manifest_path = Path(tmpdir) / "isolated" / "adapter-a" / ADAPTER_MANIFEST_NAME
        approval_path = Path(tmpdir) / "approval.json"
        gate_path = Path(tmpdir) / "gate.json"
        approval_path.write_text(json.dumps(promotion), encoding="utf-8")
        gate_path.write_text(json.dumps(report), encoding="utf-8")

        with pytest.raises(ValueError, match="Before benchmark report"):
            evaluate_replay_adapter_promotion_gate_files(
                adapter_manifest_path=manifest_path,
                comparison_report_path=None,
                before_benchmark_path=None,
                after_benchmark_path=Path(tmpdir) / "after.json",
                gate_report_path=gate_path,
                approval_path=approval_path,
                output_path=Path(tmpdir) / "promotion.json",
                useful_behavior_note="Documented useful behavior.",
            )


def test_promotion_refuses_worse_safety_verdict() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _bundle_value, report, manifest, comparison, promotion = _artifacts(tmpdir)
        gate = evaluate_replay_adapter_promotion_gate(
            adapter_manifest=manifest,
            comparison_report=comparison,
            before_benchmark=_benchmark("alive"),
            after_benchmark=_benchmark("degraded", unsafe=True),
            gate_report=report,
            approval=promotion,
            useful_behavior_note="Documented useful behavior.",
        )

    assert gate["eligible_for_experimental_promotion"] is False
    assert gate["checks"]["runtime_truth_no_regression"] is False
    assert gate["checks"]["unsafe_action_or_replay_not_increased"] is False


def test_promotion_refuses_missing_operator_approval() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _bundle_value, report, manifest, comparison, promotion = _artifacts(tmpdir)
        promotion["operator_id"] = ""

        with pytest.raises(ValueError, match="operator_id"):
            evaluate_replay_adapter_promotion_gate(
                adapter_manifest=manifest,
                comparison_report=comparison,
                before_benchmark=_benchmark("partial"),
                after_benchmark=_benchmark("alive"),
                gate_report=report,
                approval=promotion,
                useful_behavior_note="Documented useful behavior.",
            )


def test_promotion_blocks_without_improvement_or_useful_behavior_note() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        _bundle_value, report, manifest, comparison, promotion = _artifacts(tmpdir)
        gate = evaluate_replay_adapter_promotion_gate(
            adapter_manifest=manifest,
            comparison_report=comparison,
            before_benchmark=_benchmark("partial"),
            after_benchmark=_benchmark("alive"),
            gate_report=report,
            approval=promotion,
            useful_behavior_note="",
        )

    assert gate["eligible_for_experimental_promotion"] is False
    assert gate["checks"]["benchmark_improvement_or_documented_useful_behavior"] is False
