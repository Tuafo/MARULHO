from __future__ import annotations

import json
from pathlib import Path
import tempfile

from hecsn.evaluation.replay_adaptation_experiment_1 import evaluate_replay_adaptation_experiment_1
from hecsn.evaluation.replay_adapter_promotion_gate import evaluate_replay_adapter_promotion_gate
from hecsn.evaluation.replay_training_approval import (
    EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
    ISOLATED_ADAPTER_TRAINING_SCOPE,
    build_replay_training_approval,
)
from hecsn.evaluation.replay_training_plan import build_replay_training_plan
from hecsn.training.replay_adapter_experiment import ADAPTER_DELTA_NAME, COMPARISON_REPORT_NAME, run_isolated_replay_adapter_experiment

from tests.test_replay_adapter_experiment import _benchmark, _bundle, _gate_report, _long_test


def test_replay_adaptation_experiment_1_passes_with_isolated_artifact() -> None:
    bundle = _bundle()
    gate_report = _gate_report(bundle)
    dry_run_approval = build_replay_training_approval(bundle, gate_report, operator_id="operator-a")
    isolated_approval = build_replay_training_approval(
        bundle,
        gate_report,
        operator_id="operator-a",
        scope=ISOLATED_ADAPTER_TRAINING_SCOPE,
    )
    plan = build_replay_training_plan(bundle, gate_report, dry_run_approval)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "isolated" / "adapter-a"
        manifest = run_isolated_replay_adapter_experiment(
            bundle=bundle,
            gate_report=gate_report,
            training_plan=plan,
            approval=isolated_approval,
            before_benchmark=_benchmark("alive"),
            after_benchmark=_benchmark("alive"),
            before_long_test=_long_test("alive"),
            after_long_test=_long_test("alive"),
            output_dir=output_dir,
            repo_root=Path(tmpdir),
        )
        comparison = json.loads((output_dir / COMPARISON_REPORT_NAME).read_text(encoding="utf-8"))
        promotion_seed = {
            "status": "seed",
            "checks": [{"name": "decontamination", "passed": True}],
        }
        promotion_approval = build_replay_training_approval(
            bundle,
            promotion_seed,
            operator_id="operator-a",
            scope=EXPERIMENTAL_ADAPTER_PROMOTION_SCOPE,
            intended_target="experimental_replay_adapter_promotion_gate",
        )
        promotion = evaluate_replay_adapter_promotion_gate(
            adapter_manifest=manifest,
            comparison_report=comparison,
            before_benchmark=_benchmark("alive"),
            after_benchmark=_benchmark("alive"),
            gate_report=promotion_seed,
            approval=promotion_approval,
            useful_behavior_note="No production promotion; isolated evidence only.",
        )
        report_path = Path(tmpdir) / "phase12.json"
        report = evaluate_replay_adaptation_experiment_1(
            adapter_manifest=manifest,
            comparison_report=comparison,
            promotion_gate_report=promotion,
            before_benchmark=_benchmark("alive"),
            after_benchmark=_benchmark("alive"),
            before_long_test=_long_test("alive"),
            after_long_test=_long_test("alive"),
            holdout_report={"status": "passed", "checks": {"holdout_split_present": True}},
            output_path=report_path,
        )

        assert (output_dir / ADAPTER_DELTA_NAME).exists()
        assert report_path.exists()
        assert (report_path.parent / "README.md").exists()

    assert report["passed"] is True
    assert report["checks"]["isolated_artifact_exists"] is True
    assert report["checks"]["production_runtime_unchanged"] is True
    assert report["safety_flags"]["production_model_switch"] is False


def test_replay_adaptation_experiment_1_blocks_missing_holdout_evidence() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "isolated" / "adapter-a"
        output_dir.mkdir(parents=True)
        (output_dir / ADAPTER_DELTA_NAME).write_text("{}", encoding="utf-8")
        manifest = {
            "adapter": {
                "path": str(output_dir),
                "delta_file": ADAPTER_DELTA_NAME,
                "production_runtime_target": False,
                "production_runtime_switched": False,
            },
            "side_effects": {"production_runtime_switched": False},
            "rollback": {"rollback_path": "delete isolated artifact"},
        }
        comparison = {
            "status": "passed",
            "checks": {"runtime_truth_no_regression": True},
            "unsafe_flag_increases": {},
        }
        promotion = {
            "eligible_for_production_promotion": False,
            "checks": {"unsafe_action_or_replay_not_increased": True},
            "rollback_metadata": {"production_runtime_changed": False},
        }
        report = evaluate_replay_adaptation_experiment_1(
            adapter_manifest=manifest,
            comparison_report=comparison,
            promotion_gate_report=promotion,
            before_benchmark={"success": True},
            after_benchmark={"success": True},
            before_long_test={"health_verdict": "alive"},
            after_long_test={"health_verdict": "alive"},
            holdout_report={},
        )

    assert report["passed"] is False
    assert report["checks"]["holdout_evidence_saved"] is False
