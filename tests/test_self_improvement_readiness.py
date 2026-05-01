from __future__ import annotations

import json
from pathlib import Path
import tempfile

from hecsn.evaluation.self_improvement_readiness import (
    evaluate_self_improvement_readiness,
    evaluate_self_improvement_readiness_files,
)


def _phase12() -> dict[str, object]:
    return {"status": "passed_isolated_adaptation_evidence", "passed": True}


def _phase13() -> dict[str, object]:
    return {"status": "executed_approved_workspace_action", "accepted": True}


def _phase14() -> dict[str, object]:
    return {
        "status": "passed_multi_hour_living_evidence",
        "passed": True,
        "health_verdict": "alive",
        "runtime_truth_verdict": "alive",
        "replay_safety_status": {
            "training_started": False,
            "memory_mutated": False,
            "digital_action_executed": False,
        },
    }


def _promotion_gate() -> dict[str, object]:
    return {
        "status": "passed_experimental_promotion_allowed",
        "eligible_for_experimental_promotion": True,
        "eligible_for_production_promotion": False,
        "checks": {"benchmark_improvement_or_documented_useful_behavior": True},
        "rollback_metadata": {
            "production_runtime_changed": False,
            "rollback_action": "Remove the experimental adapter path reference.",
            "adapter_artifact_path": "reports/isolated/adapter",
        },
    }


def _benchmark() -> dict[str, object]:
    return {"success": True, "total_latency_ms": 100.0}


def test_phase15_readiness_passes_when_all_evidence_is_present() -> None:
    report = evaluate_self_improvement_readiness(
        phase12_report=_phase12(),
        phase13_report=_phase13(),
        phase14_report=_phase14(),
        promotion_gate_report=_promotion_gate(),
        benchmark_report=_benchmark(),
        operator_id="operator-a",
        useful_behavior_note="The isolated adapter improved replay holdout behavior.",
    )

    assert report["passed"] is True
    assert report["status"] == "ready_for_bounded_level_5_experiment"
    assert report["production_model_switch_allowed"] is False
    assert report["checks"]["autonomy_ladder_level_5_passed"] is True


def test_phase15_blocks_missing_phase14_evidence() -> None:
    phase14 = _phase14()
    phase14["passed"] = False
    phase14["status"] = "blocked_multi_hour_living_claim"

    report = evaluate_self_improvement_readiness(
        phase12_report=_phase12(),
        phase13_report=_phase13(),
        phase14_report=phase14,
        promotion_gate_report=_promotion_gate(),
        benchmark_report=_benchmark(),
        operator_id="operator-a",
        useful_behavior_note="Useful behavior was documented.",
    )

    assert report["passed"] is False
    assert report["checks"]["phase14_multi_hour_evidence_passed"] is False
    assert report["status"] == "blocked_for_bounded_level_5_experiment"


def test_phase15_keeps_production_switch_blocked_even_when_ready() -> None:
    report = evaluate_self_improvement_readiness(
        phase12_report=_phase12(),
        phase13_report=_phase13(),
        phase14_report=_phase14(),
        promotion_gate_report=_promotion_gate(),
        benchmark_report=_benchmark(),
        operator_id="operator-a",
        useful_behavior_note="Useful behavior was documented.",
    )

    assert report["production_model_switch_allowed"] is False
    assert report["checks"]["production_model_switch_blocked"] is True
    assert report["autonomy_ladder_report"]["safety_flags"]["production_model_switch"] is False


def test_phase15_file_writes_readme_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = {
            "phase12": root / "phase12.json",
            "phase13": root / "phase13.json",
            "phase14": root / "phase14.json",
            "promotion": root / "promotion.json",
            "benchmark": root / "benchmark.json",
        }
        paths["phase12"].write_text(json.dumps(_phase12()), encoding="utf-8")
        paths["phase13"].write_text(json.dumps(_phase13()), encoding="utf-8")
        paths["phase14"].write_text(json.dumps(_phase14()), encoding="utf-8")
        paths["promotion"].write_text(json.dumps(_promotion_gate()), encoding="utf-8")
        paths["benchmark"].write_text(json.dumps(_benchmark()), encoding="utf-8")
        output_path = root / "phase15.json"

        report = evaluate_self_improvement_readiness_files(
            phase12_report_path=paths["phase12"],
            phase13_report_path=paths["phase13"],
            phase14_report_path=paths["phase14"],
            promotion_gate_report_path=paths["promotion"],
            benchmark_report_path=paths["benchmark"],
            output_path=output_path,
            operator_id="operator-a",
            useful_behavior_note="Useful behavior was documented.",
        )

        assert report["passed"] is True
        assert output_path.exists()
        assert (output_path.parent / "README.md").exists()
