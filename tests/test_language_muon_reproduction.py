from __future__ import annotations

import json

import pytest

from marulho.evaluation.language_muon_falsification import (
    ARTIFACT_KIND as QUALIFICATION_ARTIFACT_KIND,
    ARM_NAMES,
    MuonFalsificationConfig,
)
from marulho.evaluation.language_muon_reproduction import (
    REJECT_DECISION,
    REQUIRED_QUALIFICATION_DECISION,
    SAVE_DECISION,
    load_qualification_report,
    reproduction_decision,
)


def _qualification() -> dict:
    return {
        "artifact_kind": QUALIFICATION_ARTIFACT_KIND,
        "decision": REQUIRED_QUALIFICATION_DECISION,
        "arms": {name: {} for name in ARM_NAMES},
        "optimizer_comparison": {
            "adamw_heldout_loss": 4.26,
            "adamw_free_relation_accuracy": 0.05,
        },
    }


def _row(loss: float, free: float, *, gradients: bool = True) -> dict:
    return {
        "all_parameters_received_final_gradient": gradients,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v29_reproduction_requires_joint_gate_and_checkpoint_fidelity() -> None:
    config = MuonFalsificationConfig()
    qualification = _qualification()
    assert reproduction_decision(
        _row(4.20, 0.08),
        qualification,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == SAVE_DECISION
    assert reproduction_decision(
        _row(4.20, 0.08),
        qualification,
        config=config,
        checkpoint_fidelity_passed=False,
    ) == REJECT_DECISION
    assert reproduction_decision(
        _row(4.255, 0.08),
        qualification,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == REJECT_DECISION
    assert reproduction_decision(
        _row(4.20, 0.06),
        qualification,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == REJECT_DECISION
    assert reproduction_decision(
        _row(4.20, 0.08, gradients=False),
        qualification,
        config=config,
        checkpoint_fidelity_passed=True,
    ) == REJECT_DECISION


def test_load_v29_qualification_rejects_wrong_decision(tmp_path) -> None:
    path = tmp_path / "qualification.json"
    report = _qualification()
    path.write_text(json.dumps(report), encoding="utf-8")
    assert (
        load_qualification_report(path)["decision"]
        == REQUIRED_QUALIFICATION_DECISION
    )
    report["decision"] = "retire"
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="did not advance"):
        load_qualification_report(path)
