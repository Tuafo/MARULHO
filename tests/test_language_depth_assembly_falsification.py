from __future__ import annotations

from marulho.evaluation.language_depth_assembly_falsification import (
    ADVANCE_DECISION,
    BASELINE_ARM,
    CANDIDATE_ARM,
    INVALID_DECISION,
    RETIRE_DECISION,
    DepthAssemblyFalsificationConfig,
    select_depth_assembly,
)


def _row(loss: float, tps: float, peak: int, *, routes: int = 0) -> dict:
    return {
        "all_parameters_received_final_gradient": True,
        "heldout": {"heldout_loss": loss},
        "training": {
            "tokens_per_second": tps,
            "peak_cuda_memory_bytes": peak,
        },
        "diagnostics": {
            "parameter_count": routes,
            "nonzero_parameter_count": routes,
        },
    }


def test_v37_budget_is_exactly_batch256_aligned() -> None:
    config = DepthAssemblyFalsificationConfig()
    assert config.physical_batch_size == 256
    assert config.token_budget == 910 * 256 * 72


def test_depth_assembly_selection_requires_joint_gain() -> None:
    config = DepthAssemblyFalsificationConfig()
    advancing = {
        BASELINE_ARM: _row(3.20, 25_000.0, 8_000_000_000),
        CANDIDATE_ARM: _row(3.17, 23_000.0, 9_000_000_000, routes=45),
    }
    assert select_depth_assembly(advancing, config=config) == ADVANCE_DECISION

    slow = {**advancing, CANDIDATE_ARM: _row(3.17, 20_000.0, 9_000_000_000, routes=45)}
    assert select_depth_assembly(slow, config=config) == RETIRE_DECISION
    weak = {**advancing, CANDIDATE_ARM: _row(3.19, 24_000.0, 9_000_000_000, routes=45)}
    assert select_depth_assembly(weak, config=config) == RETIRE_DECISION
    heavy = {**advancing, CANDIDATE_ARM: _row(3.17, 24_000.0, 11_000_000_000, routes=45)}
    assert select_depth_assembly(heavy, config=config) == RETIRE_DECISION


def test_depth_assembly_selection_rejects_invalid_evidence() -> None:
    config = DepthAssemblyFalsificationConfig()
    missing = {BASELINE_ARM: _row(3.2, 25_000.0, 8_000_000_000)}
    assert select_depth_assembly(missing, config=config) == INVALID_DECISION
    dead_routes = {
        BASELINE_ARM: _row(3.2, 25_000.0, 8_000_000_000),
        CANDIDATE_ARM: _row(3.1, 25_000.0, 8_000_000_000, routes=0),
    }
    dead_routes[CANDIDATE_ARM]["diagnostics"] = {
        "parameter_count": 45,
        "nonzero_parameter_count": 44,
    }
    assert select_depth_assembly(dead_routes, config=config) == INVALID_DECISION
