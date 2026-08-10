from marulho.evaluation.language_continual_replay_falsification import (
    ADVANCE_DECISION,
    ARM_FRACTIONS,
    INVALID_DECISION,
    REDESIGN_DOMAIN_DECISION,
    RETIRE_REPLAY_DECISION,
    ContinualReplayConfig,
    select_continual_replay,
)


def _row(free: float, ranked: float, loss: float) -> dict:
    return {
        "all_parameters_received_final_gradient": True,
        "relation": {"generation_exact_accuracy": free, "accuracy": ranked},
        "heldout": {"heldout_loss": loss},
    }


def test_v38_budget_is_batch256_aligned() -> None:
    config = ContinualReplayConfig()
    assert config.token_budget == 910 * 256 * 72


def test_v38_selects_joint_free_generation_and_retention() -> None:
    arms = {
        "relation100": _row(0.80, 0.95, 5.0),
        "relation50_replay50": _row(0.61, 0.91, 3.20),
        "relation20_replay80": _row(0.55, 0.98, 3.15),
    }
    selected, decision = select_continual_replay(
        arms,
        initial_relation={"generation_exact_accuracy": 0.0},
        initial_general_loss=3.14,
        config=ContinualReplayConfig(),
    )
    assert selected == "relation50_replay50"
    assert decision == ADVANCE_DECISION


def test_v38_distinguishes_replay_failure_from_domain_failure() -> None:
    arms = {
        "relation100": _row(0.80, 0.95, 5.0),
        "relation50_replay50": _row(0.45, 0.90, 3.18),
        "relation20_replay80": _row(0.30, 0.98, 3.14),
    }
    assert select_continual_replay(
        arms,
        initial_relation={"generation_exact_accuracy": 0.0},
        initial_general_loss=3.14,
        config=ContinualReplayConfig(),
    )[1] == RETIRE_REPLAY_DECISION
    arms["relation100"] = _row(0.20, 0.95, 5.0)
    assert select_continual_replay(
        arms,
        initial_relation={"generation_exact_accuracy": 0.0},
        initial_general_loss=3.14,
        config=ContinualReplayConfig(),
    )[1] == REDESIGN_DOMAIN_DECISION
    assert select_continual_replay(
        {key: arms[key] for key in tuple(ARM_FRACTIONS)[:2]},
        initial_relation={"generation_exact_accuracy": 0.0},
        initial_general_loss=3.14,
        config=ContinualReplayConfig(),
    )[1] == INVALID_DECISION
