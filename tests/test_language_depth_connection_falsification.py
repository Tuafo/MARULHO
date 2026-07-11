from marulho.evaluation.language_depth_connection_falsification import (
    ARM_NAMES,
    depth_connection_decision,
)


def _arm(name: str, *, loss: float, free: float, tokens: int = 16_777_216):
    return {
        "name": name,
        "processed_tokens": tokens,
        "heldout": {"heldout_loss": loss},
        "relation": {"generation_exact_accuracy": free},
    }


def test_v9_arm_contract_contains_transformer_and_all_controls() -> None:
    assert ARM_NAMES == (
        "transformer",
        "identity",
        "fixed_mean",
        "fixed_random",
        "learned_unconstrained",
        "learned_simplex",
    )


def test_v9_decision_promotes_learned_mode_only_over_every_fixed_control() -> None:
    arms = [
        _arm("transformer", loss=4.10, free=0.15),
        _arm("identity", loss=4.10, free=0.15),
        _arm("fixed_mean", loss=4.08, free=0.17),
        _arm("fixed_random", loss=4.09, free=0.16),
        _arm("learned_unconstrained", loss=4.05, free=0.22),
        _arm("learned_simplex", loss=4.07, free=0.20),
    ]
    assert depth_connection_decision(arms) == (
        "replicate_v9_learned_unconstrained_before_scale"
    )


def test_v9_decision_preserves_fixed_control_win_without_learned_claim() -> None:
    arms = [
        _arm("transformer", loss=4.10, free=0.15),
        _arm("identity", loss=4.10, free=0.15),
        _arm("fixed_mean", loss=4.07, free=0.20),
        _arm("fixed_random", loss=4.09, free=0.17),
        _arm("learned_unconstrained", loss=4.08, free=0.18),
        _arm("learned_simplex", loss=4.08, free=0.18),
    ]
    assert depth_connection_decision(arms) == (
        "replicate_v9_fixed_mean_without_learned_connection_claim"
    )


def test_v9_decision_does_not_promote_disjoint_signals() -> None:
    arms = [
        _arm("transformer", loss=4.10, free=0.15),
        _arm("identity", loss=4.10, free=0.15),
        _arm("fixed_mean", loss=4.08, free=0.15),
        _arm("fixed_random", loss=4.11, free=0.20),
        _arm("learned_unconstrained", loss=4.10, free=0.15),
        _arm("learned_simplex", loss=4.10, free=0.15),
    ]
    assert depth_connection_decision(arms) == (
        "redesign_v9_disjoint_loss_and_behavior_signals"
    )


def test_v9_decision_labels_short_or_missing_runs() -> None:
    short = [
        _arm(name, loss=4.10, free=0.15, tokens=20_000) for name in ARM_NAMES
    ]
    assert depth_connection_decision(short) == "incomplete_v9_mechanism_smoke"
    assert depth_connection_decision(short[:-1]) == (
        "incomplete_v9_depth_connection_comparison"
    )


def test_v9_decision_retires_no_gain() -> None:
    arms = [_arm(name, loss=4.10, free=0.15) for name in ARM_NAMES]
    assert depth_connection_decision(arms) == (
        "retire_v9_depth_connections_no_quality_gain"
    )
