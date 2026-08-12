from marulho.evaluation.language_multiview_transducer_falsification import (
    MultiViewTransducerFalsificationConfig,
    multiview_transducer_gate,
)


def _source(accuracy: float):
    return {
        "valid": True,
        "intact_source": {"exact_answer_accuracy": accuracy},
        "intact_gain_over_stronger_control": accuracy,
    }


def _row(accuracy: float = 0.50):
    return {
        "processed_tokens": 8_847_360,
        "epoch_count": 15,
        "cache_plus_training_seconds": 100.0,
        "all_parameters_received_final_gradient": True,
        "all_parameters_received_final_nonzero_gradient": True,
        "view_mode_counts": {
            "both": 1_350,
            "bidirectional_only": 285,
            "causal_only": 285,
        },
        "source_grounding": _source(accuracy),
    }


def _parent():
    return {
        "checkpoint_file_exact": True,
        "state_exact": True,
        "logits_exact": True,
        "general_loss_exact": True,
        "relation_exact": True,
    }


def test_v55_gate_requires_capability_and_multiview_advantage() -> None:
    config = MultiViewTransducerFalsificationConfig()
    passing = multiview_transducer_gate(
        _row(),
        ablations={
            "bidirectional_only": _source(0.421875),
            "causal_only": _source(0.40625),
        },
        parent=_parent(),
        checkpoint_fidelity={"passed": True},
        transducer_parameters=2_100_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert passing["passed"]

    no_synergy = multiview_transducer_gate(
        _row(),
        ablations={
            "bidirectional_only": _source(0.46875),
            "causal_only": _source(0.40),
        },
        parent=_parent(),
        checkpoint_fidelity={"passed": True},
        transducer_parameters=2_100_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert not no_synergy["passed"]
    assert not no_synergy["checks"]["minimum_multiview_advantage"]

    weak = multiview_transducer_gate(
        _row(0.484375),
        ablations={
            "bidirectional_only": _source(0.40),
            "causal_only": _source(0.39),
        },
        parent=_parent(),
        checkpoint_fidelity={"passed": True},
        transducer_parameters=2_100_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert not weak["passed"]
    assert not weak["checks"]["minimum_grounding_accuracy"]
