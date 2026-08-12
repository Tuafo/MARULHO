from marulho.evaluation.language_span_encoder_falsification import (
    SpanEncoderFalsificationConfig,
    span_encoder_gate,
)


def _row(intact: float = 0.296875):
    return {
        "processed_tokens": 2_096_640,
        "elapsed_seconds": 100.0,
        "all_parameters_received_final_gradient": True,
        "all_parameters_received_final_nonzero_gradient": True,
        "source_grounding": {
            "valid": True,
            "intact_source": {"exact_answer_accuracy": intact},
            "intact_gain_over_stronger_control": intact,
        },
    }


def _parent():
    return {
        "checkpoint_file_exact": True,
        "state_exact": True,
        "logits_exact": True,
        "general_loss_exact": True,
        "relation_exact": True,
    }


def test_v54_gate_requires_capability_and_exact_parent() -> None:
    config = SpanEncoderFalsificationConfig()
    passing = span_encoder_gate(
        _row(),
        parent=_parent(),
        checkpoint_fidelity={"passed": True},
        encoder_parameters=700_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert passing["passed"]

    weak = span_encoder_gate(
        _row(0.28125),
        parent=_parent(),
        checkpoint_fidelity={"passed": True},
        encoder_parameters=700_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert not weak["passed"]
    assert not weak["checks"]["minimum_grounding_accuracy"]

    changed_parent = _parent()
    changed_parent["state_exact"] = False
    corrupted = span_encoder_gate(
        _row(),
        parent=changed_parent,
        checkpoint_fidelity={"passed": True},
        encoder_parameters=700_000,
        parent_parameters=100_000_000,
        config=config,
    )
    assert not corrupted["passed"]
    assert not corrupted["checks"]["parent_state_exact"]
