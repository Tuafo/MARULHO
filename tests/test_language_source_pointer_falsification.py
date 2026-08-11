from marulho.evaluation.language_source_pointer_falsification import (
    SourcePointerFalsificationConfig,
    source_pointer_gate,
)


def _row(*, intact: float):
    return {
        "source_grounding": {
            "valid": True,
            "intact_source": {"exact_answer_accuracy": intact},
            "intact_gain_over_stronger_control": intact,
        },
        "processed_tokens": 2_096_640,
        "all_parameters_received_final_gradient": True,
    }


def _parent(*, exact: bool = True):
    return {
        "checkpoint_file_exact": exact,
        "state_exact": True,
        "logits_exact": True,
        "general_loss_exact": True,
        "relation_exact": True,
    }


def test_v53_gate_requires_compact_capability_and_exact_parent() -> None:
    gate = source_pointer_gate(
        _row(intact=0.28125),
        parent=_parent(),
        pointer_parameters=100_000,
        parent_parameters=100_000_000,
        config=SourcePointerFalsificationConfig(),
    )
    assert gate["passed"]


def test_v53_gate_rejects_parent_drift_or_two_case_regression() -> None:
    gate = source_pointer_gate(
        _row(intact=0.265625),
        parent=_parent(exact=False),
        pointer_parameters=100_000,
        parent_parameters=100_000_000,
        config=SourcePointerFalsificationConfig(),
    )
    assert not gate["passed"]
    assert not gate["checks"]["minimum_grounding_accuracy"]
    assert not gate["checks"]["parent_checkpoint_file_exact"]
