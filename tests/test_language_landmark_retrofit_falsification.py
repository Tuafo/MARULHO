from marulho.evaluation.language_landmark_retrofit_falsification import (
    LandmarkRetrofitFalsificationConfig,
    _training_schedule,
    landmark_retrofit_gate,
)


def _passing_row(config):
    return {
        "processed_adapter_positions": config.adapter_position_budget,
        "epoch_count": config.epoch_count,
        "all_parameters_received_final_gradient": True,
        "all_parameters_received_final_nonzero_gradient": True,
        "cache_plus_training_seconds": 100.0,
        "source_grounding": {
            "valid": True,
            "predicted_top2_block_union_answer_coverage": 0.85,
            "intact_gain_over_stronger_control": 0.50,
            "conditions": {
                "predicted_top2": {"exact_answer_count": 70},
                "oracle": {"exact_answer_count": 76},
                "shuffled": {"exact_answer_count": 4},
            },
        },
    }


def _exact_parent():
    return {
        "checkpoint_file_exact": True,
        "state_exact": True,
        "tokenizer_exact": True,
        "logits_exact": True,
        "general_loss_exact": True,
        "relation_exact": True,
    }


def test_v56_schedule_is_seeded_full_epoch_permutations() -> None:
    schedule, digest = _training_schedule(batch_count=4, epoch_count=3, seed=56)
    repeated, repeated_digest = _training_schedule(
        batch_count=4, epoch_count=3, seed=56
    )

    assert schedule == repeated
    assert digest == repeated_digest
    assert len(digest) == 64
    assert all(
        sorted(schedule[start : start + 4]) == [0, 1, 2, 3] for start in range(0, 12, 4)
    )


def test_v56_gate_requires_joint_capability_and_fidelity() -> None:
    config = LandmarkRetrofitFalsificationConfig()
    gate = landmark_retrofit_gate(
        _passing_row(config),
        parent=_exact_parent(),
        checkpoint_fidelity={"passed": True},
        retrofit_parameters=2_000_000,
        parent_parameters=100_000_000,
        config=config,
    )

    assert gate["passed"]
    failed = _passing_row(config)
    failed["source_grounding"]["conditions"]["predicted_top2"] = {
        "exact_answer_count": 63
    }
    failed_gate = landmark_retrofit_gate(
        failed,
        parent=_exact_parent(),
        checkpoint_fidelity={"passed": True},
        retrofit_parameters=2_000_000,
        parent_parameters=100_000_000,
        config=config,
    )

    assert not failed_gate["passed"]
    assert not failed_gate["checks"]["minimum_predicted_answer_count"]
