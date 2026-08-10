import torch

from marulho.evaluation.language_answer_objective_falsification import (
    ADVANCE_DECISION,
    INVALID_DECISION,
    RETIRE_DECISION,
    AnswerObjectiveConfig,
    answer_target_mask,
    select_answer_objective,
)


def test_answer_target_mask_selects_through_eos_prediction() -> None:
    inputs = torch.tensor([[1, 7, 8, 10, 11, 2, 1, 5, 7, 8, 12]])
    mask = answer_target_mask(inputs, marker_ids=torch.tensor([7, 8]), eos_id=2)
    assert mask.tolist() == [[False, False, True, True, True, False, False, False, False, True, True]]


def _row(free: float, ranked: float, loss: float, gradients: bool = True) -> dict:
    return {
        "all_parameters_received_final_gradient": gradients,
        "relation": {"generation_exact_accuracy": free, "accuracy": ranked},
        "heldout": {"heldout_loss": loss},
    }


def test_v39_selects_answer_objective_only_on_joint_gate() -> None:
    config = AnswerObjectiveConfig()
    arms = {
        "answer_weight2": _row(0.52, 1.0, 3.14),
        "answer_weight4": _row(0.49, 1.0, 3.10),
    }
    assert select_answer_objective(
        arms, initial_general_loss=3.16, config=config
    ) == ("answer_weight2", ADVANCE_DECISION)
    arms["answer_weight2"] = _row(0.52, 1.0, 3.30)
    assert select_answer_objective(
        arms, initial_general_loss=3.16, config=config
    )[1] == RETIRE_DECISION
    arms["answer_weight2"] = _row(0.52, 1.0, 3.14, gradients=False)
    assert select_answer_objective(
        arms, initial_general_loss=3.16, config=config
    )[1] == INVALID_DECISION
