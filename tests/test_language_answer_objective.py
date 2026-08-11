import pytest
import torch
from torch import nn

from marulho.training.language_answer_objective import (
    answer_target_mask,
    answer_weighted_next_token_loss,
)


def test_answer_target_mask_selects_through_eos_prediction() -> None:
    inputs = torch.tensor([[1, 7, 8, 10, 11, 2, 1, 5, 7, 8, 12]])
    mask = answer_target_mask(inputs, marker_ids=torch.tensor([7, 8]), eos_id=2)
    assert mask.tolist() == [
        [False, False, True, True, True, False, False, False, False, True, True]
    ]


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 6, 16))

    def forward(self, input_ids, *, collect_telemetry=False):
        del input_ids, collect_telemetry
        return {"logits": self.logits}


def test_answer_weighted_loss_is_finite_and_differentiable() -> None:
    model = _ToyModel()
    inputs = torch.tensor([[1, 7, 8, 10, 11, 2]])
    targets = torch.tensor([[7, 8, 10, 11, 2, 1]])
    loss = answer_weighted_next_token_loss(
        model,
        inputs,
        targets,
        marker_ids=torch.tensor([7, 8]),
        eos_id=2,
        answer_weight=4.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.logits.grad is not None
    with pytest.raises(ValueError, match="at least one"):
        answer_weighted_next_token_loss(
            model,
            inputs,
            targets,
            marker_ids=torch.tensor([7, 8]),
            eos_id=2,
            answer_weight=0.5,
        )


def test_answer_weighted_loss_ignores_post_eos_padding() -> None:
    model = _ToyModel()
    inputs = torch.tensor([[1, 7, 8, 10, 2, 0]])
    targets = torch.tensor([[7, 8, 10, 2, 0, 0]])
    loss = answer_weighted_next_token_loss(
        model,
        inputs,
        targets,
        marker_ids=torch.tensor([7, 8]),
        eos_id=2,
        answer_weight=4.0,
        pad_id=0,
    )
    loss.backward()
    assert torch.count_nonzero(model.logits.grad[:, 4:]) == 0
