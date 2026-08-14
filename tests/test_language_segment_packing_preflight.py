from __future__ import annotations

import torch
from torch import nn

from marulho.evaluation.language_quality_continuation import _episode
from marulho.evaluation.language_segment_packing_preflight import (
    admission_checks,
    packed_episode,
)


class _TokenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, 12)
        self.output = nn.Linear(12, 64, bias=False)

    def forward(self, input_ids: torch.Tensor, **_: object) -> dict:
        return {"logits": self.output(self.embedding(input_ids))}


def test_packed_episode_matches_independent_segment_objective() -> None:
    torch.manual_seed(801)
    model = _TokenModel()
    documents = torch.randint(0, 64, (4, 961))
    baseline = _episode(model, documents)
    packed = packed_episode(model, documents)
    assert torch.allclose(baseline["loss"], packed["loss"], atol=1.0e-6)
    assert torch.allclose(
        baseline["per_document_segment_losses"],
        packed["per_document_segment_losses"],
        atol=1.0e-6,
    )


def test_v80_admission_requires_parity_speed_and_memory() -> None:
    timing = {
        "median_positions_per_second": 20_000.0,
        "warmup_gradient_audit": {"passed": True},
        "model_state_finite": True,
        "peak_cuda_allocated_bytes": 4 * 1024**3,
    }
    packed = {**timing, "median_positions_per_second": 22_100.0}
    checks = admission_checks(
        forward_loss_delta=0.001,
        per_document_max_delta=0.01,
        gradient={"cosine": 0.99995, "relative_l2_error": 0.01},
        state={"relative_l2_error": 0.001},
        post_update_loss_delta=0.001,
        baseline=timing,
        packed=packed,
    )
    assert all(checks.values())
    packed["median_positions_per_second"] = 21_900.0
    checks = admission_checks(
        forward_loss_delta=0.001,
        per_document_max_delta=0.01,
        gradient={"cosine": 0.99995, "relative_l2_error": 0.01},
        state={"relative_l2_error": 0.001},
        post_update_loss_delta=0.001,
        baseline=timing,
        packed=packed,
    )
    assert not checks["packed_throughput_gain_at_least_1_10"]
