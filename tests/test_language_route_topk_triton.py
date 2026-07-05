from __future__ import annotations

import torch

from marulho.core.language_route_topk_triton import (
    can_use_language_route_topk_triton,
    language_route_topk,
    language_route_topk_torch_reference,
    language_route_topk_triton_stats,
)


def _route_inputs(device: torch.device | str) -> dict[str, torch.Tensor]:
    torch.manual_seed(20260704)
    rows = 12
    cols = 16
    expert_count = 10
    route_candidate_count = 4
    candidate_offsets = torch.arange(
        route_candidate_count,
        device=device,
        dtype=torch.long,
    ).view(1, route_candidate_count)
    row_offsets = (
        torch.arange(rows, device=device, dtype=torch.long).view(rows, 1)
        * route_candidate_count
    )
    return {
        "hidden": torch.randn(rows, cols, device=device, dtype=torch.float32),
        "candidate_ids": (row_offsets + candidate_offsets).remainder(expert_count),
        "route_keys": torch.randn(expert_count, cols, device=device, dtype=torch.float32),
        "route_bias": torch.randn(expert_count, device=device, dtype=torch.float32),
    }


def test_language_route_topk_reference_returns_selected_ids_scores_and_weights() -> None:
    inputs = _route_inputs("cpu")

    selected_ids, top_scores, top_weights = language_route_topk_torch_reference(
        **inputs,
        active_count=2,
    )

    assert selected_ids.shape == (12, 2)
    assert top_scores.shape == (12, 2)
    assert top_weights.shape == (12, 2)
    torch.testing.assert_close(top_weights.sum(dim=-1), torch.ones(12))
    assert selected_ids.dtype == torch.long
    assert selected_ids.min().item() >= 0
    assert selected_ids.max().item() < 10


def test_language_route_topk_wrapper_matches_reference_on_cpu() -> None:
    inputs = _route_inputs("cpu")

    output = language_route_topk(**inputs, active_count=3)
    reference = language_route_topk_torch_reference(**inputs, active_count=3)

    assert torch.equal(output[0], reference[0])
    torch.testing.assert_close(output[1], reference[1])
    torch.testing.assert_close(output[2], reference[2])


def test_language_route_topk_triton_matches_reference_when_available() -> None:
    if not torch.cuda.is_available() or not language_route_topk_triton_stats()[
        "triton_available"
    ]:
        return
    inputs = _route_inputs("cuda")
    assert can_use_language_route_topk_triton(**inputs, active_count=2) is True

    output = language_route_topk(
        **inputs,
        active_count=2,
        prefer_triton=True,
        force_triton=True,
    )
    reference = language_route_topk_torch_reference(**inputs, active_count=2)
    torch.cuda.synchronize()

    assert torch.equal(output[0], reference[0])
    torch.testing.assert_close(output[1], reference[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(output[2], reference[2], atol=1e-4, rtol=1e-4)
