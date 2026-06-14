from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.evaluation.fused_route_vote_triton import fused_route_vote_cuda
from marulho.core.fused_route_vote_cuda import fused_route_vote_kernel_variant


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.parametrize(
    ("vector_count", "column_dim", "location_dim", "candidate_count"),
    [
        (32, 8, 4, 5),
        (1024, 64, 64, 10),
    ],
)
@pytest.mark.parametrize("silent", [False, True])
def test_fused_route_vote_matches_tensor_routing_and_vote(
    vector_count: int,
    column_dim: int,
    location_dim: int,
    candidate_count: int,
    silent: bool,
) -> None:
    torch.manual_seed(20260612)
    device = torch.device("cuda")
    routing_key = F.normalize(torch.rand(column_dim, device=device), dim=0)
    routing_vectors = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    routing_ids = torch.randperm(vector_count, device=device, dtype=torch.long)
    prototypes = F.normalize(
        torch.rand(vector_count, column_dim, device=device),
        dim=1,
    )
    thresholds = torch.full(
        (vector_count,),
        4.0 if silent else 0.2,
        device=device,
    )
    locations = torch.rand(vector_count, location_dim, device=device)
    previous_winner = torch.tensor([7], dtype=torch.long, device=device)

    search_key = F.normalize(routing_key.unsqueeze(0), dim=1)
    scores = search_key @ routing_vectors.T
    positions = torch.topk(scores, k=candidate_count, dim=1).indices[0]
    expected_candidates = routing_ids[positions]
    competition_key = F.normalize(routing_key.clamp(min=1e-6), dim=0)
    location_similarity = F.cosine_similarity(
        locations[expected_candidates],
        locations[previous_winner.item()].unsqueeze(0),
        dim=1,
    )
    combined = (
        prototypes[expected_candidates] @ competition_key
    ) * (1.0 + 0.3 * location_similarity.clamp(-1.0, 1.0))
    activation = torch.relu(combined - thresholds[expected_candidates])
    expected_local = (
        torch.argmax(activation)
        if bool((activation.max() > 0.0).item())
        else torch.argmax(combined)
    )
    expected_winner = expected_candidates[expected_local]

    scores_out = torch.empty(vector_count, device=device)
    candidates_out = torch.empty(
        candidate_count,
        dtype=torch.long,
        device=device,
    )
    winner_out = torch.empty(1, dtype=torch.long, device=device)
    strength_out = torch.empty(1, device=device)
    had_positive = torch.empty((), dtype=torch.bool, device=device)
    reconstruction_error_out = torch.empty(1, device=device)
    assert (
        fused_route_vote_kernel_variant(routing_vectors, candidates_out)
        == "two_stage_route_vote"
    )
    fused_route_vote_cuda(
        routing_key=routing_key,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        previous_winner=previous_winner,
        scores_out=scores_out,
        candidates_out=candidates_out,
        winner_out=winner_out,
        strength_out=strength_out,
        competition_had_positive=had_positive,
        reconstruction_error_out=reconstruction_error_out,
    )
    torch.cuda.synchronize()

    assert torch.equal(candidates_out, expected_candidates)
    assert int(winner_out.item()) == int(expected_winner.item())
    assert int(previous_winner.item()) == int(expected_winner.item())
    assert float(strength_out.item()) == 1.0
    assert bool(had_positive.item()) is bool((activation.max() > 0.0).item())
    expected_reconstruction_error = torch.clamp(1.0 - scores.max(), min=0.0)
    assert torch.allclose(
        reconstruction_error_out.squeeze(0),
        expected_reconstruction_error.squeeze(0),
        rtol=0.0,
        atol=1e-6,
    )
