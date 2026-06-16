from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.evaluation.route_candidate_bank_quality_gate import (
    evaluate_route_candidate_bank_quality_from_tensors,
)


def _quality_report(keys: torch.Tensor, *, bank_size: int = 2) -> dict:
    vectors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    ids = torch.arange(int(vectors.shape[0]), dtype=torch.long)
    prototypes = vectors.clone()
    thresholds = torch.zeros(int(vectors.shape[0]), dtype=torch.float32)
    locations = torch.eye(int(vectors.shape[0]), dtype=torch.float32)
    return evaluate_route_candidate_bank_quality_from_tensors(
        routing_keys=F.normalize(keys.float(), dim=1),
        routing_vectors=vectors,
        routing_ids=ids,
        prototypes=prototypes,
        thresholds=thresholds,
        prediction_location=locations,
        k_routing=2,
        bank_size=bank_size,
        previous_winner=0,
    )


def test_route_candidate_bank_quality_passes_stable_local_source() -> None:
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.05, 0.0],
            [0.92, 0.08, 0.0],
            [0.91, 0.09, 0.0],
        ],
        dtype=torch.float32,
    )

    report = _quality_report(keys)

    assert report["surface"] == "route_candidate_bank_quality_gate.v1"
    assert report["hot_path_all_column_oracle"] is False
    assert report["exact_seed_count"] == 1
    assert report["steady_bank_score_rows"]["mean"] == 2.0
    assert report["quality"]["exact_top1_in_bank_rate"] == 1.0
    assert report["quality"]["exact_winner_match_rate"] == 1.0
    assert report["promotion_status"] == "passes_real_source_route_bank_quality_gate"


def test_route_candidate_bank_quality_rejects_relevance_shift_without_reseed() -> None:
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.95, 0.05],
        ],
        dtype=torch.float32,
    )

    report = _quality_report(keys)

    assert report["hot_path_all_column_oracle"] is False
    assert report["quality"]["exact_top1_in_bank_rate"] < 1.0
    assert report["quality"]["exact_winner_match_rate"] < 1.0
    assert report["quality"]["worst_consecutive_exact_top1_miss"] >= 1
    assert (
        report["promotion_status"]
        == "requires_reseed_policy_or_wider_bank_before_quality_claim"
    )


def test_route_candidate_probe_lane_recovers_bounded_relevance_shift() -> None:
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.95, 0.05],
        ],
        dtype=torch.float32,
    )
    vectors = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.9, 0.1],
                [0.0, 0.0, 1.0],
                [0.1, 0.0, 0.9],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )

    report = evaluate_route_candidate_bank_quality_from_tensors(
        routing_keys=F.normalize(keys.float(), dim=1),
        routing_vectors=vectors,
        routing_ids=torch.arange(6, dtype=torch.long),
        prototypes=vectors.clone(),
        thresholds=torch.zeros(6, dtype=torch.float32),
        prediction_location=torch.eye(6, dtype=torch.float32),
        k_routing=2,
        bank_size=2,
        previous_winner=0,
        route_candidate_probe_rows=2,
        route_candidate_bank_refresh_interval=16,
    )

    assert report["hot_path_all_column_oracle"] is False
    assert report["route_candidate_probe"]["enabled"] is True
    assert report["route_candidate_probe"]["probe_rows"] == 2
    assert report["steady_route_score_rows"]["mean"] == 4.0
    assert report["steady_route_score_rows"]["max"] < report["total_columns"]
    assert report["route_candidate_probe"]["exact_top1_discovered_from_probe_rows"] >= 1
    assert report["quality"]["exact_top1_in_bank_rate"] == 1.0
    assert report["quality"]["exact_winner_match_rate"] == 1.0
    assert report["promotion_status"] == "passes_real_source_route_bank_quality_gate"


def test_route_candidate_graph_neighbors_recover_bounded_relevance_shift() -> None:
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.95, 0.05],
        ],
        dtype=torch.float32,
    )

    report = evaluate_route_candidate_bank_quality_from_tensors(
        routing_keys=F.normalize(keys.float(), dim=1),
        routing_vectors=F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.9, 0.1],
                ],
                dtype=torch.float32,
            ),
            dim=1,
        ),
        routing_ids=torch.arange(4, dtype=torch.long),
        prototypes=F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.9, 0.1],
                ],
                dtype=torch.float32,
            ),
            dim=1,
        ),
        thresholds=torch.zeros(4, dtype=torch.float32),
        prediction_location=torch.eye(4, dtype=torch.float32),
        k_routing=2,
        bank_size=2,
        route_candidate_graph_neighbor_count=3,
    )

    assert report["route_candidate_graph"]["enabled"] is True
    assert report["route_candidate_graph"]["valid_rows"]["mean"] == 4.0
    assert report["steady_bank_score_rows"]["mean"] == 4.0
    assert report["quality"]["exact_top1_in_bank_rate"] == 1.0
    assert report["quality"]["exact_winner_match_rate"] == 1.0
    assert report["promotion_status"] == "passes_real_source_route_bank_quality_gate"
