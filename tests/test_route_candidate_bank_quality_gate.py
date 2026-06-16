from __future__ import annotations

import math

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
    assert report["quality"]["exact_winner_in_bank_rate"] == 1.0
    assert (
        report["quality"]["bank_candidates_with_exact_previous_winner_match_rate"]
        == 1.0
    )
    assert report["quality"]["bank_previous_winner_drift_match_gap"] == 0.0
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
    assert report["quality"]["exact_winner_in_bank_rate"] < 1.0
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
    assert report["quality"]["exact_winner_in_bank_rate"] == 1.0
    assert report["promotion_status"] == "passes_real_source_route_bank_quality_gate"


def test_route_candidate_hypercube_neighbors_recover_local_bitflip_shift() -> None:
    keys = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.05, 0.95],
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
                [0.1, 0.9, 0.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )
    kwargs = {
        "routing_keys": F.normalize(keys.float(), dim=1),
        "routing_vectors": vectors,
        "routing_ids": torch.arange(8, dtype=torch.long),
        "prototypes": vectors.clone(),
        "thresholds": torch.zeros(8, dtype=torch.float32),
        "prediction_location": torch.zeros(8, 3, dtype=torch.float32),
        "k_routing": 2,
        "bank_size": 2,
        "previous_winner": 0,
        "route_candidate_bank_refresh_interval": 1,
    }

    base_report = evaluate_route_candidate_bank_quality_from_tensors(**kwargs)
    neighbor_report = evaluate_route_candidate_bank_quality_from_tensors(
        **kwargs,
        route_candidate_hypercube_neighbor_rows=3,
    )

    assert base_report["quality"]["exact_top1_in_bank_rate"] < 1.0
    assert neighbor_report["route_candidate_hypercube_neighbors"]["enabled"] is True
    assert neighbor_report["route_candidate_hypercube_neighbors"]["neighbor_rows"] == 3
    assert neighbor_report["route_candidate_hypercube_neighbors"]["valid_rows"]["max"] <= 3
    assert neighbor_report["steady_route_score_rows"]["max"] < neighbor_report["total_columns"]
    assert neighbor_report["quality"]["exact_top1_in_bank_rate"] == 1.0
    assert neighbor_report["quality"]["exact_winner_match_rate"] == 1.0
    assert (
        neighbor_report["promotion_status"]
        == "passes_real_source_route_bank_quality_gate"
    )


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


def test_route_candidate_graph_walk_recovers_multi_hop_shift_with_bounded_rows() -> None:
    angles = torch.tensor(
        [math.radians(value) for value in range(0, 100, 10)],
        dtype=torch.float32,
    )
    vectors = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1)
    keys = torch.stack((vectors[0], vectors[6], vectors[6]))

    base_kwargs = {
        "routing_keys": keys,
        "routing_vectors": vectors,
        "routing_ids": torch.arange(int(vectors.shape[0]), dtype=torch.long),
        "prototypes": vectors.clone(),
        "thresholds": torch.zeros(int(vectors.shape[0]), dtype=torch.float32),
        "prediction_location": torch.zeros(
            int(vectors.shape[0]),
            int(vectors.shape[1]),
            dtype=torch.float32,
        ),
        "k_routing": 2,
        "bank_size": 2,
        "previous_winner": 0,
        "route_candidate_graph_neighbor_count": 3,
        "route_candidate_graph_capacity_rows": 8,
        "route_candidate_bank_refresh_interval": 16,
    }

    one_hop_report = evaluate_route_candidate_bank_quality_from_tensors(
        **base_kwargs,
    )
    walk_report = evaluate_route_candidate_bank_quality_from_tensors(
        **base_kwargs,
        route_candidate_graph_walk_beam=1,
        route_candidate_graph_walk_rounds=4,
    )

    assert one_hop_report["route_candidate_graph"]["mode"] == "one_hop_capacity"
    assert one_hop_report["quality"]["exact_top1_in_bank_rate"] < 1.0
    assert walk_report["route_candidate_graph"]["mode"] == "bounded_walk"
    assert walk_report["route_candidate_graph_walk"]["enabled"] is True
    assert walk_report["route_candidate_graph_walk"]["beam"] == 1
    assert walk_report["route_candidate_graph_walk"]["rounds"] == 4
    assert walk_report["steady_route_score_rows"]["max"] < walk_report["total_columns"]
    assert walk_report["quality"]["exact_top1_in_bank_rate"] == 1.0
    assert walk_report["quality"]["exact_winner_match_rate"] == 1.0
    assert walk_report["promotion_status"] == "passes_real_source_route_bank_quality_gate"
