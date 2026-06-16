from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.evaluation.route_candidate_discovery_probe import (
    evaluate_discovery_probe_from_tensors,
)


def _cluster_vectors() -> torch.Tensor:
    return F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.95, 0.05, 0.0],
                [0.0, 1.0, 0.0],
                [0.05, 0.95, 0.0],
                [0.0, 0.0, 1.0],
                [0.05, 0.0, 0.95],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )


def test_landmark_bucket_probe_recovers_controlled_relevance_shift() -> None:
    vectors = _cluster_vectors()
    keys = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.95, 0.05],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )

    report = evaluate_discovery_probe_from_tensors(
        mode="landmark_bucket",
        routing_keys=keys,
        routing_vectors=vectors,
        routing_ids=torch.arange(6, dtype=torch.long),
        prototypes=vectors.clone(),
        thresholds=torch.zeros(6, dtype=torch.float32),
        prediction_location=torch.eye(6, dtype=torch.float32),
        k_routing=2,
        bank_size=2,
        refresh_interval=1,
        landmark_count=3,
        top_landmarks=1,
        bucket_rows=2,
    )

    assert report["surface"] == "route_candidate_discovery_probe.v1"
    assert report["mode"] == "landmark_bucket"
    assert report["hot_path_all_column_scan"] is False
    assert report["selector_rows_scored_on_refresh"] == 3
    assert report["steady_route_score_rows"]["max"] < report["total_columns"]
    assert report["quality"]["exact_top1_in_candidate_rate"] == 1.0
    assert report["quality"]["exact_winner_match_rate"] == 1.0
    assert report["promotion_status"] == "passes_bounded_discovery_quality_gate"


def test_random_projection_probe_reports_bounded_rows_without_promotion_claim() -> None:
    vectors = _cluster_vectors()
    keys = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    )

    report = evaluate_discovery_probe_from_tensors(
        mode="random_projection_bucket",
        routing_keys=keys,
        routing_vectors=vectors,
        routing_ids=torch.arange(6, dtype=torch.long),
        prototypes=vectors.clone(),
        thresholds=torch.zeros(6, dtype=torch.float32),
        prediction_location=torch.eye(6, dtype=torch.float32),
        k_routing=2,
        bank_size=2,
        refresh_interval=1,
        projection_count=4,
        top_projections=1,
        bucket_rows=2,
        seed=7,
    )

    assert report["mode"] == "random_projection_bucket"
    assert report["hot_path_all_column_scan"] is False
    assert report["selector_rows_scored_on_refresh"] == 4
    assert report["steady_route_score_rows"]["max"] <= report["total_columns"]
    assert "exact_top1_in_candidate_rate" in report["quality"]


def test_discovery_probe_rejects_unknown_mode() -> None:
    vectors = _cluster_vectors()
    with pytest.raises(ValueError, match="mode must be"):
        evaluate_discovery_probe_from_tensors(
            mode="bad",
            routing_keys=vectors[:1],
            routing_vectors=vectors,
            routing_ids=torch.arange(6, dtype=torch.long),
            prototypes=vectors.clone(),
            thresholds=torch.zeros(6, dtype=torch.float32),
            prediction_location=torch.eye(6, dtype=torch.float32),
            k_routing=2,
            bank_size=2,
        )
