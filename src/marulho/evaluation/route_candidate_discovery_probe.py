"""Evaluate bounded route-discovery probes before runtime promotion.

This module is deliberately evaluation-only. It may build offline helper
indexes from the full routing cache, but the simulated hot-path step scores
only a fixed candidate set selected by the probe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from marulho.evaluation.route_candidate_bank_quality_gate import (
    _candidate_winner,
    _float_stats,
    _ordered_unique_limited_candidate_ids,
    _ordered_unique_limited_positions,
    _position_map,
    _random_patterns,
    _text_patterns,
)
from marulho.evaluation.continuous_runtime_quantum_benchmark import DEFAULT_SOURCE_TEXT
from marulho.training.checkpointing import load_trainer_checkpoint


def _select_farthest_landmarks(
    vectors: torch.Tensor,
    *,
    count: int,
) -> torch.Tensor:
    total = int(vectors.shape[0])
    target = max(1, min(int(count), total))
    mean = F.normalize(vectors.mean(dim=0, keepdim=True), dim=1)
    first = int(torch.argmin((vectors @ mean.T).flatten()).item())
    selected = [first]
    best_similarity = vectors @ vectors[first]
    for _ in range(1, target):
        next_position = int(torch.argmin(best_similarity).item())
        selected.append(next_position)
        best_similarity = torch.maximum(
            best_similarity,
            vectors @ vectors[next_position],
        )
    return torch.tensor(selected, dtype=torch.long, device=vectors.device)


@torch.no_grad()
def evaluate_discovery_probe_from_tensors(
    *,
    mode: str,
    routing_keys: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    k_routing: int,
    bank_size: int,
    refresh_interval: int = 16,
    landmark_count: int = 0,
    top_landmarks: int = 0,
    bucket_rows: int = 0,
    projection_count: int = 0,
    top_projections: int = 0,
    seed: int = 20260616,
    previous_winner: int = 0,
) -> dict[str, Any]:
    if routing_keys.dim() != 2:
        raise ValueError("routing_keys must be a rank-2 tensor")
    if routing_vectors.dim() != 2:
        raise ValueError("routing_vectors must be a rank-2 tensor")
    if int(routing_keys.shape[1]) != int(routing_vectors.shape[1]):
        raise ValueError("routing key width must match routing vector width")
    if int(routing_ids.numel()) != int(routing_vectors.shape[0]):
        raise ValueError("routing_ids must match routing vector rows")
    if mode not in {"landmark_bucket", "random_projection_bucket"}:
        raise ValueError("mode must be landmark_bucket or random_projection_bucket")

    device = routing_vectors.device
    keys = F.normalize(routing_keys.to(device=device), dim=1)
    vectors = F.normalize(routing_vectors.to(device=device), dim=1)
    ids = routing_ids.to(device=device, dtype=torch.long)
    prototype_rows = prototypes.to(device=device)
    threshold_rows = thresholds.to(device=device)
    prediction_rows = prediction_location.to(device=device)
    k = max(1, min(int(k_routing), int(ids.numel())))
    bank_capacity = max(1, min(int(bank_size), int(ids.numel())))
    interval = max(1, int(refresh_interval))
    total_columns = max(
        int(prototype_rows.shape[0]),
        int(ids.max().item()) + 1 if int(ids.numel()) > 0 else 0,
    )
    positions_by_column = _position_map(ids, total_columns=total_columns)
    exact_scores = keys @ vectors.T
    exact_offsets = torch.topk(exact_scores, k=k, dim=1).indices
    exact_candidates = ids[exact_offsets]

    route_count = int(ids.numel())
    route_width = int(vectors.shape[1])
    offline_precompute_rows = route_count
    landmark_positions: torch.Tensor | None = None
    landmark_vectors: torch.Tensor | None = None
    landmark_buckets: torch.Tensor | None = None
    projection_planes: torch.Tensor | None = None
    projection_positive_buckets: torch.Tensor | None = None
    projection_negative_buckets: torch.Tensor | None = None

    bucket_limit = max(1, min(int(bucket_rows), route_count))
    if mode == "landmark_bucket":
        landmark_total = max(1, min(int(landmark_count or 1), route_count))
        top_count = max(1, min(int(top_landmarks or 1), landmark_total))
        landmark_positions = _select_farthest_landmarks(
            vectors,
            count=landmark_total,
        )
        landmark_vectors = vectors.index_select(0, landmark_positions)
        landmark_scores = landmark_vectors @ vectors.T
        landmark_buckets = torch.topk(
            landmark_scores,
            k=bucket_limit,
            dim=1,
        ).indices.to(dtype=torch.long)
        selector_rows = landmark_total
    else:
        projection_total = max(1, int(projection_count or route_width))
        top_count = max(1, min(int(top_projections or 1), projection_total))
        generator = torch.Generator(device=device).manual_seed(int(seed))
        projection_planes = F.normalize(
            torch.randn(
                projection_total,
                route_width,
                generator=generator,
                device=device,
            ),
            dim=1,
        )
        projection_scores = vectors @ projection_planes.T
        projection_positive_buckets = torch.topk(
            projection_scores.T,
            k=bucket_limit,
            dim=1,
        ).indices.to(dtype=torch.long)
        projection_negative_buckets = torch.topk(
            (-projection_scores).T,
            k=bucket_limit,
            dim=1,
        ).indices.to(dtype=torch.long)
        selector_rows = projection_total

    current_bank_ids = exact_candidates[0, :bank_capacity].clone()
    current_score_positions = positions_by_column.index_select(0, current_bank_ids)
    exact_previous = int(previous_winner)
    bank_previous = int(previous_winner)
    scored_since_refresh = interval
    refresh_count = 0
    exact_top1_in_candidates = 0
    exact_winner_match_count = 0
    positive_match_count = 0
    consecutive_top1_miss = 0
    worst_consecutive_top1_miss = 0
    overlap_fractions: list[float] = []
    score_rows: list[int] = []
    candidate_samples_exact: list[list[int]] = []
    candidate_samples_probe: list[list[int]] = []

    def _refresh_positions(tick: int, bank_ids: torch.Tensor) -> torch.Tensor:
        nonlocal refresh_count
        bank_positions = positions_by_column.index_select(0, bank_ids)
        if mode == "landmark_bucket":
            assert landmark_vectors is not None
            assert landmark_buckets is not None
            scores = keys[tick].unsqueeze(0) @ landmark_vectors.T
            selected = torch.topk(scores, k=top_count, dim=1).indices.flatten()
            expanded = torch.cat(
                (
                    bank_positions,
                    landmark_buckets.index_select(0, selected).flatten(),
                ),
                dim=0,
            )
        else:
            assert projection_planes is not None
            assert projection_positive_buckets is not None
            assert projection_negative_buckets is not None
            scores = keys[tick] @ projection_planes.T
            selected = torch.topk(
                scores.abs(),
                k=top_count,
            ).indices.flatten()
            buckets = []
            for offset, projection_idx in enumerate(selected.detach().cpu().tolist()):
                source = (
                    projection_positive_buckets
                    if bool(scores[selected[offset]].item() >= 0.0)
                    else projection_negative_buckets
                )
                buckets.append(source[int(projection_idx)])
            expanded = torch.cat((bank_positions, *buckets), dim=0)
        refresh_count += 1
        return _ordered_unique_limited_positions(
            expanded,
            limit=int(bank_capacity + top_count * bucket_limit),
        )

    for tick in range(int(keys.shape[0])):
        exact_row = exact_candidates[tick]
        seeded = tick == 0
        if seeded:
            probe_row = exact_row[:k].clone()
            rows_scored = route_count
            current_score_positions = _refresh_positions(tick, current_bank_ids)
            scored_since_refresh = interval
        else:
            if scored_since_refresh >= interval:
                current_score_positions = _refresh_positions(tick, current_bank_ids)
                scored_since_refresh = 0
            scores = keys[tick].unsqueeze(0) @ vectors.index_select(
                0,
                current_score_positions,
            ).T
            probe_row = _ordered_unique_limited_candidate_ids(
                scores=scores,
                positions=current_score_positions,
                routing_ids=ids,
                limit=k,
            )
            rows_scored = int(current_score_positions.numel())

        exact_set = {int(value) for value in exact_row.detach().cpu().tolist()}
        probe_set = {int(value) for value in probe_row.detach().cpu().tolist()}
        overlap_fractions.append(
            float(len(exact_set.intersection(probe_set))) / float(max(1, len(exact_set)))
        )
        exact_top1 = int(exact_row[0].item())
        if exact_top1 in probe_set:
            exact_top1_in_candidates += 1
            consecutive_top1_miss = 0
        else:
            consecutive_top1_miss += 1
            worst_consecutive_top1_miss = max(
                worst_consecutive_top1_miss,
                consecutive_top1_miss,
            )

        exact_winner, exact_positive = _candidate_winner(
            routing_key=keys[tick],
            candidates=exact_row,
            prototypes=prototype_rows,
            thresholds=threshold_rows,
            prediction_location=prediction_rows,
            previous_winner=exact_previous,
        )
        probe_winner, probe_positive = _candidate_winner(
            routing_key=keys[tick],
            candidates=probe_row,
            prototypes=prototype_rows,
            thresholds=threshold_rows,
            prediction_location=prediction_rows,
            previous_winner=bank_previous,
        )
        exact_winner_match_count += int(exact_winner == probe_winner)
        positive_match_count += int(exact_positive == probe_positive)
        exact_previous = exact_winner if exact_winner >= 0 else exact_previous
        bank_previous = probe_winner if probe_winner >= 0 else bank_previous
        score_rows.append(rows_scored)
        candidate_samples_exact.append(
            [int(value) for value in exact_row.detach().cpu().tolist()]
        )
        candidate_samples_probe.append(
            [int(value) for value in probe_row.detach().cpu().tolist()]
        )
        if not seeded:
            current_bank_ids = probe_row[:bank_capacity].clone()
            scored_since_refresh += 1

    checked = int(keys.shape[0])
    top1_rate = float(exact_top1_in_candidates) / float(max(1, checked))
    winner_rate = float(exact_winner_match_count) / float(max(1, checked))
    quality_passes = (
        top1_rate >= 0.95
        and winner_rate >= 0.95
        and int(worst_consecutive_top1_miss) <= 2
    )
    steady_rows = [
        float(value)
        for index, value in enumerate(score_rows)
        if index > 0
    ]
    return {
        "surface": "route_candidate_discovery_probe.v1",
        "mode": mode,
        "scope": "evaluation_only_bounded_candidate_discovery_probe",
        "claim_boundary": (
            "offline precompute may inspect the full routing cache, but simulated "
            "steady hot-path scoring uses fixed bounded candidate rows"
        ),
        "checked_ticks": checked,
        "total_columns": int(total_columns),
        "k_routing": int(k),
        "bank_size": int(bank_capacity),
        "refresh_interval_tokens": int(interval),
        "offline_precompute_rows": int(offline_precompute_rows),
        "hot_path_all_column_scan": False,
        "selector_rows_scored_on_refresh": int(selector_rows),
        "selector_rows_amortized_per_tick": float(selector_rows / interval),
        "bucket_rows": int(bucket_limit),
        "selected_bucket_count": int(top_count),
        "steady_route_score_rows": _float_stats(steady_rows),
        "quality": {
            "mean_topk_overlap_fraction": _float_stats(overlap_fractions)["mean"],
            "min_topk_overlap_fraction": _float_stats(overlap_fractions)["min"],
            "exact_top1_in_candidate_rate": top1_rate,
            "exact_winner_match_rate": winner_rate,
            "positive_branch_match_rate": float(positive_match_count)
            / float(max(1, checked)),
            "worst_consecutive_exact_top1_miss": int(
                worst_consecutive_top1_miss
            ),
        },
        "candidate_samples": {
            "exact_first": candidate_samples_exact[: min(5, checked)],
            "probe_first": candidate_samples_probe[: min(5, checked)],
            "exact_last": candidate_samples_exact[-min(5, checked) :],
            "probe_last": candidate_samples_probe[-min(5, checked) :],
        },
        "promotion_status": (
            "passes_bounded_discovery_quality_gate"
            if quality_passes
            else "rejected_until_fused_or_higher_quality_discovery"
        ),
    }


@torch.no_grad()
def run_discovery_probe(
    checkpoint: Path,
    *,
    mode: str,
    samples: int = 512,
    seed: int = 20260616,
    source_mode: str = "default_text",
    source_path: Path | None = None,
    bank_size: int = 0,
    refresh_interval: int = 16,
    landmark_count: int = 256,
    top_landmarks: int = 8,
    bucket_rows: int = 128,
    projection_count: int = 512,
    top_projections: int = 32,
) -> dict[str, Any]:
    trainer, metadata = load_trainer_checkpoint(checkpoint)
    device = trainer.model.device
    if source_mode == "random":
        patterns = _random_patterns(trainer, samples=samples, seed=seed)
        source_summary = {"mode": "random", "seed": int(seed)}
    else:
        if source_mode == "file":
            if source_path is None:
                raise ValueError("--source-path is required when --source-mode=file")
            source_text = Path(source_path).read_text(encoding="utf-8")
            source_name = str(source_path)
        elif source_path is None:
            source_text = DEFAULT_SOURCE_TEXT * max(1, int(samples // 256) + 1)
            source_name = "DEFAULT_SOURCE_TEXT"
        else:
            source_text = Path(source_path).read_text(encoding="utf-8")
            source_name = str(source_path)
        patterns = _text_patterns(
            trainer,
            source_text=source_text,
            samples=samples,
        )
        source_summary = {
            "mode": source_mode,
            "source": source_name,
            "characters": len(source_text),
        }

    routing_keys = torch.stack(
        [trainer.model.routing_key_from_pattern(pattern) for pattern in patterns]
    )
    routing_vectors, routing_ids = trainer.model.routing_index.routing_tensor_cache()
    report = evaluate_discovery_probe_from_tensors(
        mode=mode,
        routing_keys=routing_keys,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        prototypes=trainer.model.competitive.prototypes,
        thresholds=trainer.model.competitive.thresholds,
        prediction_location=trainer.model.predictive.location,
        k_routing=int(trainer.config.k_routing),
        bank_size=int(
            bank_size
            or trainer.config.route_candidate_bank_size
            or trainer.config.k_routing
        ),
        refresh_interval=int(refresh_interval),
        landmark_count=int(landmark_count),
        top_landmarks=int(top_landmarks),
        bucket_rows=int(bucket_rows),
        projection_count=int(projection_count),
        top_projections=int(top_projections),
        seed=int(seed),
        previous_winner=(
            0
            if trainer.last_winner is None
            else int(trainer.last_winner)
        ),
    )
    report.update(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_metadata": metadata,
            "device": {
                "type": device.type,
                "cuda_available": torch.cuda.is_available(),
                "name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else str(device)
                ),
            },
            "source": source_summary,
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--mode",
        choices=["landmark_bucket", "random_projection_bucket"],
        required=True,
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--source-mode",
        choices=["default_text", "file", "random"],
        default="default_text",
    )
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--bank-size", type=int, default=0)
    parser.add_argument("--refresh-interval", type=int, default=16)
    parser.add_argument("--landmark-count", type=int, default=256)
    parser.add_argument("--top-landmarks", type=int, default=8)
    parser.add_argument("--bucket-rows", type=int, default=128)
    parser.add_argument("--projection-count", type=int, default=512)
    parser.add_argument("--top-projections", type=int, default=32)
    args = parser.parse_args()
    if args.source_mode == "file" and args.source_path is None:
        parser.error("--source-path is required when --source-mode=file")
    report = run_discovery_probe(
        args.checkpoint,
        mode=args.mode,
        samples=args.samples,
        seed=args.seed,
        source_mode=args.source_mode,
        source_path=args.source_path if args.source_mode == "file" else None,
        bank_size=args.bank_size,
        refresh_interval=args.refresh_interval,
        landmark_count=args.landmark_count,
        top_landmarks=args.top_landmarks,
        bucket_rows=args.bucket_rows,
        projection_count=args.projection_count,
        top_projections=args.top_projections,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
