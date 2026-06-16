"""Evaluate bounded route-bank quality against an exact routing oracle.

This is an evaluation gate, not a runtime scheduler. It may score the full
routing cache as an oracle, but the report keeps that work outside the hot path
and separates the explicit seed from steady bank scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable
import json

import torch
import torch.nn.functional as F

from marulho.evaluation.continuous_runtime_quantum_benchmark import (
    DEFAULT_SOURCE_TEXT,
)
from marulho.training.checkpointing import load_trainer_checkpoint


def _float_stats(values: Iterable[float]) -> dict[str, float | int | None]:
    materialized = [float(value) for value in values]
    if not materialized:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(materialized),
        "min": float(min(materialized)),
        "mean": float(sum(materialized) / max(1, len(materialized))),
        "max": float(max(materialized)),
    }


def _position_map(
    routing_ids: torch.Tensor,
    *,
    total_columns: int,
) -> torch.Tensor:
    positions = torch.full(
        (max(1, int(total_columns)),),
        -1,
        dtype=torch.long,
        device=routing_ids.device,
    )
    if int(routing_ids.numel()) > 0:
        positions[routing_ids.long()] = torch.arange(
            int(routing_ids.numel()),
            dtype=torch.long,
            device=routing_ids.device,
        )
    return positions


def _ordered_unique_limited_positions(
    positions: torch.Tensor,
    *,
    limit: int,
) -> torch.Tensor:
    if int(limit) <= 0 or int(positions.numel()) <= 0:
        return torch.empty((0,), dtype=torch.long, device=positions.device)
    seen: set[int] = set()
    kept: list[int] = []
    for value in positions.detach().cpu().tolist():
        position = int(value)
        if position < 0 or position in seen:
            continue
        seen.add(position)
        kept.append(position)
        if len(kept) >= int(limit):
            break
    return torch.tensor(kept, dtype=torch.long, device=positions.device)


def _ordered_new_limited_positions(
    positions: torch.Tensor,
    *,
    seen: set[int],
    limit: int,
) -> torch.Tensor:
    if int(limit) <= 0 or int(positions.numel()) <= 0:
        return torch.empty((0,), dtype=torch.long, device=positions.device)
    kept: list[int] = []
    for value in positions.detach().cpu().tolist():
        position = int(value)
        if position < 0 or position in seen:
            continue
        seen.add(position)
        kept.append(position)
        if len(kept) >= int(limit):
            break
    return torch.tensor(kept, dtype=torch.long, device=positions.device)


def _ordered_unique_limited_candidate_ids(
    *,
    scores: torch.Tensor,
    positions: torch.Tensor,
    routing_ids: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    if int(limit) <= 0 or int(positions.numel()) <= 0:
        return torch.empty((0,), dtype=torch.long, device=routing_ids.device)
    ordered_offsets = torch.argsort(scores.flatten(), descending=True)
    seen: set[int] = set()
    kept: list[int] = []
    for offset in ordered_offsets.detach().cpu().tolist():
        position = int(positions[int(offset)].item())
        if position < 0:
            continue
        candidate_id = int(routing_ids[position].item())
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        kept.append(candidate_id)
        if len(kept) >= int(limit):
            break
    return torch.tensor(kept, dtype=torch.long, device=routing_ids.device)


def _candidate_winner(
    *,
    routing_key: torch.Tensor,
    candidates: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    previous_winner: int,
) -> tuple[int, bool]:
    if int(candidates.numel()) <= 0:
        return -1, False
    previous = max(0, min(int(previous_winner), int(prototypes.shape[0]) - 1))
    key = F.normalize(routing_key.clamp(min=1e-6), dim=0)
    candidate_ids = candidates.to(dtype=torch.long)
    location_similarity = F.cosine_similarity(
        prediction_location.index_select(0, candidate_ids),
        prediction_location[previous].unsqueeze(0),
        dim=1,
    )
    combined = (
        prototypes.index_select(0, candidate_ids) @ key
    ) * (1.0 + 0.3 * location_similarity.clamp(-1.0, 1.0))
    activation = torch.relu(combined - thresholds.index_select(0, candidate_ids))
    positive = bool((activation.max() > 0.0).item())
    local_index = torch.argmax(activation if positive else combined)
    return int(candidate_ids[local_index].item()), positive


@torch.no_grad()
def evaluate_route_candidate_bank_quality_from_tensors(
    *,
    routing_keys: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    k_routing: int,
    bank_size: int,
    previous_winner: int = 0,
    exact_reseed_interval: int = 0,
    route_candidate_graph_neighbor_count: int = 0,
    route_candidate_graph_capacity_rows: int = 0,
    route_candidate_graph_walk_beam: int = 0,
    route_candidate_graph_walk_rounds: int = 0,
    route_candidate_probe_rows: int = 0,
    route_candidate_bank_refresh_interval: int = 1,
) -> dict[str, Any]:
    if routing_keys.dim() != 2:
        raise ValueError("routing_keys must be a rank-2 tensor")
    if routing_vectors.dim() != 2:
        raise ValueError("routing_vectors must be a rank-2 tensor")
    if int(routing_vectors.shape[1]) != int(routing_keys.shape[1]):
        raise ValueError("routing key width must match routing vector width")
    if int(routing_ids.numel()) != int(routing_vectors.shape[0]):
        raise ValueError("routing_ids must match routing vector rows")
    if int(routing_keys.shape[0]) <= 0:
        raise ValueError("at least one routing key is required")

    device = routing_vectors.device
    keys = F.normalize(routing_keys.to(device=device), dim=1)
    vectors = F.normalize(routing_vectors.to(device=device), dim=1)
    ids = routing_ids.to(device=device, dtype=torch.long)
    prototype_rows = prototypes.to(device=device)
    threshold_rows = thresholds.to(device=device)
    prediction_rows = prediction_location.to(device=device)
    k = max(1, min(int(k_routing), int(ids.numel())))
    bank_capacity = max(1, min(int(bank_size), int(ids.numel())))
    total_columns = max(
        int(prototype_rows.shape[0]),
        int(ids.max().item()) + 1 if int(ids.numel()) > 0 else 0,
    )
    positions_by_column = _position_map(ids, total_columns=total_columns)
    graph_neighbor_count = min(
        max(0, int(route_candidate_graph_neighbor_count)),
        max(0, int(ids.numel()) - 1),
    )
    graph_enabled = graph_neighbor_count > 0
    probe_rows = min(
        max(0, int(route_candidate_probe_rows)),
        max(0, int(ids.numel()) - int(bank_capacity)),
    )
    probe_enabled = probe_rows > 0
    refresh_interval = max(1, int(route_candidate_bank_refresh_interval))
    probe_offsets = torch.arange(
        int(probe_rows),
        dtype=torch.long,
        device=device,
    )
    probe_cursor = int(bank_capacity)
    probe_refresh_count = 0
    probe_deferred_tick_count = 0
    default_graph_capacity = int(bank_capacity * (graph_neighbor_count + 1))
    graph_capacity = int(
        min(
            int(ids.numel()),
            max(
                int(bank_capacity),
                int(route_candidate_graph_capacity_rows or default_graph_capacity),
            ),
        )
    )
    graph_walk_beam = min(
        max(0, int(route_candidate_graph_walk_beam)),
        int(ids.numel()),
    )
    graph_walk_rounds = max(0, int(route_candidate_graph_walk_rounds))
    graph_walk_enabled = bool(
        graph_enabled and graph_walk_beam > 0 and graph_walk_rounds > 0
    )
    neighbor_positions: torch.Tensor | None = None
    if graph_enabled:
        neighbor_scores = vectors @ vectors.T
        neighbor_scores.fill_diagonal_(-float("inf"))
        neighbor_positions = torch.topk(
            neighbor_scores,
            k=graph_neighbor_count,
            dim=1,
        ).indices.to(dtype=torch.long)

    exact_scores = keys @ vectors.T
    exact_offsets = torch.topk(exact_scores, k=k, dim=1).indices
    exact_candidates = ids[exact_offsets]

    bank_candidates_by_tick: list[list[int]] = []
    exact_candidates_by_tick: list[list[int]] = []
    overlap_fractions: list[float] = []
    exact_top1_in_bank_count = 0
    exact_winner_match_count = 0
    positive_match_count = 0
    consecutive_top1_miss = 0
    worst_consecutive_top1_miss = 0
    bank_score_rows: list[int] = []
    graph_valid_rows: list[int] = []
    graph_walk_valid_rows: list[int] = []
    seed_count = 0
    reseed_count = 0
    probe_exact_top1_seen_count = 0
    probe_exact_top1_discovery_count = 0

    current_bank_ids = exact_candidates[0, :bank_capacity].clone()
    exact_previous = int(previous_winner)
    bank_previous = int(previous_winner)
    scored_since_refresh = 0
    current_score_positions: torch.Tensor | None = None
    current_bank_positions = positions_by_column.index_select(0, current_bank_ids)
    current_probe_positions = torch.empty((0,), dtype=torch.long, device=device)

    def _graph_walk_score_positions(
        routing_key: torch.Tensor,
        seed_positions: torch.Tensor,
    ) -> torch.Tensor:
        assert neighbor_positions is not None
        seen_positions: set[int] = set()
        score_position_parts: list[torch.Tensor] = []
        frontier = seed_positions
        remaining = int(graph_capacity)
        for walk_round in range(int(graph_walk_rounds) + 1):
            current_frontier = _ordered_new_limited_positions(
                frontier,
                seen=seen_positions,
                limit=remaining,
            )
            if int(current_frontier.numel()) <= 0:
                break
            score_position_parts.append(current_frontier)
            remaining -= int(current_frontier.numel())
            if remaining <= 0 or walk_round >= int(graph_walk_rounds):
                break
            frontier_scores = routing_key.unsqueeze(0) @ vectors.index_select(
                0,
                current_frontier,
            ).T
            parent_count = min(int(graph_walk_beam), int(current_frontier.numel()))
            if parent_count <= 0:
                break
            parent_offsets = torch.topk(
                frontier_scores.flatten(),
                k=parent_count,
            ).indices
            parent_positions = current_frontier.index_select(0, parent_offsets)
            frontier = neighbor_positions.index_select(
                0,
                parent_positions,
            ).flatten()
        if not score_position_parts:
            return torch.empty((0,), dtype=torch.long, device=device)
        return torch.cat(score_position_parts, dim=0)

    def _refresh_score_positions(bank_ids: torch.Tensor) -> None:
        nonlocal current_score_positions
        nonlocal current_bank_positions
        nonlocal current_probe_positions
        nonlocal probe_cursor
        nonlocal probe_refresh_count
        current_bank_positions = positions_by_column.index_select(0, bank_ids)
        if bool((current_bank_positions < 0).any().item()):
            raise ValueError("route bank contains id missing from routing cache")
        if not probe_enabled:
            current_probe_positions = torch.empty(
                (0,),
                dtype=torch.long,
                device=device,
            )
            current_score_positions = current_bank_positions
            return
        route_count = int(ids.numel())
        current_probe_positions = torch.remainder(
            probe_offsets + int(probe_cursor),
            route_count,
        ).to(dtype=torch.long)
        probe_cursor = (int(probe_cursor) + int(probe_rows)) % route_count
        probe_refresh_count += 1
        current_score_positions = torch.cat(
            (current_bank_positions, current_probe_positions),
            dim=0,
        )

    for tick in range(int(keys.shape[0])):
        exact_row = exact_candidates[tick]
        seeded = tick == 0 or (
            int(exact_reseed_interval) > 0
            and tick > 0
            and tick % int(exact_reseed_interval) == 0
        )
        if seeded:
            bank_row = exact_row[:k].clone()
            current_bank_ids = exact_row[:bank_capacity].clone()
            bank_rows_scored = int(ids.numel())
            _refresh_score_positions(current_bank_ids)
            scored_since_refresh = 0
            if tick == 0:
                seed_count += 1
            else:
                reseed_count += 1
        else:
            bank_positions = current_bank_positions
            if graph_enabled:
                assert neighbor_positions is not None
                graph_seed_positions = (
                    current_score_positions
                    if current_score_positions is not None and probe_enabled
                    else bank_positions
                )
                if graph_walk_enabled:
                    expanded_positions = _graph_walk_score_positions(
                        keys[tick],
                        graph_seed_positions,
                    )
                    graph_walk_valid_rows.append(int(expanded_positions.numel()))
                else:
                    expanded_positions = torch.cat(
                        (
                            graph_seed_positions,
                            neighbor_positions.index_select(
                                0,
                                graph_seed_positions,
                            ).flatten(),
                        ),
                        dim=0,
                    )
                    expanded_positions = _ordered_unique_limited_positions(
                        expanded_positions,
                        limit=graph_capacity,
                    )
                bank_scores = keys[tick].unsqueeze(0) @ vectors.index_select(
                    0,
                    expanded_positions,
                ).T
                bank_row = _ordered_unique_limited_candidate_ids(
                    scores=bank_scores,
                    positions=expanded_positions,
                    routing_ids=ids,
                    limit=k,
                )
                bank_rows_scored = int(expanded_positions.numel())
                graph_valid_rows.append(bank_rows_scored)
            else:
                score_positions = (
                    current_score_positions
                    if current_score_positions is not None
                    else bank_positions
                )
                bank_scores = keys[tick].unsqueeze(0) @ vectors.index_select(
                    0,
                    score_positions,
                ).T
                bank_row = _ordered_unique_limited_candidate_ids(
                    scores=bank_scores,
                    positions=score_positions,
                    routing_ids=ids,
                    limit=k,
                )
                bank_rows_scored = int(score_positions.numel())

        exact_set = {int(value) for value in exact_row.detach().cpu().tolist()}
        bank_set = {int(value) for value in bank_row.detach().cpu().tolist()}
        bank_source_set = {
            int(value)
            for value in ids.index_select(0, current_bank_positions)
            .detach()
            .cpu()
            .tolist()
        }
        probe_source_set = {
            int(value)
            for value in ids.index_select(0, current_probe_positions)
            .detach()
            .cpu()
            .tolist()
        }
        overlap = len(exact_set.intersection(bank_set))
        overlap_fractions.append(float(overlap) / float(max(1, len(exact_set))))
        exact_top1 = int(exact_row[0].item())
        if exact_top1 in bank_set:
            exact_top1_in_bank_count += 1
            consecutive_top1_miss = 0
        else:
            consecutive_top1_miss += 1
            worst_consecutive_top1_miss = max(
                worst_consecutive_top1_miss,
                consecutive_top1_miss,
            )
        if not seeded and probe_enabled:
            if exact_top1 in probe_source_set and exact_top1 not in bank_source_set:
                probe_exact_top1_seen_count += 1
                if exact_top1 in bank_set:
                    probe_exact_top1_discovery_count += 1

        exact_winner, exact_positive = _candidate_winner(
            routing_key=keys[tick],
            candidates=exact_row,
            prototypes=prototype_rows,
            thresholds=threshold_rows,
            prediction_location=prediction_rows,
            previous_winner=exact_previous,
        )
        bank_winner, bank_positive = _candidate_winner(
            routing_key=keys[tick],
            candidates=bank_row,
            prototypes=prototype_rows,
            thresholds=threshold_rows,
            prediction_location=prediction_rows,
            previous_winner=bank_previous,
        )
        exact_winner_match_count += int(exact_winner == bank_winner)
        positive_match_count += int(exact_positive == bank_positive)
        exact_previous = exact_winner if exact_winner >= 0 else exact_previous
        bank_previous = bank_winner if bank_winner >= 0 else bank_previous
        exact_candidates_by_tick.append([int(value) for value in exact_row.detach().cpu().tolist()])
        bank_candidates_by_tick.append([int(value) for value in bank_row.detach().cpu().tolist()])
        bank_score_rows.append(bank_rows_scored)
        if not seeded:
            scored_since_refresh += 1
            if scored_since_refresh >= refresh_interval:
                current_bank_ids = bank_row[:bank_capacity].clone()
                _refresh_score_positions(current_bank_ids)
                scored_since_refresh = 0
            elif probe_enabled:
                probe_deferred_tick_count += 1

    checked = int(keys.shape[0])
    steady_ticks = max(0, checked - seed_count - reseed_count)
    top1_rate = float(exact_top1_in_bank_count) / float(max(1, checked))
    winner_rate = float(exact_winner_match_count) / float(max(1, checked))
    positive_rate = float(positive_match_count) / float(max(1, checked))
    quality_passes = (
        top1_rate >= 0.95
        and winner_rate >= 0.95
        and int(worst_consecutive_top1_miss) <= 2
    )
    steady_bank_rows = [
        float(value)
        for index, value in enumerate(bank_score_rows)
        if index > 0
        and not (
            int(exact_reseed_interval) > 0
            and index % int(exact_reseed_interval) == 0
        )
    ]
    return {
        "surface": "route_candidate_bank_quality_gate.v1",
        "scope": (
            "evaluation_only_exact_oracle_compared_to_training_owned_"
            "route_candidate_bank_plus_optional_probe_lane"
        ),
        "claim_boundary": (
            "scores full route cache only as an offline oracle; runtime steady "
            "route bank/probe lane still scores bounded rows after the explicit seed"
        ),
        "checked_ticks": checked,
        "steady_ticks": steady_ticks,
        "total_columns": int(total_columns),
        "k_routing": int(k),
        "bank_size": int(bank_capacity),
        "exact_seed_count": int(seed_count),
        "exact_reseed_count": int(reseed_count),
        "exact_reseed_interval": int(exact_reseed_interval),
        "hot_path_all_column_oracle": False,
        "offline_oracle_score_rows_per_tick": int(ids.numel()),
        "evaluation_exact_reseed_only": int(exact_reseed_interval) > 0,
        "route_candidate_graph": {
            "enabled": bool(graph_enabled),
            "mode": (
                "bounded_walk"
                if graph_walk_enabled
                else "one_hop_capacity"
                if graph_enabled
                else "disabled"
            ),
            "neighbor_count": int(graph_neighbor_count),
            "capacity_rows": int(graph_capacity if graph_enabled else bank_capacity),
            "default_capacity_rows": int(
                default_graph_capacity if graph_enabled else bank_capacity
            ),
            "valid_rows": _float_stats(graph_valid_rows),
            "offline_precompute_rows": int(ids.numel() if graph_enabled else 0),
            "hot_path_precompute": False,
            "claim_boundary": (
                "offline_quality_simulation_of_bounded_neighbor_expansion_without_hot_path_all_column_scan"
            ),
        },
        "route_candidate_graph_walk": {
            "enabled": bool(graph_walk_enabled),
            "beam": int(graph_walk_beam if graph_enabled else 0),
            "rounds": int(graph_walk_rounds if graph_enabled else 0),
            "capacity_rows": int(graph_capacity if graph_walk_enabled else 0),
            "seed_rows": int(
                (bank_capacity + probe_rows) if graph_walk_enabled else 0
            ),
            "valid_rows": _float_stats(graph_walk_valid_rows),
            "offline_precompute_rows": int(ids.numel() if graph_walk_enabled else 0),
            "hot_path_precompute": False,
            "claim_boundary": (
                "offline_quality_simulation_of_fixed_degree_graph_walk_without_hot_path_all_column_scan"
            ),
        },
        "route_candidate_probe": {
            "enabled": bool(probe_enabled),
            "probe_rows": int(probe_rows),
            "refresh_interval_tokens": int(refresh_interval),
            "refresh_count": int(probe_refresh_count),
            "deferred_tick_count": int(probe_deferred_tick_count),
            "exact_top1_seen_in_probe_rows": int(probe_exact_top1_seen_count),
            "exact_top1_discovered_from_probe_rows": int(
                probe_exact_top1_discovery_count
            ),
            "exact_top1_probe_seen_rate": float(
                probe_exact_top1_seen_count / max(1, steady_ticks)
            ),
            "exact_top1_probe_discovery_rate": float(
                probe_exact_top1_discovery_count / max(1, steady_ticks)
            ),
            "hot_path_all_column_probe_scan": False,
            "claim_boundary": (
                "offline_quality_simulation_of_live_fixed_probe_lane_without_hot_path_all_column_scan"
            ),
        },
        "steady_bank_score_rows": _float_stats(steady_bank_rows),
        "steady_route_score_rows": _float_stats(steady_bank_rows),
        "oracle_score_rows": int(ids.numel()),
        "quality": {
            "mean_topk_overlap_fraction": _float_stats(overlap_fractions)["mean"],
            "min_topk_overlap_fraction": _float_stats(overlap_fractions)["min"],
            "exact_top1_in_bank_rate": top1_rate,
            "exact_winner_match_rate": winner_rate,
            "positive_branch_match_rate": positive_rate,
            "worst_consecutive_exact_top1_miss": int(worst_consecutive_top1_miss),
        },
        "candidate_samples": {
            "exact_first": exact_candidates_by_tick[: min(5, len(exact_candidates_by_tick))],
            "bank_first": bank_candidates_by_tick[: min(5, len(bank_candidates_by_tick))],
            "exact_last": exact_candidates_by_tick[-min(5, len(exact_candidates_by_tick)) :],
            "bank_last": bank_candidates_by_tick[-min(5, len(bank_candidates_by_tick)) :],
        },
        "promotion_status": (
            "passes_real_source_route_bank_quality_gate"
            if quality_passes
            else "graph_walk_bounded_but_requires_stronger_discovery_router_before_quality_claim"
            if graph_walk_enabled
            else "probe_lane_bounded_but_requires_stronger_discovery_router_before_quality_claim"
            if probe_enabled
            else "requires_reseed_policy_or_wider_bank_before_quality_claim"
        ),
    }


def _text_patterns(
    trainer: Any,
    *,
    source_text: str,
    samples: int,
) -> torch.Tensor:
    windows = []
    for _raw_window, pattern in trainer.encoder.iter_char_patterns(
        source_text,
        int(trainer.config.window_size),
        learn=False,
    ):
        windows.append(pattern.to(trainer.model.device))
        if len(windows) >= int(samples):
            break
    if len(windows) < int(samples):
        raise RuntimeError(
            f"source text yielded {len(windows)} patterns, need {int(samples)}"
        )
    return torch.stack(windows)


def _random_patterns(trainer: Any, *, samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=trainer.model.device).manual_seed(int(seed))
    return torch.rand(
        int(samples),
        int(trainer.config.input_dim),
        generator=generator,
        device=trainer.model.device,
    )


@torch.no_grad()
def run_route_candidate_bank_quality_gate(
    checkpoint: Path,
    *,
    samples: int = 512,
    seed: int = 20260616,
    source_mode: str = "default_text",
    source_path: Path | None = None,
    bank_size: int = 0,
    exact_reseed_interval: int = 0,
    route_candidate_graph_neighbor_count: int = 0,
    route_candidate_graph_capacity_rows: int = 0,
    route_candidate_graph_walk_beam: int = 0,
    route_candidate_graph_walk_rounds: int = 0,
    route_candidate_probe_rows: int = 2,
    route_candidate_bank_refresh_interval: int = 16,
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
    quality = evaluate_route_candidate_bank_quality_from_tensors(
        routing_keys=routing_keys,
        routing_vectors=routing_vectors,
        routing_ids=routing_ids,
        prototypes=trainer.model.competitive.prototypes,
        thresholds=trainer.model.competitive.thresholds,
        prediction_location=trainer.model.predictive.location,
        k_routing=int(trainer.config.k_routing),
        bank_size=int(bank_size or trainer.config.k_routing),
        previous_winner=0 if trainer.last_winner is None else int(trainer.last_winner),
        exact_reseed_interval=int(exact_reseed_interval),
        route_candidate_graph_neighbor_count=int(route_candidate_graph_neighbor_count),
        route_candidate_graph_capacity_rows=int(route_candidate_graph_capacity_rows),
        route_candidate_graph_walk_beam=int(route_candidate_graph_walk_beam),
        route_candidate_graph_walk_rounds=int(route_candidate_graph_walk_rounds),
        route_candidate_probe_rows=int(route_candidate_probe_rows),
        route_candidate_bank_refresh_interval=int(
            route_candidate_bank_refresh_interval
        ),
    )
    quality.update(
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
    return quality


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument(
        "--source-mode",
        choices=["default_text", "file", "random"],
        default="default_text",
    )
    parser.add_argument("--source-path", type=Path)
    parser.add_argument("--bank-size", type=int, default=0)
    parser.add_argument("--exact-reseed-interval", type=int, default=0)
    parser.add_argument("--route-candidate-graph-neighbor-count", type=int, default=0)
    parser.add_argument("--route-candidate-graph-capacity-rows", type=int, default=0)
    parser.add_argument("--route-candidate-graph-walk-beam", type=int, default=0)
    parser.add_argument("--route-candidate-graph-walk-rounds", type=int, default=0)
    parser.add_argument("--route-candidate-probe-rows", type=int, default=2)
    parser.add_argument(
        "--route-candidate-bank-refresh-interval",
        type=int,
        default=16,
    )
    args = parser.parse_args()
    if args.source_mode == "file" and args.source_path is None:
        parser.error("--source-path is required when --source-mode=file")
    source_path = args.source_path if args.source_mode == "file" else None
    report = run_route_candidate_bank_quality_gate(
        args.checkpoint,
        samples=args.samples,
        seed=args.seed,
        source_mode=args.source_mode,
        source_path=source_path,
        bank_size=args.bank_size,
        exact_reseed_interval=args.exact_reseed_interval,
        route_candidate_graph_neighbor_count=(
            args.route_candidate_graph_neighbor_count
        ),
        route_candidate_graph_capacity_rows=(
            args.route_candidate_graph_capacity_rows
        ),
        route_candidate_graph_walk_beam=args.route_candidate_graph_walk_beam,
        route_candidate_graph_walk_rounds=args.route_candidate_graph_walk_rounds,
        route_candidate_probe_rows=args.route_candidate_probe_rows,
        route_candidate_bank_refresh_interval=(
            args.route_candidate_bank_refresh_interval
        ),
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
