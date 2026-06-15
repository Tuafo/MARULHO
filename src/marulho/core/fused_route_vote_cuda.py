"""Triton exact-cache routing plus predictive column voting.

The production lifecycle remains checkpoint-opt-in and limits this kernel to
text/idle ticks. Sensory ticks retain ordinary tensor routing because grounded
benchmark evidence does not show a throughput benefit there.
"""

from __future__ import annotations

import torch

from marulho.core.inplace_column_cuda import ensure_windows_triton_compiler

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _routing_scores_kernel(
        routing_key,
        routing_vectors,
        scores_out,
        vector_count: tl.constexpr,
        column_dim: tl.constexpr,
        block_d: tl.constexpr,
    ):
        row = tl.program_id(0)
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < column_dim
        query = tl.load(
            routing_key + dimensions,
            mask=dimension_mask,
            other=0.0,
        )
        vector = tl.load(
            routing_vectors + row * column_dim + dimensions,
            mask=dimension_mask,
            other=0.0,
        )
        score = tl.sum(query * vector, axis=0)
        tl.store(scores_out + row, score, mask=row < vector_count)

    @triton.jit
    def _select_route_vote_kernel(
        routing_key,
        routing_scores,
        routing_ids,
        steps_since_win,
        prototypes,
        thresholds,
        prediction_location,
        previous_winner,
        route_filter_control,
        route_filter_state_out,
        candidates_out,
        winner_out,
        strength_out,
        competition_had_positive,
        reconstruction_error_out,
        vector_count: tl.constexpr,
        column_dim: tl.constexpr,
        location_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        block_location: tl.constexpr,
    ):
        score_offsets = tl.arange(0, block_n)
        score_mask = score_offsets < vector_count
        remaining_scores = tl.load(
            routing_scores + score_offsets,
            mask=score_mask,
            other=-float("inf"),
        )
        candidate_ids_by_position = tl.load(
            routing_ids + score_offsets,
            mask=score_mask,
            other=0,
        )
        filter_enabled = tl.load(route_filter_control) != 0
        deep_sleep_threshold = tl.load(route_filter_control + 1)
        candidate_steps = tl.load(
            steps_since_win + candidate_ids_by_position,
            mask=score_mask,
            other=0,
        )
        route_awake_mask = candidate_steps < deep_sleep_threshold
        eligible_count = tl.sum(
            tl.where(score_mask & route_awake_mask, 1, 0),
            axis=0,
        )
        deep_sleep_count = vector_count - eligible_count
        apply_filter = filter_enabled & (eligible_count >= candidate_count)
        primary_scores = tl.where(
            apply_filter & score_mask & (~route_awake_mask),
            -float("inf"),
            remaining_scores,
        )
        best_awake_route_score = tl.max(primary_scores, axis=0)
        best_unfiltered_route_score = tl.max(remaining_scores, axis=0)
        best_route_score = tl.where(
            apply_filter,
            best_awake_route_score,
            best_unfiltered_route_score,
        )

        dimension_offsets = tl.arange(0, block_d)
        dimension_mask = dimension_offsets < column_dim
        raw_query = tl.load(
            routing_key + dimension_offsets,
            mask=dimension_mask,
            other=0.0,
        )
        clamped_query = tl.maximum(raw_query, 1e-6)
        query_norm = tl.sqrt(tl.sum(clamped_query * clamped_query, axis=0))
        competition_query = clamped_query / tl.maximum(query_norm, 1e-8)

        prior_winner = tl.load(previous_winner)
        has_prior_winner = prior_winner >= 0
        safe_prior_winner = tl.maximum(prior_winner, 0)
        location_offsets = tl.arange(0, block_location)
        location_mask = location_offsets < location_dim
        prior_location = tl.load(
            prediction_location
            + safe_prior_winner * location_dim
            + location_offsets,
            mask=location_mask,
            other=0.0,
        )
        prior_norm = tl.sqrt(tl.sum(prior_location * prior_location, axis=0))

        best_positive_score = -float("inf")
        best_positive_id = 0
        best_combined_score = -float("inf")
        best_combined_id = 0

        for candidate_offset in tl.static_range(0, candidate_count):
            routing_position = tl.argmax(primary_scores, axis=0)
            candidate_id = tl.load(routing_ids + routing_position)
            tl.store(candidates_out + candidate_offset, candidate_id)
            primary_scores = tl.where(
                score_offsets == routing_position,
                -float("inf"),
                primary_scores,
            )

            prototype = tl.load(
                prototypes
                + candidate_id * column_dim
                + dimension_offsets,
                mask=dimension_mask,
                other=0.0,
            )
            similarity = tl.sum(prototype * competition_query, axis=0)

            candidate_location = tl.load(
                prediction_location
                + candidate_id * location_dim
                + location_offsets,
                mask=location_mask,
                other=0.0,
            )
            location_dot = tl.sum(candidate_location * prior_location, axis=0)
            candidate_norm = tl.sqrt(
                tl.sum(candidate_location * candidate_location, axis=0)
            )
            location_similarity = location_dot / tl.maximum(
                candidate_norm * prior_norm,
                1e-8,
            )
            location_similarity = tl.maximum(
                -1.0,
                tl.minimum(1.0, location_similarity),
            )
            consensus_gain = tl.where(
                has_prior_winner,
                1.0 + 0.3 * location_similarity,
                1.0,
            )
            combined = similarity * consensus_gain
            threshold = tl.load(thresholds + candidate_id)
            activation = tl.maximum(combined - threshold, 0.0)

            better_combined = combined > best_combined_score
            best_combined_score = tl.where(
                better_combined,
                combined,
                best_combined_score,
            )
            best_combined_id = tl.where(
                better_combined,
                candidate_id,
                best_combined_id,
            )
            better_positive = activation > best_positive_score
            best_positive_score = tl.where(
                better_positive,
                activation,
                best_positive_score,
            )
            best_positive_id = tl.where(
                better_positive,
                candidate_id,
                best_positive_id,
            )

        had_positive = best_positive_score > 0.0
        winner = tl.where(
            had_positive,
            best_positive_id,
            best_combined_id,
        )
        tl.store(winner_out, winner)
        tl.store(previous_winner, winner)
        tl.store(strength_out, 1.0)
        tl.store(competition_had_positive, had_positive)
        tl.store(
            reconstruction_error_out,
            tl.maximum(1.0 - best_route_score, 0.0),
        )
        filtered_count = tl.where(
            apply_filter,
            deep_sleep_count,
            0,
        )
        fallback_code = tl.where(
            filter_enabled & (~apply_filter) & (eligible_count <= 0),
            2,
            tl.where(
                filter_enabled & (~apply_filter),
                1,
                0,
            ),
        )
        tl.store(route_filter_state_out + 0, tl.where(filter_enabled, 1, 0))
        tl.store(route_filter_state_out + 1, tl.where(apply_filter, 1, 0))
        tl.store(route_filter_state_out + 2, filtered_count)
        tl.store(
            route_filter_state_out + 3,
            tl.where(filter_enabled, eligible_count, vector_count),
        )
        tl.store(route_filter_state_out + 4, fallback_code)
        tl.store(route_filter_state_out + 5, vector_count)
        tl.store(route_filter_state_out + 6, candidate_count)
        tl.store(
            route_filter_state_out + 7,
            0,
        )


def fused_route_vote_available() -> bool:
    return bool(triton is not None and torch.cuda.is_available())


def fused_route_vote_kernel_variant(
    routing_vectors: torch.Tensor,
    candidates_out: torch.Tensor,
) -> str:
    if triton is None or routing_vectors.dim() != 2:
        return "unavailable"
    vector_count, column_dim = map(int, routing_vectors.shape)
    candidate_count = int(candidates_out.numel())
    if (
        vector_count > 0
        and candidate_count > 0
        and candidate_count <= vector_count
        and column_dim > 0
    ):
        return "two_stage_route_vote"
    return "unavailable"


def warmup_fused_route_vote_cuda(
    *,
    routing_key: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    steps_since_win: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    previous_winner: torch.Tensor,
    route_filter_control: torch.Tensor,
    route_filter_state_out: torch.Tensor,
    scores_out: torch.Tensor,
    candidates_out: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
    reconstruction_error_out: torch.Tensor,
) -> None:
    """Compile the bounded route/vote shape without launching or mutating it."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    vector_count, column_dim = map(int, routing_vectors.shape)
    candidate_count = int(candidates_out.numel())
    ensure_windows_triton_compiler()
    _routing_scores_kernel.warmup(
        routing_key,
        routing_vectors,
        scores_out,
        vector_count=vector_count,
        column_dim=column_dim,
        block_d=triton.next_power_of_2(column_dim),
        num_warps=1,
        grid=(vector_count,),
    )
    _select_route_vote_kernel.warmup(
        routing_key,
        scores_out,
        routing_ids,
        steps_since_win,
        prototypes,
        thresholds,
        prediction_location,
        previous_winner,
        route_filter_control,
        route_filter_state_out,
        candidates_out,
        winner_out,
        strength_out,
        competition_had_positive,
        reconstruction_error_out,
        vector_count=vector_count,
        column_dim=column_dim,
        location_dim=int(prediction_location.shape[1]),
        candidate_count=candidate_count,
        block_n=triton.next_power_of_2(vector_count),
        block_d=triton.next_power_of_2(column_dim),
        block_location=triton.next_power_of_2(int(prediction_location.shape[1])),
        num_warps=8,
        grid=(1,),
    )


def fused_route_vote_cuda(
    *,
    routing_key: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    steps_since_win: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    previous_winner: torch.Tensor,
    route_filter_control: torch.Tensor,
    route_filter_state_out: torch.Tensor,
    scores_out: torch.Tensor,
    candidates_out: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
    reconstruction_error_out: torch.Tensor,
) -> None:
    """Run the production-owned two-launch exact text route/vote kernel."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not routing_key.is_cuda:
        raise ValueError("fused route/vote requires CUDA tensors")
    tensors = (
        routing_key,
        routing_vectors,
        routing_ids,
        steps_since_win,
        prototypes,
        thresholds,
        prediction_location,
        previous_winner,
        route_filter_control,
        route_filter_state_out,
        scores_out,
        candidates_out,
        winner_out,
        strength_out,
        competition_had_positive,
        reconstruction_error_out,
    )
    if any(tensor.device != routing_key.device for tensor in tensors):
        raise ValueError("all fused route/vote tensors must share one CUDA device")
    if routing_vectors.dim() != 2:
        raise ValueError("routing_vectors must be rank 2")
    vector_count, column_dim = map(int, routing_vectors.shape)
    if vector_count <= 0:
        raise ValueError("routing cache must not be empty")
    if int(routing_key.numel()) != column_dim:
        raise ValueError("routing_key must match routing vector width")
    if int(routing_ids.numel()) != vector_count:
        raise ValueError("routing_ids must match routing vector rows")
    if int(steps_since_win.numel()) != int(prototypes.shape[0]):
        raise ValueError("steps_since_win must match prototype rows")
    if steps_since_win.dtype != torch.long:
        raise ValueError("steps_since_win must use int64")
    if route_filter_control.dtype != torch.long or int(route_filter_control.numel()) < 2:
        raise ValueError("route_filter_control must contain at least two int64 values")
    if route_filter_state_out.dtype != torch.long or int(route_filter_state_out.numel()) < 8:
        raise ValueError("route_filter_state_out must contain at least eight int64 values")
    if int(scores_out.numel()) != vector_count:
        raise ValueError("scores_out must match routing vector rows")
    candidate_count = int(candidates_out.numel())
    if candidate_count <= 0 or candidate_count > vector_count:
        raise ValueError("candidate output count must be within routing cache size")
    if int(prototypes.shape[1]) != column_dim:
        raise ValueError("prototype width must match routing vector width")
    if int(thresholds.numel()) != int(prototypes.shape[0]):
        raise ValueError("thresholds must match prototype rows")
    if int(prediction_location.shape[0]) != int(prototypes.shape[0]):
        raise ValueError("prediction locations must match prototype rows")
    if previous_winner.dtype != torch.long or int(previous_winner.numel()) != 1:
        raise ValueError("previous_winner must be one int64 value")
    if candidates_out.dtype != torch.long:
        raise ValueError("candidates_out must use int64")
    if int(reconstruction_error_out.numel()) != 1:
        raise ValueError("reconstruction_error_out must be one scalar value")

    ensure_windows_triton_compiler()
    _routing_scores_kernel[(vector_count,)](
        routing_key,
        routing_vectors,
        scores_out,
        vector_count=vector_count,
        column_dim=column_dim,
        block_d=triton.next_power_of_2(column_dim),
        num_warps=1,
    )
    _select_route_vote_kernel[(1,)](
        routing_key,
        scores_out,
        routing_ids,
        steps_since_win,
        prototypes,
        thresholds,
        prediction_location,
        previous_winner,
        route_filter_control,
        route_filter_state_out,
        candidates_out,
        winner_out,
        strength_out,
        competition_had_positive,
        reconstruction_error_out,
        vector_count=vector_count,
        column_dim=column_dim,
        location_dim=int(prediction_location.shape[1]),
        candidate_count=candidate_count,
        block_n=triton.next_power_of_2(vector_count),
        block_d=triton.next_power_of_2(column_dim),
        block_location=triton.next_power_of_2(int(prediction_location.shape[1])),
        num_warps=8,
    )
