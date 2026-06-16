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
        route_positions,
        scores_out,
        vector_count: tl.constexpr,
        column_dim: tl.constexpr,
        block_d: tl.constexpr,
        has_route_positions: tl.constexpr,
    ):
        row = tl.program_id(0)
        vector_row = tl.load(route_positions + row) if has_route_positions else row
        dimensions = tl.arange(0, block_d)
        dimension_mask = dimensions < column_dim
        query = tl.load(
            routing_key + dimensions,
            mask=dimension_mask,
            other=0.0,
        )
        vector = tl.load(
            routing_vectors + vector_row * column_dim + dimensions,
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
        route_positions,
        route_bank_positions_out,
        route_probe_cursor,
        steps_since_win,
        steps_since_win_last_update_step,
        state_transition_step_counter,
        state_transition_all_materialized_step,
        prototypes,
        thresholds,
        prediction_location,
        memory_pressure,
        usefulness,
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
        has_route_positions: tl.constexpr,
        refresh_route_positions: tl.constexpr,
        probe_rows: tl.constexpr,
        route_count: tl.constexpr,
    ):
        score_offsets = tl.arange(0, block_n)
        score_mask = score_offsets < vector_count
        remaining_scores = tl.load(
            routing_scores + score_offsets,
            mask=score_mask,
            other=-float("inf"),
        )
        routing_positions = tl.load(
            route_positions + score_offsets,
            mask=score_mask,
            other=0,
        )
        routing_positions = tl.where(
            has_route_positions,
            routing_positions,
            score_offsets,
        )
        candidate_ids_by_position = tl.load(
            routing_ids + routing_positions,
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
        candidate_last_update_step = tl.load(
            steps_since_win_last_update_step + candidate_ids_by_position,
            mask=score_mask,
            other=0,
        )
        current_state_step = tl.load(state_transition_step_counter)
        all_materialized_step = tl.load(state_transition_all_materialized_step)
        effective_last_update_step = tl.maximum(
            candidate_last_update_step,
            all_materialized_step,
        )
        candidate_steps = candidate_steps + tl.maximum(
            current_state_step - effective_last_update_step,
            0,
        )
        candidate_pressure = tl.load(
            memory_pressure + candidate_ids_by_position,
            mask=score_mask,
            other=0.0,
        )
        candidate_usefulness = tl.load(
            usefulness + candidate_ids_by_position,
            mask=score_mask,
            other=1.0,
        )
        route_awake_mask = candidate_steps < deep_sleep_threshold
        pressure_enabled = tl.load(route_filter_control + 2) != 0
        pressure_threshold = (
            tl.load(route_filter_control + 3).to(tl.float32) * 0.000001
        )
        route_pressure_mask = candidate_pressure <= pressure_threshold
        usefulness_enabled = tl.load(route_filter_control + 4) != 0
        usefulness_threshold = (
            tl.load(route_filter_control + 5).to(tl.float32) * 0.000001
        )
        route_usefulness_mask = candidate_usefulness >= usefulness_threshold
        deep_eligible_mask = score_mask & tl.where(
            filter_enabled,
            route_awake_mask,
            True,
        )
        eligible_count = tl.sum(
            tl.where(deep_eligible_mask, 1, 0),
            axis=0,
        )
        deep_sleep_count = vector_count - eligible_count
        final_eligible_mask = deep_eligible_mask & tl.where(
            pressure_enabled,
            route_pressure_mask,
            True,
        )
        pressure_eligible_count = tl.sum(
            tl.where(final_eligible_mask, 1, 0),
            axis=0,
        )
        pressure_filtered_count = tl.where(
            pressure_enabled,
            eligible_count - pressure_eligible_count,
            0,
        )
        usefulness_eligible_mask = final_eligible_mask & tl.where(
            usefulness_enabled,
            route_usefulness_mask,
            True,
        )
        usefulness_eligible_count = tl.sum(
            tl.where(usefulness_eligible_mask, 1, 0),
            axis=0,
        )
        usefulness_filtered_count = tl.where(
            usefulness_enabled,
            pressure_eligible_count - usefulness_eligible_count,
            0,
        )
        any_filter_enabled = filter_enabled | pressure_enabled | usefulness_enabled
        apply_filter = any_filter_enabled & (
            usefulness_eligible_count >= candidate_count
        )
        primary_scores = tl.where(
            apply_filter & score_mask & (~usefulness_eligible_mask),
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
            routing_cache_position = tl.max(
                tl.where(score_offsets == routing_position, routing_positions, 0),
                axis=0,
            )
            candidate_id = tl.load(routing_ids + routing_cache_position)
            tl.store(candidates_out + candidate_offset, candidate_id)
            if refresh_route_positions:
                tl.store(route_bank_positions_out + candidate_offset, routing_cache_position)
                tl.store(route_positions + candidate_offset, routing_cache_position)
            primary_scores = tl.where(
                candidate_ids_by_position == candidate_id,
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
            apply_filter & filter_enabled,
            deep_sleep_count,
            0,
        )
        fallback_code = tl.where(
            filter_enabled & (~apply_filter) & (eligible_count <= 0),
            2,
            tl.where(
                pressure_enabled & (~apply_filter) & (pressure_eligible_count <= 0),
                4,
                tl.where(
                    usefulness_enabled
                    & (~apply_filter)
                    & (usefulness_eligible_count <= 0),
                    6,
                    tl.where(
                        usefulness_enabled & (~apply_filter),
                        5,
                        tl.where(
                            pressure_enabled & (~apply_filter),
                            3,
                            tl.where(
                                filter_enabled & (~apply_filter),
                                1,
                                0,
                            ),
                        ),
                    ),
                ),
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
        tl.store(route_filter_state_out + 8, tl.where(pressure_enabled, 1, 0))
        tl.store(
            route_filter_state_out + 9,
            tl.where(apply_filter & pressure_enabled, 1, 0),
        )
        tl.store(route_filter_state_out + 10, pressure_filtered_count)
        tl.store(route_filter_state_out + 11, pressure_eligible_count)
        tl.store(route_filter_state_out + 12, tl.where(usefulness_enabled, 1, 0))
        tl.store(
            route_filter_state_out + 13,
            tl.where(apply_filter & usefulness_enabled, 1, 0),
        )
        tl.store(route_filter_state_out + 14, usefulness_filtered_count)
        tl.store(route_filter_state_out + 15, usefulness_eligible_count)
        if refresh_route_positions and probe_rows > 0:
            probe_offsets = tl.arange(0, block_n)
            probe_mask = probe_offsets < probe_rows
            cursor = tl.load(route_probe_cursor)
            next_probe_positions = (cursor + probe_offsets) % route_count
            tl.store(
                route_positions + candidate_count + probe_offsets,
                next_probe_positions,
                mask=probe_mask,
            )
            tl.store(route_probe_cursor, (cursor + probe_rows) % route_count)


def fused_route_vote_available() -> bool:
    return bool(triton is not None and torch.cuda.is_available())


def fused_route_vote_kernel_variant(
    routing_vectors: torch.Tensor,
    candidates_out: torch.Tensor,
    route_positions: torch.Tensor | None = None,
    *,
    device_route_bank_refresh: bool = False,
) -> str:
    if triton is None or routing_vectors.dim() != 2:
        return "unavailable"
    vector_count, column_dim = map(int, routing_vectors.shape)
    if route_positions is not None:
        vector_count = int(route_positions.numel())
    candidate_count = int(candidates_out.numel())
    if (
        vector_count > 0
        and candidate_count > 0
        and candidate_count <= vector_count
        and column_dim > 0
    ):
        if route_positions is not None:
            if device_route_bank_refresh:
                return "indexed_route_bank_vote_device_refresh"
            return "indexed_route_bank_vote"
        return "two_stage_route_vote"
    return "unavailable"


def warmup_fused_route_vote_cuda(
    *,
    routing_key: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    route_positions: torch.Tensor | None = None,
    route_bank_positions_out: torch.Tensor | None = None,
    route_probe_cursor: torch.Tensor | None = None,
    steps_since_win: torch.Tensor,
    steps_since_win_last_update_step: torch.Tensor,
    state_transition_step_counter: torch.Tensor,
    state_transition_all_materialized_step: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    memory_pressure: torch.Tensor,
    usefulness: torch.Tensor,
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
    full_vector_count, column_dim = map(int, routing_vectors.shape)
    vector_count = (
        int(route_positions.numel())
        if route_positions is not None
        else int(full_vector_count)
    )
    candidate_count = int(candidates_out.numel())
    route_position_tensor = (
        route_positions
        if route_positions is not None
        else routing_ids
    )
    refresh_route_positions = (
        route_positions is not None
        and route_bank_positions_out is not None
        and route_probe_cursor is not None
    )
    route_bank_position_tensor = (
        route_bank_positions_out
        if route_bank_positions_out is not None
        else route_position_tensor
    )
    route_probe_cursor_tensor = (
        route_probe_cursor
        if route_probe_cursor is not None
        else previous_winner
    )
    probe_rows = (
        max(0, vector_count - candidate_count)
        if refresh_route_positions
        else 0
    )
    ensure_windows_triton_compiler()
    _routing_scores_kernel.warmup(
        routing_key,
        routing_vectors,
        route_position_tensor,
        scores_out,
        vector_count=vector_count,
        column_dim=column_dim,
        block_d=triton.next_power_of_2(column_dim),
        has_route_positions=route_positions is not None,
        num_warps=1,
        grid=(vector_count,),
    )
    _select_route_vote_kernel.warmup(
        routing_key,
        scores_out,
        routing_ids,
        route_position_tensor,
        route_bank_position_tensor,
        route_probe_cursor_tensor,
        steps_since_win,
        steps_since_win_last_update_step,
        state_transition_step_counter,
        state_transition_all_materialized_step,
        prototypes,
        thresholds,
        prediction_location,
        memory_pressure,
        usefulness,
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
        has_route_positions=route_positions is not None,
        refresh_route_positions=refresh_route_positions,
        probe_rows=probe_rows,
        route_count=full_vector_count,
        num_warps=8,
        grid=(1,),
    )

def fused_route_vote_cuda(
    *,
    routing_key: torch.Tensor,
    routing_vectors: torch.Tensor,
    routing_ids: torch.Tensor,
    route_positions: torch.Tensor | None = None,
    route_bank_positions_out: torch.Tensor | None = None,
    route_probe_cursor: torch.Tensor | None = None,
    steps_since_win: torch.Tensor,
    steps_since_win_last_update_step: torch.Tensor,
    state_transition_step_counter: torch.Tensor,
    state_transition_all_materialized_step: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    memory_pressure: torch.Tensor,
    usefulness: torch.Tensor,
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
        *((route_positions,) if route_positions is not None else ()),
        *((route_bank_positions_out,) if route_bank_positions_out is not None else ()),
        *((route_probe_cursor,) if route_probe_cursor is not None else ()),
        steps_since_win,
        steps_since_win_last_update_step,
        state_transition_step_counter,
        state_transition_all_materialized_step,
        prototypes,
        thresholds,
        prediction_location,
        memory_pressure,
        usefulness,
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
    full_vector_count, column_dim = map(int, routing_vectors.shape)
    if full_vector_count <= 0:
        raise ValueError("routing cache must not be empty")
    if int(routing_key.numel()) != column_dim:
        raise ValueError("routing_key must match routing vector width")
    if int(routing_ids.numel()) != full_vector_count:
        raise ValueError("routing_ids must match routing vector rows")
    vector_count = int(full_vector_count)
    route_position_tensor = routing_ids
    if route_positions is not None:
        if route_positions.dtype != torch.long:
            raise ValueError("route_positions must use int64")
        if route_positions.device != routing_key.device:
            raise ValueError("route_positions must share the route/vote device")
        vector_count = int(route_positions.numel())
        if vector_count <= 0:
            raise ValueError("route_positions must not be empty")
        route_position_tensor = route_positions
    refresh_route_positions = (
        route_positions is not None
        and route_bank_positions_out is not None
        and route_probe_cursor is not None
    )
    if (route_bank_positions_out is None) != (route_probe_cursor is None):
        raise ValueError(
            "route_bank_positions_out and route_probe_cursor must be provided together"
        )
    if route_bank_positions_out is not None:
        if route_positions is None:
            raise ValueError(
                "device route-bank refresh requires route_positions"
            )
        if route_bank_positions_out.dtype != torch.long:
            raise ValueError("route_bank_positions_out must use int64")
        if route_bank_positions_out.device != routing_key.device:
            raise ValueError(
                "route_bank_positions_out must share the route/vote device"
            )
    if route_probe_cursor is not None:
        if route_positions is None:
            raise ValueError("route_probe_cursor requires route_positions")
        if route_probe_cursor.dtype != torch.long or int(route_probe_cursor.numel()) != 1:
            raise ValueError("route_probe_cursor must be one int64 value")
    if int(steps_since_win.numel()) != int(prototypes.shape[0]):
        raise ValueError("steps_since_win must match prototype rows")
    if steps_since_win.dtype != torch.long:
        raise ValueError("steps_since_win must use int64")
    if int(steps_since_win_last_update_step.numel()) != int(prototypes.shape[0]):
        raise ValueError("steps_since_win_last_update_step must match prototype rows")
    if steps_since_win_last_update_step.dtype != torch.long:
        raise ValueError("steps_since_win_last_update_step must use int64")
    if (
        state_transition_step_counter.dtype != torch.long
        or int(state_transition_step_counter.numel()) != 1
    ):
        raise ValueError("state_transition_step_counter must be one int64 value")
    if (
        state_transition_all_materialized_step.dtype != torch.long
        or int(state_transition_all_materialized_step.numel()) != 1
    ):
        raise ValueError(
            "state_transition_all_materialized_step must be one int64 value"
        )
    if route_filter_control.dtype != torch.long or int(route_filter_control.numel()) < 6:
        raise ValueError("route_filter_control must contain at least six int64 values")
    if route_filter_state_out.dtype != torch.long or int(route_filter_state_out.numel()) < 16:
        raise ValueError("route_filter_state_out must contain at least sixteen int64 values")
    if int(scores_out.numel()) != vector_count:
        raise ValueError("scores_out must match active routing rows")
    candidate_count = int(candidates_out.numel())
    if candidate_count <= 0 or candidate_count > vector_count:
        raise ValueError("candidate output count must be within routing cache size")
    if route_bank_positions_out is not None and int(route_bank_positions_out.numel()) < candidate_count:
        raise ValueError(
            "route_bank_positions_out must hold at least candidate_count rows"
        )
    if int(prototypes.shape[1]) != column_dim:
        raise ValueError("prototype width must match routing vector width")
    if int(thresholds.numel()) != int(prototypes.shape[0]):
        raise ValueError("thresholds must match prototype rows")
    if int(prediction_location.shape[0]) != int(prototypes.shape[0]):
        raise ValueError("prediction locations must match prototype rows")
    if int(memory_pressure.numel()) != int(prototypes.shape[0]):
        raise ValueError("memory pressure must match prototype rows")
    if int(usefulness.numel()) != int(prototypes.shape[0]):
        raise ValueError("usefulness must match prototype rows")
    if previous_winner.dtype != torch.long or int(previous_winner.numel()) != 1:
        raise ValueError("previous_winner must be one int64 value")
    if candidates_out.dtype != torch.long:
        raise ValueError("candidates_out must use int64")
    if int(reconstruction_error_out.numel()) != 1:
        raise ValueError("reconstruction_error_out must be one scalar value")

    ensure_windows_triton_compiler()
    route_bank_position_tensor = (
        route_bank_positions_out
        if route_bank_positions_out is not None
        else route_position_tensor
    )
    route_probe_cursor_tensor = (
        route_probe_cursor
        if route_probe_cursor is not None
        else previous_winner
    )
    probe_rows = (
        max(0, vector_count - candidate_count)
        if refresh_route_positions
        else 0
    )
    _routing_scores_kernel[(vector_count,)](
        routing_key,
        routing_vectors,
        route_position_tensor,
        scores_out,
        vector_count=vector_count,
        column_dim=column_dim,
        block_d=triton.next_power_of_2(column_dim),
        has_route_positions=route_positions is not None,
        num_warps=1,
    )
    _select_route_vote_kernel[(1,)](
        routing_key,
        scores_out,
        routing_ids,
        route_position_tensor,
        route_bank_position_tensor,
        route_probe_cursor_tensor,
        steps_since_win,
        steps_since_win_last_update_step,
        state_transition_step_counter,
        state_transition_all_materialized_step,
        prototypes,
        thresholds,
        prediction_location,
        memory_pressure,
        usefulness,
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
        has_route_positions=route_positions is not None,
        refresh_route_positions=refresh_route_positions,
        probe_rows=probe_rows,
        route_count=full_vector_count,
        num_warps=8,
    )
