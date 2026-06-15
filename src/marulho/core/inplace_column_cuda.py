from __future__ import annotations

import os
from pathlib import Path

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA dependency
    triton = None
    tl = None


def ensure_windows_triton_compiler() -> str | None:
    if os.environ.get("CC"):
        return os.environ["CC"]
    if os.name != "nt" or triton is None:
        return None
    compiler = Path(triton.__file__).parent / "runtime" / "tcc" / "tcc.exe"
    if compiler.exists():
        os.environ["CC"] = str(compiler)
        return str(compiler)
    return None


if triton is not None:

    @triton.jit(do_not_specialize_on_alignment=["candidates"])
    def _single_winner_selection_kernel(
        combined,
        inhibition,
        candidates,
        winner_out,
        strength_out,
        competition_had_positive,
        candidate_count: tl.constexpr,
        block_candidates: tl.constexpr,
    ):
        offsets = tl.arange(0, block_candidates)
        mask = offsets < candidate_count
        combined_values = tl.load(
            combined + offsets,
            mask=mask,
            other=-float("inf"),
        )
        inhibition_values = tl.load(
            inhibition + offsets,
            mask=mask,
            other=0.0,
        )
        activations = tl.maximum(combined_values - inhibition_values, 0.0)
        had_positive = tl.max(activations, axis=0) > 0.0
        selection_scores = tl.where(
            had_positive,
            activations,
            combined_values,
        )
        local_winner = tl.argmax(selection_scores, axis=0)
        winner = tl.load(candidates + local_winner)
        tl.store(winner_out, winner)
        tl.store(strength_out, 1.0)
        tl.store(competition_had_positive, had_positive)

    @triton.jit(do_not_specialize_on_alignment=["candidates"])
    def _fused_vote_competition_kernel(
        routing_key,
        prototypes,
        thresholds,
        prediction_location,
        candidates,
        previous_winner,
        winner_out,
        strength_out,
        competition_had_positive,
        column_dim: tl.constexpr,
        location_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        block_candidates: tl.constexpr,
        block_d: tl.constexpr,
        block_location: tl.constexpr,
    ):
        candidate_offsets = tl.arange(0, block_candidates)
        candidate_mask = candidate_offsets < candidate_count
        candidate_ids = tl.load(
            candidates + candidate_offsets,
            mask=candidate_mask,
            other=0,
        )

        dimension_offsets = tl.arange(0, block_d)
        dimension_mask = dimension_offsets < column_dim
        prototype_values = tl.load(
            prototypes
            + candidate_ids[:, None] * column_dim
            + dimension_offsets[None, :],
            mask=candidate_mask[:, None] & dimension_mask[None, :],
            other=0.0,
        )
        routing_values = tl.load(
            routing_key + dimension_offsets,
            mask=dimension_mask,
            other=0.0,
        )
        similarity = tl.sum(
            prototype_values * routing_values[None, :],
            axis=1,
        )

        prior_winner = tl.load(previous_winner)
        has_prior_winner = prior_winner >= 0
        safe_prior_winner = tl.maximum(prior_winner, 0)
        location_offsets = tl.arange(0, block_location)
        location_mask = location_offsets < location_dim
        candidate_locations = tl.load(
            prediction_location
            + candidate_ids[:, None] * location_dim
            + location_offsets[None, :],
            mask=candidate_mask[:, None] & location_mask[None, :],
            other=0.0,
        )
        prior_location = tl.load(
            prediction_location
            + safe_prior_winner * location_dim
            + location_offsets,
            mask=location_mask,
            other=0.0,
        )
        location_dot = tl.sum(
            candidate_locations * prior_location[None, :],
            axis=1,
        )
        candidate_norm = tl.sqrt(
            tl.sum(candidate_locations * candidate_locations, axis=1)
        )
        prior_norm = tl.sqrt(tl.sum(prior_location * prior_location, axis=0))
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
        candidate_thresholds = tl.load(
            thresholds + candidate_ids,
            mask=candidate_mask,
            other=float("inf"),
        )
        activations = tl.maximum(combined - candidate_thresholds, 0.0)
        had_positive = tl.max(
            tl.where(candidate_mask, activations, -float("inf")),
            axis=0,
        ) > 0.0
        selection_scores = tl.where(
            had_positive,
            activations,
            combined,
        )
        selection_scores = tl.where(
            candidate_mask,
            selection_scores,
            -float("inf"),
        )
        local_winner = tl.argmax(selection_scores, axis=0)
        winner = tl.load(candidates + local_winner)
        tl.store(winner_out, winner)
        tl.store(previous_winner, winner)
        tl.store(strength_out, 1.0)
        tl.store(competition_had_positive, had_positive)

    @triton.jit(
        do_not_specialize=[
            "base_modulator",
            "dopamine",
            "serotonin",
            "competitive_learning_rate",
            "strong_threshold",
            "has_previous_routing_key",
        ],
        do_not_specialize_on_alignment=["winners"],
    )
    def _inplace_column_transition_kernel(
        prototypes,
        routing_vectors,
        routing_position_by_column,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        recent_spike_window,
        assembly,
        prediction_boost_out,
        effective_modulator_out,
        result_out,
        neuromodulator_state,
        burst_result_ring,
        burst_routing_ring,
        burst_assembly_ring,
        burst_strong_flags,
        burst_strong_count,
        burst_slot,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        consolidation,
        transition_parameters,
        base_modulator,
        dopamine,
        serotonin,
        competitive_learning_rate,
        competition_had_positive,
        recent_spike_row,
        strong_threshold,
        has_previous_routing_key,
        use_transition_parameters: tl.constexpr,
        use_result_packet: tl.constexpr,
        use_burst_event_packet: tl.constexpr,
        persist_previous_routing_key: tl.constexpr,
        advance_recent_spike_row: tl.constexpr,
        update_routing_vectors: tl.constexpr,
        spike_history_window: tl.constexpr,
        burst_event_capacity: tl.constexpr,
        result_width: tl.constexpr,
        n_columns: tl.constexpr,
        column_dim: tl.constexpr,
        location_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        prototype_momentum: tl.constexpr,
        homeostasis_beta: tl.constexpr,
        homeostasis_lr: tl.constexpr,
        target_firing_rate: tl.constexpr,
        threshold_min: tl.constexpr,
        threshold_max: tl.constexpr,
        prediction_error_ema_alpha: tl.constexpr,
        prediction_failure_streak_threshold: tl.constexpr,
        prediction_learning_rate: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
        block_location: tl.constexpr,
        block_candidates: tl.constexpr,
    ):
        if use_transition_parameters:
            base_modulator = tl.load(transition_parameters)
            dopamine = tl.load(transition_parameters + 1)
            serotonin = tl.load(transition_parameters + 2)
            competitive_learning_rate = tl.load(transition_parameters + 3)
            has_previous_routing_key = tl.load(transition_parameters + 4)

        column_offsets = tl.arange(0, block_n)
        column_mask = column_offsets < n_columns
        winner = tl.load(winners)
        is_winner = column_offsets == winner

        location_offsets = tl.arange(0, block_location)
        location_mask = location_offsets < location_dim
        location_ptrs = (
            location
            + column_offsets[:, None] * location_dim
            + location_offsets[None, :]
        )
        location_velocity_ptrs = (
            location_velocity
            + column_offsets[:, None] * location_dim
            + location_offsets[None, :]
        )
        prediction_weight_ptrs = (
            prediction_weights
            + column_offsets[:, None] * location_dim
            + location_offsets[None, :]
        )
        matrix_mask = column_mask[:, None] & location_mask[None, :]
        current_location = tl.load(location_ptrs, mask=matrix_mask, other=0.0)
        current_location_velocity = tl.load(
            location_velocity_ptrs,
            mask=matrix_mask,
            other=0.0,
        )
        current_prediction_weights = tl.load(
            prediction_weight_ptrs,
            mask=matrix_mask,
            other=0.0,
        )

        prediction_logit = tl.sum(
            current_location * current_prediction_weights,
            axis=1,
        )
        prediction = 1.0 / (1.0 + tl.exp(-prediction_logit))
        actual = is_winner.to(tl.float32)
        raw_error = tl.abs(prediction - actual)
        old_error = tl.load(prediction_error + column_offsets, mask=column_mask)
        next_error = (
            prediction_error_ema_alpha * raw_error
            + (1.0 - prediction_error_ema_alpha) * old_error
        )
        tl.store(prediction_error + column_offsets, next_error, mask=column_mask)

        old_failure_streak = tl.load(
            prediction_failure_streak + column_offsets,
            mask=column_mask,
        )
        next_failure_streak = tl.where(
            raw_error > prediction_failure_streak_threshold,
            old_failure_streak + 1,
            0,
        )
        tl.store(
            prediction_failure_streak + column_offsets,
            next_failure_streak,
            mask=column_mask,
        )
        old_confidence = tl.load(confidence + column_offsets, mask=column_mask)
        next_confidence = tl.maximum(
            0.0,
            tl.minimum(
                1.0,
                0.95 * old_confidence + 0.05 * (1.0 - raw_error),
            ),
        )
        tl.store(confidence + column_offsets, next_confidence, mask=column_mask)

        movement = tl.load(
            routing_key + location_offsets,
            mask=location_mask,
            other=0.0,
        ) - tl.load(
            previous_routing_key + location_offsets,
            mask=location_mask,
            other=0.0,
        )
        moved_location_velocity = 0.9 * current_location_velocity
        moved_location_velocity += (
            is_winner[:, None].to(tl.float32)
            * 0.1
            * movement[None, :]
        )
        has_previous = has_previous_routing_key != 0
        next_location_velocity = tl.where(
            has_previous,
            moved_location_velocity,
            current_location_velocity,
        )
        next_location = tl.where(
            has_previous,
            current_location
            + is_winner[:, None].to(tl.float32) * moved_location_velocity,
            current_location,
        )

        location_norm = tl.sqrt(tl.sum(next_location * next_location, axis=1))
        location_scale = tl.where(
            location_norm > 5.0,
            5.0 / tl.maximum(location_norm, 1e-8),
            1.0,
        )
        next_location *= location_scale[:, None]
        tl.store(location_ptrs, next_location, mask=matrix_mask)
        tl.store(
            location_velocity_ptrs,
            next_location_velocity,
            mask=matrix_mask,
        )

        next_prediction_weights = current_prediction_weights + (
            is_winner[:, None].to(tl.float32)
            * prediction_learning_rate
            * next_location
        )
        updated_prediction_logit = tl.sum(
            next_location * next_prediction_weights,
            axis=1,
        )
        updated_prediction = 1.0 / (
            1.0 + tl.exp(-updated_prediction_logit)
        )
        decay = (~is_winner) & (updated_prediction > 0.5)
        next_prediction_weights = tl.where(
            decay[:, None],
            next_prediction_weights * (1.0 - 0.5 * prediction_learning_rate),
            next_prediction_weights,
        )
        tl.store(
            prediction_weight_ptrs,
            next_prediction_weights,
            mask=matrix_mask,
        )

        prediction_boost = tl.sum(
            tl.where(is_winner & column_mask, next_error, 0.0),
            axis=0,
        )
        prediction_boost = tl.maximum(0.5, tl.minimum(2.0, prediction_boost))
        winner_consolidation = tl.sum(
            tl.where(
                is_winner & column_mask,
                tl.load(consolidation + column_offsets, mask=column_mask),
                0.0,
            ),
            axis=0,
        )
        wake_scale = tl.maximum(0.2, 1.0 - 0.8 * winner_consolidation)
        wake_scale *= tl.maximum(0.2, 1.0 - 0.6 * serotonin)
        dopamine_gain = 0.8 + 0.4 * dopamine
        effective_modulator = (
            base_modulator
            * wake_scale
            * dopamine_gain
            * prediction_boost
        )
        tl.store(prediction_boost_out, prediction_boost)
        tl.store(effective_modulator_out, effective_modulator)

        feature_offsets = tl.arange(0, block_d)
        feature_mask = feature_offsets < column_dim
        routing = tl.load(
            routing_key + feature_offsets,
            mask=feature_mask,
            other=0.0,
        )
        routing = tl.maximum(routing, 1e-6)
        routing_norm = tl.sqrt(tl.sum(routing * routing, axis=0))
        routing /= tl.maximum(routing_norm, 1e-8)
        winner_row_offsets = winner * column_dim + feature_offsets
        winner_prototype = tl.load(
            prototypes + winner_row_offsets,
            mask=feature_mask,
            other=0.0,
        )
        winner_velocity = tl.load(
            prototype_velocity + winner_row_offsets,
            mask=feature_mask,
            other=0.0,
        )
        winner_similarity = tl.sum(
            winner_prototype * routing,
            axis=0,
        )
        prototype_lr = competitive_learning_rate * tl.maximum(
            0.0,
            tl.minimum(1.0, tl.abs(effective_modulator) + 0.1),
        )
        next_winner_velocity = (
            prototype_momentum * winner_velocity
            + prototype_lr * (routing - winner_prototype)
        )
        next_winner_prototype = tl.maximum(
            winner_prototype + next_winner_velocity,
            1e-6,
        )
        prototype_norm = tl.sqrt(
            tl.sum(next_winner_prototype * next_winner_prototype, axis=0)
        )
        next_winner_prototype /= tl.maximum(prototype_norm, 1e-8)
        surprise_delta = next_winner_prototype - routing
        competitive_surprise = tl.sqrt(
            tl.sum(surprise_delta * surprise_delta, axis=0)
        )
        tl.store(
            prototype_velocity + winner_row_offsets,
            next_winner_velocity,
            mask=feature_mask,
        )
        tl.store(
            prototypes + winner_row_offsets,
            next_winner_prototype,
            mask=feature_mask,
        )
        if update_routing_vectors:
            routing_cache_row = tl.load(
                routing_position_by_column + winner,
            )
            routing_cache_offsets = (
                routing_cache_row * column_dim + feature_offsets
            )
            tl.store(
                routing_vectors + routing_cache_offsets,
                next_winner_prototype,
                mask=feature_mask & (routing_cache_row >= 0),
            )

        next_assembly = tl.where(is_winner, winner_similarity, 0.0)
        tl.store(assembly + column_offsets, next_assembly, mask=column_mask)
        if use_result_packet:
            tl.store(result_out + 1, tl.load(neuromodulator_state))
            tl.store(result_out + 2, tl.load(neuromodulator_state + 1))
            tl.store(result_out + 3, tl.load(neuromodulator_state + 2))
            tl.store(result_out + 4, tl.load(neuromodulator_state + 3))
            tl.store(result_out + 5, tl.load(neuromodulator_state + 4))
            tl.store(result_out + 6, winner.to(tl.float32))
            tl.store(result_out + 7, effective_modulator)
            if result_width >= 9:
                tl.store(result_out + 8, competitive_surprise)
        if use_burst_event_packet:
            reconstruction_error = tl.load(result_out)
            event_slot = tl.load(burst_slot)
            event_base = event_slot * result_width
            strong_event = reconstruction_error > strong_threshold
            tl.store(burst_result_ring + event_base, reconstruction_error)
            tl.store(burst_result_ring + event_base + 1, tl.load(neuromodulator_state))
            tl.store(
                burst_result_ring + event_base + 2,
                tl.load(neuromodulator_state + 1),
            )
            tl.store(
                burst_result_ring + event_base + 3,
                tl.load(neuromodulator_state + 2),
            )
            tl.store(
                burst_result_ring + event_base + 4,
                tl.load(neuromodulator_state + 3),
            )
            tl.store(
                burst_result_ring + event_base + 5,
                tl.load(neuromodulator_state + 4),
            )
            tl.store(burst_result_ring + event_base + 6, winner.to(tl.float32))
            tl.store(burst_result_ring + event_base + 7, effective_modulator)
            if result_width >= 9:
                tl.store(burst_result_ring + event_base + 8, competitive_surprise)
            tl.store(burst_strong_flags + event_slot, strong_event)
            current_strong_count = tl.load(burst_strong_count)
            tl.store(
                burst_strong_count,
                current_strong_count + 1,
                mask=strong_event,
            )
            tl.store(
                burst_routing_ring + event_slot * column_dim + feature_offsets,
                routing,
                mask=feature_mask & strong_event,
            )
            tl.store(
                burst_assembly_ring + event_slot * n_columns + column_offsets,
                next_assembly,
                mask=column_mask & strong_event,
            )
            next_event_slot = event_slot + 1
            next_event_slot = tl.where(
                next_event_slot >= burst_event_capacity,
                0,
                next_event_slot,
            )
            tl.store(burst_slot, next_event_slot)
        old_steps = tl.load(steps_since_win + column_offsets, mask=column_mask)
        tl.store(
            steps_since_win + column_offsets,
            tl.where(is_winner, 0, old_steps + 1),
            mask=column_mask,
        )
        spike_row = tl.load(recent_spike_row)
        spike_row_ptrs = (
            recent_spike_window
            + spike_row * n_columns
            + column_offsets
        )
        tl.store(
            spike_row_ptrs,
            is_winner.to(tl.float32),
            mask=column_mask,
        )
        if advance_recent_spike_row:
            tl.store(
                recent_spike_row,
                (spike_row + 1) % spike_history_window,
            )

        candidate_offsets = tl.arange(0, block_candidates)
        candidate_mask = candidate_offsets < candidate_count
        candidate_ids = tl.load(
            candidates + candidate_offsets,
            mask=candidate_mask,
            other=-1,
        )
        homeostasis_selected = tl.sum(
            (
                column_offsets[:, None] == candidate_ids[None, :]
            ).to(tl.int32),
            axis=1,
        ) > 0
        old_threshold = tl.load(thresholds + column_offsets, mask=column_mask)
        had_positive = tl.load(competition_had_positive) != 0
        fallback_threshold = tl.maximum(
                threshold_min,
                tl.minimum(threshold_max, old_threshold * 0.995),
        )
        competition_threshold = tl.where(
            had_positive,
            old_threshold,
            fallback_threshold,
        )
        old_win_rate = tl.load(win_rate_ema + column_offsets, mask=column_mask)
        activity = is_winner.to(tl.float32)
        next_win_rate = (
            (1.0 - homeostasis_beta) * old_win_rate
            + homeostasis_beta * activity
        )
        next_threshold = tl.maximum(
            threshold_min,
            tl.minimum(
                threshold_max,
                competition_threshold
                + homeostasis_lr
                * (next_win_rate - target_firing_rate),
            ),
        )
        update_homeostasis = column_mask & homeostasis_selected
        tl.store(
            win_rate_ema + column_offsets,
            next_win_rate,
            mask=update_homeostasis,
        )
        tl.store(
            thresholds + column_offsets,
            tl.where(
                homeostasis_selected,
                next_threshold,
                competition_threshold,
            ),
            mask=column_mask,
        )
        if persist_previous_routing_key:
            feature_offsets = tl.arange(0, block_d)
            feature_mask = feature_offsets < column_dim
            current_routing = tl.load(
                routing_key + feature_offsets,
                mask=feature_mask,
                other=0.0,
            )
            tl.store(
                previous_routing_key + feature_offsets,
                current_routing,
                mask=feature_mask,
            )
            tl.store(transition_parameters + 4, 1.0)


    @triton.jit(do_not_specialize_on_alignment=["candidates", "winners"])
    def _candidate_predictive_writeback_kernel(
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        prediction_error_ema_alpha: tl.constexpr,
        prediction_failure_streak_threshold: tl.constexpr,
        prediction_learning_rate: tl.constexpr,
        has_previous_routing_key: tl.constexpr,
        routing_dim: tl.constexpr,
        location_dim: tl.constexpr,
        candidate_count: tl.constexpr,
        block_candidates: tl.constexpr,
        block_location: tl.constexpr,
    ):
        candidate_offsets = tl.arange(0, block_candidates)
        candidate_mask = candidate_offsets < candidate_count
        candidate_ids = tl.load(
            candidates + candidate_offsets,
            mask=candidate_mask,
            other=0,
        )
        winner = tl.load(winners)
        is_winner = candidate_ids == winner

        location_offsets = tl.arange(0, block_location)
        location_mask = location_offsets < location_dim
        routing_mask = location_offsets < routing_dim
        matrix_mask = candidate_mask[:, None] & location_mask[None, :]
        row_offsets = candidate_ids[:, None] * location_dim + location_offsets[None, :]
        location_ptrs = location + row_offsets
        velocity_ptrs = location_velocity + row_offsets
        weight_ptrs = prediction_weights + row_offsets
        current_location = tl.load(location_ptrs, mask=matrix_mask, other=0.0)
        current_velocity = tl.load(velocity_ptrs, mask=matrix_mask, other=0.0)
        current_weights = tl.load(weight_ptrs, mask=matrix_mask, other=0.0)

        prediction = 1.0 / (
            1.0 + tl.exp(-tl.sum(current_location * current_weights, axis=1))
        )
        raw_error = tl.abs(prediction - is_winner.to(tl.float32))
        old_error = tl.load(
            prediction_error + candidate_ids,
            mask=candidate_mask,
            other=0.0,
        )
        next_error = (
            prediction_error_ema_alpha * raw_error
            + (1.0 - prediction_error_ema_alpha) * old_error
        )
        tl.store(prediction_error + candidate_ids, next_error, mask=candidate_mask)

        old_failure_streak = tl.load(
            prediction_failure_streak + candidate_ids,
            mask=candidate_mask,
            other=0,
        )
        next_failure_streak = tl.where(
            raw_error > prediction_failure_streak_threshold,
            old_failure_streak + 1,
            0,
        )
        tl.store(
            prediction_failure_streak + candidate_ids,
            next_failure_streak,
            mask=candidate_mask,
        )

        old_confidence = tl.load(
            confidence + candidate_ids,
            mask=candidate_mask,
            other=0.0,
        )
        next_confidence = tl.maximum(
            0.0,
            tl.minimum(1.0, 0.95 * old_confidence + 0.05 * (1.0 - raw_error)),
        )
        tl.store(confidence + candidate_ids, next_confidence, mask=candidate_mask)

        movement = tl.load(
            routing_key + location_offsets,
            mask=location_mask & routing_mask,
            other=0.0,
        ) - tl.load(
            previous_routing_key + location_offsets,
            mask=location_mask & routing_mask,
            other=0.0,
        )
        moved_velocity = 0.9 * current_velocity
        moved_velocity += is_winner[:, None].to(tl.float32) * 0.1 * movement[None, :]
        has_previous = has_previous_routing_key != 0
        next_velocity = tl.where(
            has_previous,
            moved_velocity,
            current_velocity,
        )
        next_location = tl.where(
            has_previous,
            current_location + is_winner[:, None].to(tl.float32) * moved_velocity,
            current_location,
        )
        location_norm = tl.sqrt(tl.sum(next_location * next_location, axis=1))
        location_scale = tl.where(
            location_norm > 5.0,
            5.0 / tl.maximum(location_norm, 1e-8),
            1.0,
        )
        next_location *= location_scale[:, None]
        tl.store(location_ptrs, next_location, mask=matrix_mask)
        tl.store(velocity_ptrs, next_velocity, mask=matrix_mask)

        next_weights = current_weights + (
            is_winner[:, None].to(tl.float32)
            * prediction_learning_rate
            * next_location
        )
        updated_prediction = 1.0 / (
            1.0 + tl.exp(-tl.sum(next_location * next_weights, axis=1))
        )
        decay = (~is_winner) & candidate_mask & (updated_prediction > 0.5)
        next_weights = tl.where(
            decay[:, None],
            next_weights * (1.0 - 0.5 * prediction_learning_rate),
            next_weights,
        )
        tl.store(weight_ptrs, next_weights, mask=matrix_mask)


def inplace_column_transition_cuda(
    *,
    prototypes: torch.Tensor,
    routing_vectors: torch.Tensor | None = None,
    routing_position_by_column: torch.Tensor | None = None,
    prototype_velocity: torch.Tensor,
    thresholds: torch.Tensor,
    win_rate_ema: torch.Tensor,
    steps_since_win: torch.Tensor,
    location: torch.Tensor,
    location_velocity: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_failure_streak: torch.Tensor,
    confidence: torch.Tensor,
    recent_spike_window: torch.Tensor,
    assembly: torch.Tensor,
    prediction_boost_out: torch.Tensor,
    effective_modulator_out: torch.Tensor,
    routing_key: torch.Tensor,
    previous_routing_key: torch.Tensor,
    winners: torch.Tensor,
    candidates: torch.Tensor,
    consolidation: torch.Tensor,
    transition_parameters: torch.Tensor | None = None,
    base_modulator: float,
    dopamine: float,
    serotonin: float,
    competitive_learning_rate: float,
    recent_spike_row: torch.Tensor,
    has_previous_routing_key: bool,
    persist_previous_routing_key: bool = False,
    advance_recent_spike_row: bool = False,
    spike_history_window: int = 1,
    competition_had_positive: torch.Tensor,
    prototype_momentum: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    target_firing_rate: float,
    threshold_min: float,
    threshold_max: float,
    prediction_error_ema_alpha: float,
    prediction_failure_streak_threshold: float,
    prediction_learning_rate: float,
    result_out: torch.Tensor | None = None,
    neuromodulator_state: torch.Tensor | None = None,
    burst_result_ring: torch.Tensor | None = None,
    burst_routing_ring: torch.Tensor | None = None,
    burst_assembly_ring: torch.Tensor | None = None,
    burst_strong_flags: torch.Tensor | None = None,
    burst_strong_count: torch.Tensor | None = None,
    burst_slot: torch.Tensor | None = None,
    strong_threshold: float = 1.0,
) -> None:
    """Mutate one steady-state column transition in one Triton launch."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not prototypes.is_cuda:
        raise ValueError("in-place column transition requires CUDA tensors")
    use_result_packet = result_out is not None or neuromodulator_state is not None
    if use_result_packet and (result_out is None or neuromodulator_state is None):
        raise ValueError("result_out and neuromodulator_state must be provided together")
    use_burst_event_packet = any(
        tensor is not None
        for tensor in (
            burst_result_ring,
            burst_routing_ring,
            burst_assembly_ring,
            burst_strong_flags,
            burst_strong_count,
            burst_slot,
        )
    )
    if use_burst_event_packet and not all(
        tensor is not None
        for tensor in (
            result_out,
            neuromodulator_state,
            burst_result_ring,
            burst_routing_ring,
            burst_assembly_ring,
            burst_strong_flags,
            burst_strong_count,
            burst_slot,
        )
    ):
        raise ValueError("burst event packet tensors must be provided together")

    result_tensor = result_out if result_out is not None else effective_modulator_out
    neuromodulator_tensor = (
        neuromodulator_state
        if neuromodulator_state is not None
        else effective_modulator_out.reshape(1)
    )
    burst_result_tensor = (
        burst_result_ring if burst_result_ring is not None else result_tensor
    )
    burst_routing_tensor = (
        burst_routing_ring if burst_routing_ring is not None else routing_key
    )
    burst_assembly_tensor = (
        burst_assembly_ring if burst_assembly_ring is not None else assembly
    )
    burst_flags_tensor = (
        burst_strong_flags if burst_strong_flags is not None else competition_had_positive
    )
    burst_strong_count_tensor = (
        burst_strong_count if burst_strong_count is not None else competition_had_positive
    )
    burst_slot_tensor = burst_slot if burst_slot is not None else recent_spike_row

    tensors = (
        prototypes,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        recent_spike_window,
        recent_spike_row,
        assembly,
        prediction_boost_out,
        effective_modulator_out,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        consolidation,
        effective_modulator_out
        if transition_parameters is None
        else transition_parameters,
        competition_had_positive,
        result_tensor,
        neuromodulator_tensor,
        burst_result_tensor,
        burst_routing_tensor,
        burst_assembly_tensor,
        burst_flags_tensor,
        burst_strong_count_tensor,
        burst_slot_tensor,
    )
    if any(tensor.device != prototypes.device for tensor in tensors):
        raise ValueError("all in-place transition tensors must share one CUDA device")
    if int(winners.numel()) != 1:
        raise ValueError("in-place transition currently requires exactly one winner")
    if location.shape != location_velocity.shape:
        raise ValueError("location and location_velocity shapes must match")
    if location.shape != prediction_weights.shape:
        raise ValueError("location and prediction_weights shapes must match")
    n_columns, column_dim = prototypes.shape
    if int(location.shape[0]) != int(n_columns):
        raise ValueError("predictive and competitive column counts must match")
    if int(consolidation.numel()) != int(n_columns):
        raise ValueError("consolidation must have one value per column")
    if int(assembly.numel()) != int(n_columns):
        raise ValueError("assembly must have one value per column")
    if transition_parameters is not None and int(transition_parameters.numel()) < 5:
        raise ValueError("transition_parameters must contain at least five values")
    result_width = int(result_tensor.numel())
    if use_result_packet and result_width < 8:
        raise ValueError("result_out must contain at least eight values")
    if use_result_packet and int(neuromodulator_tensor.numel()) < 5:
        raise ValueError("neuromodulator_state must contain five values")
    if use_burst_event_packet:
        if int(burst_result_tensor.ndim) != 2:
            raise ValueError("burst_result_ring must be a two-dimensional tensor")
        if int(burst_result_tensor.shape[1]) != result_width:
            raise ValueError("burst_result_ring width must match result_out")
        if int(burst_routing_tensor.shape[-1]) != int(prototypes.shape[1]):
            raise ValueError("burst_routing_ring width must match column_dim")
        if int(burst_assembly_tensor.shape[-1]) != int(prototypes.shape[0]):
            raise ValueError("burst_assembly_ring width must match n_columns")
        if int(burst_flags_tensor.numel()) < int(burst_result_tensor.shape[0]):
            raise ValueError("burst_strong_flags must cover the event ring")
        if int(burst_strong_count_tensor.numel()) != 1:
            raise ValueError("burst_strong_count must be scalar")
        if int(burst_slot_tensor.numel()) != 1:
            raise ValueError("burst_slot must be scalar")
    if advance_recent_spike_row and int(spike_history_window) <= 0:
        raise ValueError("spike_history_window must be positive")
    update_routing_vectors = (
        routing_vectors is not None
        or routing_position_by_column is not None
    )
    if update_routing_vectors:
        if routing_vectors is None or routing_position_by_column is None:
            raise ValueError(
                "routing_vectors and routing_position_by_column must be provided together"
            )
        if routing_vectors.device != prototypes.device:
            raise ValueError("routing vectors must share the prototype device")
        if routing_vectors.shape[1:] != prototypes.shape[1:]:
            raise ValueError("routing vectors must match prototype width")
        if routing_position_by_column.device != prototypes.device:
            raise ValueError("routing positions must share the prototype device")
        if routing_position_by_column.dtype != torch.long:
            raise ValueError("routing positions must use torch.long")
        if int(routing_position_by_column.numel()) != int(n_columns):
            raise ValueError("routing positions must contain one row per column")

    ensure_windows_triton_compiler()
    parameter_tensor = (
        effective_modulator_out
        if transition_parameters is None
        else transition_parameters
    )
    _inplace_column_transition_kernel[(1,)](
        prototypes,
        prototypes if routing_vectors is None else routing_vectors,
        winners if routing_position_by_column is None else routing_position_by_column,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        recent_spike_window,
        assembly,
        prediction_boost_out,
        effective_modulator_out,
        result_tensor,
        neuromodulator_tensor,
        burst_result_tensor,
        burst_routing_tensor,
        burst_assembly_tensor,
        burst_flags_tensor,
        burst_strong_count_tensor,
        burst_slot_tensor,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        consolidation,
        parameter_tensor,
        float(base_modulator),
        float(dopamine),
        float(serotonin),
        float(competitive_learning_rate),
        competition_had_positive,
        recent_spike_row,
        float(strong_threshold),
        has_previous_routing_key=int(bool(has_previous_routing_key)),
        use_transition_parameters=int(transition_parameters is not None),
        use_result_packet=int(bool(use_result_packet)),
        use_burst_event_packet=int(bool(use_burst_event_packet)),
        persist_previous_routing_key=int(bool(persist_previous_routing_key)),
        advance_recent_spike_row=int(bool(advance_recent_spike_row)),
        update_routing_vectors=int(bool(update_routing_vectors)),
        spike_history_window=int(spike_history_window),
        burst_event_capacity=int(
            burst_result_tensor.shape[0] if use_burst_event_packet else 1
        ),
        result_width=int(result_width),
        n_columns=int(n_columns),
        column_dim=int(column_dim),
        location_dim=int(location.shape[1]),
        candidate_count=int(candidates.numel()),
        prototype_momentum=float(prototype_momentum),
        homeostasis_beta=float(homeostasis_beta),
        homeostasis_lr=float(homeostasis_lr),
        target_firing_rate=float(target_firing_rate),
        threshold_min=float(threshold_min),
        threshold_max=float(threshold_max),
        prediction_error_ema_alpha=float(prediction_error_ema_alpha),
        prediction_failure_streak_threshold=float(
            prediction_failure_streak_threshold
        ),
        prediction_learning_rate=float(prediction_learning_rate),
        block_n=triton.next_power_of_2(int(n_columns)),
        block_d=triton.next_power_of_2(int(column_dim)),
        block_location=triton.next_power_of_2(int(location.shape[1])),
        block_candidates=triton.next_power_of_2(int(candidates.numel())),
        num_warps=8,
    )


def candidate_predictive_writeback_cuda(
    *,
    location: torch.Tensor,
    location_velocity: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_failure_streak: torch.Tensor,
    confidence: torch.Tensor,
    routing_key: torch.Tensor,
    previous_routing_key: torch.Tensor,
    winners: torch.Tensor,
    candidates: torch.Tensor,
    has_previous_routing_key: bool,
    prediction_error_ema_alpha: float,
    prediction_failure_streak_threshold: float,
    prediction_learning_rate: float,
) -> None:
    """Mutate predictive state for a routed CUDA candidate set in one launch."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not location.is_cuda:
        raise ValueError("candidate predictive writeback requires CUDA tensors")
    tensors = (
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
    )
    if any(tensor.device != location.device for tensor in tensors):
        raise ValueError("all candidate predictive tensors must share one CUDA device")
    if int(winners.numel()) != 1:
        raise ValueError("candidate predictive writeback requires exactly one winner")
    if int(candidates.numel()) <= 0:
        raise ValueError("candidate predictive writeback requires candidates")
    if candidates.dtype != torch.long or winners.dtype != torch.long:
        raise ValueError("winners and candidates must use torch.long")
    if location.shape != location_velocity.shape:
        raise ValueError("location and location_velocity shapes must match")
    if location.shape != prediction_weights.shape:
        raise ValueError("location and prediction_weights shapes must match")
    n_columns, location_dim = location.shape
    if int(prediction_error.numel()) != int(n_columns):
        raise ValueError("prediction_error must have one value per column")
    if int(prediction_failure_streak.numel()) != int(n_columns):
        raise ValueError("prediction_failure_streak must have one value per column")
    if int(confidence.numel()) != int(n_columns):
        raise ValueError("confidence must have one value per column")

    ensure_windows_triton_compiler()
    _candidate_predictive_writeback_kernel[(1,)](
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        prediction_error_ema_alpha=float(prediction_error_ema_alpha),
        prediction_failure_streak_threshold=float(prediction_failure_streak_threshold),
        prediction_learning_rate=float(prediction_learning_rate),
        has_previous_routing_key=int(bool(has_previous_routing_key)),
        routing_dim=int(routing_key.numel()),
        location_dim=int(location_dim),
        candidate_count=int(candidates.numel()),
        block_candidates=triton.next_power_of_2(int(candidates.numel())),
        block_location=triton.next_power_of_2(int(location_dim)),
        num_warps=1,
    )


def select_single_winner_cuda(
    *,
    combined: torch.Tensor,
    inhibition: torch.Tensor,
    candidates: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
) -> None:
    """Select one candidate winner without synchronizing the CUDA stream."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not combined.is_cuda:
        raise ValueError("single-winner selection requires CUDA tensors")
    tensors = (
        combined,
        inhibition,
        candidates,
        winner_out,
        strength_out,
        competition_had_positive,
    )
    if any(tensor.device != combined.device for tensor in tensors):
        raise ValueError("all single-winner tensors must share one CUDA device")
    candidate_count = int(candidates.numel())
    if candidate_count <= 0:
        raise ValueError("single-winner selection requires candidates")
    if int(combined.numel()) != candidate_count:
        raise ValueError("combined must have one value per candidate")
    if int(inhibition.numel()) != candidate_count:
        raise ValueError("inhibition must have one value per candidate")
    if int(winner_out.numel()) != 1 or winner_out.dtype != torch.long:
        raise ValueError("winner_out must be one int64 value")
    if int(strength_out.numel()) != 1:
        raise ValueError("strength_out must contain one value")
    if (
        int(competition_had_positive.numel()) != 1
        or competition_had_positive.dtype != torch.bool
    ):
        raise ValueError("competition_had_positive must be one bool value")

    ensure_windows_triton_compiler()
    _single_winner_selection_kernel[(1,)](
        combined,
        inhibition,
        candidates,
        winner_out,
        strength_out,
        competition_had_positive,
        candidate_count=candidate_count,
        block_candidates=triton.next_power_of_2(candidate_count),
        num_warps=1,
    )


def select_fused_vote_competition_cuda(
    *,
    routing_key: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    candidates: torch.Tensor,
    previous_winner: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
) -> None:
    """Fuse predictive vote, candidate score, inhibition, and selection."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not routing_key.is_cuda:
        raise ValueError("fused vote competition requires CUDA tensors")
    tensors = (
        routing_key,
        prototypes,
        thresholds,
        prediction_location,
        candidates,
        previous_winner,
        winner_out,
        strength_out,
        competition_had_positive,
    )
    if any(tensor.device != routing_key.device for tensor in tensors):
        raise ValueError("all fused vote competition tensors must share one CUDA device")
    candidate_count = int(candidates.numel())
    if candidate_count <= 0:
        raise ValueError("fused vote competition requires candidates")
    if int(routing_key.numel()) != int(prototypes.shape[1]):
        raise ValueError("routing_key must match prototype column dimension")
    if int(thresholds.numel()) != int(prototypes.shape[0]):
        raise ValueError("thresholds must have one value per prototype")
    if int(prediction_location.shape[0]) != int(prototypes.shape[0]):
        raise ValueError("prediction_location must have one row per prototype")
    if int(previous_winner.numel()) != 1 or previous_winner.dtype != torch.long:
        raise ValueError("previous_winner must be one int64 value")

    ensure_windows_triton_compiler()
    _fused_vote_competition_kernel[(1,)](
        routing_key,
        prototypes,
        thresholds,
        prediction_location,
        candidates,
        previous_winner,
        winner_out,
        strength_out,
        competition_had_positive,
        column_dim=int(prototypes.shape[1]),
        location_dim=int(prediction_location.shape[1]),
        candidate_count=candidate_count,
        block_candidates=triton.next_power_of_2(candidate_count),
        block_d=triton.next_power_of_2(int(prototypes.shape[1])),
        block_location=triton.next_power_of_2(int(prediction_location.shape[1])),
        num_warps=4,
    )


def warmup_fused_vote_competition_cuda(
    *,
    routing_key: torch.Tensor,
    prototypes: torch.Tensor,
    thresholds: torch.Tensor,
    prediction_location: torch.Tensor,
    candidates: torch.Tensor,
    previous_winner: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
) -> None:
    """Compile one fused vote/competition shape without launching it."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    candidate_count = int(candidates.numel())
    ensure_windows_triton_compiler()
    _fused_vote_competition_kernel.warmup(
        routing_key,
        prototypes,
        thresholds,
        prediction_location,
        candidates,
        previous_winner,
        winner_out,
        strength_out,
        competition_had_positive,
        column_dim=int(prototypes.shape[1]),
        location_dim=int(prediction_location.shape[1]),
        candidate_count=candidate_count,
        block_candidates=triton.next_power_of_2(candidate_count),
        block_d=triton.next_power_of_2(int(prototypes.shape[1])),
        block_location=triton.next_power_of_2(int(prediction_location.shape[1])),
        num_warps=4,
        grid=(1,),
    )


def warmup_single_winner_cuda(
    *,
    combined: torch.Tensor,
    inhibition: torch.Tensor,
    candidates: torch.Tensor,
    winner_out: torch.Tensor,
    strength_out: torch.Tensor,
    competition_had_positive: torch.Tensor,
) -> None:
    """Compile one selector shape without launching or mutating state."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not combined.is_cuda:
        raise ValueError("single-winner warmup requires CUDA tensors")
    candidate_count = int(candidates.numel())
    if candidate_count <= 0:
        raise ValueError("single-winner warmup requires candidates")
    ensure_windows_triton_compiler()
    _single_winner_selection_kernel.warmup(
        combined,
        inhibition,
        candidates,
        winner_out,
        strength_out,
        competition_had_positive,
        candidate_count=candidate_count,
        block_candidates=triton.next_power_of_2(candidate_count),
        num_warps=1,
        grid=(1,),
    )


def warmup_inplace_column_transition_cuda(
    *,
    prototypes: torch.Tensor,
    prototype_velocity: torch.Tensor,
    thresholds: torch.Tensor,
    win_rate_ema: torch.Tensor,
    steps_since_win: torch.Tensor,
    location: torch.Tensor,
    location_velocity: torch.Tensor,
    prediction_weights: torch.Tensor,
    prediction_error: torch.Tensor,
    prediction_failure_streak: torch.Tensor,
    confidence: torch.Tensor,
    recent_spike_window: torch.Tensor,
    assembly: torch.Tensor,
    prediction_boost_out: torch.Tensor,
    effective_modulator_out: torch.Tensor,
    routing_key: torch.Tensor,
    previous_routing_key: torch.Tensor,
    winners: torch.Tensor,
    candidates: torch.Tensor,
    consolidation: torch.Tensor,
    competition_had_positive: torch.Tensor,
    recent_spike_row: torch.Tensor,
    prototype_momentum: float,
    homeostasis_beta: float,
    homeostasis_lr: float,
    target_firing_rate: float,
    threshold_min: float,
    threshold_max: float,
    prediction_error_ema_alpha: float,
    prediction_failure_streak_threshold: float,
    prediction_learning_rate: float,
) -> None:
    """Compile one candidate shape without launching or mutating state."""

    if triton is None:
        raise RuntimeError("Triton is not installed")
    if not prototypes.is_cuda:
        raise ValueError("in-place column transition warmup requires CUDA tensors")
    n_columns, column_dim = prototypes.shape
    ensure_windows_triton_compiler()
    _inplace_column_transition_kernel.warmup(
        prototypes,
        prototypes,
        winners,
        prototype_velocity,
        thresholds,
        win_rate_ema,
        steps_since_win,
        location,
        location_velocity,
        prediction_weights,
        prediction_error,
        prediction_failure_streak,
        confidence,
        recent_spike_window,
        assembly,
        prediction_boost_out,
        effective_modulator_out,
        effective_modulator_out,
        effective_modulator_out,
        effective_modulator_out,
        effective_modulator_out,
        routing_key,
        assembly,
        competition_had_positive,
        recent_spike_row,
        routing_key,
        previous_routing_key,
        winners,
        candidates,
        consolidation,
        effective_modulator_out,
        0.0,
        0.0,
        0.0,
        0.0,
        competition_had_positive,
        recent_spike_row,
        1.0,
        has_previous_routing_key=0,
        use_transition_parameters=0,
        use_result_packet=0,
        use_burst_event_packet=0,
        persist_previous_routing_key=0,
        advance_recent_spike_row=0,
        update_routing_vectors=0,
        spike_history_window=1,
        burst_event_capacity=1,
        result_width=1,
        n_columns=int(n_columns),
        column_dim=int(column_dim),
        location_dim=int(location.shape[1]),
        candidate_count=int(candidates.numel()),
        prototype_momentum=float(prototype_momentum),
        homeostasis_beta=float(homeostasis_beta),
        homeostasis_lr=float(homeostasis_lr),
        target_firing_rate=float(target_firing_rate),
        threshold_min=float(threshold_min),
        threshold_max=float(threshold_max),
        prediction_error_ema_alpha=float(prediction_error_ema_alpha),
        prediction_failure_streak_threshold=float(
            prediction_failure_streak_threshold
        ),
        prediction_learning_rate=float(prediction_learning_rate),
        block_n=triton.next_power_of_2(int(n_columns)),
        block_d=triton.next_power_of_2(int(column_dim)),
        block_location=triton.next_power_of_2(int(location.shape[1])),
        block_candidates=triton.next_power_of_2(int(candidates.numel())),
        num_warps=8,
        grid=(1,),
    )
