"""Tests for CompetitiveColumnLayer, focusing on state_dict roundtrip."""

from __future__ import annotations

import torch
import pytest

from marulho.core.columns import CompetitiveColumnLayer


def _make_layer(**overrides) -> CompetitiveColumnLayer:
    defaults = dict(
        n_columns=16,
        column_dim=8,
        input_dim=32,
        device=torch.device("cpu"),
    )
    defaults.update(overrides)
    return CompetitiveColumnLayer(**defaults)


class TestStateDict:
    """Verify state_dict / load_state_dict roundtrip fidelity."""

    def test_device_report_exposes_live_tensor_devices(self):
        layer = _make_layer(plasticity_mode="local_stdp")
        assert layer.local_plasticity is not None

        report = layer.device_report()

        assert report["device"] == "cpu"
        assert report["W_project_device"] == str(layer.W_project.device)
        assert report["input_weights_device"] == str(layer.input_weights.device)
        assert report["prototypes_device"] == str(layer.prototypes.device)
        assert report["prototype_velocity_device"] == str(layer.prototype_velocity.device)
        assert report["thresholds_device"] == str(layer.thresholds.device)
        assert report["recent_spike_window_device"] == str(layer.recent_spike_window.device)
        assert report["local_plasticity"] is not None
        assert report["local_plasticity"]["pre_trace_device"] == str(layer.local_plasticity.pre_trace.device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices(self):
        layer = _make_layer(
            device=torch.device("cuda"),
            plasticity_mode="local_stdp",
        )
        assert layer.local_plasticity is not None

        report = layer.device_report()

        assert report["device"].startswith("cuda")
        assert str(report["prototypes_device"]).startswith("cuda")
        assert str(report["W_project_device"]).startswith("cuda")
        assert str(report["local_plasticity"]["pre_trace_device"]).startswith("cuda")

    def test_roundtrip_preserves_prototypes(self):
        layer = _make_layer()
        layer.prototypes += torch.randn_like(layer.prototypes) * 0.1
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.prototypes, layer2.prototypes)

    def test_roundtrip_preserves_weights(self):
        layer = _make_layer()
        layer.W_project += 0.01
        layer.input_weights *= 1.1
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.W_project, layer2.W_project)
        assert torch.allclose(layer.input_weights, layer2.input_weights)

    def test_roundtrip_preserves_homeostasis(self):
        layer = _make_layer()
        layer.thresholds.fill_(0.42)
        layer.win_rate_ema.fill_(0.03)
        layer.homeostasis_step_count = 17
        layer.homeostasis_last_update_step[:] = torch.arange(layer.n_columns)
        layer.threshold_relaxation_history = [False, True, False, True]
        layer.threshold_relaxation_last_applied_step[:] = torch.arange(layer.n_columns)
        layer.steps_since_win[0] = 999
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.thresholds, layer2.thresholds)
        assert torch.allclose(layer.win_rate_ema, layer2.win_rate_ema)
        assert layer2.homeostasis_step_count == 17
        assert torch.equal(
            layer.homeostasis_last_update_step,
            layer2.homeostasis_last_update_step,
        )
        assert layer2.threshold_relaxation_history == [False, True, False, True]
        assert torch.equal(
            layer.threshold_relaxation_last_applied_step,
            layer2.threshold_relaxation_last_applied_step,
        )
        assert layer2.steps_since_win[0].item() == 999

    def test_roundtrip_materializes_lazy_state_transition_snapshot(self):
        layer = _make_layer(n_columns=6)
        layer.steps_since_win[:] = torch.tensor([0, 1, 2, 3, 4, 5])
        layer.state_transition_step_count = 5
        layer.steps_since_win_last_update_step[:] = torch.tensor([5, 3, 5, 0, 4, 2])
        expected = layer.state_transition_steps_snapshot()

        snap = layer.state_dict()
        layer2 = _make_layer(n_columns=6)
        layer2.load_state_dict(snap)

        assert torch.equal(snap["steps_since_win"], expected.cpu())
        assert layer2.state_transition_step_count == 5
        assert torch.equal(layer2.steps_since_win, expected)
        assert torch.equal(layer2.state_transition_steps_snapshot(), expected)
        assert torch.equal(
            layer2.steps_since_win_last_update_step,
            torch.full((6,), 5, dtype=torch.long),
        )
        assert layer2.state_transition_all_materialized_step == 5
        assert layer2.state_transition_last_update_tensor_materialized_step == 5

    def test_dense_materialized_marker_avoids_lazy_double_count(self):
        layer = _make_layer(n_columns=4)
        layer.steps_since_win[:] = torch.tensor([0, 2, 4, 6])
        layer.state_transition_step_count = 8
        layer.steps_since_win_last_update_step.zero_()

        layer._mark_all_state_transition_materialized(
            8,
            sync_last_update_tensor=False,
        )

        assert torch.equal(
            layer.state_transition_steps_snapshot(),
            torch.tensor([0, 2, 4, 6]),
        )
        assert torch.equal(
            layer.steps_since_win_last_update_step,
            torch.zeros(4, dtype=torch.long),
        )

        layer.materialize_state_transition(torch.tensor([1]))

        assert layer.steps_since_win[1].item() == 2
        assert layer.steps_since_win_last_update_step[1].item() == 8

    def test_roundtrip_preserves_spike_history_window(self):
        layer = _make_layer()
        layer.recent_spike_window.zero_()
        layer.recent_spike_window[0, 0] = 1.0
        layer.recent_spike_window[1, 1] = 1.0
        layer.recent_spike_window_cursor = 2
        layer.recent_spike_window_count = 2
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)

        assert torch.allclose(layer.recent_spike_window, layer2.recent_spike_window)
        assert layer2.recent_spike_window_cursor == 2
        assert layer2.recent_spike_window_count == 2

    def test_roundtrip_preserves_scalars(self):
        layer = _make_layer()
        layer.update_count = 42
        layer.target_firing_rate = 0.123
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert layer2.update_count == 42
        assert abs(layer2.target_firing_rate - 0.123) < 1e-6

    def test_roundtrip_preserves_velocity(self):
        layer = _make_layer()
        layer.prototype_velocity += 0.5
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.prototype_velocity, layer2.prototype_velocity)

    def test_roundtrip_handles_none_optional_tensors(self):
        layer = _make_layer()
        assert layer.last_input_pattern is None
        snap = layer.state_dict()
        assert snap["last_input_pattern"] is None

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert layer2.last_input_pattern is None

    def test_roundtrip_preserves_optional_tensors(self):
        layer = _make_layer()
        layer.last_input_pattern = torch.randn(32)
        layer.last_projected_input = torch.randn(8)
        snap = layer.state_dict()

        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        assert torch.allclose(layer.last_input_pattern, layer2.last_input_pattern)
        assert torch.allclose(layer.last_projected_input, layer2.last_projected_input)

    def test_roundtrip_with_local_stdp(self):
        layer = _make_layer(plasticity_mode="local_stdp")
        assert layer.local_plasticity is not None
        snap = layer.state_dict()
        assert snap["local_plasticity"] is not None

        layer2 = _make_layer(plasticity_mode="local_stdp")
        layer2.load_state_dict(snap)

    def test_state_dict_tensors_on_cpu(self):
        layer = _make_layer()
        snap = layer.state_dict()
        for key, val in snap.items():
            if isinstance(val, torch.Tensor):
                assert val.device == torch.device("cpu"), f"{key} not on CPU"

    def test_compete_after_roundtrip_matches(self):
        """Loaded layer produces identical competition results."""
        layer = _make_layer()
        routing_key = torch.rand(8)  # column_dim
        candidates = torch.arange(16)
        result_before = layer.compete(routing_key, candidate_indices=candidates)

        snap = layer.state_dict()
        layer2 = _make_layer()
        layer2.load_state_dict(snap)
        result_after = layer2.compete(routing_key, candidate_indices=candidates)

        assert torch.equal(result_before[0], result_after[0])

    def test_zero_input_weight_blend_skips_dead_input_drive_work(self):
        layer = _make_layer(input_weight_blend=0.0)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("input drive must not run when its blend is zero")

        layer._input_drive = fail_if_called
        pattern = torch.rand(layer.input_dim)
        routing_key = layer.project_input(pattern)

        assembly = layer.assembly_from_input(pattern)
        winners, strengths, candidates = layer.compete(
            routing_key,
            candidate_indices=torch.arange(layer.n_columns),
        )

        assert assembly.shape == (layer.n_columns,)
        assert winners.numel() == layer.n_winners
        assert strengths.numel() == layer.n_winners
        assert candidates.numel() == layer.n_columns
        assert layer._cached_raw_drive is None

    def test_zero_input_weight_blend_skips_dormant_lite_input_plasticity(self):
        layer = _make_layer(input_weight_blend=0.0, plasticity_mode="lite")
        layer.last_input_pattern = torch.full(
            (layer.input_dim,),
            1.0 / layer.input_dim,
        )
        routing_key = torch.rand(layer.column_dim)
        input_weights_before = layer.input_weights.clone()
        prototypes_before = layer.prototypes.clone()

        layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=0.5,
        )

        assert torch.equal(layer.input_weights, input_weights_before)
        assert not torch.equal(layer.prototypes, prototypes_before)
        assert layer.last_input_plasticity_mode == "skipped_zero_blend"
        assert layer.input_plasticity_update_count == 0
        assert layer.input_plasticity_skip_count == 1
        report = layer.execution_report()
        assert report["dormant_input_plasticity_skipped"] is True
        assert report["input_weight_blend"] == 0.0

    def test_nonzero_input_weight_blend_keeps_lite_input_plasticity_active(self):
        layer = _make_layer(input_weight_blend=0.25, plasticity_mode="lite")
        layer.last_input_pattern = torch.full(
            (layer.input_dim,),
            1.0 / layer.input_dim,
        )
        input_weights_before = layer.input_weights.clone()

        layer.process(
            torch.rand(layer.column_dim),
            torch.tensor([0]),
            modulator=0.5,
        )

        assert not torch.equal(layer.input_weights, input_weights_before)
        assert layer.last_input_plasticity_mode == "lite_active"
        assert layer.input_plasticity_update_count == 1
        assert layer.input_plasticity_skip_count == 0

    def test_full_input_weight_blend_uses_only_input_drive(self):
        layer = _make_layer(input_weight_blend=1.0)
        pattern = torch.rand(layer.input_dim)
        layer.last_input_pattern = layer._normalized_input_pattern(pattern)
        similarity = torch.rand(layer.n_columns)
        expected = layer._input_drive(layer.last_input_pattern)

        combined = layer._combine_similarity_and_input_drive(similarity)

        assert torch.allclose(combined, expected)

    def test_candidate_routing_scores_only_retrieved_columns(self):
        layer = _make_layer(n_columns=16, k_routing=4, input_weight_blend=0.02)
        pattern = torch.rand(layer.input_dim)
        candidates = torch.tensor([1, 4, 7, 12])

        routing_key = layer.prepare_input_for_candidate_routing(pattern)
        winners, strengths, selected = layer.compete(
            routing_key,
            candidate_indices=candidates,
        )
        report = layer.execution_report()

        assert winners.numel() == layer.n_winners
        assert strengths.numel() == layer.n_winners
        assert torch.equal(selected, candidates)
        assert report["mode"] == "candidate_subset"
        assert report["candidate_count"] == 4
        assert report["scored_column_count"] == 4
        assert report["scored_column_fraction"] == 0.25
        assert report["sparse_candidate_execution_observed"] is True
        assert report["tensor_device"] == "cpu"

    def test_lite_process_does_not_read_winner_strengths(self):
        layer = _make_layer(plasticity_mode="lite")
        layer.last_input_pattern = torch.full((layer.input_dim,), 1.0 / layer.input_dim)
        routing_key = torch.rand(layer.column_dim)

        class _ForbiddenStrengths:
            def numel(self):
                raise AssertionError("lite plasticity must not read winner_strengths")

        assembly = layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=0.5,
            winner_strengths=_ForbiddenStrengths(),
        )

        assert assembly.shape == (layer.n_columns,)
        assert layer.update_count == 1

    def test_dense_assembly_reports_all_columns_scored(self):
        layer = _make_layer(n_columns=16)

        layer.assembly_from_input(torch.rand(layer.input_dim))
        report = layer.execution_report()

        assert report["mode"] == "dense_assembly"
        assert report["candidate_count"] == 16
        assert report["scored_column_count"] == 16
        assert report["scored_column_fraction"] == 1.0
        assert report["sparse_candidate_execution_observed"] is False

    def test_full_candidate_set_is_reported_without_false_fallback(self):
        layer = _make_layer(n_columns=4)
        routing_key = layer.prepare_input_for_candidate_routing(
            torch.rand(layer.input_dim)
        )

        layer.compete(
            routing_key,
            candidate_indices=torch.arange(layer.n_columns),
        )
        report = layer.execution_report()

        assert report["mode"] == "all_columns_candidate_set"
        assert report["fallback_reason"] == "candidate_set_covers_all_columns"
        assert report["sparse_candidate_execution_observed"] is False

    def test_process_records_bounded_recent_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        layer.process(routing_key, torch.tensor([0]), modulator=0.5)
        layer.process(routing_key, torch.tensor([1]), modulator=0.5)

        report = layer.spike_health_report()

        assert report["correlation_evidence_available"] is False
        assert report["correlation"]["status"] == "insufficient_window"
        assert report["correlation"]["sample_count"] == 2
        assert layer.recent_spike_window_count == 2

    def test_sparse_spike_window_overwrite_clears_prior_dense_row(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        layer.process(routing_key, torch.tensor([2]), modulator=0.5)

        for _ in range(layer.spike_history_window - 1):
            layer.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.5,
                homeostasis_update_indices=torch.tensor([0]),
            )

        layer.process(
            routing_key,
            torch.tensor([1]),
            modulator=0.5,
            homeostasis_update_indices=torch.tensor([1]),
        )

        assert torch.equal(
            layer.recent_spike_window[0],
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
        )
        assert layer.recent_spike_window_active_ids[0, 0].item() == 1

    def test_process_without_global_state_does_not_record_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        layer.process(
            routing_key,
            torch.tensor([0]),
            modulator=0.5,
            update_global_state=False,
        )

        assert layer.recent_spike_window_count == 0

    def test_process_can_scope_homeostasis_to_awake_candidates(self):
        layer = _make_layer(
            n_columns=6,
            column_dim=4,
            input_dim=8,
            homeostasis_beta=0.5,
            dead_column_steps=3,
        )
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        layer.win_rate_ema.fill_(0.25)
        layer.thresholds.fill_(0.5)
        layer.steps_since_win[:] = torch.tensor([0, 0, 0, 0, 3, 5])
        win_before = layer.win_rate_ema.clone()
        thresholds_before = layer.thresholds.clone()

        layer.process(
            routing_key,
            torch.tensor([1]),
            modulator=0.5,
            homeostasis_update_indices=torch.tensor([1, 3]),
        )

        assert layer.last_homeostasis_update_mode == "candidate_subset"
        assert layer.last_homeostasis_update_count == 2
        assert not torch.allclose(layer.win_rate_ema[1], win_before[1])
        assert not torch.allclose(layer.win_rate_ema[3], win_before[3])
        assert torch.allclose(layer.win_rate_ema[0], win_before[0])
        assert torch.allclose(layer.win_rate_ema[2], win_before[2])
        assert torch.allclose(layer.win_rate_ema[4], win_before[4])
        assert torch.allclose(layer.win_rate_ema[5], win_before[5])
        assert torch.allclose(layer.thresholds[0], thresholds_before[0])
        assert torch.allclose(layer.thresholds[4], thresholds_before[4])
        assert torch.equal(layer.steps_since_win, torch.tensor([0, 0, 0, 1, 3, 5]))
        assert torch.equal(
            layer.state_transition_steps_snapshot(),
            torch.tensor([1, 0, 1, 1, 4, 6]),
        )

        report = layer.execution_report()
        assert report["homeostasis_update_mode"] == "candidate_subset"
        assert report["homeostasis_update_count"] == 2
        assert report["homeostasis_update_fraction"] == round(2 / 6, 6)
        assert report["state_transition_mode"] == "candidate_subset_lazy_state_transition"
        assert report["state_transition_column_count"] == 2
        assert report["state_transition_cached_count"] == 4
        assert report["state_transition_runs_all_columns"] is False
        assert report["runs_all_columns"] is False
        assert report["fallback_reason"] is None

    def test_candidate_state_transition_matches_dense_logical_steps(self):
        all_column = _make_layer(
            n_columns=8,
            column_dim=4,
            input_dim=8,
            input_weight_blend=0.0,
        )
        scoped = _make_layer(
            n_columns=8,
            column_dim=4,
            input_dim=8,
            input_weight_blend=0.0,
        )
        scoped.load_state_dict(all_column.state_dict())
        all_column.steps_since_win[:] = torch.arange(8, dtype=torch.long)
        scoped.steps_since_win[:] = torch.arange(8, dtype=torch.long)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        plans = (
            (0, torch.tensor([0, 2])),
            (1, torch.tensor([1, 3])),
            (0, torch.tensor([0, 4])),
            (5, torch.tensor([5])),
            (1, torch.tensor([1, 6])),
        )

        for winner, candidates in plans:
            all_column.last_input_pattern = torch.full((8,), 1.0 / 8.0)
            scoped.last_input_pattern = torch.full((8,), 1.0 / 8.0)
            all_column.process(
                routing_key,
                torch.tensor([winner]),
                modulator=0.0,
                homeostasis_update_indices=None,
            )
            scoped.process(
                routing_key,
                torch.tensor([winner]),
                modulator=0.0,
                homeostasis_update_indices=candidates,
            )

            assert torch.equal(
                scoped.state_transition_steps_snapshot(),
                all_column.steps_since_win,
            )
            report = scoped.execution_report()
            assert report["state_transition_mode"] == "candidate_subset_lazy_state_transition"
            assert report["state_transition_runs_all_columns"] is False
            assert report["state_transition_cached_count"] >= 6

        assert not torch.equal(scoped.steps_since_win, all_column.steps_since_win)

        scoped.materialize_state_transition(torch.tensor([7]))

        assert scoped.steps_since_win[7].item() == all_column.steps_since_win[7].item()
        assert scoped.last_state_transition_materialize_mode == "candidate_subset"
        assert scoped.last_state_transition_materialize_count == 1
        assert scoped.last_state_transition_materialize_max_age == len(plans)

    def test_candidate_compete_materializes_idle_homeostasis_before_scoring(self):
        torch.manual_seed(42)
        all_column = _make_layer(
            n_columns=6,
            column_dim=4,
            input_dim=8,
            input_weight_blend=0.0,
            homeostasis_beta=0.5,
            homeostasis_lr=0.2,
        )
        scoped = _make_layer(
            n_columns=6,
            column_dim=4,
            input_dim=8,
            input_weight_blend=0.0,
            homeostasis_beta=0.5,
            homeostasis_lr=0.2,
        )
        scoped.load_state_dict(all_column.state_dict())
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        all_column.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        scoped.last_input_pattern = torch.full((8,), 1.0 / 8.0)

        for _ in range(4):
            all_column.process(routing_key, torch.tensor([0]), modulator=0.5)
            scoped.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.5,
                homeostasis_update_indices=torch.tensor([0]),
            )

        assert scoped.homeostasis_step_count == all_column.homeostasis_step_count
        assert scoped.homeostasis_last_update_step[1].item() == 0
        assert not torch.allclose(scoped.thresholds[1], all_column.thresholds[1])

        scoped.compete(routing_key, candidate_indices=torch.tensor([1, 0]))

        assert torch.allclose(scoped.win_rate_ema[1], all_column.win_rate_ema[1])
        assert torch.allclose(scoped.thresholds[1], all_column.thresholds[1])
        assert scoped.homeostasis_last_update_step[1].item() == all_column.homeostasis_step_count
        report = scoped.execution_report()
        assert report["homeostasis_materialize_mode"] == "candidate_subset"
        assert report["homeostasis_materialize_count"] == 1
        assert report["homeostasis_materialize_max_age"] == 4

    def test_candidate_homeostasis_materializes_missed_threshold_relaxation_order(self):
        torch.manual_seed(7)
        all_column = _make_layer(
            n_columns=4,
            column_dim=3,
            input_dim=5,
            homeostasis_beta=0.5,
            homeostasis_lr=0.2,
        )
        scoped = _make_layer(
            n_columns=4,
            column_dim=3,
            input_dim=5,
            homeostasis_beta=0.5,
            homeostasis_lr=0.2,
        )
        scoped.load_state_dict(all_column.state_dict())
        routing_key = torch.rand(3)
        full_indices = torch.arange(4)
        scoped_indices = torch.tensor([0])

        for _ in range(3):
            all_step = all_column._record_threshold_relaxation()
            all_column._apply_threshold_relaxation(full_indices, step=all_step)
            all_column.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.0,
                homeostasis_update_indices=None,
            )

            scoped_step = scoped._record_threshold_relaxation()
            scoped._apply_threshold_relaxation(scoped_indices, step=scoped_step)
            scoped.process(
                routing_key,
                torch.tensor([0]),
                modulator=0.0,
                homeostasis_update_indices=scoped_indices,
            )

        assert int(scoped.homeostasis_last_update_step[1].item()) == 0

        scoped.materialize_homeostasis(torch.tensor([1]))

        assert torch.allclose(scoped.thresholds[1], all_column.thresholds[1])
        assert torch.allclose(scoped.win_rate_ema[1], all_column.win_rate_ema[1])
        assert int(scoped.threshold_relaxation_last_applied_step[1].item()) == (
            all_column.homeostasis_step_count
        )
        assert int(scoped.homeostasis_last_update_step[1].item()) == (
            all_column.homeostasis_step_count
        )
        assert scoped.last_homeostasis_materialize_mode == "candidate_subset"
        assert scoped.last_homeostasis_materialize_count == 1
        assert scoped.last_homeostasis_materialize_max_age == 3

    def test_process_marks_stale_columns_without_hot_path_revival(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8, dead_column_steps=1)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        prototypes_before = layer.prototypes.clone()

        layer.process(routing_key, torch.tensor([0]), modulator=0.5)

        assert int(layer.last_revived_indices.numel()) == 0
        assert torch.equal(layer.steps_since_win[1:], torch.ones(3, dtype=torch.long))
        assert torch.allclose(layer.prototypes[1:], prototypes_before[1:])
        report = layer.spike_health_report()
        assert report["stale_fraction"] == 0.75
        assert report["activity_state"] == "stale_routing_risk"

    def test_force_revive_dead_columns_remains_explicit_maintenance_path(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8, dead_column_steps=1)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        layer.steps_since_win.fill_(1)

        revived = layer.force_revive_dead_columns(torch.tensor([1.0, 0.2, 0.1, 0.1]))

        assert revived == 4
        assert int(layer.last_revived_indices.numel()) == 4
        assert torch.equal(layer.steps_since_win, torch.zeros(4, dtype=torch.long))

    def test_spike_health_report_is_read_only_for_spike_window(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])
        layer.process(routing_key, torch.tensor([0]), modulator=0.5)
        before_count = layer.recent_spike_window_count
        before_window = layer.recent_spike_window.clone()

        layer.spike_health_report()
        layer.spike_health_report()

        assert layer.recent_spike_window_count == before_count
        assert torch.allclose(layer.recent_spike_window, before_window)

    def test_spike_health_reports_windowed_correlation_when_enough_samples(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        samples = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        layer.recent_spike_window[:4] = samples
        layer.recent_spike_window_cursor = 4
        layer.recent_spike_window_count = 4

        report = layer.spike_health_report()

        assert report["correlation_evidence_available"] is True
        assert report["correlation"]["status"] == "overcorrelated_risk"
        assert report["correlation"]["active_columns"] >= 2
        assert report["correlation"]["mean_abs_offdiag_correlation"] is not None
        assert report["missing_evidence"] == []

    def test_spike_window_is_bounded(self):
        layer = _make_layer(n_columns=4, column_dim=4, input_dim=8)
        layer.last_input_pattern = torch.full((8,), 1.0 / 8.0)
        routing_key = torch.tensor([1.0, 0.2, 0.1, 0.1])

        for idx in range(layer.spike_history_window + 5):
            layer.process(
                routing_key,
                torch.tensor([idx % layer.n_columns]),
                modulator=0.5,
            )

        assert layer.recent_spike_window_count == layer.spike_history_window
        assert 0 <= layer.recent_spike_window_cursor < layer.spike_history_window
