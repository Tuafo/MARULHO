"""Tests for PredictiveColumnState (Thousand Brains Theory extension)."""

from __future__ import annotations

import torch
import pytest

from marulho.config.model_config import MarulhoConfig
from marulho.core.predictive_columns import PredictiveColumnState


class TestPredictiveColumnState:
    """Test the Thousand Brains predictive column implementation."""

    def test_initialization(self):
        state = PredictiveColumnState(n_columns=32, location_dim=8)
        assert state.location.shape == (32, 8)
        assert state.velocity.shape == (32, 8)
        assert state.confidence.shape == (32,)
        assert state.prediction_error.shape == (32,)
        assert state.prediction_failure_streak.shape == (32,)
        # Initial confidence is 0.5
        assert abs(state.confidence.mean().item() - 0.5) < 0.01

    def test_device_report_exposes_live_tensor_devices(self):
        state = PredictiveColumnState(
            n_columns=8,
            location_dim=4,
            device=torch.device("cpu"),
        )
        report = state.device_report()

        assert report["device"] == "cpu"
        assert report["location_device"] == str(state.location.device)
        assert report["velocity_device"] == str(state.velocity.device)
        assert report["prediction_weights_device"] == str(state._prediction_weights.device)
        assert report["prediction_error_device"] == str(state.prediction_error.device)
        assert report["prediction_failure_streak_device"] == str(state.prediction_failure_streak.device)
        assert report["prediction_failure_streak_available"] is True
        assert report["last_prediction_update_mode"] == "not_run"
        assert report["last_prediction_update_count"] == 0
        assert report["last_prediction_update_fraction"] == 0.0
        assert report["last_prediction_update_runs_all_columns"] is False
        assert report["last_location_update_mode"] == "not_run"
        assert report["last_location_update_count"] == 0
        assert report["last_location_update_runs_all_columns"] is False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices(self):
        state = PredictiveColumnState(
            n_columns=8,
            location_dim=4,
            device=torch.device("cuda"),
        )
        report = state.device_report()

        assert report["device"].startswith("cuda")
        assert str(report["location_device"]).startswith("cuda")
        assert str(report["prediction_weights_device"]).startswith("cuda")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_compiled_dense_transition_preserves_state_across_repeated_cuda_steps(self):
        torch.manual_seed(23)
        legacy = PredictiveColumnState(
            n_columns=16,
            location_dim=4,
            device=torch.device("cuda"),
        )
        compiled = PredictiveColumnState(
            n_columns=16,
            location_dim=4,
            device=torch.device("cuda"),
        )
        compiled.load_state_dict(legacy.state_dict())
        previous = torch.randn(16, device="cuda")

        for step in range(3):
            routing_key = torch.randn(16, device="cuda")
            winners = torch.tensor([step + 1], dtype=torch.long, device="cuda")
            winner_list = winners.cpu().tolist()
            legacy.compute_prediction_error(winner_list, routing_key)
            legacy.update_location(winner_list, routing_key, previous)
            legacy.update_predictions(winner_list, learning_rate=0.005)
            compiled.apply_dense_transition(
                winners,
                routing_key,
                previous,
                transition_mode="compiled",
            )
            previous = routing_key

        torch.cuda.synchronize()
        assert compiled.last_dense_transition_mode == "compiled"
        assert compiled.last_dense_transition_fallback_reason is None
        assert torch.allclose(compiled.location, legacy.location, atol=1e-5)
        assert torch.allclose(compiled.velocity, legacy.velocity, atol=1e-5)
        assert torch.allclose(
            compiled._prediction_weights,
            legacy._prediction_weights,
            atol=1e-5,
        )
        assert torch.allclose(
            compiled.prediction_error,
            legacy.prediction_error,
            atol=1e-5,
        )
        assert torch.equal(
            compiled.prediction_failure_streak,
            legacy.prediction_failure_streak,
        )
        assert torch.allclose(compiled.confidence, legacy.confidence, atol=1e-5)

    def test_predict_returns_correct_shape(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        pred = state.predict(column_dim=16)
        assert pred.shape == (16,)
        # Predictions are sigmoid outputs, so in [0, 1]
        assert pred.min() >= 0.0
        assert pred.max() <= 1.0

    def test_update_location_moves_winners(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        loc_before = state.location[0].clone()

        routing_key = torch.randn(16)
        prev_key = torch.randn(16)
        state.update_location([0, 1], routing_key, prev_key)

        # Winner locations should have moved
        assert not torch.allclose(state.location[0], loc_before)

    def test_update_location_without_prev_key(self):
        """First step has no previous key — should not crash."""
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        routing_key = torch.randn(16)
        state.update_location([0], routing_key, None)  # No prev key
        # Should not crash, locations unchanged (no movement signal)

    def test_candidate_scoped_update_location_keeps_idle_location_cached(self):
        state = PredictiveColumnState(n_columns=6, location_dim=3)
        state.location = torch.ones(6, 3) * 0.1
        state.velocity = torch.ones(6, 3)
        location_before = state.location.clone()
        velocity_before = state.velocity.clone()

        state.update_location(
            [1],
            torch.randn(6),
            torch.randn(6),
            candidate_indices=torch.tensor([1, 3], dtype=torch.long),
        )

        assert state.last_location_update_mode == "candidate_subset"
        assert state.last_location_update_count == 2
        assert state.last_location_cached_count == 4
        assert state.last_location_update_runs_all_columns is False
        assert not torch.allclose(state.location[1], location_before[1])
        assert torch.allclose(state.velocity[3], velocity_before[3] * 0.9)
        assert torch.allclose(state.location[[0, 2, 4, 5]], location_before[[0, 2, 4, 5]])
        assert torch.allclose(state.velocity[[0, 2, 4, 5]], velocity_before[[0, 2, 4, 5]])

    def test_prediction_error_computation(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        routing_key = torch.randn(16)

        error = state.compute_prediction_error([0, 1], routing_key)
        assert error.shape == (16,)
        assert error.min() >= 0.0
        assert error.max() <= 1.0

    @pytest.mark.parametrize("with_previous", [False, True])
    def test_dense_transition_matches_legacy_predictive_sequence(self, with_previous):
        torch.manual_seed(17)
        legacy = PredictiveColumnState(n_columns=16, location_dim=4)
        fused = PredictiveColumnState(n_columns=16, location_dim=4)
        fused.load_state_dict(legacy.state_dict())
        routing_key = torch.randn(16)
        previous = torch.randn(16) if with_previous else None
        winner_ids = [2]
        winners = torch.tensor(winner_ids, dtype=torch.long)

        legacy_error = legacy.compute_prediction_error(winner_ids, routing_key)
        legacy.update_location(winner_ids, routing_key, previous)
        legacy.update_predictions(winner_ids, learning_rate=0.005)
        fused_error = fused.apply_dense_transition(
            winners,
            routing_key,
            previous,
            learning_rate=0.005,
        )

        assert torch.allclose(fused_error, legacy_error, atol=1e-6)
        assert torch.allclose(fused.location, legacy.location, atol=1e-6)
        assert torch.allclose(fused.velocity, legacy.velocity, atol=1e-6)
        assert torch.allclose(
            fused._prediction_weights,
            legacy._prediction_weights,
            atol=1e-6,
        )
        assert torch.allclose(fused.prediction_error, legacy.prediction_error, atol=1e-6)
        assert torch.equal(
            fused.prediction_failure_streak,
            legacy.prediction_failure_streak,
        )
        assert torch.allclose(fused.confidence, legacy.confidence, atol=1e-6)

    def test_candidate_scoped_prediction_error_keeps_idle_state_cached(self):
        state = PredictiveColumnState(n_columns=6, location_dim=3)
        state.location.fill_(1.0)
        state._prediction_weights.fill_(5.0)
        state.prediction_error.fill_(0.25)
        state.confidence.fill_(0.5)
        state.prediction_failure_streak.fill_(7)

        candidates = torch.tensor([1, 3], dtype=torch.long)
        error = state.compute_prediction_error(
            [1],
            torch.randn(6),
            candidate_indices=candidates,
        )

        assert error.shape == (6,)
        assert state.last_prediction_update_mode == "candidate_subset"
        assert state.last_prediction_update_count == 2
        assert state.last_prediction_update_fraction == pytest.approx(2 / 6)
        report = state.prediction_update_execution_report()
        assert report["surface"] == "predictive_column_update_scheduler.v1"
        assert report["updated_column_count"] == 2
        assert report["cached_state_count"] == 4
        assert report["location_update_mode"] == "not_run"
        assert report["location_update_count"] == 0
        assert report["location_update_runs_all_columns"] is False
        assert report["runs_all_columns"] is False
        assert state.prediction_failure_streak[1].item() == 0
        assert state.prediction_failure_streak[3].item() == 8
        assert torch.equal(
            state.prediction_failure_streak[[0, 2, 4, 5]],
            torch.full((4,), 7, dtype=torch.int32),
        )
        assert torch.allclose(
            state.prediction_error[[0, 2, 4, 5]],
            torch.full((4,), 0.25),
        )
        assert torch.allclose(
            state.confidence[[0, 2, 4, 5]],
            torch.full((4,), 0.5),
        )

    def test_candidate_scoped_prediction_decay_keeps_non_candidates_cached(self):
        state = PredictiveColumnState(n_columns=6, location_dim=3)
        state.location.fill_(1.0)
        state._prediction_weights.fill_(5.0)
        before = state._prediction_weights.clone()

        candidates = torch.tensor([1, 3], dtype=torch.long)
        state.update_predictions([1], learning_rate=0.1, candidate_indices=candidates)

        assert state.last_prediction_update_mode == "candidate_subset"
        assert state.last_prediction_update_count == 2
        assert torch.all(state._prediction_weights[1] > before[1])
        assert torch.all(state._prediction_weights[3] < before[3])
        assert torch.allclose(state._prediction_weights[[0, 2, 4, 5]], before[[0, 2, 4, 5]])

    def test_candidate_vote_materializes_idle_predictive_state_before_scoring(self):
        torch.manual_seed(123)
        full = PredictiveColumnState(n_columns=6, location_dim=3)
        scoped = PredictiveColumnState(n_columns=6, location_dim=3)
        scoped.load_state_dict(full.state_dict())
        routing_keys = [torch.randn(6) for _ in range(4)]
        scoped_candidates = torch.tensor([0], dtype=torch.long)
        previous = None

        for routing_key in routing_keys:
            full.compute_prediction_error([0], routing_key)
            full.update_location([0], routing_key, previous)
            full.update_predictions([0], learning_rate=0.005)

            scoped.compute_prediction_error(
                [0],
                routing_key,
                candidate_indices=scoped_candidates,
            )
            scoped.update_location(
                [0],
                routing_key,
                previous,
                candidate_indices=scoped_candidates,
            )
            scoped.update_predictions(
                [0],
                learning_rate=0.005,
                candidate_indices=scoped_candidates,
            )
            previous = routing_key

        assert int(scoped.predictive_last_update_step[1].item()) == 0

        scoped.vote([0], torch.randn(6), candidate_indices=torch.tensor([1]))

        assert torch.allclose(scoped.location[1], full.location[1], atol=1e-6)
        assert torch.allclose(scoped.velocity[1], full.velocity[1], atol=1e-6)
        assert torch.allclose(
            scoped._prediction_weights[1],
            full._prediction_weights[1],
            atol=1e-6,
        )
        assert torch.allclose(
            scoped.prediction_error[1],
            full.prediction_error[1],
            atol=1e-6,
        )
        assert torch.allclose(scoped.confidence[1], full.confidence[1], atol=1e-6)
        assert torch.equal(
            scoped.prediction_failure_streak[1],
            full.prediction_failure_streak[1],
        )
        assert int(scoped.predictive_last_update_step[1].item()) == full.predictive_step_count
        assert scoped.last_predictive_materialize_mode == "candidate_subset"
        assert scoped.last_predictive_materialize_count == 1
        assert scoped.last_predictive_materialize_max_age == 4

    def test_candidate_materialization_matches_long_full_predictive_replay(self):
        torch.manual_seed(20260615)
        full = PredictiveColumnState(n_columns=12, location_dim=4)
        scoped = PredictiveColumnState(n_columns=12, location_dim=4)

        location = torch.randn(12, 4)
        location[1] *= 8.0
        location[2] *= -7.0
        weights = torch.randn(12, 4)
        weights[1] = location[1].sign() * 2.0
        weights[2] = -location[2].sign() * 2.0

        full.location.copy_(location)
        full.velocity.copy_(torch.randn(12, 4))
        full._prediction_weights.copy_(weights)
        full.prediction_error.copy_(torch.linspace(0.05, 0.35, 12))
        full.confidence.copy_(torch.linspace(0.2, 0.8, 12))
        full.prediction_failure_streak.copy_(torch.arange(12) % 5)
        scoped.load_state_dict(full.state_dict())

        scoped_candidates = torch.tensor([0], dtype=torch.long)
        previous = None
        for _ in range(96):
            routing_key = torch.randn(12)
            full.compute_prediction_error([0], routing_key)
            full.update_location([0], routing_key, previous)
            full.update_predictions([0], learning_rate=0.005)

            scoped.compute_prediction_error(
                [0],
                routing_key,
                candidate_indices=scoped_candidates,
            )
            scoped.update_location(
                [0],
                routing_key,
                previous,
                candidate_indices=scoped_candidates,
            )
            scoped.update_predictions(
                [0],
                learning_rate=0.005,
                candidate_indices=scoped_candidates,
            )
            previous = routing_key

        skipped = torch.arange(1, 12, dtype=torch.long)
        scoped.materialize_predictive_state(skipped)

        assert torch.allclose(scoped.location[skipped], full.location[skipped], atol=1e-6)
        assert torch.allclose(scoped.velocity[skipped], full.velocity[skipped], atol=1e-6)
        assert torch.allclose(
            scoped._prediction_weights[skipped],
            full._prediction_weights[skipped],
            atol=1e-6,
        )
        assert torch.allclose(
            scoped.prediction_error[skipped],
            full.prediction_error[skipped],
            atol=1e-6,
        )
        assert torch.allclose(
            scoped.confidence[skipped],
            full.confidence[skipped],
            atol=1e-6,
        )
        assert torch.equal(
            scoped.prediction_failure_streak[skipped],
            full.prediction_failure_streak[skipped],
        )
        assert torch.equal(
            scoped.predictive_last_update_step[skipped],
            torch.full_like(skipped, full.predictive_step_count),
        )

    def test_prediction_failure_streak_tracks_repeated_raw_failures(self):
        state = PredictiveColumnState(n_columns=4, location_dim=2)
        state.location.fill_(1.0)
        state._prediction_weights.fill_(5.0)

        for _ in range(3):
            state.compute_prediction_error([0], torch.randn(4))

        assert int(state.prediction_failure_streak[0].item()) == 0
        assert int(state.prediction_failure_streak[1].item()) == 3
        assert int(state.prediction_failure_streak[2].item()) == 3
        assert int(state.prediction_failure_streak[3].item()) == 3

        state.compute_prediction_error([1, 2, 3], torch.randn(4))

        assert int(state.prediction_failure_streak[0].item()) == 1
        assert int(state.prediction_failure_streak[1].item()) == 0
        assert int(state.prediction_failure_streak[2].item()) == 0
        assert int(state.prediction_failure_streak[3].item()) == 0

    def test_prediction_error_decreases_with_consistent_winners(self):
        state = PredictiveColumnState(n_columns=8, location_dim=4)
        routing_key = torch.randn(8)

        # Get initial prediction for column 0
        initial_pred = state.predict(8)[0].item()

        # Repeatedly tell column 0 it won — predictions should increase
        for _ in range(100):
            state.compute_prediction_error([0], routing_key)
            state.update_predictions([0], learning_rate=0.1)

        # After training, column 0's prediction should be higher
        final_pred = state.predict(8)[0].item()
        assert final_pred > initial_pred  # Learned to predict its own winning

    def test_voting_returns_gain_vector(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        routing_key = torch.randn(16)

        gain = state.vote([0, 1], routing_key)
        assert gain.shape == (16,)
        # Gains should be centered around 1.0
        assert gain.mean().item() > 0.5
        assert gain.mean().item() < 1.5

    def test_voting_boosts_similar_locations(self):
        state = PredictiveColumnState(n_columns=8, location_dim=4)
        # Make columns 0,1,2 have similar locations
        state.location[0:3] = torch.ones(3, 4) * 0.5
        # Make columns 5,6,7 have opposite locations
        state.location[5:8] = torch.ones(3, 4) * -0.5

        gain = state.vote([0], torch.randn(8))
        # Columns near the winner should get higher gain
        assert gain[1].item() > gain[6].item()

    def test_candidate_scoped_voting_updates_awake_columns_and_caches_idle_votes(self):
        scoped = PredictiveColumnState(n_columns=8, location_dim=4)
        full = PredictiveColumnState(n_columns=8, location_dim=4)
        full.location.copy_(scoped.location)
        full._prediction_weights.copy_(scoped._prediction_weights)
        candidates = torch.tensor([1, 3, 5], dtype=torch.long)
        routing_key = torch.randn(8)

        full_gain = full.vote([0], routing_key)
        scoped_gain = scoped.vote([0], routing_key, candidate_indices=candidates)
        report = scoped.vote_execution_report()

        assert torch.allclose(scoped_gain[candidates], full_gain[candidates])
        idle = torch.tensor([2, 4, 6, 7], dtype=torch.long)
        assert torch.allclose(scoped_gain[idle], torch.ones_like(scoped_gain[idle]))
        assert report["mode"] == "awake_mask_cached_vote"
        assert report["updated_column_count"] == 3
        assert report["cached_vote_use_count"] == 5
        assert report["runs_all_columns"] is False
        assert report["fallback_reason"] is None

    def test_candidate_scoped_voting_falls_back_truthfully_for_empty_or_full_awake_masks(self):
        state = PredictiveColumnState(n_columns=4, location_dim=2)
        cached = torch.tensor([1.1, 0.9, 1.0, 1.2])
        state.cached_consensus_gain.copy_(cached)

        empty_gain = state.vote([0], torch.randn(4), candidate_indices=torch.empty(0, dtype=torch.long))
        empty_report = state.vote_execution_report()

        assert torch.equal(empty_gain, cached)
        assert empty_report["mode"] == "cached_vote_no_awake_candidates"
        assert empty_report["updated_column_count"] == 0
        assert empty_report["cached_vote_use_count"] == 4
        assert empty_report["runs_all_columns"] is False
        assert empty_report["fallback_reason"] == "no_awake_candidates_cached_vote"

        state.vote([0], torch.randn(4), candidate_indices=torch.arange(4))
        full_report = state.vote_execution_report()

        assert full_report["mode"] == "all_columns_candidate_set"
        assert full_report["updated_column_count"] == 4
        assert full_report["cached_vote_use_count"] == 0
        assert full_report["runs_all_columns"] is True
        assert full_report["fallback_reason"] == "candidate_set_covers_all_columns"

    def test_prediction_error_modulation(self):
        state = PredictiveColumnState(n_columns=8, location_dim=4)
        state.prediction_error = torch.tensor([0.0, 0.1, 0.5, 0.9, 0.0, 0.0, 0.0, 0.0])

        mod = state.prediction_error_modulation()
        # High error = high modulation (more learning)
        assert mod[3].item() > mod[0].item()
        # Modulation is 1.0 + 2.0 * error
        assert abs(mod[3].item() - 2.8) < 0.01

    def test_state_dict_and_load(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        # Modify state
        state.location += 1.0
        state.confidence.fill_(0.9)
        state.prediction_failure_streak.fill_(4)
        state.predictive_step_count = 5
        state.predictive_last_update_step[:] = torch.arange(16)

        saved = state.state_dict()
        state2 = PredictiveColumnState(n_columns=16, location_dim=4)
        state2.load_state_dict(saved)

        assert torch.allclose(state.location, state2.location)
        assert torch.allclose(state.confidence, state2.confidence)
        assert torch.equal(state.prediction_failure_streak, state2.prediction_failure_streak)
        assert state2.predictive_step_count == 5
        assert torch.equal(
            state.predictive_last_update_step,
            state2.predictive_last_update_step,
        )

    def test_legacy_hypothesis_checkpoint_field_is_ignored(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        saved = state.state_dict()
        saved["hypothesis"] = torch.ones(16)

        restored = PredictiveColumnState(n_columns=16, location_dim=4)
        restored.load_state_dict(saved)

        assert "hypothesis" not in restored.state_dict()
        assert not hasattr(restored, "hypothesis")

    def test_reset(self):
        state = PredictiveColumnState(n_columns=8, location_dim=4)
        state.location += 10.0
        state.confidence.fill_(0.9)
        state.prediction_error.fill_(0.5)
        state.prediction_failure_streak.fill_(5)

        state.reset()
        assert state.location.abs().max() < 1.0  # Re-randomized near 0
        assert abs(state.confidence.mean().item() - 0.5) < 0.01
        assert state.prediction_error.sum().item() == 0.0
        assert state.prediction_failure_streak.sum().item() == 0

    def test_location_normalization_prevents_drift(self):
        state = PredictiveColumnState(n_columns=4, location_dim=4)
        routing_key = torch.randn(16)
        prev_key = torch.randn(16)

        # Many updates with large movements
        for _ in range(1000):
            state.update_location([0], routing_key, prev_key)

        # Location should be bounded (norm <= 5.0 + epsilon)
        norms = state.location.norm(dim=1)
        assert norms.max().item() <= 5.5


class TestPredictiveColumnsInTrainer:
    """Test that predictive columns are wired into the training loop."""

    def test_hnsw_buffer_reuses_known_winner_ids(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(n_columns=16, column_latent_dim=8)
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        indices = torch.tensor([7], dtype=torch.long)
        vectors = trainer.model.competitive.prototypes[[3]].detach()

        trainer._buffer_hnsw_update(
            indices,
            vectors,
            known_ids=[3],
        )

        assert trainer._hnsw_buffer_ids == [3]

    def test_model_has_predictive_state(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel

        cfg = MarulhoConfig(n_columns=16, column_latent_dim=8)
        model = MarulhoModel(cfg)
        assert hasattr(model, 'predictive')
        assert model.predictive.n_columns == 16

    def test_trainer_uses_predictive_columns(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(n_columns=16, column_latent_dim=8, bootstrap_tokens=0, memory_capacity=32)
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True

        # Run a few steps
        for i in range(10):
            x = torch.randn(cfg.input_dim)
            metrics = trainer.train_step(x, raw_window=f'tok_{i}')

        # Predictive state should have evolved
        assert model.predictive.prediction_error.sum().item() > 0
        assert model.predictive.prediction_failure_streak.sum().item() >= 0
        assert trainer._prev_routing_key is not None
        assert metrics["trainer_telemetry_interval_tokens"] == cfg.trainer_telemetry_interval_tokens

    def test_trainer_caches_episode_terms_for_stream_text_refresh(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(n_columns=8, column_latent_dim=4, bootstrap_tokens=0, memory_capacity=16)
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)

        first = trainer._update_stream_text("alpha beta memory")
        terms = trainer._cached_episode_terms
        second = trainer._update_stream_text("alpha beta memory")

        assert first == second
        assert terms == {"alpha", "beta", "memory"}
        assert trainer._cached_episode_terms is terms

    def test_train_step_archives_raw_window_without_stream_episode_rebuild(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            slow_memory_archive_interval_tokens=1,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
        )
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        trainer.train_step(torch.randn(cfg.input_dim), raw_window="alpha")
        metrics = trainer.train_step(torch.randn(cfg.input_dim), raw_window="alpha beta")

        assert metrics["slow_memory_archive_due"] == 1
        assert trainer.model.memory_store.slow_texts[-1] == "alpha beta"
        assert trainer._cached_episode_text is None
        assert trainer._recent_stream_text == ""

    def test_trainer_train_step_profile_is_opt_in(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(n_columns=8, column_latent_dim=4, bootstrap_tokens=0, memory_capacity=16)
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)

        trainer.train_step(torch.randn(cfg.input_dim), raw_window="alpha beta")
        disabled_report = trainer.train_step_profile_report()
        assert disabled_report["enabled"] is False
        assert disabled_report["count"] == 0

        trainer.enable_train_step_profile()
        trainer.train_step(torch.randn(cfg.input_dim), raw_window="gamma delta")
        report = trainer.train_step_profile_report()

        assert report["enabled"] is True
        assert report["count"] == 1
        assert report["totals_ms"]["total"] > 0.0
        assert "routing_prepare" in report["per_tick_ms"]
        assert "metrics_build" in report["per_tick_ms"]

    def test_train_step_can_skip_metrics_packet_without_skipping_cognition(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(n_columns=8, column_latent_dim=4, bootstrap_tokens=0, memory_capacity=16)
        trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
        trainer.enable_train_step_profile()

        skipped = trainer.train_step(
            torch.randn(cfg.input_dim),
            raw_window="fast source window",
            return_metrics=False,
        )
        full = trainer.train_step(
            torch.randn(cfg.input_dim),
            raw_window="runtime truth window",
        )
        report = trainer.train_step_profile_report()

        assert skipped == {}
        assert trainer.token_count == 2
        assert full["train_step_metrics_full_count"] == 1
        assert full["train_step_metrics_skip_count"] == 1
        assert report["count"] == 2
        assert "metrics_build_skipped" in report["per_tick_ms"]
        assert "metrics_build" in report["per_tick_ms"]

    def test_trainer_cadences_slow_memory_archival(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            slow_memory_archive_interval_tokens=3,
            slow_memory_archive_strong_capture_threshold=999.0,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)

        metrics = {}
        for step in range(5):
            metrics = trainer.train_step(
                torch.randn(cfg.input_dim),
                raw_window=f"memory cadence {step}",
            )

        assert model.memory_store.update_calls == 2
        assert metrics["slow_memory_archive_interval_tokens"] == 3
        assert metrics["slow_memory_archive_count"] == 2
        assert metrics["slow_memory_archive_skip_count"] == 3
        assert metrics["slow_memory_archive_reason"] == "cadence_skip"

    def test_trainer_cadences_awake_ripple_tagging_with_archive(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            bootstrap_tokens=0,
            memory_capacity=16,
            slow_memory_archive_interval_tokens=3,
            slow_memory_archive_strong_capture_threshold=999.0,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.model.surprise.dopamine = 0.95
        trainer.model.surprise.update_neuromodulators = lambda *args, **kwargs: None

        metrics = {}
        for step in range(5):
            metrics = trainer.train_step(
                torch.randn(cfg.input_dim),
                raw_window=f"ripple cadence {step}",
            )

        assert model.memory_store.ripple_scalar_scan_count == 2
        assert metrics["awake_ripple_tag_count"] == 2
        assert metrics["awake_ripple_tag_skip_count"] == 3
        assert metrics["awake_ripple_last_reason"] == "cadence_skip"

    def test_trainer_delays_candidate_scoped_predictive_updates(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            dead_column_steps=100,
            candidate_predictive_update_start_tokens=2,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        pattern = torch.randn(cfg.input_dim)

        trainer.train_step(pattern, raw_window="early")
        assert model.predictive.last_prediction_update_mode == "all_columns"
        assert model.predictive.last_prediction_update_count == cfg.n_columns

        trainer.token_count = cfg.candidate_predictive_update_start_tokens
        trainer.train_step(pattern, raw_window="scoped")
        assert model.predictive.last_prediction_update_mode == "candidate_subset"
        assert model.predictive.last_prediction_update_count == cfg.k_routing
        assert model.predictive.last_prediction_update_fraction < 1.0

    def test_candidate_homeostasis_can_start_before_dead_column_scope(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            dead_column_steps=100,
            candidate_homeostasis_start_tokens=1,
            candidate_predictive_update_start_tokens=100,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        trainer.token_count = cfg.candidate_homeostasis_start_tokens

        trainer.train_step(torch.randn(cfg.input_dim), raw_window="scoped homeostasis")

        competitive = model.competitive.execution_report()
        assert competitive["homeostasis_update_mode"] == "candidate_subset"
        assert competitive["homeostasis_update_count"] == cfg.k_routing
        assert model.predictive.last_prediction_update_mode == "all_columns"
        assert model.predictive.last_prediction_update_count == cfg.n_columns
        assert model.predictive.last_prediction_update_fallback_reason == (
            "candidate_predictive_update_not_due"
        )

    def test_candidate_predictive_update_can_start_before_dead_column_scope(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            dead_column_steps=100,
            candidate_predictive_update_start_tokens=1,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        trainer.token_count = cfg.candidate_predictive_update_start_tokens

        trainer.train_step(torch.randn(cfg.input_dim), raw_window="scoped prediction")
        report = model.predictive.prediction_update_execution_report()

        assert report["mode"] == "candidate_subset"
        assert report["updated_column_count"] == cfg.k_routing
        assert report["cached_state_count"] == cfg.n_columns - cfg.k_routing
        assert report["runs_all_columns"] is False
        assert report["fallback_reason"] is None

    def test_trainer_scopes_predictive_vote_to_awake_candidates(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)

        trainer.train_step(torch.randn(cfg.input_dim), raw_window="first winner")
        trainer.train_step(torch.randn(cfg.input_dim), raw_window="scoped vote")
        report = model.predictive.vote_execution_report()

        assert report["mode"] == "awake_mask_cached_vote"
        assert report["updated_column_count"] == cfg.k_routing
        assert report["cached_vote_use_count"] == cfg.n_columns - cfg.k_routing
        assert report["runs_all_columns"] is False
        assert report["claim_boundary"] == (
            "training_owned_awake_mask_predictive_vote_cache_skips_non_awake_columns"
        )

    def test_candidate_deep_sleep_filter_skips_only_retrieved_stale_candidates(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            k_routing=3,
            bootstrap_tokens=0,
            memory_capacity=16,
            routing_index_mode="torch_topk",
            dead_column_steps=5,
            candidate_deep_sleep_filter_start_tokens=0,
            candidate_deep_sleep_backfill_factor=2,
            device="cpu",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        model.competitive.steps_since_win[:] = torch.tensor([5, 0, 7, 1, 9, 2, 0, 0])

        filtered = trainer._filter_candidate_deep_sleep(
            torch.tensor([0, 1, 2, 3, 4, 5]),
            target_count=cfg.k_routing,
        )

        assert filtered.tolist() == [1, 3, 5]
        report = model.column_runtime_report(token_count=0)
        sleep_filter = report["candidate_sleep_filter_execution"]
        assert sleep_filter["mode"] == "candidate_deep_sleep_filter"
        assert sleep_filter["input_candidate_count"] == 6
        assert sleep_filter["output_candidate_count"] == 3
        assert sleep_filter["filtered_deep_sleep_count"] == 3
        assert sleep_filter["runs_all_columns"] is False
        assert sleep_filter["claim_boundary"] == (
            "training_owned_candidate_deep_sleep_filter_skips_deep_sleep_candidates_without_all_column_scan"
        )

    def test_trainer_filters_deep_sleep_candidates_before_update_and_vote(self, monkeypatch):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=8,
            column_latent_dim=4,
            k_routing=2,
            bootstrap_tokens=0,
            memory_capacity=16,
            routing_index_mode="torch_topk",
            predictive_dense_transition_mode="legacy",
            dead_column_steps=3,
            candidate_homeostasis_start_tokens=0,
            candidate_predictive_update_start_tokens=0,
            candidate_deep_sleep_filter_start_tokens=0,
            candidate_deep_sleep_backfill_factor=2,
            micro_sleep_interval_tokens=10**9,
            deep_sleep_interval_tokens=10**9,
            slow_memory_archive_interval_tokens=10**9,
            device="cpu",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.last_winner = 1
        model.competitive.steps_since_win[:] = torch.tensor([3, 0, 3, 0, 0, 0, 0, 0])
        candidates_seen = {}

        def fake_wake_plan(routing_key, *, apply_sleep_filter=False):
            candidates_seen["apply_sleep_filter"] = apply_sleep_filter
            raw = torch.tensor([0, 1, 2, 3], dtype=torch.long)
            if apply_sleep_filter:
                return trainer._filter_candidate_deep_sleep_plan(
                    raw,
                    target_count=cfg.k_routing,
                )
            return trainer._build_column_wake_plan(
                mode="candidate_routing",
                awake_indices=raw[: cfg.k_routing],
                input_candidate_count=cfg.k_routing,
                filtered_deep_sleep_count=0,
                backfill_candidate_count=0,
                fallback_reason=None,
                wake_reason="retrieved_candidate",
                sleep_reason=None,
            )

        monkeypatch.setattr(trainer, "_routing_wake_plan", fake_wake_plan)

        metrics = trainer.train_step(
            torch.rand(cfg.input_dim),
            raw_window="candidate sleep filter test",
            allow_sleep_maintenance=False,
        )

        assert candidates_seen["apply_sleep_filter"] is True
        assert metrics["candidate_deep_sleep_filter_mode"] == "candidate_deep_sleep_filter"
        assert metrics["candidate_deep_sleep_filtered_count"] == 2
        report = model.column_runtime_report(
            token_count=trainer.token_count,
            last_winner=trainer.last_winner,
        )
        sleep_filter = report["candidate_sleep_filter_execution"]
        assert sleep_filter["output_candidate_count"] == cfg.k_routing
        assert sleep_filter["filtered_deep_sleep_count"] == 2
        wake_plan = report["column_wake_plan"]
        assert wake_plan["surface"] == "column_wake_plan.v1"
        assert wake_plan["mode"] == "candidate_deep_sleep_filter"
        assert wake_plan["awake_count"] == cfg.k_routing
        assert wake_plan["awake_column_ids_sample"] == [1, 3]
        assert wake_plan["bounded"] is True
        assert wake_plan["runs_all_columns"] is False
        assert wake_plan["wake_reason"] == "retrieved_candidate_not_in_deep_sleep"
        assert wake_plan["sleep_reason"] == "deep_sleep_candidate_filtered_from_awake_mask"
        assert wake_plan["claim_boundary"] == (
            "training_owned_column_wake_plan_bounds_specialist_execution_without_all_column_sleep_scan"
        )
        assert report["predictive_update_execution"]["updated_column_count"] == cfg.k_routing
        assert report["predictive_update_execution"]["location_update_count"] == cfg.k_routing
        assert report["predictive_update_execution"]["location_cached_count"] == (
            cfg.n_columns - cfg.k_routing
        )
        assert report["predictive_vote_execution"]["updated_column_count"] == cfg.k_routing
        assert report["execution"]["homeostasis_update_count"] == cfg.k_routing
        assert report["runs_all_columns"] is False

    def test_candidate_homeostasis_start_tokens_must_be_non_negative(self):
        from marulho.config.model_config import MarulhoConfig

        with pytest.raises(ValueError, match="candidate_homeostasis_start_tokens"):
            MarulhoConfig(candidate_homeostasis_start_tokens=-1)

    def test_candidate_predictive_update_start_tokens_must_be_non_negative(self):
        from marulho.config.model_config import MarulhoConfig

        with pytest.raises(ValueError, match="candidate_predictive_update_start_tokens"):
            MarulhoConfig(candidate_predictive_update_start_tokens=-1)

    def test_candidate_deep_sleep_filter_start_tokens_must_be_non_negative(self):
        from marulho.config.model_config import MarulhoConfig

        with pytest.raises(ValueError, match="candidate_deep_sleep_filter_start_tokens"):
            MarulhoConfig(candidate_deep_sleep_filter_start_tokens=-1)

    def test_candidate_deep_sleep_backfill_factor_must_be_positive(self):
        from marulho.config.model_config import MarulhoConfig

        with pytest.raises(ValueError, match="candidate_deep_sleep_backfill_factor"):
            MarulhoConfig(candidate_deep_sleep_backfill_factor=0)

    def test_predictive_route_vote_mode_must_be_supported(self):
        with pytest.raises(ValueError, match="predictive_route_vote_mode"):
            MarulhoConfig(predictive_route_vote_mode="always")  # type: ignore[arg-type]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_trainer_reports_cuda_predictive_scope_fallback(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            dead_column_steps=1,
            candidate_predictive_update_start_tokens=1,
            device="cuda",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        trainer.token_count = cfg.dead_column_steps

        trainer.train_step(torch.randn(cfg.input_dim, device=model.device), raw_window="cuda-first")
        trainer.train_step(torch.randn(cfg.input_dim, device=model.device), raw_window="cuda-steady")
        report = model.predictive.device_report()

        assert report["last_prediction_update_mode"] == "all_columns"
        assert report["last_prediction_update_count"] == cfg.n_columns
        assert report["last_prediction_update_runs_all_columns"] is True
        assert report["last_prediction_update_fallback_reason"] == (
            "cuda_sparse_prediction_update_launch_bound_dense_retained"
        )
        assert report["last_dense_transition_mode"] == "compiled"
        assert report["dense_transition_compile_count"] == 1
        assert report["last_dense_transition_fallback_reason"] is None
