"""Tests for PredictiveColumnState (Thousand Brains Theory extension)."""

from __future__ import annotations

import torch
import pytest

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
        assert report["last_prediction_update_mode"] == "all_columns"
        assert report["last_prediction_update_count"] == 8
        assert report["last_prediction_update_fraction"] == 1.0

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

    def test_prediction_error_computation(self):
        state = PredictiveColumnState(n_columns=16, location_dim=4)
        routing_key = torch.randn(16)

        error = state.compute_prediction_error([0, 1], routing_key)
        assert error.shape == (16,)
        assert error.min() >= 0.0
        assert error.max() <= 1.0

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

        saved = state.state_dict()
        state2 = PredictiveColumnState(n_columns=16, location_dim=4)
        state2.load_state_dict(saved)

        assert torch.allclose(state.location, state2.location)
        assert torch.allclose(state.confidence, state2.confidence)
        assert torch.equal(state.prediction_failure_streak, state2.prediction_failure_streak)

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
            trainer.train_step(x, raw_window=f'tok_{i}')

        # Predictive state should have evolved
        assert model.predictive.prediction_error.sum().item() > 0
        assert model.predictive.prediction_failure_streak.sum().item() >= 0
        assert trainer._prev_routing_key is not None

    def test_trainer_delays_candidate_scoped_predictive_updates(self):
        from marulho.config.model_config import MarulhoConfig
        from marulho.training.model import MarulhoModel
        from marulho.training.trainer import MarulhoTrainer

        cfg = MarulhoConfig(
            n_columns=16,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=32,
            dead_column_steps=2,
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        pattern = torch.randn(cfg.input_dim)

        trainer.train_step(pattern, raw_window="early")
        assert model.predictive.last_prediction_update_mode == "all_columns"
        assert model.predictive.last_prediction_update_count == cfg.n_columns

        trainer.token_count = cfg.dead_column_steps
        trainer.train_step(pattern, raw_window="scoped")
        assert model.predictive.last_prediction_update_mode == "candidate_subset"
        assert model.predictive.last_prediction_update_count == cfg.k_routing
        assert model.predictive.last_prediction_update_fraction < 1.0

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
            device="cuda",
        )
        model = MarulhoModel(cfg)
        trainer = MarulhoTrainer(model, cfg)
        trainer.memory_warm_started = True
        trainer.token_count = cfg.dead_column_steps

        trainer.train_step(torch.randn(cfg.input_dim, device=model.device), raw_window="cuda")
        report = model.predictive.device_report()

        assert report["last_prediction_update_mode"] == "all_columns"
        assert report["last_prediction_update_count"] == cfg.n_columns
        assert report["last_prediction_update_fallback_reason"] == (
            "cuda_sparse_prediction_update_launch_bound_dense_retained"
        )
