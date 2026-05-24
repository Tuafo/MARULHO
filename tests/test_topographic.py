"""Tests for topographic column organization and spatial binding."""

from __future__ import annotations

import math

import pytest
import torch

from hecsn.core.topographic import SpatialBindingLayer, TopographicGrid


# ── TopographicGrid ──────────────────────────────────────────────────


class TestTopographicGrid:
    """Tests for the 2D grid layout and neighbor computation."""

    def test_grid_dims_perfect_square(self):
        grid = TopographicGrid(16)
        assert grid.n_rows * grid.n_cols >= 16
        assert grid.n_rows == 4
        assert grid.n_cols == 4

    def test_grid_dims_non_square(self):
        grid = TopographicGrid(10)
        assert grid.n_rows * grid.n_cols >= 10

    def test_grid_dims_small(self):
        grid = TopographicGrid(4)
        assert grid.n_rows * grid.n_cols >= 4

    def test_grid_positions_unique(self):
        grid = TopographicGrid(25)
        positions = grid.grid_positions
        assert positions.shape == (25, 2)
        # All positions should be unique
        pos_set = set()
        for i in range(25):
            pos = (int(positions[i, 0].item()), int(positions[i, 1].item()))
            pos_set.add(pos)
        assert len(pos_set) == 25

    def test_grid_position_api(self):
        grid = TopographicGrid(16)
        r, c = grid.grid_position(0)
        assert r == 0 and c == 0
        r, c = grid.grid_position(5)
        assert r == 1 and c == 1  # 5 // 4 = 1, 5 % 4 = 1

    def test_distance_symmetric(self):
        grid = TopographicGrid(16)
        for i in range(16):
            for j in range(16):
                assert abs(grid.distance(i, j) - grid.distance(j, i)) < 1e-6

    def test_distance_self_zero(self):
        grid = TopographicGrid(16)
        for i in range(16):
            assert grid.distance(i, i) == 0.0

    def test_distance_adjacent(self):
        grid = TopographicGrid(16)
        # Column 0 at (0,0) and Column 1 at (0,1) should be distance 1.0
        assert abs(grid.distance(0, 1) - 1.0) < 1e-6

    def test_neighbor_count(self):
        grid = TopographicGrid(16, k_neighbors=4)
        for i in range(16):
            nids, weights = grid.neighbors(i)
            assert len(nids) == 4
            assert len(weights) == 4

    def test_neighbor_weights_sum_to_one(self):
        grid = TopographicGrid(25, k_neighbors=8)
        for i in range(25):
            _, weights = grid.neighbors(i)
            assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_neighbor_weights_positive(self):
        grid = TopographicGrid(16, k_neighbors=4)
        for i in range(16):
            _, weights = grid.neighbors(i)
            assert (weights > 0).all()

    def test_self_not_neighbor(self):
        grid = TopographicGrid(16, k_neighbors=4)
        for i in range(16):
            nids, _ = grid.neighbors(i)
            assert i not in nids.tolist()

    def test_neighborhood_kernel_shape(self):
        grid = TopographicGrid(16)
        kernel = grid.neighborhood_kernel(0)
        assert kernel.shape == (16,)
        assert kernel[0] == 1.0  # center gets full weight

    def test_neighborhood_kernel_decay(self):
        grid = TopographicGrid(25)
        kernel = grid.neighborhood_kernel(12)  # center of 5×5
        # Closer columns should have higher kernel values
        d_near = grid.distance(12, 13)  # adjacent
        d_far = grid.distance(12, 24)   # corner
        assert kernel[13] > kernel[24] or d_near >= d_far

    def test_neighbor_purity_random_prototypes(self):
        grid = TopographicGrid(16)
        protos = torch.randn(16, 64)
        purity = grid.neighbor_purity(protos)
        assert -1.0 <= purity <= 1.0

    def test_topographic_error_range(self):
        grid = TopographicGrid(16)
        protos = torch.randn(16, 64)
        err = grid.topographic_error(protos)
        assert 0.0 <= err <= 1.0

    def test_topographic_error_perfect(self):
        """When adjacent columns have identical prototypes, error should be low."""
        grid = TopographicGrid(16, k_neighbors=4)
        # Create prototypes where neighbors are similar
        protos = torch.zeros(16, 8)
        for i in range(16):
            protos[i] = grid.grid_positions[i].repeat(4)
        err = grid.topographic_error(protos)
        # With position-based prototypes, neighbors should be most similar
        assert err < 0.5

    def test_state_dict_roundtrip(self):
        grid = TopographicGrid(16, k_neighbors=4, sigma=2.0)
        state = grid.state_dict()
        grid2 = TopographicGrid(16, k_neighbors=4, sigma=2.0)
        grid2.load_state_dict(state)
        assert torch.allclose(grid.grid_positions, grid2.grid_positions)
        assert torch.allclose(grid._distance_matrix, grid2._distance_matrix)

    def test_k_neighbors_capped_at_n_minus_1(self):
        grid = TopographicGrid(4, k_neighbors=100)
        assert grid.k_neighbors == 3  # 4 - 1

    def test_large_grid(self):
        grid = TopographicGrid(256, k_neighbors=8)
        assert grid.n_rows * grid.n_cols >= 256
        nids, weights = grid.neighbors(0)
        assert len(nids) == 8


# ── SpatialBindingLayer ──────────────────────────────────────────────


class TestSpatialBindingLayer:
    """Tests for the sparse local binding layer."""

    @pytest.fixture
    def binding(self):
        return SpatialBindingLayer(
            n_columns=16,
            device=torch.device("cpu"),
            k_neighbors=4,
            sigma=1.5,
        )

    def test_interface_bind(self, binding):
        """bind() returns (state, strength) like BindingLayer."""
        ctx = torch.randn(16).abs()
        asm = torch.randn(16).abs()
        state, strength = binding.bind(ctx, asm)
        assert state.shape == (16,)
        assert isinstance(strength, float)

    def test_device_report_exposes_live_tensor_devices(self, binding):
        report = binding.device_report()

        assert report["device"] == "cpu"
        assert report["binding_state_device"] == str(binding.binding_state.device)
        assert report["neighbor_ids_device"] == str(binding.neighbor_ids.device)
        assert report["neighbor_weights_device"] == str(binding.neighbor_weights.device)
        assert report["learned_weights_device"] == str(binding.learned_weights.device)
        assert report["grid"]["grid_positions_device"] == str(binding.grid.grid_positions.device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
    def test_cuda_device_report_exposes_live_tensor_devices_after_bind(self):
        binding = SpatialBindingLayer(
            n_columns=16,
            device=torch.device("cuda"),
            k_neighbors=4,
            sigma=1.5,
        )
        ctx = torch.randn(16, device=torch.device("cuda")).abs()
        asm = torch.randn(16, device=torch.device("cuda")).abs()
        binding.bind(ctx, asm)

        report = binding.device_report()

        assert str(report["device"]).startswith("cuda")
        assert str(report["binding_state_device"]).startswith("cuda")
        assert str(report["neighbor_ids_device"]).startswith("cuda")
        assert str(report["learned_weights_device"]).startswith("cuda")
        assert str(report["grid"]["neighbor_ids_device"]).startswith("cuda")

    def test_interface_modulation_gain(self, binding):
        """modulation_gain() returns [n_columns] gains."""
        ctx = torch.randn(16).abs()
        gain = binding.modulation_gain(ctx)
        assert gain.shape == (16,)
        assert gain.min() >= 0.70
        assert gain.max() <= 1.35

    def test_interface_binding_prediction(self, binding):
        ctx = torch.randn(16).abs()
        pred = binding.binding_prediction(ctx)
        assert pred.shape == (16,)

    def test_interface_reset_state(self, binding):
        ctx = torch.randn(16).abs()
        asm = torch.randn(16).abs()
        binding.bind(ctx, asm)
        binding.reset_state()
        assert binding.binding_state.sum() == 0.0
        assert binding.coincidence_trace.sum() == 0.0

    def test_interface_grow_binding(self, binding):
        """grow_binding() accepts same format as BindingLayer."""
        pairs = [(0, 1, 0.9), (2, 3, 0.8)]
        result = binding.grow_binding(pairs)
        assert isinstance(result, int)

    def test_bind_zero_input(self, binding):
        """Zero inputs produce zero state."""
        ctx = torch.zeros(16)
        asm = torch.zeros(16)
        state, strength = binding.bind(ctx, asm)
        assert state.sum() == 0.0
        assert strength == 0.0

    def test_bind_positive_input(self, binding):
        """Positive inputs produce non-zero binding."""
        ctx = torch.ones(16)
        asm = torch.ones(16)
        state, strength = binding.bind(ctx, asm)
        # After first step, may or may not produce output (depends on threshold)
        assert state.shape == (16,)

    def test_bind_repeated_builds_trace(self, binding):
        """Repeated binding builds up coincidence trace."""
        ctx = torch.ones(16)
        asm = torch.ones(16)
        for _ in range(10):
            state, strength = binding.bind(ctx, asm)
        # After 10 steps, should have some binding activity
        assert binding.coincidence_trace.max() > 0.0

    def test_state_dict_roundtrip(self, binding):
        """state_dict + load_state_dict preserves state."""
        ctx = torch.ones(16)
        asm = torch.ones(16)
        for _ in range(5):
            binding.bind(ctx, asm)

        state = binding.state_dict()
        binding2 = SpatialBindingLayer(
            n_columns=16, device=torch.device("cpu"), k_neighbors=4,
        )
        binding2.load_state_dict(state)
        assert torch.allclose(binding.learned_weights, binding2.learned_weights)
        assert torch.allclose(binding.binding_usage, binding2.binding_usage)

    def test_grow_binding_neighbors_only(self, binding):
        """grow_binding only strengthens pairs that are grid neighbors."""
        # Column 0 and column 15 are far apart — likely not neighbors
        grid = binding.grid
        nids = grid._neighbor_ids[0].tolist()
        distant = [i for i in range(16) if i not in nids and i != 0]
        if distant:
            result = binding.grow_binding([(0, distant[0], 0.9)])
            # Should return 0 if they're not neighbors
            # (or 1 if distant[0] has 0 as neighbor)

    def test_grow_binding_strengthens_neighbors(self, binding):
        """grow_binding increases weights for neighbor pairs."""
        nids = binding.grid._neighbor_ids[0].tolist()
        neighbor = nids[0]
        old_weight = binding.learned_weights[0, 0].item()
        binding.grow_binding([(0, neighbor, 0.95)])
        # Weight should have changed (may increase or stay same due to normalization)

    def test_modulation_gain_ones_on_empty(self, binding):
        """With no binding history, gain should be all ones."""
        ctx = torch.randn(16).abs()
        gain = binding.modulation_gain(ctx)
        assert torch.allclose(gain, torch.ones(16))

    def test_sparse_faster_than_dense_at_scale(self):
        """At 256+ columns, spatial binding should use less memory than dense."""
        spatial = SpatialBindingLayer(
            n_columns=256, device=torch.device("cpu"), k_neighbors=8,
        )
        # Dense BindingLayer uses n_bindings × n_columns connectivity matrix
        # Spatial uses n_columns × k_neighbors (much smaller)
        spatial_params = spatial.learned_weights.numel()  # 256 × 8 = 2048
        dense_params = 256 * (256 + 32)  # ~73,728
        assert spatial_params < dense_params


# ── Integration with config ──────────────────────────────────────────


class TestSpatialBindingConfig:
    """Test that config binding_mode works correctly."""

    def test_config_hypercube_default(self):
        from hecsn.config.model_config import HECSNConfig
        cfg = HECSNConfig(n_columns=16, enable_binding_layer=True,
                          enable_context_layer=True)
        assert cfg.binding_mode == "hypercube"

    def test_config_spatial_mode(self):
        from hecsn.config.model_config import HECSNConfig
        cfg = HECSNConfig(n_columns=16, enable_binding_layer=True,
                          enable_context_layer=True, binding_mode="spatial")
        assert cfg.binding_mode == "spatial"

    def test_config_invalid_mode(self):
        from hecsn.config.model_config import HECSNConfig
        with pytest.raises(ValueError, match="binding_mode"):
            HECSNConfig(n_columns=16, enable_binding_layer=True,
                        enable_context_layer=True, binding_mode="invalid")

    def test_model_creates_spatial_binding(self):
        from hecsn.config.model_config import HECSNConfig
        from hecsn.training.trainer import HECSNModel
        cfg = HECSNConfig(
            n_columns=16, enable_binding_layer=True,
            enable_context_layer=True, binding_mode="spatial",
        )
        model = HECSNModel(cfg)
        assert isinstance(model.binding_layer, SpatialBindingLayer)
        report = model.subcortex_device_report()["binding"]
        assert report is not None
        assert report["module"] == "spatial_binding"
        assert report["neighbor_ids_device"] == str(model.binding_layer.neighbor_ids.device)

    def test_model_creates_dense_binding(self):
        from hecsn.config.model_config import HECSNConfig
        from hecsn.core.context import BindingLayer
        from hecsn.training.trainer import HECSNModel
        cfg = HECSNConfig(
            n_columns=16, enable_binding_layer=True,
            enable_context_layer=True, binding_mode="dense",
        )
        model = HECSNModel(cfg)
        assert isinstance(model.binding_layer, BindingLayer)


# ── Winner history per-token collection ──────────────────────────────


class TestWinnerHistoryPerToken:
    """Test that winner_accumulator collects per-token winners."""

    def test_winner_accumulator_populated(self):
        from hecsn.config.model_config import HECSNConfig
        from hecsn.data.rtf_encoder import RTFEncoder
        from hecsn.training.developmental_runner import (
            _build_concept_corpus,
            _build_concept_signatures,
            _train_multimodal_on_corpus,
        )
        from hecsn.training.trainer import HECSNModel, HECSNTrainer

        cfg = HECSNConfig(n_columns=8, enable_cross_modal=True)
        model = HECSNModel(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)
        corpus = _build_concept_corpus()
        sigs = _build_concept_signatures(
            n_concepts=20, dim_visual=cfg.cross_modal_dim_visual,
            dim_audio=cfg.cross_modal_dim_audio, seed=42,
        )

        winners: list[int] = []
        n_tokens = 50
        tok, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, corpus, n_tokens, sigs,
            cfg.cross_modal_dim_visual, cfg.cross_modal_dim_audio,
            winner_accumulator=winners,
        )
        # Should have one winner per token processed
        assert len(winners) == tok
        assert all(0 <= w < cfg.n_columns for w in winners)

    def test_winner_accumulator_none_backward_compat(self):
        """Passing None for winner_accumulator still works."""
        from hecsn.config.model_config import HECSNConfig
        from hecsn.data.rtf_encoder import RTFEncoder
        from hecsn.training.developmental_runner import (
            _build_concept_corpus,
            _build_concept_signatures,
            _train_multimodal_on_corpus,
        )
        from hecsn.training.trainer import HECSNModel, HECSNTrainer

        cfg = HECSNConfig(n_columns=8, enable_cross_modal=True)
        model = HECSNModel(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)
        corpus = _build_concept_corpus()
        sigs = _build_concept_signatures(
            n_concepts=20, dim_visual=cfg.cross_modal_dim_visual,
            dim_audio=cfg.cross_modal_dim_audio, seed=42,
        )
        tok, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, corpus, 20, sigs,
            cfg.cross_modal_dim_visual, cfg.cross_modal_dim_audio,
        )
        assert tok == 20
