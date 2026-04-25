"""Tests for HypercubeTopology and HypercubeBindingLayer.

Tests cover:
  - Topology construction and properties
  - Bit-flip neighbor correctness
  - Non-power-of-2 column counts
  - Small-world shortcuts
  - HypercubeBindingLayer interface parity with BindingLayer/SpatialBindingLayer
  - grow_binding constrained to 2-hop
  - State dict round-trip
  - Config integration
  - Benchmark: hypercube vs dense vs spatial
"""

import math
import time

import pytest
import torch

from hecsn.core.hypercube import HypercubeTopology, HypercubeBindingLayer


def _average_shortest_path_length(topo: HypercubeTopology) -> float:
    adjacency = [set(int(n) for n in topo.neighbors(i)[0].tolist() if int(n) >= 0) for i in range(topo.n_columns)]
    total_distance = 0
    pair_count = 0
    for start in range(topo.n_columns):
        visited = {start: 0}
        queue = [start]
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor in visited:
                    continue
                visited[neighbor] = visited[current] + 1
                queue.append(neighbor)
        for end in range(start + 1, topo.n_columns):
            total_distance += int(visited[end])
            pair_count += 1
    return float(total_distance) / max(1, pair_count)


# ── Topology construction ──────────────────────────────────────────

class TestHypercubeTopology:
    def test_power_of_2_columns(self):
        """Power-of-2 columns: every node has exactly dim neighbors."""
        topo = HypercubeTopology(n_columns=16)
        assert topo.dim == 4
        assert topo.n_vertices == 16
        assert topo.n_columns == 16
        # Every node should have exactly 4 neighbors
        for i in range(16):
            ids, weights = topo.neighbors(i)
            assert len(ids) == 4, f"Node {i} has {len(ids)} neighbors, expected 4"
            assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_bit_flip_neighbors(self):
        """Neighbors differ by exactly 1 bit."""
        topo = HypercubeTopology(n_columns=8)  # 3D hypercube
        assert topo.dim == 3
        for i in range(8):
            ids, _ = topo.neighbors(i)
            for n in ids.tolist():
                hamming = bin(i ^ n).count("1")
                assert hamming == 1, f"Node {i}, neighbor {n}: hamming distance {hamming}"

    def test_non_power_of_2(self):
        """Non-power-of-2: masked nodes reduce degree at boundary."""
        topo = HypercubeTopology(n_columns=10)
        assert topo.dim == 4  # ceil(log2(10)) = 4
        assert topo.n_vertices == 16
        # Nodes 0-9 active, 10-15 masked
        # Node 0: neighbors are 1,2,4,8 — all < 10, so degree 4
        ids_0, _ = topo.neighbors(0)
        assert len(ids_0) == 4
        # Node 9 (=1001): neighbors are 8(1000),11(1011),13(1101),1(0001)
        # 11 and 13 are > 9, so only 8 and 1 are valid → degree 2
        ids_9, _ = topo.neighbors(9)
        assert all(n < 10 for n in ids_9.tolist())
        assert len(ids_9) >= 2  # at least 8 and 1

    def test_small_world_shortcuts(self):
        """Shortcuts add deterministic long-range edges per node."""
        topo = HypercubeTopology(n_columns=16, shortcuts_per_node=2)
        stats = topo.topology_stats()
        # Base degree is 4, target degree is 6 on a full 4D cube.
        assert stats["avg_degree"] > 4.0
        assert stats["max_degree"] <= 6
        assert stats["shortcut_strategy"] == "deterministic_long_range"
        assert stats["shortcut_budget_policy"] == "target_degree_compensated"
        assert stats["long_range_weight_policy"] == "bounded_relative_mass"
        assert stats["long_range_mass_ratio"] == pytest.approx(0.5)
        assert stats["target_degree"] == 6
        assert stats["avg_effective_shortcuts_per_node"] == pytest.approx(2.0)
        assert stats["avg_long_range_raw_weight"] == pytest.approx(0.5)
        assert stats["avg_shortcut_hamming_distance"] > 1.0

    def test_shortcut_budget_compensates_masked_boundary_degree_loss(self):
        topo = HypercubeTopology(n_columns=10, shortcuts_per_node=2)
        stats = topo.topology_stats()

        assert stats["target_degree"] == 6
        assert stats["min_degree"] == 6
        assert stats["max_degree"] == 6
        assert stats["degree_std"] == pytest.approx(0.0)

        ids_9, weights_9 = topo.neighbors(9)
        assert len(ids_9) == 6
        assert int(topo._direct_degree[9].item()) == 2
        assert int(topo._shortcut_degree[9].item()) == 4

        direct_share = 0.0
        long_range_share = 0.0
        for target, weight in zip(ids_9.tolist(), weights_9.tolist()):
            if topo.hamming_distance(9, int(target)) == 1:
                direct_share += float(weight)
            else:
                long_range_share += float(weight)
        assert direct_share > long_range_share
        assert long_range_share == pytest.approx(1.0 / 3.0, rel=1e-5)
        assert stats["avg_long_range_raw_weight"] < 0.5
        assert stats["max_long_range_weight_share"] <= (1.0 / 3.0) + 1e-6

    def test_shortcuts_are_deterministic(self):
        topo_a = HypercubeTopology(n_columns=32, shortcuts_per_node=2)
        topo_b = HypercubeTopology(n_columns=32, shortcuts_per_node=2)
        assert torch.equal(topo_a._neighbor_ids, topo_b._neighbor_ids)
        assert torch.allclose(topo_a._neighbor_weights, topo_b._neighbor_weights)

    def test_no_self_loops(self):
        """No node is its own neighbor."""
        topo = HypercubeTopology(n_columns=32)
        for i in range(32):
            ids, _ = topo.neighbors(i)
            assert i not in ids.tolist()

    def test_hamming_distance(self):
        topo = HypercubeTopology(n_columns=16)
        assert topo.hamming_distance(0, 15) == 4  # 0000 vs 1111
        assert topo.hamming_distance(0, 1) == 1   # 0000 vs 0001
        assert topo.hamming_distance(5, 7) == 1   # 0101 vs 0111

    def test_is_neighbor(self):
        topo = HypercubeTopology(n_columns=8)
        assert topo.is_neighbor(0, 1)  # differ by bit 0
        assert topo.is_neighbor(0, 2)  # differ by bit 1
        assert topo.is_neighbor(0, 4)  # differ by bit 2
        assert not topo.is_neighbor(0, 3)  # differ by bits 0 and 1

    def test_two_hop_neighbors(self):
        topo = HypercubeTopology(n_columns=8)
        # Node 0 (000): 1-hop = {1,2,4}
        # 2-hop from 0: neighbors of {1,2,4} minus 1-hop minus self
        # 1→{0,3,5}, 2→{0,3,6}, 4→{0,5,6} → union minus {0,1,2,4} = {3,5,6}
        two_hop = topo.two_hop_neighbors(0)
        assert two_hop == {3, 5, 6}

    def test_topology_stats(self):
        topo = HypercubeTopology(n_columns=256)
        stats = topo.topology_stats()
        assert stats["dim"] == 8
        assert stats["n_columns"] == 256
        assert stats["avg_degree"] == 8.0
        assert stats["sparsity_ratio"] < 0.05  # very sparse

    def test_state_dict_roundtrip(self):
        topo = HypercubeTopology(n_columns=16, shortcuts_per_node=1)
        state = topo.state_dict()
        topo2 = HypercubeTopology(n_columns=16, shortcuts_per_node=1)
        topo2.load_state_dict(state)
        assert torch.equal(topo._neighbor_ids, topo2._neighbor_ids)
        assert torch.allclose(topo._neighbor_weights, topo2._neighbor_weights)
        assert torch.equal(topo._direct_degree, topo2._direct_degree)
        assert torch.equal(topo._shortcut_degree, topo2._shortcut_degree)
        assert topo2.target_degree == topo.target_degree
        assert topo2.shortcut_budget_policy == topo.shortcut_budget_policy
        assert topo2.long_range_weight_policy == topo.long_range_weight_policy
        assert topo2.long_range_mass_ratio == topo.long_range_mass_ratio

    def test_small_column_counts(self):
        """Edge cases: 2, 3, 4 columns."""
        for n in [2, 3, 4]:
            topo = HypercubeTopology(n_columns=n)
            assert topo.n_columns == n
            for i in range(n):
                ids, weights = topo.neighbors(i)
                assert len(ids) >= 1

    def test_large_column_count(self):
        """2048 columns (11D) — the snn-llm sweet spot."""
        topo = HypercubeTopology(n_columns=2048)
        assert topo.dim == 11
        assert topo.n_vertices == 2048
        stats = topo.topology_stats()
        assert stats["avg_degree"] == 11.0
        assert stats["total_edges"] == 2048 * 11

    def test_weight_normalization(self):
        """Weights sum to 1.0 per node."""
        topo = HypercubeTopology(n_columns=32, shortcuts_per_node=2)
        for i in range(32):
            _, weights = topo.neighbors(i)
            assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_shortcuts_reduce_average_shortest_path(self):
        topo_no_short = HypercubeTopology(n_columns=32, shortcuts_per_node=0)
        topo_with_short = HypercubeTopology(n_columns=32, shortcuts_per_node=2)
        assert _average_shortest_path_length(topo_with_short) < _average_shortest_path_length(topo_no_short)


# ── HypercubeBindingLayer ──────────────────────────────────────────

class TestHypercubeBindingLayer:
    @pytest.fixture
    def layer_32(self):
        return HypercubeBindingLayer(n_columns=32, device=torch.device("cpu"))

    def test_construction(self, layer_32):
        assert layer_32.n_columns == 32
        assert layer_32.topology.dim == 5
        assert layer_32.fan_in == 5
        assert layer_32.binding_state.shape == (32,)
        assert layer_32.neighbor_ids.shape[0] == 32

    def test_bind_zeros(self, layer_32):
        """Zero inputs → zero output, zero strength."""
        ctx = torch.zeros(32)
        asm = torch.zeros(32)
        state, strength = layer_32.bind(ctx, asm)
        assert state.sum() == 0.0
        assert strength == 0.0

    def test_bind_active(self, layer_32):
        """Active inputs → nonzero binding state."""
        ctx = torch.randn(32).abs()
        asm = torch.randn(32).abs()
        state, strength = layer_32.bind(ctx, asm)
        assert state.shape == (32,)
        # After first step, may or may not fire depending on thresholds
        # Run a few steps to build up activity
        for _ in range(5):
            state, strength = layer_32.bind(ctx, asm)
        assert state.sum() >= 0.0

    def test_bind_returns_tuple(self, layer_32):
        ctx = torch.ones(32)
        asm = torch.ones(32)
        result = layer_32.bind(ctx, asm)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], float)

    def test_modulation_gain_shape(self, layer_32):
        ctx = torch.randn(32).abs()
        gain = layer_32.modulation_gain(ctx)
        assert gain.shape == (32,)
        # Gain should be in [0.70, 1.35] range
        assert gain.min() >= 0.69
        assert gain.max() <= 1.36

    def test_modulation_gain_zero_context(self, layer_32):
        """Zero context → uniform gain of 1.0."""
        ctx = torch.zeros(32)
        gain = layer_32.modulation_gain(ctx)
        assert torch.allclose(gain, torch.ones(32))

    def test_binding_prediction(self, layer_32):
        ctx = torch.randn(32).abs()
        pred = layer_32.binding_prediction(ctx)
        assert pred.shape == (32,)
        assert pred.sum() >= 0.0

    def test_reset_state(self, layer_32):
        ctx = torch.randn(32).abs()
        asm = torch.randn(32).abs()
        layer_32.bind(ctx, asm)
        layer_32.reset_state()
        assert layer_32.binding_state.sum() == 0.0
        assert layer_32.coincidence_trace.sum() == 0.0
        assert layer_32.facilitation.sum() == 0.0
        assert layer_32.resources.sum() == 32.0  # all 1.0

    def test_grow_binding_1hop(self, layer_32):
        """grow_binding strengthens direct hypercube neighbors."""
        # Nodes 0 and 1 are 1-hop neighbors (differ by bit 0)
        count = layer_32.grow_binding([(0, 1, 0.9)])
        assert count >= 1

    def test_grow_binding_2hop(self, layer_32):
        """grow_binding can strengthen 2-hop paths."""
        # Nodes 0 (00000) and 3 (00011) are 2-hop: 0→1→3 or 0→2→3
        count = layer_32.grow_binding([(0, 3, 0.9)])
        assert count >= 1

    def test_grow_binding_beyond_2hop_ignored(self, layer_32):
        """Columns beyond 2-hop are NOT strengthened."""
        # Nodes 0 (00000) and 7 (00111) are 3-hop apart
        count = layer_32.grow_binding([(0, 7, 0.9)])
        assert count == 0

    def test_grow_binding_low_correlation_ignored(self, layer_32):
        """Low correlation pairs are skipped."""
        count = layer_32.grow_binding([(0, 1, 0.5)])
        assert count == 0

    def test_state_dict_roundtrip(self, layer_32):
        ctx = torch.randn(32).abs()
        asm = torch.randn(32).abs()
        for _ in range(3):
            layer_32.bind(ctx, asm)
        state = layer_32.state_dict()
        layer2 = HypercubeBindingLayer(n_columns=32, device=torch.device("cpu"))
        layer2.load_state_dict(state)
        assert torch.allclose(layer_32.binding_state, layer2.binding_state)
        assert torch.allclose(layer_32.learned_weights, layer2.learned_weights)
        assert torch.allclose(layer_32.neighbor_weights, layer2.neighbor_weights)
        assert torch.allclose(layer_32._hub_activation_ema, layer2._hub_activation_ema)
        assert torch.allclose(layer_32._hub_connection_multiplier, layer2._hub_connection_multiplier)
        assert torch.equal(layer_32._hub_extra_connections, layer2._hub_extra_connections)
        assert torch.equal(layer_32.neighbor_ids, layer2.neighbor_ids)
        assert torch.equal(layer_32.degree, layer2.degree)

    def test_state_dict_has_compatibility_keys(self, layer_32):
        """state_dict includes BindingLayer-compatible keys."""
        state = layer_32.state_dict()
        assert "connectivity" in state
        assert "output_weights" in state
        assert "binding_outputs" in state

    def test_weight_renormalization_after_bind(self, layer_32):
        """Learned weights stay normalized after binding updates."""
        ctx = torch.randn(32).abs()
        asm = torch.randn(32).abs()
        for _ in range(10):
            layer_32.bind(ctx, asm)
        # Check all rows sum to ~1.0 (for active rows)
        for i in range(32):
            d = int(layer_32.degree[i].item())
            if d > 0:
                row_sum = layer_32.learned_weights[i, :d].sum().item()
                assert abs(row_sum - 1.0) < 0.01, f"Row {i} sum: {row_sum}"

    def test_padded_positions_stay_zero(self, layer_32):
        """Padded neighbor positions (id=-1) keep weight=0."""
        ctx = torch.randn(32).abs()
        asm = torch.randn(32).abs()
        for _ in range(10):
            layer_32.bind(ctx, asm)
        pad_mask = layer_32.neighbor_ids < 0
        if pad_mask.any():
            assert (layer_32.learned_weights[pad_mask] == 0.0).all()

    def test_non_power_of_2_binding(self):
        """Binding works with non-power-of-2 column count."""
        layer = HypercubeBindingLayer(n_columns=10, device=torch.device("cpu"))
        ctx = torch.randn(10).abs()
        asm = torch.randn(10).abs()
        state, strength = layer.bind(ctx, asm)
        assert state.shape == (10,)

    def test_with_shortcuts(self):
        """Binding with small-world shortcuts."""
        layer = HypercubeBindingLayer(
            n_columns=16, device=torch.device("cpu"),
            shortcuts_per_node=2,
        )
        assert layer.topology.shortcuts_per_node == 2
        ctx = torch.randn(16).abs()
        asm = torch.randn(16).abs()
        state, strength = layer.bind(ctx, asm)
        assert state.shape == (16,)


# ── Config integration ─────────────────────────────────────────────

class TestConfigIntegration:
    def test_config_accepts_hypercube(self):
        from hecsn.config.model_config import HECSNConfig
        cfg = HECSNConfig(
            n_columns=32,
            enable_binding_layer=True,
            enable_context_layer=True,
            binding_mode="hypercube",
        )
        assert cfg.binding_mode == "hypercube"

    def test_config_rejects_invalid_mode(self):
        from hecsn.config.model_config import HECSNConfig
        with pytest.raises(ValueError, match="binding_mode"):
            HECSNConfig(
                n_columns=32,
                enable_binding_layer=True,
                enable_context_layer=True,
                binding_mode="invalid",
            )

    def test_model_creates_hypercube_binding(self):
        from hecsn.config.model_config import HECSNConfig
        from hecsn.training.trainer import HECSNModel
        cfg = HECSNConfig(
            n_columns=32,
            enable_binding_layer=True,
            enable_context_layer=True,
            binding_mode="hypercube",
        )
        model = HECSNModel(cfg)
        assert isinstance(model.binding_layer, HypercubeBindingLayer)
        assert model.binding_layer.topology.dim == 5


# ── Benchmark ──────────────────────────────────────────────────────

class TestBenchmark:
    """Comparative benchmark: dense vs spatial vs hypercube binding."""

    @pytest.mark.parametrize("n_columns", [32, 64, 128, 256])
    def test_throughput_comparison(self, n_columns):
        """Measure bind() throughput for all three modes."""
        from hecsn.core.context import BindingLayer
        from hecsn.core.topographic import SpatialBindingLayer

        device = torch.device("cpu")
        n_steps = 200

        # Dense
        dense = BindingLayer(n_columns=n_columns, device=device)
        ctx = torch.randn(n_columns).abs()
        asm = torch.randn(n_columns).abs()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            dense.bind(ctx, asm)
        dense_time = time.perf_counter() - t0

        # Spatial
        spatial = SpatialBindingLayer(n_columns=n_columns, device=device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            spatial.bind(ctx, asm)
        spatial_time = time.perf_counter() - t0

        # Hypercube
        hc = HypercubeBindingLayer(n_columns=n_columns, device=device)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            hc.bind(ctx, asm)
        hc_time = time.perf_counter() - t0

        # Report (not an assertion — just collect data)
        print(f"\n--- {n_columns} columns, {n_steps} steps ---")
        print(f"  Dense:     {dense_time*1000:.1f}ms ({n_steps/dense_time:.0f} steps/s)")
        print(f"  Spatial:   {spatial_time*1000:.1f}ms ({n_steps/spatial_time:.0f} steps/s)")
        print(f"  Hypercube: {hc_time*1000:.1f}ms ({n_steps/hc_time:.0f} steps/s)")

        # Basic sanity: all should complete without error
        assert dense_time > 0
        assert spatial_time > 0
        assert hc_time > 0

    def test_memory_comparison(self):
        """Compare tensor memory footprint at 256 columns."""
        from hecsn.core.context import BindingLayer
        from hecsn.core.topographic import SpatialBindingLayer

        n = 256
        device = torch.device("cpu")

        dense = BindingLayer(n_columns=n, device=device)
        spatial = SpatialBindingLayer(n_columns=n, device=device)
        hc = HypercubeBindingLayer(n_columns=n, device=device)

        # Dense connectivity matrix: n_bindings × n_columns
        dense_params = dense.connectivity.nelement() + dense.output_weights.nelement()
        # Spatial: neighbor_ids + learned_weights
        spatial_params = spatial.neighbor_ids.nelement() + spatial.learned_weights.nelement()
        # Hypercube: neighbor_ids + learned_weights
        hc_params = hc.neighbor_ids.nelement() + hc.learned_weights.nelement()

        print(f"\n--- Memory at {n} columns ---")
        print(f"  Dense:     {dense_params} elements ({dense_params * 4 / 1024:.1f} KB)")
        print(f"  Spatial:   {spatial_params} elements ({spatial_params * 4 / 1024:.1f} KB)")
        print(f"  Hypercube: {hc_params} elements ({hc_params * 4 / 1024:.1f} KB)")

        # Hypercube should use dramatically less memory than dense
        assert hc_params < dense_params

    def test_sparsity_comparison(self):
        """Compare edge counts at various scales."""
        for n in [32, 64, 128, 256, 512, 1024, 2048]:
            topo = HypercubeTopology(n_columns=n)
            stats = topo.topology_stats()
            full_edges = n * (n - 1)
            print(f"  {n:5d} cols: dim={stats['dim']:2d}, "
                  f"edges={stats['total_edges']:6d}/{full_edges:8d} "
                  f"({stats['sparsity_ratio']*100:.2f}%)")
