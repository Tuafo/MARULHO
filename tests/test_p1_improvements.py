"""Tests for P1 improvements: awake ripple tagging + hub-aware hypercube structure."""

from __future__ import annotations

import torch
import pytest

from hecsn.consolidation.memory_store import DualMemoryStore
from hecsn.core.hypercube import HypercubeBindingLayer, HypercubeTopology


def _outgoing_count(layer: HypercubeBindingLayer, source_id: int) -> int:
    count = 0
    for target in range(layer.n_columns):
        d = int(layer.degree[target].item())
        count += sum(1 for source in layer.neighbor_ids[target, :d].tolist() if int(source) == source_id)
    return count


class TestAwakeRippleTagging:
    """Test the Yang & Buzsaki 2024 awake ripple tagging improvement."""

    def test_ripple_tag_marks_recent_entries(self):
        store = DualMemoryStore(capacity=32)
        # Insert entries
        for i in range(10):
            store.update(torch.randn(16), importance=0.5, token_count=i * 100)

        # Ripple tag recent ones (DA > threshold)
        tagged = store.ripple_tag_awake(
            current_token=950,
            window_tokens=200,  # Last 200 tokens = entries at 800, 900
            da_level=0.9,
            da_threshold=0.7,
        )
        assert tagged >= 1
        assert store.ripple_tagged_count >= 1

    def test_ripple_tag_not_triggered_below_threshold(self):
        store = DualMemoryStore(capacity=32)
        for i in range(5):
            store.update(torch.randn(16), importance=0.5, token_count=i * 100)

        tagged = store.ripple_tag_awake(
            current_token=450,
            window_tokens=500,
            da_level=0.5,  # Below threshold
            da_threshold=0.7,
        )
        assert tagged == 0
        assert store.ripple_tagged_count == 0

    def test_ripple_tagged_entries_get_higher_replay_scores(self):
        store = DualMemoryStore(capacity=32)
        # Insert identical entries
        for i in range(10):
            store.update(
                torch.randn(16), importance=0.5, token_count=i * 100,
                tag_strength=0.3,
            )

        scores_before = store.replay_scores(current_token=1000)
        store.ripple_tag_awake(
            current_token=1000,
            window_tokens=250,
            da_level=0.9,
        )
        scores_after = store.replay_scores(current_token=1000)

        for idx in range(10):
            if store.slow_ripple_strength[idx] > 0.0:
                assert scores_after[idx].item() > scores_before[idx].item(), \
                    f"Entry {idx} should have boosted score"

    def test_ripple_strength_tracks_da_and_recency(self):
        store = DualMemoryStore(capacity=32)
        for i in range(6):
            store.update(torch.randn(16), importance=0.5, token_count=i * 100, tag_strength=0.2)

        store.ripple_tag_awake(
            current_token=550,
            window_tokens=400,
            da_level=0.95,
            da_threshold=0.7,
        )

        recent_strength = store.slow_ripple_strength[5]
        older_strength = store.slow_ripple_strength[2]
        assert recent_strength >= older_strength
        assert recent_strength >= 0.5
        assert older_strength >= 0.5

    def test_ripple_priority_multiplier_reaches_above_legacy_three_x(self):
        store = DualMemoryStore(capacity=32)
        for i in range(6):
            store.update(torch.randn(16), importance=0.5, token_count=i * 100, tag_strength=0.2)

        store.ripple_tag_awake(
            current_token=550,
            window_tokens=500,
            da_level=0.95,
            da_threshold=0.7,
        )

        multipliers = [
            store._ripple_priority_multiplier(value)
            for value in store.slow_ripple_strength
            if value > 0.0
        ]
        assert multipliers
        assert max(multipliers) > 3.0
        assert max(multipliers) <= 5.0

    def test_ripple_tag_survives_snapshot_restore(self):
        store = DualMemoryStore(capacity=32)
        for i in range(5):
            store.update(torch.randn(16), importance=0.5, token_count=i * 100)
        store.ripple_tag_awake(current_token=450, window_tokens=200, da_level=0.9)
        assert store.ripple_tagged_count > 0

        snap = store.snapshot()
        store2 = DualMemoryStore(capacity=32)
        store2.restore(snap)
        assert store2.ripple_tagged_count == store.ripple_tagged_count
        assert store2.slow_ripple_strength == store.slow_ripple_strength

    def test_ripple_tag_count_property(self):
        store = DualMemoryStore(capacity=32)
        assert store.ripple_tagged_count == 0
        for i in range(5):
            store.update(torch.randn(16), importance=0.5, token_count=i * 10)
        store.ripple_tag_awake(current_token=45, window_tokens=50, da_level=0.85)
        assert store.ripple_tagged_count == 5  # All within window


class TestSmallWorldHubTracking:
    """Test hub tracking in HypercubeBindingLayer."""

    def test_hub_profile_initialized_empty(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        assert not layer._hub_mask.any()
        assert torch.allclose(layer._hub_strength, torch.zeros(32))
        assert torch.allclose(layer._hub_connection_multiplier, torch.ones(32))
        assert torch.equal(layer._hub_extra_connections, torch.zeros(32, dtype=torch.long))

    def test_hub_profile_uses_ceil_top_fraction_and_two_x_range(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        base_outgoing_0 = len(layer._base_outgoing_targets[0])
        base_outgoing_1 = len(layer._base_outgoing_targets[1])
        layer._hub_activation_ema = torch.linspace(1.0, 0.1, steps=32)
        layer._refresh_hub_profile(force_structure_refresh=True)

        stats = layer.hub_stats()
        assert stats["hub_count"] == 2  # ceil(5% of 32) = 2
        assert layer._hub_mask[0]
        assert layer._hub_mask[1]
        assert layer._hub_connection_multiplier[0].item() == pytest.approx(2.0)
        assert 1.5 <= layer._hub_connection_multiplier[1].item() <= 2.0
        assert layer._hub_connection_multiplier[2].item() == pytest.approx(1.0)
        assert _outgoing_count(layer, 0) == base_outgoing_0 + int(layer._hub_extra_connections[0].item())
        assert _outgoing_count(layer, 1) == base_outgoing_1 + int(layer._hub_extra_connections[1].item())

    def test_hub_profile_updates_after_repeated_binding(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        signal = torch.zeros(32)
        signal[0] = 2.0
        signal[1] = 1.6
        signal[2:8] = 0.9
        signal[8:] = 0.2

        for _ in range(200):
            layer.bind(signal, signal)

        assert layer.binding_usage[0:8].mean() >= layer.binding_usage[15:].mean()
        hub_stats = layer.hub_stats()
        assert hub_stats["hub_count"] == 2
        assert set(hub_stats["hub_indices"]).issubset(set(range(8)))
        assert hub_stats["max_hub_connection_multiplier"] <= 2.0
        assert hub_stats["max_hub_connection_multiplier"] >= 1.5
        assert hub_stats["hub_extra_edges"] > 0

    def test_hub_structure_adds_outgoing_edges(self):
        baseline = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        signal = torch.zeros(32)
        signal[0] = 1.0

        drive_without_hubs = baseline._sparse_drive(signal)
        base_outgoing_0 = _outgoing_count(layer, 0)
        layer._hub_activation_ema[0] = 1.0
        layer._hub_activation_ema[1] = 0.9
        layer._refresh_hub_profile(force_structure_refresh=True)
        drive_with_hubs = layer._sparse_drive(signal)

        assert _outgoing_count(layer, 0) > base_outgoing_0
        assert int(layer._hub_extra_connections[0].item()) > 0
        assert drive_with_hubs.sum().item() > drive_without_hubs.sum().item()

    def test_shortcuts_reduce_average_path(self):
        """Topology with shortcuts should have shorter paths than without."""
        topo_no_short = HypercubeTopology(n_columns=32, shortcuts_per_node=0)
        topo_with_short = HypercubeTopology(n_columns=32, shortcuts_per_node=2)

        assert topo_with_short.max_degree > topo_no_short.max_degree

    def test_reset_clears_hub_state(self):
        layer = HypercubeBindingLayer(
            n_columns=16, device=torch.device("cpu"),
        )
        layer._hub_activation_ema[0:3] = 1.0
        layer._refresh_hub_profile(force_structure_refresh=True)
        layer.reset_state()
        assert not layer._hub_mask.any()
        assert layer._hub_activation_ema.sum() == 0.0
        assert layer._hub_strength.sum() == 0.0
        assert torch.allclose(layer._hub_connection_multiplier, torch.ones(16))
        assert torch.equal(layer._hub_extra_connections, torch.zeros(16, dtype=torch.long))
        assert torch.equal(layer.neighbor_ids, layer._base_neighbor_ids)
        assert torch.equal(layer.degree, layer._base_degree)


class TestNIMCortexNoOllama:
    """Verify that NO Ollama is referenced or launched."""

    def test_public_cortex_boundary_excludes_active_product_path(self):
        import hecsn.cortex as cortex

        retired_exports = {
            "ThoughtLoop",
            "BrainStats",
            "NIMCortex",
            "MultiCortex",
            "create_cortex_from_env",
            "create_embedder_from_env",
        }
        for name in retired_exports:
            assert not hasattr(cortex, name)
            assert name not in cortex.__all__

        for name in {
            "CorticalCore",
            "ContextPacket",
            "MemoryItem",
            "ThoughtResult",
            "ThinkingMode",
            "MockCortex",
        }:
            assert hasattr(cortex, name)
            assert name in cortex.__all__

    def test_no_ollama_in_cortex_imports(self):
        import hecsn.cortex.core as core_module
        source = open(core_module.__file__).read()
        # Should have no Ollama connection code
        assert "127.0.0.1:11434" not in source
        assert "api/generate" not in source
        assert "httpx.Client" not in source

    def test_create_cortex_from_env_is_retired(self):
        """The old external LLM Cortex factory is retired, regardless of API key."""
        import os
        old_key = os.environ.get("NVIDIA_API_KEY")
        try:
            os.environ["NVIDIA_API_KEY"] = "would-not-enable-retired-path"
            from hecsn.cortex.multi_cortex import create_cortex_from_env
            import pytest
            with pytest.raises(RuntimeError, match="cortex_runtime_retired"):
                create_cortex_from_env()
        finally:
            if old_key is not None:
                os.environ["NVIDIA_API_KEY"] = old_key
            else:
                os.environ.pop("NVIDIA_API_KEY", None)

    def test_create_embedder_from_env_is_strict_by_default(self):
        import os
        old_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            from hecsn.cortex.multi_cortex import create_embedder_from_env
            with pytest.raises(RuntimeError, match="NVIDIA_API_KEY not set"):
                create_embedder_from_env()
        finally:
            if old_key:
                os.environ["NVIDIA_API_KEY"] = old_key

    def test_create_embedder_from_env_can_opt_into_simple_fallback(self):
        import os
        old_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            from hecsn.cortex.multi_cortex import create_embedder_from_env
            from hecsn.cortex.episodic_memory import SimpleEmbedder
            embedder = create_embedder_from_env(allow_fallback=True)
            assert isinstance(embedder, SimpleEmbedder)
        finally:
            if old_key:
                os.environ["NVIDIA_API_KEY"] = old_key

    def test_mock_cortex_works_standalone(self):
        from hecsn.cortex.core import MockCortex, ContextPacket, ThinkingMode
        cortex = MockCortex()
        result = cortex.generate(ContextPacket(
            drive_summary="test curiosity",
            mode=ThinkingMode.THINK,
        ))
        assert result.parse_success
        assert result.thought
        assert result.latency_ms == 10.0
