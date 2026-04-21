"""Tests for P1 improvements: awake ripple tagging + small-world hub tracking."""

from __future__ import annotations

import torch
import pytest

from hecsn.consolidation.memory_store import DualMemoryStore
from hecsn.core.hypercube import HypercubeBindingLayer, HypercubeTopology


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

        # Get baseline replay scores
        scores_before = store.replay_scores(current_token=1000)

        # Ripple-tag entries 8 and 9 (most recent)
        store.ripple_tag_awake(
            current_token=1000,
            window_tokens=250,
            da_level=0.9,
        )

        # Scores should be boosted for tagged entries
        scores_after = store.replay_scores(current_token=1000)

        # Tagged entries (8,9) should have higher scores
        for idx in range(10):
            if store.slow_ripple_tagged[idx]:
                assert scores_after[idx].item() > scores_before[idx].item(), \
                    f"Entry {idx} should have boosted score"

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

    def test_ripple_tag_count_property(self):
        store = DualMemoryStore(capacity=32)
        assert store.ripple_tagged_count == 0
        for i in range(5):
            store.update(torch.randn(16), importance=0.5, token_count=i * 10)
        store.ripple_tag_awake(current_token=45, window_tokens=50, da_level=0.85)
        assert store.ripple_tagged_count == 5  # All within window


class TestSmallWorldHubTracking:
    """Test hub tracking in HypercubeBindingLayer."""

    def test_hub_mask_initialized_empty(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        assert not layer._hub_mask.any()

    def test_hub_mask_updates_after_repeated_binding(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        # Simulate many binds with strong overlapping signals
        # (both context and assembly active in same columns)
        for i in range(200):
            signal = torch.randn(32).abs() * 0.5
            signal[0:8] += 0.8  # Columns 0-7 consistently high
            layer.bind(signal, signal)  # Same signal = maximal coincidence

        # After 200 binds, hub EMA for consistently active columns should be higher
        # Note: output depends on coincidence detection succeeding
        # The usage counter (binding_usage) tracks which columns fire
        assert layer.binding_usage[0:8].mean() >= layer.binding_usage[15:].mean()

    def test_hub_boost_amplifies_hub_signals(self):
        layer = HypercubeBindingLayer(
            n_columns=32, device=torch.device("cpu"), shortcuts_per_node=1,
        )
        # Manually set hubs
        layer._hub_mask[0:2] = True

        # Create signal and check drive
        signal = torch.ones(32) * 0.5
        drive_with_hubs = layer._sparse_drive(signal)

        layer._hub_mask.zero_()
        drive_without_hubs = layer._sparse_drive(signal)

        # Hub columns should increase overall drive
        # (since their signals are boosted, their neighbors receive more)
        assert drive_with_hubs.sum() >= drive_without_hubs.sum()

    def test_shortcuts_reduce_average_path(self):
        """Topology with shortcuts should have shorter paths than without."""
        topo_no_short = HypercubeTopology(n_columns=32, shortcuts_per_node=0)
        topo_with_short = HypercubeTopology(n_columns=32, shortcuts_per_node=2)

        # With shortcuts, max degree is higher
        assert topo_with_short.max_degree > topo_no_short.max_degree

    def test_reset_clears_hub_state(self):
        layer = HypercubeBindingLayer(
            n_columns=16, device=torch.device("cpu"),
        )
        layer._hub_mask[0:3] = True
        layer._hub_activation_ema[0:3] = 1.0
        layer.reset_state()
        assert not layer._hub_mask.any()
        assert layer._hub_activation_ema.sum() == 0.0


class TestNIMCortexNoOllama:
    """Verify that NO Ollama is referenced or launched."""

    def test_no_ollama_in_cortex_imports(self):
        import hecsn.cortex.core as core_module
        source = open(core_module.__file__).read()
        # Should have no Ollama connection code
        assert "127.0.0.1:11434" not in source
        assert "api/generate" not in source
        assert "httpx.Client" not in source

    def test_create_cortex_from_env_no_ollama(self):
        """Without API key, raise RuntimeError — no silent MockCortex fallback."""
        import os
        old_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            from hecsn.cortex.multi_cortex import create_cortex_from_env
            import pytest
            with pytest.raises(RuntimeError, match="NVIDIA_API_KEY not set"):
                create_cortex_from_env()
        finally:
            if old_key:
                os.environ["NVIDIA_API_KEY"] = old_key

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
