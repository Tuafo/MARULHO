"""Tests for P1 improvements: awake ripple tagging + hub-aware hypercube structure."""

from __future__ import annotations

import torch
import pytest

from marulho.consolidation.memory_store import DualMemoryStore
from marulho.core.hypercube import HypercubeBindingLayer, HypercubeTopology


def _module_missing(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is None
    except ModuleNotFoundError:
        return True


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
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 100,
                bucket_id=i,
            )

        # Ripple tag recent ones (DA > threshold)
        tagged = store.ripple_tag_awake(
            current_token=950,
            window_tokens=200,  # Last 200 tokens = entries at 800, 900
            da_level=0.9,
            da_threshold=0.7,
            awake_bucket_ids=[8, 9],
        )
        assert tagged >= 1
        assert store.ripple_tagged_count >= 1

    def test_ripple_tag_not_triggered_below_threshold(self):
        store = DualMemoryStore(capacity=32)
        for i in range(5):
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 100,
                bucket_id=i,
            )

        tagged = store.ripple_tag_awake(
            current_token=450,
            window_tokens=500,
            da_level=0.5,  # Below threshold
            da_threshold=0.7,
            awake_bucket_ids=[0, 1, 2, 3, 4],
        )
        assert tagged == 0
        assert store.ripple_tagged_count == 0

    def test_ripple_tagged_entries_get_higher_replay_scores(self):
        store = DualMemoryStore(capacity=32)
        # Insert identical entries
        for i in range(10):
            store.update(
                torch.randn(16), importance=0.5, token_count=i * 100,
                tag_strength=0.3, bucket_id=i,
            )

        candidate_indices = list(range(len(store.slow_buffer)))
        scores_before = store.replay_scores_for_indices(
            candidate_indices,
            current_token=1000,
        )
        store.ripple_tag_awake(
            current_token=1000,
            window_tokens=250,
            da_level=0.9,
            awake_bucket_ids=[8, 9],
        )
        scores_after = store.replay_scores_for_indices(
            candidate_indices,
            current_token=1000,
        )

        for idx in range(10):
            if store.slow_ripple_strength[idx] > 0.0:
                assert scores_after[idx] > scores_before[idx], \
                    f"Entry {idx} should have boosted score"

    def test_ripple_strength_tracks_da_and_recency(self):
        store = DualMemoryStore(capacity=32)
        for i in range(6):
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 100,
                tag_strength=0.2,
                bucket_id=i,
            )

        store.ripple_tag_awake(
            current_token=550,
            window_tokens=400,
            da_level=0.95,
            da_threshold=0.7,
            awake_bucket_ids=[2, 3, 4, 5],
        )

        recent_strength = store.slow_ripple_strength[5]
        older_strength = store.slow_ripple_strength[2]
        assert recent_strength >= older_strength
        assert recent_strength >= 0.5
        assert older_strength >= 0.5

    def test_ripple_priority_multiplier_reaches_above_legacy_three_x(self):
        store = DualMemoryStore(capacity=32)
        for i in range(6):
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 100,
                tag_strength=0.2,
                bucket_id=i,
            )

        store.ripple_tag_awake(
            current_token=550,
            window_tokens=500,
            da_level=0.95,
            da_threshold=0.7,
            awake_bucket_ids=[1, 2, 3, 4, 5],
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
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 100,
                bucket_id=i,
            )
        store.ripple_tag_awake(
            current_token=450,
            window_tokens=200,
            da_level=0.9,
            awake_bucket_ids=[3, 4],
        )
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
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 10,
                bucket_id=i,
            )
        store.ripple_tag_awake(
            current_token=45,
            window_tokens=50,
            da_level=0.85,
            awake_bucket_ids=[0, 1, 2, 3, 4],
        )
        assert store.ripple_tagged_count == 5  # All within window

    def test_vectorized_ripple_tag_matches_retained_scalar_formula(self):
        store = DualMemoryStore(capacity=32)
        for i in range(12):
            store.update(
                torch.randn(16),
                importance=0.5,
                token_count=i * 17,
                tag_strength=0.1 + i * 0.01,
                bucket_id=i,
            )
        retained = DualMemoryStore(capacity=32)
        retained.restore(store.snapshot())

        current_token = 220
        window_tokens = 95
        da_level = 0.91
        da_threshold = 0.7
        floor_token = max(0, current_token - window_tokens)
        window_span = max(1.0, float(window_tokens))
        da_scale = max(
            0.0,
            min(
                1.0,
                (da_level - da_threshold) / max(1e-6, 1.0 - da_threshold),
            ),
        )
        retained._advance_state(current_token)
        retained_tagged = 0
        for idx, timestamp in enumerate(retained.slow_entry_timestamps):
            entry_token = int(timestamp)
            if entry_token < floor_token:
                continue
            recency_scale = max(
                0.0,
                min(
                    1.0,
                    (float(entry_token) - float(floor_token)) / window_span,
                ),
            )
            strength = retained._clip_ripple_strength(
                0.5 + 0.30 * da_scale + 0.20 * recency_scale
            )
            was_untagged = retained.slow_ripple_strength[idx] <= 0.0
            retained.slow_ripple_strength[idx] = max(
                retained.slow_ripple_strength[idx],
                strength,
            )
            retained.slow_capture_tag[idx] = min(
                1.0,
                retained.slow_capture_tag[idx] + 0.10 + 0.25 * strength,
            )
            retained_tagged += int(was_untagged)

        tagged = store.ripple_tag_awake(
            current_token=current_token,
            window_tokens=window_tokens,
            da_level=da_level,
            da_threshold=da_threshold,
            allow_global_diagnostic=True,
        )

        assert tagged == retained_tagged
        assert list(store.slow_ripple_strength) == pytest.approx(
            list(retained.slow_ripple_strength),
            abs=1e-12,
        )
        assert list(store.slow_capture_tag) == pytest.approx(
            list(retained.slow_capture_tag),
            abs=1e-12,
        )


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
        layer.refresh_hub_topology(reason="test_ranked_hub_profile")

        stats = layer.hub_stats()
        assert stats["hub_count"] == 2  # ceil(5% of 32) = 2
        assert layer._hub_mask[0]
        assert layer._hub_mask[1]
        assert layer._hub_connection_multiplier[0].item() == pytest.approx(2.0)
        assert 1.5 <= layer._hub_connection_multiplier[1].item() <= 2.0
        assert layer._hub_connection_multiplier[2].item() == pytest.approx(1.0)
        assert _outgoing_count(layer, 0) == base_outgoing_0 + int(layer._hub_extra_connections[0].item())
        assert _outgoing_count(layer, 1) == base_outgoing_1 + int(layer._hub_extra_connections[1].item())

    def test_repeated_binding_collects_hub_evidence_until_explicit_refresh(self):
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
        assert hub_stats["hub_count"] == 0
        assert hub_stats["evidence_update_count"] == 200
        assert hub_stats["topology_refresh_count"] == 0

        layer.refresh_hub_topology(reason="test_repeated_binding_evidence")
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
        layer.refresh_hub_topology(reason="test_hub_outreach")
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
        layer.refresh_hub_topology(reason="test_reset_hub_profile")
        layer.reset_state()
        assert not layer._hub_mask.any()
        assert layer._hub_activation_ema.sum() == 0.0
        assert layer._hub_strength.sum() == 0.0
        assert torch.allclose(layer._hub_connection_multiplier, torch.ones(16))
        assert torch.equal(layer._hub_extra_connections, torch.zeros(16, dtype=torch.long))
        assert torch.equal(layer.neighbor_ids, layer._base_neighbor_ids)
        assert torch.equal(layer.degree, layer._base_degree)


class TestRetiredCortexExternalLLMPath:
    """Verify that external LLM Cortex paths are deleted from the public surface."""

    def test_cortex_package_is_deleted(self):
        assert _module_missing("marulho.cortex")

    def test_retired_llm_core_module_is_deleted(self):
        assert _module_missing("marulho.cortex.core")

    def test_retired_llm_prompts_module_is_deleted(self):
        assert _module_missing("marulho.cortex.prompts")

    def test_external_llm_adapter_module_is_deleted(self):
        assert _module_missing("marulho.cortex.multi_cortex")

    def test_mock_cortex_is_not_public_cognition_contract(self):
        """Mock Cortex must not be used as a replacement language/thought path."""
        assert _module_missing("marulho.cortex")

    def test_thought_loop_body_is_deleted(self):
        """The retired loop must not remain importable as hidden Cortex code."""
        assert _module_missing("marulho.cortex.thought_loop")
