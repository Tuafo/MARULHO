from __future__ import annotations

import pytest
import torch

from marulho.consolidation.memory_store import DualMemoryStore


def _selected_replay_store(entries: int = 32, buckets: int = 8) -> DualMemoryStore:
    store = DualMemoryStore(
        capacity=entries,
        ema_alpha=0.1,
        slow_mean_decay=1.0,
        capture_tag_decay=1.0,
        capture_release=0.5,
        consolidation_rate=1.0,
        prp_synthesis_rate=0.5,
    )
    for index in range(entries):
        store.update(
            torch.tensor([float(index % 7), 1.0], dtype=torch.float32),
            token_count=index,
            importance=1.0 + float(index % 3),
            bucket_id=index % buckets,
            capture_tag=1.0,
        )
        store.slow_local_prp[index] = 1.0
        store.slow_consolidation_level[index] = 0.10
    store.global_prp_pool = 10.0
    for bucket in range(buckets):
        store.bucket_prp_pool[bucket] = 10.0
    store._invalidate_bucket_consolidation_cache()
    store.bucket_consolidation_tensor(buckets, device=torch.device("cpu"))
    return store


def test_bucket_consolidation_tensor_matches_scalar_lookup_and_invalidates() -> None:
    store = DualMemoryStore(capacity=8)
    store.slow_buffer = [torch.zeros(1) for _ in range(3)]
    store.slow_bucket_ids = [1, 3, 1]
    store.slow_importance = [1.0, 1.0, 3.0]
    store.slow_consolidation_level = [0.2, 0.6, 0.8]

    first = store.bucket_consolidation_tensor(5, device=torch.device("cpu"))
    first_generation = store.bucket_consolidation_cache_generation

    assert torch.allclose(first, torch.tensor([0.0, 0.65, 0.0, 0.6, 0.0]))
    assert first[1].item() == pytest.approx(store.bucket_consolidation_level(1))
    assert first[3].item() == pytest.approx(store.bucket_consolidation_level(3))
    report = store.device_report()["last_bucket_consolidation_level_report"]
    assert report["surface"] == "bucket_consolidation_level_cache_lookup.v1"
    assert report["status"] == "cache_hit"
    assert report["full_memory_scan"] is False
    assert report["scan_entry_count"] == 0

    store.slow_consolidation_level[1] = 0.9
    store._invalidate_bucket_consolidation_cache()
    assert store.bucket_consolidation_cache_generation == first_generation + 1
    second = store.bucket_consolidation_tensor(5, device=torch.device("cpu"))

    assert second[3].item() == pytest.approx(0.9)
    assert second.data_ptr() != first.data_ptr()
    assert store.bucket_consolidation_cache_generation == first_generation + 1


def test_bucket_consolidation_tensor_updates_one_cached_bucket_on_append() -> None:
    store = DualMemoryStore(capacity=8)
    store.slow_buffer = [torch.zeros(1)]
    store.slow_bucket_ids = [2]
    store.slow_importance = [1.0]
    store.slow_consolidation_level = [0.8]
    cached = store.bucket_consolidation_tensor(4, device=torch.device("cpu"))
    generation = store.bucket_consolidation_cache_generation

    store.slow_buffer.append(torch.zeros(1))
    store.slow_bucket_ids.append(2)
    store.slow_importance.append(3.0)
    store.slow_consolidation_level.append(0.0)
    store._adjust_bucket_consolidation_cache(
        2,
        importance=3.0,
        consolidation=0.0,
        sign=1.0,
    )

    assert cached[2].item() == pytest.approx(0.2)
    assert store.bucket_consolidation_cache_generation == generation


def test_bucket_consolidation_level_missing_cache_does_not_scan() -> None:
    store = DualMemoryStore(capacity=8)
    store.slow_buffer = [torch.zeros(1) for _ in range(3)]
    store.slow_bucket_ids = [1, 3, 1]
    store.slow_importance = [1.0, 1.0, 3.0]
    store.slow_consolidation_level = [0.2, 0.6, 0.8]
    store._invalidate_bucket_consolidation_cache()

    assert store.bucket_consolidation_level(1) == pytest.approx(0.0)
    report = store.device_report()["last_bucket_consolidation_level_report"]

    assert report["status"] == "cache_missing_no_scan"
    assert report["cache_hit"] is False
    assert report["full_memory_scan"] is False
    assert report["scan_entry_count"] == 0
    assert store.bucket_consolidation_cache_rebuild_count == 0


def test_selected_replay_consolidation_missing_cache_does_not_rebuild_full_archive() -> None:
    store = _selected_replay_store(entries=64, buckets=16)
    store._invalidate_bucket_consolidation_cache()
    before_level = float(store.slow_consolidation_level[3])
    before_rebuilds = int(store.bucket_consolidation_cache_rebuild_count)
    before_scan_entries = int(store.bucket_consolidation_cache_rebuild_scan_entry_count)

    report = store.consolidate_replay(
        [3, 7, -1, 99],
        current_token=100,
        blend=0.5,
        protein_synthesis_level=1.25,
    )
    device_report = store.device_report()

    assert report["surface"] == "bounded_selected_replay_consolidation.v1"
    assert report["status"] == "updated_selected_window"
    assert report["selected_indices"] == [3, 7]
    assert report["selected_valid_index_count"] == 2
    assert report["replayed_count"] == 2
    assert report["consolidated_count"] == 2
    assert report["cache_adjustment_mode"] == "cache_missing_deferred_no_full_rebuild"
    assert report["cache_rebuild_count_delta"] == 0
    assert report["cache_rebuild_scan_entry_count"] == 0
    assert report["full_memory_scan"] is False
    assert report["scan_entry_count"] == 0
    assert report["runs_live_tick"] is False
    assert report["runs_every_token"] is False
    assert report["raw_text_payload_loaded"] is False
    assert report["language_reasoning"] is False
    assert store.bucket_consolidation_cache_rebuild_count == before_rebuilds
    assert store.bucket_consolidation_cache_rebuild_scan_entry_count == before_scan_entries
    assert device_report["bucket_consolidation_cache_entries"] == 0
    assert device_report["last_replay_consolidation_report"] == report
    assert store.slow_consolidation_level[3] > before_level
    assert store.slow_replay_count[3] == 1


def test_selected_replay_consolidation_matches_cached_diagnostic_without_cache_rebuild() -> None:
    source = _selected_replay_store(entries=32, buckets=8)
    bounded = DualMemoryStore(capacity=1)
    diagnostic = DualMemoryStore(capacity=1)
    bounded.restore(source.snapshot())
    diagnostic.restore(source.snapshot())
    bounded._invalidate_bucket_consolidation_cache()
    selected = [2, 10, 18]

    bounded_report = bounded.consolidate_replay(
        selected,
        current_token=120,
        blend=0.4,
        protein_synthesis_level=1.1,
    )
    diagnostic_report = diagnostic.consolidate_replay(
        selected,
        current_token=120,
        blend=0.4,
        protein_synthesis_level=1.1,
    )

    assert bounded_report["cache_adjustment_mode"] == (
        "cache_missing_deferred_no_full_rebuild"
    )
    assert diagnostic_report["cache_adjustment_mode"] == "selected_bucket_delta_update"
    assert bounded_report["cache_rebuild_scan_entry_count"] == 0
    assert bounded_report["cache_rebuild_count_delta"] == 0
    assert diagnostic_report["cache_rebuild_scan_entry_count"] == 0
    for index in selected:
        assert bounded.slow_consolidation_level[index] == pytest.approx(
            diagnostic.slow_consolidation_level[index]
        )
        assert bounded.slow_replay_count[index] == diagnostic.slow_replay_count[index]
        assert bounded.slow_consolidation_events[index] == (
            diagnostic.slow_consolidation_events[index]
        )
        assert bounded.slow_capture_tag[index] == pytest.approx(
            diagnostic.slow_capture_tag[index]
        )
    assert torch.allclose(bounded.fast_ema, diagnostic.fast_ema)

    restored = DualMemoryStore(capacity=1)
    restored.restore(bounded.snapshot())
    assert restored.last_replay_consolidation_report["surface"] == (
        "bounded_selected_replay_consolidation.v1"
    )
    assert restored.last_replay_consolidation_report["selected_indices"] == selected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_bucket_consolidation_tensor_keeps_cached_compute_copy_on_cuda() -> None:
    store = DualMemoryStore(capacity=4)
    store.slow_bucket_ids = [2]
    store.slow_importance = [1.0]
    store.slow_consolidation_level = [0.75]

    first = store.bucket_consolidation_tensor(4, device=torch.device("cuda"))
    second = store.bucket_consolidation_tensor(4, device=torch.device("cuda"))
    report = store.device_report()

    assert first.is_cuda
    assert first.data_ptr() == second.data_ptr()
    assert first[2].item() == pytest.approx(0.75)
    assert any(
        device.startswith("cuda")
        for device in report["bucket_consolidation_cache_devices"]
    )
