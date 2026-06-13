from __future__ import annotations

import pytest
import torch

from marulho.consolidation.memory_store import DualMemoryStore


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
