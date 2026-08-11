from __future__ import annotations

import torch

from marulho.training.language_hidden_state_memory import (
    HiddenStateEpisodicMemory,
    HiddenStateMemoryConfig,
    load_hidden_state_memory,
    memory_state_sha256,
    save_hidden_state_memory,
)


def test_hidden_state_memory_fuses_only_active_similar_queries() -> None:
    memory = HiddenStateEpisodicMemory(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([2, 3]),
        config=HiddenStateMemoryConfig(
            top_k=1,
            similarity_threshold=0.8,
            interpolation_weight=0.75,
            temperature=0.1,
        ),
    )
    logits = torch.zeros(2, 5)
    fused, evidence = memory.fuse_logits(
        logits,
        torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )

    assert int(fused[0].argmax().item()) == 2
    assert torch.allclose(fused[1], torch.log_softmax(logits[1], dim=-1))
    assert evidence["active_query_count"] == 1
    assert memory.metrics()["search_is_dense"] is True


def test_hidden_state_memory_exact_reload(tmp_path) -> None:
    memory = HiddenStateEpisodicMemory(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([5, 6]),
        config=HiddenStateMemoryConfig(top_k=2),
        metadata={"parent": "v39"},
    )
    path = save_hidden_state_memory(tmp_path / "memory.pt", memory)
    restored = load_hidden_state_memory(path)

    assert memory_state_sha256(restored) == memory_state_sha256(memory)
    assert restored.metadata == {"parent": "v39"}
