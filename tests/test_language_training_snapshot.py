from __future__ import annotations

import torch
from torch.nn import functional as F

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_training_snapshot import (
    load_language_training_snapshot,
    save_language_training_snapshot,
    tree_sha256,
    tree_to_cpu,
)


def _model_and_tokenizer() -> tuple[MarulhoLanguageModel, ByteLevelLanguageTokenizer]:
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=8,
            transformer_mlp_ratio=2.0,
            transformer_dropout=0.0,
        )
    ).to(dtype=torch.bfloat16)
    return model, tokenizer


def _optimizer(model: MarulhoLanguageModel) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=1.0e-3)


def _step(model: MarulhoLanguageModel, optimizer: torch.optim.Optimizer) -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens[:, :-1], collect_telemetry=False)["logits"]
    loss = F.cross_entropy(logits.float().flatten(0, 1), tokens[:, 1:].flatten())
    loss.backward()
    optimizer.step()


def test_tree_hash_is_mapping_order_independent_and_tensor_sensitive() -> None:
    first = {"b": [2, torch.tensor([3])], "a": 1}
    second = {"a": 1, "b": [2, torch.tensor([3])]}
    assert tree_sha256(first) == tree_sha256(second)
    second["b"][1][0] = 4
    assert tree_sha256(first) != tree_sha256(second)
    assert tree_to_cpu(first)["b"][1].device.type == "cpu"


def test_training_snapshot_restores_model_optimizer_and_counters(tmp_path) -> None:
    torch.manual_seed(7)
    model, tokenizer = _model_and_tokenizer()
    optimizer = _optimizer(model)
    _step(model, optimizer)
    snapshot = tmp_path / "continuation.pt"
    saved = save_language_training_snapshot(
        snapshot,
        model,
        tokenizer,
        optimizer,
        completed_steps=1,
        schedule_sha256="schedule",
        next_schedule_offset=32,
        training_state={"curve": [{"step": 1}]},
    )
    assert saved["verification"]["passed"]
    restored_model, restored_tokenizer, restored_optimizer, continuation, audit = (
        load_language_training_snapshot(
            snapshot,
            optimizer_builder=_optimizer,
            device=torch.device("cpu"),
            expected_schedule_sha256="schedule",
        )
    )
    assert audit["passed"]
    assert continuation["completed_steps"] == 1
    assert continuation["next_schedule_offset"] == 32
    assert continuation["training_state"]["curve"] == [{"step": 1}]
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert tree_sha256(restored_optimizer.state_dict()) == tree_sha256(
        optimizer.state_dict()
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            restored_model.state_dict().values(),
            model.state_dict().values(),
            strict=True,
        )
    )
