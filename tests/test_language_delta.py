from __future__ import annotations

import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_delta import (
    DeltaLanguageConfig,
    MarulhoDeltaLanguageModel,
    load_delta_language_checkpoint,
    save_delta_language_checkpoint,
)
from marulho.training.language_model import load_language_model_checkpoint
from marulho.training.language_protocol import CausalLanguageModel


def _config(**overrides) -> DeltaLanguageConfig:
    values = {
        "vocab_size": 64,
        "width": 32,
        "layers": 2,
        "memory_heads": 4,
        "memory_head_dim": 8,
        "attention_heads": 4,
        "context_length": 8,
        "mlp_dim": 64,
    }
    values.update(overrides)
    return DeltaLanguageConfig(**values)


def _assert_state_close(
    left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]
) -> None:
    assert set(left) == set(right)
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=1.0e-5, atol=1.0e-6)


@pytest.mark.parametrize("local_attention_every", [0, 2])
def test_delta_full_scan_matches_streaming_steps(local_attention_every: int) -> None:
    torch.manual_seed(7)
    model = MarulhoDeltaLanguageModel(
        _config(local_attention_every=local_attention_every)
    ).eval()
    token_ids = torch.randint(0, 64, (2, 8))

    full = model(token_ids, collect_telemetry=False)
    state = None
    logits = []
    for index in range(token_ids.shape[1]):
        step = model.forward_step(
            token_ids[:, index], state, collect_telemetry=False
        )
        logits.append(step["logits"])
        state = step["state"]
    assert state is not None
    torch.testing.assert_close(
        full["logits"], torch.cat(logits, dim=1), rtol=1.0e-4, atol=1.0e-6
    )
    _assert_state_close(full["state"], state)


@pytest.mark.parametrize("local_attention_every", [0, 2])
def test_delta_is_causal_and_all_parameters_receive_gradients(
    local_attention_every: int,
) -> None:
    torch.manual_seed(11)
    model = MarulhoDeltaLanguageModel(
        _config(local_attention_every=local_attention_every)
    )
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    changed = torch.tensor([[1, 2, 3, 4, 20, 21, 22, 23]])
    first = model(prefix, collect_telemetry=False)["logits"]
    second = model(changed, collect_telemetry=False)["logits"]
    torch.testing.assert_close(first[:, :4], second[:, :4], rtol=0.0, atol=0.0)

    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    loss = model.next_token_loss(
        prefix, targets, collect_telemetry=False
    )["loss"]
    loss.backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    assert missing == []
    assert nonfinite == []


def test_delta_runtime_state_round_trip_is_exact() -> None:
    torch.manual_seed(13)
    model = MarulhoDeltaLanguageModel(_config(local_attention_every=2)).eval()
    first = model(torch.randint(0, 64, (2, 6)), collect_telemetry=False)
    serialized = model.serialize_state(first["state"])
    restored = model.load_state(serialized)
    next_ids = torch.tensor([7, 9])
    expected = model.forward_step(
        next_ids, first["state"], collect_telemetry=False
    )
    actual = model.forward_step(next_ids, restored, collect_telemetry=False)
    torch.testing.assert_close(expected["logits"], actual["logits"])
    _assert_state_close(expected["state"], actual["state"])


def test_delta_matched_arms_are_within_point_one_percent() -> None:
    transformer_parameters = 20_976_128
    pure = MarulhoDeltaLanguageModel(
        DeltaLanguageConfig(vocab_size=8192, context_length=72)
    )
    hybrid = MarulhoDeltaLanguageModel(
        DeltaLanguageConfig(
            vocab_size=8192,
            context_length=72,
            local_attention_every=4,
        )
    )
    pure_parameters = sum(parameter.numel() for parameter in pure.parameters())
    hybrid_parameters = sum(parameter.numel() for parameter in hybrid.parameters())
    assert pure_parameters == 20_978_176
    assert hybrid_parameters == 20_977_664
    assert abs(pure_parameters - transformer_parameters) / transformer_parameters < 0.001
    assert abs(hybrid_parameters - transformer_parameters) / transformer_parameters < 0.001


def test_delta_implements_shared_language_protocol_and_batched_generation() -> None:
    model = MarulhoDeltaLanguageModel(_config())
    assert isinstance(model, CausalLanguageModel)
    result = model.generate(
        torch.tensor([[1, 2, 3], [4, 5, 6]]),
        max_new_tokens=3,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
    )
    assert result["generated_ids"].shape == (2, 6)
    assert result["new_token_count"] == 3
    assert result["external_llm_used"] is False


def test_delta_checkpoint_preserves_weights_tokenizer_metadata_and_state(
    tmp_path,
) -> None:
    torch.manual_seed(29)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDeltaLanguageModel(
        _config(vocab_size=tokenizer.vocab_size, local_attention_every=2)
    ).eval()
    first = model(
        torch.tensor([[tokenizer.bos_id, 7, 8, 9]]), collect_telemetry=False
    )
    path = save_delta_language_checkpoint(
        tmp_path / "delta.pt",
        model,
        tokenizer,
        metadata={"cumulative_update_tokens": 1234},
        runtime_state=first["state"],
    )
    with pytest.raises(ValueError, match="legacy language checkpoint"):
        load_language_model_checkpoint(path)

    restored, restored_tokenizer, metadata, restored_state = (
        load_delta_language_checkpoint(path)
    )
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata == {"cumulative_update_tokens": 1234}
    assert restored_state is not None
    assert restored.lm_head.weight is restored.token_embedding.weight
    expected = model.forward_step(
        torch.tensor([10]), first["state"], collect_telemetry=False
    )
    actual = restored.forward_step(
        torch.tensor([10]), restored_state, collect_telemetry=False
    )
    torch.testing.assert_close(expected["logits"], actual["logits"])
    _assert_state_close(expected["state"], actual["state"])
