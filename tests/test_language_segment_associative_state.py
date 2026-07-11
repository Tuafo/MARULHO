from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_segment_associative_state import (
    SEGMENT_ASSOCIATIVE_MODES,
    MarulhoSegmentAssociativeLanguageModel,
    SegmentAssociativeConfig,
    build_segment_associative_model,
    load_segment_associative_checkpoint,
    save_segment_associative_checkpoint,
    segment_associative_checkpoint_payload,
)


def _base(*, vocab_size: int = 96, mode: str = "token_hash"):
    return MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=vocab_size,
            width=32,
            layers=3,
            attention_heads=4,
            context_length=32,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=64,
            routing_heads=2,
            experts_per_head=2,
            mode=mode,
        )
    )


def _segment_config(**overrides) -> SegmentAssociativeConfig:
    values = {
        "segment_length": 4,
        "memory_layer_index": 0,
        "memory_heads": 2,
        "key_width": 4,
        "value_width": 8,
        "mode": "gated_delta",
    }
    values.update(overrides)
    return SegmentAssociativeConfig(**values)


def _model(**overrides) -> MarulhoSegmentAssociativeLanguageModel:
    return build_segment_associative_model(
        _base(),
        _segment_config(**overrides),
    )


def test_segment_modes_attach_with_exact_base_logits() -> None:
    torch.manual_seed(127)
    base = _base().eval()
    input_ids = torch.randint(0, 96, (2, 16))
    expected = base(input_ids, collect_telemetry=False)["logits"].detach()
    model = build_segment_associative_model(base, _segment_config()).eval()
    for mode in SEGMENT_ASSOCIATIVE_MODES:
        model.set_segment_associative_mode(mode)
        actual = model(input_ids, collect_telemetry=False)["logits"].detach()
        assert torch.equal(actual, expected)
    report = model.segment_parameter_report()
    assert report["segment_associative_parameters"] == 1696
    assert report["total_model_parameters"] == sum(
        value.numel() for value in base.parameters()
    ) + 1696


@pytest.mark.parametrize("mode", SEGMENT_ASSOCIATIVE_MODES)
def test_segment_associative_state_is_causal_and_streaming_equivalent(
    mode: str,
) -> None:
    torch.manual_seed(131)
    model = _model(mode=mode).eval()
    with torch.no_grad():
        model.state_block.associative.output.weight.normal_(0.0, 0.02)
    first = torch.randint(0, 96, (2, 16))
    second = first.clone()
    second[:, 9:] = torch.randint(0, 96, second[:, 9:].shape)
    with torch.no_grad():
        first_logits = model(first, collect_telemetry=False)["logits"]
        second_logits = model(second, collect_telemetry=False)["logits"]
        prompt = model(first[:, :3], collect_telemetry=False)
        state = prompt["state"]
        incremental = [prompt["logits"][:, -1]]
        for index in range(3, int(first.shape[1])):
            step = model.forward_step(
                first[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            incremental.append(step["logits"][:, -1])
    torch.testing.assert_close(first_logits[:, :9], second_logits[:, :9])
    torch.testing.assert_close(
        torch.stack(incremental, dim=1),
        first_logits[:, 2:],
        atol=3e-5,
        rtol=2e-5,
    )
    assert state["segment_memory"].shape == (2, 2, 4, 8)
    assert int(state["segment_count"]) == 0


def test_gated_segment_state_trains_every_associative_parameter() -> None:
    torch.manual_seed(137)
    model = _model(mode="gated_delta").train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    input_ids = torch.randint(0, 96, (3, 16))
    target_ids = torch.randint(0, 96, (3, 16))
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids, collect_telemetry=False)["logits"]
        loss = F.cross_entropy(logits.reshape(-1, 96), target_ids.reshape(-1))
        loss.backward()
        optimizer.step()
    report = model.final_segment_gradient_report()
    assert report["mode"] == "gated_delta"
    assert report["all_parameters_received_gradient"] is True
    assert all(
        row["nonzero_gradient_elements"] > 0 for row in report["parameters"]
    )


@pytest.mark.parametrize("mode", SEGMENT_ASSOCIATIVE_MODES)
def test_segment_diagnostic_is_label_safe_and_mode_specific(mode: str) -> None:
    torch.manual_seed(138)
    model = _model(mode=mode).eval()
    with torch.no_grad():
        model.state_block.associative.output.weight.normal_(0.0, 0.02)
    report = model.segment_diagnostic_report(torch.randint(0, 96, (2, 16)))
    state = report["state"]
    assert report["mode"] == mode
    assert state["mode"] == mode
    assert state["write_policy_uses_labels"] is False
    assert state["promotion_metric"] is False
    assert state["memory_state_bytes"] > 0
    assert state["theoretical_active_multiplies_per_token"] >= 0
    if mode in {"delta", "gated_delta"}:
        assert state["memory_update_count"] == 8
        assert state["final_memory_frobenius_norm"] > 0.0
        assert state["final_memory_matrix_rank_mean"] > 0.0
        assert state["state_trajectory_effective_rank"] > 0.0
        assert state["state_perturbation_gain"] >= 0.0
    else:
        assert state["memory_update_count"] == 0
        assert state["final_memory_frobenius_norm"] == 0.0
    if mode == "off":
        assert state["mean_write_gate"] is None
        assert state["theoretical_active_multiplies_per_token"] == 0
    else:
        assert state["mean_write_gate"] is not None


def test_segment_associative_checkpoint_round_trip_is_strict(tmp_path) -> None:
    torch.manual_seed(139)
    tokenizer = ByteLevelLanguageTokenizer()
    base = _base(vocab_size=tokenizer.vocab_size).eval()
    model = build_segment_associative_model(base, _segment_config()).train()
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 12))
    target_ids = torch.randint(0, tokenizer.vocab_size, (2, 12))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(
            model(input_ids, collect_telemetry=False)["logits"].reshape(
                -1, tokenizer.vocab_size
            ),
            target_ids.reshape(-1),
        )
        loss.backward()
        optimizer.step()
    model.eval()
    expected = model(input_ids, collect_telemetry=False)["logits"].detach()
    path = save_segment_associative_checkpoint(
        tmp_path / "v14.pt",
        model,
        tokenizer,
        metadata={"decision": "qualified"},
    )
    restored, restored_tokenizer, metadata = load_segment_associative_checkpoint(
        path
    )
    restored.eval()
    actual = restored(input_ids, collect_telemetry=False)["logits"]
    torch.testing.assert_close(actual, expected)
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata == {"decision": "qualified"}
    assert restored.lm_head.weight.data_ptr() == restored.token_embedding.weight.data_ptr()
    assert restored.state_block.associative._mode_name == "gated_delta"


def test_segment_checkpoint_rejects_non_gated_mode() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model = build_segment_associative_model(
        _base(vocab_size=tokenizer.vocab_size),
        _segment_config(),
    )
    model.set_segment_associative_mode("local")
    with pytest.raises(ValueError, match="gated_delta"):
        segment_associative_checkpoint_payload(model, tokenizer)


@pytest.mark.parametrize(
    ("configuration", "match"),
    [
        (_segment_config(segment_length=1), "segment_length"),
        (_segment_config(memory_layer_index=2), "precede"),
        (_segment_config(memory_heads=0), "memory_heads"),
        (_segment_config(key_width=0), "key/value"),
        (_segment_config(mode="reservoir"), "mode must be"),
    ],
)
def test_segment_associative_rejects_invalid_config(
    configuration: SegmentAssociativeConfig,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        build_segment_associative_model(_base(), configuration)


def test_segment_associative_requires_token_hash_base() -> None:
    with pytest.raises(ValueError, match="token_hash"):
        build_segment_associative_model(
            _base(mode="shared_only"),
            _segment_config(),
        )
