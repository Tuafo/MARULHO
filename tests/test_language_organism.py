from __future__ import annotations

from pathlib import Path

import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    load_language_model_checkpoint,
)
from marulho.training.language_organism import (
    DistributedLanguageConfig,
    MarulhoDistributedLanguageModel,
    load_distributed_language_checkpoint,
    save_distributed_language_checkpoint,
)
from marulho.training.language_protocol import CausalLanguageModel


def _tiny_config(vocab_size: int, **overrides: object) -> DistributedLanguageConfig:
    values: dict[str, object] = {
        "vocab_size": vocab_size,
        "width": 32,
        "layers": 2,
        "attention_heads": 4,
        "context_length": 8,
        "unit_groups": 4,
        "workspace_slots": 2,
        "episodic_slots": 4,
        "state_update_interval": 4,
        "mlp_dim": 64,
        "counterfactual_rate": 0.0,
    }
    values.update(overrides)
    return DistributedLanguageConfig(**values)


def test_distributed_candidate_matches_transformer_parameter_budget() -> None:
    baseline = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=8192,
            embedding_dim=512,
            state_dim=512,
            state_layers=4,
            attention_heads=8,
            transformer_context_length=72,
            transformer_mlp_ratio=4.0,
        )
    )
    candidate = MarulhoDistributedLanguageModel(
        DistributedLanguageConfig(vocab_size=8192)
    )
    baseline_count = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    assert baseline_count == 20_976_128
    assert abs(candidate_count - baseline_count) / baseline_count < 0.0003


def test_distributed_candidate_is_causal_and_protocol_complete() -> None:
    torch.manual_seed(3)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(_tiny_config(tokenizer.vocab_size)).eval()
    assert isinstance(model, CausalLanguageModel)
    first = torch.tensor([[1, 8, 9, 10, 11]], dtype=torch.long)
    second = first.clone()
    second[:, -1] = 12
    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    torch.testing.assert_close(first_logits[:, :-1], second_logits[:, :-1])


def test_distributed_scan_matches_incremental_state() -> None:
    torch.manual_seed(5)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(_tiny_config(tokenizer.vocab_size)).eval()
    token_ids = torch.tensor([[1, 4, 8, 15, 16, 23]], dtype=torch.long)
    scanned = model(token_ids, collect_telemetry=False)
    state = None
    rows = []
    for token_index in range(int(token_ids.shape[1])):
        step = model.forward_step(
            token_ids[:, token_index],
            state,
            collect_telemetry=False,
        )
        state = step["state"]
        rows.append(step["logits"])
    stepped_logits = torch.cat(rows, dim=1)
    torch.testing.assert_close(
        scanned["logits"], stepped_logits, atol=2.0e-6, rtol=2.0e-5
    )
    assert state is not None
    assert int(state["position"].item()) == int(token_ids.shape[1])
    torch.testing.assert_close(
        scanned["state"]["layer_1_units"], state["layer_1_units"]
    )
    torch.testing.assert_close(
        scanned["state"]["layer_1_episode_usage"],
        state["layer_1_episode_usage"],
    )


def test_distributed_loss_trains_all_parameters_and_counterfactual_credit() -> None:
    torch.manual_seed(7)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(
        _tiny_config(
            tokenizer.vocab_size,
            counterfactual_rate=1.0,
            episode_counterfactual_fraction=0.5,
            utility_loss_weight=0.1,
        )
    ).train()
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    target_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    result = model.next_token_loss(input_ids, target_ids)
    assert torch.isfinite(result["loss"])
    assert result["loss_evidence"]["counterfactual"]["ran"] is True
    assert result["loss_evidence"]["counterfactual"]["horizon"] >= 1
    result["loss"].backward()
    missing = [
        name for name, parameter in model.named_parameters() if parameter.grad is None
    ]
    assert missing == []
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def test_auxiliary_fast_path_preserves_loss_state_and_gradients() -> None:
    torch.manual_seed(9)
    tokenizer = ByteLevelLanguageTokenizer()
    config = _tiny_config(tokenizer.vocab_size, counterfactual_rate=0.0)
    fast = MarulhoDistributedLanguageModel(config).train()
    instrumented = MarulhoDistributedLanguageModel(config).train()
    instrumented.load_state_dict(fast.state_dict(), strict=True)
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    target_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    fast_result = fast.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
        return_evidence=False,
    )
    instrumented_result = instrumented.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=True,
        return_evidence=True,
    )
    torch.testing.assert_close(fast_result["loss"], instrumented_result["loss"])
    for key, value in fast_result["state"].items():
        torch.testing.assert_close(value, instrumented_result["state"][key])
    fast_result["loss"].backward()
    instrumented_result["loss"].backward()
    for (fast_name, fast_parameter), (full_name, full_parameter) in zip(
        fast.named_parameters(), instrumented.named_parameters(), strict=True
    ):
        assert fast_name == full_name
        assert fast_parameter.grad is not None
        assert full_parameter.grad is not None
        torch.testing.assert_close(fast_parameter.grad, full_parameter.grad)


def test_counterfactual_metadata_survives_evidence_suppression() -> None:
    torch.manual_seed(10)
    tokenizer = ByteLevelLanguageTokenizer()
    config = _tiny_config(tokenizer.vocab_size, counterfactual_rate=1.0)
    model = MarulhoDistributedLanguageModel(config).train()
    instrumented = MarulhoDistributedLanguageModel(config).train()
    instrumented.load_state_dict(model.state_dict(), strict=True)
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    target_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    rng_state = torch.get_rng_state()
    result = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
        return_evidence=False,
    )
    torch.set_rng_state(rng_state)
    instrumented_result = instrumented.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=True,
        return_evidence=True,
    )
    assert result["loss_evidence"] == {}
    assert result["training_aux"]["counterfactual"]["ran"] is True
    assert result["training_aux"]["counterfactual"]["kind"] in {"unit", "episode"}
    torch.testing.assert_close(result["loss"], instrumented_result["loss"])
    for key, value in result["state"].items():
        torch.testing.assert_close(value, instrumented_result["state"][key])
    result["loss"].backward()
    instrumented_result["loss"].backward()
    for (name, parameter), (instrumented_name, instrumented_parameter) in zip(
        model.named_parameters(), instrumented.named_parameters(), strict=True
    ):
        assert name == instrumented_name
        assert parameter.grad is not None
        assert instrumented_parameter.grad is not None
        torch.testing.assert_close(parameter.grad, instrumented_parameter.grad)


def test_counterfactual_probe_can_be_scheduled_explicitly() -> None:
    torch.manual_seed(12)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(
        _tiny_config(tokenizer.vocab_size, counterfactual_rate=1.0)
    ).train()
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    target_ids = torch.randint(0, tokenizer.vocab_size, (2, 7))
    ordinary = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
        return_evidence=False,
        counterfactual_probe=False,
    )
    probe = model.next_token_loss(
        input_ids,
        target_ids,
        collect_telemetry=False,
        return_evidence=False,
        counterfactual_probe=True,
    )
    assert ordinary["training_aux"]["counterfactual"]["ran"] is False
    assert probe["training_aux"]["counterfactual"]["ran"] is True


def test_distributed_generation_is_owned_and_stateful() -> None:
    torch.manual_seed(11)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(_tiny_config(tokenizer.vocab_size)).eval()
    prompt = torch.tensor(
        tokenizer.encode("A key moved", add_bos=True), dtype=torch.long
    )
    result = model.generate(prompt, max_new_tokens=4, temperature=0.0)
    assert result["owned_by_marulho"] is True
    assert result["external_llm_used"] is False
    assert result["generated_token_count"] == 4
    assert result["generation_decode"]["full_model_vocab_logits_materialized"] is True
    assert int(result["state"]["position"].item()) == int(prompt.numel()) + 4
    assert float(result["state"]["layer_0_episode_usage"].sum()) > 0.0


def test_distributed_checkpoint_roundtrip_is_strict_and_preserves_runtime(
    tmp_path: Path,
) -> None:
    torch.manual_seed(13)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoDistributedLanguageModel(_tiny_config(tokenizer.vocab_size)).eval()
    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    before = model(token_ids, collect_telemetry=False)
    checkpoint = save_distributed_language_checkpoint(
        tmp_path / "candidate.pt",
        model,
        tokenizer,
        metadata={"optimizer_steps": 9},
        runtime_state=before["state"],
    )
    restored, restored_tokenizer, metadata, restored_state = (
        load_distributed_language_checkpoint(checkpoint)
    )
    after = restored(token_ids, collect_telemetry=False)
    torch.testing.assert_close(before["logits"], after["logits"])
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata["optimizer_steps"] == 9
    assert metadata["checkpoint_size_bytes"] == checkpoint.stat().st_size
    assert restored_state is not None
    torch.testing.assert_close(
        restored_state["layer_0_episode_values"],
        before["state"]["layer_0_episode_values"],
    )
    with pytest.raises(ValueError, match="Rejected legacy language checkpoint"):
        load_language_model_checkpoint(checkpoint)
