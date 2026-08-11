import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_conditional_lora import (
    MarulhoConditionalLoRALanguageModel,
    load_conditional_lora_checkpoint,
    parent_state_sha256,
    save_conditional_lora_checkpoint,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=ByteLevelLanguageTokenizer().vocab_size,
        embedding_dim=16,
        state_dim=16,
        state_layers=2,
        attention_heads=4,
        transformer_context_length=16,
        transformer_mlp_ratio=2.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
    )


def test_conditional_lora_inactive_path_is_bit_exact() -> None:
    torch.manual_seed(23)
    parent = MarulhoLanguageModel(_config()).eval()
    candidate = MarulhoConditionalLoRALanguageModel.from_parent(
        parent,
        rank=4,
    ).eval()
    inputs = torch.randint(0, 64, (3, 9))

    expected = parent(inputs, collect_telemetry=False)
    observed = candidate(inputs, collect_telemetry=False)

    assert torch.equal(observed["logits"], expected["logits"])
    assert parent_state_sha256(candidate) == parent_state_sha256(parent)
    assert not candidate.conditional_lora_enabled
    assert candidate.conditional_lora_parameter_count() > 0
    assert all(
        parameter.requires_grad
        for parameter in candidate.conditional_lora_parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in candidate.named_parameters()
        if ".lora_a." not in name and ".lora_b." not in name
    )


def test_conditional_lora_active_path_streams_and_trains() -> None:
    torch.manual_seed(29)
    parent = MarulhoLanguageModel(_config()).eval()
    candidate = MarulhoConditionalLoRALanguageModel.from_parent(parent, rank=4)
    inputs = torch.randint(0, 64, (2, 7))
    targets = torch.randint(0, 64, (2, 7))
    candidate.set_conditional_lora_enabled(True)
    assert candidate.conditional_lora_enabled

    candidate.eval()
    full = candidate(inputs, collect_telemetry=False)
    state = None
    logits = []
    for position in range(inputs.shape[1]):
        step = candidate.forward_step(
            inputs[:, position],
            state,
            collect_telemetry=False,
        )
        state = step["state"]
        logits.append(step["logits"])
    assert torch.allclose(
        torch.cat(logits, dim=1),
        full["logits"],
        atol=1e-5,
        rtol=1e-5,
    )

    candidate.train()
    loss = candidate.next_token_loss(
        inputs,
        targets,
        collect_telemetry=False,
        return_evidence=False,
    )["loss"]
    loss.backward()
    trainable = tuple(candidate.conditional_lora_named_parameters())
    assert all(parameter.grad is not None for _name, parameter in trainable)
    assert all(
        parameter.grad is None
        for name, parameter in candidate.named_parameters()
        if ".lora_a." not in name and ".lora_b." not in name
    )


def test_conditional_lora_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(31)
    parent = MarulhoLanguageModel(_config())
    candidate = MarulhoConditionalLoRALanguageModel.from_parent(parent, rank=4)
    tokenizer = ByteLevelLanguageTokenizer()
    output = tmp_path / "lora.pt"
    save_conditional_lora_checkpoint(
        output,
        candidate,
        tokenizer,
        {"decision": "test"},
    )

    restored, restored_tokenizer, metadata = load_conditional_lora_checkpoint(output)

    assert not restored.conditional_lora_enabled
    assert metadata == {"decision": "test"}
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert restored.state_dict().keys() == candidate.state_dict().keys()
    assert all(
        torch.equal(restored.state_dict()[name], value)
        for name, value in candidate.state_dict().items()
    )
