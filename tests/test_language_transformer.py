from __future__ import annotations

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_transformer import (
    MarulhoCausalTransformerStateBlock,
)


def _config(**overrides) -> LanguageModelConfig:
    values = {
        "vocab_size": 96,
        "embedding_dim": 32,
        "state_dim": 32,
        "state_core": "transformer",
        "state_layers": 2,
        "attention_heads": 4,
        "transformer_context_length": 16,
        "transformer_mlp_ratio": 2.0,
        "transformer_dropout": 0.0,
        "tie_embeddings": True,
    }
    values.update(overrides)
    return LanguageModelConfig(**values)


def test_transformer_is_causal_and_backpropagates() -> None:
    torch.manual_seed(7)
    model = MarulhoLanguageModel(_config()).eval()
    first = torch.tensor([[1, 8, 9, 10, 11, 12]], dtype=torch.long)
    second = torch.tensor([[1, 8, 9, 44, 45, 46]], dtype=torch.long)

    first_logits = model(first, collect_telemetry=False)["logits"]
    second_logits = model(second, collect_telemetry=False)["logits"]
    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)

    model.train()
    result = model(first)
    loss = F.cross_entropy(
        result["logits"][:, :-1].reshape(-1, model.config.vocab_size),
        first[:, 1:].reshape(-1),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert model.token_embedding.weight.grad is not None
    assert torch.isfinite(model.token_embedding.weight.grad).all()
    assert result["telemetry"]["state_core"] == "transformer"
    assert result["telemetry"]["attention_heads"] == 4


def test_transformer_incremental_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(13)
    model = MarulhoLanguageModel(_config()).eval()
    token_ids = torch.tensor([[1, 3, 5, 7, 9, 11, 13, 15]], dtype=torch.long)
    with torch.no_grad():
        full = model(token_ids, collect_telemetry=False)["logits"]
        prompt = model(token_ids[:, :3], collect_telemetry=False)
        state = prompt["state"]
        incremental_logits = [prompt["logits"][:, -1]]
        for index in range(3, int(token_ids.shape[1])):
            step = model.forward_step(
                token_ids[:, index : index + 1],
                state,
                collect_telemetry=False,
            )
            state = step["state"]
            incremental_logits.append(step["logits"][:, -1])

    expected = full[:, 2:]
    actual = torch.stack(incremental_logits, dim=1)
    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-5)
    assert int(state["position"].item()) == int(token_ids.shape[1])
    assert int(state["layer_0_key"].shape[2]) == int(token_ids.shape[1])


def test_transformer_cache_is_bounded_by_context() -> None:
    torch.manual_seed(17)
    block = MarulhoCausalTransformerStateBlock(
        16,
        16,
        state_layers=2,
        attention_heads=4,
        context_length=8,
        mlp_ratio=2.0,
    ).eval()
    state = None
    with torch.no_grad():
        for _ in range(20):
            _hidden, state, telemetry = block.step(torch.randn(1, 16), state)

    assert state is not None
    assert int(state["position"].item()) == 20
    assert int(state["layer_0_key"].shape[2]) == 8
    assert int(state["layer_1_value"].shape[2]) == 8
    assert telemetry["kv_cache_tokens"] == 8


def test_transformer_checkpoint_restores_tied_weights_and_generation(tmp_path) -> None:
    torch.manual_seed(19)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        _config(
            vocab_size=tokenizer.vocab_size,
            transformer_context_length=32,
        )
    ).eval()
    path = save_language_model_checkpoint(
        tmp_path / "transformer.pt",
        model,
        tokenizer,
        metadata={"architecture": "transformer"},
    )

    restored, restored_tokenizer, metadata = load_language_model_checkpoint(path)
    assert restored.config.state_core == "transformer"
    assert restored.lm_head.weight.data_ptr() == restored.token_embedding.weight.data_ptr()
    assert restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
    assert metadata == {"architecture": "transformer"}

    prompt = torch.tensor(tokenizer.encode("MARULHO", add_eos=False), dtype=torch.long)
    generated = restored.generate(prompt, max_new_tokens=4, eos_id=tokenizer.eos_id)
    assert generated["new_token_count"] >= 1
    assert generated["external_llm_used"] is False
    assert generated["generation_decode"]["decode_strategy"] == "greedy_argmax"

    sampled_first = restored.generate(
        prompt,
        max_new_tokens=8,
        eos_id=tokenizer.eos_id,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )
    sampled_second = restored.generate(
        prompt,
        max_new_tokens=8,
        eos_id=tokenizer.eos_id,
        temperature=0.8,
        top_p=0.9,
        seed=23,
    )
    assert torch.equal(
        sampled_first["generated_ids"],
        sampled_second["generated_ids"],
    )
    assert sampled_first["generation_decode"]["decode_strategy"] == "nucleus_sampling"
    assert sampled_first["generation_decode"]["top_p_applied"] is True
    assert sampled_first["generation_decode"]["sampling_seed"] == 23


def test_rejected_recurrent_language_config_is_not_accepted() -> None:
    config = _config(state_core="gru")
    try:
        MarulhoLanguageModel(config)
    except ValueError as error:
        assert "only state_core='transformer'" in str(error)
    else:  # pragma: no cover
        raise AssertionError("recurrent language configuration was accepted")
