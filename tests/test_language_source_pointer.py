import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    language_model_state_sha256,
)
from marulho.training.language_source_pointer import (
    FrozenSourcePointerLanguageModel,
    load_source_pointer_checkpoint,
    save_source_pointer_checkpoint,
    source_pointer_answer_loss,
    structural_source_mask,
)


def _pointer_model():
    tokenizer = ByteLevelLanguageTokenizer()
    base = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=64,
            transformer_mlp_ratio=2.0,
        )
    )
    context = torch.tensor(
        tokenizer.encode("Context:", add_bos=False, add_eos=False)
    )
    question = torch.tensor(
        tokenizer.encode("\nQuestion:", add_bos=False, add_eos=False)
    )
    return (
        FrozenSourcePointerLanguageModel(
            base,
            context_marker_ids=context,
            question_marker_ids=question,
            pointer_rank=8,
        ),
        tokenizer,
    )


def test_structural_source_mask_requires_explicit_context_and_question() -> None:
    model, tokenizer = _pointer_model()
    intact = torch.tensor(
        [tokenizer.encode("Context: red cat\nQuestion: color?\nAnswer:")]
    )
    mask = structural_source_mask(
        intact,
        context_marker_ids=model.context_marker_ids,
        question_marker_ids=model.question_marker_ids,
    )
    assert mask.any()
    assert tokenizer.decode(intact[0][mask[0]].tolist()).strip() == "red cat"
    question_only = torch.tensor(
        [tokenizer.encode("Question: color?\nAnswer:")]
    )
    assert not structural_source_mask(
        question_only,
        context_marker_ids=model.context_marker_ids,
        question_marker_ids=model.question_marker_ids,
    ).any()


def test_source_pointer_backpropagates_only_into_pointer() -> None:
    torch.manual_seed(53)
    model, tokenizer = _pointer_model()
    text = "Context: red cat\nQuestion: color?\nAnswer: red"
    ids = tokenizer.encode(text)
    row = torch.full((65,), tokenizer.pad_id, dtype=torch.long)
    row[: len(ids)] = torch.tensor(ids)
    inputs = row[:-1].unsqueeze(0)
    targets = row[1:].unsqueeze(0)
    answer = torch.tensor(
        tokenizer.encode("Answer:", add_bos=False, add_eos=False)
    )
    parent_before = language_model_state_sha256(model.base)
    loss = source_pointer_answer_loss(
        model,
        inputs,
        targets,
        answer_marker_ids=answer,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )
    assert language_model_state_sha256(model.base) == parent_before


def test_source_pointer_implements_inactive_next_token_loss() -> None:
    model, tokenizer = _pointer_model()
    ids = torch.tensor([tokenizer.encode("ordinary text")])
    result = model.next_token_loss(
        ids[:, :-1],
        ids[:, 1:],
        collect_telemetry=False,
    )
    assert torch.isfinite(result["loss"])
    assert result["evidence"]["full_vocab_logits_materialized"]


def test_source_pointer_checkpoint_is_parent_strict(tmp_path) -> None:
    model, _tokenizer = _pointer_model()
    output = tmp_path / "pointer.pt"
    save_source_pointer_checkpoint(
        output,
        model,
        parent_checkpoint_sha256="parent123",
        metadata={"decision": "test"},
    )
    restored_base = _pointer_model()[0].base
    restored, metadata = load_source_pointer_checkpoint(
        output,
        restored_base,
        expected_parent_checkpoint_sha256="parent123",
    )
    assert metadata == {"decision": "test"}
    assert restored.pointer_state_dict().keys() == model.pointer_state_dict().keys()
    for name, value in model.pointer_state_dict().items():
        assert torch.equal(restored.pointer_state_dict()[name], value)
