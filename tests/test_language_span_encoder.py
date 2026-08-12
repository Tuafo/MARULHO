import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    language_model_state_sha256,
)
from marulho.training.language_span_encoder import (
    FrozenBaseSpanEncoder,
    build_span_supervision_batches,
    load_span_encoder_checkpoint,
    save_span_encoder_checkpoint,
    span_encoder_type_ids,
)


def _span_model():
    tokenizer = ByteLevelLanguageTokenizer()
    base = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=16,
            state_dim=16,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=128,
            transformer_mlp_ratio=2.0,
        )
    )
    model = FrozenBaseSpanEncoder(
        base,
        context_marker_ids=torch.tensor(
            tokenizer.encode("Context:", add_bos=False, add_eos=False)
        ),
        question_marker_ids=torch.tensor(
            tokenizer.encode("\nQuestion:", add_bos=False, add_eos=False)
        ),
        answer_marker_ids=torch.tensor(
            tokenizer.encode("\nAnswer:", add_bos=False, add_eos=False)
        ),
        width=16,
        layers=1,
        heads=4,
    )
    return model, tokenizer


def _manifest():
    cases = []
    for index, (source, question, answer) in enumerate(
        (
            ("The cat is red.", "What color is the cat?", "red"),
            ("The key is bronze.", "What is the key made of?", "bronze"),
        )
    ):
        cases.append(
            {
                "case_id": f"case-{index}",
                "source_text": source,
                "question": question,
                "answers": [answer],
                "prompt": f"Context: {source}\nQuestion: {question}\nAnswer:",
            }
        )
    return {"cases": cases}


def test_span_supervision_maps_visible_answers_to_source_tokens() -> None:
    _model, tokenizer = _span_model()
    batches, report = build_span_supervision_batches(
        _manifest(), tokenizer, sequence_length=128, batch_size=2
    )
    batch = batches[0]

    assert report["all_gold_spans_source_contained"]
    for index in range(2):
        selected = batch.input_ids[
            index,
            batch.start_positions[index] : batch.end_positions[index] + 1,
        ]
        assert tokenizer.decode(selected.tolist()) == _manifest()["cases"][index][
            "answers"
        ][0]
        assert batch.source_mask[index, batch.start_positions[index]]


def test_span_encoder_backpropagates_without_mutating_parent() -> None:
    torch.manual_seed(54)
    model, tokenizer = _span_model()
    batch = build_span_supervision_batches(
        _manifest(), tokenizer, sequence_length=128, batch_size=2
    )[0][0]
    parent_before = language_model_state_sha256(model.base)

    loss = model.loss(batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )
    assert language_model_state_sha256(model.base) == parent_before


def test_span_types_require_all_three_markers() -> None:
    model, tokenizer = _span_model()
    intact = torch.tensor(
        [tokenizer.encode("Context: red cat\nQuestion: color?\nAnswer:")]
    )
    types, source = span_encoder_type_ids(
        intact,
        context_marker_ids=model.context_marker_ids,
        question_marker_ids=model.question_marker_ids,
        answer_marker_ids=model.answer_marker_ids,
    )
    assert source.any()
    assert set(types[source].tolist()) == {1}
    question_only = torch.tensor(
        [tokenizer.encode("Question: color?\nAnswer:")]
    )
    assert not span_encoder_type_ids(
        question_only,
        context_marker_ids=model.context_marker_ids,
        question_marker_ids=model.question_marker_ids,
        answer_marker_ids=model.answer_marker_ids,
    )[1].any()


def test_span_checkpoint_is_compact_and_parent_strict(tmp_path) -> None:
    model, _tokenizer = _span_model()
    output = tmp_path / "span.pt"
    save_span_encoder_checkpoint(
        output,
        model,
        parent_checkpoint_sha256="parent123",
        metadata={"decision": "test"},
    )
    restored_base = _span_model()[0].base
    restored_base.load_state_dict(model.base.state_dict(), strict=True)
    restored, metadata = load_span_encoder_checkpoint(
        output,
        restored_base,
        expected_parent_checkpoint_sha256="parent123",
    )
    assert metadata == {"decision": "test"}
    assert language_model_state_sha256(restored) == language_model_state_sha256(
        model
    )
    assert output.stat().st_size < 1_000_000
    with pytest.raises(ValueError, match="parent checkpoint differs"):
        load_span_encoder_checkpoint(
            output,
            _span_model()[0].base,
            expected_parent_checkpoint_sha256="wrong-parent",
        )
