import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    language_model_state_sha256,
)
from marulho.training.language_multiview_transducer import (
    FrozenBaseMultiViewAnswerTransducer,
    build_multiview_supervision_batches,
    load_multiview_transducer_checkpoint,
    multiview_type_ids,
    save_multiview_transducer_checkpoint,
)


def _model():
    tokenizer = ByteLevelLanguageTokenizer()
    base = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=24,
            state_dim=24,
            state_layers=1,
            attention_heads=4,
            transformer_context_length=128,
            transformer_mlp_ratio=2.0,
        )
    )
    model = FrozenBaseMultiViewAnswerTransducer(
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
        bos_id=tokenizer.bos_id,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        width=24,
        encoder_layers=1,
        decoder_layers=1,
        heads=4,
        maximum_answer_tokens=8,
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


def test_multiview_supervision_emits_complete_answer_then_eos() -> None:
    _candidate, tokenizer = _model()
    batches, report = build_multiview_supervision_batches(
        _manifest(),
        tokenizer,
        sequence_length=128,
        batch_size=2,
        maximum_answer_tokens=8,
    )
    batch = batches[0]

    assert report["all_gold_spans_source_contained"]
    for row in range(2):
        targets = batch.pointer_targets[row]
        positions = targets[(targets >= 0) & (targets < 128)]
        selected = batch.input_ids[row].gather(0, positions)
        assert tokenizer.decode(selected.tolist()) == _manifest()["cases"][row][
            "answers"
        ][0]
        assert targets[len(positions)].item() == 128


def test_multiview_loss_reaches_every_trainable_path_without_parent_gradients() -> None:
    torch.manual_seed(55)
    model, tokenizer = _model()
    batch = build_multiview_supervision_batches(
        _manifest(),
        tokenizer,
        sequence_length=128,
        batch_size=2,
        maximum_answer_tokens=8,
    )[0][0]
    parent_before = language_model_state_sha256(model.base)

    loss, components = model.loss(batch, view_mode="both")
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(components["pointer_loss"])
    assert torch.isfinite(components["span_loss"])
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )
    assert language_model_state_sha256(model.base) == parent_before


def test_multiview_modes_require_marked_source_and_keep_shape() -> None:
    model, tokenizer = _model()
    intact = torch.tensor(
        [tokenizer.encode("Context: red cat\nQuestion: color?\nAnswer:")]
    )
    types, source = multiview_type_ids(
        intact,
        context_marker_ids=model.context_marker_ids,
        question_marker_ids=model.question_marker_ids,
        answer_marker_ids=model.answer_marker_ids,
    )
    assert source.any()
    assert set(types[source].tolist()) == {1}
    for mode in ("both", "bidirectional_only", "causal_only"):
        memory, attention, mode_source = model.encode_memory(
            intact, view_mode=mode
        )
        assert memory.shape == (1, intact.shape[1], model.width)
        assert attention.shape == intact.shape
        assert torch.equal(mode_source, source)


def test_multiview_checkpoint_is_compact_and_parent_strict(tmp_path) -> None:
    model, _tokenizer = _model()
    output = tmp_path / "multiview.pt"
    save_multiview_transducer_checkpoint(
        output,
        model,
        parent_checkpoint_sha256="parent123",
        metadata={"decision": "test"},
    )
    restored_base = _model()[0].base
    restored_base.load_state_dict(model.base.state_dict(), strict=True)
    restored, metadata = load_multiview_transducer_checkpoint(
        output,
        restored_base,
        expected_parent_checkpoint_sha256="parent123",
    )
    assert metadata == {"decision": "test"}
    assert language_model_state_sha256(restored) == language_model_state_sha256(
        model
    )
    assert output.stat().st_size < 2_000_000
    with pytest.raises(ValueError, match="parent checkpoint differs"):
        load_multiview_transducer_checkpoint(
            output,
            _model()[0].base,
            expected_parent_checkpoint_sha256="wrong-parent",
        )
