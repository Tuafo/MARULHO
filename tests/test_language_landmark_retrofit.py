import pytest
import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_landmark_retrofit_falsification import (
    _evaluate_landmark_grounding,
)
from marulho.training.language_landmark_retrofit import (
    FrozenBaseLandmarkRetrofit,
    build_landmark_retrofit_batches,
    cache_landmark_retrofit_hidden,
    load_landmark_retrofit_checkpoint,
    save_landmark_retrofit_checkpoint,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    language_model_state_sha256,
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
            transformer_context_length=64,
            transformer_mlp_ratio=2.0,
        )
    )
    model = FrozenBaseLandmarkRetrofit(
        base,
        tokenizer=tokenizer,
        pad_id=tokenizer.pad_id,
        eos_id=tokenizer.eos_id,
        block_tokens=16,
        maximum_blocks=5,
        retrieval_width=16,
        adapter_width=24,
        adapter_layers=1,
        adapter_heads=4,
    )
    return model, tokenizer


def _manifest():
    rows = (
        (
            "aaaaaaaaaaaaaa bronze key rests beside the old gate.",
            "What rests beside the gate?",
            "bronze key",
        ),
        (
            "A quiet room contains maps and a silver bell on the desk.",
            "What is on the desk?",
            "silver bell",
        ),
    )
    cases = []
    for index, (source, question, answer) in enumerate(rows):
        mismatched_source = rows[(index + 1) % len(rows)][0]
        question_only = f"Question: {question}\nAnswer:"
        cases.append(
            {
                "case_id": f"case-{index}",
                "source_text": source,
                "question": question,
                "answers": [answer],
                "question_only_prompt": question_only,
                "prompt": f"Context: {source}\nQuestion: {question}\nAnswer:",
                "mismatched_source_text": mismatched_source,
                "mismatched_prompt": (
                    f"Context: {mismatched_source}\nQuestion: {question}\nAnswer:"
                ),
            }
        )
    return {"cases": cases}


def test_landmark_batches_preserve_answer_and_hide_it_from_retrieval() -> None:
    model, tokenizer = _model()
    batches, report = build_landmark_retrofit_batches(
        _manifest(),
        tokenizer,
        batch_size=2,
        block_tokens=model.block_tokens,
        maximum_blocks=model.maximum_blocks,
        query_length=model.context_length,
    )
    batch = batches[0]

    assert report["all_gold_evidence_contains_answer_union"]
    assert report["all_retrieval_queries_answer_free"]
    assert report["boundary_spanning_answer_count"] == 1
    for row, case in enumerate(_manifest()["cases"]):
        evidence = batch.source_ids[row, batch.gold_evidence_indices[row]].flatten()
        evidence = evidence[evidence != tokenizer.pad_id]
        assert case["answers"][0] in tokenizer.decode(evidence.tolist())
        query = batch.retrieval_query_ids[row][
            batch.retrieval_query_attention_mask[row]
        ]
        assert case["answers"][0] not in tokenizer.decode(query.tolist())
        targets = batch.generator_target_ids[row][batch.generator_loss_mask[row]]
        assert tokenizer.decode(targets.tolist()) == case["answers"][0]


def test_runtime_split_matches_training_blocks_exactly() -> None:
    model, tokenizer = _model()
    batches, _report = build_landmark_retrofit_batches(
        _manifest(),
        tokenizer,
        batch_size=2,
        block_tokens=model.block_tokens,
        maximum_blocks=model.maximum_blocks,
        query_length=model.context_length,
    )
    prompt = torch.tensor(
        tokenizer.encode(_manifest()["cases"][0]["prompt"], add_eos=False)
    )
    source_ids, source_mask, valid, generator_query, retrieval_query = (
        model._split_long_prompt(prompt.unsqueeze(0))
    )

    assert torch.equal(source_ids[0], batches[0].source_ids[0])
    assert torch.equal(source_mask[0], batches[0].source_attention_mask[0])
    assert torch.equal(valid[0], batches[0].block_valid_mask[0])
    expected_retrieval_query = batches[0].retrieval_query_ids[0][
        batches[0].retrieval_query_attention_mask[0]
    ]
    expected_generator_query = torch.tensor(
        tokenizer.encode(
            f"{_manifest()['cases'][0]['question_only_prompt']} ", add_eos=False
        )
    )
    assert torch.equal(retrieval_query[0], expected_retrieval_query)
    assert torch.equal(generator_query[0], expected_generator_query)


def test_landmark_loss_reaches_every_retrofit_path_without_parent_gradients() -> None:
    torch.manual_seed(56)
    model, tokenizer = _model()
    batches = build_landmark_retrofit_batches(
        _manifest(),
        tokenizer,
        batch_size=2,
        block_tokens=model.block_tokens,
        maximum_blocks=model.maximum_blocks,
        query_length=model.context_length,
    )[0]
    cached, cache_report = cache_landmark_retrofit_hidden(
        model.base, batches, device=torch.device("cpu")
    )
    parent_before = language_model_state_sha256(model.base)

    loss, components = model.loss(cached[0])
    loss.backward()

    assert cache_report["case_count"] == 2
    assert torch.isfinite(loss)
    assert torch.isfinite(components["generator_loss"])
    assert torch.isfinite(components["retrieval_loss"])
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert all(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if not name.startswith("base.")
    )
    assert language_model_state_sha256(model.base) == parent_before


def test_source_absent_generation_is_exact_parent_bypass() -> None:
    model, tokenizer = _model()
    model.eval()
    prompt = torch.tensor(tokenizer.encode("Question: What is rain?\nAnswer:"))

    parent = model.base.generate(
        prompt,
        max_new_tokens=4,
        eos_id=tokenizer.eos_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )
    candidate = model.generate(
        prompt,
        max_new_tokens=4,
        eos_id=tokenizer.eos_id,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    assert torch.equal(candidate["generated_ids"], parent["generated_ids"])


def test_landmark_grounding_evaluator_runs_all_interventions() -> None:
    model, tokenizer = _model()
    batches = build_landmark_retrofit_batches(
        _manifest(),
        tokenizer,
        batch_size=2,
        block_tokens=model.block_tokens,
        maximum_blocks=model.maximum_blocks,
        query_length=model.context_length,
    )[0]
    cached = cache_landmark_retrofit_hidden(
        model.base, batches, device=torch.device("cpu")
    )[0]

    report = _evaluate_landmark_grounding(
        model,
        tokenizer,
        _manifest(),
        cached,
        maximum_answer_tokens=2,
    )

    assert report["valid"]
    assert set(report["conditions"]) == {
        "predicted_top2",
        "predicted_top1",
        "oracle",
        "shuffled",
        "question_only",
        "mismatched_source",
    }
    assert all(value["case_count"] == 2 for value in report["conditions"].values())


def test_landmark_checkpoint_is_compact_and_parent_strict(tmp_path) -> None:
    model, _tokenizer = _model()
    output = tmp_path / "landmark.pt"
    save_landmark_retrofit_checkpoint(
        output,
        model,
        parent_checkpoint_sha256="parent123",
        metadata={"decision": "test"},
    )
    restored_base = _model()[0].base
    restored_base.load_state_dict(model.base.state_dict(), strict=True)
    restored, metadata = load_landmark_retrofit_checkpoint(
        output,
        restored_base,
        _model()[1],
        expected_parent_checkpoint_sha256="parent123",
    )

    assert metadata == {"decision": "test"}
    assert language_model_state_sha256(restored) == language_model_state_sha256(model)
    assert output.stat().st_size < 1_000_000
    with pytest.raises(ValueError, match="parent checkpoint differs"):
        load_landmark_retrofit_checkpoint(
            output,
            _model()[0].base,
            _model()[1],
            expected_parent_checkpoint_sha256="wrong-parent",
        )
