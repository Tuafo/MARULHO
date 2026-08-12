from marulho.data.language_tokenizer import (
    ByteLevelLanguageTokenizer,
    LANGUAGE_DOCUMENT_SEPARATOR,
)
from marulho.evaluation.language_source_grounding import (
    build_squad_grounding_cases,
    build_squad_long_context_cases,
    materialize_squad_training_corpus,
)


def _row(index: int, title: str, context: str, question: str, answer: str):
    return {
        "row_idx": index,
        "row": {
            "id": f"case-{index}",
            "title": title,
            "context": context,
            "question": question,
            "answers": {
                "text": [answer],
                "answer_start": [context.index(answer)],
            },
        },
    }


def test_squad_grounding_cases_keep_answer_visible_and_controls_clean() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    rows = [
        _row(
            0,
            "alpha",
            "The observatory stands in Serra Azul. It opened in 1998.",
            "Where does the observatory stand?",
            "Serra Azul",
        ),
        _row(
            1,
            "beta",
            "The archive stores copper tablets. Researchers visit weekly.",
            "What does the archive store?",
            "copper tablets",
        ),
    ]
    cases = build_squad_grounding_cases(
        rows,
        tokenizer,
        case_count=2,
        maximum_prompt_tokens=128,
        maximum_answer_tokens=32,
    )

    assert [case["case_id"] for case in cases] == ["case-0", "case-1"]
    assert all(case["answers"][0] in case["source_text"] for case in cases)
    assert all(case["answers"][0] not in case["question"] for case in cases)
    assert all(
        case["answers"][0] not in case["mismatched_source_text"] for case in cases
    )
    assert all(case["prompt_token_count"] <= 128 for case in cases)
    assert all(case["mismatched_prompt_token_count"] <= 128 for case in cases)


def test_squad_grounding_cases_reject_question_answer_leakage() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    leaking = _row(
        0,
        "leak",
        "The answer is cobalt.",
        "Is the answer cobalt?",
        "cobalt",
    )
    valid_rows = [
        _row(
            1,
            "one",
            "The vessel carried saffron.",
            "What did the vessel carry?",
            "saffron",
        ),
        _row(
            2,
            "two",
            "The station opened in winter.",
            "When did the station open?",
            "winter",
        ),
    ]
    cases = build_squad_grounding_cases(
        [leaking, *valid_rows],
        tokenizer,
        case_count=2,
        maximum_prompt_tokens=128,
        maximum_answer_tokens=32,
    )

    assert {case["case_id"] for case in cases} == {"case-1", "case-2"}


def test_squad_training_corpus_uses_prompt_answer_documents(tmp_path) -> None:
    output = tmp_path / "train.txt"
    report = materialize_squad_training_corpus(
        {
            "contract_sha256": "frozen",
            "cases": [
                {"prompt": "Context: A\nQuestion: Q\nAnswer:", "answers": ["one"]},
                {"prompt": "Context: B\nQuestion: R\nAnswer:", "answers": ["two"]},
            ],
        },
        output_path=output,
    )

    assert output.read_text(encoding="utf-8") == (
        "Context: A\nQuestion: Q\nAnswer: one"
        + LANGUAGE_DOCUMENT_SEPARATOR
        + "Context: B\nQuestion: R\nAnswer: two"
    )
    assert report["document_count"] == 2
    assert report["manifest_contract_sha256"] == "frozen"


def test_long_context_cases_keep_multiple_blocks_and_contextual_span_bounds() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    rows = [
        _row(
            0,
            "alpha",
            "The first instrument is quiet. The observatory stands in Serra Azul. "
            "It opened in 1998 and keeps several historical records for visitors.",
            "Where does the observatory stand?",
            "Serra Azul",
        ),
        _row(
            1,
            "beta",
            "Researchers arrive every Monday. The archive stores copper tablets. "
            "A reading room nearby contains maps and handwritten field journals.",
            "What does the archive store?",
            "copper tablets",
        ),
        _row(
            2,
            "gamma",
            "The harbor closes during storms. The vessel carried saffron in sealed "
            "containers. Inspectors recorded every package before the ship departed.",
            "What did the vessel carry?",
            "saffron",
        ),
    ]
    cases = build_squad_long_context_cases(
        rows,
        tokenizer,
        case_count=2,
        maximum_prompt_tokens=256,
        minimum_source_tokens=64,
        maximum_answer_tokens=32,
        maximum_cases_per_title=2,
        excluded_case_ids=("case-0",),
        retrieval_block_tokens=32,
        maximum_causal_sequence_tokens=256,
        maximum_oracle_prompt_tokens=128,
    )

    assert [case["case_id"] for case in cases] == ["case-1", "case-2"]
    assert all(case["retrieval_block_count"] >= 2 for case in cases)
    assert all(case["source_token_count"] >= 64 for case in cases)
    assert all(case["answer_in_context_token_count"] <= 32 for case in cases)
    assert all(case["prompt_token_count"] <= 256 for case in cases)
    assert all(case["causal_prompt"].endswith("Answer: ") for case in cases)
    assert all(case["causal_sequence_token_count"] - 1 <= 256 for case in cases)
    assert all(case["oracle_prompt_token_count"] <= 128 for case in cases)
    assert all(case["answers"][0] in case["oracle_source_text"] for case in cases)
    assert all(case["oracle_causal_prompt"].endswith("Answer: ") for case in cases)
    assert all(
        tokenizer.encode(case["causal_prompt"], add_bos=True, add_eos=False)
        == tokenizer.encode(f"{case['prompt']} ", add_bos=True, add_eos=False)
        for case in cases
    )
    assert all(
        case["question_only_prompt_token_count"]
        + case["answer_in_context_token_count"]
        + 1
        <= 72
        for case in cases
    )
    assert all(
        case["answers"][0] not in case["mismatched_source_text"] for case in cases
    )


def test_long_context_cases_reject_empty_normalized_unicode_answers() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    rows = [
        _row(
            0,
            "unicode",
            "Several old letters appear in this historical passage, including ƿ. "
            "The surrounding explanation is deliberately long enough for two blocks.",
            "Which old letter appears?",
            "ƿ",
        ),
        _row(
            1,
            "valid",
            "The first room is empty. The archive stores copper tablets for study. "
            "Researchers inspect the collection every week and record each visit.",
            "What does the archive store?",
            "copper tablets",
        ),
        _row(
            2,
            "control",
            "The harbor closes during storms. The vessel carried saffron in sealed "
            "containers. Inspectors recorded every package before departure.",
            "What did the vessel carry?",
            "saffron",
        ),
    ]
    cases = build_squad_long_context_cases(
        rows,
        tokenizer,
        case_count=2,
        maximum_prompt_tokens=256,
        minimum_source_tokens=64,
        maximum_answer_tokens=32,
        retrieval_block_tokens=32,
    )

    assert [case["case_id"] for case in cases] == ["case-1", "case-2"]
