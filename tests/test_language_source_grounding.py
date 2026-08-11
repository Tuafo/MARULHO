from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_source_grounding import (
    build_squad_grounding_cases,
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
