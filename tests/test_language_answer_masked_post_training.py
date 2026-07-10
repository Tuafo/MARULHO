from pathlib import Path

from marulho.data.language_tokenizer import (
    ByteLevelLanguageTokenizer,
    LANGUAGE_DOCUMENT_SEPARATOR,
)
from marulho.evaluation.language_answer_masked_post_training import (
    build_masked_answer_batches,
    masked_post_training_branch_decision,
)


def test_answer_mask_excludes_prompt_and_padding() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    corpus = (
        "### source=test"
        f"{LANGUAGE_DOCUMENT_SEPARATOR}Nora moved the cup. Question: Where is "
        "the coin? Answer: The coin is in the cup."
        f"{LANGUAGE_DOCUMENT_SEPARATOR}Mara gave Eli a key. Question: Who has "
        "the key? Answer: Eli has the key.\n"
    )
    batches, report = build_masked_answer_batches(
        corpus,
        tokenizer,
        batch_size=2,
        context_length=512,
        max_documents=10,
    )

    assert report["document_count"] == 2
    assert report["prompt_loss_masked"] is True
    assert report["padding_loss_masked"] is True
    batch = batches[0]
    assert int(batch.answer_mask.sum().item()) == report["answer_token_count"]
    assert bool(batch.answer_mask[:, 0].any().item()) is False
    assert bool(batch.answer_mask.any().item()) is True
    padding_positions = batch.input_ids == tokenizer.pad_id
    assert bool(batch.answer_mask[padding_positions].any().item()) is False
    expected_answer_tokens = sum(
        len(tokenizer.encode(f" {answer}", add_bos=False, add_eos=True))
        for answer in ("The coin is in the cup.", "Eli has the key.")
    )
    assert int(batch.answer_mask.sum().item()) == expected_answer_tokens


def test_masked_post_training_decision_requires_free_answers_and_retention() -> None:
    assert masked_post_training_branch_decision(
        free_accuracy_after=0.70,
        candidate_accuracy_after=0.90,
        general_loss_delta=0.08,
    ) == "answer_masked_post_training_promising"
    assert masked_post_training_branch_decision(
        free_accuracy_after=0.70,
        candidate_accuracy_after=0.90,
        general_loss_delta=0.30,
    ) == "answer_masked_post_training_forgets_general_language"
    assert masked_post_training_branch_decision(
        free_accuracy_after=0.30,
        candidate_accuracy_after=0.90,
        general_loss_delta=0.02,
    ) == "answer_masking_keeps_ranking_but_not_free_binding"
    assert masked_post_training_branch_decision(
        free_accuracy_after=0.30,
        candidate_accuracy_after=0.70,
        general_loss_delta=0.02,
    ) == "answer_masked_objective_falsified"
