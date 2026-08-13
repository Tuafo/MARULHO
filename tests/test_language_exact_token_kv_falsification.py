from __future__ import annotations

from dataclasses import replace

import torch

from marulho.evaluation.language_exact_token_kv_falsification import (
    ExactTokenKVController,
    V63Config,
    _gradient_audit,
    _schedule_indices,
    active_zero_parity,
    prepare_cases,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


class CharacterTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    @staticmethod
    def _token(character: str) -> int:
        return 3 + ord(character)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> list[int]:
        result = [self._token(character) for character in text]
        if add_bos:
            result.insert(0, self.bos_id)
        if add_eos:
            result.append(self.eos_id)
        return result

    def encode_with_offsets(
        self,
        text: str,
        *,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        ids = [self._token(character) for character in text]
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_bos:
            ids.insert(0, self.bos_id)
            offsets.insert(0, (0, 0))
        if add_eos:
            ids.append(self.eos_id)
            offsets.append((len(text), len(text)))
        return ids, offsets


def _small_parent() -> MarulhoLanguageModel:
    torch.manual_seed(1201)
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=64,
            embedding_dim=16,
            state_dim=16,
            state_layers=2,
            attention_heads=2,
            transformer_context_length=16,
            transformer_mlp_ratio=2.0,
            transformer_dropout=0.0,
        )
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _small_controller() -> ExactTokenKVController:
    return ExactTokenKVController(
        state_layers=2,
        attention_heads=2,
        head_dim=8,
        correction_scale=0.25,
        model_seed=1211,
    )


def test_prepare_cases_uses_exact_offset_boundary_and_answer_only_loss() -> None:
    tokenizer = CharacterTokenizer()
    question = "Where?"
    source = "A fact."
    row = {
        "case_id": "case-1",
        "title": "heldout",
        "question": question,
        "answers": ["Here"],
        "source_text": source,
        "oracle_source_text": "Fact.",
        "mismatched_source_text": "Wrong.",
        "causal_prompt": f"Context: {source}\nQuestion: {question}\nAnswer: ",
        "oracle_causal_prompt": f"Context: Fact.\nQuestion: {question}\nAnswer: ",
        "mismatched_prompt": f"Context: Wrong.\nQuestion: {question}\nAnswer:",
        "question_only_prompt": f"Question: {question}\nAnswer:",
    }
    config = replace(V63Config(), context_length=96)
    prepared = prepare_cases((row,), tokenizer, config)
    evidence = prepared.boundary_evidence[0]

    assert prepared.input_ids.shape == (1, 96)
    assert evidence["token_boundary_exact"] is True
    assert evidence["delimiter_normalized_suffix_exact"] is True
    assert evidence["right_padding_only_after_eos"] is True
    assert prepared.source_mask[0, 0].item() is False
    assert int(prepared.source_mask.sum()) == len(f"Context: {source}")
    assert not bool(
        (prepared.source_mask & prepared.answer_mask).any()
    )
    assert int(prepared.answer_mask.sum()) == len("Here") + 1


def test_zero_controller_is_bit_exact_for_hidden_logits_and_all_kv_state() -> None:
    parent = _small_parent()
    controller = _small_controller()
    ids = torch.randint(0, 64, (2, 12))
    source_mask = torch.zeros_like(ids, dtype=torch.bool)
    source_mask[:, 1:7] = True

    parity = active_zero_parity(parent, controller, ids, source_mask)

    assert parity["hidden_exact"] is True
    assert parity["logits_exact"] is True
    assert parity["state_exact"] is True
    assert parity["state_key_count"] == 5


def test_all_layer_head_key_value_corrections_receive_gradients() -> None:
    parent = _small_parent()
    controller = _small_controller()
    controller.train()
    ids = torch.randint(0, 64, (3, 12))
    source_mask = torch.zeros_like(ids, dtype=torch.bool)
    source_mask[:, 1:8] = True

    hidden, _state = controller.forward_parent(parent, ids, source_mask)
    loss = hidden[:, -1].float().square().mean()
    loss.backward()
    audit = _gradient_audit(controller)

    assert audit["all_trainable_tensors_nonzero"] is True
    assert audit["matrix_count"] == 8
    assert audit["all_correction_matrices_nonzero"] is True
    assert all(parameter.grad is None for parameter in parent.parameters())


def test_schedule_is_deterministic_and_epoch_complete() -> None:
    config = replace(V63Config(), epochs=3, data_seed=91)
    first, first_hash = _schedule_indices(8, config)
    second, second_hash = _schedule_indices(8, config)

    assert torch.equal(first, second)
    assert first_hash == second_hash
    assert len(first) == 24
    for epoch in first.view(3, 8):
        assert sorted(epoch.tolist()) == list(range(8))


def test_frozen_v39_shape_has_exact_preregistered_parameter_budget() -> None:
    controller = ExactTokenKVController(
        state_layers=10,
        attention_heads=12,
        head_dim=64,
        correction_scale=0.25,
        model_seed=63_131,
    )

    assert sum(parameter.numel() for parameter in controller.parameters()) == 983_040
    assert controller.correction_matrix_count == 240
    assert {parameter.dtype for parameter in controller.parameters()} == {
        torch.float32
    }
    assert all(int(torch.count_nonzero(parameter)) == 0 for parameter in controller.parameters())
