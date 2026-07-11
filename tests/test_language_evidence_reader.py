from __future__ import annotations

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_evidence_reader import (
    EvidenceReaderConfig,
    MarulhoEvidenceReaderLanguageModel,
)
from marulho.training.language_hashed_micro_experts import (
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)


def _reader() -> tuple[ByteLevelLanguageTokenizer, MarulhoEvidenceReaderLanguageModel]:
    tokenizer = ByteLevelLanguageTokenizer()
    cortex = MarulhoHashedMicroExpertLanguageModel(
        HashedMicroExpertConfig(
            vocab_size=tokenizer.vocab_size,
            width=32,
            layers=2,
            attention_heads=4,
            context_length=64,
            baseline_hidden_width=64,
            shared_hidden_width=32,
            expert_layer_index=1,
            expert_pool_size=128,
            routing_heads=2,
            experts_per_head=2,
        )
    )
    return tokenizer, MarulhoEvidenceReaderLanguageModel(
        cortex,
        EvidenceReaderConfig(width=32, attention_heads=4),
    )


def test_gate_zero_is_exact_cortex_parity() -> None:
    torch.manual_seed(3)
    tokenizer, model = _reader()
    query = torch.randint(0, tokenizer.vocab_size, (2, 17))
    expected = model.cortex.forward(query, collect_telemetry=False)["logits"]
    observed = model.forward(
        query, None, interface="gate_zero", collect_telemetry=False
    )["logits"]
    assert torch.equal(observed, expected)


def test_raw_context_reproduces_direct_concatenated_cortex() -> None:
    torch.manual_seed(5)
    tokenizer, model = _reader()
    evidence = torch.randint(0, tokenizer.vocab_size, (2, 11))
    query = torch.randint(0, tokenizer.vocab_size, (2, 17))
    direct = model.cortex.forward(
        torch.cat((evidence, query), dim=1), collect_telemetry=False
    )["logits"][:, -17:]
    observed = model.forward(
        query, evidence, interface="raw_context", collect_telemetry=False
    )["logits"]
    assert torch.equal(observed, direct)


def test_separate_reader_changes_logits_without_consuming_query_positions() -> None:
    torch.manual_seed(7)
    tokenizer, model = _reader()
    query = torch.randint(0, tokenizer.vocab_size, (2, 17))
    first = torch.randint(0, tokenizer.vocab_size, (2, 11))
    second = torch.randint(0, tokenizer.vocab_size, (2, 11))
    left = model.forward(
        query, first, interface="separate_reader", collect_telemetry=True
    )
    right = model.forward(
        query, second, interface="separate_reader", collect_telemetry=False
    )
    assert left["logits"].shape == right["logits"].shape == (
        2,
        17,
        tokenizer.vocab_size,
    )
    assert not torch.equal(left["logits"], right["logits"])
    assert left["telemetry"]["local_query_positions"] == 17
    assert left["telemetry"]["evidence_positions"] == 11
    assert left["telemetry"]["reader_active"] is True


def test_masked_reader_loss_reaches_cortex_cross_attention_norms_and_gate() -> None:
    torch.manual_seed(11)
    tokenizer, model = _reader()
    query = torch.randint(0, tokenizer.vocab_size, (2, 17))
    evidence = torch.randint(0, tokenizer.vocab_size, (2, 11))
    targets = torch.randint(0, tokenizer.vocab_size, (2, 17))
    mask = torch.zeros(2, 17, dtype=torch.bool)
    mask[:, -4:] = True
    loss = model.masked_next_token_loss(
        query,
        targets,
        mask,
        evidence,
        interface="separate_reader",
    )
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    reader_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("cortex.")
    }
    assert all(
        int(torch.count_nonzero(parameter.grad)) > 0
        for parameter in reader_parameters.values()
        if parameter.grad is not None
    )


def test_reader_generation_is_owned_and_keeps_evidence_out_of_returned_sequence() -> None:
    torch.manual_seed(13)
    tokenizer, model = _reader()
    query = torch.randint(0, tokenizer.vocab_size, (2, 9))
    evidence = torch.randint(0, tokenizer.vocab_size, (2, 7))
    row = model.generate_with_evidence(
        query,
        evidence,
        interface="separate_reader",
        max_new_tokens=3,
        eos_id=None,
    )
    assert row["generated_ids"].shape == (2, 12)
    assert torch.equal(row["generated_ids"][:, :9], query)
    assert row["external_llm_used"] is False
    report = model.reader_parameter_report()
    assert report["reader_parameters"] > 0
    assert report["reader_parameter_tensors"] > 0
