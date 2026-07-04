from __future__ import annotations

import json

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_sustained_runtime_evidence import (
    SURFACE,
    run_language_sustained_runtime_evidence,
    run_language_sustained_runtime_evidence_from_checkpoint,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    save_language_model_checkpoint,
)


def _language_runtime_fixture(*, expert_count: int = 2) -> tuple[
    MarulhoLanguageModel,
    ByteLevelLanguageTokenizer,
]:
    torch.manual_seed(20260703)
    tokenizer = ByteLevelLanguageTokenizer()
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=8,
            state_dim=12,
            expert_count=expert_count,
            active_expert_count=1,
            route_candidate_count=max(1, expert_count),
        )
    )
    return model, tokenizer


def test_language_sustained_evidence_writes_final_json_report(tmp_path) -> None:
    model, tokenizer = _language_runtime_fixture()
    model.train()
    output = tmp_path / "language-final.json"

    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=output,
        target_tokens=5,
        prompt="marulho",
        tick_tokens=4,
        quantum_tokens=2,
        timeout_seconds=5.0,
        collect_environment=False,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["report_status"] == "final"
    assert report["success"] is True
    assert report["token_delta"] == 5
    assert report["active_language_path"] == "marulho_lm_head"
    assert report["runtime_owner"] == "MarulhoLanguageModel"
    assert report["external_llm_used"] is False
    assert report["thought_loop_used"] is False
    assert report["cortex_used"] is False
    assert report["device_backend"]["promoted_hot_path"] is False
    assert report["failure_fallback_counters"]["triton_kernel_fallback_count"] == 5
    assert report["last_trace"]["event"] == "language_lm_head_stream"
    assert report["promotion_gate"]["short_run_is_smoke_only"] is True
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert (tmp_path / "README.md").exists()
    assert model.training is True


def test_language_sustained_evidence_writes_timeout_report(tmp_path) -> None:
    model, tokenizer = _language_runtime_fixture(expert_count=0)
    output = tmp_path / "language-timeout.json"

    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=output,
        target_tokens=3,
        timeout_seconds=0.0,
        collect_environment=False,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["success"] is False
    assert report["report_status"] == "timeout"
    assert report["failure_reason"] == "target_tokens_not_reached_before_timeout"
    assert report["evidence_state"]["timeout"] is True
    assert written["report_status"] == "timeout"
    assert written["token_delta"] == 0


def test_language_sustained_evidence_writes_manual_stop_partial_report(tmp_path) -> None:
    model, tokenizer = _language_runtime_fixture()
    calls = 0

    def should_stop() -> bool:
        nonlocal calls
        calls += 1
        return calls > 2

    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=tmp_path / "language-manual-stop.json",
        target_tokens=8,
        should_stop=should_stop,
        collect_environment=False,
    )

    assert report["success"] is False
    assert report["report_status"] == "partial"
    assert report["failure_reason"] == "manual_stop"
    assert report["token_delta"] == 2
    assert report["evidence_state"]["manual_stop"] is True


def test_language_sustained_evidence_writes_exception_report(tmp_path) -> None:
    model, tokenizer = _language_runtime_fixture()

    def fail_forward(*_args, **_kwargs):
        raise RuntimeError("simulated lm failure")

    model.forward = fail_forward  # type: ignore[method-assign]
    output = tmp_path / "language-exception.json"

    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=output,
        target_tokens=4,
        collect_environment=False,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["success"] is False
    assert report["report_status"] == "exception"
    assert report["failure_reason"] == "exception:RuntimeError"
    assert report["exception"]["type"] == "RuntimeError"
    assert written["exception"]["message"] == "simulated lm failure"


def test_language_sustained_evidence_loads_checkpoint_metadata(tmp_path) -> None:
    model, tokenizer = _language_runtime_fixture()
    checkpoint = save_language_model_checkpoint(
        tmp_path / "language.pt",
        model,
        tokenizer,
        metadata={"split_hash": "unit-test"},
    )
    output = tmp_path / "language-checkpoint-run.json"

    report = run_language_sustained_runtime_evidence_from_checkpoint(
        checkpoint,
        output_path=output,
        target_tokens=2,
        timeout_seconds=5.0,
        collect_environment=False,
        map_location="cpu",
    )

    assert report["success"] is True
    assert report["checkpoint_path"] == str(checkpoint)
    assert report["checkpoint_metadata"]["split_hash"] == "unit-test"
    assert report["device_backend"]["device"] == "cpu"
    assert json.loads(output.read_text(encoding="utf-8"))["token_delta"] == 2


def test_language_sustained_evidence_reports_padded_vocab_decode_policy(tmp_path) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    model_vocab_size = tokenizer.vocab_size + 32
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            vocab_size=model_vocab_size,
            embedding_dim=8,
            state_dim=12,
            generation_vocab_size=tokenizer.vocab_size,
        )
    )
    with torch.no_grad():
        model.lm_head.bias[tokenizer.vocab_size :].fill_(1_000_000.0)
    checkpoint = save_language_model_checkpoint(
        tmp_path / "padded-language.pt",
        model,
        tokenizer,
        metadata={"policy": "padded-vocab-decode-limit"},
    )

    report = run_language_sustained_runtime_evidence_from_checkpoint(
        checkpoint,
        output_path=tmp_path / "padded-language-run.json",
        target_tokens=4,
        timeout_seconds=5.0,
        collect_environment=False,
        map_location="cpu",
    )

    assert report["success"] is True
    assert report["model_vocab_size"] == model_vocab_size
    assert report["tokenizer_vocab_size"] == tokenizer.vocab_size
    assert report["generation_vocab_size"] == tokenizer.vocab_size
    assert report["padded_vocab_rows"] == 32
    assert report["generation_decode"]["full_model_vocab_logits_materialized"] is False
    assert max(report["generated_tail_ids"]) < tokenizer.vocab_size
