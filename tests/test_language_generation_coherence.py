from __future__ import annotations

import json
from types import SimpleNamespace

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_generation_coherence import (
    SURFACE,
    LanguageGenerationPromptCase,
    run_language_generation_coherence_report,
)


class _FakeGenerationModel:
    def __init__(
        self,
        tokenizer: ByteLevelLanguageTokenizer,
        *,
        continuation_text: str,
        external_llm_used: bool = False,
    ) -> None:
        self.config = SimpleNamespace(active_language_path="marulho_lm_head")
        self._tokenizer = tokenizer
        self._continuation_ids = tokenizer.encode(continuation_text, add_eos=False)
        self._external_llm_used = bool(external_llm_used)

    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
    ) -> dict[str, object]:
        del eos_id
        flat_prompt = prompt_ids.detach().cpu().reshape(-1).to(torch.long)
        continuation = torch.tensor(
            self._continuation_ids[: max(0, int(max_new_tokens))],
            dtype=torch.long,
        )
        return {
            "surface": "marulho_language_generation.v1",
            "generated_ids": torch.cat([flat_prompt, continuation]).unsqueeze(0),
            "new_token_count": int(continuation.numel()),
            "active_language_path": "marulho_lm_head",
            "external_llm_used": self._external_llm_used,
            "owned_by_marulho": not self._external_llm_used,
            "loads_external_checkpoint": False,
        }


def test_language_generation_coherence_report_passes_grounded_prompt_suite(
    tmp_path,
) -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    output = tmp_path / "generation-coherence.json"
    source_text = "MARULHO learns runtime evidence from local source windows."

    report = run_language_generation_coherence_report(
        _FakeGenerationModel(tokenizer, continuation_text=" learns runtime evidence"),
        tokenizer,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="MARULHO",
                source_text=source_text,
                max_new_tokens=32,
                min_new_tokens=8,
                min_prefix_match_chars=8,
                min_prefix_match_fraction=0.20,
            ),
        ),
        output_path=output,
    )
    written = json.loads(output.read_text(encoding="utf-8"))

    assert report["surface"] == SURFACE
    assert written["surface"] == SURFACE
    assert report["promotion_gate"]["generation_coherence_available"] is True
    assert report["promotion_gate"]["grounded_prompt_suite_available"] is True
    assert report["promotion_gate"]["human_review_available"] is False
    assert report["promotion_gate"]["promotes_runtime_claim"] is False
    assert report["summary"]["passed_case_count"] == 1
    assert report["cases"][0]["passed"] is True
    assert report["cases"][0]["external_llm_used"] is False
    assert "learns runtime evidence" in report["cases"][0]["continuation_text"]
    assert (tmp_path / "README.md").exists()


def test_language_generation_coherence_report_blocks_unsupported_generation() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    source_text = "MARULHO learns runtime evidence from local source windows."

    report = run_language_generation_coherence_report(
        _FakeGenerationModel(tokenizer, continuation_text=" zzzzzzzzzzzzzz"),
        tokenizer,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="MARULHO",
                source_text=source_text,
                max_new_tokens=16,
                min_new_tokens=8,
                min_prefix_match_chars=8,
                min_prefix_match_fraction=0.20,
            ),
        ),
    )

    assert report["promotion_gate"]["generation_coherence_available"] is False
    assert report["summary"]["passed_case_count"] == 0
    assert report["cases"][0]["passed"] is False
    assert "source_prefix_match_below_threshold" in report["cases"][0]["failure_reasons"]


def test_language_generation_coherence_report_blocks_external_generation() -> None:
    tokenizer = ByteLevelLanguageTokenizer()
    source_text = "MARULHO learns runtime evidence from local source windows."

    report = run_language_generation_coherence_report(
        _FakeGenerationModel(
            tokenizer,
            continuation_text=" learns runtime evidence",
            external_llm_used=True,
        ),
        tokenizer,
        prompt_cases=(
            LanguageGenerationPromptCase(
                prompt_text="MARULHO",
                source_text=source_text,
                max_new_tokens=32,
                min_new_tokens=8,
                min_prefix_match_chars=8,
                min_prefix_match_fraction=0.20,
            ),
        ),
    )

    assert report["external_llm_used"] is True
    assert report["owned_by_marulho"] is False
    assert report["promotion_gate"]["generation_coherence_available"] is False
    assert report["cases"][0]["passed"] is False
    assert "generation_not_marulho_owned" in report["cases"][0]["failure_reasons"]
