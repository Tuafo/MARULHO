"""Grounding-support evidence for MARULHO-owned LM generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.semantics.grounding_diagnostics import GroundingDiagnostics
from marulho.semantics.grounding_text import (
    match_terms,
    query_focused_text,
    salient_query_terms,
)
from marulho.training.language_model import MarulhoLanguageModel


SURFACE = "marulho_language_grounding_support_report.v1"
ARTIFACT_KIND = "marulho_language_grounding_support_report"


def _ordered_terms(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = str(value).strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _coverage(matched: Sequence[str], required: Sequence[str]) -> float:
    if not required:
        return 0.0
    return float(len(matched)) / float(len(required))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _decoded_generation_text(
    generation: dict[str, Any],
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    prompt_token_count: int,
) -> dict[str, Any]:
    generated_ids = generation["generated_ids"]
    if not isinstance(generated_ids, torch.Tensor):
        generated_ids = torch.as_tensor(generated_ids, dtype=torch.long)
    flat_ids = [int(item) for item in generated_ids.detach().cpu().reshape(-1).tolist()]
    continuation_ids = flat_ids[int(prompt_token_count) :]
    return {
        "generated_text": tokenizer.decode(flat_ids),
        "continuation_text": tokenizer.decode(continuation_ids),
        "generated_token_count": int(len(flat_ids)),
        "continuation_token_count": int(len(continuation_ids)),
        "sequence_hash": tokenizer.sequence_hash(flat_ids),
        "continuation_sequence_hash": tokenizer.sequence_hash(continuation_ids),
    }


def run_language_grounding_support_report(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    prompt_text: str,
    source_text: str,
    required_terms: Sequence[str] | None = None,
    max_new_tokens: int = 0,
    min_source_term_coverage: float = 1.0,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Measure whether a prompt has source-term support for LM generation.

    This is a support report, not a generation-quality review. It verifies that
    MARULHO's own LM path can emit a generation artifact under a source-term
    coverage gate while keeping unsupported generated terms visible for review.
    """

    prompt_ids = torch.tensor(
        tokenizer.encode(prompt_text, add_eos=False),
        dtype=torch.long,
    )
    generation = model.generate(
        prompt_ids,
        max_new_tokens=max(0, int(max_new_tokens)),
        eos_id=tokenizer.eos_id,
    )
    decoded = _decoded_generation_text(
        generation,
        tokenizer,
        prompt_token_count=int(prompt_ids.numel()),
    )
    required = _ordered_terms(
        required_terms if required_terms is not None else salient_query_terms(prompt_text)
    )
    source_terms = _ordered_terms(salient_query_terms(source_text))
    matched_required = _ordered_terms(match_terms(required, source_text))
    matched_set = set(matched_required)
    missing_required = [term for term in required if term not in matched_set]
    source_term_coverage = _coverage(matched_required, required)
    coverage_gate_passed = (
        bool(required)
        and source_term_coverage >= float(min_source_term_coverage)
        and not missing_required
    )

    generated_terms = _ordered_terms(salient_query_terms(decoded["generated_text"]))
    continuation_terms = _ordered_terms(salient_query_terms(decoded["continuation_text"]))
    generated_supported = _ordered_terms(
        match_terms(generated_terms, f"{source_text} {prompt_text}")
    )
    continuation_supported = _ordered_terms(
        match_terms(continuation_terms, f"{source_text} {prompt_text}")
    )
    generated_supported_set = set(generated_supported)
    continuation_supported_set = set(continuation_supported)
    unsupported_generated = [
        term for term in generated_terms if term not in generated_supported_set
    ]
    unsupported_continuation = [
        term for term in continuation_terms if term not in continuation_supported_set
    ]

    support_available = (
        coverage_gate_passed
        and bool(generation["owned_by_marulho"])
        and not bool(generation["external_llm_used"])
        and str(generation["active_language_path"]) == "marulho_transformer"
    )
    diagnostics = GroundingDiagnostics(
        kind="lm_source_term_support",
        target=str(prompt_text),
        target_terms=tuple(required),
        matched_target_terms=tuple(matched_required),
        evidence_supported_terms=tuple(matched_required),
        grounded_evidence_count=len(matched_required),
        response_coverage=source_term_coverage,
        evidence_coverage=_coverage(matched_required, source_terms),
        evidence_alignment=source_term_coverage,
        alignment_score=source_term_coverage,
        fallback_used=False,
    ).to_dict()

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "source": {
            "source_text_hash": _sha256_text(source_text),
            "source_term_count": len(source_terms),
            "focused_source_text": query_focused_text(source_text, required),
        },
        "source_terms": {
            "required_terms": required,
            "matched_required_terms": matched_required,
            "missing_required_terms": missing_required,
            "source_term_coverage": source_term_coverage,
            "min_source_term_coverage": float(min_source_term_coverage),
            "source_term_coverage_gate_passed": coverage_gate_passed,
        },
        "grounding_diagnostics": diagnostics,
        "generation": {
            "surface": generation["surface"],
            "prompt_text": str(prompt_text),
            "prompt_token_count": int(prompt_ids.numel()),
            "max_new_tokens": max(0, int(max_new_tokens)),
            "new_token_count": int(generation["new_token_count"]),
            "active_language_path": generation["active_language_path"],
            "external_llm_used": bool(generation["external_llm_used"]),
            "owned_by_marulho": bool(generation["owned_by_marulho"]),
            "generated_text": decoded["generated_text"],
            "continuation_text": decoded["continuation_text"],
            "generated_token_count": decoded["generated_token_count"],
            "continuation_token_count": decoded["continuation_token_count"],
            "sequence_hash": decoded["sequence_hash"],
            "continuation_sequence_hash": decoded["continuation_sequence_hash"],
            "generated_terms": generated_terms,
            "continuation_terms": continuation_terms,
            "supported_generated_terms": generated_supported,
            "supported_continuation_terms": continuation_supported,
            "unsupported_generated_terms": unsupported_generated,
            "unsupported_continuation_terms": unsupported_continuation,
            "review_kind": "support_terms_only_not_human_generation_quality_review",
        },
        "promotion_gate": {
            "status": (
                "grounding_support_available"
                if support_available
                else "blocked_missing_source_terms"
            ),
            "grounding_support_available": support_available,
            "source_term_coverage_gate_passed": coverage_gate_passed,
            "requires_human_grounded_generation_review": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
    }
    if output_path is not None:
        path = Path(output_path)
        report["output_path"] = str(path)
        write_json_report_with_readme(path, report)
    return report
