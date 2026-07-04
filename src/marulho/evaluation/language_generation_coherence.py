"""Grounded prompt-suite coherence evidence for MARULHO-owned LM generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    load_language_model_checkpoint,
)


SURFACE = "marulho_language_generation_coherence_report.v1"
ARTIFACT_KIND = "marulho_language_generation_coherence_report"


@dataclass(frozen=True)
class LanguageGenerationPromptCase:
    prompt_text: str
    source_text: str
    max_new_tokens: int = 64
    min_new_tokens: int = 8
    min_prefix_match_chars: int = 8
    min_prefix_match_fraction: float = 0.10
    min_printable_fraction: float = 0.95
    min_distinct_bigram_fraction: float = 0.20
    max_token_run_length: int = 8


def default_generation_coherence_prompt_cases(
    source_text: str = DEFAULT_CORPUS,
) -> tuple[LanguageGenerationPromptCase, ...]:
    return (
        LanguageGenerationPromptCase(prompt_text="MARULHO", source_text=source_text),
        LanguageGenerationPromptCase(
            prompt_text="Replay protects",
            source_text=source_text,
        ),
        LanguageGenerationPromptCase(
            prompt_text="Structural pressure",
            source_text=source_text,
        ),
        LanguageGenerationPromptCase(
            prompt_text="Long sustained runs",
            source_text=source_text,
        ),
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _printable_fraction(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\n\t")
    return float(printable) / float(len(text))


def _distinct_bigram_fraction(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 2:
        return 1.0 if token_ids else 0.0
    bigrams = list(zip(token_ids, token_ids[1:]))
    return float(len(set(bigrams))) / float(len(bigrams))


def _max_token_run_length(token_ids: Sequence[int]) -> int:
    if not token_ids:
        return 0
    longest = 1
    current = 1
    previous = int(token_ids[0])
    for token_id in token_ids[1:]:
        token_id = int(token_id)
        if token_id == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
            previous = token_id
    return max(longest, current)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _expected_source_continuation(
    *,
    prompt_text: str,
    source_text: str,
    max_chars: int,
) -> dict[str, Any]:
    source_index = str(source_text).find(str(prompt_text))
    if source_index < 0:
        return {
            "source_prompt_found": False,
            "expected_source_continuation": "",
        }
    start = source_index + len(str(prompt_text))
    stop = start + max(0, int(max_chars))
    expected = str(source_text)[start:stop]
    return {
        "source_prompt_found": True,
        "expected_source_continuation": expected,
    }


def _decoded_generation_case(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    case: LanguageGenerationPromptCase,
    *,
    generation_repetition_penalty: float = 1.0,
    generation_no_repeat_ngram_size: int = 0,
) -> dict[str, Any]:
    repetition_penalty = max(1.0, float(generation_repetition_penalty))
    no_repeat_ngram_size = max(0, int(generation_no_repeat_ngram_size))
    prompt_ids = torch.tensor(
        tokenizer.encode(case.prompt_text, add_eos=False),
        dtype=torch.long,
    )
    generation = model.generate(
        prompt_ids,
        max_new_tokens=max(0, int(case.max_new_tokens)),
        eos_id=tokenizer.eos_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    generated_ids = generation["generated_ids"]
    if not isinstance(generated_ids, torch.Tensor):
        generated_ids = torch.as_tensor(generated_ids, dtype=torch.long)
    flat_ids = [int(token_id) for token_id in generated_ids.detach().cpu().reshape(-1).tolist()]
    prompt_count = int(prompt_ids.numel())
    continuation_ids = flat_ids[prompt_count:]
    generated_text = tokenizer.decode(flat_ids)
    continuation_text = tokenizer.decode(continuation_ids)
    expected = _expected_source_continuation(
        prompt_text=case.prompt_text,
        source_text=case.source_text,
        max_chars=len(continuation_text),
    )
    expected_text = str(expected["expected_source_continuation"])
    prefix_match_chars = _common_prefix_length(expected_text, continuation_text)
    prefix_match_fraction = (
        float(prefix_match_chars) / float(len(expected_text)) if expected_text else 0.0
    )
    printable_fraction = _printable_fraction(continuation_text)
    distinct_bigram_fraction = _distinct_bigram_fraction(continuation_ids)
    max_run = _max_token_run_length(continuation_ids)
    source_prompt_found = bool(expected["source_prompt_found"])
    next_character_matches_source = bool(
        expected_text and continuation_text and expected_text[0] == continuation_text[0]
    )
    continuation_long_enough = int(len(continuation_ids)) >= int(case.min_new_tokens)
    prefix_gate_passed = (
        prefix_match_chars >= int(case.min_prefix_match_chars)
        and prefix_match_fraction >= float(case.min_prefix_match_fraction)
    )
    printable_gate_passed = printable_fraction >= float(case.min_printable_fraction)
    diversity_gate_passed = (
        distinct_bigram_fraction >= float(case.min_distinct_bigram_fraction)
        and max_run <= int(case.max_token_run_length)
    )
    owned_generation = (
        bool(generation.get("owned_by_marulho"))
        and not bool(generation.get("external_llm_used"))
        and str(generation.get("active_language_path")) == "marulho_lm_head"
    )
    passed = (
        owned_generation
        and source_prompt_found
        and continuation_long_enough
        and prefix_gate_passed
        and printable_gate_passed
        and diversity_gate_passed
    )
    failure_reasons: list[str] = []
    if not owned_generation:
        failure_reasons.append("generation_not_marulho_owned")
    if not source_prompt_found:
        failure_reasons.append("prompt_not_found_in_source")
    if not continuation_long_enough:
        failure_reasons.append("continuation_too_short")
    if not prefix_gate_passed:
        failure_reasons.append("source_prefix_match_below_threshold")
    if not printable_gate_passed:
        failure_reasons.append("printable_fraction_below_threshold")
    if not diversity_gate_passed:
        failure_reasons.append("token_repetition_or_low_bigram_diversity")
    return {
        "surface": "marulho_language_generation_coherence_case.v1",
        "prompt_text": case.prompt_text,
        "source_text_hash": _sha256_text(case.source_text),
        "generation_surface": generation.get("surface"),
        "generation_decode": dict(generation.get("generation_decode") or {}),
        "active_language_path": generation.get("active_language_path"),
        "external_llm_used": bool(generation.get("external_llm_used")),
        "owned_by_marulho": bool(generation.get("owned_by_marulho")),
        "prompt_token_count": prompt_count,
        "generated_token_count": len(flat_ids),
        "continuation_token_count": len(continuation_ids),
        "new_token_count": int(generation.get("new_token_count", 0) or 0),
        "generated_text": generated_text,
        "continuation_text": continuation_text,
        "sequence_hash": tokenizer.sequence_hash(flat_ids),
        "continuation_sequence_hash": tokenizer.sequence_hash(continuation_ids),
        "source_prompt_found": source_prompt_found,
        "expected_source_continuation": expected_text,
        "next_character_matches_source": next_character_matches_source,
        "prefix_match_chars": int(prefix_match_chars),
        "prefix_match_fraction": float(prefix_match_fraction),
        "printable_fraction": float(printable_fraction),
        "distinct_bigram_fraction": float(distinct_bigram_fraction),
        "max_token_run_length": int(max_run),
        "thresholds": {
            "min_new_tokens": int(case.min_new_tokens),
            "min_prefix_match_chars": int(case.min_prefix_match_chars),
            "min_prefix_match_fraction": float(case.min_prefix_match_fraction),
            "min_printable_fraction": float(case.min_printable_fraction),
            "min_distinct_bigram_fraction": float(case.min_distinct_bigram_fraction),
            "max_token_run_length": int(case.max_token_run_length),
        },
        "passed": bool(passed),
        "failure_reasons": failure_reasons,
    }


def _coherence_summary(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    case_count = len(cases)
    passed_cases = [case for case in cases if bool(case.get("passed"))]
    return {
        "surface": "marulho_language_generation_coherence_summary.v1",
        "case_count": int(case_count),
        "passed_case_count": int(len(passed_cases)),
        "case_pass_rate": float(len(passed_cases)) / float(case_count) if case_count else 0.0,
        "mean_prefix_match_chars": _mean(
            [float(case.get("prefix_match_chars", 0.0) or 0.0) for case in cases]
        ),
        "mean_prefix_match_fraction": _mean(
            [float(case.get("prefix_match_fraction", 0.0) or 0.0) for case in cases]
        ),
        "mean_printable_fraction": _mean(
            [float(case.get("printable_fraction", 0.0) or 0.0) for case in cases]
        ),
        "mean_distinct_bigram_fraction": _mean(
            [float(case.get("distinct_bigram_fraction", 0.0) or 0.0) for case in cases]
        ),
        "max_token_run_length": max(
            [int(case.get("max_token_run_length", 0) or 0) for case in cases],
            default=0,
        ),
        "next_character_match_rate": _mean(
            [
                1.0 if bool(case.get("next_character_matches_source")) else 0.0
                for case in cases
            ]
        ),
    }


def run_language_generation_coherence_report(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    prompt_cases: Sequence[LanguageGenerationPromptCase] | None = None,
    min_case_pass_rate: float = 1.0,
    checkpoint_path: str | Path | None = None,
    output_path: str | Path | None = None,
    generation_repetition_penalty: float = 1.0,
    generation_no_repeat_ngram_size: int = 0,
) -> dict[str, Any]:
    cases = tuple(prompt_cases or default_generation_coherence_prompt_cases())
    if not cases:
        raise ValueError("At least one generation coherence prompt case is required")
    repetition_penalty = max(1.0, float(generation_repetition_penalty))
    no_repeat_ngram_size = max(0, int(generation_no_repeat_ngram_size))
    case_reports = [
        _decoded_generation_case(
            model,
            tokenizer,
            case,
            generation_repetition_penalty=repetition_penalty,
            generation_no_repeat_ngram_size=no_repeat_ngram_size,
        )
        for case in cases
    ]
    summary = _coherence_summary(case_reports)
    case_pass_rate = float(summary["case_pass_rate"])
    external_llm_used = any(bool(case.get("external_llm_used")) for case in case_reports)
    active_language_paths = {
        str(case.get("active_language_path")) for case in case_reports
    }
    active_language_path = (
        active_language_paths.pop()
        if len(active_language_paths) == 1
        else "mixed_or_unknown"
    )
    owned_by_marulho = all(
        bool(case.get("owned_by_marulho"))
        and str(case.get("active_language_path")) == "marulho_lm_head"
        and not bool(case.get("external_llm_used"))
        for case in case_reports
    )
    coherence_available = (
        case_pass_rate >= float(min_case_pass_rate)
        and owned_by_marulho
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": bool(owned_by_marulho),
        "external_llm_used": bool(external_llm_used),
        "loads_external_checkpoint": False,
        "active_language_path": active_language_path,
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "prompt_suite": {
            "surface": "marulho_language_generation_coherence_prompt_suite.v1",
            "case_count": len(cases),
            "min_case_pass_rate": float(min_case_pass_rate),
            "review_kind": "automated_grounded_prompt_suite_not_human_review",
            "generation_decode_controls": {
                "repetition_penalty": float(repetition_penalty),
                "repetition_penalty_applied": bool(repetition_penalty > 1.0),
                "no_repeat_ngram_size": int(no_repeat_ngram_size),
                "no_repeat_ngram_applied": bool(no_repeat_ngram_size > 0),
                "decode_controls_requested": bool(
                    repetition_penalty > 1.0 or no_repeat_ngram_size > 0
                ),
            },
            "prompt_cases": [asdict(case) for case in cases],
        },
        "cases": case_reports,
        "summary": summary,
        "promotion_gate": {
            "status": (
                "generation_coherence_available"
                if coherence_available
                else "blocked_generation_coherence"
            ),
            "generation_coherence_available": bool(coherence_available),
            "grounded_prompt_suite_available": bool(coherence_available),
            "human_review_available": False,
            "promotes_prompt_suite_coherence_claim": bool(coherence_available),
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "requires_long_run_pairing": True,
        },
    }
    if output_path is not None:
        path = Path(output_path)
        report["output_path"] = str(path)
        write_json_report_with_readme(path, report)
    return report


def _prompt_case_from_arg(value: str, *, source_text: str) -> LanguageGenerationPromptCase:
    parts = [part.strip() for part in str(value).split("|")]
    if not parts or not parts[0]:
        raise argparse.ArgumentTypeError("prompt case must start with prompt text")
    kwargs: dict[str, Any] = {"prompt_text": parts[0], "source_text": source_text}
    if len(parts) > 1 and parts[1]:
        kwargs["max_new_tokens"] = int(parts[1])
    if len(parts) > 2 and parts[2]:
        kwargs["min_prefix_match_chars"] = int(parts[2])
    if len(parts) > 3 and parts[3]:
        kwargs["min_prefix_match_fraction"] = float(parts[3])
    return LanguageGenerationPromptCase(**kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-location", default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--prompt-case",
        action="append",
        default=[],
        help=(
            "Prompt case as prompt|max_new_tokens|min_prefix_chars|min_prefix_fraction. "
            "May be passed multiple times."
        ),
    )
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.0)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=0)
    args = parser.parse_args()
    source_text = (
        args.source.read_text(encoding="utf-8") if args.source is not None else DEFAULT_CORPUS
    )
    model, tokenizer, _metadata = load_language_model_checkpoint(
        args.checkpoint,
        map_location=args.map_location,
    )
    if args.map_location is not None:
        model = model.to(torch.device(args.map_location))
    prompt_cases = (
        tuple(_prompt_case_from_arg(value, source_text=source_text) for value in args.prompt_case)
        if args.prompt_case
        else default_generation_coherence_prompt_cases(source_text=source_text)
    )
    report = run_language_generation_coherence_report(
        model,
        tokenizer,
        prompt_cases=prompt_cases,
        min_case_pass_rate=float(args.min_case_pass_rate),
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        generation_repetition_penalty=max(1.0, float(args.generation_repetition_penalty)),
        generation_no_repeat_ngram_size=max(
            0,
            int(args.generation_no_repeat_ngram_size),
        ),
    )
    return 0 if report["promotion_gate"]["generation_coherence_available"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
