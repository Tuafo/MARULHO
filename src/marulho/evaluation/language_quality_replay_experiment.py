"""Checkpoint-backed hard-prompt replay experiment for MARULHO LM quality."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    default_generation_coherence_prompt_cases,
    run_language_generation_coherence_report,
)
from marulho.evaluation.language_runtime_benchmark_suite import (
    run_language_runtime_benchmark_suite,
)
from marulho.evaluation.language_sustained_runtime_evidence import (
    run_language_sustained_runtime_evidence,
)
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageBatch,
    build_language_model_splits,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


SURFACE = "marulho_language_quality_replay_experiment.v1"
ARTIFACT_KIND = "marulho_language_quality_replay_experiment"


@dataclass(frozen=True)
class LanguageQualityReplayExperimentConfig:
    sequence_length: int = 64
    stride: int = 16
    batch_size: int = 16
    hard_prompt_repeat: int = 16
    hard_prompt_context_chars: int = 192
    max_new_batches: int = 8
    max_replay_batches: int = 8
    max_old_eval_batches: int = 8
    max_new_eval_batches: int = 8
    max_steps: int = 2
    learning_rate: float = 8e-4
    replay_loss_weight: float = 0.35
    forgetting_tolerance: float = 100.0
    replay_retention_tolerance: float = 100.0
    rollback_on_forgetting: bool = False
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 8
    collect_training_telemetry: bool = False
    min_case_pass_rate: float = 1.0
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3
    sustained_target_tokens: int = 0
    sustained_target_token_counts: tuple[int, ...] = ()
    sustained_prompt: str = "MARULHO"
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 600.0
    benchmark_suite_output_path: str = ""
    benchmark_gpu_kernel_evidence_paths: tuple[str, ...] = ()
    collect_environment: bool = False
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def _read_text(path: str | Path | None, *, default: str) -> tuple[str, str]:
    if path is None:
        return default, "default"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Language quality replay source is empty: {resolved}")
    return text, str(resolved)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trim_batches(
    batches: Sequence[LanguageBatch],
    limit: int,
) -> tuple[LanguageBatch, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _sustained_targets(
    config: LanguageQualityReplayExperimentConfig,
) -> tuple[int, ...]:
    targets: list[int] = []
    single_target = max(0, int(config.sustained_target_tokens))
    if single_target > 0:
        targets.append(single_target)
    for value in config.sustained_target_token_counts:
        target = max(0, int(value))
        if target > 0:
            targets.append(target)
    unique_targets: list[int] = []
    for target in targets:
        if target not in unique_targets:
            unique_targets.append(target)
    return tuple(unique_targets)


def _prompt_segment(
    case: LanguageGenerationPromptCase,
    *,
    context_chars: int,
) -> tuple[str, dict[str, Any]]:
    prompt = str(case.prompt_text)
    source = str(case.source_text)
    source_index = source.find(prompt)
    if source_index < 0:
        fallback = f"{prompt} {source[: max(0, int(context_chars))]}"
        return fallback, {
            "prompt_text": prompt,
            "source_prompt_found": False,
            "source_index": -1,
            "segment_character_count": len(fallback),
        }
    stop = source_index + len(prompt) + max(0, int(context_chars))
    segment = source[source_index:stop]
    return segment, {
        "prompt_text": prompt,
        "source_prompt_found": True,
        "source_index": int(source_index),
        "segment_character_count": len(segment),
    }


def _ensure_minimum_tokens(
    text: str,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    minimum_tokens: int,
) -> str:
    if len(tokenizer.encode(text, add_bos=True, add_eos=True)) >= int(minimum_tokens):
        return text
    expanded = str(text)
    while len(tokenizer.encode(expanded, add_bos=True, add_eos=True)) < int(
        minimum_tokens
    ):
        expanded = f"{expanded}\n{expanded}"
    return expanded


def _build_hard_prompt_corpus(
    prompt_cases: Sequence[LanguageGenerationPromptCase],
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    sequence_length: int,
    repeat: int,
    context_chars: int,
) -> tuple[str, dict[str, Any]]:
    segments: list[str] = []
    segment_reports: list[dict[str, Any]] = []
    for case in prompt_cases:
        segment, segment_report = _prompt_segment(
            case,
            context_chars=int(context_chars),
        )
        segment_reports.append(segment_report)
        if segment.strip():
            segments.extend([segment] * max(1, int(repeat)))
    if not segments:
        raise ValueError("Hard-prompt replay produced no source segments")
    hard_text = "\n".join(segments)
    hard_text = _ensure_minimum_tokens(
        hard_text,
        tokenizer,
        minimum_tokens=max(2, int(sequence_length) + 1),
    )
    encoded_count = len(tokenizer.encode(hard_text, add_bos=True, add_eos=True))
    return hard_text, {
        "surface": "marulho_language_hard_prompt_replay_corpus.v1",
        "case_count": len(prompt_cases),
        "prompt_repeat": max(1, int(repeat)),
        "context_chars": max(0, int(context_chars)),
        "segment_reports": segment_reports,
        "source_prompt_found_count": sum(
            1 for item in segment_reports if bool(item["source_prompt_found"])
        ),
        "character_count": len(hard_text),
        "token_count": int(encoded_count),
        "corpus_hash": _sha256_text(hard_text),
    }


def _coherence_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_summary = dict(before.get("summary") or {})
    after_summary = dict(after.get("summary") or {})
    before_cases = {
        str(case.get("prompt_text")): case for case in before.get("cases", ())
    }
    repaired_prompts: list[str] = []
    regressed_prompts: list[str] = []
    for case in after.get("cases", ()):
        prompt = str(case.get("prompt_text"))
        prior = before_cases.get(prompt)
        if prior is None:
            continue
        if not bool(prior.get("passed")) and bool(case.get("passed")):
            repaired_prompts.append(prompt)
        if bool(prior.get("passed")) and not bool(case.get("passed")):
            regressed_prompts.append(prompt)
    return {
        "surface": "marulho_language_quality_replay_coherence_delta.v1",
        "passed_case_count_delta": int(
            after_summary.get("passed_case_count", 0) or 0
        )
        - int(before_summary.get("passed_case_count", 0) or 0),
        "case_pass_rate_delta": float(after_summary.get("case_pass_rate", 0.0) or 0.0)
        - float(before_summary.get("case_pass_rate", 0.0) or 0.0),
        "mean_prefix_match_chars_delta": float(
            after_summary.get("mean_prefix_match_chars", 0.0) or 0.0
        )
        - float(before_summary.get("mean_prefix_match_chars", 0.0) or 0.0),
        "mean_prefix_match_fraction_delta": float(
            after_summary.get("mean_prefix_match_fraction", 0.0) or 0.0
        )
        - float(before_summary.get("mean_prefix_match_fraction", 0.0) or 0.0),
        "next_character_match_rate_delta": float(
            after_summary.get("next_character_match_rate", 0.0) or 0.0
        )
        - float(before_summary.get("next_character_match_rate", 0.0) or 0.0),
        "repaired_prompt_count": len(repaired_prompts),
        "repaired_prompts": repaired_prompts,
        "regressed_prompt_count": len(regressed_prompts),
        "regressed_prompts": regressed_prompts,
        "promotes_generation_quality_claim": False,
    }


def _sustained_summary(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    successful = [report for report in reports if bool(report.get("success"))]
    target_tokens = [int(report.get("target_tokens", 0) or 0) for report in reports]
    house_scale_reports = [
        report for report in reports if int(report.get("target_tokens", 0) or 0) >= 524288
    ]
    long_run_reports = [
        report for report in reports if int(report.get("target_tokens", 0) or 0) >= 131072
    ]
    diagnostic_reports = [
        report for report in reports if int(report.get("target_tokens", 0) or 0) >= 8192
    ]
    token_rates = [
        float(report.get("tokens_per_second", 0.0) or 0.0)
        for report in successful
    ]
    return {
        "surface": "marulho_language_quality_replay_sustained_summary.v1",
        "enabled": bool(reports),
        "report_count": len(reports),
        "successful_report_count": len(successful),
        "all_success": bool(reports) and len(successful) == len(reports),
        "target_tokens": target_tokens,
        "diagnostic_8192_available": bool(diagnostic_reports),
        "long_run_131072_available": bool(long_run_reports),
        "house_scale_524288_available": bool(house_scale_reports),
        "max_target_tokens": max(target_tokens, default=0),
        "max_tokens_per_second": max(token_rates, default=0.0),
        "min_tokens_per_second": min(token_rates, default=0.0),
        "reports": [
            {
                "path": str(report.get("output_path") or ""),
                "checkpoint_path": str(report.get("checkpoint_path") or ""),
                "target_tokens": int(report.get("target_tokens", 0) or 0),
                "token_delta": int(report.get("token_delta", 0) or 0),
                "tokens_per_second": float(
                    report.get("tokens_per_second", 0.0) or 0.0
                ),
                "success": bool(report.get("success")),
                "report_status": report.get("report_status"),
                "backend": dict(report.get("execution_evidence") or {}).get(
                    "backend"
                ),
            }
            for report in reports
        ],
    }


def _disabled_sustained_summary() -> dict[str, Any]:
    return {
        "surface": "marulho_language_quality_replay_sustained_summary.v1",
        "enabled": False,
        "reason": "sustained_target_tokens_not_requested",
        "report_count": 0,
        "successful_report_count": 0,
        "all_success": False,
        "target_tokens": [],
        "diagnostic_8192_available": False,
        "long_run_131072_available": False,
        "house_scale_524288_available": False,
        "max_target_tokens": 0,
        "max_tokens_per_second": 0.0,
        "min_tokens_per_second": 0.0,
        "reports": [],
    }


def _parse_prompt_case(
    value: str,
    *,
    source_text: str,
) -> LanguageGenerationPromptCase:
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


def run_language_quality_replay_experiment(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    source_path: str | Path | None = None,
    prompt_cases: Sequence[LanguageGenerationPromptCase] | None = None,
    child_checkpoint_path: str | Path | None = None,
    config: LanguageQualityReplayExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageQualityReplayExperimentConfig()
    device = _resolve_device(cfg.device)
    output = Path(output_path)
    parent_checkpoint = Path(checkpoint_path)
    if not parent_checkpoint.is_file():
        raise FileNotFoundError(f"Language checkpoint not found: {parent_checkpoint}")
    parent_hash = _sha256_file(parent_checkpoint)
    source_text, source_label = _read_text(source_path, default=DEFAULT_CORPUS)
    cases = tuple(prompt_cases or default_generation_coherence_prompt_cases(source_text))
    if not cases:
        raise ValueError("At least one hard-prompt case is required")

    model, tokenizer, parent_metadata = load_language_model_checkpoint(
        parent_checkpoint,
        map_location="cpu",
    )
    model.to(device)
    hard_text, hard_corpus_report = _build_hard_prompt_corpus(
        cases,
        tokenizer,
        sequence_length=int(cfg.sequence_length),
        repeat=int(cfg.hard_prompt_repeat),
        context_chars=int(cfg.hard_prompt_context_chars),
    )
    replay_text = _ensure_minimum_tokens(
        source_text,
        tokenizer,
        minimum_tokens=max(2, int(cfg.sequence_length) + 1),
    )
    old_split = build_language_model_splits(
        [replay_text],
        tokenizer,
        sequence_length=int(cfg.sequence_length),
        eval_fraction=0.2,
        stride=int(cfg.stride),
        batch_size=int(cfg.batch_size),
        device=device,
    )
    hard_split = build_language_model_splits(
        [hard_text],
        tokenizer,
        sequence_length=int(cfg.sequence_length),
        eval_fraction=0.2,
        stride=int(cfg.stride),
        batch_size=int(cfg.batch_size),
        device=device,
    )
    before_coherence_path = output.with_name(f"{output.stem}-parent-coherence.json")
    before_coherence = run_language_generation_coherence_report(
        model,
        tokenizer,
        prompt_cases=cases,
        min_case_pass_rate=float(cfg.min_case_pass_rate),
        checkpoint_path=parent_checkpoint,
        output_path=before_coherence_path,
        generation_repetition_penalty=float(cfg.generation_repetition_penalty),
        generation_no_repeat_ngram_size=int(cfg.generation_no_repeat_ngram_size),
    )
    learning_config = LanguageContinualLearningConfig(
        learning_rate=float(cfg.learning_rate),
        max_steps=int(cfg.max_steps),
        replay_loss_weight=float(cfg.replay_loss_weight),
        forgetting_tolerance=float(cfg.forgetting_tolerance),
        replay_retention_tolerance=float(cfg.replay_retention_tolerance),
        rollback_on_forgetting=bool(cfg.rollback_on_forgetting),
        sparse_vocab_optimizer=bool(model.config.sampled_vocab_size > 0),
        max_grad_norm=float(cfg.max_grad_norm),
        gradient_clip_interval=max(0, int(cfg.gradient_clip_interval)),
        collect_training_telemetry=bool(cfg.collect_training_telemetry),
    )
    new_batches = _trim_batches(hard_split.train, int(cfg.max_new_batches))
    replay_batches = _trim_batches(old_split.train, int(cfg.max_replay_batches))
    old_eval_batches = _trim_batches(old_split.eval, int(cfg.max_old_eval_batches))
    new_eval_batches = _trim_batches(hard_split.eval, int(cfg.max_new_eval_batches))
    learning_report = run_language_continual_learning_window(
        model,
        new_batches=new_batches,
        old_eval_batches=old_eval_batches,
        new_eval_batches=new_eval_batches,
        replay_batches=replay_batches,
        config=learning_config,
    )

    child_checkpoint = (
        Path(child_checkpoint_path)
        if child_checkpoint_path is not None
        else output.with_name(f"{output.stem}-child-checkpoint.pt")
    )
    child_metadata = {
        "parent_checkpoint_path": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_hash,
        "parent_checkpoint_metadata": parent_metadata,
        "quality_replay_report": str(output),
        "hard_prompt_corpus_hash": hard_corpus_report["corpus_hash"],
        "learning_status": learning_report.get("status"),
        "learning_final_state_hash": dict(
            learning_report.get("rollback_evidence") or {}
        ).get("final_state_hash"),
    }
    save_language_model_checkpoint(
        child_checkpoint,
        model,
        tokenizer,
        metadata=child_metadata,
    )
    child_hash = _sha256_file(child_checkpoint)
    after_coherence_path = output.with_name(f"{output.stem}-child-coherence.json")
    after_coherence = run_language_generation_coherence_report(
        model,
        tokenizer,
        prompt_cases=cases,
        min_case_pass_rate=float(cfg.min_case_pass_rate),
        checkpoint_path=child_checkpoint,
        output_path=after_coherence_path,
        generation_repetition_penalty=float(cfg.generation_repetition_penalty),
        generation_no_repeat_ngram_size=int(cfg.generation_no_repeat_ngram_size),
    )

    sustained_targets = _sustained_targets(cfg)
    sustained_reports: list[dict[str, Any]] = []
    for sustained_target in sustained_targets:
        sustained_path = output.with_name(
            f"{output.stem}-child-sustained-{sustained_target}.json"
        )
        sustained_reports.append(
            run_language_sustained_runtime_evidence(
                model,
                tokenizer,
                output_path=sustained_path,
                target_tokens=sustained_target,
                checkpoint_path=child_checkpoint,
                checkpoint_metadata=child_metadata,
                prompt=str(cfg.sustained_prompt),
                tick_tokens=int(cfg.sustained_tick_tokens),
                quantum_tokens=int(cfg.sustained_quantum_tokens),
                timeout_seconds=float(cfg.sustained_timeout_seconds),
                generation_repetition_penalty=float(cfg.generation_repetition_penalty),
                generation_no_repeat_ngram_size=int(cfg.generation_no_repeat_ngram_size),
                collect_environment=bool(cfg.collect_environment),
            )
        )
    sustained_summary = (
        _sustained_summary(sustained_reports)
        if sustained_reports
        else _disabled_sustained_summary()
    )
    sustained_report: dict[str, Any] = (
        sustained_reports[0]
        if len(sustained_reports) == 1
        else sustained_summary
    )

    benchmark_suite_report: dict[str, Any]
    benchmark_suite_output = str(cfg.benchmark_suite_output_path or "").strip()
    if benchmark_suite_output:
        benchmark_suite_report = run_language_runtime_benchmark_suite(
            output_path=benchmark_suite_output,
            sustained_target_tokens=8,
            sustained_evidence_paths=[
                str(report.get("output_path") or "")
                for report in sustained_reports
                if report.get("output_path")
            ],
            generation_coherence_evidence_paths=(str(after_coherence_path),),
            gpu_kernel_evidence_paths=tuple(cfg.benchmark_gpu_kernel_evidence_paths),
        )
    else:
        benchmark_suite_report = {
            "surface": "marulho_language_quality_replay_benchmark_suite.v1",
            "enabled": False,
            "reason": "benchmark_suite_output_path_not_requested",
        }

    coherence_delta = _coherence_delta(before_coherence, after_coherence)
    sustained_success = bool(sustained_summary.get("all_success", False))
    same_child_quality = bool(
        dict(after_coherence.get("promotion_gate") or {}).get(
            "generation_coherence_available",
            False,
        )
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "output_path": str(output),
        "config": asdict(cfg),
        "parent_checkpoint_path": str(parent_checkpoint),
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_path": str(child_checkpoint),
        "child_checkpoint_sha256": child_hash,
        "checkpoint_lineage": {
            "surface": "marulho_language_quality_replay_checkpoint_lineage.v1",
            "parent_checkpoint_path": str(parent_checkpoint),
            "child_checkpoint_path": str(child_checkpoint),
            "parent_checkpoint_sha256": parent_hash,
            "child_checkpoint_sha256": child_hash,
            "writes_child_checkpoint": True,
            "mutates_parent_checkpoint": False,
            "rollback_available": True,
        },
        "source": {
            "surface": "marulho_language_quality_replay_source.v1",
            "source": source_label,
            "source_hash": _sha256_text(source_text),
            "source_character_count": len(source_text),
            "replay_character_count": len(replay_text),
        },
        "hard_prompt_replay": hard_corpus_report,
        "split": {
            "old_replay": old_split.report,
            "hard_prompt": hard_split.report,
            "used_new_batches": len(new_batches),
            "used_replay_batches": len(replay_batches),
            "used_old_eval_batches": len(old_eval_batches),
            "used_new_eval_batches": len(new_eval_batches),
        },
        "learning_evidence": learning_report,
        "generation_coherence_before": before_coherence,
        "generation_coherence_after": after_coherence,
        "generation_coherence_delta": coherence_delta,
        "sustained_runtime_evidence": sustained_report,
        "sustained_runtime_evidence_reports": sustained_reports,
        "sustained_runtime_evidence_summary": sustained_summary,
        "benchmark_suite_report": benchmark_suite_report,
        "experiment_review": {
            "surface": "marulho_language_quality_replay_review.v1",
            "fast_mutable_experiment": True,
            "records_hard_prompt_training_pressure": True,
            "records_actual_continual_learning": bool(
                dict(learning_report.get("learning_evidence") or {}).get(
                    "update_token_count",
                    0,
                )
                > 0
            ),
            "records_replay_retention": bool(
                "general_replay_retention_delta"
                in dict(learning_report.get("learning_evidence") or {})
            ),
            "records_same_child_generation_coherence": True,
            "records_same_child_sustained_runtime": bool(sustained_reports),
            "records_multiple_sustained_targets": bool(len(sustained_reports) > 1),
            "records_house_scale_sustained_runtime": bool(
                sustained_summary["house_scale_524288_available"]
            ),
            "records_benchmark_suite_aggregation": bool(
                benchmark_suite_report.get("surface")
                == "marulho_language_runtime_benchmark_suite.v1"
            ),
            "same_child_generation_coherence_available": bool(same_child_quality),
            "same_child_sustained_runtime_success": bool(sustained_success),
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--child-checkpoint", type=Path, default=None)
    parser.add_argument("--prompt-case", action="append", default=[])
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hard-prompt-repeat", type=int, default=16)
    parser.add_argument("--hard-prompt-context-chars", type=int, default=192)
    parser.add_argument("--max-new-batches", type=int, default=8)
    parser.add_argument("--max-replay-batches", type=int, default=8)
    parser.add_argument("--max-old-eval-batches", type=int, default=8)
    parser.add_argument("--max-new-eval-batches", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--replay-loss-weight", type=float, default=0.35)
    parser.add_argument("--forgetting-tolerance", type=float, default=100.0)
    parser.add_argument("--replay-retention-tolerance", type=float, default=100.0)
    parser.add_argument("--rollback-on-forgetting", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=8)
    parser.add_argument("--collect-training-telemetry", action="store_true")
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--sustained-target-tokens", type=int, action="append", default=[])
    parser.add_argument("--sustained-prompt", default="MARULHO")
    parser.add_argument("--sustained-tick-tokens", type=int, default=128)
    parser.add_argument("--sustained-quantum-tokens", type=int, default=16)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--benchmark-suite-output", type=Path, default=None)
    parser.add_argument("--gpu-kernel-evidence", action="append", default=[])
    parser.add_argument("--collect-environment", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    source_text, _source_label = _read_text(args.source, default=DEFAULT_CORPUS)
    cases = (
        tuple(
            _parse_prompt_case(value, source_text=source_text)
            for value in args.prompt_case
        )
        if args.prompt_case
        else default_generation_coherence_prompt_cases(source_text)
    )
    report = run_language_quality_replay_experiment(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        source_path=args.source,
        child_checkpoint_path=args.child_checkpoint,
        prompt_cases=cases,
        config=LanguageQualityReplayExperimentConfig(
            sequence_length=args.sequence_length,
            stride=args.stride,
            batch_size=args.batch_size,
            hard_prompt_repeat=args.hard_prompt_repeat,
            hard_prompt_context_chars=args.hard_prompt_context_chars,
            max_new_batches=args.max_new_batches,
            max_replay_batches=args.max_replay_batches,
            max_old_eval_batches=args.max_old_eval_batches,
            max_new_eval_batches=args.max_new_eval_batches,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            replay_loss_weight=args.replay_loss_weight,
            forgetting_tolerance=args.forgetting_tolerance,
            replay_retention_tolerance=args.replay_retention_tolerance,
            rollback_on_forgetting=bool(args.rollback_on_forgetting),
            max_grad_norm=args.max_grad_norm,
            gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
            collect_training_telemetry=bool(args.collect_training_telemetry),
            min_case_pass_rate=args.min_case_pass_rate,
            generation_repetition_penalty=args.generation_repetition_penalty,
            generation_no_repeat_ngram_size=args.generation_no_repeat_ngram_size,
            sustained_target_tokens=0,
            sustained_target_token_counts=tuple(args.sustained_target_tokens),
            sustained_prompt=args.sustained_prompt,
            sustained_tick_tokens=args.sustained_tick_tokens,
            sustained_quantum_tokens=args.sustained_quantum_tokens,
            sustained_timeout_seconds=args.sustained_timeout_seconds,
            benchmark_suite_output_path=(
                "" if args.benchmark_suite_output is None else str(args.benchmark_suite_output)
            ),
            benchmark_gpu_kernel_evidence_paths=tuple(str(path) for path in args.gpu_kernel_evidence),
            collect_environment=bool(args.collect_environment),
            device=args.device,
        ),
    )
    if args.sustained_target_tokens:
        return 0 if bool(report["sustained_runtime_evidence_summary"]["all_success"]) else 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
