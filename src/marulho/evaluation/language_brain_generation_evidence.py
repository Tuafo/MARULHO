"""Evidence runner for generation quality through an installed MarulhoBrain."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from marulho.brain import MarulhoBrain
from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    default_generation_coherence_prompt_cases,
    run_language_generation_coherence_report,
)
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_language_brain_installed_generation_evidence.v1"
ARTIFACT_KIND = "marulho_language_brain_installed_generation_evidence"


@dataclass(frozen=True)
class BrainInstalledGenerationEvidenceConfig:
    min_case_pass_rate: float = 1.0
    generation_repetition_penalty: float = 1.15
    generation_no_repeat_ngram_size: int = 3
    seed: int = 20260706


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: str | Path | None) -> tuple[str, str]:
    if path is None:
        return DEFAULT_CORPUS, "default_inline"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Brain generation source is empty: {resolved}")
    return text, str(resolved)


def _status_read_mutates(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return (
        int(before.get("token_count", 0) or 0) != int(after.get("token_count", 0) or 0)
        or int(before.get("queued_tokens", 0) or 0)
        != int(after.get("queued_tokens", 0) or 0)
        or str(before.get("active_language_path") or "")
        != str(after.get("active_language_path") or "")
    )


def _language_state(brain: MarulhoBrain) -> Mapping[str, Any]:
    state = brain.export_state()
    language_state = state.get("language_model")
    if not isinstance(language_state, Mapping):
        raise RuntimeError("MARULHO language model runtime is not installed")
    return language_state


def _language_tokenizer(language_state: Mapping[str, Any]) -> ByteLevelLanguageTokenizer:
    tokenizer_state = language_state.get("tokenizer")
    if not isinstance(tokenizer_state, Mapping):
        raise RuntimeError("Installed language runtime has no checkpointed tokenizer")
    return ByteLevelLanguageTokenizer.load_state_dict(tokenizer_state)


class _BrainGenerationAdapter:
    def __init__(
        self,
        brain: MarulhoBrain,
        tokenizer: ByteLevelLanguageTokenizer,
    ) -> None:
        self._brain = brain
        self._tokenizer = tokenizer
        self.config = SimpleNamespace(active_language_path="marulho_lm_head")

    def generate(
        self,
        prompt_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> dict[str, Any]:
        del eos_id
        flat_prompt = [
            int(token_id)
            for token_id in prompt_ids.detach().cpu().reshape(-1).to(torch.long).tolist()
        ]
        prompt_text = self._tokenizer.decode(flat_prompt)
        generation = self._brain.generate(
            prompt_text,
            max_tokens=int(max_new_tokens),
            generation_repetition_penalty=float(repetition_penalty),
            generation_no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
        generated_ids = [
            int(token_id) for token_id in list(generation.get("generated_token_ids") or [])
        ]
        if not generated_ids:
            generated_ids = list(flat_prompt)
        return {
            "surface": "marulho_brain_generation_adapter.v1",
            "brain_generation_surface": generation.get("surface"),
            "generated_ids": torch.tensor([generated_ids], dtype=torch.long),
            "new_token_count": int(generation.get("emitted_tokens", 0) or 0),
            "active_language_path": generation.get("active_language_path"),
            "external_llm_used": bool(generation.get("external_llm_used", False)),
            "owned_by_marulho": bool(generation.get("owned_by_marulho", False)),
            "loads_external_checkpoint": bool(
                generation.get("loads_external_checkpoint", False)
            ),
            "generation_decode": dict(generation.get("generation_decode") or {}),
        }


def _prompt_case_from_arg(
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


def build_language_brain_installed_generation_evidence(
    *,
    output_path: str | Path,
    brain_checkpoint_path: str | Path,
    source_path: str | Path | None = None,
    prompt_cases: Sequence[LanguageGenerationPromptCase] | None = None,
    config: BrainInstalledGenerationEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Restore an installed brain checkpoint and score generation through MarulhoBrain."""

    cfg = config or BrainInstalledGenerationEvidenceConfig()
    output = Path(output_path)
    report: dict[str, Any] = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "partial",
        "report_status": "partial",
        "output_path": str(output),
        "brain_checkpoint_path": str(brain_checkpoint_path),
        "runtime_owner": "MarulhoBrain",
        "cuda_available": bool(torch.cuda.is_available()),
        "config": asdict(cfg),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "service_owned_cognition": False,
        "status_read_mutation": False,
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }
    exception: BaseException | None = None
    try:
        torch.manual_seed(int(cfg.seed))
        checkpoint = Path(brain_checkpoint_path)
        if not checkpoint.is_file():
            report.update(
                {
                    "status": "blocked_brain_installed_generation_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_checkpoint_missing",
                }
            )
            return report
        brain = MarulhoBrain.load(checkpoint)
        restored_status = brain.status()
        language_model = (
            restored_status.get("language_model")
            if isinstance(restored_status.get("language_model"), Mapping)
            else {}
        )
        if not bool(language_model.get("available", False)):
            report.update(
                {
                    "status": "blocked_brain_installed_generation_evidence",
                    "report_status": "partial",
                    "failure_reason": "brain_language_runtime_missing",
                }
            )
            return report
        before_status = brain.status()
        after_status = brain.status()
        status_read_mutation = _status_read_mutates(before_status, after_status)
        language_state = _language_state(brain)
        tokenizer = _language_tokenizer(language_state)
        tokenizer_hash_matches_installed = (
            tokenizer.vocabulary_hash() == language_model.get("tokenizer_hash")
        )
        source_text, source = _read_text(source_path)
        cases = tuple(prompt_cases or default_generation_coherence_prompt_cases(source_text))
        adapter = _BrainGenerationAdapter(brain, tokenizer)
        coherence = run_language_generation_coherence_report(
            adapter,  # type: ignore[arg-type]
            tokenizer,
            prompt_cases=cases,
            min_case_pass_rate=float(cfg.min_case_pass_rate),
            checkpoint_path=checkpoint,
            generation_repetition_penalty=float(cfg.generation_repetition_penalty),
            generation_no_repeat_ngram_size=int(cfg.generation_no_repeat_ngram_size),
        )
        generation_status = brain.status()
        gate = (
            coherence.get("promotion_gate")
            if isinstance(coherence.get("promotion_gate"), Mapping)
            else {}
        )
        summary = (
            coherence.get("summary")
            if isinstance(coherence.get("summary"), Mapping)
            else {}
        )
        restored_ok = (
            restored_status.get("active_language_path") == "marulho_lm_head"
            and bool(language_model.get("checkpointed_language_components", False))
        )
        recorded_generation = bool(coherence.get("cases"))
        success = (
            restored_ok
            and not bool(status_read_mutation)
            and bool(tokenizer_hash_matches_installed)
            and bool(recorded_generation)
            and coherence.get("active_language_path") == "marulho_lm_head"
            and coherence.get("owned_by_marulho") is True
            and coherence.get("external_llm_used") is False
            and gate.get("promotes_runtime_claim") is False
            and gate.get("promotes_generation_quality_claim") is False
        )
        report.update(
            {
                "status": (
                    "final" if success else "blocked_brain_installed_generation_evidence"
                ),
                "report_status": "final" if success else "partial",
                "failure_reason": None if success else "evidence_gate_not_satisfied",
                "brain_checkpoint": {
                    "path": str(checkpoint),
                    "sha256": _sha256_file(checkpoint),
                    "restore_verified": bool(restored_ok),
                    "active_language_path": restored_status.get("active_language_path"),
                    "language_model": dict(language_model),
                },
                "status_read": {
                    "surface": "marulho_brain_status_read_check.v1",
                    "mutates_runtime_state": bool(status_read_mutation),
                    "before_token_count": int(before_status.get("token_count", 0) or 0),
                    "after_token_count": int(after_status.get("token_count", 0) or 0),
                    "active_language_path": before_status.get("active_language_path"),
                },
                "tokenizer": {
                    "surface": "marulho_brain_generation_tokenizer_check.v1",
                    "tokenizer_hash": tokenizer.vocabulary_hash(),
                    "tokenizer_hash_matches_installed_runtime": bool(
                        tokenizer_hash_matches_installed
                    ),
                    "used_for_prompt_scoring_only": True,
                },
                "source": {
                    "path": source,
                    "character_count": len(source_text),
                },
                "generation_coherence": dict(coherence),
                "generation_summary": dict(summary),
                "post_generation_status": {
                    "surface": "marulho_brain_post_generation_status_summary.v1",
                    "active_language_path": generation_status.get("active_language_path"),
                    "device": generation_status.get("device"),
                    "language_model": dict(generation_status.get("language_model") or {}),
                    "last_generation": dict(generation_status.get("last_generation") or {}),
                    "last_trace": dict(generation_status.get("last_trace") or {}),
                },
                "active_language_path": generation_status.get("active_language_path"),
                "status_read_mutation": bool(status_read_mutation),
            }
        )
        report["promotion_gate"] = {
            "surface": "marulho_language_brain_installed_generation_gate.v1",
            "loaded_installed_brain_checkpoint": True,
            "brain_checkpoint_restore_verified": bool(restored_ok),
            "batch_tokenizer_matches_installed_runtime": bool(
                tokenizer_hash_matches_installed
            ),
            "generation_runs_through_marulho_brain": bool(recorded_generation),
            "generation_coherence_available": bool(
                gate.get("generation_coherence_available", False)
            ),
            "grounded_prompt_suite_available": bool(
                gate.get("grounded_prompt_suite_available", False)
            ),
            "case_count": int(summary.get("case_count", 0) or 0),
            "passed_case_count": int(summary.get("passed_case_count", 0) or 0),
            "case_pass_rate": float(summary.get("case_pass_rate", 0.0) or 0.0),
            "status_read_mutation_absent": not bool(status_read_mutation),
            "external_llm_absent": not bool(report.get("external_llm_used", False)),
            "service_owned_cognition_absent": not bool(
                report.get("service_owned_cognition", False)
            ),
            "ready_for_runtime_claim_review": False,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
        }
        return report
    except BaseException as exc:  # pragma: no cover - report persistence guard
        exception = exc
        report.update(
            {
                "status": "exception",
                "report_status": "exception",
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        return report
    finally:
        if exception is not None:
            report.setdefault(
                "promotion_gate",
                {
                    "surface": "marulho_language_brain_installed_generation_gate.v1",
                    "ready_for_runtime_claim_review": False,
                    "promotes_runtime_claim": False,
                    "promotes_generation_quality_claim": False,
                },
            )
        write_json_report_with_readme(output, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--brain-checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--prompt-case",
        action="append",
        default=[],
        help=(
            "Prompt case as prompt|max_new_tokens|min_prefix_chars|min_prefix_fraction. "
            "Defaults to the standard MARULHO prompt suite."
        ),
    )
    parser.add_argument("--min-case-pass-rate", type=float, default=1.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260706)
    args = parser.parse_args()

    source_text, _source = _read_text(args.source)
    cases = (
        tuple(_prompt_case_from_arg(value, source_text=source_text) for value in args.prompt_case)
        if args.prompt_case
        else default_generation_coherence_prompt_cases(source_text)
    )
    report = build_language_brain_installed_generation_evidence(
        output_path=args.output,
        brain_checkpoint_path=args.brain_checkpoint,
        source_path=args.source,
        prompt_cases=cases,
        config=BrainInstalledGenerationEvidenceConfig(
            min_case_pass_rate=float(args.min_case_pass_rate),
            generation_repetition_penalty=max(
                1.0,
                float(args.generation_repetition_penalty),
            ),
            generation_no_repeat_ngram_size=max(
                0,
                int(args.generation_no_repeat_ngram_size),
            ),
            seed=int(args.seed),
        ),
    )
    return 0 if report.get("report_status") == "final" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
