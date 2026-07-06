"""Scale-ladder definitions and estimates for the MARULHO LM head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)
from marulho.training.language_model_parameters import estimate_language_model_parameters


SURFACE = "marulho_language_scale_ladder.v1"
ARTIFACT_KIND = "marulho_language_scale_ladder"


@dataclass(frozen=True)
class LanguageScaleLadderEntry:
    name: str
    target_class: str
    purpose: str
    config: LanguageModelConfig
    min_total_parameters: int
    max_total_parameters: int
    required_evidence: tuple[str, ...]
    instantiate_by_default: bool = False


def default_language_scale_ladder() -> tuple[LanguageScaleLadderEntry, ...]:
    return (
        LanguageScaleLadderEntry(
            name="small_fixture",
            target_class="ci_correctness_fixture",
            purpose="deterministic encode/loss/checkpoint/rollback smoke",
            config=LanguageModelConfig(
                vocab_size=256,
                embedding_dim=16,
                state_dim=24,
                expert_count=4,
                active_expert_count=1,
                route_candidate_count=2,
                expert_hidden_dim=48,
            ),
            min_total_parameters=1,
            max_total_parameters=1_000_000,
            required_evidence=(
                "deterministic_tokenizer",
                "heldout_loss",
                "checkpoint_round_trip",
                "rollback",
            ),
            instantiate_by_default=True,
        ),
        LanguageScaleLadderEntry(
            name="nord_140m_class",
            target_class="140M-class",
            purpose="first meaningful sparse LM comparison class",
            config=LanguageModelConfig(
                vocab_size=32_768,
                embedding_dim=512,
                state_dim=768,
                expert_count=40,
                active_expert_count=2,
                route_candidate_count=8,
                expert_hidden_dim=1_536,
            ),
            min_total_parameters=100_000_000,
            max_total_parameters=200_000_000,
            required_evidence=(
                "train_tokens",
                "heldout_loss",
                "heldout_perplexity",
                "active_compute_per_token",
                "replay_retention",
            ),
        ),
        LanguageScaleLadderEntry(
            name="growth_500m_class",
            target_class="500M-class",
            purpose="growth and routing stability class",
            config=LanguageModelConfig(
                vocab_size=49_152,
                embedding_dim=768,
                state_dim=1_280,
                expert_count=64,
                active_expert_count=4,
                route_candidate_count=16,
                expert_hidden_dim=2_560,
            ),
            min_total_parameters=400_000_000,
            max_total_parameters=650_000_000,
            required_evidence=(
                "route_saturation",
                "active_columns",
                "memory_footprint",
                "forgetting_metrics",
                "checkpoint_restore_fidelity",
            ),
        ),
        LanguageScaleLadderEntry(
            name="neuronspark_0_9b_class",
            target_class="0.9B-class",
            purpose="NeuronSpark-scale comparison class",
            config=LanguageModelConfig(
                vocab_size=49_152,
                embedding_dim=1_024,
                state_dim=1_536,
                expert_count=80,
                active_expert_count=4,
                route_candidate_count=16,
                expert_hidden_dim=3_072,
            ),
            min_total_parameters=800_000_000,
            max_total_parameters=1_100_000_000,
            required_evidence=(
                "long_run_stability",
                "kernel_coverage",
                "restore_fidelity",
                "generation_quality_review",
            ),
        ),
        LanguageScaleLadderEntry(
            name="research_2b_plus_class",
            target_class="2B+ research",
            purpose="larger recurrent sparse research class when kernels and memory allow",
            config=LanguageModelConfig(
                vocab_size=65_536,
                embedding_dim=1_536,
                state_dim=2_048,
                expert_count=128,
                active_expert_count=4,
                route_candidate_count=16,
                expert_hidden_dim=4_096,
            ),
            min_total_parameters=2_000_000_000,
            max_total_parameters=3_000_000_000,
            required_evidence=(
                "memory_budget",
                "throughput",
                "long_run_stability",
                "checkpoint_restore_fidelity",
                "generation_quality_review",
            ),
        ),
    )


def _status_for_entry(
    entry: LanguageScaleLadderEntry,
    estimate: Mapping[str, Any],
) -> dict[str, Any]:
    total = int(estimate["total_parameters"])
    in_class = int(entry.min_total_parameters) <= total <= int(entry.max_total_parameters)
    return {
        "surface": "marulho_language_scale_ladder_entry_gate.v1",
        "target_class": entry.target_class,
        "total_parameters_in_target_range": bool(in_class),
        "min_total_parameters": int(entry.min_total_parameters),
        "max_total_parameters": int(entry.max_total_parameters),
        "missing_required_evidence": list(entry.required_evidence),
        "trained": False,
        "promoted": False,
        "claim": "configuration_defined_not_trained",
    }


def build_language_scale_ladder_report(
    *,
    entries: Sequence[LanguageScaleLadderEntry] | None = None,
    output_path: str | Path | None = None,
    smoke_model: MarulhoLanguageModel | None = None,
    smoke_tokenizer: ByteLevelLanguageTokenizer | None = None,
    smoke_eval_batches: Sequence[LanguageBatch] = (),
) -> dict[str, Any]:
    ladder = tuple(entries or default_language_scale_ladder())
    entry_reports: list[dict[str, Any]] = []
    for entry in ladder:
        estimate = estimate_language_model_parameters(entry.config)
        entry_reports.append(
            {
                "name": entry.name,
                "target_class": entry.target_class,
                "purpose": entry.purpose,
                "config": asdict(entry.config),
                "estimate": estimate,
                "required_evidence": list(entry.required_evidence),
                "instantiate_by_default": bool(entry.instantiate_by_default),
                "gate": _status_for_entry(entry, estimate),
            }
        )
    smoke_report: dict[str, Any] | None = None
    if smoke_model is not None and smoke_tokenizer is not None and smoke_eval_batches:
        eval_report = evaluate_language_model(smoke_model, smoke_eval_batches)
        generation = smoke_model.generate(
            smoke_eval_batches[0].input_ids[0],
            max_new_tokens=2,
            eos_id=smoke_tokenizer.eos_id,
        )
        smoke_report = {
            "surface": "marulho_language_scale_ladder_smoke_run.v1",
            "ran": True,
            "model_config": asdict(smoke_model.config),
            "tokenizer_hash": smoke_tokenizer.vocabulary_hash(),
            "heldout_loss": eval_report["heldout_loss"],
            "heldout_perplexity": eval_report["heldout_perplexity"],
            "eval_token_count": eval_report["eval_token_count"],
            "generated_token_count": int(generation["new_token_count"]),
            "external_llm_used": False,
            "promotes_scale_claim": False,
        }
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": None if output_path is None else str(output_path),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "entry_count": len(entry_reports),
        "entries": entry_reports,
        "smoke_fixture": smoke_report
        or {
            "surface": "marulho_language_scale_ladder_smoke_run.v1",
            "ran": False,
        },
        "promotion_gate": {
            "scale_ladder_defined": True,
            "large_ladders_instantiated": False,
            "large_ladders_trained": False,
            "frontier_competitiveness_claimed": False,
            "requires_long_run_evidence": True,
            "requires_kernel_coverage": True,
            "status": "defined_not_promoted",
        },
    }
    if output_path is not None:
        write_json_report_with_readme(output_path, report)
    return report


def build_smoke_fixture_report(output_path: str | Path) -> dict[str, Any]:
    tokenizer = ByteLevelLanguageTokenizer()
    split = build_language_model_splits(
        [
            "scale ladder smoke verifies heldout loss and owned generation. " * 4,
            "large ladder classes stay estimates until long evidence exists. " * 4,
        ],
        tokenizer,
        sequence_length=10,
        eval_fraction=0.25,
    )
    config = default_language_scale_ladder()[0].config
    model = MarulhoLanguageModel(
        LanguageModelConfig(
            **{
                **asdict(config),
                "vocab_size": tokenizer.vocab_size,
            }
        )
    )
    return build_language_scale_ladder_report(
        output_path=output_path,
        smoke_model=model,
        smoke_tokenizer=tokenizer,
        smoke_eval_batches=split.eval,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-smoke-fixture", action="store_true")
    args = parser.parse_args()
    if bool(args.include_smoke_fixture):
        build_smoke_fixture_report(args.output)
    else:
        build_language_scale_ladder_report(output_path=args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
