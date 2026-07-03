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


def _state_block_parameter_count(config: LanguageModelConfig) -> int:
    embedding_dim = int(config.embedding_dim)
    state_dim = int(config.state_dim)
    return int(
        (8 * embedding_dim * state_dim)
        + (3 * state_dim * state_dim)
        + (13 * state_dim)
        + embedding_dim
    )


def _expert_hidden_dim(config: LanguageModelConfig) -> int:
    hidden = int(config.expert_hidden_dim)
    return hidden if hidden > 0 else int(config.state_dim) * 2


def _expert_parameter_count_per_column(config: LanguageModelConfig) -> int:
    if int(config.expert_count) <= 0:
        return 0
    state_dim = int(config.state_dim)
    hidden_dim = _expert_hidden_dim(config)
    return int((2 * state_dim * hidden_dim) + hidden_dim + state_dim)


def estimate_language_model_parameters(
    config: LanguageModelConfig,
    *,
    dtype_bytes: int = 2,
) -> dict[str, Any]:
    vocab_size = int(config.vocab_size)
    embedding_dim = int(config.embedding_dim)
    state_dim = int(config.state_dim)
    expert_count = max(0, int(config.expert_count))
    active_expert_count = max(1, int(config.active_expert_count))
    route_candidate_count = (
        expert_count
        if int(config.route_candidate_count) <= 0
        else min(expert_count, int(config.route_candidate_count))
    )
    token_embedding = int(vocab_size * embedding_dim)
    state_block = _state_block_parameter_count(config)
    route_bank = int((expert_count * state_dim) + expert_count) if expert_count else 0
    per_expert = _expert_parameter_count_per_column(config)
    expert_total = int(expert_count * per_expert)
    lm_head = int((state_dim * vocab_size) + vocab_size)
    total = int(token_embedding + state_block + route_bank + expert_total + lm_head)
    active_experts = min(active_expert_count, max(0, route_candidate_count))
    route_candidate_scored_parameters = (
        int(route_candidate_count * (state_dim + 1)) if expert_count else 0
    )
    active_parameters_per_token = int(
        embedding_dim
        + state_block
        + lm_head
        + (active_experts * per_expert)
        + route_candidate_scored_parameters
    )
    dtype_size = max(1, int(dtype_bytes))
    return {
        "surface": "marulho_language_model_parameter_estimate.v1",
        "config": asdict(config),
        "total_parameters": total,
        "parameter_breakdown": {
            "token_embedding": token_embedding,
            "selective_spiking_state_block": state_block,
            "route_bank": route_bank,
            "routed_experts": expert_total,
            "lm_head_dense_vocab": lm_head,
        },
        "expert_parameters_per_column": int(per_expert),
        "active_parameters_per_token_estimate": active_parameters_per_token,
        "active_parameter_fraction_estimate": (
            float(active_parameters_per_token) / float(total) if total > 0 else 0.0
        ),
        "active_expert_count_per_token": int(active_experts),
        "route_candidate_count": int(route_candidate_count),
        "route_candidate_rows_scored_per_token": int(route_candidate_count),
        "dense_vocab_head_active": True,
        "sampled_or_adaptive_vocab_xent_present": False,
        "parameter_memory_mib": float(total * dtype_size / (1024 * 1024)),
        "parameter_memory_mib_fp16": float(total * 2 / (1024 * 1024)),
        "parameter_memory_mib_fp32": float(total * 4 / (1024 * 1024)),
        "adamw_train_state_mib_fp32_estimate": float(total * 16 / (1024 * 1024)),
    }


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
