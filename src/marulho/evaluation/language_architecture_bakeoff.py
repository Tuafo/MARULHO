"""Matched MARULHO base-LM architecture bakeoff with explicit branch decisions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Sequence

import torch

from marulho.evaluation.language_training_experiment import (
    DEFAULT_CORPUS,
    LanguageTrainingExperimentConfig,
    run_language_training_experiment,
)
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    run_language_generation_coherence_report,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_language_architecture_bakeoff.v1"
ARTIFACT_KIND = "marulho_language_architecture_bakeoff"
REFERENCE_VARIANT = "selective_spiking_routed"
DEFAULT_VARIANTS = (
    REFERENCE_VARIANT,
    "selective_spiking_dense",
    "selective_continuous_routed",
    "gru_routed",
    "gru_dense",
    "transformer_dense",
)


@dataclass(frozen=True)
class LanguageArchitectureBakeoffConfig:
    training: LanguageTrainingExperimentConfig = field(
        default_factory=lambda: LanguageTrainingExperimentConfig(
            embedding_dim=64,
            state_dim=128,
            expert_count=8,
            active_expert_count=2,
            route_candidate_count=4,
            expert_hidden_dim=192,
            sequence_length=128,
            stride=64,
            batch_size=8,
            max_train_batches=256,
            max_eval_batches=128,
            train_epochs=1,
            learning_rate=1e-3,
            generation_tokens=128,
            sustained_target_tokens=8192,
            sustained_timeout_seconds=600.0,
        )
    )
    variants: tuple[str, ...] = DEFAULT_VARIANTS
    epoch_budgets: tuple[int, ...] = (1, 4)
    seeds: tuple[int, ...] = (1337,)
    heldout_prompt_count: int = 4
    heldout_prompt_characters: int = 32
    rescore_repetition_penalty: float = 1.1
    rescore_no_repeat_ngram_size: int = 3


def _read_corpus(path: str | Path | None) -> tuple[str, str]:
    if path is None:
        return DEFAULT_CORPUS, "default_inline"
    corpus_path = Path(path)
    corpus = corpus_path.read_text(encoding="utf-8")
    if not corpus.strip():
        raise ValueError(f"Architecture bakeoff corpus is empty: {corpus_path}")
    return corpus, str(corpus_path)


def _heldout_prompts(
    corpus: str,
    *,
    count: int,
    prompt_characters: int,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    split_offset = max(1, min(len(corpus) - 1, int(len(corpus) * 0.80)))
    training_prefix = corpus[:split_offset]
    heldout = corpus[split_offset:]
    width = max(8, int(prompt_characters))
    desired = max(1, int(count))
    prompts: list[str] = []
    overlap_flags: list[bool] = []
    max_start = max(0, len(heldout) - width)
    primary_positions = (
        [0]
        if desired == 1 or max_start <= 0
        else [round(index * max_start / float(desired - 1)) for index in range(desired)]
    )
    fallback_step = max(1, width // 2)
    candidate_positions = primary_positions + list(
        range(0, max_start + 1, fallback_step)
    )
    for start in candidate_positions:
        prompt = heldout[start : start + width]
        if not prompt.strip() or prompt in prompts:
            continue
        overlaps_training = prompt in training_prefix
        if overlaps_training and any(not flag for flag in overlap_flags):
            continue
        prompts.append(prompt)
        overlap_flags.append(overlaps_training)
        if len(prompts) >= desired:
            break
    if not prompts:
        prompt = heldout[:width] or corpus[:width]
        prompts = [prompt]
        overlap_flags = [prompt in training_prefix]
    return tuple(prompts), {
        "surface": "marulho_language_bakeoff_heldout_prompt_bank.v1",
        "corpus_character_split_offset": int(split_offset),
        "prompt_count": len(prompts),
        "prompt_characters": int(width),
        "prompts": list(prompts),
        "prompt_present_in_training_prefix": list(overlap_flags),
        "all_prompts_absent_from_training_prefix": not any(overlap_flags),
        "selection_policy": "deterministic_final_20_percent_quantile_windows",
    }


def _variant_training_config(
    base: LanguageTrainingExperimentConfig,
    *,
    variant: str,
    epochs: int,
) -> LanguageTrainingExperimentConfig:
    common = {"train_epochs": max(1, int(epochs))}
    if variant == REFERENCE_VARIANT:
        return replace(base, state_core="selective_spiking", **common)
    if variant == "selective_spiking_dense":
        return replace(
            base,
            state_core="selective_spiking",
            expert_count=0,
            active_expert_count=1,
            route_candidate_count=0,
            expert_hidden_dim=0,
            **common,
        )
    if variant == "selective_continuous_routed":
        return replace(base, state_core="selective_continuous", **common)
    if variant == "gru_routed":
        return replace(base, state_core="gru", **common)
    if variant == "gru_dense":
        return replace(
            base,
            state_core="gru",
            expert_count=0,
            active_expert_count=1,
            route_candidate_count=0,
            expert_hidden_dim=0,
            **common,
        )
    if variant == "transformer_dense":
        return replace(
            base,
            state_core="transformer",
            recurrent_gradient_horizon=0,
            tie_embeddings=bool(base.embedding_dim == base.state_dim),
            expert_count=0,
            active_expert_count=1,
            route_candidate_count=0,
            expert_hidden_dim=0,
            **common,
        )
    raise ValueError(f"Unsupported architecture bakeoff variant: {variant!r}")


def _run_summary(
    *,
    variant: str,
    seed: int,
    epochs: int,
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    quality = dict(report.get("generation_quality_after") or {})
    inventory = dict(report.get("parameter_inventory") or {})
    return {
        "variant": variant,
        "state_core": report.get("state_core"),
        "seed": int(seed),
        "train_epochs": int(epochs),
        "report_path": str(report_path),
        "checkpoint_path": report.get("checkpoint_path"),
        "train_split_hash": report.get("split", {}).get("train_split_hash"),
        "eval_split_hash": report.get("split", {}).get("eval_split_hash"),
        "training_tokens": int(report.get("training", {}).get("token_count", 0) or 0),
        "training_tokens_per_second": float(
            report.get("training", {}).get("tokens_per_second", 0.0) or 0.0
        ),
        "heldout_loss_before": float(report["eval_before"]["heldout_loss"]),
        "heldout_loss_after": float(report["eval_after"]["heldout_loss"]),
        "heldout_loss_delta": float(report["language_delta"]["heldout_loss_delta"]),
        "heldout_perplexity_after": float(report["eval_after"]["heldout_perplexity"]),
        "mean_source_prefix_match_chars": float(
            quality.get("mean_source_prefix_match_chars", 0.0) or 0.0
        ),
        "next_character_match_rate": float(
            quality.get("next_character_match_rate", 0.0) or 0.0
        ),
        "mean_distinct_bigram_fraction": float(
            quality.get("mean_distinct_bigram_fraction", 0.0) or 0.0
        ),
        "total_parameters": int(inventory.get("total_parameters", 0) or 0),
        "state_core_parameters": int(inventory.get("state_core_parameters", 0) or 0),
        "routed_expert_parameters": int(
            inventory.get("routed_expert_parameters", 0) or 0
        ),
        "sustained_success": bool(report.get("sustained_summary", {}).get("success")),
        "sustained_tokens_per_second": float(
            report.get("sustained_summary", {}).get("tokens_per_second", 0.0) or 0.0
        ),
        "external_llm_used": bool(report.get("external_llm_used", True)),
    }


def _aggregate_final_budget(
    runs: Sequence[dict[str, Any]],
    *,
    variants: Sequence[str],
    final_epoch_budget: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in variants:
        selected = [
            run
            for run in runs
            if run["variant"] == variant
            and int(run["train_epochs"]) == int(final_epoch_budget)
        ]
        if not selected:
            continue
        losses = [float(run["heldout_loss_after"]) for run in selected]
        deltas = [float(run["heldout_loss_delta"]) for run in selected]
        prefixes = [float(run["mean_source_prefix_match_chars"]) for run in selected]
        throughputs = [float(run["training_tokens_per_second"]) for run in selected]
        summaries.append(
            {
                "variant": variant,
                "run_count": len(selected),
                "mean_heldout_loss_after": fmean(losses),
                "heldout_loss_after_stddev": pstdev(losses) if len(losses) > 1 else 0.0,
                "mean_heldout_loss_delta": fmean(deltas),
                "mean_source_prefix_match_chars": fmean(prefixes),
                "mean_training_tokens_per_second": fmean(throughputs),
                "mean_total_parameters": fmean(
                    [float(run["total_parameters"]) for run in selected]
                ),
                "all_sustained_success": all(
                    bool(run["sustained_success"]) for run in selected
                ),
                "external_llm_absent": all(
                    not bool(run["external_llm_used"]) for run in selected
                ),
            }
        )
    return sorted(
        summaries,
        key=lambda row: (
            float(row["mean_heldout_loss_after"]),
            -float(row["mean_source_prefix_match_chars"]),
        ),
    )


def _branch_decision(
    summaries: Sequence[dict[str, Any]],
    *,
    seed_count: int,
) -> dict[str, Any]:
    if not summaries:
        return {
            "branch": "insufficient_bakeoff_evidence",
            "winner": None,
            "reason": "no final-budget architecture summaries",
        }
    winner = dict(summaries[0])
    if float(winner["mean_heldout_loss_delta"]) >= 0.0:
        branch = "redesign_training_setup_before_scaling"
        reason = "no winning architecture improved heldout loss"
    else:
        branch = {
            "selective_spiking_routed": "scale_selective_spiking_routed",
            "selective_spiking_dense": "redesign_remove_routing",
            "selective_continuous_routed": "redesign_toward_continuous_or_hybrid",
            "gru_routed": "redesign_toward_gru_or_hybrid",
            "gru_dense": "redesign_toward_gru_remove_routing",
            "transformer_dense": "retire_recurrent_language_base_scale_transformer",
        }[str(winner["variant"])]
        reason = "lowest final-budget heldout loss; generation quality is tie-break only"
    return {
        "surface": "marulho_language_architecture_branch_decision.v1",
        "branch": branch,
        "winner": winner["variant"],
        "reason": reason,
        "confidence": "multi_seed" if int(seed_count) > 1 else "single_seed_provisional",
        "throughput_is_primary_selector": False,
        "prompt_pass_count_is_primary_selector": False,
        "selection_order": [
            "lowest_mean_heldout_loss_after",
            "highest_mean_source_prefix_match_chars_tiebreak",
        ],
    }


def _rescore_final_checkpoints(
    runs: Sequence[dict[str, Any]],
    *,
    output: Path,
    corpus: str,
    prompts: Sequence[str],
    final_epoch_budget: int,
    generation_tokens: int,
    device_name: str,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
) -> list[dict[str, Any]]:
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else (
            "cpu" if device_name == "auto" else device_name
        )
    )
    cases = tuple(
        LanguageGenerationPromptCase(
            prompt_text=str(prompt),
            source_text=corpus,
            max_new_tokens=max(8, int(generation_tokens)),
        )
        for prompt in prompts
    )
    rescored: list[dict[str, Any]] = []
    for run in runs:
        if int(run["train_epochs"]) != int(final_epoch_budget):
            continue
        checkpoint = Path(str(run["checkpoint_path"]))
        report_path = output.with_name(
            f"{output.stem}-{run['variant']}-seed{run['seed']}-epochs{final_epoch_budget}"
            "-diverse-heldout.json"
        )
        model, tokenizer, _metadata = load_language_model_checkpoint(
            checkpoint,
            map_location="cpu",
        )
        model = model.to(device)
        coherence = run_language_generation_coherence_report(
            model,
            tokenizer,
            prompt_cases=cases,
            checkpoint_path=checkpoint,
            output_path=report_path,
            generation_repetition_penalty=max(1.0, float(repetition_penalty)),
            generation_no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
        )
        rescored.append(
            {
                "variant": run["variant"],
                "seed": int(run["seed"]),
                "train_epochs": int(final_epoch_budget),
                "checkpoint_path": str(checkpoint),
                "report_path": str(report_path),
                "summary": coherence["summary"],
                "external_llm_used": bool(coherence["external_llm_used"]),
                "owned_by_marulho": bool(coherence["owned_by_marulho"]),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rescored


def run_language_architecture_bakeoff(
    *,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    config: LanguageArchitectureBakeoffConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageArchitectureBakeoffConfig()
    variants = tuple(dict.fromkeys(str(value) for value in cfg.variants))
    budgets = tuple(sorted({max(1, int(value)) for value in cfg.epoch_budgets}))
    seeds = tuple(dict.fromkeys(int(value) for value in cfg.seeds))
    if not variants:
        raise ValueError("Architecture bakeoff requires at least one variant")
    if not seeds:
        raise ValueError("Architecture bakeoff requires at least one seed")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus, corpus_source = _read_corpus(corpus_path)
    prompts, prompt_bank = _heldout_prompts(
        corpus,
        count=int(cfg.heldout_prompt_count),
        prompt_characters=int(cfg.heldout_prompt_characters),
    )
    runs: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            for epochs in budgets:
                torch.manual_seed(int(seed))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(seed))
                    torch.cuda.empty_cache()
                run_config = _variant_training_config(
                    cfg.training,
                    variant=variant,
                    epochs=int(epochs),
                )
                run_path = output.with_name(
                    f"{output.stem}-{variant}-seed{seed}-epochs{epochs}.json"
                )
                print(
                    f"[bakeoff] start variant={variant} seed={seed} epochs={epochs}",
                    flush=True,
                )
                if run_path.exists():
                    run_report = json.loads(run_path.read_text(encoding="utf-8"))
                    expected_config = asdict(run_config)
                    if (
                        run_report.get("config") != expected_config
                        or run_report.get("corpus", {}).get("source") != corpus_source
                        or int(run_report.get("corpus", {}).get("character_count", -1))
                        != len(corpus)
                    ):
                        raise RuntimeError(
                            f"Refusing to reuse mismatched bakeoff arm: {run_path}"
                        )
                    print(f"[bakeoff] reuse report={run_path}", flush=True)
                else:
                    run_report = run_language_training_experiment(
                        output_path=run_path,
                        corpus_path=corpus_path,
                        prompts=prompts,
                        config=run_config,
                    )
                runs.append(
                    _run_summary(
                        variant=variant,
                        seed=int(seed),
                        epochs=int(epochs),
                        report_path=run_path,
                        report=run_report,
                    )
                )
                print(
                    "[bakeoff] complete "
                    f"variant={variant} seed={seed} epochs={epochs} "
                    f"heldout_loss={run_report['eval_after']['heldout_loss']:.6f}",
                    flush=True,
                )

    final_budget = max(budgets)
    summaries = _aggregate_final_budget(
        runs,
        variants=variants,
        final_epoch_budget=final_budget,
    )
    diverse_heldout_rescore = _rescore_final_checkpoints(
        runs,
        output=output,
        corpus=corpus,
        prompts=prompts,
        final_epoch_budget=final_budget,
        generation_tokens=int(cfg.training.generation_tokens),
        device_name=str(cfg.training.device),
        repetition_penalty=float(cfg.rescore_repetition_penalty),
        no_repeat_ngram_size=int(cfg.rescore_no_repeat_ngram_size),
    )
    train_hashes = {str(run["train_split_hash"]) for run in runs}
    eval_hashes = {str(run["eval_split_hash"]) for run in runs}
    token_counts_by_budget = {
        str(budget): sorted(
            {
                int(run["training_tokens"])
                for run in runs
                if int(run["train_epochs"]) == int(budget)
            }
        )
        for budget in budgets
    }
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "corpus": {
            "source": corpus_source,
            "character_count": len(corpus),
            "sha256": hashlib.sha256(corpus.encode("utf-8")).hexdigest(),
        },
        "config": asdict(cfg),
        "heldout_prompt_bank": prompt_bank,
        "fairness": {
            "surface": "marulho_language_architecture_bakeoff_fairness.v1",
            "same_corpus": True,
            "same_tokenizer": True,
            "same_optimizer_policy": True,
            "same_shape_except_dense_routing_ablation": True,
            "same_train_split": len(train_hashes) == 1,
            "same_eval_split": len(eval_hashes) == 1,
            "train_split_hashes": sorted(train_hashes),
            "eval_split_hashes": sorted(eval_hashes),
            "training_token_counts_by_epoch_budget": token_counts_by_budget,
            "parameter_counts_recorded": all(
                int(run["total_parameters"]) > 0 for run in runs
            ),
            "throughput_used_as_quality_proxy": False,
        },
        "quality_curves": runs,
        "final_budget": {
            "train_epochs": int(final_budget),
            "variant_summaries": summaries,
        },
        "diverse_heldout_rescore": diverse_heldout_rescore,
        "decision": _branch_decision(summaries, seed_count=len(seeds)),
        "experiment_review": {
            "breaking_architecture_comparison": True,
            "repair_sweep": False,
            "base_model_quality_first": True,
            "reintroduce_continual_learning_after_base_quality": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--tokenizer-kind", choices=("byte", "bpe"), default="byte")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--epoch-budget", action="append", type=int, default=[])
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--state-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-context-length", type=int, default=256)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=192)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=256)
    parser.add_argument("--max-eval-batches", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--sustained-target-tokens", type=int, default=8192)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rescore-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--rescore-no-repeat-ngram-size", type=int, default=3)
    args = parser.parse_args()

    training = LanguageTrainingExperimentConfig(
        tokenizer_kind=args.tokenizer_kind,
        tokenizer_vocab_size=max(512, int(args.tokenizer_vocab_size)),
        tokenizer_min_frequency=max(1, int(args.tokenizer_min_frequency)),
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        state_layers=max(1, int(args.state_layers)),
        attention_heads=max(1, int(args.attention_heads)),
        transformer_context_length=max(2, int(args.transformer_context_length)),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        transformer_dropout=float(args.transformer_dropout),
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        learning_rate=args.learning_rate,
        generation_tokens=args.generation_tokens,
        sustained_target_tokens=args.sustained_target_tokens,
        sustained_timeout_seconds=args.sustained_timeout_seconds,
        device=args.device,
    )
    config = LanguageArchitectureBakeoffConfig(
        training=training,
        variants=tuple(args.variant) or DEFAULT_VARIANTS,
        seeds=tuple(args.seed) or (1337,),
        epoch_budgets=tuple(args.epoch_budget) or (1, 4),
        rescore_repetition_penalty=max(1.0, float(args.rescore_repetition_penalty)),
        rescore_no_repeat_ngram_size=max(0, int(args.rescore_no_repeat_ngram_size)),
    )
    report = run_language_architecture_bakeoff(
        output_path=args.output,
        corpus_path=args.corpus,
        config=config,
    )
    return 0 if report["decision"]["winner"] is not None else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
