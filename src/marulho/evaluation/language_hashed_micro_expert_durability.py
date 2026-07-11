"""Larger-budget durability test for pruned hashed micro-experts v11."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_geometry import transformer_depth_geometry_report
from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    PreparedMatchedLanguageData,
    parameter_sha256,
    prepare_matched_language_data,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    HASHED_MICRO_EXPERT_MODES,
    HashedMicroExpertConfig,
    MarulhoHashedMicroExpertLanguageModel,
)
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)


SURFACE = "marulho_hashed_micro_expert_durability.v1"
ARTIFACT_KIND = "marulho_hashed_micro_expert_durability"
ARM_NAMES = ("transformer", *HASHED_MICRO_EXPERT_MODES)
ARCHITECTURES = ("transformer", "hashed_micro_experts")


@dataclass(frozen=True)
class HashedMicroExpertDurabilityConfig:
    token_budget: int = 67_108_864
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    relation_fraction: float = 0.20
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 2026
    model_seed: int = 2026
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    width: int = 512
    layers: int = 4
    attention_heads: int = 8
    baseline_hidden_width: int = 2048
    shared_hidden_width: int = 1024
    expert_layer_index: int = 2
    expert_pool_size: int = 16_384
    routing_heads: int = 4
    experts_per_head: int = 2
    hash_seed: int = 10_729
    routing_vector_samples: int = 256
    geometry_max_samples: int = 4096
    execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3


def _architecture(name: str) -> str:
    return "transformer" if name == "transformer" else "hashed_micro_experts"


def _data_config(
    config: HashedMicroExpertDurabilityConfig,
) -> MatchedLanguageDataConfig:
    return MatchedLanguageDataConfig(
        token_budget=config.token_budget,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        eval_batches=config.eval_batches,
        relation_fraction=config.relation_fraction,
        seed=config.seed,
        sample_bytes_per_train_source=config.sample_bytes_per_train_source,
        sample_bytes_per_eval_source=config.sample_bytes_per_eval_source,
        sample_range_count=config.sample_range_count,
    )


def _training_config(
    config: HashedMicroExpertDurabilityConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        learning_rate=config.learning_rate,
        minimum_learning_rate_fraction=config.minimum_learning_rate_fraction,
        warmup_fraction=config.warmup_fraction,
        weight_decay=config.weight_decay,
        precision=config.precision,
        execution_backend=config.execution_backend,
        compile_loss_tolerance=config.compile_loss_tolerance,
    )


def _build_model(
    architecture: str,
    *,
    vocab_size: int,
    config: HashedMicroExpertDurabilityConfig,
) -> MarulhoLanguageModel | MarulhoHashedMicroExpertLanguageModel:
    if architecture == "transformer":
        return MarulhoLanguageModel(
            LanguageModelConfig(
                vocab_size=int(vocab_size),
                embedding_dim=config.width,
                state_dim=config.width,
                state_layers=config.layers,
                attention_heads=config.attention_heads,
                transformer_context_length=config.sequence_length,
                transformer_mlp_ratio=(
                    config.baseline_hidden_width / config.width
                ),
                tie_embeddings=True,
                active_language_path="marulho_transformer_v11_control",
            )
        )
    if architecture == "hashed_micro_experts":
        return MarulhoHashedMicroExpertLanguageModel(
            HashedMicroExpertConfig(
                vocab_size=int(vocab_size),
                width=config.width,
                layers=config.layers,
                attention_heads=config.attention_heads,
                context_length=config.sequence_length,
                baseline_hidden_width=config.baseline_hidden_width,
                shared_hidden_width=config.shared_hidden_width,
                expert_layer_index=config.expert_layer_index,
                expert_pool_size=config.expert_pool_size,
                routing_heads=config.routing_heads,
                experts_per_head=config.experts_per_head,
                hash_seed=config.hash_seed,
                mode="token_hash",
            )
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _configure_model(model, name: str) -> None:
    if isinstance(model, MarulhoHashedMicroExpertLanguageModel):
        model.set_hashed_micro_expert_mode(name)


def _common_parameter_hash(
    model,
    *,
    architecture: str,
    expert_layer_index: int,
) -> str:
    prefix = f"state_block.layers.{int(expert_layer_index)}"
    if architecture == "transformer":
        excluded = (
            f"{prefix}.gate_up.weight",
            f"{prefix}.down.weight",
        )
    else:
        excluded = (
            f"{prefix}.shared_gate_up.weight",
            f"{prefix}.shared_down.weight",
            f"{prefix}.expert_input.weight",
            f"{prefix}.expert_output.weight",
        )
    return parameter_sha256(model, excluded_names=excluded)


def _diagnostics(
    model,
    input_ids: torch.Tensor,
    *,
    max_vector_samples: int,
    geometry_max_samples: int,
) -> dict[str, Any]:
    precision = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if model.device.type == "cuda"
        else nullcontext()
    )
    with precision:
        geometry = transformer_depth_geometry_report(
            model,
            input_ids,
            max_samples=geometry_max_samples,
        )
        routing = (
            model.routing_report(
                input_ids,
                max_vector_samples=max_vector_samples,
            )
            if isinstance(model, MarulhoHashedMicroExpertLanguageModel)
            else None
        )
    if not isinstance(model, MarulhoHashedMicroExpertLanguageModel):
        return {"depth_geometry": geometry}
    return {
        "active_parameters": model.active_parameter_report(),
        "final_gradients": model.final_gradient_report(),
        "routing": routing,
        "depth_geometry": geometry,
    }


def hashed_micro_expert_durability_decision(
    arms: Sequence[Mapping[str, Any]],
    *,
    minimum_tokens: int = 67_108_864,
) -> str:
    rows = {str(row["name"]): row for row in arms}
    if any(name not in rows for name in ARM_NAMES):
        return "incomplete_v11_hashed_micro_expert_comparison"
    if min(int(rows[name].get("processed_tokens") or 0) for name in ARM_NAMES) < int(
        minimum_tokens
    ):
        return "incomplete_v11_mechanism_smoke"

    def loss(name: str) -> float:
        return float(rows[name]["heldout"]["heldout_loss"])

    def free(name: str) -> float:
        return float(rows[name]["relation"]["generation_exact_accuracy"])

    def qualified(candidate: str, control: str) -> bool:
        return (
            loss(candidate) <= loss(control) - 0.005
            and free(candidate) >= free(control) + 0.02
        )

    throughput_ratio = float(
        rows["token_hash"]["training"]["tokens_per_second"]
    ) / max(
        float(rows["transformer"]["training"]["tokens_per_second"]),
        1.0e-12,
    )
    if qualified("token_hash", "transformer") and qualified(
        "token_hash", "shared_only"
    ):
        if throughput_ratio < 0.50:
            return "redesign_v11_durable_quality_but_local_throughput_collapse"
        return "promote_v11_hash_for_checkpoint_and_unseen_generation"
    loss_signal = loss("token_hash") <= loss("transformer") - 0.005
    behavior_signal = free("token_hash") >= free("transformer") + 0.02
    if loss_signal and behavior_signal:
        return "redesign_v11_durable_transformer_win_shared_behavior_unresolved"
    if loss_signal:
        return "redesign_v11_durable_loss_without_joint_behavior"
    if behavior_signal:
        return "redesign_v11_durable_behavior_without_loss"
    return "retire_v11_hashed_micro_experts_not_durable"


def _assemble_report(
    *,
    config: HashedMicroExpertDurabilityConfig,
    prepared: PreparedMatchedLanguageData,
    combined: Mapping[str, Mapping[str, Any]],
    architecture_runs: Mapping[str, Mapping[str, Any]],
    executed_names: Sequence[str],
    elapsed_seconds: float,
) -> dict[str, Any]:
    arms = [dict(combined[name]) for name in ARM_NAMES if name in combined]
    counts = {row["name"]: int(row["parameters"]) for row in arms}
    transformer_count = counts.get("transformer")
    candidate_count = next(
        (
            int(row["parameters"])
            for row in arms
            if row["architecture"] == "hashed_micro_experts"
        ),
        None,
    )
    active_report = next(
        (
            row.get("diagnostics", {}).get("active_parameters")
            for row in arms
            if row["architecture"] == "hashed_micro_experts"
            and row.get("diagnostics", {}).get("active_parameters")
        ),
        None,
    )
    candidate_rates = [
        float(row["training"]["tokens_per_second"])
        for row in arms
        if row["architecture"] == "hashed_micro_experts"
    ]
    common_hashes = {
        str(row["common_initialization_sha256"])
        for row in arms
        if row.get("common_initialization_sha256")
    }
    requested_compiles = sum(
        len(row.get("arms_sharing_graph") or [])
        for row in architecture_runs.values()
        if row.get("executed_this_run")
    )
    actual_compiles = sum(
        bool(row.get("executed_this_run")) for row in architecture_runs.values()
    )
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "tokenizer": {
            "checkpoint": str(prepared.tokenizer_checkpoint),
            "checkpoint_sha256": sha256_file(prepared.tokenizer_checkpoint),
            "vocab_size": prepared.tokenizer.vocab_size,
            "hash": prepared.tokenizer.vocabulary_hash(),
        },
        "relation_cases": {
            "path": str(prepared.relation_cases_file),
            "sha256": sha256_file(prepared.relation_cases_file),
            "count": len(prepared.cases),
            "labels_metrics_only": True,
        },
        "source_selections": prepared.source_selections,
        "schedule": {
            "steps": len(prepared.schedule),
            "processed_tokens": (
                len(prepared.schedule)
                * config.batch_size
                * config.sequence_length
            ),
            "relation_steps": sum(
                kind == "relation" for kind, _index in prepared.schedule
            ),
            "sha256": prepared.schedule_sha256,
            "identical_for_all_arms": True,
            "staged_once_on_device": True,
            "staging_seconds": prepared.staged.elapsed_seconds,
            "staging_storage_bytes": prepared.staged.storage_bytes,
        },
        "parameter_budget": {
            "counts": counts,
            "candidate_minus_transformer_fraction": (
                (candidate_count - transformer_count) / transformer_count
                if candidate_count is not None and transformer_count
                else None
            ),
            "active_compute": active_report,
            "stored_parameters_are_not_active_parameters": True,
        },
        "initialization_match": {
            "common_parameter_sha256_values": sorted(common_hashes),
            "all_common_initial_parameters_identical": len(common_hashes) <= 1,
            "replaced_mlp_excluded_from_common_hash": True,
        },
        "compile_reuse": {
            "architecture_runs": dict(architecture_runs),
            "loss_graph_compile_count_actual": actual_compiles,
            "loss_graph_compile_count_without_reuse": requested_compiles,
            "loss_graph_compiles_avoided": max(
                0,
                requested_compiles - actual_compiles,
            ),
            "compile_seconds_this_run": sum(
                float(row.get("compile_seconds_total") or 0.0)
                for row in architecture_runs.values()
                if row.get("executed_this_run")
            ),
        },
        "control_compute": {
            "same_candidate_parameter_objects": True,
            "same_candidate_compiled_loss_graph": True,
            "same_eight_expert_granularity": True,
            "steady_tokens_per_second_min": (
                min(candidate_rates) if candidate_rates else None
            ),
            "steady_tokens_per_second_max": (
                max(candidate_rates) if candidate_rates else None
            ),
            "max_to_min_ratio": (
                max(candidate_rates) / min(candidate_rates)
                if candidate_rates
                else None
            ),
        },
        "arms": arms,
        "arms_executed_this_run": list(executed_names),
        "experiment_wall_seconds_this_run": elapsed_seconds,
        "decision": hashed_micro_expert_durability_decision(arms),
        "decision_contract": {
            "heldout_loss_margin": 0.005,
            "free_relation_margin": 0.02,
            "hash_must_beat_transformer_and_shared_only": True,
            "minimum_local_throughput_fraction": 0.50,
            "promotion_requires_checkpoint_and_unseen_generation_next": True,
            "routing_and_geometry_are_diagnostic_only": True,
        },
        "anti_cheat": {
            "route_uses_input_token_ids_only": True,
            "route_uses_relation_labels": False,
            "correct_index_metrics_only": True,
        },
        "quality_boundary": {
            "promotes_runtime_installation": False,
            "promotes_unseen_generation": False,
            "throughput_is_not_quality": True,
        },
        "research_provenance": {
            "peer_singleton_experts": "https://arxiv.org/abs/2407.04153",
            "v10_learned_router_result": (
                "reports/language_scaling/"
                "micro-experts-v10-replication-seed7331-16m-20260711.json"
            ),
        },
    }


def run_hashed_micro_expert_durability(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: HashedMicroExpertDurabilityConfig = (
        HashedMicroExpertDurabilityConfig()
    ),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    if config.execution_backend not in {"eager", "inductor"}:
        raise ValueError("execution_backend must be 'eager' or 'inductor'")
    if config.compile_loss_tolerance <= 0.0:
        raise ValueError("compile_loss_tolerance must be positive")
    resolved = _resolve_device(device)
    if config.execution_backend == "inductor" and resolved.type != "cuda":
        raise ValueError("Inductor v11 execution is admitted only for CUDA runs")
    requested_arms = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested_arms or any(name not in ARM_NAMES for name in requested_arms):
        raise ValueError("arm_names must contain valid unique v11 arm names")

    run_started = time.perf_counter()
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=tokenizer_checkpoint_path,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=_data_config(config),
        device=resolved,
    )
    combined: dict[str, Mapping[str, Any]] = {}
    architecture_runs: dict[str, Mapping[str, Any]] = {}
    executed_names: list[str] = []
    expected_common_hash: str | None = None
    output = Path(output_path)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        for architecture in ARCHITECTURES:
            group_arms = [
                name
                for name in requested_arms
                if _architecture(name) == architecture
            ]
            if not group_arms:
                continue
            print(f"[hashed-micro-v11] preparing {architecture} graph", flush=True)
            torch.manual_seed(config.model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(config.model_seed)
            model = _build_model(
                architecture,
                vocab_size=prepared.tokenizer.vocab_size,
                config=config,
            )
            common_hash = _common_parameter_hash(
                model,
                architecture=architecture,
                expert_layer_index=config.expert_layer_index,
            )
            if expected_common_hash is None:
                expected_common_hash = common_hash
            elif common_hash != expected_common_hash:
                raise RuntimeError("Common V11 initial parameters are not identical")
            initial_state = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            model = model.to(resolved)
            training_config = _training_config(config)
            warm_batch = LanguageBatch(
                prepared.staged.input_ids[0],
                prepared.staged.target_ids[0],
            )
            model.train()
            training_loss, execution = _prepare_language_loss_backend(
                model,
                warm_batch,
                training_config,
            )
            compile_seconds = float(execution["compile_seconds"])
            model.eval()
            initial_heldout = evaluate_language_model(model, prepared.eval_batches)
            architecture_runs[architecture] = {
                "executed_this_run": True,
                "arms_sharing_graph": list(group_arms),
                "same_model_object_reloaded_from_exact_initial_state": True,
                "fresh_optimizer_per_arm": True,
                "loss_execution": execution,
                "compile_seconds_total": compile_seconds,
                "initial_heldout": initial_heldout,
            }
            allocated_compile = compile_seconds / max(1, len(group_arms))
            for name in group_arms:
                print(f"[hashed-micro-v11] starting {name}", flush=True)
                row = run_matched_training_arm(
                    name,
                    architecture=architecture,
                    model=model,
                    initial_state=initial_state,
                    training_loss=training_loss,
                    execution={
                        **execution,
                        "loss_graph_shared_between_hash_controls": (
                            architecture == "hashed_micro_experts"
                        ),
                    },
                    allocated_compile_seconds=allocated_compile,
                    prepared=prepared,
                    training_config=training_config,
                    gradient_clip=config.gradient_clip,
                    precision=config.precision,
                    relation_eval_batch_size=config.relation_eval_batch_size,
                    model_seed=config.model_seed,
                    device=resolved,
                    progress_prefix="hashed-micro-v11",
                    configure_model=_configure_model,
                    diagnostic_builder=lambda active_model, input_ids: _diagnostics(
                        active_model,
                        input_ids,
                        max_vector_samples=config.routing_vector_samples,
                        geometry_max_samples=config.geometry_max_samples,
                    ),
                    extra_row={
                        "hashed_micro_expert_mode": (
                            None if name == "transformer" else name
                        ),
                        "common_initialization_sha256": common_hash,
                    },
                )
                combined[name] = row
                executed_names.append(name)
                print(
                    f"[hashed-micro-v11] completed {name}: loss "
                    f"{row['heldout']['heldout_loss']:.4f}, free "
                    f"{row['relation']['generation_exact_accuracy']:.3f}",
                    flush=True,
                )
                write_json_report_with_readme(
                    output,
                    _assemble_report(
                        config=config,
                        prepared=prepared,
                        combined=combined,
                        architecture_runs=architecture_runs,
                        executed_names=executed_names,
                        elapsed_seconds=time.perf_counter() - run_started,
                    ),
                    title="MARULHO Hashed Micro-Experts v11 Durability",
                )
            del training_loss, model
            gc.collect()
            if resolved.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    report = _assemble_report(
        config=config,
        prepared=prepared,
        combined=combined,
        architecture_runs=architecture_runs,
        executed_names=executed_names,
        elapsed_seconds=time.perf_counter() - run_started,
    )
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Hashed Micro-Experts v11 Durability",
    )
    print(f"[hashed-micro-v11] decision {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=67_108_864)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, default=[])
    parser.add_argument(
        "--execution-backend",
        choices=("eager", "inductor"),
        default="eager",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = HashedMicroExpertDurabilityConfig(
        token_budget=int(args.token_budget),
        seed=int(args.seed),
        model_seed=int(args.model_seed),
        sample_bytes_per_train_source=int(args.train_sample_mib) * 1024 * 1024,
        sample_bytes_per_eval_source=int(args.eval_sample_mib) * 1024 * 1024,
        execution_backend=str(args.execution_backend),
    )
    run_hashed_micro_expert_durability(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        output_path=args.output,
        config=config,
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
