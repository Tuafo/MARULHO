"""Token-matched V36 screen for faster MARULHO training on one consumer GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_general_context_falsification import (
    GeneralContextFalsificationConfig,
    _prepare_data,
    _training_config,
    model_state_sha256,
)
from marulho.evaluation.language_matched_support import (
    grouped_staged_batch,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_training_experiment import (
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    load_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_consumer_gpu_throughput.v1"
ARTIFACT_KIND = "marulho_consumer_gpu_throughput"
BASELINE_ARM = "batch32_whole_qkv_lr3e4"
ADVANCE_DECISION = "adopt_v36_consumer_gpu_recipe_for_next_training_stage"
RETAIN_DECISION = "retain_v35r_training_recipe_no_quality_safe_speedup"
INVALID_DECISION = "invalid_v36_consumer_gpu_throughput_evidence"
V35R_CHECKPOINT_SHA256 = (
    "48bfe82a70d9c537f10dc6d898c3cf18906716bd90acfefb7089ccd30477d9df"
)


@dataclass(frozen=True)
class ThroughputArm:
    name: str
    microbatches_per_optimizer_step: int
    learning_rate: float
    per_head_attention_qkv: bool

    @property
    def physical_batch_size(self) -> int:
        return 32 * int(self.microbatches_per_optimizer_step)


ARMS = (
    ThroughputArm(BASELINE_ARM, 1, 3.0e-4, False),
    ThroughputArm("batch32_per_head_lr3e4", 1, 3.0e-4, True),
    ThroughputArm("batch256_whole_qkv_lr3e4", 8, 3.0e-4, False),
    ThroughputArm("batch256_whole_qkv_lr8p5e4", 8, 8.5e-4, False),
    ThroughputArm("batch256_whole_qkv_lr1p2e3", 8, 1.2e-3, False),
    ThroughputArm("batch256_per_head_lr8p5e4", 8, 8.5e-4, True),
)


@dataclass(frozen=True)
class ConsumerGpuThroughputConfig:
    token_budget: int = 2_359_296
    sequence_length: int = 72
    microbatch_size: int = 32
    eval_batches: int = 16
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 36_121
    model_seed: int = 36_131
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    maximum_heldout_loss_regression: float = 0.01
    minimum_large_batch_speedup: float = 1.80
    maximum_per_head_loss_regression: float = 0.005
    minimum_per_head_speedup: float = 1.03


def select_throughput_arm(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    config: ConsumerGpuThroughputConfig,
) -> tuple[str | None, str]:
    if set(arms) != {arm.name for arm in ARMS}:
        return None, INVALID_DECISION
    if not all(bool(row["all_parameters_received_final_gradient"]) for row in arms.values()):
        return None, INVALID_DECISION
    baseline = arms[BASELINE_ARM]
    baseline_loss = float(baseline["heldout"]["heldout_loss"])
    baseline_tps = float(baseline["training"]["tokens_per_second"])
    candidates: list[tuple[float, str]] = []
    for spec in ARMS:
        if int(spec.microbatches_per_optimizer_step) == 1:
            continue
        row = arms[spec.name]
        loss_regression = float(row["heldout"]["heldout_loss"]) - baseline_loss
        speedup = float(row["training"]["tokens_per_second"]) / baseline_tps
        if (
            loss_regression <= float(config.maximum_heldout_loss_regression)
            and speedup >= float(config.minimum_large_batch_speedup)
        ):
            candidates.append((speedup, spec.name))
    if not candidates:
        return None, RETAIN_DECISION
    selected = max(candidates)[1]
    return selected, ADVANCE_DECISION


def per_head_optimizer_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    config: ConsumerGpuThroughputConfig,
) -> str:
    baseline = arms[BASELINE_ARM]
    candidate = arms["batch32_per_head_lr3e4"]
    loss_regression = float(candidate["heldout"]["heldout_loss"]) - float(
        baseline["heldout"]["heldout_loss"]
    )
    speedup = float(candidate["training"]["tokens_per_second"]) / float(
        baseline["training"]["tokens_per_second"]
    )
    if (
        loss_regression <= float(config.maximum_per_head_loss_regression)
        and speedup >= float(config.minimum_per_head_speedup)
    ):
        return "adopt_per_head_qkv_muon"
    return "retain_whole_qkv_muon"


def _data_config(config: ConsumerGpuThroughputConfig) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.microbatch_size),
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=8,
        relation_case_limit=0,
        relation_fraction=0.0,
        learning_rate=3.0e-4,
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        data_seed=int(config.data_seed),
        model_seed=int(config.model_seed),
        sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
        sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
        sample_range_count=int(config.sample_range_count),
        schedule_mode=str(config.schedule_mode),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
        width=768,
        layers=10,
        heads=12,
        mlp_ratio=4.0,
    )


def _validate_sources(paths: Sequence[str | Path], expected: Mapping[str, str]) -> None:
    requested = {Path(path).name: Path(path) for path in paths}
    if set(requested) != set(expected) or len(tuple(paths)) != len(expected):
        raise ValueError("V36 source manifest differs from the preregistered files")
    for name, expected_hash in expected.items():
        if sha256_file(requested[name]) != expected_hash:
            raise ValueError(f"V36 source hash differs for {name}")


def run_consumer_gpu_throughput(
    *,
    checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    report_output_path: str | Path,
    config: ConsumerGpuThroughputConfig = ConsumerGpuThroughputConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V36 consumer-GPU throughput screen requires CUDA")
    if int(config.microbatch_size) != 32 or int(config.sequence_length) != 72:
        raise ValueError("V36 locks the V35R microbatch shape at 32x72")
    if int(config.token_budget) % (32 * 72 * 8) != 0:
        raise ValueError("V36 token budget must divide exactly into batch-256 steps")
    checkpoint = Path(checkpoint_path)
    if sha256_file(checkpoint) != V35R_CHECKPOINT_SHA256:
        raise ValueError("V36 requires the exact qualified V35R checkpoint")

    expected_train = {
        "fineweb-edu-train-75k-shard0-20260710.txt": "75f07f85c15c971e1d6eeba623c3f8e20d794e81b9c356ad6fadff2366c99434",
        "cosmopedia-v2-train-150k-shard1-20260710.txt": "c4c846e1d08965c2c3f0e615b67d5b23554965e9222eb72bbb9ecaa4d7199b65",
        "cosmopedia-v2-train-75k-shard3-20260710.txt": "3a135b5f9c8386ca2edd7c18deefec82cafc6e5922691324428d050158d6da51",
    }
    expected_eval = {
        "fineweb-edu-eval-10k-shard1-20260710.txt": "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a",
        "cosmopedia-v2-eval-10k-shard2-20260710.txt": "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491",
    }
    _validate_sources(general_train_paths, expected_train)
    _validate_sources(general_eval_paths, expected_eval)
    if sha256_file(relation_corpus_path) != "5db37c256d3de62209f0b30ae0f5c1aa206569e4d82913cddc8f73566ba4e8c7":
        raise ValueError("V36 relation corpus hash differs")
    if sha256_file(relation_cases_path) != "620f51974ce5e39b20da090c15e12e1e3dc1535a2c662ded4e1244ef5f18560d":
        raise ValueError("V36 relation cases hash differs")

    data_config = _data_config(config)
    prepared = _prepare_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.microbatch_size),
        config=data_config,
        device=resolved,
    )
    if len(prepared.schedule) != len(set(prepared.schedule)):
        raise ValueError("V36 schedule repeats a source batch")

    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if tokenizer.vocabulary_hash() != prepared.tokenizer.vocabulary_hash():
        raise ValueError("V36 tokenizer differs from V35R")
    model = model.to(resolved)
    initial_hash = model_state_sha256(model)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_heldout = evaluate_language_model(model, prepared.eval_batches)

    graph_by_group: dict[int, tuple[Any, Mapping[str, Any]]] = {}
    for group in sorted({arm.microbatches_per_optimizer_step for arm in ARMS}):
        physical_batch = grouped_staged_batch(
            prepared.staged, start=0, count=group, device=resolved
        )
        graph_config = _training_config(
            data_config,
            sequence_length=int(config.sequence_length),
            batch_size=int(config.microbatch_size) * group,
        )
        print(f"[V36] compiling batch {int(config.microbatch_size) * group}", flush=True)
        graph_by_group[group] = _prepare_language_loss_backend(
            model, physical_batch, graph_config
        )

    arm_count_by_group = {
        group: sum(arm.microbatches_per_optimizer_step == group for arm in ARMS)
        for group in graph_by_group
    }
    rows: dict[str, dict[str, Any]] = {}
    for spec in ARMS:
        training_loss, execution = graph_by_group[spec.microbatches_per_optimizer_step]
        arm_data_config = GeneralContextFalsificationConfig(
            **{
                **asdict(data_config),
                "learning_rate": float(spec.learning_rate),
            }
        )
        training_config = _training_config(
            arm_data_config,
            sequence_length=int(config.sequence_length),
            batch_size=int(spec.physical_batch_size),
        )

        def optimizer_builder(model_value, config_value, *, per_head=spec.per_head_attention_qkv):
            return build_language_muon(
                model_value,
                learning_rate=float(config_value.learning_rate),
                weight_decay=float(config_value.weight_decay),
                adamw_betas=(
                    float(config_value.adam_beta1),
                    float(config_value.adam_beta2),
                ),
                per_head_attention_qkv=bool(per_head),
            )

        print(f"[V36] training {spec.name}", flush=True)
        row = run_matched_training_arm(
            spec.name,
            architecture="v35r_causal_transformer_optimizer_throughput_screen",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution=execution,
            allocated_compile_seconds=float(execution["compile_seconds"])
            / int(arm_count_by_group[spec.microbatches_per_optimizer_step]),
            prepared=prepared,
            training_config=training_config,
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=8,
            model_seed=int(config.model_seed),
            device=resolved,
            progress_prefix="V36",
            extra_row={
                "initial_heldout": initial_heldout,
                "physical_batch_size": int(spec.physical_batch_size),
                "learning_rate": float(spec.learning_rate),
                "per_head_attention_qkv": bool(spec.per_head_attention_qkv),
            },
            optimizer_builder=optimizer_builder,
            optimizer_warmup_steps=3,
            microbatches_per_optimizer_step=int(
                spec.microbatches_per_optimizer_step
            ),
        )
        rows[spec.name] = row
        gc.collect()
        torch.cuda.empty_cache()

    selected_arm, decision = select_throughput_arm(rows, config=config)
    per_head_decision = per_head_optimizer_decision(rows, config=config)
    baseline_tps = float(rows[BASELINE_ARM]["training"]["tokens_per_second"])
    for row in rows.values():
        row["heldout_loss_delta_vs_baseline"] = float(
            row["heldout"]["heldout_loss"]
        ) - float(rows[BASELINE_ARM]["heldout"]["heldout_loss"])
        row["throughput_speedup_vs_baseline"] = float(
            row["training"]["tokens_per_second"]
        ) / baseline_tps

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "created_at_unix": time.time(),
        "external_llm_used": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": V35R_CHECKPOINT_SHA256,
            "metadata": metadata,
            "initial_state_sha256": initial_hash,
        },
        "config": asdict(config),
        "arms_manifest": [asdict(arm) for arm in ARMS],
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_microbatch_count": int(prepared.staged.step_count),
            "tokens_per_microbatch": int(prepared.staged.tokens_per_step),
            "processed_tokens_per_arm": int(config.token_budget),
            "schedule_unique": True,
            "source_selections": prepared.source_selections,
        },
        "initial_heldout": initial_heldout,
        "execution_graphs": {
            str(int(config.microbatch_size) * group): dict(execution)
            for group, (_training_loss, execution) in graph_by_group.items()
        },
        "arms": rows,
        "gates": {
            "maximum_heldout_loss_regression": float(
                config.maximum_heldout_loss_regression
            ),
            "minimum_large_batch_speedup": float(config.minimum_large_batch_speedup),
            "maximum_per_head_loss_regression": float(
                config.maximum_per_head_loss_regression
            ),
            "minimum_per_head_speedup": float(config.minimum_per_head_speedup),
        },
        "selected_arm": selected_arm,
        "per_head_optimizer_decision": per_head_decision,
        "decision": decision,
        "checkpoint_saved": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title="MARULHO V36 Consumer-GPU Throughput Screen",
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--relation-corpus", required=True)
    parser.add_argument("--relation-cases", required=True)
    parser.add_argument("--general-train", action="append", required=True)
    parser.add_argument("--general-eval", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_consumer_gpu_throughput(
        checkpoint_path=args.checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        report_output_path=args.report,
        device=args.device,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
