"""V37 matched falsification of MARULHO learned depth assembly."""

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
from marulho.training.language_depth_assembly import (
    MarulhoDepthAssemblyStateBlock,
    install_depth_assembly,
)
from marulho.training.language_model import (
    evaluate_language_model,
    load_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_depth_assembly_falsification.v1"
ARTIFACT_KIND = "marulho_depth_assembly_falsification"
BASELINE_ARM = "v35r_transformer"
CANDIDATE_ARM = "v37_depth_assembly"
ADVANCE_DECISION = "advance_v37_depth_assembly_to_durable_confirmation"
RETIRE_DECISION = "retire_v37_depth_assembly_no_quality_per_second_gain"
INVALID_DECISION = "invalid_v37_depth_assembly_evidence"
V35R_CHECKPOINT_SHA256 = (
    "48bfe82a70d9c537f10dc6d898c3cf18906716bd90acfefb7089ccd30477d9df"
)


@dataclass(frozen=True)
class DepthAssemblyFalsificationConfig:
    token_budget: int = 16_773_120
    sequence_length: int = 72
    microbatch_size: int = 32
    microbatches_per_optimizer_step: int = 8
    eval_batches: int = 16
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 37_121
    model_seed: int = 37_131
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    minimum_heldout_loss_improvement: float = 0.02
    minimum_throughput_ratio: float = 0.90
    maximum_peak_memory_ratio: float = 1.25

    @property
    def physical_batch_size(self) -> int:
        return int(self.microbatch_size) * int(
            self.microbatches_per_optimizer_step
        )


def select_depth_assembly(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    config: DepthAssemblyFalsificationConfig,
) -> str:
    if set(arms) != {BASELINE_ARM, CANDIDATE_ARM}:
        return INVALID_DECISION
    if not all(bool(row["all_parameters_received_final_gradient"]) for row in arms.values()):
        return INVALID_DECISION
    candidate_diagnostics = arms[CANDIDATE_ARM].get("diagnostics", {})
    if int(candidate_diagnostics.get("nonzero_parameter_count", 0)) != int(
        candidate_diagnostics.get("parameter_count", -1)
    ):
        return INVALID_DECISION
    baseline = arms[BASELINE_ARM]
    candidate = arms[CANDIDATE_ARM]
    improvement = float(baseline["heldout"]["heldout_loss"]) - float(
        candidate["heldout"]["heldout_loss"]
    )
    throughput_ratio = float(candidate["training"]["tokens_per_second"]) / float(
        baseline["training"]["tokens_per_second"]
    )
    baseline_peak = max(1, int(baseline["training"]["peak_cuda_memory_bytes"]))
    memory_ratio = int(candidate["training"]["peak_cuda_memory_bytes"]) / baseline_peak
    if (
        improvement >= float(config.minimum_heldout_loss_improvement)
        and throughput_ratio >= float(config.minimum_throughput_ratio)
        and memory_ratio <= float(config.maximum_peak_memory_ratio)
    ):
        return ADVANCE_DECISION
    return RETIRE_DECISION


def _data_config(
    config: DepthAssemblyFalsificationConfig,
) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.microbatch_size),
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=8,
        relation_case_limit=0,
        relation_fraction=0.0,
        learning_rate=float(config.learning_rate),
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
        raise ValueError("V37 source manifest differs from the preregistered files")
    for name, expected_hash in expected.items():
        if sha256_file(requested[name]) != expected_hash:
            raise ValueError(f"V37 source hash differs for {name}")


def _optimizer_builder(model, config):
    return build_language_muon(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        adamw_betas=(float(config.adam_beta1), float(config.adam_beta2)),
        per_head_attention_qkv=False,
    )


def run_depth_assembly_falsification(
    *,
    checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    report_output_path: str | Path,
    config: DepthAssemblyFalsificationConfig = DepthAssemblyFalsificationConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V37 requires CUDA")
    if int(config.physical_batch_size) != 256:
        raise ValueError("V37 locks the advancing physical batch at 256")
    tokens_per_step = int(config.physical_batch_size) * int(config.sequence_length)
    if int(config.token_budget) % tokens_per_step != 0:
        raise ValueError("V37 token budget must divide into complete optimizer steps")
    checkpoint = Path(checkpoint_path)
    if sha256_file(checkpoint) != V35R_CHECKPOINT_SHA256:
        raise ValueError("V37 requires the exact qualified V35R checkpoint")

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
        raise ValueError("V37 relation corpus hash differs")
    if sha256_file(relation_cases_path) != "620f51974ce5e39b20da090c15e12e1e3dc1535a2c662ded4e1244ef5f18560d":
        raise ValueError("V37 relation cases hash differs")

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
        raise ValueError("V37 schedule repeats a source batch")
    training_config = _training_config(
        data_config,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.physical_batch_size),
    )
    example_batch = grouped_staged_batch(
        prepared.staged,
        start=0,
        count=int(config.microbatches_per_optimizer_step),
        device=resolved,
    )

    baseline, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if tokenizer.vocabulary_hash() != prepared.tokenizer.vocabulary_hash():
        raise ValueError("V37 tokenizer differs from V35R")
    baseline = baseline.to(resolved)
    baseline_state = {
        name: value.detach().cpu().clone()
        for name, value in baseline.state_dict().items()
    }
    initial_hash = model_state_sha256(baseline)
    initial_heldout = evaluate_language_model(baseline, prepared.eval_batches)
    print("[V37] compiling baseline", flush=True)
    baseline_loss, baseline_execution = _prepare_language_loss_backend(
        baseline, example_batch, training_config
    )
    baseline_row = run_matched_training_arm(
        BASELINE_ARM,
        architecture="v35r_causal_transformer",
        model=baseline,
        initial_state=baseline_state,
        training_loss=baseline_loss,
        execution=baseline_execution,
        allocated_compile_seconds=float(baseline_execution["compile_seconds"]),
        prepared=prepared,
        training_config=training_config,
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        relation_eval_batch_size=8,
        model_seed=int(config.model_seed),
        device=resolved,
        progress_prefix="V37",
        optimizer_builder=_optimizer_builder,
        optimizer_warmup_steps=3,
        microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
        extra_row={"initial_heldout": initial_heldout},
    )
    del baseline, baseline_loss
    gc.collect()
    torch.cuda.empty_cache()

    candidate, candidate_tokenizer, _ = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if candidate_tokenizer.vocabulary_hash() != tokenizer.vocabulary_hash():
        raise ValueError("V37 candidate tokenizer differs from baseline")
    depth_block = install_depth_assembly(candidate)
    candidate = candidate.to(resolved)
    candidate_state = {
        name: value.detach().cpu().clone()
        for name, value in candidate.state_dict().items()
    }
    candidate_initial = evaluate_language_model(candidate, prepared.eval_batches)
    initial_loss_delta = abs(
        float(candidate_initial["heldout_loss"])
        - float(initial_heldout["heldout_loss"])
    )
    if initial_loss_delta != 0.0:
        raise ValueError("V37 identity initialization changes heldout loss")
    print("[V37] compiling depth assembly", flush=True)
    candidate_loss, candidate_execution = _prepare_language_loss_backend(
        candidate, example_batch, training_config
    )
    candidate_row = run_matched_training_arm(
        CANDIDATE_ARM,
        architecture="marulho_depth_assembly_transformer_v1",
        model=candidate,
        initial_state=candidate_state,
        training_loss=candidate_loss,
        execution=candidate_execution,
        allocated_compile_seconds=float(candidate_execution["compile_seconds"]),
        prepared=prepared,
        training_config=training_config,
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        relation_eval_batch_size=8,
        model_seed=int(config.model_seed),
        device=resolved,
        progress_prefix="V37",
        optimizer_builder=_optimizer_builder,
        optimizer_warmup_steps=3,
        microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
        diagnostic_builder=lambda model_value, _input_ids: (
            model_value.state_block.route_report()
            if isinstance(model_value.state_block, MarulhoDepthAssemblyStateBlock)
            else {}
        ),
        extra_row={
            "initial_heldout": candidate_initial,
            "identity_initial_loss_absolute_delta": initial_loss_delta,
            "depth_route_parameter_count": int(depth_block.depth_routes.numel()),
        },
    )
    rows = {BASELINE_ARM: baseline_row, CANDIDATE_ARM: candidate_row}
    decision = select_depth_assembly(rows, config=config)
    candidate_row["heldout_loss_improvement_vs_baseline"] = float(
        baseline_row["heldout"]["heldout_loss"]
    ) - float(candidate_row["heldout"]["heldout_loss"])
    candidate_row["throughput_ratio_vs_baseline"] = float(
        candidate_row["training"]["tokens_per_second"]
    ) / float(baseline_row["training"]["tokens_per_second"])
    candidate_row["peak_memory_ratio_vs_baseline"] = int(
        candidate_row["training"]["peak_cuda_memory_bytes"]
    ) / max(1, int(baseline_row["training"]["peak_cuda_memory_bytes"]))

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
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_microbatch_count": int(prepared.staged.step_count),
            "tokens_per_microbatch": int(prepared.staged.tokens_per_step),
            "processed_tokens_per_arm": int(config.token_budget),
            "schedule_unique": True,
            "source_selections": prepared.source_selections,
        },
        "initial_heldout": initial_heldout,
        "arms": rows,
        "gates": {
            "minimum_heldout_loss_improvement": float(
                config.minimum_heldout_loss_improvement
            ),
            "minimum_throughput_ratio": float(config.minimum_throughput_ratio),
            "maximum_peak_memory_ratio": float(config.maximum_peak_memory_ratio),
        },
        "decision": decision,
        "checkpoint_saved": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title="MARULHO V37 Depth-Assembly Falsification",
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
    report = run_depth_assembly_falsification(
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
