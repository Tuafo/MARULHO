"""V39 answer-emphasized continual-learning falsification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

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
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    _precision_context,
    _prepare_triton_compiler_compatibility,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_answer_objective_falsification.v1"
ARTIFACT_KIND = "marulho_answer_objective_falsification"
ARMS = {"answer_weight2": 2.0, "answer_weight4": 4.0}
ADVANCE_DECISION = "advance_v39_answer_objective_continual_checkpoint"
RETIRE_DECISION = "retire_v39_answer_weighting_no_free_generation_gain"
INVALID_DECISION = "invalid_v39_answer_objective_evidence"
V35R_CHECKPOINT_SHA256 = (
    "48bfe82a70d9c537f10dc6d898c3cf18906716bd90acfefb7089ccd30477d9df"
)
V38_REPORT_SHA256 = (
    "e356bf9a44ccb7fd1986be256c41128c2bb79a086c903d1be7bd110a841cc1d2"
)


@dataclass(frozen=True)
class AnswerObjectiveConfig:
    token_budget: int = 16_773_120
    sequence_length: int = 72
    microbatch_size: int = 32
    microbatches_per_optimizer_step: int = 8
    eval_batches: int = 16
    relation_fraction: float = 0.50
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 38_121
    model_seed: int = 38_131
    sample_bytes_per_train_source: int = 128 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    minimum_free_relation_accuracy: float = 0.50
    minimum_candidate_relation_accuracy: float = 0.80
    maximum_general_loss_regression: float = 0.10

    @property
    def physical_batch_size(self) -> int:
        return int(self.microbatch_size) * int(self.microbatches_per_optimizer_step)


def answer_target_mask(
    input_ids: torch.Tensor,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
) -> torch.Tensor:
    """Select targets after the latest Answer marker and before the next EOS."""

    if input_ids.ndim != 2 or marker_ids.ndim != 1:
        raise ValueError("answer mask expects [batch,time] ids and a flat marker")
    marker_size = int(marker_ids.numel())
    if marker_size < 1 or int(input_ids.shape[1]) < marker_size:
        raise ValueError("answer marker must fit inside the training sequence")
    matches = input_ids.unfold(1, marker_size, 1).eq(marker_ids).all(dim=-1)
    marker_ends = F.pad(matches, (marker_size - 1, 0))
    positions = torch.arange(
        1,
        int(input_ids.shape[1]) + 1,
        device=input_ids.device,
        dtype=torch.long,
    ).unsqueeze(0)
    last_marker = torch.cummax(
        torch.where(marker_ends, positions, torch.zeros_like(positions)), dim=1
    ).values
    last_eos = torch.cummax(
        torch.where(input_ids.eq(int(eos_id)), positions, torch.zeros_like(positions)),
        dim=1,
    ).values
    return last_marker > last_eos


def select_answer_objective(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    initial_general_loss: float,
    config: AnswerObjectiveConfig,
) -> tuple[str | None, str]:
    if set(arms) != set(ARMS):
        return None, INVALID_DECISION
    if not all(bool(row["all_parameters_received_final_gradient"]) for row in arms.values()):
        return None, INVALID_DECISION
    qualified: list[tuple[float, float, str]] = []
    for name, row in arms.items():
        free = float(row["relation"]["generation_exact_accuracy"])
        ranked = float(row["relation"]["accuracy"])
        regression = float(row["heldout"]["heldout_loss"]) - float(
            initial_general_loss
        )
        if (
            free >= float(config.minimum_free_relation_accuracy)
            and ranked >= float(config.minimum_candidate_relation_accuracy)
            and regression <= float(config.maximum_general_loss_regression)
        ):
            qualified.append((free, -regression, name))
    if qualified:
        return max(qualified)[2], ADVANCE_DECISION
    return None, RETIRE_DECISION


def _data_config(config: AnswerObjectiveConfig) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.microbatch_size),
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=8,
        relation_case_limit=0,
        relation_fraction=float(config.relation_fraction),
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


def _prepare_weighted_loss(
    model,
    example_batch,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
    answer_weight: float,
    config: AnswerObjectiveConfig,
):
    def eager_loss(input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits = model.forward(input_ids, collect_telemetry=False)["logits"]
        token_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1),
            reduction="none",
        ).reshape(target_ids.shape)
        mask = answer_target_mask(
            input_ids, marker_ids=marker_ids, eos_id=int(eos_id)
        )
        weights = 1.0 + mask.to(token_losses.dtype) * (float(answer_weight) - 1.0)
        return (token_losses * weights).sum() / weights.sum()

    device_batch = example_batch.to(model.device)
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all()
    with torch.no_grad(), _precision_context(model.device, config.precision):
        reference = eager_loss(device_batch.input_ids, device_batch.target_ids)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
    compatibility = _prepare_triton_compiler_compatibility()
    compiled = torch.compile(eager_loss, backend="inductor", fullgraph=True, dynamic=False)
    torch.cuda.reset_peak_memory_stats(model.device)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with _precision_context(model.device, config.precision):
        observed = compiled(device_batch.input_ids, device_batch.target_ids)
    observed.backward()
    torch.cuda.synchronize(model.device)
    compile_seconds = time.perf_counter() - started
    delta = abs(float(reference.detach().float().cpu()) - float(observed.detach().float().cpu()))
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state_all(cuda_state)
    if delta > float(config.compile_loss_tolerance):
        raise RuntimeError("V39 compiled/eager weighted loss drift exceeds tolerance")
    return compiled, {
        "requested_backend": "inductor",
        "effective_backend": "inductor",
        "ordinary_step_backend": "torch_compile_inductor_fullgraph",
        "compile_fullgraph": True,
        "compile_dynamic_shapes": False,
        "compile_seconds": compile_seconds,
        "compile_peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "triton_compiler_compatibility": compatibility,
        "warmup_loss_parity": {
            "performed": True,
            "eager_loss": float(reference.detach().float().cpu()),
            "compiled_loss": float(observed.detach().float().cpu()),
            "absolute_delta": delta,
            "tolerance": float(config.compile_loss_tolerance),
            "passed": True,
        },
    }


def _optimizer_builder(model, config):
    return build_language_muon(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        adamw_betas=(float(config.adam_beta1), float(config.adam_beta2)),
        per_head_attention_qkv=False,
    )


def run_answer_objective_falsification(
    *,
    checkpoint_path: str | Path,
    v38_report_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    report_output_path: str | Path,
    checkpoint_output_path: str | Path,
    config: AnswerObjectiveConfig = AnswerObjectiveConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda" or int(config.physical_batch_size) != 256:
        raise ValueError("V39 requires CUDA and physical batch 256")
    if int(config.token_budget) % (256 * int(config.sequence_length)) != 0:
        raise ValueError("V39 token budget must divide into complete optimizer steps")
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("V39 requires exactly two replay and two eval sources")
    checkpoint = Path(checkpoint_path)
    v38_report = Path(v38_report_path)
    expected = {
        checkpoint: V35R_CHECKPOINT_SHA256,
        v38_report: V38_REPORT_SHA256,
        Path(relation_corpus_path): "2cb66bc468ad244d1c0846f0b1b06e1ec2c2d3ad754bab2483dda51b12d7bdc3",
        Path(relation_cases_path): "5506af0f51a2e0744b237bcebf2c90862a84c9cc4a69d74814909d2c466ce85a",
        Path(general_train_paths[0]): "034a3a00ea86ec097b913f6002485a6081c6adb2b66c14ddc82be7d57b13751c",
        Path(general_train_paths[1]): "7b6f41e3b3d2c1871d0124dc19f212713e3c8136e9f66cb462c845354e267aa7",
        Path(general_eval_paths[0]): "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a",
        Path(general_eval_paths[1]): "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V39 source hash differs for {path.name}")

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
        raise ValueError("V39 schedule repeats a source batch")
    model, tokenizer, metadata = load_language_model_checkpoint(checkpoint, map_location="cpu")
    model = model.to(resolved)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_hash = model_state_sha256(model)
    initial_heldout = evaluate_language_model(model, prepared.eval_batches)
    initial_relation = evaluate_relation_binding_cases_batched(
        model, tokenizer, prepared.cases, batch_size=8
    )
    marker_values = tokenizer.encode(" Answer:", add_bos=False, add_eos=False)
    marker_ids = torch.tensor(marker_values, dtype=torch.long, device=resolved)
    example = grouped_staged_batch(
        prepared.staged,
        start=0,
        count=int(config.microbatches_per_optimizer_step),
        device=resolved,
    )
    mask_fraction = float(
        answer_target_mask(example.input_ids, marker_ids=marker_ids, eos_id=tokenizer.eos_id)
        .float()
        .mean()
        .cpu()
    )
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("V39 answer mask has invalid coverage")
    training_config = _training_config(
        data_config,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.physical_batch_size),
    )
    graphs = {}
    for name, weight in ARMS.items():
        print(f"[V39] compiling {name}", flush=True)
        graphs[name] = _prepare_weighted_loss(
            model,
            example,
            marker_ids=marker_ids,
            eos_id=tokenizer.eos_id,
            answer_weight=weight,
            config=config,
        )

    rows: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, weight in ARMS.items():
        training_loss, execution = graphs[name]
        print(f"[V39] training {name}", flush=True)
        rows[name] = run_matched_training_arm(
            name,
            architecture="v35r_transformer_answer_emphasized_continual",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution=execution,
            allocated_compile_seconds=float(execution["compile_seconds"]),
            prepared=prepared,
            training_config=training_config,
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=8,
            model_seed=int(config.model_seed),
            device=resolved,
            progress_prefix="V39",
            optimizer_builder=_optimizer_builder,
            optimizer_warmup_steps=3,
            microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
            extra_row={"answer_weight": weight, "answer_mask_fraction": mask_fraction},
        )
        states[name] = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }

    selected, decision = select_answer_objective(
        rows,
        initial_general_loss=float(initial_heldout["heldout_loss"]),
        config=config,
    )
    for row in rows.values():
        row["general_loss_regression"] = float(row["heldout"]["heldout_loss"]) - float(
            initial_heldout["heldout_loss"]
        )
    fidelity: dict[str, Any] = {"performed": False}
    checkpoint_saved = False
    if selected is not None:
        model.load_state_dict(states[selected], strict=True)
        selected_hash = model_state_sha256(model)
        output = save_language_model_checkpoint(
            checkpoint_output_path,
            model,
            tokenizer,
            metadata={
                **metadata,
                "architecture": "v35r_transformer_answer_objective_v39",
                "parent_checkpoint_sha256": V35R_CHECKPOINT_SHA256,
                "selected_arm": selected,
                "cumulative_tokens": 201_335_040 + int(config.token_budget),
                "external_llm_used": False,
            },
        )
        restored, restored_tokenizer, _ = load_language_model_checkpoint(output, map_location="cpu")
        restored_hash = model_state_sha256(restored)
        fidelity = {
            "performed": True,
            "checkpoint_sha256": sha256_file(output),
            "model_state_sha256_before_save": selected_hash,
            "model_state_sha256_after_reload": restored_hash,
            "model_state_exact": restored_hash == selected_hash,
            "tokenizer_exact": restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash(),
        }
        checkpoint_saved = bool(fidelity["model_state_exact"] and fidelity["tokenizer_exact"])
        if not checkpoint_saved:
            raise RuntimeError("V39 checkpoint fidelity failed")

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "created_at_unix": time.time(),
        "external_llm_used": False,
        "parent_checkpoint": {
            "path": str(checkpoint),
            "sha256": V35R_CHECKPOINT_SHA256,
            "initial_state_sha256": initial_hash,
        },
        "v38_baseline_report": {
            "path": str(v38_report),
            "sha256": V38_REPORT_SHA256,
            "best_free_accuracy": 0.46875,
            "best_general_loss": 3.112370491027832,
        },
        "config": asdict(config),
        "answer_marker_ids": marker_values,
        "answer_mask_fraction": mask_fraction,
        "initial_heldout": initial_heldout,
        "initial_relation": initial_relation,
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_selections": prepared.source_selections,
        },
        "arms": rows,
        "selected_arm": selected,
        "decision": decision,
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_fidelity": fidelity,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path, report, title="MARULHO V39 Answer-Objective Falsification"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v38-report", required=True)
    parser.add_argument("--relation-corpus", required=True)
    parser.add_argument("--relation-cases", required=True)
    parser.add_argument("--general-train", action="append", required=True)
    parser.add_argument("--general-eval", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint-output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = run_answer_objective_falsification(
        checkpoint_path=args.checkpoint,
        v38_report_path=args.v38_report,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        report_output_path=args.report,
        checkpoint_output_path=args.checkpoint_output,
        device=args.device,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
