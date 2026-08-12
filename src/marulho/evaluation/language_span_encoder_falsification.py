"""Run V54 frozen-base trainable source-encoder falsification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    prepare_matched_language_data,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_source_grounding import (
    evaluate_source_grounding,
    load_squad_grounding_manifest,
)
from marulho.evaluation.language_source_grounding_continual import (
    _stratified_relation_cases,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_span_encoder import (
    FrozenBaseSpanEncoder,
    build_span_supervision_batches,
    load_span_encoder_checkpoint,
    save_span_encoder_checkpoint,
)


SURFACE = "marulho_span_encoder_falsification.v1"
ARCHITECTURE = "frozen_v39_plus_bidirectional_width128_span_encoder"


@dataclass(frozen=True)
class SpanEncoderFalsificationConfig:
    token_budget: int = 2_096_640
    sequence_length: int = 72
    batch_size: int = 64
    encoder_width: int = 128
    encoder_layers: int = 2
    encoder_heads: int = 4
    maximum_answer_tokens: int = 8
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 54121
    model_seed: int = 54131
    relation_case_count: int = 64
    relation_eval_batch_size: int = 8
    relation_generation_tokens: int = 16
    eval_batches: int = 16
    maximum_training_seconds: float = 1_200.0
    maximum_parameter_fraction: float = 0.0075
    minimum_grounding_accuracy: float = 0.296875
    minimum_source_gain: float = 0.25


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def span_encoder_gate(
    row: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    checkpoint_fidelity: Mapping[str, Any],
    encoder_parameters: int,
    parent_parameters: int,
    config: SpanEncoderFalsificationConfig,
) -> dict[str, Any]:
    source = dict(row["source_grounding"])
    intact = float(source["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source["intact_gain_over_stronger_control"])
    parameter_fraction = int(encoder_parameters) / int(parent_parameters)
    checks = {
        "source_valid": bool(source["valid"]),
        "minimum_grounding_accuracy": intact
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "exact_token_budget": int(row["processed_tokens"])
        == int(config.token_budget),
        "all_encoder_parameters_received_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "all_encoder_parameters_received_nonzero_gradient": bool(
            row["all_parameters_received_final_nonzero_gradient"]
        ),
        "maximum_parameter_fraction": parameter_fraction
        <= float(config.maximum_parameter_fraction),
        "bounded_training_time": float(row["elapsed_seconds"])
        <= float(config.maximum_training_seconds),
        "parent_checkpoint_file_exact": bool(parent["checkpoint_file_exact"]),
        "parent_state_exact": bool(parent["state_exact"]),
        "parent_logits_exact": bool(parent["logits_exact"]),
        "parent_general_loss_exact": bool(parent["general_loss_exact"]),
        "parent_relation_exact": bool(parent["relation_exact"]),
        "checkpoint_fidelity": bool(checkpoint_fidelity["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "observed": {
            "grounding_accuracy": intact,
            "source_gain": source_gain,
            "encoder_parameters": int(encoder_parameters),
            "parent_parameters": int(parent_parameters),
            "parameter_fraction": parameter_fraction,
            "elapsed_seconds": float(row["elapsed_seconds"]),
        },
        "thresholds": asdict(config),
    }


def _learning_rate_factor(
    step: int,
    *,
    step_count: int,
    warmup_steps: int,
    minimum_fraction: float,
) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, step_count - warmup_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return float(minimum_fraction) + (1.0 - float(minimum_fraction)) * cosine


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        int(value.numel() * value.element_size())
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )


def _train_span_encoder(
    model: FrozenBaseSpanEncoder,
    batches,
    *,
    config: SpanEncoderFalsificationConfig,
) -> dict[str, Any]:
    tokens_per_step = int(config.batch_size) * int(config.sequence_length)
    if int(config.token_budget) % tokens_per_step:
        raise ValueError("V54 token budget must divide into exact optimizer steps")
    step_count = int(config.token_budget) // tokens_per_step
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [parameter for _name, parameter in trainable],
        lr=float(config.learning_rate),
        betas=(0.9, 0.95),
        weight_decay=float(config.weight_decay),
        fused=model.device.type == "cuda",
    )
    warmup_steps = max(1, math.ceil(step_count * float(config.warmup_fraction)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.data_seed))
    schedule: list[int] = []
    while len(schedule) < step_count:
        schedule.extend(torch.randperm(len(batches), generator=generator).tolist())
    schedule = schedule[:step_count]
    schedule_sha256 = hashlib.sha256(
        torch.tensor(schedule, dtype=torch.int64).numpy().tobytes()
    ).hexdigest()
    trace = []
    received_gradient = {name: False for name, _parameter in trainable}
    received_nonzero_gradient = {name: False for name, _parameter in trainable}
    model.train()
    torch.cuda.reset_peak_memory_stats(model.device)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    initial_loss = None
    final_loss = None
    for step, batch_index in enumerate(schedule):
        batch = batches[batch_index].to(model.device)
        optimizer.zero_grad(set_to_none=True)
        factor = _learning_rate_factor(
            step,
            step_count=step_count,
            warmup_steps=warmup_steps,
            minimum_fraction=float(config.minimum_learning_rate_fraction),
        )
        learning_rate = float(config.learning_rate) * factor
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=str(config.precision) == "bfloat16",
        ):
            loss = model.loss(batch)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"V54 loss became non-finite at step {step}")
        loss.backward()
        for name, parameter in trainable:
            if parameter.grad is not None:
                received_gradient[name] = True
                if bool(parameter.grad.detach().ne(0).any().item()):
                    received_nonzero_gradient[name] = True
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for _name, parameter in trainable],
            max_norm=float(config.gradient_clip),
        )
        optimizer.step()
        value = float(loss.detach().item())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        if step in {0, warmup_steps - 1, step_count // 2, step_count - 1}:
            trace.append(
                {
                    "step": step + 1,
                    "loss": value,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.detach().item()),
                }
            )
        if (step + 1) % 50 == 0 or step + 1 == step_count:
            print(
                f"[span-encoder-v54] step {step + 1}/{step_count} "
                f"loss={value:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    final_gradient = {
        name: parameter.grad is not None for name, parameter in trainable
    }
    final_nonzero_gradient = {
        name: parameter.grad is not None
        and bool(parameter.grad.detach().ne(0).any().item())
        for name, parameter in trainable
    }
    return {
        "architecture": ARCHITECTURE,
        "processed_tokens": int(config.token_budget),
        "optimizer_steps": step_count,
        "tokens_per_step": tokens_per_step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": int(config.token_budget) / elapsed,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
        "schedule_sha256": schedule_sha256,
        "warmup_steps": warmup_steps,
        "trace": trace,
        "all_parameters_received_gradient": all(received_gradient.values()),
        "all_parameters_received_nonzero_gradient": all(
            received_nonzero_gradient.values()
        ),
        "all_parameters_received_final_gradient": all(final_gradient.values()),
        "all_parameters_received_final_nonzero_gradient": all(
            final_nonzero_gradient.values()
        ),
        "missing_gradient_parameters": [
            name for name, seen in received_gradient.items() if not seen
        ],
        "zero_gradient_parameters": [
            name for name, seen in received_nonzero_gradient.items() if not seen
        ],
        "final_missing_gradient_parameters": [
            name for name, seen in final_gradient.items() if not seen
        ],
        "final_zero_gradient_parameters": [
            name for name, seen in final_nonzero_gradient.items() if not seen
        ],
    }


def run_span_encoder_falsification(
    *,
    checkpoint_path: str | Path,
    grounding_training_manifest_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    grounding_validation_manifest_path: str | Path,
    output_path: str | Path,
    arm_artifact_path: str | Path,
    candidate_checkpoint_path: str | Path,
    config: SpanEncoderFalsificationConfig = SpanEncoderFalsificationConfig(),
) -> dict[str, Any]:
    if len(general_train_paths) < 2 or len(general_eval_paths) != 2:
        raise ValueError("V54 requires at least two train and exactly two eval sources")
    if not torch.cuda.is_available():
        raise ValueError("V54 is a CUDA-only evidence run")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    base, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if int(base.context_length) != int(config.sequence_length):
        raise ValueError("V54 must preserve the V39 context length")
    parent_parameters = sum(parameter.numel() for parameter in base.parameters())
    parent_state_before = language_model_state_sha256(base)
    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    base = base.to(device)
    model = FrozenBaseSpanEncoder(
        base,
        context_marker_ids=torch.tensor(
            tokenizer.encode("Context:", add_bos=False, add_eos=False)
        ),
        question_marker_ids=torch.tensor(
            tokenizer.encode("\nQuestion:", add_bos=False, add_eos=False)
        ),
        answer_marker_ids=torch.tensor(
            tokenizer.encode("\nAnswer:", add_bos=False, add_eos=False)
        ),
        width=int(config.encoder_width),
        layers=int(config.encoder_layers),
        heads=int(config.encoder_heads),
        maximum_answer_tokens=int(config.maximum_answer_tokens),
    ).to(device)
    encoder_parameters = model.encoder_parameter_count()
    if encoder_parameters / parent_parameters > float(
        config.maximum_parameter_fraction
    ):
        raise ValueError("V54 encoder exceeds the preregistered parameter fraction")
    training_manifest = load_squad_grounding_manifest(
        grounding_training_manifest_path, tokenizer
    )
    training_manifest["path"] = str(grounding_training_manifest_path)
    training_batches, supervision = build_span_supervision_batches(
        training_manifest,
        tokenizer,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
    )
    validation = load_squad_grounding_manifest(
        grounding_validation_manifest_path, tokenizer
    )
    validation["path"] = str(grounding_validation_manifest_path)
    print("[span-encoder-v54] preparing immutable parent audits", flush=True)
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=checkpoint,
        relation_corpus_path=grounding_training_manifest_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.batch_size) * int(config.sequence_length),
            sequence_length=int(config.sequence_length),
            batch_size=8,
            eval_batches=int(config.eval_batches),
            relation_fraction=0.0,
            seed=int(config.data_seed),
            sample_bytes_per_train_source=1 * 1024 * 1024,
            sample_bytes_per_eval_source=1 * 1024 * 1024,
            sample_range_count=8,
            schedule_mode="indexed_host",
        ),
        device=device,
    )
    prepared = replace(
        prepared,
        cases=_stratified_relation_cases(
            prepared.cases, case_count=int(config.relation_case_count)
        ),
    )
    baseline_general = evaluate_language_model(base, prepared.eval_batches)
    baseline_relation = evaluate_relation_binding_cases_batched(
        base,
        tokenizer,
        prepared.cases,
        batch_size=int(config.relation_eval_batch_size),
        max_new_tokens=int(config.relation_generation_tokens),
    )
    sample = prepared.eval_batches[0].to(device)
    with torch.no_grad():
        baseline_logits = base(sample.input_ids, collect_telemetry=False)[
            "logits"
        ].detach().cpu()
    frozen_contract = {
        "surface": SURFACE,
        "configuration": asdict(config),
        "checkpoint_sha256": checkpoint_sha_before,
        "parent_state_sha256": parent_state_before,
        "training_manifest_contract": training_manifest["contract_sha256"],
        "validation_manifest_contract": validation["contract_sha256"],
        "supervision": supervision,
    }
    experiment_contract_sha256 = _canonical_sha256(frozen_contract)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        print("[span-encoder-v54] training direct span objective", flush=True)
        row = _train_span_encoder(model, training_batches, config=config)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    row.update(
        {
            "parent_frozen": True,
            "replay_used": False,
            "encoder_parameters": encoder_parameters,
            "parent_parameters": parent_parameters,
            "experiment_contract_sha256": experiment_contract_sha256,
        }
    )
    arm = Path(arm_artifact_path)
    candidate = Path(candidate_checkpoint_path)
    arm.unlink(missing_ok=True)
    candidate.unlink(missing_ok=True)
    save_span_encoder_checkpoint(
        arm,
        model,
        parent_checkpoint_sha256=checkpoint_sha_before,
        metadata={
            "source_experiment": SURFACE,
            "processed_tokens": int(config.token_budget),
            "experiment_contract_sha256": experiment_contract_sha256,
        },
    )
    source_path = Path(output_path).with_name(f"{Path(output_path).stem}-source.json")
    row["source_grounding"] = evaluate_source_grounding(
        model,
        tokenizer,
        validation,
        checkpoint_path=arm,
        output_path=source_path,
        max_new_tokens=int(config.maximum_answer_tokens),
    )
    checkpoint_sha_after = sha256_file(checkpoint)
    parent_state_after = language_model_state_sha256(model.base)
    final_general = evaluate_language_model(base, prepared.eval_batches)
    final_relation = evaluate_relation_binding_cases_batched(
        base,
        tokenizer,
        prepared.cases,
        batch_size=int(config.relation_eval_batch_size),
        max_new_tokens=int(config.relation_generation_tokens),
    )
    with torch.no_grad():
        final_logits = base(sample.input_ids, collect_telemetry=False)[
            "logits"
        ].detach().cpu()
    parent = {
        "checkpoint_sha256_before": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_sha256_before": parent_state_before,
        "state_sha256_after": parent_state_after,
        "state_exact": parent_state_before == parent_state_after,
        "logits_exact": torch.equal(baseline_logits, final_logits),
        "general_loss_before": float(baseline_general["heldout_loss"]),
        "general_loss_after": float(final_general["heldout_loss"]),
        "general_loss_exact": float(baseline_general["heldout_loss"])
        == float(final_general["heldout_loss"]),
        "relation_before": baseline_relation,
        "relation_after": final_relation,
        "relation_exact": baseline_relation == final_relation,
    }
    expected_state = language_model_state_sha256(model)
    restored_base, restored_tokenizer, _metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    restored, restored_metadata = load_span_encoder_checkpoint(
        arm,
        restored_base,
        expected_parent_checkpoint_sha256=checkpoint_sha_before,
    )
    checkpoint_fidelity = {
        "performed": True,
        "arm_checkpoint_path": str(arm),
        "arm_checkpoint_sha256": sha256_file(arm),
        "arm_checkpoint_size_bytes": arm.stat().st_size,
        "expected_state_sha256": expected_state,
        "restored_state_sha256": language_model_state_sha256(restored),
        "tokenizer_hash_before": tokenizer.vocabulary_hash(),
        "tokenizer_hash_after": restored_tokenizer.vocabulary_hash(),
        "metadata": restored_metadata,
    }
    checkpoint_fidelity["passed"] = bool(
        checkpoint_fidelity["expected_state_sha256"]
        == checkpoint_fidelity["restored_state_sha256"]
        and checkpoint_fidelity["tokenizer_hash_before"]
        == checkpoint_fidelity["tokenizer_hash_after"]
    )
    gate = span_encoder_gate(
        row,
        parent=parent,
        checkpoint_fidelity=checkpoint_fidelity,
        encoder_parameters=encoder_parameters,
        parent_parameters=parent_parameters,
        config=config,
    )
    decision = (
        "advance_v54_span_encoder_to_scale_and_routing"
        if bool(gate["passed"])
        else "retire_v54_span_encoder_insufficient_grounding"
    )
    if bool(gate["passed"]):
        candidate.parent.mkdir(parents=True, exist_ok=True)
        os.replace(arm, candidate)
        checkpoint_fidelity["candidate_checkpoint_path"] = str(candidate)
        checkpoint_fidelity["candidate_checkpoint_sha256"] = sha256_file(candidate)
    else:
        arm.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
    report = {
        "surface": SURFACE,
        "decision": decision,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "configuration": asdict(config),
        "experiment_contract_sha256": experiment_contract_sha256,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha_before,
            "metadata": checkpoint_metadata,
        },
        "data": {
            "training_manifest_path": str(grounding_training_manifest_path),
            "training_manifest_sha256": sha256_file(
                grounding_training_manifest_path
            ),
            "training_manifest_contract_sha256": training_manifest[
                "contract_sha256"
            ],
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "validation_manifest_contract_sha256": validation["contract_sha256"],
            "general_source_selections": prepared.source_selections,
            "supervision": supervision,
        },
        "baseline": {"general": baseline_general, "relation": baseline_relation},
        "arm": row,
        "parent": parent,
        "checkpoint_fidelity": checkpoint_fidelity,
        "gate": gate,
        "boundary": (
            "V54 tests a frozen-parent bidirectional source/question encoder with "
            "direct extractive span supervision. It does not prove synthesis, "
            "multi-document retrieval, open-domain reasoning, or routing."
        ),
    }
    write_json_report_with_readme(
        output_path, report, title="MARULHO V54 Trainable Source Encoder"
    )
    torch.cuda.empty_cache()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--grounding-training-manifest", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", type=Path, action="append", required=True)
    parser.add_argument("--general-eval", type=Path, action="append", required=True)
    parser.add_argument("--grounding-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm-artifact", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    args = parser.parse_args()
    report = run_span_encoder_falsification(
        checkpoint_path=args.checkpoint,
        grounding_training_manifest_path=args.grounding_training_manifest,
        relation_cases_path=args.relation_cases,
        general_train_paths=tuple(args.general_train),
        general_eval_paths=tuple(args.general_eval),
        grounding_validation_manifest_path=args.grounding_validation_manifest,
        output_path=args.output,
        arm_artifact_path=args.arm_artifact,
        candidate_checkpoint_path=args.candidate_checkpoint,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
