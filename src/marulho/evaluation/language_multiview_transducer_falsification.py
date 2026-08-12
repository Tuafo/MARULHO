"""Run V55 multi-view autoregressive answer-transducer falsification."""

from __future__ import annotations

import argparse
from collections import Counter
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
from marulho.training.language_multiview_transducer import (
    FrozenBaseMultiViewAnswerTransducer,
    build_multiview_supervision_batches,
    cache_frozen_causal_hidden,
    load_multiview_transducer_checkpoint,
    save_multiview_transducer_checkpoint,
)


SURFACE = "marulho_multiview_transducer_falsification.v1"
ARCHITECTURE = "frozen_v39_multiview_autoregressive_pointer_transducer"


@dataclass(frozen=True)
class MultiViewTransducerFalsificationConfig:
    epoch_count: int = 15
    token_budget: int = 8_847_360
    sequence_length: int = 72
    batch_size: int = 64
    encoder_width: int = 192
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 6
    maximum_answer_tokens: int = 8
    span_loss_weight: float = 0.25
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 55121
    model_seed: int = 55131
    both_view_fraction: float = 0.70
    bidirectional_only_fraction: float = 0.15
    causal_only_fraction: float = 0.15
    relation_case_count: int = 64
    relation_eval_batch_size: int = 8
    relation_generation_tokens: int = 16
    eval_batches: int = 16
    maximum_cache_plus_training_seconds: float = 1_200.0
    maximum_parameter_fraction: float = 0.025
    minimum_grounding_accuracy: float = 0.50
    minimum_source_gain: float = 0.45
    minimum_multiview_advantage: float = 0.0625


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def multiview_transducer_gate(
    row: Mapping[str, Any],
    *,
    ablations: Mapping[str, Mapping[str, Any]],
    parent: Mapping[str, Any],
    checkpoint_fidelity: Mapping[str, Any],
    transducer_parameters: int,
    parent_parameters: int,
    config: MultiViewTransducerFalsificationConfig,
) -> dict[str, Any]:
    source = dict(row["source_grounding"])
    intact = float(source["intact_source"]["exact_answer_accuracy"])
    source_gain = float(source["intact_gain_over_stronger_control"])
    single_view_accuracies = {
        name: float(report["intact_source"]["exact_answer_accuracy"])
        for name, report in ablations.items()
    }
    best_single = max(single_view_accuracies.values())
    multiview_advantage = intact - best_single
    parameter_fraction = int(transducer_parameters) / int(parent_parameters)
    checks = {
        "source_valid": bool(source["valid"]),
        "ablations_valid": all(
            bool(report["valid"]) for report in ablations.values()
        ),
        "minimum_grounding_accuracy": intact
        >= float(config.minimum_grounding_accuracy),
        "minimum_source_gain": source_gain >= float(config.minimum_source_gain),
        "minimum_multiview_advantage": multiview_advantage
        >= float(config.minimum_multiview_advantage),
        "exact_token_budget": int(row["processed_tokens"])
        == int(config.token_budget),
        "exact_epoch_count": int(row["epoch_count"]) == int(config.epoch_count),
        "all_transducer_parameters_received_gradient": bool(
            row["all_parameters_received_final_gradient"]
        ),
        "all_transducer_parameters_received_nonzero_gradient": bool(
            row["all_parameters_received_final_nonzero_gradient"]
        ),
        "all_view_modes_trained": all(
            int(row["view_mode_counts"].get(name, 0)) > 0
            for name in ("both", "bidirectional_only", "causal_only")
        ),
        "maximum_parameter_fraction": parameter_fraction
        <= float(config.maximum_parameter_fraction),
        "bounded_cache_plus_training_time": float(
            row["cache_plus_training_seconds"]
        )
        <= float(config.maximum_cache_plus_training_seconds),
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
            "single_view_accuracies": single_view_accuracies,
            "best_single_view_accuracy": best_single,
            "multiview_advantage": multiview_advantage,
            "transducer_parameters": int(transducer_parameters),
            "parent_parameters": int(parent_parameters),
            "parameter_fraction": parameter_fraction,
            "cache_plus_training_seconds": float(
                row["cache_plus_training_seconds"]
            ),
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


def _training_schedule(
    *,
    batch_count: int,
    config: MultiViewTransducerFalsificationConfig,
) -> tuple[list[int], list[str], str]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.data_seed))
    batch_schedule: list[int] = []
    view_schedule: list[str] = []
    both_count = int(round(batch_count * float(config.both_view_fraction)))
    bidirectional_count = int(
        round(batch_count * float(config.bidirectional_only_fraction))
    )
    causal_count = batch_count - both_count - bidirectional_count
    if min(both_count, bidirectional_count, causal_count) < 1:
        raise ValueError("V55 view schedule must contain every mode per epoch")
    epoch_modes = (
        ["both"] * both_count
        + ["bidirectional_only"] * bidirectional_count
        + ["causal_only"] * causal_count
    )
    for _epoch in range(int(config.epoch_count)):
        batch_schedule.extend(
            torch.randperm(batch_count, generator=generator).tolist()
        )
        mode_order = torch.randperm(batch_count, generator=generator).tolist()
        view_schedule.extend(epoch_modes[index] for index in mode_order)
    if view_schedule[-1] != "both":
        swap = next(
            index
            for index in range(len(view_schedule) - 2, -1, -1)
            if view_schedule[index] == "both"
        )
        view_schedule[swap], view_schedule[-1] = (
            view_schedule[-1],
            view_schedule[swap],
        )
    encoded = torch.tensor(
        [
            (batch_index << 2)
            | {"both": 0, "bidirectional_only": 1, "causal_only": 2}[mode]
            for batch_index, mode in zip(
                batch_schedule, view_schedule, strict=True
            )
        ],
        dtype=torch.int64,
    )
    return (
        batch_schedule,
        view_schedule,
        hashlib.sha256(encoded.numpy().tobytes()).hexdigest(),
    )


def _train_transducer(
    model: FrozenBaseMultiViewAnswerTransducer,
    batches,
    *,
    cache_report: Mapping[str, Any],
    config: MultiViewTransducerFalsificationConfig,
) -> dict[str, Any]:
    batch_schedule, view_schedule, schedule_sha256 = _training_schedule(
        batch_count=len(batches), config=config
    )
    step_count = len(batch_schedule)
    tokens_per_step = int(config.batch_size) * int(config.sequence_length)
    if step_count * tokens_per_step != int(config.token_budget):
        raise ValueError("V55 epoch schedule differs from the token budget")
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
    received_gradient = {name: False for name, _parameter in trainable}
    received_nonzero_gradient = {name: False for name, _parameter in trainable}
    trace = []
    mode_counts = Counter(view_schedule)
    model.train()
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    initial_loss = None
    final_loss = None
    final_pointer_loss = None
    final_span_loss = None
    for step, (batch_index, view_mode) in enumerate(
        zip(batch_schedule, view_schedule, strict=True)
    ):
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
            loss, components = model.loss(batch, view_mode=view_mode)
        if not bool(torch.isfinite(loss).item()):
            raise ValueError(f"V55 loss became non-finite at step {step}")
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
        pointer_value = float(components["pointer_loss"].item())
        span_value = float(components["span_loss"].item())
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        final_pointer_loss = pointer_value
        final_span_loss = span_value
        if step in {
            0,
            warmup_steps - 1,
            step_count // 4,
            step_count // 2,
            3 * step_count // 4,
            step_count - 1,
        }:
            trace.append(
                {
                    "step": step + 1,
                    "epoch": step // len(batches) + 1,
                    "view_mode": view_mode,
                    "loss": value,
                    "pointer_loss": pointer_value,
                    "span_loss": span_value,
                    "learning_rate": learning_rate,
                    "gradient_norm": float(gradient_norm.detach().item()),
                }
            )
        if (step + 1) % 128 == 0 or step + 1 == step_count:
            print(
                f"[multiview-v55] epoch {(step + 1) // len(batches)}/"
                f"{config.epoch_count} step {step + 1}/{step_count} "
                f"loss={value:.4f} pointer={pointer_value:.4f}",
                flush=True,
            )
    torch.cuda.synchronize(model.device)
    training_seconds = time.perf_counter() - started
    final_gradient = {
        name: parameter.grad is not None for name, parameter in trainable
    }
    final_nonzero_gradient = {
        name: parameter.grad is not None
        and bool(parameter.grad.detach().ne(0).any().item())
        for name, parameter in trainable
    }
    cache_plus_training = float(cache_report["elapsed_seconds"]) + training_seconds
    return {
        "architecture": ARCHITECTURE,
        "processed_tokens": int(config.token_budget),
        "epoch_count": int(config.epoch_count),
        "optimizer_steps": step_count,
        "tokens_per_step": tokens_per_step,
        "training_seconds": training_seconds,
        "cache_plus_training_seconds": cache_plus_training,
        "training_tokens_per_second": int(config.token_budget) / training_seconds,
        "cache_amortized_tokens_per_second": int(config.token_budget)
        / cache_plus_training,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_pointer_loss": final_pointer_loss,
        "final_span_loss": final_span_loss,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
        "optimizer_state_bytes": _optimizer_state_bytes(optimizer),
        "schedule_sha256": schedule_sha256,
        "view_mode_counts": dict(sorted(mode_counts.items())),
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


def run_multiview_transducer_falsification(
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
    config: MultiViewTransducerFalsificationConfig = (
        MultiViewTransducerFalsificationConfig()
    ),
) -> dict[str, Any]:
    if len(general_train_paths) < 2 or len(general_eval_paths) != 2:
        raise ValueError("V55 requires at least two train and exactly two eval sources")
    if not torch.cuda.is_available():
        raise ValueError("V55 is a CUDA-only evidence run")
    device = torch.device("cuda")
    checkpoint = Path(checkpoint_path)
    checkpoint_sha_before = sha256_file(checkpoint)
    base, tokenizer, checkpoint_metadata = load_language_model_checkpoint(
        checkpoint, map_location="cpu"
    )
    if int(base.context_length) != int(config.sequence_length):
        raise ValueError("V55 must preserve the V39 context length")
    parent_parameters = sum(parameter.numel() for parameter in base.parameters())
    parent_state_before = language_model_state_sha256(base)
    torch.manual_seed(int(config.model_seed))
    torch.cuda.manual_seed_all(int(config.model_seed))
    base = base.to(device)
    model = FrozenBaseMultiViewAnswerTransducer(
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
        bos_id=int(tokenizer.bos_id),
        pad_id=int(tokenizer.pad_id),
        eos_id=int(tokenizer.eos_id),
        width=int(config.encoder_width),
        encoder_layers=int(config.encoder_layers),
        decoder_layers=int(config.decoder_layers),
        heads=int(config.heads),
        maximum_answer_tokens=int(config.maximum_answer_tokens),
        span_loss_weight=float(config.span_loss_weight),
    ).to(device)
    transducer_parameters = model.transducer_parameter_count()
    if transducer_parameters / parent_parameters > float(
        config.maximum_parameter_fraction
    ):
        raise ValueError("V55 transducer exceeds the parameter-fraction gate")
    training_manifest = load_squad_grounding_manifest(
        grounding_training_manifest_path, tokenizer
    )
    training_manifest["path"] = str(grounding_training_manifest_path)
    training_batches, supervision = build_multiview_supervision_batches(
        training_manifest,
        tokenizer,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        maximum_answer_tokens=int(config.maximum_answer_tokens),
    )
    if len(training_batches) * int(config.batch_size) != 8_192:
        raise ValueError("V55 requires the frozen 8,192-case manifest")
    validation = load_squad_grounding_manifest(
        grounding_validation_manifest_path, tokenizer
    )
    validation["path"] = str(grounding_validation_manifest_path)
    training_ids = {str(case["case_id"]) for case in training_manifest["cases"]}
    validation_ids = {str(case["case_id"]) for case in validation["cases"]}
    if training_ids & validation_ids:
        raise ValueError("V55 training and validation case IDs overlap")
    print("[multiview-v55] preparing immutable parent audits", flush=True)
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
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
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
            baseline_logits = base(
                sample.input_ids, collect_telemetry=False
            )["logits"].detach().cpu()
        torch.cuda.reset_peak_memory_stats(device)
        print("[multiview-v55] caching frozen V39 causal states", flush=True)
        cached_batches, cache_report = cache_frozen_causal_hidden(
            base, training_batches, device=device
        )
        print("[multiview-v55] training multi-view transducer", flush=True)
        row = _train_transducer(
            model, cached_batches, cache_report=cache_report, config=config
        )
        row.update(
            {
                "parent_frozen": True,
                "replay_used": False,
                "transducer_parameters": transducer_parameters,
                "parent_parameters": parent_parameters,
            }
        )
        frozen_contract = {
            "surface": SURFACE,
            "configuration": asdict(config),
            "checkpoint_sha256": checkpoint_sha_before,
            "parent_state_sha256": parent_state_before,
            "training_manifest_contract": training_manifest["contract_sha256"],
            "validation_manifest_contract": validation["contract_sha256"],
            "supervision": supervision,
            "cache_content_sha256": cache_report["content_sha256"],
            "schedule_sha256": row["schedule_sha256"],
        }
        experiment_contract_sha256 = _canonical_sha256(frozen_contract)
        row["experiment_contract_sha256"] = experiment_contract_sha256
        arm = Path(arm_artifact_path)
        candidate = Path(candidate_checkpoint_path)
        arm.unlink(missing_ok=True)
        candidate.unlink(missing_ok=True)
        save_multiview_transducer_checkpoint(
            arm,
            model,
            parent_checkpoint_sha256=checkpoint_sha_before,
            metadata={
                "source_experiment": SURFACE,
                "processed_tokens": int(config.token_budget),
                "experiment_contract_sha256": experiment_contract_sha256,
            },
        )
        source_reports = {}
        for view_mode in ("both", "bidirectional_only", "causal_only"):
            model.set_inference_view_mode(view_mode)
            source_path = Path(output_path).with_name(
                f"{Path(output_path).stem}-{view_mode}-source.json"
            )
            print(f"[multiview-v55] evaluating {view_mode}", flush=True)
            source_reports[view_mode] = evaluate_source_grounding(
                model,
                tokenizer,
                validation,
                checkpoint_path=arm,
                output_path=source_path,
                max_new_tokens=int(config.maximum_answer_tokens),
            )
        model.set_inference_view_mode("both")
        row["source_grounding"] = source_reports["both"]
        ablations = {
            name: report
            for name, report in source_reports.items()
            if name != "both"
        }
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
            final_logits = base(
                sample.input_ids, collect_telemetry=False
            )["logits"].detach().cpu()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
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
    restored, restored_metadata = load_multiview_transducer_checkpoint(
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
    gate = multiview_transducer_gate(
        row,
        ablations=ablations,
        parent=parent,
        checkpoint_fidelity=checkpoint_fidelity,
        transducer_parameters=transducer_parameters,
        parent_parameters=parent_parameters,
        config=config,
    )
    decision = (
        "advance_v55_multiview_transducer_to_multisource_routing"
        if bool(gate["passed"])
        else "retire_v55_multiview_transducer_capability_or_ablation_failure"
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
            "training_validation_case_overlap": len(training_ids & validation_ids),
            "validation_manifest_path": str(grounding_validation_manifest_path),
            "validation_manifest_contract_sha256": validation["contract_sha256"],
            "general_source_selections": prepared.source_selections,
            "supervision": supervision,
            "causal_hidden_cache": cache_report,
        },
        "baseline": {"general": baseline_general, "relation": baseline_relation},
        "arm": row,
        "source_ablations": ablations,
        "parent": parent,
        "checkpoint_fidelity": checkpoint_fidelity,
        "gate": gate,
        "boundary": (
            "V55 tests a frozen-parent two-view autoregressive extractive source "
            "organ. It does not prove multi-document retrieval, abstractive "
            "synthesis, open-domain reasoning, or learned base-model routing."
        ),
    }
    write_json_report_with_readme(
        output_path, report, title="MARULHO V55 Multi-View Answer Transducer"
    )
    del cached_batches
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
    report = run_multiview_transducer_falsification(
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
