"""Falsify V33 editable matrix state against the exact Transformer control."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import gc
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    PreparedMatchedLanguageData,
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
from marulho.training.language_editable_state_hybrid import (
    EDITABLE_STATE_HYBRID_CHECKPOINT_SURFACE,
    MarulhoEditableStateHybridLanguageModel,
    load_editable_state_hybrid_checkpoint,
    save_editable_state_hybrid_checkpoint,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)
from marulho.training.language_muon import (
    build_language_muon,
    warm_language_muon_orthogonalizer_shapes,
)


SURFACE = "marulho_editable_state_falsification.v1"
ARTIFACT_KIND = "marulho_editable_state_falsification"
ARM_NAMES = ("transformer", "editable_hybrid")
MINIMUM_DECISION_TOKENS = 16_000_000


def _tensor_mapping_sha256(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class EditableStateFalsificationConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 32
    eval_batches: int = 16
    relation_eval_batch_size: int = 8
    relation_case_limit: int = 0
    learning_rate: float = 1.0e-3
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 16121
    model_seed: int = 16131
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    width: int = 512
    layers: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    local_attention_window: int = 24
    matrix_chunk_size: int = 72
    matrix_decay_scale: float = 64.0
    minimum_loss_gain: float = 0.02
    minimum_throughput_ratio: float = 0.25
    maximum_parameter_delta_fraction: float = 0.0


def _model_config(
    *, vocab_size: int, config: EditableStateFalsificationConfig, active_path: str
) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=int(vocab_size),
        embedding_dim=int(config.width),
        state_dim=int(config.width),
        state_layers=int(config.layers),
        attention_heads=int(config.heads),
        transformer_context_length=int(config.sequence_length),
        transformer_mlp_ratio=float(config.mlp_ratio),
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path=active_path,
    )


def build_arm_model(
    arm: str, *, vocab_size: int, config: EditableStateFalsificationConfig
) -> MarulhoLanguageModel:
    if arm == "transformer":
        return MarulhoLanguageModel(
            _model_config(
                vocab_size=vocab_size,
                config=config,
                active_path="marulho_transformer_v33_control",
            )
        )
    if arm == "editable_hybrid":
        return MarulhoEditableStateHybridLanguageModel(
            _model_config(
                vocab_size=vocab_size,
                config=config,
                active_path="marulho_editable_state_hybrid_v33",
            ),
            local_attention_window=int(config.local_attention_window),
            matrix_chunk_size=int(config.matrix_chunk_size),
            matrix_decay_scale=float(config.matrix_decay_scale),
        )
    raise ValueError(f"unknown V33 arm: {arm}")


def copy_shared_initialization(
    baseline: MarulhoLanguageModel,
    candidate: MarulhoEditableStateHybridLanguageModel,
) -> dict[str, Any]:
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    shared_names = tuple(
        name
        for name, value in candidate_state.items()
        if name in baseline_state and baseline_state[name].shape == value.shape
    )
    with torch.no_grad():
        for name in shared_names:
            candidate_state[name].copy_(baseline_state[name])
    baseline_shared = {name: baseline_state[name] for name in shared_names}
    candidate_shared = {name: candidate_state[name] for name in shared_names}
    baseline_hash = _tensor_mapping_sha256(baseline_shared)
    candidate_hash = _tensor_mapping_sha256(candidate_shared)
    return {
        "shared_tensor_names": list(shared_names),
        "shared_tensor_count": len(shared_names),
        "shared_parameter_elements": sum(
            int(candidate_state[name].numel()) for name in shared_names
        ),
        "baseline_shared_sha256": baseline_hash,
        "candidate_shared_sha256": candidate_hash,
        "shared_tensors_bit_exact": baseline_hash == candidate_hash,
    }


def _training_config(
    config: EditableStateFalsificationConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        sequence_length=int(config.sequence_length),
        batch_size=int(config.batch_size),
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.gradient_clip),
        precision=str(config.precision),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
        device="cuda",
    )


def architecture_comparison(
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    if set(arms) != set(ARM_NAMES):
        return None
    baseline = arms["transformer"]
    candidate = arms["editable_hybrid"]
    baseline_loss = float(baseline["heldout"]["heldout_loss"])
    candidate_loss = float(candidate["heldout"]["heldout_loss"])
    baseline_rate = float(baseline["training"]["tokens_per_second"])
    candidate_rate = float(candidate["training"]["tokens_per_second"])
    baseline_memory = int(baseline["training"]["peak_cuda_memory_bytes"])
    candidate_memory = int(candidate["training"]["peak_cuda_memory_bytes"])
    return {
        "transformer_heldout_loss": baseline_loss,
        "editable_hybrid_heldout_loss": candidate_loss,
        "editable_hybrid_loss_gain": baseline_loss - candidate_loss,
        "transformer_free_relation_accuracy": float(
            baseline["relation"]["generation_exact_accuracy"]
        ),
        "editable_hybrid_free_relation_accuracy": float(
            candidate["relation"]["generation_exact_accuracy"]
        ),
        "transformer_tokens_per_second": baseline_rate,
        "editable_hybrid_tokens_per_second": candidate_rate,
        "editable_hybrid_throughput_ratio": candidate_rate
        / max(baseline_rate, 1.0e-9),
        "transformer_peak_cuda_memory_bytes": baseline_memory,
        "editable_hybrid_peak_cuda_memory_bytes": candidate_memory,
        "editable_hybrid_peak_memory_ratio": candidate_memory
        / max(baseline_memory, 1),
    }


def editable_state_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    processed_tokens: int,
    parameter_delta_fraction: float,
    shared_initialization_passed: bool,
    config: EditableStateFalsificationConfig,
) -> str:
    if set(arms) != set(ARM_NAMES):
        return "incomplete_v33_missing_architecture_arm"
    if float(parameter_delta_fraction) > float(
        config.maximum_parameter_delta_fraction
    ):
        return "invalid_v33_parameter_mismatch"
    if not bool(shared_initialization_passed):
        return "invalid_v33_shared_initialization_mismatch"
    if not all(
        bool(row["all_parameters_received_final_gradient"])
        and int(row["nonzero_final_gradient_elements"]) > 0
        for row in arms.values()
    ):
        return "invalid_v33_incomplete_gradient_coverage"
    if not all(
        bool(row["execution"]["warmup_loss_parity"]["passed"])
        for row in arms.values()
    ):
        return "invalid_v33_compiled_eager_parity"
    if int(processed_tokens) < MINIMUM_DECISION_TOKENS:
        return "diagnostic_v33_below_durable_token_floor"
    comparison = architecture_comparison(arms)
    if comparison is None:
        return "incomplete_v33_missing_architecture_comparison"
    if float(comparison["editable_hybrid_throughput_ratio"]) < float(
        config.minimum_throughput_ratio
    ):
        return "retire_v33_editable_state_execution_not_viable"
    if float(comparison["editable_hybrid_loss_gain"]) >= float(
        config.minimum_loss_gain
    ):
        return "advance_v33_editable_state_to_unseen_generation"
    return "retire_v33_editable_state_no_heldout_language_win"


def _assemble_report(
    *,
    config: EditableStateFalsificationConfig,
    prepared: PreparedMatchedLanguageData,
    arms: Mapping[str, Mapping[str, Any]],
    executed_arms: Sequence[str],
    shared_initialization: Mapping[str, Any],
    optimizer_warmup: Mapping[str, Any],
    tokenizer_checkpoint: Path,
    relation_cases: Path,
    checkpoint: Mapping[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    counts = {name: int(row["parameters"]) for name, row in arms.items()}
    parameter_delta = max(counts.values()) - min(counts.values()) if counts else None
    parameter_delta_fraction = (
        float(parameter_delta) / float(next(iter(counts.values())))
        if parameter_delta is not None and counts
        else None
    )
    processed_tokens = (
        min(int(row["processed_tokens"]) for row in arms.values()) if arms else 0
    )
    decision = editable_state_decision(
        arms,
        processed_tokens=processed_tokens,
        parameter_delta_fraction=(
            float("inf") if parameter_delta_fraction is None else parameter_delta_fraction
        ),
        shared_initialization_passed=bool(
            shared_initialization["shared_tensors_bit_exact"]
        ),
        config=config,
    )
    if decision == "advance_v33_editable_state_to_unseen_generation" and not bool(
        (checkpoint or {}).get("roundtrip_passed", False)
    ):
        decision = "invalid_v33_candidate_checkpoint_fidelity"
    return {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "hypothesis": (
            "exact local attention plus a separately gated continuous matrix state "
            "improves general heldout language at an exactly matched parameter budget"
        ),
        "null_hypothesis": (
            "the editable state is only a slower capacity redistribution and does "
            "not beat the Transformer on identical data"
        ),
        "architecture_boundary": {
            "event_control_enabled": False,
            "event_control_admission": (
                "only a fully active state branch with a heldout win may receive "
                "always-on, fixed-budget, and learned delta-event controls"
            ),
            "installed_runtime_changed": False,
            "checkpoint_surface_added": True,
            "checkpoint_surface": EDITABLE_STATE_HYBRID_CHECKPOINT_SURFACE,
        },
        "tokenizer": {
            "checkpoint_path": str(tokenizer_checkpoint),
            "checkpoint_sha256": sha256_file(tokenizer_checkpoint),
            "vocab_size": int(prepared.tokenizer.vocab_size),
            "vocabulary_hash": prepared.tokenizer.vocabulary_hash(),
        },
        "relation_cases": {
            "path": str(relation_cases),
            "sha256": sha256_file(relation_cases),
            "training_fraction": 0.0,
            "metrics_only": True,
        },
        "schedule": {
            "sha256": prepared.schedule_sha256,
            "step_count": int(prepared.staged.step_count),
            "tokens_per_step": int(prepared.staged.tokens_per_step),
            "processed_tokens_per_complete_arm": int(
                prepared.staged.step_count * prepared.staged.tokens_per_step
            ),
            "source_selections": prepared.source_selections,
        },
        "shared_initialization": dict(shared_initialization),
        "optimizer_orthogonalizer_warmup": dict(optimizer_warmup),
        "parameter_counts": counts,
        "parameter_delta": parameter_delta,
        "parameter_delta_fraction": parameter_delta_fraction,
        "executed_arms": list(executed_arms),
        "arms": dict(arms),
        "comparison": architecture_comparison(arms),
        "checkpoint": dict(checkpoint or {"saved": False, "reason": "not_yet_qualified"}),
        "decision": decision,
        "elapsed_seconds": float(elapsed_seconds),
        "promotion_boundary": {
            "minimum_decision_tokens": MINIMUM_DECISION_TOKENS,
            "minimum_heldout_loss_gain": float(config.minimum_loss_gain),
            "minimum_throughput_ratio": float(config.minimum_throughput_ratio),
            "unseen_generation_required_after_loss_gate": True,
            "checkpoint_saved": bool((checkpoint or {}).get("saved", False)),
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
        },
    }


def save_and_verify_qualified_checkpoint(
    *,
    path: str | Path,
    model: MarulhoEditableStateHybridLanguageModel,
    tokenizer,
    sample_input_ids: torch.Tensor,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Save once, strict-reload, and prove tensor/tokenizer/logit fidelity."""
    output = Path(path)
    expected_state_hash = _tensor_mapping_sha256(model.state_dict())
    model.eval()
    with torch.no_grad():
        expected_logits = model(
            sample_input_ids.cpu(), collect_telemetry=False
        )["logits"].detach().cpu()
    try:
        save_editable_state_hybrid_checkpoint(
            output, model, tokenizer, metadata=metadata
        )
        restored, restored_tokenizer, restored_metadata = (
            load_editable_state_hybrid_checkpoint(output, map_location="cpu")
        )
        restored.eval()
        with torch.no_grad():
            actual_logits = restored(
                sample_input_ids.cpu(), collect_telemetry=False
            )["logits"].detach().cpu()
        restored_state_hash = _tensor_mapping_sha256(restored.state_dict())
        state_passed = restored_state_hash == expected_state_hash
        tokenizer_passed = (
            restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash()
        )
        metadata_passed = restored_metadata == dict(metadata)
        logits_passed = torch.equal(actual_logits, expected_logits)
        passed = state_passed and tokenizer_passed and metadata_passed and logits_passed
        if not passed:
            output.unlink(missing_ok=True)
        return {
            "path": str(output),
            "surface": EDITABLE_STATE_HYBRID_CHECKPOINT_SURFACE,
            "saved": passed,
            "roundtrip_passed": passed,
            "state_sha256_before_save": expected_state_hash,
            "state_sha256_after_reload": restored_state_hash,
            "state_bit_exact": state_passed,
            "tokenizer_hash_exact": tokenizer_passed,
            "metadata_exact": metadata_passed,
            "sample_logits_bit_exact": logits_passed,
            "sha256": sha256_file(output) if passed else None,
        }
    except Exception as error:
        output.unlink(missing_ok=True)
        return {
            "path": str(output),
            "surface": EDITABLE_STATE_HYBRID_CHECKPOINT_SURFACE,
            "saved": False,
            "roundtrip_passed": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }


def run_editable_state_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    checkpoint_output_path: str | Path,
    config: EditableStateFalsificationConfig = EditableStateFalsificationConfig(),
    device: str = "auto",
    arm_names: Sequence[str] = ARM_NAMES,
) -> dict[str, Any]:
    resolved = _resolve_device(device)
    if resolved.type != "cuda":
        raise ValueError("V33 editable-state falsification requires CUDA")
    requested = tuple(dict.fromkeys(str(name) for name in arm_names))
    if not requested or any(name not in ARM_NAMES for name in requested):
        raise ValueError("arm_names must contain valid unique V33 arms")
    started = time.perf_counter()
    tokenizer_checkpoint = Path(tokenizer_checkpoint_path)
    relation_cases = Path(relation_cases_path)
    checkpoint_output = Path(checkpoint_output_path)
    if checkpoint_output.exists():
        raise FileExistsError(
            f"refusing to overwrite V33 candidate checkpoint: {checkpoint_output}"
        )
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=tokenizer_checkpoint,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        config=MatchedLanguageDataConfig(
            token_budget=int(config.token_budget),
            sequence_length=int(config.sequence_length),
            batch_size=int(config.batch_size),
            eval_batches=int(config.eval_batches),
            relation_fraction=0.0,
            seed=int(config.data_seed),
            sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
            sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
            sample_range_count=int(config.sample_range_count),
            schedule_mode=str(config.schedule_mode),
        ),
        device=resolved,
    )
    if int(config.relation_case_limit) > 0:
        prepared = replace(
            prepared, cases=prepared.cases[: int(config.relation_case_limit)]
        )
    if int(prepared.tokenizer.vocab_size) != 8192:
        raise ValueError("V33 requires the 8,192-token BPE")

    torch.manual_seed(int(config.model_seed))
    baseline = build_arm_model(
        "transformer", vocab_size=int(prepared.tokenizer.vocab_size), config=config
    )
    torch.manual_seed(int(config.model_seed))
    candidate = build_arm_model(
        "editable_hybrid",
        vocab_size=int(prepared.tokenizer.vocab_size),
        config=config,
    )
    assert isinstance(candidate, MarulhoEditableStateHybridLanguageModel)
    shared_initialization = copy_shared_initialization(baseline, candidate)
    models = {"transformer": baseline, "editable_hybrid": candidate}
    initial_states = {
        name: {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        for name, model in models.items()
    }
    if len({sum(p.numel() for p in model.parameters()) for model in models.values()}) != 1:
        raise ValueError("V33 architecture parameter counts differ")
    muon_shapes: set[tuple[int, int, int]] = set()
    for model in models.values():
        shape_counts = Counter(
            tuple(int(value) for value in parameter.shape)
            for name, parameter in model.named_parameters()
            if parameter.ndim == 2
            and not name.startswith("token_embedding.")
            and not name.startswith("lm_head.")
        )
        muon_shapes.update(
            (int(count), int(shape[0]), int(shape[1]))
            for shape, count in shape_counts.items()
        )
    optimizer_warmup = warm_language_muon_orthogonalizer_shapes(
        muon_shapes, device=resolved
    )

    output = Path(output_path)
    rows: dict[str, Mapping[str, Any]] = {}
    executed: list[str] = []
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    try:
        for arm in requested:
            model = models[arm].to(resolved)
            model.eval()
            initial_heldout = evaluate_language_model(model, prepared.eval_batches)
            model.train()
            warm_batch = prepared.staged.batch(0, resolved)
            training_config = _training_config(config)
            print(f"[editable-v33] compiling {arm}", flush=True)
            training_loss, execution = _prepare_language_loss_backend(
                model, warm_batch, training_config
            )

            def optimizer_builder(model_value, config_value):
                return build_language_muon(
                    model_value,
                    learning_rate=float(config_value.learning_rate),
                    weight_decay=float(config_value.weight_decay),
                    adamw_betas=(
                        float(config_value.adam_beta1),
                        float(config_value.adam_beta2),
                    ),
                )

            print(f"[editable-v33] training {arm}", flush=True)
            row = run_matched_training_arm(
                arm,
                architecture=(
                    "causal_transformer_control"
                    if arm == "transformer"
                    else "editable_state_local_attention_hybrid"
                ),
                model=model,
                initial_state=initial_states[arm],
                training_loss=training_loss,
                execution=execution,
                allocated_compile_seconds=float(execution["compile_seconds"]),
                prepared=prepared,
                training_config=training_config,
                gradient_clip=float(config.gradient_clip),
                precision=str(config.precision),
                relation_eval_batch_size=int(config.relation_eval_batch_size),
                model_seed=int(config.model_seed),
                device=resolved,
                progress_prefix="editable-v33",
                extra_row={
                    "initial_heldout": initial_heldout,
                    "initial_state_sha256": _tensor_mapping_sha256(
                        initial_states[arm]
                    ),
                    "relation_training_fraction": 0.0,
                },
                optimizer_builder=optimizer_builder,
                optimizer_warmup_steps=3,
            )
            rows[arm] = row
            executed.append(arm)
            if len(rows) < len(ARM_NAMES):
                report = _assemble_report(
                    config=config,
                    prepared=prepared,
                    arms=rows,
                    executed_arms=executed,
                    shared_initialization=shared_initialization,
                    optimizer_warmup=optimizer_warmup,
                    tokenizer_checkpoint=tokenizer_checkpoint,
                    relation_cases=relation_cases,
                    checkpoint=None,
                    elapsed_seconds=time.perf_counter() - started,
                )
                write_json_report_with_readme(
                    output, report, title="MARULHO V33 Editable-State Falsification"
                )
            print(
                f"[editable-v33] {arm} loss={row['heldout']['heldout_loss']:.4f} "
                f"rate={row['training']['tokens_per_second']:.0f}",
                flush=True,
            )
            models[arm] = model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
        torch.set_float32_matmul_precision(previous_precision)
    preliminary = editable_state_decision(
        rows,
        processed_tokens=min(int(row["processed_tokens"]) for row in rows.values()),
        parameter_delta_fraction=0.0,
        shared_initialization_passed=bool(
            shared_initialization["shared_tensors_bit_exact"]
        ),
        config=config,
    )
    checkpoint: Mapping[str, Any]
    if preliminary == "advance_v33_editable_state_to_unseen_generation":
        qualified = models["editable_hybrid"]
        assert isinstance(qualified, MarulhoEditableStateHybridLanguageModel)
        checkpoint = save_and_verify_qualified_checkpoint(
            path=checkpoint_output,
            model=qualified,
            tokenizer=prepared.tokenizer,
            sample_input_ids=prepared.eval_batches[0].input_ids[:1],
            metadata={
                "source_surface": SURFACE,
                "source_report": str(output),
                "processed_tokens": int(rows["editable_hybrid"]["processed_tokens"]),
                "decision_target": "unseen_generation_only",
            },
        )
    else:
        checkpoint = {
            "path": str(checkpoint_output),
            "surface": EDITABLE_STATE_HYBRID_CHECKPOINT_SURFACE,
            "saved": False,
            "roundtrip_passed": False,
            "reason": "candidate_did_not_pass_durable_loss_and_execution_gate",
        }
    report = _assemble_report(
        config=config,
        prepared=prepared,
        arms=rows,
        executed_arms=executed,
        shared_initialization=shared_initialization,
        optimizer_warmup=optimizer_warmup,
        tokenizer_checkpoint=tokenizer_checkpoint,
        relation_cases=relation_cases,
        checkpoint=checkpoint,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json_report_with_readme(
        output, report, title="MARULHO V33 Editable-State Falsification"
    )
    print(f"[editable-v33] decision {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_777_216)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--relation-case-limit", type=int, default=0)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--arm", action="append", choices=ARM_NAMES, default=[])
    parser.add_argument(
        "--execution-backend", choices=("eager", "inductor"), default="inductor"
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_editable_state_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        checkpoint_output_path=args.checkpoint_output,
        config=EditableStateFalsificationConfig(
            token_budget=max(1, int(args.token_budget)),
            batch_size=max(1, int(args.batch_size)),
            eval_batches=max(1, int(args.eval_batches)),
            relation_case_limit=max(0, int(args.relation_case_limit)),
            sample_bytes_per_train_source=max(1, int(args.train_sample_mib))
            * 1024
            * 1024,
            sample_bytes_per_eval_source=max(1, int(args.eval_sample_mib))
            * 1024
            * 1024,
            execution_backend=str(args.execution_backend),
        ),
        device=args.device,
        arm_names=tuple(args.arm) or ARM_NAMES,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
