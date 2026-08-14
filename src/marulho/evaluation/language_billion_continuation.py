"""Run V80's billion-position all-source Transformer continuation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import torch

from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_TOKENS,
    EFFECTIVE_BATCH,
    ROOT,
    SEGMENT_LENGTH,
    SEGMENTS,
    TOKENIZER_SHA256,
    _atomic_json,
    _episode,
    _gradient_audit,
    checkpoint_fidelity,
    file_sha256,
)
from marulho.evaluation.language_scale_corpus_materialization import (
    SURFACE as CORPUS_SURFACE,
    _token_tensor_sha256,
)
from marulho.evaluation.language_scale_schedule import (
    SOURCE_NAMES,
    SOURCE_SLOT_COUNTS,
    SURFACE as SCHEDULE_SURFACE,
    TOTAL_POSITIONS,
    TOTAL_SLOTS,
    _schedule_sha256,
)
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    load_language_model_state,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon
from marulho.training.language_training_snapshot import (
    capture_rng_state,
    load_language_training_snapshot,
    save_language_training_snapshot,
    tree_sha256,
)


SURFACE = "marulho_language_billion_continuation.v80"
RESUME_PREFLIGHT_SURFACE = (
    "marulho_language_billion_continuation.v80_resume_fidelity_preflight"
)
PARENT = (
    ROOT
    / "reports/language_scaling/"
    "v78-unique-document-qualified-100m-257m-20260814.pt"
)
PARENT_SHA256 = "b66753983316b5a0cf61b293d36e4fda9b15929168067a59ed95ef816da4313b"
PARENT_STATE_SHA256 = "4ebf6ae3a500a0a77a256be80bb652a3439e47310eb008c002d14312bb34b75e"
PARENT_BF16_STATE_SHA256 = (
    "299ad6daf789748008f8a4460d447acd4bfaa9582590421d742100c2b4907703"
)
PARENT_CUMULATIVE_POSITIONS = 257_429_760
TARGET_CUMULATIVE_POSITIONS = PARENT_CUMULATIVE_POSITIONS + TOTAL_POSITIONS
SCHEDULE = (
    ROOT
    / "reports/language_curriculum/"
    "v80-billion-position-schedule-20260814.pt"
)
SCHEDULE_ARTIFACT_SHA256 = "2785aa0a96ce7c26d139d88298f05bde99626f25a0c67fe3223a9d29c15ea460"
SCHEDULE_SHA256 = "a886ef762ef32f077d688f9701269ebe5a293c37661ba091c4dd73cfffb449fa"
RESUME_FIDELITY_REPORT = (
    ROOT
    / "reports/language_scaling/"
    "v80-resume-fidelity-preflight-20260814.json"
)
RESUME_FIDELITY_REPORT_SHA256 = (
    "a1ce9559d252deb4ee4cf1aa8d38704b9019278a87feb72cc1b3d82b886c4dac"
)
EVAL_ARTIFACT = (
    ROOT
    / "reports/language_curriculum/"
    "v80-scale-three-source-eval-20260814.pt"
)
EVAL_ARTIFACT_SHA256 = "4a7afb30299c79b083cdf00d5e76317159b41f72d2ab67a6f1c030c6a859bd89"
EVAL_TOKEN_SHA256 = "ce57f5b33a08fe1a8adc238e3e0e751d5c662110d4b0004307eef14af837b679"
SOURCE_ARTIFACTS = {
    "fineweb_edu": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-fineweb-edu-train-20260814.pt",
        "dc182d9d8da5bcf70d727cc64ef239269ef46d0498478a9546b209543cbce73b",
        "203750d238058d93426db243d0e3ee02b466a719d2392ce72464ce5b70017e8f",
        58_999,
    ),
    "cosmopedia_v2": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-cosmopedia-v2-train-20260814.pt",
        "24d000c88f65a554ca2d6d38a0826b2146ac63fad7fd24ebb218d41e86ed3871",
        "a13f2d07d9a284dd4332fdd9066f156ffa723da3151d294746341cf834fd5573",
        62_298,
    ),
    "dclm_edu": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-dclm-edu-train-20260814.pt",
        "72cbb6ba6c0e9723e7b52b27b380bf50041709b9a946e3ac68e8069956a07a99",
        "71730172e0d74d277efac157a8062cb21074c75a943282ed444d4d00bed6971f",
        150_910,
    ),
}
MODEL_SEED = 80_131
TRAIN_STEPS = 32_768
WARMUP_STEPS = 256
COOLDOWN_STEPS = 6_554
COOLDOWN_START_STEP = TRAIN_STEPS - COOLDOWN_STEPS
PEAK_LEARNING_RATE = 1.5e-4
MINIMUM_LEARNING_RATE = 1.5e-5
INITIAL_LEARNING_RATE = 3.0e-5
PHYSICAL_BATCH = 8
ACCUMULATION_STEPS = EFFECTIVE_BATCH // PHYSICAL_BATCH
MAXIMUM_PEAK_BYTES = 8 * 1024**3
EVAL_BATCH = 8
EVAL_DOCUMENTS_PER_SOURCE = 512
EVAL_EVERY_STEPS = 2_048
SNAPSHOT_EVERY_STEPS = 1_024
PROGRESS_EVERY_STEPS = 32
SNAPSHOTS_TO_KEEP = 2
REFERENCE_INITIAL_LATER_LOSS = 2.9828618367513022
REFERENCE_INITIAL_SOURCE_LOSSES = {
    "fineweb_edu": 3.1576766967773438,
    "cosmopedia_v2": 2.438720703125,
    "dclm_edu": 3.3521881103515625,
}
MINIMUM_OVERALL_IMPROVEMENT = 0.25
MAXIMUM_OLD_SOURCE_REGRESSION = 0.02
MINIMUM_DCLM_IMPROVEMENT = 0.20
MAXIMUM_DIFFERING_FRACTION = 1.0e-6
MAXIMUM_GRADIENT_OR_OPTIMIZER_ABSOLUTE_DIFFERENCE = 1.0e-6
MAXIMUM_GRADIENT_OR_OPTIMIZER_RELATIVE_L2 = 1.0e-7
MAXIMUM_MODEL_ABSOLUTE_DIFFERENCE = 5.0e-4
MAXIMUM_MODEL_RELATIVE_L2 = 1.0e-8


def _learning_rate(step: int) -> float:
    if step < 0 or step >= TRAIN_STEPS:
        raise ValueError("V80 step is outside its frozen schedule")
    if step < WARMUP_STEPS:
        return INITIAL_LEARNING_RATE + (
            PEAK_LEARNING_RATE - INITIAL_LEARNING_RATE
        ) * ((step + 1) / WARMUP_STEPS)
    if step < COOLDOWN_START_STEP:
        return PEAK_LEARNING_RATE
    progress = (step + 1 - COOLDOWN_START_STEP) / COOLDOWN_STEPS
    return MINIMUM_LEARNING_RATE + 0.5 * (
        PEAK_LEARNING_RATE - MINIMUM_LEARNING_RATE
    ) * (1.0 + math.cos(math.pi * progress))


def _optimizer(model: MarulhoLanguageModel):
    return build_language_muon(
        model,
        learning_rate=PEAK_LEARNING_RATE,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )


def _load_parent() -> tuple[MarulhoLanguageModel, Any, dict[str, Any]]:
    checkpoint_hash = file_sha256(PARENT)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V80 parent checkpoint changed: {checkpoint_hash}")
    model, tokenizer, metadata = load_language_model_checkpoint(PARENT, map_location="cpu")
    state_hash = language_model_state_sha256(model)
    checks = {
        "state_exact": state_hash == PARENT_STATE_SHA256,
        "tokenizer_exact": tokenizer.vocabulary_hash() == TOKENIZER_SHA256,
        "parameter_count_exact": sum(p.numel() for p in model.parameters())
        == 100_679_424,
        "context_exact": int(model.config.transformer_context_length)
        == SEGMENT_LENGTH,
        "dropout_disabled": float(model.config.transformer_dropout) == 0.0,
        "metadata_exact": metadata.get("decision")
        == "save_v78_unique_document_checkpoint_for_unseen_generation",
        "cumulative_positions_exact": int(
            metadata.get("cumulative_processed_tokens", -1)
        )
        == PARENT_CUMULATIVE_POSITIONS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V80 parent validation failed: {checks}")
    return model, tokenizer, {
        "path": str(PARENT),
        "sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "checks": checks,
    }


def _load_token_artifact(
    name: str,
    path: Path,
    artifact_hash: str,
    token_hash: str,
    documents: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    actual_hash = file_sha256(path)
    if actual_hash != artifact_hash:
        raise RuntimeError(f"V80 {name} artifact changed: {actual_hash}")
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    tokens: torch.Tensor = artifact["tokens"].to(dtype=torch.int32)
    checks = {
        "surface_exact": artifact.get("surface") == CORPUS_SURFACE,
        "source_exact": artifact.get("source_name") == name,
        "shape_exact": tuple(tokens.shape) == (documents, DOCUMENT_TOKENS),
        "token_hash_exact": _token_tensor_sha256(tokens) == token_hash,
        "artifact_token_hash_exact": artifact.get("token_sha256") == token_hash,
        "external_llm_absent": artifact.get("external_llm_used") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V80 {name} token artifact failed: {checks}")
    return tokens, {
        "path": str(path),
        "sha256": actual_hash,
        "token_sha256": token_hash,
        "documents": documents,
        "checks": checks,
    }


def load_v80_data() -> dict[str, Any]:
    sources: dict[str, torch.Tensor] = {}
    source_audits: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        path, artifact_hash, token_hash, documents = SOURCE_ARTIFACTS[name]
        sources[name], source_audits[name] = _load_token_artifact(
            name,
            path,
            artifact_hash,
            token_hash,
            documents,
        )
    eval_tokens, eval_audit = _load_token_artifact(
        "three_source_holdout",
        EVAL_ARTIFACT,
        EVAL_ARTIFACT_SHA256,
        EVAL_TOKEN_SHA256,
        3 * 512,
    )
    schedule_artifact_hash = file_sha256(SCHEDULE)
    if schedule_artifact_hash != SCHEDULE_ARTIFACT_SHA256:
        raise RuntimeError(f"V80 schedule artifact changed: {schedule_artifact_hash}")
    schedule = torch.load(SCHEDULE, map_location="cpu", weights_only=False)
    source_ids: torch.Tensor = schedule["source_ids"].to(dtype=torch.int8)
    row_ids: torch.Tensor = schedule["row_ids"].to(dtype=torch.int32)
    computed_schedule_hash = _schedule_sha256(source_ids, row_ids)
    schedule_checks = {
        "surface_exact": schedule.get("surface") == SCHEDULE_SURFACE,
        "source_names_exact": tuple(schedule.get("source_names", ())) == SOURCE_NAMES,
        "shape_exact": tuple(source_ids.shape) == (TOTAL_SLOTS,)
        and tuple(row_ids.shape) == (TOTAL_SLOTS,),
        "hash_exact": computed_schedule_hash == SCHEDULE_SHA256
        and schedule.get("schedule_sha256") == SCHEDULE_SHA256,
        "slot_counts_exact": all(
            int(torch.count_nonzero(source_ids == index).item())
            == SOURCE_SLOT_COUNTS[name]
            for index, name in enumerate(SOURCE_NAMES)
        ),
        "total_positions_exact": schedule.get("total_positions") == TOTAL_POSITIONS,
        "external_llm_absent": schedule.get("external_llm_used") is False,
    }
    if not all(schedule_checks.values()):
        raise RuntimeError(f"V80 schedule validation failed: {schedule_checks}")
    return {
        "sources": sources,
        "eval_tokens": eval_tokens,
        "source_ids": source_ids,
        "row_ids": row_ids,
        "audits": {
            "sources": source_audits,
            "eval": eval_audit,
            "schedule": {
                "path": str(SCHEDULE),
                "sha256": schedule_artifact_hash,
                "schedule_sha256": computed_schedule_hash,
                "checks": schedule_checks,
            },
        },
    }


def _scheduled_documents(
    data: Mapping[str, Any],
    *,
    offset: int,
    count: int,
) -> torch.Tensor:
    if offset < 0 or count < 1 or offset + count > TOTAL_SLOTS:
        raise ValueError("V80 scheduled document slice is out of bounds")
    source_ids: torch.Tensor = data["source_ids"][offset : offset + count]
    row_ids: torch.Tensor = data["row_ids"][offset : offset + count]
    documents = torch.empty((count, DOCUMENT_TOKENS), dtype=torch.int32)
    for index, name in enumerate(SOURCE_NAMES):
        positions = torch.nonzero(source_ids == index, as_tuple=False).flatten()
        if int(positions.numel()) == 0:
            continue
        selected_rows = row_ids.index_select(0, positions).long()
        selected = data["sources"][name].index_select(0, selected_rows)
        documents.index_copy_(0, positions, selected)
    return documents


def _schedule_slice_sha256(data: Mapping[str, Any], *, offset: int, count: int) -> str:
    return _schedule_sha256(
        data["source_ids"][offset : offset + count],
        data["row_ids"][offset : offset + count],
    )


def _clone_to_cuda(parent: MarulhoLanguageModel, *, device: torch.device) -> MarulhoLanguageModel:
    model = MarulhoLanguageModel(parent.config)
    load_language_model_state(model, parent.state_dict())
    return model.to(device=device, dtype=torch.bfloat16)


def _comparison_state(
    model: MarulhoLanguageModel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    optimizer_tensors: dict[str, torch.Tensor] = {}
    optimizer_state = optimizer.state_dict()["state"]
    for parameter_id, state in optimizer_state.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                optimizer_tensors[f"{parameter_id}.{key}"] = (
                    value.detach().cpu().clone()
                )
    return {
        "model": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
        "gradients": {
            name: parameter.grad.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        },
        "optimizer": optimizer_tensors,
    }


def _tensor_mapping_difference(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if tuple(reference) != tuple(candidate):
        raise RuntimeError("V80 comparison tensor names changed")
    element_count = 0
    differing_element_count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    reference_squared_sum = 0.0
    maximum_absolute = 0.0
    maximum_tensor = None
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise RuntimeError(f"V80 comparison tensor metadata changed: {name}")
        difference = left.float().sub(right.float())
        tensor_maximum = float(difference.abs().max().item())
        if tensor_maximum > maximum_absolute:
            maximum_absolute = tensor_maximum
            maximum_tensor = name
        element_count += int(left.numel())
        differing_element_count += int(torch.count_nonzero(difference).item())
        absolute_sum += float(difference.double().abs().sum().item())
        squared_sum += float(difference.double().square().sum().item())
        reference_squared_sum += float(left.double().square().sum().item())
        del difference
    return {
        "tensor_count": len(reference),
        "element_count": element_count,
        "differing_element_count": differing_element_count,
        "differing_fraction": differing_element_count / max(1, element_count),
        "mean_absolute": absolute_sum / max(1, element_count),
        "maximum_absolute": maximum_absolute,
        "maximum_absolute_tensor": maximum_tensor,
        "relative_l2": math.sqrt(squared_sum / max(reference_squared_sum, 1.0e-30)),
    }


def _numerically_equivalent(
    difference: Mapping[str, Any],
    *,
    maximum_absolute: float,
    maximum_relative_l2: float,
) -> bool:
    return bool(
        float(difference["differing_fraction"]) <= MAXIMUM_DIFFERING_FRACTION
        and float(difference["maximum_absolute"]) <= maximum_absolute
        and float(difference["relative_l2"]) <= maximum_relative_l2
    )


def _logical_step(
    model: MarulhoLanguageModel,
    optimizer: torch.optim.Optimizer,
    *,
    documents: torch.Tensor,
    device: torch.device,
    audit_gradients: bool,
) -> dict[str, Any]:
    if tuple(documents.shape) != (EFFECTIVE_BATCH, DOCUMENT_TOKENS):
        raise ValueError("V80 logical step requires 32 complete documents")
    optimizer.zero_grad(set_to_none=True)
    losses = torch.zeros(SEGMENTS, dtype=torch.float64)
    scalar_losses: list[float] = []
    for micro in range(ACCUMULATION_STEPS):
        batch = documents[
            micro * PHYSICAL_BATCH : (micro + 1) * PHYSICAL_BATCH
        ].to(device=device, dtype=torch.long)
        result = _episode(model, batch)
        if not bool(torch.isfinite(result["loss"]).item()):
            raise RuntimeError("V80 encountered nonfinite training loss")
        (result["loss"] / ACCUMULATION_STEPS).backward()
        losses += result["segment_losses"].cpu().double()
        scalar_losses.append(float(result["loss"].detach().float().item()))
        del batch, result
    gradient = _gradient_audit(model) if audit_gradients else None
    gradient_state_sha256 = (
        tree_sha256(
            {
                name: parameter.grad
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
        if audit_gradients
        else None
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "loss": sum(scalar_losses) / len(scalar_losses),
        "segment_losses": (losses / ACCUMULATION_STEPS).tolist(),
        "gradient_audit": gradient,
        "gradient_state_sha256": gradient_state_sha256,
    }


def _set_step_learning_rate(optimizer: torch.optim.Optimizer, step: int) -> float:
    learning_rate = _learning_rate(step)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _run_two_step_reference(
    parent: MarulhoLanguageModel,
    data: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = _clone_to_cuda(parent, device=device)
    initial_hash = language_model_state_sha256(model)
    optimizer, optimizer_report = _optimizer(model)
    results: list[dict[str, Any]] = []
    peak = 0
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(2):
        _set_step_learning_rate(optimizer, step)
        documents = _scheduled_documents(
            data,
            offset=step * EFFECTIVE_BATCH,
            count=EFFECTIVE_BATCH,
        )
        results.append(
            _logical_step(
                model,
                optimizer,
                documents=documents,
                device=device,
                audit_gradients=True,
            )
        )
        results[-1]["post_model_state_sha256"] = language_model_state_sha256(model)
        results[-1]["post_optimizer_state_sha256"] = tree_sha256(
            optimizer.state_dict()
        )
        peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
    state = {
        "initial_model_state_sha256": initial_hash,
        "final_model_state_sha256": language_model_state_sha256(model),
        "optimizer_state_sha256": tree_sha256(optimizer.state_dict()),
        "rng_state_sha256": tree_sha256(capture_rng_state()),
        "step_results": results,
        "next_schedule_offset": 2 * EFFECTIVE_BATCH,
        "next_schedule_slice_sha256": _schedule_slice_sha256(
            data,
            offset=2 * EFFECTIVE_BATCH,
            count=EFFECTIVE_BATCH,
        ),
        "peak_cuda_allocated_bytes": peak,
        "optimizer": optimizer_report,
    }
    comparison_state = _comparison_state(model, optimizer)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return state, results[0]["gradient_audit"], comparison_state


def _run_interrupted_path(
    parent: MarulhoLanguageModel,
    tokenizer: Any,
    data: Mapping[str, Any],
    *,
    snapshot_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    torch.cuda.reset_peak_memory_stats(device)
    model = _clone_to_cuda(parent, device=device)
    initial_hash = language_model_state_sha256(model)
    optimizer, _optimizer_report = _optimizer(model)
    _set_step_learning_rate(optimizer, 0)
    first_documents = _scheduled_documents(data, offset=0, count=EFFECTIVE_BATCH)
    first = _logical_step(
        model,
        optimizer,
        documents=first_documents,
        device=device,
        audit_gradients=True,
    )
    first["post_model_state_sha256"] = language_model_state_sha256(model)
    first["post_optimizer_state_sha256"] = tree_sha256(optimizer.state_dict())
    if first["gradient_audit"] is None or not first["gradient_audit"]["passed"]:
        raise RuntimeError(f"V80 interrupted first gradients failed: {first}")
    snapshot = save_language_training_snapshot(
        snapshot_path,
        model,
        tokenizer,
        optimizer,
        completed_steps=1,
        schedule_sha256=SCHEDULE_SHA256,
        next_schedule_offset=EFFECTIVE_BATCH,
        training_state={
            "initial_model_state_sha256": initial_hash,
            "gradient_audit": first["gradient_audit"],
            "curve": [],
            "run_peak_cuda_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
        },
        metadata={
            "architecture": "marulho_transformer_v80_billion_continuation",
            "parent_checkpoint_sha256": PARENT_SHA256,
            "target_cumulative_processed_tokens": TARGET_CUMULATIVE_POSITIONS,
        },
    )
    snapshot_hash = file_sha256(snapshot_path)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    restored_model, restored_tokenizer, restored_optimizer, continuation, load_audit = (
        load_language_training_snapshot(
            snapshot_path,
            optimizer_builder=_optimizer,
            device=device,
            expected_schedule_sha256=SCHEDULE_SHA256,
            restore_rng=True,
        )
    )
    if restored_tokenizer.vocabulary_hash() != TOKENIZER_SHA256:
        raise RuntimeError("V80 resumed tokenizer changed")
    restored_pre_step_model_hash = language_model_state_sha256(restored_model)
    restored_pre_step_optimizer_hash = tree_sha256(restored_optimizer.state_dict())
    completed_steps = int(continuation["completed_steps"])
    next_offset = int(continuation["next_schedule_offset"])
    _set_step_learning_rate(restored_optimizer, completed_steps)
    second_documents = _scheduled_documents(
        data,
        offset=next_offset,
        count=EFFECTIVE_BATCH,
    )
    second = _logical_step(
        restored_model,
        restored_optimizer,
        documents=second_documents,
        device=device,
        audit_gradients=True,
    )
    second["post_model_state_sha256"] = language_model_state_sha256(restored_model)
    second["post_optimizer_state_sha256"] = tree_sha256(
        restored_optimizer.state_dict()
    )
    result = {
        "initial_model_state_sha256": initial_hash,
        "final_model_state_sha256": language_model_state_sha256(restored_model),
        "optimizer_state_sha256": tree_sha256(restored_optimizer.state_dict()),
        "rng_state_sha256": tree_sha256(capture_rng_state()),
        "step_results": [first, second],
        "resumed_completed_steps": completed_steps,
        "resumed_next_schedule_offset": next_offset,
        "restored_pre_step_model_state_sha256": restored_pre_step_model_hash,
        "restored_pre_step_optimizer_state_sha256": restored_pre_step_optimizer_hash,
        "next_schedule_offset": next_offset + EFFECTIVE_BATCH,
        "next_schedule_slice_sha256": _schedule_slice_sha256(
            data,
            offset=next_offset + EFFECTIVE_BATCH,
            count=EFFECTIVE_BATCH,
        ),
        "snapshot": {
            **snapshot,
            "sha256": snapshot_hash,
        },
        "load_audit": load_audit,
        "restored_training_state": continuation["training_state"],
        "peak_cuda_allocated_bytes": max(
            int(continuation["training_state"]["run_peak_cuda_allocated_bytes"]),
            int(torch.cuda.max_memory_allocated(device)),
        ),
    }
    gradient_audit = first["gradient_audit"]
    comparison_state = _comparison_state(restored_model, restored_optimizer)
    del restored_optimizer, restored_model
    gc.collect()
    torch.cuda.empty_cache()
    return result, gradient_audit, comparison_state


def run_resume_preflight(*, output_path: Path, snapshot_path: Path) -> dict[str, Any]:
    if output_path.exists() or snapshot_path.exists():
        raise ValueError("V80 resume preflight output already exists")
    if not torch.cuda.is_available():
        raise RuntimeError("V80 resume preflight requires CUDA")
    parent, tokenizer, parent_audit = _load_parent()
    data = load_v80_data()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    reference, reference_gradients, reference_comparison = _run_two_step_reference(
        parent,
        data,
        device=device,
    )
    interrupted, interrupted_gradients, interrupted_comparison = _run_interrupted_path(
        parent,
        tokenizer,
        data,
        snapshot_path=snapshot_path,
        device=device,
    )
    numerical_difference = {
        name: _tensor_mapping_difference(
            reference_comparison[name], interrupted_comparison[name]
        )
        for name in ("model", "gradients", "optimizer")
    }
    del reference_comparison, interrupted_comparison
    exactness_observations = {
        "first_step_gradients_exact": reference["step_results"][0][
            "gradient_state_sha256"
        ]
        == interrupted["step_results"][0]["gradient_state_sha256"],
        "first_step_model_state_exact": reference["step_results"][0][
            "post_model_state_sha256"
        ]
        == interrupted["step_results"][0]["post_model_state_sha256"],
        "first_step_optimizer_state_exact": reference["step_results"][0][
            "post_optimizer_state_sha256"
        ]
        == interrupted["step_results"][0]["post_optimizer_state_sha256"],
        "second_step_gradients_exact": reference["step_results"][1][
            "gradient_state_sha256"
        ]
        == interrupted["step_results"][1]["gradient_state_sha256"],
        "final_model_state_exact": reference["final_model_state_sha256"]
        == interrupted["final_model_state_sha256"],
        "final_optimizer_state_exact": reference["optimizer_state_sha256"]
        == interrupted["optimizer_state_sha256"],
    }
    checks = {
        "initial_model_state_exact": reference["initial_model_state_sha256"]
        == interrupted["initial_model_state_sha256"],
        "first_step_loss_exact": reference["step_results"][0]["loss"]
        == interrupted["step_results"][0]["loss"],
        "second_step_loss_exact": reference["step_results"][1]["loss"]
        == interrupted["step_results"][1]["loss"],
        "second_step_segments_exact": reference["step_results"][1]["segment_losses"]
        == interrupted["step_results"][1]["segment_losses"],
        "restored_model_state_exact": reference["step_results"][0][
            "post_model_state_sha256"
        ]
        == interrupted["restored_pre_step_model_state_sha256"],
        "restored_optimizer_state_exact": reference["step_results"][0][
            "post_optimizer_state_sha256"
        ]
        == interrupted["restored_pre_step_optimizer_state_sha256"],
        "post_step_model_numerically_equivalent": _numerically_equivalent(
            numerical_difference["model"],
            maximum_absolute=MAXIMUM_MODEL_ABSOLUTE_DIFFERENCE,
            maximum_relative_l2=MAXIMUM_MODEL_RELATIVE_L2,
        ),
        "post_step_gradients_numerically_equivalent": _numerically_equivalent(
            numerical_difference["gradients"],
            maximum_absolute=MAXIMUM_GRADIENT_OR_OPTIMIZER_ABSOLUTE_DIFFERENCE,
            maximum_relative_l2=MAXIMUM_GRADIENT_OR_OPTIMIZER_RELATIVE_L2,
        ),
        "post_step_optimizer_numerically_equivalent": _numerically_equivalent(
            numerical_difference["optimizer"],
            maximum_absolute=MAXIMUM_GRADIENT_OR_OPTIMIZER_ABSOLUTE_DIFFERENCE,
            maximum_relative_l2=MAXIMUM_GRADIENT_OR_OPTIMIZER_RELATIVE_L2,
        ),
        "final_rng_state_exact": reference["rng_state_sha256"]
        == interrupted["rng_state_sha256"],
        "resumed_step_exact": interrupted["resumed_completed_steps"] == 1,
        "resumed_offset_exact": interrupted["resumed_next_schedule_offset"]
        == EFFECTIVE_BATCH,
        "next_offset_exact": reference["next_schedule_offset"]
        == interrupted["next_schedule_offset"]
        == 2 * EFFECTIVE_BATCH,
        "next_schedule_slice_exact": reference["next_schedule_slice_sha256"]
        == interrupted["next_schedule_slice_sha256"],
        "reference_gradients_complete": bool(reference_gradients["passed"]),
        "interrupted_gradients_complete": bool(interrupted_gradients["passed"]),
        "reference_second_gradients_complete": bool(
            reference["step_results"][1]["gradient_audit"]["passed"]
        ),
        "interrupted_second_gradients_complete": bool(
            interrupted["step_results"][1]["gradient_audit"]["passed"]
        ),
        "snapshot_save_verified": bool(
            interrupted["snapshot"]["verification"]["passed"]
        ),
        "snapshot_load_verified": bool(interrupted["load_audit"]["passed"]),
        "training_state_restored": interrupted["restored_training_state"][
            "initial_model_state_sha256"
        ]
        == interrupted["initial_model_state_sha256"],
        "reference_peak_within_8_gib": int(reference["peak_cuda_allocated_bytes"])
        <= MAXIMUM_PEAK_BYTES,
        "interrupted_peak_within_8_gib": int(
            interrupted["peak_cuda_allocated_bytes"]
        )
        <= MAXIMUM_PEAK_BYTES,
    }
    passed = all(checks.values())
    snapshot_record = dict(interrupted["snapshot"])
    snapshot_path.unlink(missing_ok=True)
    snapshot_record["deleted_after_validation"] = not snapshot_path.exists()
    interrupted["snapshot"] = snapshot_record
    payload = {
        "surface": RESUME_PREFLIGHT_SURFACE,
        "artifact_kind": "marulho_language_resume_fidelity_preflight",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": (
            "admit_v80_billion_position_training"
            if passed
            else "stop_v80_resume_fidelity_failed"
        ),
        "parent": parent_audit,
        "data": data["audits"],
        "configuration": {
            "model_seed": MODEL_SEED,
            "physical_batch": PHYSICAL_BATCH,
            "effective_batch": EFFECTIVE_BATCH,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "compiled": False,
            "dtype": "torch.bfloat16",
            "schedule_sha256": SCHEDULE_SHA256,
            "resume_numerical_tolerances": {
                "maximum_differing_fraction": MAXIMUM_DIFFERING_FRACTION,
                "gradient_or_optimizer_maximum_absolute": (
                    MAXIMUM_GRADIENT_OR_OPTIMIZER_ABSOLUTE_DIFFERENCE
                ),
                "gradient_or_optimizer_maximum_relative_l2": (
                    MAXIMUM_GRADIENT_OR_OPTIMIZER_RELATIVE_L2
                ),
                "model_maximum_absolute": MAXIMUM_MODEL_ABSOLUTE_DIFFERENCE,
                "model_maximum_relative_l2": MAXIMUM_MODEL_RELATIVE_L2,
            },
        },
        "reference": reference,
        "interrupted": interrupted,
        "numerical_difference": numerical_difference,
        "exactness_observations": exactness_observations,
        "checks": checks,
        "seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    _atomic_json(output_path, payload)
    return payload


def _validate_resume_fidelity_report() -> dict[str, Any]:
    actual_hash = file_sha256(RESUME_FIDELITY_REPORT)
    if actual_hash != RESUME_FIDELITY_REPORT_SHA256:
        raise RuntimeError(f"V80 resume-fidelity report changed: {actual_hash}")
    payload = json.loads(RESUME_FIDELITY_REPORT.read_text(encoding="utf-8"))
    checks = {
        "surface_exact": payload.get("surface") == RESUME_PREFLIGHT_SURFACE,
        "passed": payload.get("passed") is True,
        "decision_exact": payload.get("decision")
        == "admit_v80_billion_position_training",
        "all_checks_pass": all(bool(value) for value in payload.get("checks", {}).values()),
        "external_llm_absent": payload.get("external_llm_used") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V80 resume-fidelity admission failed: {checks}")
    return {
        "path": str(RESUME_FIDELITY_REPORT),
        "sha256": actual_hash,
        "checks": checks,
    }


@torch.no_grad()
def _evaluate_v80(
    model: MarulhoLanguageModel,
    data: Mapping[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    documents: torch.Tensor = data["eval_tokens"]
    expected_documents = len(SOURCE_NAMES) * EVAL_DOCUMENTS_PER_SOURCE
    if tuple(documents.shape) != (expected_documents, DOCUMENT_TOKENS):
        raise RuntimeError("V80 evaluation tensor changed")
    was_training = model.training
    model.eval()
    collected: list[torch.Tensor] = []
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for offset in range(0, expected_documents, EVAL_BATCH):
        batch = documents[offset : offset + EVAL_BATCH].to(
            device=device,
            dtype=torch.long,
        )
        result = _episode(model, batch)
        collected.append(result["per_document_segment_losses"].cpu().double())
        del batch, result
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    losses = torch.cat(collected, dim=0)
    segment_losses = losses.mean(0)
    source_losses: dict[str, float] = {}
    for index, name in enumerate(SOURCE_NAMES):
        start = index * EVAL_DOCUMENTS_PER_SOURCE
        stop = start + EVAL_DOCUMENTS_PER_SOURCE
        source_losses[name] = float(losses[start:stop, 1:].mean().item())
    positions = expected_documents * SEGMENTS * SEGMENT_LENGTH
    model.train(was_training)
    return {
        "segment_losses": segment_losses.tolist(),
        "first_segment_loss": float(segment_losses[0].item()),
        "later_segment_loss": float(losses[:, 1:].mean().item()),
        "later_loss_by_source": source_losses,
        "documents": expected_documents,
        "positions": positions,
        "seconds": seconds,
        "positions_per_second": positions / seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _snapshot_output_path(prefix: Path, completed_steps: int) -> Path:
    return prefix.parent / f"{prefix.name}-step-{completed_steps:05d}.pt"


def _snapshot_candidates(prefix: Path) -> list[Path]:
    return sorted(prefix.parent.glob(f"{prefix.name}-step-*.pt"))


def _prune_snapshots(prefix: Path, *, keep: int = SNAPSHOTS_TO_KEEP) -> list[str]:
    candidates = _snapshot_candidates(prefix)
    deleted: list[str] = []
    for path in candidates[: max(0, len(candidates) - int(keep))]:
        path.unlink()
        deleted.append(str(path))
    return deleted


def _training_state(
    *,
    initial_evaluation: Mapping[str, Any],
    curve: list[dict[str, Any]],
    training_seconds: float,
    run_peak_cuda_allocated_bytes: int,
    gradient_audit: Mapping[str, Any] | None,
    parent_audit: Mapping[str, Any],
    optimizer_report: Mapping[str, Any],
    initial_bf16_state_sha256: str,
    last_step_result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "initial_evaluation": dict(initial_evaluation),
        "curve": list(curve),
        "training_seconds": float(training_seconds),
        "run_peak_cuda_allocated_bytes": int(run_peak_cuda_allocated_bytes),
        "gradient_audit": None if gradient_audit is None else dict(gradient_audit),
        "parent_audit": dict(parent_audit),
        "optimizer_report": dict(optimizer_report),
        "initial_bf16_state_sha256": str(initial_bf16_state_sha256),
        "last_step_result": (
            None if last_step_result is None else dict(last_step_result)
        ),
    }


def _save_run_snapshot(
    *,
    prefix: Path,
    model: MarulhoLanguageModel,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    completed_steps: int,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    path = _snapshot_output_path(prefix, completed_steps)
    if path.exists():
        raise ValueError(f"V80 snapshot already exists: {path}")
    saved = save_language_training_snapshot(
        path,
        model,
        tokenizer,
        optimizer,
        completed_steps=completed_steps,
        schedule_sha256=SCHEDULE_SHA256,
        next_schedule_offset=completed_steps * EFFECTIVE_BATCH,
        training_state=state,
        metadata={
            "architecture": "marulho_transformer_v80_billion_continuation",
            "parent_checkpoint_sha256": PARENT_SHA256,
            "parent_cumulative_processed_tokens": PARENT_CUMULATIVE_POSITIONS,
            "phase_processed_tokens": TOTAL_POSITIONS,
            "target_cumulative_processed_tokens": TARGET_CUMULATIVE_POSITIONS,
            "resume_fidelity_report_sha256": RESUME_FIDELITY_REPORT_SHA256,
        },
    )
    saved["sha256"] = file_sha256(path)
    saved["deleted_older_snapshots"] = _prune_snapshots(prefix)
    return saved


def _progress_payload(
    *,
    completed_steps: int,
    training_seconds: float,
    run_peak_cuda_allocated_bytes: int,
    curve: list[dict[str, Any]],
    last_step_result: Mapping[str, Any] | None,
    latest_snapshot: Mapping[str, Any] | None,
    decision: str,
) -> dict[str, Any]:
    processed_positions = completed_steps * EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH
    return {
        "surface": "marulho_language_billion_continuation.v80_live_progress",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "decision": decision,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_steps": completed_steps,
        "train_steps": TRAIN_STEPS,
        "completion_fraction": completed_steps / TRAIN_STEPS,
        "processed_positions": processed_positions,
        "target_positions": TOTAL_POSITIONS,
        "training_seconds": training_seconds,
        "positions_per_second": (
            processed_positions / training_seconds if training_seconds > 0.0 else 0.0
        ),
        "run_peak_cuda_allocated_bytes": run_peak_cuda_allocated_bytes,
        "last_step_result": None if last_step_result is None else dict(last_step_result),
        "curve": list(curve),
        "latest_snapshot": (
            None if latest_snapshot is None else dict(latest_snapshot)
        ),
    }


def _quality_checks(
    *,
    initial: Mapping[str, Any],
    candidate: Mapping[str, Any],
    completed_steps: int,
    run_peak_cuda_allocated_bytes: int,
    gradient_audit: Mapping[str, Any] | None,
) -> dict[str, bool]:
    initial_sources = initial["later_loss_by_source"]
    candidate_sources = candidate["later_loss_by_source"]
    return {
        "completed_steps_exact": completed_steps == TRAIN_STEPS,
        "processed_positions_exact": completed_steps
        * EFFECTIVE_BATCH
        * SEGMENTS
        * SEGMENT_LENGTH
        == TOTAL_POSITIONS,
        "initial_overall_reproduced": abs(
            float(initial["later_segment_loss"]) - REFERENCE_INITIAL_LATER_LOSS
        )
        <= 0.0005,
        "initial_sources_reproduced": all(
            abs(float(initial_sources[name]) - REFERENCE_INITIAL_SOURCE_LOSSES[name])
            <= 0.0005
            for name in SOURCE_NAMES
        ),
        "overall_improves_by_0_25": float(initial["later_segment_loss"])
        - float(candidate["later_segment_loss"])
        >= MINIMUM_OVERALL_IMPROVEMENT,
        "fineweb_retained": float(candidate_sources["fineweb_edu"])
        <= REFERENCE_INITIAL_SOURCE_LOSSES["fineweb_edu"]
        + MAXIMUM_OLD_SOURCE_REGRESSION,
        "cosmopedia_retained": float(candidate_sources["cosmopedia_v2"])
        <= REFERENCE_INITIAL_SOURCE_LOSSES["cosmopedia_v2"]
        + MAXIMUM_OLD_SOURCE_REGRESSION,
        "dclm_improves_by_0_20": REFERENCE_INITIAL_SOURCE_LOSSES["dclm_edu"]
        - float(candidate_sources["dclm_edu"])
        >= MINIMUM_DCLM_IMPROVEMENT,
        "gradients_complete": bool(gradient_audit and gradient_audit.get("passed")),
        "peak_within_8_gib": run_peak_cuda_allocated_bytes <= MAXIMUM_PEAK_BYTES,
    }


def run_billion_continuation(
    *,
    report_path: Path,
    checkpoint_path: Path,
    progress_path: Path,
    snapshot_prefix: Path,
    resume_from: Path | None = None,
    stop_after_completed_steps: int = TRAIN_STEPS,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V80 billion continuation requires CUDA")
    if report_path.exists() or checkpoint_path.exists():
        raise ValueError("V80 final output already exists")
    stop_after = int(stop_after_completed_steps)
    if stop_after < 1 or stop_after > TRAIN_STEPS:
        raise ValueError("V80 stop boundary is outside the frozen schedule")
    fidelity_admission = _validate_resume_fidelity_report()
    data = load_v80_data()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    latest_snapshot: dict[str, Any] | None = None

    if resume_from is None:
        if progress_path.exists() or _snapshot_candidates(snapshot_prefix):
            raise ValueError("V80 fresh run found existing progress or snapshots")
        torch.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
        parent, tokenizer, parent_audit = _load_parent()
        model = _clone_to_cuda(parent, device=device)
        initial_bf16_state_sha256 = language_model_state_sha256(model)
        if initial_bf16_state_sha256 != PARENT_BF16_STATE_SHA256:
            raise RuntimeError("V80 BF16 parent state changed")
        del parent
        gc.collect()
        optimizer, optimizer_report = _optimizer(model)
        initial_evaluation = _evaluate_v80(model, data, device=device)
        curve: list[dict[str, Any]] = [
            {"completed_steps": 0, "evaluation": initial_evaluation}
        ]
        training_seconds = 0.0
        run_peak = int(torch.cuda.max_memory_allocated(device))
        gradient_audit: dict[str, Any] | None = None
        last_step_result: dict[str, Any] | None = None
        completed_steps = 0
    else:
        if not resume_from.exists():
            raise ValueError(f"V80 resume snapshot is missing: {resume_from}")
        model, tokenizer, optimizer, continuation, load_audit = (
            load_language_training_snapshot(
                resume_from,
                optimizer_builder=_optimizer,
                device=device,
                expected_schedule_sha256=SCHEDULE_SHA256,
                restore_rng=True,
            )
        )
        if not load_audit["passed"]:
            raise RuntimeError(f"V80 resume load failed: {load_audit}")
        completed_steps = int(continuation["completed_steps"])
        next_offset = int(continuation["next_schedule_offset"])
        if next_offset != completed_steps * EFFECTIVE_BATCH:
            raise RuntimeError("V80 resume schedule offset changed")
        if completed_steps < 0 or completed_steps > TRAIN_STEPS:
            raise RuntimeError("V80 resume step is outside the frozen schedule")
        if completed_steps >= stop_after:
            raise ValueError("V80 resume snapshot is already at the stop boundary")
        state = continuation["training_state"]
        initial_evaluation = dict(state["initial_evaluation"])
        curve = list(state["curve"])
        training_seconds = float(state["training_seconds"])
        run_peak = max(
            int(state["run_peak_cuda_allocated_bytes"]),
            int(torch.cuda.max_memory_allocated(device)),
        )
        gradient_audit = (
            None if state["gradient_audit"] is None else dict(state["gradient_audit"])
        )
        parent_audit = dict(state["parent_audit"])
        optimizer_report = dict(state["optimizer_report"])
        initial_bf16_state_sha256 = str(state["initial_bf16_state_sha256"])
        last_step_result = (
            None
            if state["last_step_result"] is None
            else dict(state["last_step_result"])
        )
        latest_snapshot = {
            "path": str(resume_from),
            "sha256": file_sha256(resume_from),
            "completed_steps": completed_steps,
        }
        resume_state_checks = {
            "parent_exact": parent_audit.get("sha256") == PARENT_SHA256,
            "initial_bf16_state_exact": initial_bf16_state_sha256
            == PARENT_BF16_STATE_SHA256,
            "optimizer_exact": optimizer_report.get("kind")
            == "marulho_muon_with_adamw_fallback",
            "curve_starts_at_zero": bool(curve)
            and int(curve[0].get("completed_steps", -1)) == 0,
            "initial_evaluation_exact": bool(curve)
            and curve[0].get("evaluation") == initial_evaluation,
        }
        if not all(resume_state_checks.values()):
            raise RuntimeError(f"V80 resumed training state changed: {resume_state_checks}")

    model.train()
    block_started = time.perf_counter()
    for step in range(completed_steps, stop_after):
        learning_rate = _set_step_learning_rate(optimizer, step)
        documents = _scheduled_documents(
            data,
            offset=step * EFFECTIVE_BATCH,
            count=EFFECTIVE_BATCH,
        )
        last_step_result = _logical_step(
            model,
            optimizer,
            documents=documents,
            device=device,
            audit_gradients=gradient_audit is None,
        )
        if gradient_audit is None:
            gradient_audit = dict(last_step_result["gradient_audit"])
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V80 incomplete gradients: {gradient_audit}")
        last_step_result["learning_rate"] = learning_rate
        completed_steps = step + 1
        boundary = (
            completed_steps % PROGRESS_EVERY_STEPS == 0
            or completed_steps == stop_after
        )
        if boundary:
            torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - block_started
            run_peak = max(run_peak, int(torch.cuda.max_memory_allocated(device)))
            _atomic_json(
                progress_path,
                _progress_payload(
                    completed_steps=completed_steps,
                    training_seconds=training_seconds,
                    run_peak_cuda_allocated_bytes=run_peak,
                    curve=curve,
                    last_step_result=last_step_result,
                    latest_snapshot=latest_snapshot,
                    decision="training",
                ),
            )
            print(
                f"V80 step={completed_steps}/{TRAIN_STEPS} "
                f"loss={last_step_result['loss']:.6f} "
                f"positions_per_second="
                f"{completed_steps * EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH / training_seconds:.1f}",
                flush=True,
            )
            block_started = time.perf_counter()
        if completed_steps % EVAL_EVERY_STEPS == 0:
            evaluation = _evaluate_v80(model, data, device=device)
            curve.append(
                {"completed_steps": completed_steps, "evaluation": evaluation}
            )
            run_peak = max(run_peak, int(torch.cuda.max_memory_allocated(device)))
            model.train()
            block_started = time.perf_counter()
        if completed_steps % SNAPSHOT_EVERY_STEPS == 0 and completed_steps < TRAIN_STEPS:
            state = _training_state(
                initial_evaluation=initial_evaluation,
                curve=curve,
                training_seconds=training_seconds,
                run_peak_cuda_allocated_bytes=run_peak,
                gradient_audit=gradient_audit,
                parent_audit=parent_audit,
                optimizer_report=optimizer_report,
                initial_bf16_state_sha256=initial_bf16_state_sha256,
                last_step_result=last_step_result,
            )
            latest_snapshot = _save_run_snapshot(
                prefix=snapshot_prefix,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                completed_steps=completed_steps,
                state=state,
            )
            _atomic_json(
                progress_path,
                _progress_payload(
                    completed_steps=completed_steps,
                    training_seconds=training_seconds,
                    run_peak_cuda_allocated_bytes=run_peak,
                    curve=curve,
                    last_step_result=last_step_result,
                    latest_snapshot=latest_snapshot,
                    decision="training_snapshot_verified",
                ),
            )
            block_started = time.perf_counter()

    if completed_steps < TRAIN_STEPS:
        if latest_snapshot is None or int(latest_snapshot["completed_steps"]) != completed_steps:
            state = _training_state(
                initial_evaluation=initial_evaluation,
                curve=curve,
                training_seconds=training_seconds,
                run_peak_cuda_allocated_bytes=run_peak,
                gradient_audit=gradient_audit,
                parent_audit=parent_audit,
                optimizer_report=optimizer_report,
                initial_bf16_state_sha256=initial_bf16_state_sha256,
                last_step_result=last_step_result,
            )
            latest_snapshot = _save_run_snapshot(
                prefix=snapshot_prefix,
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                completed_steps=completed_steps,
                state=state,
            )
        payload = _progress_payload(
            completed_steps=completed_steps,
            training_seconds=training_seconds,
            run_peak_cuda_allocated_bytes=run_peak,
            curve=curve,
            last_step_result=last_step_result,
            latest_snapshot=latest_snapshot,
            decision="paused_at_controlled_boundary",
        )
        _atomic_json(progress_path, payload)
        return payload

    final_evaluation = (
        dict(curve[-1]["evaluation"])
        if curve and int(curve[-1]["completed_steps"]) == TRAIN_STEPS
        else _evaluate_v80(model, data, device=device)
    )
    candidate_bf16_state_sha256 = language_model_state_sha256(model)
    run_peak = max(run_peak, int(torch.cuda.max_memory_allocated(device)))
    checks = _quality_checks(
        initial=initial_evaluation,
        candidate=final_evaluation,
        completed_steps=completed_steps,
        run_peak_cuda_allocated_bytes=run_peak,
        gradient_audit=gradient_audit,
    )
    quality_passed = all(checks.values())
    terminal_state = _training_state(
        initial_evaluation=initial_evaluation,
        curve=curve,
        training_seconds=training_seconds,
        run_peak_cuda_allocated_bytes=run_peak,
        gradient_audit=gradient_audit,
        parent_audit=parent_audit,
        optimizer_report=optimizer_report,
        initial_bf16_state_sha256=initial_bf16_state_sha256,
        last_step_result=last_step_result,
    )
    terminal_snapshot = _save_run_snapshot(
        prefix=snapshot_prefix,
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        completed_steps=completed_steps,
        state=terminal_state,
    )
    checkpoint_record: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "saved": False,
        "fidelity": {"performed": False, "passed": False},
    }
    if quality_passed:
        model = model.to(device="cpu", dtype=torch.float32).eval()
        gc.collect()
        torch.cuda.empty_cache()
        saved_fp32_state_sha256 = language_model_state_sha256(model)
        save_language_model_checkpoint(
            checkpoint_path,
            model,
            tokenizer,
            metadata={
                "architecture": "marulho_transformer_v80_three_source_billion",
                "decision": "save_v80_billion_checkpoint_for_unseen_generation",
                "parent_checkpoint_sha256": PARENT_SHA256,
                "parent_cumulative_processed_tokens": PARENT_CUMULATIVE_POSITIONS,
                "phase_processed_tokens": TOTAL_POSITIONS,
                "cumulative_processed_tokens": TARGET_CUMULATIVE_POSITIONS,
                "heldout_later_segment_loss": float(
                    final_evaluation["later_segment_loss"]
                ),
                "optimizer": dict(optimizer_report),
                "optimizer_state_saved": False,
                "schedule_sha256": SCHEDULE_SHA256,
                "resume_fidelity_report_sha256": RESUME_FIDELITY_REPORT_SHA256,
                "external_llm_used": False,
            },
        )
        checkpoint_hash = file_sha256(checkpoint_path)
        fidelity = checkpoint_fidelity(
            model,
            checkpoint_path,
            tokenizer_hash=tokenizer.vocabulary_hash(),
            sample_input_ids=data["eval_tokens"],
            expected_decision="save_v80_billion_checkpoint_for_unseen_generation",
            expected_cumulative_tokens=TARGET_CUMULATIVE_POSITIONS,
        )
        fidelity["performed"] = True
        checkpoint_record = {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "saved": checkpoint_path.exists(),
            "saved_fp32_state_sha256": saved_fp32_state_sha256,
            "fidelity": fidelity,
        }
        if not fidelity["passed"]:
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_record["saved"] = False
    passed = quality_passed and bool(checkpoint_record["fidelity"]["passed"])
    if passed:
        for snapshot in _snapshot_candidates(snapshot_prefix):
            snapshot.unlink()
    terminal_snapshot["retained_after_decision"] = not passed
    report = {
        "surface": SURFACE,
        "artifact_kind": "marulho_language_billion_continuation",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": (
            "admit_v80_checkpoint_to_unseen_generation"
            if passed
            else "stop_v80_billion_continuation_quality_gate"
        ),
        "parent": parent_audit,
        "resume_fidelity_admission": fidelity_admission,
        "data": data["audits"],
        "configuration": {
            "model_seed": MODEL_SEED,
            "train_steps": TRAIN_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "physical_batch": PHYSICAL_BATCH,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "phase_processed_positions": TOTAL_POSITIONS,
            "target_cumulative_processed_positions": TARGET_CUMULATIVE_POSITIONS,
            "warmup_steps": WARMUP_STEPS,
            "cooldown_steps": COOLDOWN_STEPS,
            "peak_learning_rate": PEAK_LEARNING_RATE,
            "minimum_learning_rate": MINIMUM_LEARNING_RATE,
            "evaluation_every_steps": EVAL_EVERY_STEPS,
            "snapshot_every_steps": SNAPSHOT_EVERY_STEPS,
            "snapshots_to_keep": SNAPSHOTS_TO_KEEP,
            "dtype": "torch.bfloat16",
            "compiled": False,
        },
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_bf16_state_sha256": initial_bf16_state_sha256,
        "candidate_bf16_state_sha256": candidate_bf16_state_sha256,
        "initial_evaluation": initial_evaluation,
        "curve": curve,
        "final_evaluation": final_evaluation,
        "training": {
            "seconds": training_seconds,
            "positions": TOTAL_POSITIONS,
            "positions_per_second": TOTAL_POSITIONS / training_seconds,
            "peak_cuda_allocated_bytes": run_peak,
            "gradient_audit": gradient_audit,
            "optimizer": optimizer_report,
            "final": last_step_result,
        },
        "quality_checks": checks,
        "checkpoint": checkpoint_record,
        "terminal_training_snapshot": terminal_snapshot,
        "promotion_boundary": {
            "unseen_generation_admitted": passed,
            "coherent_generation_claimed": False,
            "continual_learning_admitted": False,
            "runtime_install_allowed": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        _progress_payload(
            completed_steps=completed_steps,
            training_seconds=training_seconds,
            run_peak_cuda_allocated_bytes=run_peak,
            curve=curve,
            last_step_result=last_step_result,
            latest_snapshot=None if passed else terminal_snapshot,
            decision=report["decision"],
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-preflight", type=Path)
    parser.add_argument("--preflight-snapshot", type=Path)
    parser.add_argument("--train-report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--snapshot-prefix", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--stop-after-completed-steps",
        type=int,
        default=TRAIN_STEPS,
    )
    args = parser.parse_args()
    preflight_requested = args.resume_preflight is not None
    training_requested = args.train_report is not None
    if preflight_requested == training_requested:
        parser.error("choose exactly one of --resume-preflight or --train-report")
    if preflight_requested:
        if args.preflight_snapshot is None:
            parser.error("--resume-preflight requires --preflight-snapshot")
        result = run_resume_preflight(
            output_path=args.resume_preflight,
            snapshot_path=args.preflight_snapshot,
        )
    else:
        if any(
            value is None
            for value in (args.checkpoint, args.progress, args.snapshot_prefix)
        ):
            parser.error(
                "--train-report requires --checkpoint, --progress, and --snapshot-prefix"
            )
        result = run_billion_continuation(
            report_path=args.train_report,
            checkpoint_path=args.checkpoint,
            progress_path=args.progress,
            snapshot_prefix=args.snapshot_prefix,
            resume_from=args.resume_from,
            stop_after_completed_steps=args.stop_after_completed_steps,
        )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "passed": result.get("passed"),
                "completed_steps": result.get("completed_steps"),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
