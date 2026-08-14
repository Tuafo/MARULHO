"""Run V79's matched Cosmopedia-versus-DCLM data replacement."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
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
    EVAL_BATCH,
    EVAL_DOCUMENTS_PER_SOURCE,
    EVAL_SOURCES,
    ROOT,
    SEGMENT_LENGTH,
    SEGMENTS,
    TOKENIZER_SHA256,
    _atomic_json,
    _episode,
    _gradient_audit,
    _select_batch,
    _select_documents,
    checkpoint_fidelity,
    file_sha256,
)
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_language_dclm_replacement.v79"
PARENT = (
    ROOT
    / "reports/language_scaling/"
    "v78-unique-document-qualified-100m-257m-20260814.pt"
)
PARENT_SHA256 = "b66753983316b5a0cf61b293d36e4fda9b15929168067a59ed95ef816da4313b"
PARENT_STATE_SHA256 = "4ebf6ae3a500a0a77a256be80bb652a3439e47310eb008c002d14312bb34b75e"
PARENT_CUMULATIVE_TOKENS = 257_429_760
PARENT_LATER_LOSSES = {
    "fineweb_edu": 3.1576766967773438,
    "cosmopedia_v2": 2.438720703125,
}
SHARED_SOURCE = (
    ROOT
    / "reports/language_curriculum/"
    "fineweb-edu-train-75k-shard0-20260710.txt"
)
SHARED_SOURCE_SHA256 = "75f07f85c15c971e1d6eeba623c3f8e20d794e81b9c356ad6fadff2366c99434"
CONTROL_SOURCE = (
    ROOT
    / "reports/language_curriculum/"
    "cosmopedia-v2-train-75k-shard3-20260710.txt"
)
CONTROL_SOURCE_SHA256 = "3a135b5f9c8386ca2edd7c18deefec82cafc6e5922691324428d050158d6da51"
DCLM_ARTIFACT = (
    ROOT
    / "reports/language_curriculum/"
    "v79-dclm-edu-selected-16896-20260814.pt"
)
DCLM_ARTIFACT_SHA256 = "04812812d5f2a319a9e88132d1cd01867b98600fc45ba03f3fe78b86bf9eeea0"
DCLM_HOLDOUT_SHA256 = "906f73b29c8496f098986153fe1c01a97a47db9c7cec8317d04155b401f3c9c6"
DCLM_TRAIN_SHA256 = "fa4dc5151406c23e19c8fe28dd12872b2a5b179e078c43609ec65ab48abd530a"
SOURCE_NAMES = ("fineweb_edu", "cosmopedia_v2", "dclm_edu")
TRAIN_DOCUMENTS_PER_SOURCE = 16_384
TRAIN_DOCUMENTS = 2 * TRAIN_DOCUMENTS_PER_SOURCE
DATA_SEED = 79_121
MODEL_SEED = 79_131
TRAIN_STEPS = 1024
WARMUP_STEPS = 52
PHYSICAL_BATCH = 8
MAXIMUM_PEAK_BYTES = 8 * 1024**3
PHASE_TOKENS = TRAIN_STEPS * EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH
TARGET_CUMULATIVE_TOKENS = PARENT_CUMULATIVE_TOKENS + PHASE_TOKENS
EXPECTED_SHARED_TRAIN_SHA256 = "70cec149e29bae7b756087e3a105c7194b9da790749761a73daa31a79052ad70"
EXPECTED_CONTROL_TRAIN_SHA256 = "b97544f1ad8e3eb7b128969614a7cf1402ced8c2b19ddd57c4a140f08efe0241"
EXPECTED_FINEWEB_EVAL_SHA256 = "3235bc1b7bbfb1b390b1aeeaaf5d584d1c6880a3836d720f98b83f11192defa4"
EXPECTED_COSMOPEDIA_EVAL_SHA256 = "325f9ca1a220cee21202a857f76d43515c23054adb3c1f24d1d985cce0bdc59c"
EXPECTED_PAIR_CONTRACT_SHA256 = "e2f787fc001a432a8d1ff00fde3a0189aad184a5ab0572e39c3f73c8d9499a23"
EXPECTED_SCHEDULE_SHA256 = "fb949816cf10bbefb1cb8ce51a0d503cc8dc2f8cac9c20cfecdffb2c9cf3a360"
CONTROL_REPORT_SHA256 = ""


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def load_v79_parent(
    checkpoint: str | Path,
) -> tuple[MarulhoLanguageModel, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V79 parent checkpoint hash changed: {checkpoint_hash}")
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    tokenizer_hash = tokenizer.vocabulary_hash()
    state_hash = language_model_state_sha256(model)
    checks = {
        "tokenizer_exact": tokenizer_hash == TOKENIZER_SHA256,
        "state_exact": state_hash == PARENT_STATE_SHA256,
        "context_exact": int(model.config.transformer_context_length)
        == SEGMENT_LENGTH,
        "active_path_exact": model.config.active_language_path
        == "marulho_transformer",
        "metadata_decision_exact": metadata.get("decision")
        == "save_v78_unique_document_checkpoint_for_unseen_generation",
        "cumulative_tokens_exact": int(
            metadata.get("cumulative_processed_tokens", -1)
        )
        == PARENT_CUMULATIVE_TOKENS,
        "parameter_count_exact": sum(p.numel() for p in model.parameters())
        == 100_679_424,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V79 parent validation failed: {checks}")
    return model, tokenizer, {
        "path": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "state_sha256": state_hash,
        "tokenizer_sha256": tokenizer_hash,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "configuration": asdict(model.config),
        "metadata": metadata,
        "checks": checks,
    }


def _load_dclm_artifact() -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    artifact_hash = file_sha256(DCLM_ARTIFACT)
    if artifact_hash != DCLM_ARTIFACT_SHA256:
        raise RuntimeError(f"V79 DCLM artifact changed: {artifact_hash}")
    artifact = torch.load(DCLM_ARTIFACT, map_location="cpu", weights_only=False)
    holdout: torch.Tensor = artifact["holdout_tokens"].to(dtype=torch.int32)
    train: torch.Tensor = artifact["train_tokens"].to(dtype=torch.int32)
    checks = {
        "surface_exact": artifact.get("surface")
        == "marulho_dclm_edu_materialization.v79",
        "external_llm_absent": artifact.get("external_llm_used") is False,
        "tokenizer_exact": artifact.get("tokenizer_sha256") == TOKENIZER_SHA256,
        "holdout_shape_exact": tuple(holdout.shape)
        == (EVAL_DOCUMENTS_PER_SOURCE, DOCUMENT_TOKENS),
        "train_shape_exact": tuple(train.shape)
        == (TRAIN_DOCUMENTS_PER_SOURCE, DOCUMENT_TOKENS),
        "holdout_hash_exact": _tensor_sha256(holdout) == DCLM_HOLDOUT_SHA256,
        "train_hash_exact": _tensor_sha256(train) == DCLM_TRAIN_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V79 DCLM artifact validation failed: {checks}")
    return holdout, train, {
        "path": str(DCLM_ARTIFACT),
        "sha256": artifact_hash,
        "holdout_sha256": _tensor_sha256(holdout),
        "train_sha256": _tensor_sha256(train),
        "checks": checks,
    }


def prepare_v79_data(tokenizer: Any, *, enforce_frozen_hashes: bool) -> dict[str, Any]:
    source_hashes = {
        "shared_fineweb": file_sha256(SHARED_SOURCE),
        "control_cosmopedia": file_sha256(CONTROL_SOURCE),
    }
    if source_hashes["shared_fineweb"] != SHARED_SOURCE_SHA256:
        raise RuntimeError("V79 shared FineWeb source changed")
    if source_hashes["control_cosmopedia"] != CONTROL_SOURCE_SHA256:
        raise RuntimeError("V79 control Cosmopedia source changed")

    shared, shared_selection = _select_documents(
        SHARED_SOURCE,
        count=TRAIN_DOCUMENTS_PER_SOURCE,
        tokenizer=tokenizer,
    )
    control_other, control_selection = _select_documents(
        CONTROL_SOURCE,
        count=TRAIN_DOCUMENTS_PER_SOURCE,
        tokenizer=tokenizer,
    )
    dclm_holdout, dclm_train, dclm_audit = _load_dclm_artifact()
    eval_parts: list[torch.Tensor] = []
    eval_selections: dict[str, Any] = {}
    for name, path in zip(SOURCE_NAMES[:2], EVAL_SOURCES, strict=True):
        tensor, report = _select_documents(
            path,
            count=EVAL_DOCUMENTS_PER_SOURCE,
            tokenizer=tokenizer,
        )
        eval_parts.append(tensor)
        eval_selections[name] = report
    eval_parts.append(dclm_holdout)
    eval_selections["dclm_edu"] = {
        "path": str(DCLM_ARTIFACT),
        "requested": EVAL_DOCUMENTS_PER_SOURCE,
        "selected_token_sha256": DCLM_HOLDOUT_SHA256,
        "document_tokens": DOCUMENT_TOKENS,
        "selection": "materialized_first_eligible_in_parquet_row_order",
    }

    control_documents = torch.cat((shared, control_other), dim=0)
    candidate_documents = torch.cat((shared, dclm_train), dim=0)
    eval_documents = torch.cat(eval_parts, dim=0)
    eval_sources = torch.cat(
        [
            torch.full((EVAL_DOCUMENTS_PER_SOURCE,), index, dtype=torch.long)
            for index in range(len(SOURCE_NAMES))
        ]
    )
    schedule = torch.randperm(
        TRAIN_DOCUMENTS,
        generator=torch.Generator().manual_seed(DATA_SEED),
    )
    hashes = {
        "shared_train_sha256": _tensor_sha256(shared),
        "control_train_sha256": _tensor_sha256(control_other),
        "candidate_train_sha256": _tensor_sha256(dclm_train),
        "fineweb_eval_sha256": _tensor_sha256(eval_parts[0]),
        "cosmopedia_eval_sha256": _tensor_sha256(eval_parts[1]),
        "dclm_eval_sha256": _tensor_sha256(dclm_holdout),
        "schedule_sha256": _tensor_sha256(schedule),
    }
    pair_digest = hashlib.sha256()
    for key in (
        "shared_train_sha256",
        "control_train_sha256",
        "candidate_train_sha256",
        "fineweb_eval_sha256",
        "cosmopedia_eval_sha256",
        "dclm_eval_sha256",
        "schedule_sha256",
    ):
        pair_digest.update(key.encode("ascii"))
        pair_digest.update(b"\0")
        pair_digest.update(hashes[key].encode("ascii"))
        pair_digest.update(b"\0")
    hashes["pair_contract_sha256"] = pair_digest.hexdigest()

    if enforce_frozen_hashes:
        expected = {
            "shared_train_sha256": EXPECTED_SHARED_TRAIN_SHA256,
            "control_train_sha256": EXPECTED_CONTROL_TRAIN_SHA256,
            "candidate_train_sha256": DCLM_TRAIN_SHA256,
            "fineweb_eval_sha256": EXPECTED_FINEWEB_EVAL_SHA256,
            "cosmopedia_eval_sha256": EXPECTED_COSMOPEDIA_EVAL_SHA256,
            "dclm_eval_sha256": DCLM_HOLDOUT_SHA256,
            "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
            "pair_contract_sha256": EXPECTED_PAIR_CONTRACT_SHA256,
        }
        if not all(expected.values()):
            raise RuntimeError("V79 data hashes have not been frozen")
        changed = {
            key: {"expected": expected[key], "actual": hashes[key]}
            for key in expected
            if hashes[key] != expected[key]
        }
        if changed:
            raise RuntimeError(f"V79 data contract changed: {changed}")

    return {
        "control_documents": control_documents,
        "candidate_documents": candidate_documents,
        "eval_documents": eval_documents,
        "eval_sources": eval_sources,
        "schedule": schedule,
        "hashes": hashes,
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "source_hashes": source_hashes,
        "selections": {
            "shared_fineweb": shared_selection,
            "control_cosmopedia": control_selection,
            "candidate_dclm": dclm_audit,
            "eval": eval_selections,
        },
    }


def _learning_rate(step: int) -> float:
    if step < 0 or step >= TRAIN_STEPS:
        raise ValueError("V79 step is outside its frozen schedule")
    if step < WARMUP_STEPS:
        return 3.0e-5 + (3.0e-4 - 3.0e-5) * ((step + 1) / WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / float(TRAIN_STEPS - WARMUP_STEPS)
    return 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
        1.0 + math.cos(math.pi * progress)
    )


def _optimizer(model: MarulhoLanguageModel):
    return build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )


def _logical_step(
    model: MarulhoLanguageModel,
    optimizer: Any,
    *,
    documents: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    audit_gradients: bool,
) -> dict[str, Any]:
    accumulation = EFFECTIVE_BATCH // PHYSICAL_BATCH
    optimizer.zero_grad(set_to_none=True)
    losses = torch.zeros(SEGMENTS, dtype=torch.float64)
    for micro in range(accumulation):
        micro_indices = indices[
            micro * PHYSICAL_BATCH : (micro + 1) * PHYSICAL_BATCH
        ]
        batch = _select_batch(documents, micro_indices, device=device)
        result = _episode(model, batch)
        (result["loss"] / accumulation).backward()
        losses += result["segment_losses"].cpu().double()
        del batch, result
    gradient = _gradient_audit(model) if audit_gradients else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {
        "segment_losses": (losses / accumulation).tolist(),
        "gradient_audit": gradient,
    }


def _model_finite(model: torch.nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    )


@torch.no_grad()
def evaluate_v79(
    model: MarulhoLanguageModel,
    *,
    data: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    documents: torch.Tensor = data["eval_documents"]
    collected: list[torch.Tensor] = []
    model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for offset in range(0, int(documents.shape[0]), EVAL_BATCH):
        indices = torch.arange(offset, min(offset + EVAL_BATCH, int(documents.shape[0])))
        batch = _select_batch(documents, indices, device=device)
        result = _episode(model, batch)
        collected.append(result["per_document_segment_losses"].cpu().double())
        del batch, result
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    losses = torch.cat(collected, dim=0)
    source_ids: torch.Tensor = data["eval_sources"]
    segment_losses = losses.mean(0)
    later_by_source: dict[str, float] = {}
    for index, name in enumerate(SOURCE_NAMES):
        later_by_source[name] = float(losses[source_ids == index, 1:].mean().item())
    positions = int(losses.shape[0]) * SEGMENTS * SEGMENT_LENGTH
    return {
        "segment_losses": segment_losses.tolist(),
        "first_segment_loss": float(segment_losses[0].item()),
        "later_segment_loss": float(losses[:, 1:].mean().item()),
        "later_loss_by_source": later_by_source,
        "old_source_mean_later_loss": (
            later_by_source["fineweb_edu"] + later_by_source["cosmopedia_v2"]
        )
        / 2.0,
        "documents": int(losses.shape[0]),
        "positions": positions,
        "seconds": seconds,
        "positions_per_second": positions / seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _train_arm(
    model: MarulhoLanguageModel,
    *,
    arm: str,
    data: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    documents: torch.Tensor = data[f"{arm}_documents"]
    schedule: torch.Tensor = data["schedule"]
    optimizer, optimizer_report = _optimizer(model)
    observed_schedule = hashlib.sha256()
    gradient_audit: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    final_step: dict[str, Any] = {}
    run_peak = 0
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        learning_rate = _learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        offset = step * EFFECTIVE_BATCH
        indices = schedule[offset : offset + EFFECTIVE_BATCH]
        observed_schedule.update(indices.contiguous().view(torch.uint8).numpy().tobytes())
        final_step = _logical_step(
            model,
            optimizer,
            documents=documents,
            indices=indices,
            device=device,
            audit_gradients=step == 0,
        )
        if gradient_audit is None:
            gradient_audit = final_step["gradient_audit"]
            assert gradient_audit is not None
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V79 {arm} incomplete gradients: {gradient_audit}")
        if (step + 1) % 32 == 0:
            print(
                f"V79 {arm} step={step + 1}/{TRAIN_STEPS} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        if (step + 1) % 256 == 0:
            run_peak = max(run_peak, int(torch.cuda.max_memory_allocated(device)))
            curve.append(
                {"step": step + 1, "evaluation": evaluate_v79(model, data=data, device=device)}
            )
            model.train()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert gradient_audit is not None
    schedule_hash = observed_schedule.hexdigest()
    if schedule_hash != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"V79 observed schedule changed: {schedule_hash}")
    return {
        "arm": arm,
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * EFFECTIVE_BATCH,
        "positions": PHASE_TOKENS,
        "seconds_including_curve_evaluations": seconds,
        "positions_per_second_including_curve_evaluations": PHASE_TOKENS / seconds,
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": EFFECTIVE_BATCH // PHYSICAL_BATCH,
        "schedule_sha256": schedule_hash,
        "gradient_audit": gradient_audit,
        "model_state_finite": _model_finite(model),
        "peak_cuda_allocated_bytes": run_peak,
        "curve_evaluations_reset_interval_peak_stats": True,
        "optimizer": optimizer_report,
        "final": {
            "segment_losses": final_step["segment_losses"],
            "learning_rate": _learning_rate(TRAIN_STEPS - 1),
        },
    }, curve


def _arm_validity_checks(
    *,
    arm: str,
    initial: Mapping[str, Any],
    training: Mapping[str, Any],
    data: Mapping[str, Any],
    initial_bf16_state_sha256: str,
) -> dict[str, bool]:
    return {
        "arm_known": arm in {"control", "candidate"},
        "parent_fineweb_loss_reproduced": abs(
            float(initial["later_loss_by_source"]["fineweb_edu"])
            - PARENT_LATER_LOSSES["fineweb_edu"]
        )
        <= 0.0005,
        "parent_cosmopedia_loss_reproduced": abs(
            float(initial["later_loss_by_source"]["cosmopedia_v2"])
            - PARENT_LATER_LOSSES["cosmopedia_v2"]
        )
        <= 0.0005,
        "initial_bf16_state_recorded": len(initial_bf16_state_sha256) == 64,
        "pair_contract_exact": data["hashes"]["pair_contract_sha256"]
        == EXPECTED_PAIR_CONTRACT_SHA256,
        "schedule_exact": training["schedule_sha256"] == EXPECTED_SCHEDULE_SHA256,
        "tokenizer_exact": data["tokenizer_sha256"] == TOKENIZER_SHA256,
        "all_gradients_complete": bool(training["gradient_audit"]["passed"]),
        "model_state_finite": bool(training["model_state_finite"]),
        "all_positions_processed": int(training["positions"]) == PHASE_TOKENS,
        "physical_batch_exact": int(training["physical_batch"]) == PHYSICAL_BATCH,
        "peak_within_8_gib": int(training["peak_cuda_allocated_bytes"])
        <= MAXIMUM_PEAK_BYTES,
    }


def _load_control_report(path: Path) -> tuple[dict[str, Any], str]:
    if not CONTROL_REPORT_SHA256:
        raise RuntimeError("V79 control report hash has not been frozen")
    actual_hash = file_sha256(path)
    if actual_hash != CONTROL_REPORT_SHA256:
        raise RuntimeError(f"V79 control report changed: {actual_hash}")
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "surface_exact": report.get("surface") == SURFACE,
        "arm_exact": report.get("arm") == "control",
        "decision_exact": report.get("decision") == "freeze_v79_control_result",
        "validity_passed": report.get("validity_passed") is True,
        "parent_exact": report.get("parent", {}).get("sha256") == PARENT_SHA256,
        "contract_exact": report.get("data", {}).get("hashes", {}).get(
            "pair_contract_sha256"
        )
        == EXPECTED_PAIR_CONTRACT_SHA256,
        "schedule_exact": report.get("training", {}).get("schedule_sha256")
        == EXPECTED_SCHEDULE_SHA256,
        "positions_exact": report.get("training", {}).get("positions") == PHASE_TOKENS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V79 control report is invalid: {checks}")
    return report, actual_hash


def _candidate_quality_checks(
    *,
    control: Mapping[str, Any],
    candidate_final: Mapping[str, Any],
    candidate_training: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, float]]:
    control_final = control["final_evaluation"]
    control_training = control["training"]
    control_mean = float(control_final["later_segment_loss"])
    candidate_mean = float(candidate_final["later_segment_loss"])
    control_old = float(control_final["old_source_mean_later_loss"])
    candidate_old = float(candidate_final["old_source_mean_later_loss"])
    control_dclm = float(control_final["later_loss_by_source"]["dclm_edu"])
    candidate_dclm = float(candidate_final["later_loss_by_source"]["dclm_edu"])
    throughput_ratio = float(
        candidate_training["positions_per_second_including_curve_evaluations"]
    ) / float(control_training["positions_per_second_including_curve_evaluations"])
    deltas = {
        "three_source_gain_control_minus_candidate": control_mean - candidate_mean,
        "old_source_candidate_minus_control": candidate_old - control_old,
        "dclm_gain_control_minus_candidate": control_dclm - candidate_dclm,
        "candidate_control_throughput_ratio": throughput_ratio,
        "throughput_relative_difference": abs(throughput_ratio - 1.0),
    }
    checks = {
        "three_source_mean_gain_at_least_0_03": deltas[
            "three_source_gain_control_minus_candidate"
        ]
        >= 0.03,
        "old_source_mean_no_worse_by_0_02": deltas[
            "old_source_candidate_minus_control"
        ]
        <= 0.02,
        "dclm_gain_at_least_0_08": deltas["dclm_gain_control_minus_candidate"]
        >= 0.08,
        "throughput_within_5_percent": deltas["throughput_relative_difference"]
        <= 0.05,
    }
    return checks, deltas


def run_arm(
    *,
    arm: str,
    parent_path: Path,
    report_path: Path,
    control_report_path: Path | None = None,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if arm not in {"control", "candidate"}:
        raise ValueError("V79 arm must be control or candidate")
    if not torch.cuda.is_available():
        raise RuntimeError("V79 training requires CUDA")
    if report_path.exists():
        raise ValueError(f"V79 report already exists: {report_path}")
    if arm == "candidate":
        if control_report_path is None or checkpoint_path is None:
            raise ValueError("V79 candidate requires control report and checkpoint paths")
        if checkpoint_path.exists():
            raise ValueError(f"V79 checkpoint already exists: {checkpoint_path}")

    control: dict[str, Any] | None = None
    control_hash: str | None = None
    if arm == "candidate":
        assert control_report_path is not None
        control, control_hash = _load_control_report(control_report_path)

    parent, tokenizer, parent_audit = load_v79_parent(parent_path)
    data = prepare_v79_data(tokenizer, enforce_frozen_hashes=True)
    device = torch.device("cuda")
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = parent.to(device=device, dtype=torch.bfloat16)
    initial_bf16_state_sha256 = language_model_state_sha256(model)
    initial = evaluate_v79(model, data=data, device=device)
    print(
        f"V79 {arm} initial three_source_loss={initial['later_segment_loss']:.6f}",
        flush=True,
    )
    training, curve = _train_arm(model, arm=arm, data=data, device=device)
    final = curve[-1]["evaluation"]
    print(
        f"V79 {arm} final three_source_loss={final['later_segment_loss']:.6f}",
        flush=True,
    )
    candidate_bf16_state_sha256 = language_model_state_sha256(model)
    validity_checks = _arm_validity_checks(
        arm=arm,
        initial=initial,
        training=training,
        data=data,
        initial_bf16_state_sha256=initial_bf16_state_sha256,
    )
    if arm == "candidate":
        assert control is not None
        validity_checks["initial_bf16_state_matches_control"] = (
            initial_bf16_state_sha256 == control["initial_bf16_state_sha256"]
        )
    validity_passed = all(validity_checks.values())

    quality_checks: dict[str, bool] = {}
    deltas: dict[str, float] = {}
    quality_passed = False
    fidelity: dict[str, Any] = {"performed": False, "passed": False}
    checkpoint_sha256: str | None = None
    saved_fp32_state_sha256: str | None = None
    if arm == "candidate":
        assert control is not None and checkpoint_path is not None
        quality_checks, deltas = _candidate_quality_checks(
            control=control,
            candidate_final=final,
            candidate_training=training,
        )
        quality_passed = all(quality_checks.values())
        if validity_passed and quality_passed:
            model = model.to(device="cpu", dtype=torch.float32).eval()
            torch.cuda.empty_cache()
            saved_fp32_state_sha256 = language_model_state_sha256(model)
            metadata = {
                "architecture": "marulho_transformer_v79_dclm_replacement",
                "decision": "save_v79_dclm_checkpoint_for_generation",
                "checkpoint_reproduction": True,
                "parent_checkpoint_sha256": PARENT_SHA256,
                "parent_cumulative_processed_tokens": PARENT_CUMULATIVE_TOKENS,
                "phase_processed_tokens": PHASE_TOKENS,
                "cumulative_processed_tokens": TARGET_CUMULATIVE_TOKENS,
                "heldout_later_segment_loss": float(final["later_segment_loss"]),
                "optimizer": dict(training["optimizer"]),
                "optimizer_state_saved": False,
                "external_llm_used": False,
                "external_text_data_used": True,
                "data_contract_sha256": EXPECTED_PAIR_CONTRACT_SHA256,
                "schedule_sha256": EXPECTED_SCHEDULE_SHA256,
                "control_report_sha256": control_hash,
            }
            save_language_model_checkpoint(checkpoint_path, model, tokenizer, metadata)
            checkpoint_sha256 = file_sha256(checkpoint_path)
            fidelity = checkpoint_fidelity(
                model,
                checkpoint_path,
                tokenizer_hash=tokenizer.vocabulary_hash(),
                sample_input_ids=data["eval_documents"],
                expected_decision="save_v79_dclm_checkpoint_for_generation",
                expected_cumulative_tokens=TARGET_CUMULATIVE_TOKENS,
            )
            fidelity["performed"] = True
            if not fidelity["passed"]:
                checkpoint_path.unlink(missing_ok=True)
                checkpoint_sha256 = None

    passed = (
        validity_passed
        if arm == "control"
        else validity_passed and quality_passed and bool(fidelity["passed"])
    )
    decision = (
        "freeze_v79_control_result"
        if arm == "control" and passed
        else "reject_v79_control_invalid"
        if arm == "control"
        else "admit_v79_candidate_to_generation"
        if passed
        else "retire_v79_dclm_replacement_no_joint_language_gain"
    )
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_matched_language_data_replacement",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data_used": True,
        "arm": arm,
        "passed": passed,
        "validity_passed": validity_passed,
        "quality_passed": quality_passed,
        "decision": decision,
        "parent": parent_audit,
        "configuration": {
            "model_seed": MODEL_SEED,
            "data_seed": DATA_SEED,
            "train_steps": TRAIN_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "physical_batch": PHYSICAL_BATCH,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "phase_processed_tokens": PHASE_TOKENS,
            "target_cumulative_processed_tokens": TARGET_CUMULATIVE_TOKENS,
            "dtype": "torch.bfloat16",
            "compiled": False,
        },
        "data": {
            "hashes": data["hashes"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "source_hashes": data["source_hashes"],
            "selections": data["selections"],
        },
        "control_report": {
            "path": str(control_report_path) if control_report_path else None,
            "sha256": control_hash,
        },
        "initial_bf16_state_sha256": initial_bf16_state_sha256,
        "initial_evaluation": initial,
        "curve": curve,
        "training": training,
        "final_evaluation": final,
        "candidate_bf16_state_sha256": candidate_bf16_state_sha256,
        "saved_fp32_state_sha256": saved_fp32_state_sha256,
        "validity_checks": validity_checks,
        "quality_checks": quality_checks,
        "deltas": deltas,
        "checkpoint": {
            "path": str(checkpoint_path)
            if checkpoint_path is not None and checkpoint_path.exists()
            else None,
            "sha256": checkpoint_sha256,
            "saved": bool(checkpoint_path is not None and checkpoint_path.exists()),
            "optimizer_state_saved": False,
            "fidelity": fidelity,
        },
        "promotion_boundary": {
            "generation_admitted": arm == "candidate" and passed,
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
    _atomic_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--arm", choices=("control", "candidate"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--control-report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.validate_contract_only:
        parent, tokenizer, parent_audit = load_v79_parent(args.parent)
        data = prepare_v79_data(tokenizer, enforce_frozen_hashes=True)
        print(
            json.dumps(
                {
                    "hashes": data["hashes"],
                    "parent_state_sha256": parent_audit["state_sha256"],
                    "source_hashes": data["source_hashes"],
                    "tokenizer_sha256": data["tokenizer_sha256"],
                    "train_documents_per_arm": TRAIN_DOCUMENTS,
                    "validated": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del parent
        return
    if args.arm is None or args.report is None:
        parser.error("training requires --arm and --report")
    result = run_arm(
        arm=args.arm,
        parent_path=args.parent,
        report_path=args.report,
        control_report_path=args.control_report,
        checkpoint_path=args.checkpoint,
    )
    print(
        json.dumps(
            {
                "arm": result["arm"],
                "decision": result["decision"],
                "passed": result["passed"],
                "checkpoint": result["checkpoint"]["path"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
