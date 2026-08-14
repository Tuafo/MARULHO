"""Run V78's larger unique-document continuation from the strict V77 base."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Mapping

import torch

from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_TOKENS,
    EFFECTIVE_BATCH,
    EVAL_DOCUMENTS_PER_SOURCE,
    EVAL_SOURCES,
    ROOT,
    SEGMENT_LENGTH,
    SEGMENTS,
    SOURCE_NAMES,
    TOKENIZER_SHA256,
    TRAIN_SOURCES,
    _atomic_json,
    _episode,
    _gradient_audit,
    _select_batch,
    _select_documents,
    checkpoint_fidelity,
    evaluate_continuation,
    file_sha256,
)
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    load_language_model_state,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_language_base_scale_continuation.v78"
PARENT = ROOT / "reports/language_scaling/v77-static-long-document-qualified-100m-225m-20260813.pt"
PARENT_SHA256 = "3755bfb683b77bbf74811d58b9d3db404cdca4143b82e1f6f427077ea4487074"
PARENT_STATE_SHA256 = "1862aa585bd67a937ee1ea76f5ed74d6f6ee40a8f1ad335599b8021804848e57"
TRAIN_SKIP_PER_SOURCE = 4096
TRAIN_DOCUMENTS_PER_SOURCE = 16_384
DATA_SEED = 78_121
MODEL_SEED = 78_131
TRAIN_STEPS = 1024
WARMUP_STEPS = 52
PHASE_TOKENS = TRAIN_STEPS * EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH
PARENT_CUMULATIVE_TOKENS = 225_972_480
TARGET_CUMULATIVE_TOKENS = PARENT_CUMULATIVE_TOKENS + PHASE_TOKENS
EXPECTED_CONTRACT_SHA256 = "67e5dc4a58ba27c0ea16b9611ff92c84e1b542863018856152dee097d1742bf8"
EXPECTED_SCHEDULE_SHA256 = "cac405ffdeaa7863eef53e3086800929e47835771a48a21ef776e8d505facc91"
PREFLIGHT_SHA256 = "5a80f2bdef2c8e107e2bc5150c6824d7097b112142ad0e4c578ada7a70fa6e9f"
SELECTED_PHYSICAL_BATCH = 8
MAXIMUM_PEAK_BYTES = 8 * 1024**3
REFERENCE_V77_LATER_LOSS = 2.902099609375
REFERENCE_V77_SOURCE_LOSSES = {
    "fineweb_edu": 3.2497100830078125,
    "cosmopedia_v2": 2.5544891357421875,
}


def load_v78_parent(
    checkpoint: str | Path,
) -> tuple[MarulhoLanguageModel, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V78 parent checkpoint hash changed: {checkpoint_hash}")
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    tokenizer_hash = tokenizer.vocabulary_hash()
    if tokenizer_hash != TOKENIZER_SHA256:
        raise RuntimeError(f"V78 tokenizer hash changed: {tokenizer_hash}")
    if int(model.config.transformer_context_length) != SEGMENT_LENGTH:
        raise RuntimeError("V78 parent context is not 320")
    if model.config.active_language_path != "marulho_transformer":
        raise RuntimeError("V78 parent is not the maintained Transformer path")
    if metadata.get("decision") != "save_v77_static_checkpoint_for_unseen_generation":
        raise RuntimeError("V78 parent lacks V77 qualification metadata")
    if int(metadata.get("cumulative_processed_tokens", -1)) != PARENT_CUMULATIVE_TOKENS:
        raise RuntimeError("V78 parent cumulative token count changed")
    state_sha256 = language_model_state_sha256(model)
    if PARENT_STATE_SHA256 and state_sha256 != PARENT_STATE_SHA256:
        raise RuntimeError(f"V78 parent state changed: {state_sha256}")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 100_679_424:
        raise RuntimeError(f"V78 parent parameter count changed: {parameter_count}")
    return model, tokenizer, {
        "path": str(checkpoint_path),
        "sha256": checkpoint_hash,
        "state_sha256": state_sha256,
        "tokenizer_sha256": tokenizer_hash,
        "parameter_count": parameter_count,
        "configuration": asdict(model.config),
        "metadata": metadata,
    }


def prepare_v78_data(tokenizer: Any, *, enforce_frozen_hashes: bool) -> dict[str, Any]:
    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    selections: dict[str, Any] = {"train": {}, "eval": {}}
    for name, path in zip(SOURCE_NAMES, TRAIN_SOURCES, strict=True):
        tensor, report = _select_documents(
            path,
            count=TRAIN_DOCUMENTS_PER_SOURCE,
            tokenizer=tokenizer,
            skip_eligible=TRAIN_SKIP_PER_SOURCE,
        )
        train_parts.append(tensor)
        selections["train"][name] = report
    for name, path in zip(SOURCE_NAMES, EVAL_SOURCES, strict=True):
        tensor, report = _select_documents(
            path,
            count=EVAL_DOCUMENTS_PER_SOURCE,
            tokenizer=tokenizer,
        )
        eval_parts.append(tensor)
        selections["eval"][name] = report
    train_documents = torch.cat(train_parts, dim=0)
    eval_documents = torch.cat(eval_parts, dim=0)
    eval_sources = torch.cat(
        [
            torch.full((EVAL_DOCUMENTS_PER_SOURCE,), index, dtype=torch.long)
            for index in range(len(SOURCE_NAMES))
        ]
    )
    schedule = torch.randperm(
        int(train_documents.shape[0]),
        generator=torch.Generator().manual_seed(DATA_SEED),
    )
    digest = hashlib.sha256()
    digest.update(train_documents.numpy().tobytes())
    digest.update(eval_documents.numpy().tobytes())
    digest.update(schedule.numpy().tobytes())
    contract_sha256 = digest.hexdigest()
    schedule_sha256 = hashlib.sha256(schedule.numpy().tobytes()).hexdigest()
    if enforce_frozen_hashes:
        if not EXPECTED_CONTRACT_SHA256 or not EXPECTED_SCHEDULE_SHA256:
            raise RuntimeError("V78 data hashes have not been frozen")
        if contract_sha256 != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError(f"V78 data contract changed: {contract_sha256}")
        if schedule_sha256 != EXPECTED_SCHEDULE_SHA256:
            raise RuntimeError(f"V78 schedule changed: {schedule_sha256}")
    return {
        "train_documents": train_documents,
        "eval_documents": eval_documents,
        "eval_sources": eval_sources,
        "schedule": schedule,
        "contract_sha256": contract_sha256,
        "schedule_sha256": schedule_sha256,
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "selections": selections,
    }


def _clone_to_cuda(
    parent: MarulhoLanguageModel,
    *,
    device: torch.device,
) -> MarulhoLanguageModel:
    clone = MarulhoLanguageModel(parent.config)
    load_language_model_state(clone, parent.state_dict())
    return clone.to(device=device, dtype=torch.bfloat16)


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
    physical_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    accumulation = EFFECTIVE_BATCH // physical_batch
    optimizer.zero_grad(set_to_none=True)
    losses = torch.zeros(SEGMENTS, dtype=torch.float64)
    for micro in range(accumulation):
        micro_indices = indices[
            micro * physical_batch : (micro + 1) * physical_batch
        ]
        batch = _select_batch(documents, micro_indices, device=device)
        result = _episode(model, batch)
        (result["loss"] / accumulation).backward()
        losses += result["segment_losses"].cpu().double()
        del batch, result
    gradient = _gradient_audit(model)
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


def _preflight_arm(
    parent: MarulhoLanguageModel,
    *,
    data: Mapping[str, Any],
    physical_batch: int,
    device: torch.device,
) -> dict[str, Any]:
    if EFFECTIVE_BATCH % physical_batch:
        raise ValueError("physical batch must divide the effective batch")
    documents: torch.Tensor = data["train_documents"]
    schedule: torch.Tensor = data["schedule"]

    warm_model = _clone_to_cuda(parent, device=device)
    warm_optimizer, _ = _optimizer(warm_model)
    _logical_step(
        warm_model,
        warm_optimizer,
        documents=documents,
        indices=schedule[:EFFECTIVE_BATCH],
        physical_batch=physical_batch,
        device=device,
    )
    del warm_optimizer, warm_model
    gc.collect()
    torch.cuda.empty_cache()

    model = _clone_to_cuda(parent, device=device)
    optimizer, optimizer_report = _optimizer(model)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    step_seconds: list[float] = []
    gradient_audit: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    for step in range(4):
        offset = step * EFFECTIVE_BATCH
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        final = _logical_step(
            model,
            optimizer,
            documents=documents,
            indices=schedule[offset : offset + EFFECTIVE_BATCH],
            physical_batch=physical_batch,
            device=device,
        )
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
        if gradient_audit is None:
            gradient_audit = final["gradient_audit"]
    assert gradient_audit is not None
    median_seconds = statistics.median(step_seconds)
    peak = int(torch.cuda.max_memory_allocated(device))
    finite = _model_finite(model)
    row = {
        "physical_batch": physical_batch,
        "gradient_accumulation_steps": EFFECTIVE_BATCH // physical_batch,
        "measured_steps": 4,
        "measured_documents": 4 * EFFECTIVE_BATCH,
        "step_seconds": step_seconds,
        "median_complete_step_seconds": median_seconds,
        "median_positions_per_second": (
            EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH / median_seconds
        ),
        "peak_cuda_allocated_bytes": peak,
        "gradient_audit": gradient_audit,
        "model_state_finite": finite,
        "optimizer": optimizer_report,
        "final_segment_losses": final["segment_losses"],
    }
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_batch_preflight(
    *,
    parent_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"V78 preflight output already exists: {output_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("V78 preflight requires CUDA")
    parent, tokenizer, parent_audit = load_v78_parent(parent_path)
    data = prepare_v78_data(tokenizer, enforce_frozen_hashes=True)
    device = torch.device("cuda")
    rows = {
        str(batch): _preflight_arm(
            parent,
            data=data,
            physical_batch=batch,
            device=device,
        )
        for batch in (8, 16)
    }
    row8 = rows["8"]
    row16 = rows["16"]
    throughput_ratio = float(row16["median_positions_per_second"]) / float(
        row8["median_positions_per_second"]
    )
    batch16_passed = (
        int(row16["peak_cuda_allocated_bytes"]) <= MAXIMUM_PEAK_BYTES
        and bool(row16["gradient_audit"]["passed"])
        and bool(row16["model_state_finite"])
        and throughput_ratio >= 1.05
    )
    selected = 16 if batch16_passed else 8
    selected_row = rows[str(selected)]
    passed = (
        int(selected_row["peak_cuda_allocated_bytes"]) <= MAXIMUM_PEAK_BYTES
        and bool(selected_row["gradient_audit"]["passed"])
        and bool(selected_row["model_state_finite"])
    )
    payload = {
        "surface": "marulho_language_base_scale_continuation.v78_preflight",
        "artifact_kind": "marulho_language_training_batch_preflight",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": "admit_v78_training" if passed else "stop_v78_unsafe_training",
        "parent": parent_audit,
        "data": {
            "contract_sha256": data["contract_sha256"],
            "schedule_sha256": data["schedule_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "selections": data["selections"],
        },
        "arms": rows,
        "batch16_throughput_ratio": throughput_ratio,
        "batch16_passed_selection_gate": batch16_passed,
        "selected_physical_batch": selected,
        "selected_gradient_accumulation_steps": EFFECTIVE_BATCH // selected,
        "batch32_forbidden": True,
        "maximum_peak_cuda_allocated_bytes": MAXIMUM_PEAK_BYTES,
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


def _load_preflight(path: Path) -> dict[str, Any]:
    if not PREFLIGHT_SHA256 or not SELECTED_PHYSICAL_BATCH:
        raise RuntimeError("V78 preflight decision has not been frozen")
    actual = file_sha256(path)
    if actual != PREFLIGHT_SHA256:
        raise RuntimeError(f"V78 preflight hash changed: {actual}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed") or payload.get("decision") != "admit_v78_training":
        raise RuntimeError("V78 preflight did not admit training")
    if int(payload["selected_physical_batch"]) != SELECTED_PHYSICAL_BATCH:
        raise RuntimeError("V78 selected batch changed")
    return payload


def _learning_rate(step: int) -> float:
    if step < 0 or step >= TRAIN_STEPS:
        raise ValueError("V78 step is outside its frozen schedule")
    if step < WARMUP_STEPS:
        return 3.0e-5 + (3.0e-4 - 3.0e-5) * ((step + 1) / WARMUP_STEPS)
    progress = (step + 1 - WARMUP_STEPS) / float(TRAIN_STEPS - WARMUP_STEPS)
    return 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
        1.0 + math.cos(math.pi * progress)
    )


def _train(
    model: MarulhoLanguageModel,
    *,
    data: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    physical_batch = SELECTED_PHYSICAL_BATCH
    accumulation = EFFECTIVE_BATCH // physical_batch
    optimizer, optimizer_report = _optimizer(model)
    schedule: torch.Tensor = data["schedule"]
    documents: torch.Tensor = data["train_documents"]
    observed_schedule = hashlib.sha256()
    gradient_audit: dict[str, Any] | None = None
    curve: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
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
        observed_schedule.update(indices.numpy().tobytes())
        final = _logical_step(
            model,
            optimizer,
            documents=documents,
            indices=indices,
            physical_batch=physical_batch,
            device=device,
        )
        if gradient_audit is None:
            gradient_audit = final["gradient_audit"]
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V78 incomplete gradients: {gradient_audit}")
        if (step + 1) % 32 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"V78 step={step + 1}/{TRAIN_STEPS} elapsed={elapsed:.1f}s",
                flush=True,
            )
        if (step + 1) % 256 == 0:
            evaluation = evaluate_continuation(model, data=data, device=device)
            curve.append({"step": step + 1, "evaluation": evaluation})
            model.train()
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert gradient_audit is not None
    schedule_sha256 = observed_schedule.hexdigest()
    if schedule_sha256 != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"V78 observed schedule changed: {schedule_sha256}")
    return {
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * EFFECTIVE_BATCH,
        "positions": PHASE_TOKENS,
        "seconds_including_curve_evaluations": seconds,
        "positions_per_second_including_curve_evaluations": PHASE_TOKENS / seconds,
        "physical_batch": physical_batch,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": accumulation,
        "schedule_sha256": schedule_sha256,
        "gradient_audit": gradient_audit,
        "model_state_finite": _model_finite(model),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer": optimizer_report,
        "final": {
            "segment_losses": final["segment_losses"],
            "learning_rate": _learning_rate(TRAIN_STEPS - 1),
        },
    }, curve


def qualification_checks(
    *,
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    training: Mapping[str, Any],
    data: Mapping[str, Any],
) -> dict[str, bool]:
    return {
        "initial_v77_loss_exact": abs(
            float(initial["later_segment_loss"]) - REFERENCE_V77_LATER_LOSS
        )
        <= 0.0005,
        "final_improves_by_0_08": float(initial["later_segment_loss"])
        - float(final["later_segment_loss"])
        >= 0.08,
        "fineweb_improves_by_0_03": REFERENCE_V77_SOURCE_LOSSES["fineweb_edu"]
        - float(final["later_loss_by_source"]["fineweb_edu"])
        >= 0.03,
        "cosmopedia_improves_by_0_03": REFERENCE_V77_SOURCE_LOSSES[
            "cosmopedia_v2"
        ]
        - float(final["later_loss_by_source"]["cosmopedia_v2"])
        >= 0.03,
        "contract_exact": data["contract_sha256"] == EXPECTED_CONTRACT_SHA256,
        "schedule_exact": training["schedule_sha256"] == EXPECTED_SCHEDULE_SHA256,
        "tokenizer_exact": data["tokenizer_sha256"] == TOKENIZER_SHA256,
        "all_gradients_complete": bool(training["gradient_audit"]["passed"]),
        "model_state_finite": bool(training["model_state_finite"]),
        "all_positions_processed": int(training["positions"]) == PHASE_TOKENS,
        "selected_batch_exact": int(training["physical_batch"])
        == SELECTED_PHYSICAL_BATCH,
        "peak_within_preflight_gate": int(training["peak_cuda_allocated_bytes"])
        <= MAXIMUM_PEAK_BYTES,
    }


def run_training(
    *,
    parent_path: Path,
    preflight_path: Path,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V78 training requires CUDA")
    if checkpoint_path.exists() or report_path.exists():
        raise ValueError("V78 output already exists")
    preflight = _load_preflight(preflight_path)
    parent, tokenizer, parent_audit = load_v78_parent(parent_path)
    data = prepare_v78_data(tokenizer, enforce_frozen_hashes=True)
    device = torch.device("cuda")
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = parent.to(device=device, dtype=torch.bfloat16)
    initial = evaluate_continuation(model, data=data, device=device)
    print(f"V78 initial later_loss={initial['later_segment_loss']:.6f}", flush=True)
    training, curve = _train(model, data=data, device=device)
    final = curve[-1]["evaluation"]
    print(f"V78 final later_loss={final['later_segment_loss']:.6f}", flush=True)
    candidate_bf16_state_sha256 = language_model_state_sha256(model)
    checks = qualification_checks(
        initial=initial,
        final=final,
        training=training,
        data=data,
    )
    quality_passed = all(checks.values())
    fidelity: dict[str, Any] = {"performed": False, "passed": False}
    checkpoint_sha256: str | None = None
    saved_fp32_state_sha256: str | None = None
    if quality_passed:
        model = model.to(device="cpu", dtype=torch.float32).eval()
        torch.cuda.empty_cache()
        saved_fp32_state_sha256 = language_model_state_sha256(model)
        metadata = {
            "architecture": "marulho_transformer_v78_unique_document_scale",
            "decision": "save_v78_unique_document_checkpoint_for_unseen_generation",
            "checkpoint_reproduction": True,
            "parent_checkpoint_sha256": PARENT_SHA256,
            "parent_cumulative_processed_tokens": PARENT_CUMULATIVE_TOKENS,
            "phase_processed_tokens": PHASE_TOKENS,
            "cumulative_processed_tokens": TARGET_CUMULATIVE_TOKENS,
            "heldout_later_segment_loss": float(final["later_segment_loss"]),
            "optimizer": dict(training["optimizer"]),
            "optimizer_state_saved": False,
            "external_llm_used": False,
            "data_contract_sha256": data["contract_sha256"],
            "schedule_sha256": data["schedule_sha256"],
            "preflight_sha256": PREFLIGHT_SHA256,
        }
        save_language_model_checkpoint(checkpoint_path, model, tokenizer, metadata)
        checkpoint_sha256 = file_sha256(checkpoint_path)
        fidelity = checkpoint_fidelity(
            model,
            checkpoint_path,
            tokenizer_hash=tokenizer.vocabulary_hash(),
            sample_input_ids=data["eval_documents"],
            expected_decision="save_v78_unique_document_checkpoint_for_unseen_generation",
            expected_cumulative_tokens=TARGET_CUMULATIVE_TOKENS,
        )
        fidelity["performed"] = True
        if not fidelity["passed"]:
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_sha256 = None
    passed = quality_passed and bool(fidelity["passed"])
    decision = (
        "admit_v78_checkpoint_to_unseen_generation"
        if passed
        else "stop_v78_unique_document_scaling_diminishing_return"
    )
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_language_base_scale_continuation",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": decision,
        "parent": parent_audit,
        "preflight": {
            "path": str(preflight_path),
            "sha256": PREFLIGHT_SHA256,
            "selected_physical_batch": preflight["selected_physical_batch"],
        },
        "configuration": {
            "model_seed": MODEL_SEED,
            "data_seed": DATA_SEED,
            "train_steps": TRAIN_STEPS,
            "warmup_steps": WARMUP_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "physical_batch": SELECTED_PHYSICAL_BATCH,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "phase_processed_tokens": PHASE_TOKENS,
            "target_cumulative_processed_tokens": TARGET_CUMULATIVE_TOKENS,
            "dtype": "torch.bfloat16",
            "compiled": False,
        },
        "data": {
            "contract_sha256": data["contract_sha256"],
            "schedule_sha256": data["schedule_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "selections": data["selections"],
        },
        "initial_evaluation": initial,
        "curve": curve,
        "training": training,
        "final_evaluation": final,
        "candidate_bf16_state_sha256": candidate_bf16_state_sha256,
        "saved_fp32_state_sha256": saved_fp32_state_sha256,
        "qualification_checks": checks,
        "checkpoint": {
            "path": str(checkpoint_path) if checkpoint_path.exists() else None,
            "sha256": checkpoint_sha256,
            "saved": checkpoint_path.exists(),
            "optimizer_state_saved": False,
            "fidelity": fidelity,
        },
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
    _atomic_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=PARENT)
    parser.add_argument("--validate-contract-only", action="store_true")
    parser.add_argument("--batch-preflight", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.validate_contract_only:
        parent, tokenizer, parent_audit = load_v78_parent(args.parent)
        data = prepare_v78_data(tokenizer, enforce_frozen_hashes=False)
        print(
            json.dumps(
                {
                    "contract_sha256": data["contract_sha256"],
                    "parent_state_sha256": parent_audit["state_sha256"],
                    "schedule_sha256": data["schedule_sha256"],
                    "tokenizer_sha256": data["tokenizer_sha256"],
                    "train_documents": int(data["train_documents"].shape[0]),
                    "validated": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        del parent
        return
    if args.batch_preflight is not None:
        result = run_batch_preflight(
            parent_path=args.parent,
            output_path=args.batch_preflight,
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "passed": result["passed"],
                    "selected_physical_batch": result["selected_physical_batch"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.preflight is None or args.checkpoint is None or args.report is None:
        parser.error("training requires --preflight, --checkpoint, and --report")
    result = run_training(
        parent_path=args.parent,
        preflight_path=args.preflight,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
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
