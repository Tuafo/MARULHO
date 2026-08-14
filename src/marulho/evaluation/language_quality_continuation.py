"""Qualify MARULHO's strongest ordinary Transformer continuation as V77."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Iterable, Mapping

import torch
import torch.nn.functional as F

from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
    load_language_model_state,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


ROOT = Path(__file__).resolve().parents[3]
SURFACE = "marulho_language_quality_continuation.v77"
PARENT = ROOT / "reports/language_scaling/v39-answer-objective-qualified-100m-218m-20260810.pt"
PARENT_SHA256 = "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d"
TOKENIZER_SHA256 = "faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715"
EXPECTED_CONTRACT_SHA256 = "eb56d6828e9a89ec7a0a7092663694e5c27c4c1d29dc1104b15ad29d10739d27"
EXPECTED_SCHEDULE_SHA256 = "74b714f5f1798309dd4b78743d183a01e2dd9c000e4a19df661d913537a261ca"
TRAIN_SOURCES = (
    ROOT / "reports/language_curriculum/fineweb-edu-replay-75k-shard2-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-replay-75k-shard4-20260710.txt",
)
EVAL_SOURCES = (
    ROOT / "reports/language_curriculum/fineweb-edu-eval-10k-shard1-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-eval-10k-shard2-20260710.txt",
)
SOURCE_NAMES = ("fineweb_edu", "cosmopedia_v2")
DOCUMENT_MARKER = "<|MARULHO_DOCUMENT|>"
TRAIN_DOCUMENTS_PER_SOURCE = 4096
EVAL_DOCUMENTS_PER_SOURCE = 512
DOCUMENT_TOKENS = 961
SEGMENT_LENGTH = 320
SEGMENTS = 3
DATA_SEED = 72121
MODEL_SEED = 76131
TRAIN_STEPS = 256
EFFECTIVE_BATCH = 32
PHYSICAL_BATCH = 8
ACCUMULATION_STEPS = 4
EVAL_BATCH = 32
PARENT_CUMULATIVE_TOKENS = 218_108_160
PHASE_TOKENS = TRAIN_STEPS * EFFECTIVE_BATCH * SEGMENTS * SEGMENT_LENGTH
TARGET_CUMULATIVE_TOKENS = PARENT_CUMULATIVE_TOKENS + PHASE_TOKENS
REFERENCE_IMMUTABLE_LATER_LOSS = 3.9632034301757812
REFERENCE_STATIC_LATER_LOSS = 2.9023361206054688
REFERENCE_STATIC_SOURCE_LOSSES = {
    "fineweb_edu": 3.2500152587890625,
    "cosmopedia_v2": 2.554656982421875,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _iter_documents(path: Path) -> Iterable[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip() == DOCUMENT_MARKER:
                document = "".join(lines).strip()
                if document:
                    yield document
                lines.clear()
            else:
                lines.append(line)
    document = "".join(lines).strip()
    if document:
        yield document


def _select_documents(
    path: Path,
    *,
    count: int,
    tokenizer: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected: list[list[int]] = []
    pending: list[str] = []
    parsed = 0
    eligible = 0

    def flush() -> None:
        nonlocal eligible
        if not pending or len(selected) >= count:
            pending.clear()
            return
        for token_ids in tokenizer.encode_batch(pending, add_bos=True, add_eos=True):
            if len(token_ids) < DOCUMENT_TOKENS:
                continue
            eligible += 1
            selected.append(token_ids[:DOCUMENT_TOKENS])
            if len(selected) >= count:
                break
        pending.clear()

    for document in _iter_documents(path):
        parsed += 1
        pending.append(document)
        if len(pending) >= 128:
            flush()
        if len(selected) >= count:
            break
    flush()
    if len(selected) != count:
        raise RuntimeError(
            f"{path.name} has only {len(selected)} eligible documents; expected {count}"
        )
    tensor = torch.tensor(selected, dtype=torch.int32)
    return tensor, {
        "path": str(path),
        "requested": count,
        "parsed_before_completion": parsed,
        "eligible_before_completion": eligible,
        "selected_token_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "document_tokens": DOCUMENT_TOKENS,
        "selection": "first_eligible_in_file_order",
    }


def prepare_long_document_data(tokenizer: Any) -> dict[str, Any]:
    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    selections: dict[str, Any] = {"train": {}, "eval": {}}
    for name, path in zip(SOURCE_NAMES, TRAIN_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=TRAIN_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
        )
        train_parts.append(tensor)
        selections["train"][name] = report
    for name, path in zip(SOURCE_NAMES, EVAL_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=EVAL_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
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
    if contract_sha256 != EXPECTED_CONTRACT_SHA256:
        raise RuntimeError(f"V77 data contract changed: {contract_sha256}")
    if schedule_sha256 != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"V77 schedule changed: {schedule_sha256}")
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


def load_parent(
    checkpoint: str | Path,
) -> tuple[MarulhoLanguageModel, Any, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    checkpoint_hash = file_sha256(checkpoint_path)
    if checkpoint_hash != PARENT_SHA256:
        raise RuntimeError(f"V77 parent checkpoint hash changed: {checkpoint_hash}")
    source, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    tokenizer_hash = tokenizer.vocabulary_hash()
    if tokenizer_hash != TOKENIZER_SHA256:
        raise RuntimeError(f"V77 tokenizer hash changed: {tokenizer_hash}")
    source_hash = language_model_state_sha256(source)
    config = replace(
        source.config,
        transformer_context_length=SEGMENT_LENGTH,
        active_language_path="marulho_transformer",
    )
    extended = MarulhoLanguageModel(config)
    load_language_model_state(extended, source.state_dict())
    extended_hash = language_model_state_sha256(extended)
    if source_hash != extended_hash:
        raise RuntimeError("V77 context extension changed parent tensor state")
    return extended, tokenizer, {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "tokenizer_sha256": tokenizer_hash,
        "source_model_state_sha256": source_hash,
        "extended_model_state_sha256": extended_hash,
        "source_context_length": int(source.config.transformer_context_length),
        "extended_context_length": SEGMENT_LENGTH,
        "parameter_count": sum(parameter.numel() for parameter in extended.parameters()),
        "metadata": metadata,
    }


def _select_batch(
    documents: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    return documents.index_select(0, indices).to(
        device=device, dtype=torch.long, non_blocking=False
    )


def _episode(
    model: MarulhoLanguageModel,
    documents: torch.Tensor,
) -> dict[str, torch.Tensor]:
    per_document_losses: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for segment in range(SEGMENTS):
        start = segment * SEGMENT_LENGTH
        inputs = documents[:, start : start + SEGMENT_LENGTH]
        targets = documents[:, start + 1 : start + SEGMENT_LENGTH + 1]
        logits = model(inputs, collect_telemetry=False)["logits"]
        per_token = F.cross_entropy(
            logits.flatten(0, 1), targets.flatten(), reduction="none"
        ).reshape(int(documents.shape[0]), SEGMENT_LENGTH)
        per_document = per_token.mean(1)
        per_document_losses.append(per_document.detach())
        losses.append(per_document.mean())
    return {
        "loss": torch.stack(losses).mean(),
        "segment_losses": torch.stack([value.detach() for value in losses]),
        "per_document_segment_losses": torch.stack(per_document_losses, dim=1),
    }


def _learning_rate(step: int) -> float:
    if step < 0 or step >= TRAIN_STEPS:
        raise ValueError("V77 step is outside its frozen training schedule")
    if step < 13:
        return 3.0e-5 + (3.0e-4 - 3.0e-5) * ((step + 1) / 13.0)
    progress = (step + 1 - 13) / float(TRAIN_STEPS - 13)
    return 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
        1.0 + math.cos(math.pi * progress)
    )


def _gradient_audit(model: torch.nn.Module) -> dict[str, Any]:
    missing: list[str] = []
    zero: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
        elif not bool(torch.isfinite(gradient).all().item()):
            nonfinite.append(name)
        elif not bool(torch.count_nonzero(gradient).item()):
            zero.append(name)
    return {
        "missing": missing,
        "zero": zero,
        "nonfinite": nonfinite,
        "passed": not missing and not zero and not nonfinite,
    }


def train_continuation(
    model: MarulhoLanguageModel,
    *,
    data: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )
    schedule: torch.Tensor = data["schedule"]
    documents: torch.Tensor = data["train_documents"]
    schedule_digest = hashlib.sha256()
    gradient_audit: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        learning_rate = _learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        final_segment_losses = torch.zeros(SEGMENTS, dtype=torch.float64)
        for micro in range(ACCUMULATION_STEPS):
            offset = step * EFFECTIVE_BATCH + micro * PHYSICAL_BATCH
            indices = schedule[offset : offset + PHYSICAL_BATCH]
            schedule_digest.update(indices.numpy().tobytes())
            batch = _select_batch(documents, indices, device=device)
            result = _episode(model, batch)
            (result["loss"] / ACCUMULATION_STEPS).backward()
            final_segment_losses += result["segment_losses"].cpu().double()
            del batch, result
        if gradient_audit is None:
            gradient_audit = _gradient_audit(model)
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V77 incomplete gradients: {gradient_audit}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = {
            "segment_losses": (
                final_segment_losses / ACCUMULATION_STEPS
            ).tolist(),
            "learning_rate": learning_rate,
        }
        if (step + 1) % 16 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"V77 static step={step + 1}/{TRAIN_STEPS} elapsed={elapsed:.1f}s",
                flush=True,
            )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert gradient_audit is not None
    observed_schedule_sha256 = schedule_digest.hexdigest()
    if observed_schedule_sha256 != EXPECTED_SCHEDULE_SHA256:
        raise RuntimeError(f"V77 observed schedule changed: {observed_schedule_sha256}")
    return {
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * EFFECTIVE_BATCH,
        "positions": PHASE_TOKENS,
        "seconds": seconds,
        "positions_per_second": PHASE_TOKENS / seconds,
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "schedule_sha256": observed_schedule_sha256,
        "gradient_audit": gradient_audit,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer": optimizer_report,
        "final": final,
    }


@torch.no_grad()
def evaluate_continuation(
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
        mask = source_ids == index
        later_by_source[name] = float(losses[mask, 1:].mean().item())
    positions = int(losses.shape[0]) * SEGMENTS * SEGMENT_LENGTH
    return {
        "segment_losses": segment_losses.tolist(),
        "first_segment_loss": float(segment_losses[0].item()),
        "later_segment_loss": float(losses[:, 1:].mean().item()),
        "later_loss_by_source": later_by_source,
        "documents": int(losses.shape[0]),
        "positions": positions,
        "seconds": seconds,
        "positions_per_second": positions / seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def qualification_checks(
    *,
    initial: Mapping[str, Any],
    candidate: Mapping[str, Any],
    training: Mapping[str, Any],
    parameter_count: int,
    contract_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, bool]:
    return {
        "parameter_count_exact": int(parameter_count) == 100_679_424,
        "data_contract_exact": contract_sha256 == EXPECTED_CONTRACT_SHA256,
        "tokenizer_exact": tokenizer_sha256 == TOKENIZER_SHA256,
        "schedule_exact": training.get("schedule_sha256") == EXPECTED_SCHEDULE_SHA256,
        "initial_loss_reproduced": abs(
            float(initial["later_segment_loss"]) - REFERENCE_IMMUTABLE_LATER_LOSS
        )
        <= 0.0005,
        "candidate_later_loss_reproduced": float(candidate["later_segment_loss"])
        <= REFERENCE_STATIC_LATER_LOSS + 0.02,
        "candidate_fineweb_reproduced": float(
            candidate["later_loss_by_source"]["fineweb_edu"]
        )
        <= REFERENCE_STATIC_SOURCE_LOSSES["fineweb_edu"] + 0.02,
        "candidate_cosmopedia_reproduced": float(
            candidate["later_loss_by_source"]["cosmopedia_v2"]
        )
        <= REFERENCE_STATIC_SOURCE_LOSSES["cosmopedia_v2"] + 0.02,
        "candidate_improves_parent_by_0_50": float(initial["later_segment_loss"])
        - float(candidate["later_segment_loss"])
        >= 0.50,
        "all_gradients_complete": bool(training["gradient_audit"]["passed"]),
        "peak_below_10_gib": int(training["peak_cuda_allocated_bytes"])
        < 10 * 1024**3,
        "processed_positions_exact": int(training["positions"]) == PHASE_TOKENS,
    }


@torch.no_grad()
def checkpoint_fidelity(
    original_model: MarulhoLanguageModel,
    checkpoint_path: Path,
    *,
    tokenizer_hash: str,
    sample_input_ids: torch.Tensor,
) -> dict[str, Any]:
    restored_model, restored_tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    original_model = original_model.cpu().float().eval()
    restored_model = restored_model.cpu().float().eval()
    original_state = original_model.state_dict()
    restored_state = restored_model.state_dict()
    exact_keys = tuple(original_state) == tuple(restored_state)
    exact_tensors = exact_keys and all(
        original_state[name].dtype == restored_state[name].dtype
        and torch.equal(original_state[name], restored_state[name])
        for name in original_state
    )
    sample = sample_input_ids[:1, :32].cpu().long()
    original_logits = original_model(sample, collect_telemetry=False)["logits"]
    restored_logits = restored_model(sample, collect_telemetry=False)["logits"]
    maximum_logit_delta = float((original_logits - restored_logits).abs().max().item())
    report = {
        "strict_state_keys_equal": bool(exact_keys),
        "strict_state_tensors_bit_equal": bool(exact_tensors),
        "state_sha256_equal": language_model_state_sha256(original_model)
        == language_model_state_sha256(restored_model),
        "tokenizer_hash": restored_tokenizer.vocabulary_hash(),
        "tokenizer_hash_equal": restored_tokenizer.vocabulary_hash() == tokenizer_hash,
        "model_config_equal": asdict(restored_model.config)
        == asdict(original_model.config),
        "tied_embedding_head_restored": restored_model.token_embedding.weight.data_ptr()
        == restored_model.lm_head.weight.data_ptr(),
        "maximum_logit_absolute_delta": maximum_logit_delta,
        "logits_bit_equal": maximum_logit_delta == 0.0,
        "metadata_exact": metadata.get("decision")
        == "save_v77_static_checkpoint_for_unseen_generation"
        and int(metadata.get("cumulative_processed_tokens", -1))
        == TARGET_CUMULATIVE_TOKENS
        and metadata.get("external_llm_used") is False
        and metadata.get("optimizer_state_saved") is False,
    }
    report["passed"] = all(
        bool(report[key])
        for key in (
            "strict_state_keys_equal",
            "strict_state_tensors_bit_equal",
            "state_sha256_equal",
            "tokenizer_hash_equal",
            "model_config_equal",
            "tied_embedding_head_restored",
            "logits_bit_equal",
            "metadata_exact",
        )
    )
    return report


def run_qualification(
    *,
    parent_path: Path,
    checkpoint_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V77 requires observed CUDA execution")
    if checkpoint_path.exists():
        raise ValueError(f"V77 checkpoint output already exists: {checkpoint_path}")
    if report_path.exists():
        raise ValueError(f"V77 report output already exists: {report_path}")
    device = torch.device("cuda")
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model, tokenizer, parent_audit = load_parent(parent_path)
    data = prepare_long_document_data(tokenizer)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial_fp32_state_sha256 = language_model_state_sha256(model)
    model = model.to(device=device, dtype=torch.bfloat16)
    initial_bf16_state_sha256 = language_model_state_sha256(model)
    initial = evaluate_continuation(model, data=data, device=device)
    print(
        f"V77 immutable later_loss={initial['later_segment_loss']:.6f}",
        flush=True,
    )
    training = train_continuation(model, data=data, device=device)
    candidate = evaluate_continuation(model, data=data, device=device)
    print(
        f"V77 candidate later_loss={candidate['later_segment_loss']:.6f}",
        flush=True,
    )
    candidate_bf16_state_sha256 = language_model_state_sha256(model)
    checks = qualification_checks(
        initial=initial,
        candidate=candidate,
        training=training,
        parameter_count=parameter_count,
        contract_sha256=str(data["contract_sha256"]),
        tokenizer_sha256=str(data["tokenizer_sha256"]),
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
            "architecture": "marulho_transformer_v77_static_long_document",
            "decision": "save_v77_static_checkpoint_for_unseen_generation",
            "checkpoint_reproduction": True,
            "parent_checkpoint_sha256": PARENT_SHA256,
            "parent_cumulative_processed_tokens": PARENT_CUMULATIVE_TOKENS,
            "phase_processed_tokens": PHASE_TOKENS,
            "cumulative_processed_tokens": TARGET_CUMULATIVE_TOKENS,
            "heldout_later_segment_loss": float(candidate["later_segment_loss"]),
            "optimizer": dict(training["optimizer"]),
            "optimizer_state_saved": False,
            "external_llm_used": False,
            "data_contract_sha256": data["contract_sha256"],
            "schedule_sha256": data["schedule_sha256"],
        }
        save_language_model_checkpoint(
            checkpoint_path,
            model,
            tokenizer,
            metadata=metadata,
        )
        checkpoint_sha256 = file_sha256(checkpoint_path)
        fidelity = checkpoint_fidelity(
            model,
            checkpoint_path,
            tokenizer_hash=tokenizer.vocabulary_hash(),
            sample_input_ids=data["eval_documents"],
        )
        fidelity["performed"] = True
        if not fidelity["passed"]:
            checkpoint_path.unlink(missing_ok=True)
            checkpoint_sha256 = None
    passed = quality_passed and bool(fidelity["passed"])
    decision = (
        "admit_v77_checkpoint_to_unseen_generation"
        if passed
        else "reject_v77_static_reproduction"
    )
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_language_quality_continuation",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "passed": passed,
        "decision": decision,
        "parent": parent_audit,
        "configuration": {
            "model_seed": MODEL_SEED,
            "data_seed": DATA_SEED,
            "train_steps": TRAIN_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "physical_batch": PHYSICAL_BATCH,
            "gradient_accumulation_steps": ACCUMULATION_STEPS,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "document_tokens": DOCUMENT_TOKENS,
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
        "parameter_count": parameter_count,
        "initial_fp32_state_sha256": initial_fp32_state_sha256,
        "initial_bf16_state_sha256": initial_bf16_state_sha256,
        "candidate_bf16_state_sha256": candidate_bf16_state_sha256,
        "saved_fp32_state_sha256": saved_fp32_state_sha256,
        "initial_evaluation": initial,
        "training": training,
        "candidate_evaluation": candidate,
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
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--validate-contract-only", action="store_true")
    args = parser.parse_args()
    if args.validate_contract_only:
        model, tokenizer, parent_audit = load_parent(args.parent)
        data = prepare_long_document_data(tokenizer)
        print(
            json.dumps(
                {
                    "contract_sha256": data["contract_sha256"],
                    "parameter_count": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "parent_sha256": parent_audit["checkpoint_sha256"],
                    "schedule_sha256": data["schedule_sha256"],
                    "tokenizer_sha256": data["tokenizer_sha256"],
                    "validated": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.checkpoint is None or args.report is None:
        parser.error("--checkpoint and --report are required for qualification")
    result = run_qualification(
        parent_path=args.parent,
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
