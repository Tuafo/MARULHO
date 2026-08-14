"""V76 Stage-A1 matched 100M real-language experiment."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from marulho.evaluation.language_exact_ttt_v76_data import (
    ROOT,
    SEGMENT_LENGTH,
    SOURCE_NAMES,
    prepare_v76_language_data,
    select_document_batch,
)
from marulho.training.language_exact_ttt_100m import (
    V76ExactTTTLanguage,
    file_sha256,
    load_v76_language_parent,
)
from marulho.training.language_model import MarulhoLanguageModel
from marulho.training.language_muon import build_language_muon


Arm = Literal["immutable", "static", "first_order", "exact"]
ARMS: tuple[Arm, ...] = ("immutable", "static", "first_order", "exact")
PARENT = ROOT / "reports/language_scaling/v39-answer-objective-qualified-100m-218m-20260810.pt"
PREFLIGHT = ROOT / "reports/language_scaling/exact-ttt-v76-100m-preflight-v2-20260813.json"
PREFLIGHT_SHA256 = "44aff0655a99e973289fca8876f32e22bd3ceb9e13efff8cbefeba04e51879dd"
MODEL_SEED = 76131
TRAIN_STEPS = 256
EFFECTIVE_BATCH = 32
PHYSICAL_BATCH = 8
ACCUMULATION_STEPS = 4
EVAL_BATCH = 8


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_preflight() -> dict[str, Any]:
    actual = file_sha256(PREFLIGHT)
    if actual != PREFLIGHT_SHA256:
        raise RuntimeError(f"V76 Stage A1 preflight hash changed: {actual}")
    payload = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not payload.get("passed") or payload.get("decision") != "admit_v76_stage_a1_training":
        raise RuntimeError("V76 Stage A1 preflight did not admit training")
    if int(payload["selected_physical_batch"]) != PHYSICAL_BATCH:
        raise RuntimeError("V76 physical batch differs from preflight")
    if int(payload["gradient_accumulation_steps"]) != ACCUMULATION_STEPS:
        raise RuntimeError("V76 accumulation differs from preflight")
    return payload


def _learning_rate(step: int) -> float:
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


def _static_episode(
    model: MarulhoLanguageModel,
    documents: torch.Tensor,
) -> dict[str, torch.Tensor]:
    per_document_losses: list[torch.Tensor] = []
    losses: list[torch.Tensor] = []
    for segment in range(3):
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


def _train_ttt(
    model: V76ExactTTTLanguage,
    *,
    meta_gradient: Literal["exact", "first_order"],
    data: dict[str, Any],
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
    digest = hashlib.sha256()
    gradient_audit: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        final_segment_losses = torch.zeros(3, dtype=torch.float64)
        final_update_norms = torch.zeros(2, dtype=torch.float64)
        for micro in range(ACCUMULATION_STEPS):
            offset = step * EFFECTIVE_BATCH + micro * PHYSICAL_BATCH
            indices = schedule[offset : offset + PHYSICAL_BATCH]
            digest.update(indices.numpy().tobytes())
            batch = select_document_batch(documents, indices, device=device)
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                result = model.episode_documents(
                    batch, meta_gradient=meta_gradient, update_mode="own"
                )
                (result["loss"] / ACCUMULATION_STEPS).backward()
            final_segment_losses += result["segment_losses"].cpu().double()
            final_update_norms += result["update_norms"].cpu().double()
            del batch, result
        if gradient_audit is None:
            gradient_audit = _gradient_audit(model)
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V76 incomplete TTT gradients: {gradient_audit}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = {
            "segment_losses": (final_segment_losses / ACCUMULATION_STEPS).tolist(),
            "update_norms": (final_update_norms / ACCUMULATION_STEPS).tolist(),
            "inner_rates": F.softplus(model.inner_log_rates).detach().cpu().tolist(),
            "learning_rate": _learning_rate(step),
        }
        if (step + 1) % 32 == 0:
            print(f"V76 100M {meta_gradient} step={step + 1}/{TRAIN_STEPS}", flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert gradient_audit is not None
    return {
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * EFFECTIVE_BATCH,
        "positions": TRAIN_STEPS * EFFECTIVE_BATCH * 3 * SEGMENT_LENGTH,
        "seconds": seconds,
        "positions_per_second": TRAIN_STEPS * EFFECTIVE_BATCH * 3 * SEGMENT_LENGTH / seconds,
        "physical_batch": PHYSICAL_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": ACCUMULATION_STEPS,
        "schedule_sha256": digest.hexdigest(),
        "gradient_audit": gradient_audit,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer": optimizer_report,
        "final": final,
    }


def _train_static(
    model: MarulhoLanguageModel,
    *,
    data: dict[str, Any],
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
    digest = hashlib.sha256()
    gradient_audit: dict[str, Any] | None = None
    final: dict[str, Any] = {}
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step in range(TRAIN_STEPS):
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        offset = step * EFFECTIVE_BATCH
        indices = schedule[offset : offset + EFFECTIVE_BATCH]
        digest.update(indices.numpy().tobytes())
        batch = select_document_batch(documents, indices, device=device)
        result = _static_episode(model, batch)
        result["loss"].backward()
        if gradient_audit is None:
            gradient_audit = _gradient_audit(model)
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V76 incomplete static gradients: {gradient_audit}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = {
            "segment_losses": result["segment_losses"].tolist(),
            "learning_rate": _learning_rate(step),
        }
        del batch, result
        if (step + 1) % 32 == 0:
            print(f"V76 100M static step={step + 1}/{TRAIN_STEPS}", flush=True)
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    assert gradient_audit is not None
    return {
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * EFFECTIVE_BATCH,
        "positions": TRAIN_STEPS * EFFECTIVE_BATCH * 3 * SEGMENT_LENGTH,
        "seconds": seconds,
        "positions_per_second": TRAIN_STEPS * EFFECTIVE_BATCH * 3 * SEGMENT_LENGTH / seconds,
        "physical_batch": EFFECTIVE_BATCH,
        "effective_batch": EFFECTIVE_BATCH,
        "gradient_accumulation_steps": 1,
        "schedule_sha256": digest.hexdigest(),
        "gradient_audit": gradient_audit,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "optimizer": optimizer_report,
        "final": final,
    }


def _summarize_evaluation(
    losses: torch.Tensor,
    source_ids: torch.Tensor,
    *,
    seconds: float,
    peak: int,
) -> dict[str, Any]:
    segment_losses = losses.mean(0)
    later = losses[:, 1:].mean()
    by_source: dict[str, float] = {}
    for index, name in enumerate(SOURCE_NAMES):
        mask = source_ids == index
        by_source[name] = float(losses[mask, 1:].mean().item())
    positions = int(losses.shape[0]) * 3 * SEGMENT_LENGTH
    return {
        "segment_losses": segment_losses.tolist(),
        "first_segment_loss": float(segment_losses[0].item()),
        "later_segment_loss": float(later.item()),
        "later_loss_by_source": by_source,
        "documents": int(losses.shape[0]),
        "positions": positions,
        "seconds": seconds,
        "positions_per_second": positions / seconds,
        "peak_cuda_allocated_bytes": peak,
    }


def _evaluate_ttt(
    model: V76ExactTTTLanguage,
    *,
    update_mode: Literal["own", "discard", "shuffled"],
    data: dict[str, Any],
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
        indices = torch.arange(offset, offset + EVAL_BATCH)
        batch = select_document_batch(documents, indices, device=device)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            result = model.episode_documents(
                batch, meta_gradient="first_order", update_mode=update_mode
            )
        collected.append(result["per_document_segment_losses"].cpu().double())
        del batch, result
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    losses = torch.cat(collected, dim=0)
    return _summarize_evaluation(
        losses,
        data["eval_sources"],
        seconds=seconds,
        peak=int(torch.cuda.max_memory_allocated(device)),
    )


@torch.no_grad()
def _evaluate_static(
    model: MarulhoLanguageModel,
    *,
    data: dict[str, Any],
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
    for offset in range(0, int(documents.shape[0]), EFFECTIVE_BATCH):
        indices = torch.arange(offset, offset + EFFECTIVE_BATCH)
        batch = select_document_batch(documents, indices, device=device)
        result = _static_episode(model, batch)
        collected.append(result["per_document_segment_losses"].cpu().double())
        del batch, result
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    losses = torch.cat(collected, dim=0)
    return _summarize_evaluation(
        losses,
        data["eval_sources"],
        seconds=seconds,
        peak=int(torch.cuda.max_memory_allocated(device)),
    )


def run_arm(arm: Arm, output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V76 Stage A1 requires observed CUDA execution")
    preflight = _load_preflight()
    parent, tokenizer, parent_audit = load_v76_language_parent(PARENT)
    data = prepare_v76_language_data(tokenizer)
    if data["contract_sha256"] != preflight["contract_sha256"]:
        raise RuntimeError("V76 data contract changed after preflight")
    device = torch.device("cuda")
    training: dict[str, Any] | None = None
    evaluations: dict[str, Any]
    if arm in {"exact", "first_order"}:
        torch.manual_seed(MODEL_SEED)
        torch.cuda.manual_seed_all(MODEL_SEED)
        model = V76ExactTTTLanguage(parent).to(
            device=device, dtype=torch.bfloat16
        )
        initial_hash = _state_hash(model)
        training = _train_ttt(
            model,
            meta_gradient="exact" if arm == "exact" else "first_order",
            data=data,
            device=device,
        )
        modes = ("own", "discard", "shuffled") if arm == "exact" else ("own",)
        evaluations = {
            mode: _evaluate_ttt(model, update_mode=mode, data=data, device=device)
            for mode in modes
        }
        final_hash = _state_hash(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    else:
        model = parent.to(device=device, dtype=torch.bfloat16)
        initial_hash = _state_hash(model)
        if arm == "static":
            training = _train_static(model, data=data, device=device)
        evaluations = {"static": _evaluate_static(model, data=data, device=device)}
        final_hash = _state_hash(model)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "surface": "marulho_exact_ttt_v76.stage_a1_100m_arm.v1",
        "arm": arm,
        "parent": parent_audit,
        "preflight_sha256": PREFLIGHT_SHA256,
        "contract_sha256": data["contract_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "parameter_count": parameter_count,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": final_hash,
        "training": training,
        "evaluations": evaluations,
        "recipe": {
            "model_seed": MODEL_SEED,
            "train_steps": 0 if arm == "immutable" else TRAIN_STEPS,
            "effective_batch": EFFECTIVE_BATCH,
            "physical_batch": (
                PHYSICAL_BATCH if arm in {"exact", "first_order"} else EFFECTIVE_BATCH
            ),
            "segments": 3,
            "segment_length": SEGMENT_LENGTH,
            "dtype": "torch.bfloat16",
            "compiled": False,
            "external_llm_used": False,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
    }
    _atomic_json(output, payload)
    return payload


def aggregate(inputs: list[Path], output: Path) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    by_arm = {str(row["arm"]): row for row in rows}
    if set(by_arm) != set(ARMS):
        raise ValueError(f"V76 aggregate requires exactly {ARMS}")
    exact = by_arm["exact"]
    first = by_arm["first_order"]
    static = by_arm["static"]
    immutable = by_arm["immutable"]
    exact_own = exact["evaluations"]["own"]
    comparisons = {
        "immutable": immutable["evaluations"]["static"],
        "static": static["evaluations"]["static"],
        "first_order": first["evaluations"]["own"],
        "discard": exact["evaluations"]["discard"],
        "shuffled": exact["evaluations"]["shuffled"],
    }
    margins = {
        name: float(row["later_segment_loss"]) - float(exact_own["later_segment_loss"])
        for name, row in comparisons.items()
    }
    source_checks = {
        name: float(exact_own["later_loss_by_source"][name])
        <= float(comparisons["static"]["later_loss_by_source"][name]) + 0.02
        for name in SOURCE_NAMES
    }
    contracts_match = len({row["contract_sha256"] for row in rows}) == 1
    parents_match = len({row["parent"]["checkpoint_sha256"] for row in rows}) == 1
    initial_ttt_match = exact["initial_state_sha256"] == first["initial_state_sha256"]
    schedule_match = len(
        {
            by_arm[name]["training"]["schedule_sha256"]
            for name in ("exact", "first_order", "static")
        }
    ) == 1
    gradients_complete = all(
        bool(by_arm[name]["training"]["gradient_audit"]["passed"])
        for name in ("exact", "first_order", "static")
    )
    throughput_ratio = float(exact_own["positions_per_second"]) / float(
        comparisons["immutable"]["positions_per_second"]
    )
    checks = {
        "contracts_match": contracts_match,
        "parents_match": parents_match,
        "exact_first_order_initial_state_matches": initial_ttt_match,
        "training_schedules_match": schedule_match,
        "all_gradients_complete": gradients_complete,
        "exact_beats_immutable_by_0_02": margins["immutable"] >= 0.02,
        "exact_beats_static_by_0_02": margins["static"] >= 0.02,
        "exact_beats_first_order_by_0_02": margins["first_order"] >= 0.02,
        "discard_worsens_by_0_02": margins["discard"] >= 0.02,
        "shuffled_worsens_by_0_02": margins["shuffled"] >= 0.02,
        "first_segment_within_static_0_02": float(exact_own["first_segment_loss"])
        <= float(comparisons["static"]["first_segment_loss"]) + 0.02,
        "each_source_within_static_0_02": all(source_checks.values()),
        "test_time_throughput_at_least_50_percent": throughput_ratio >= 0.50,
        "exact_peak_below_10_gib": int(exact["training"]["peak_cuda_allocated_bytes"])
        <= 10 * 1024**3,
    }
    passed = all(checks.values())
    payload = {
        "surface": "marulho_exact_ttt_v76.stage_a1_100m_decision.v1",
        "passed": passed,
        "decision": (
            "advance_v76_exact_ttt_to_checkpoint_and_continual_validation"
            if passed
            else "retire_v76_exact_ttt_100m_language_failure"
        ),
        "checks": checks,
        "exact_later_segment_loss": exact_own["later_segment_loss"],
        "later_loss_margins_over_exact": margins,
        "exact_later_loss_by_source": exact_own["later_loss_by_source"],
        "source_checks": source_checks,
        "exact_test_time_throughput_ratio": throughput_ratio,
        "arm_reports": [
            {"path": str(path), "sha256": file_sha256(path)} for path in inputs
        ],
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--aggregate", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.arm) == bool(args.aggregate):
        raise ValueError("choose exactly one of --arm or --aggregate")
    result = (
        aggregate(args.aggregate, args.output)
        if args.aggregate
        else run_arm(args.arm, args.output)
    )
    print(
        json.dumps(
            {
                "arm": result.get("arm"),
                "decision": result.get("decision"),
                "passed": result.get("passed"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
