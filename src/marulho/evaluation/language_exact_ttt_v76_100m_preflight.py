"""Bounded CUDA safety ladder for V76 Stage-A1 exact meta-training."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from marulho.evaluation.language_exact_ttt_v76_data import (
    ROOT,
    prepare_v76_language_data,
    select_document_batch,
)
from marulho.training.language_exact_ttt_100m import (
    V76ExactTTTLanguage,
    load_v76_language_parent,
)


PARENT = ROOT / "reports/language_scaling/v39-answer-objective-qualified-100m-218m-20260810.pt"
RUNGS = (1, 2, 4, 8, 16, 32)
EFFECTIVE_BATCH = 32
MEMORY_LIMIT_BYTES = 10 * 1024**3
PROJECTED_TOTAL_LIMIT_SECONDS = 90 * 60
MODEL_SEED = 76131


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


def run(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V76 Stage A1 preflight requires CUDA")
    device = torch.device("cuda")
    parent, tokenizer, parent_audit = load_v76_language_parent(PARENT)
    data = prepare_v76_language_data(tokenizer)
    torch.manual_seed(MODEL_SEED)
    torch.cuda.manual_seed_all(MODEL_SEED)
    model = V76ExactTTTLanguage(parent).to(device=device, dtype=torch.bfloat16).train()

    parity_indices = data["schedule"][:1]
    parity_documents = select_document_batch(
        data["train_documents"], parity_indices, device=device
    )
    fast_a, fast_b = model.initial_fast_weights(1)
    with torch.no_grad(), sdpa_kernel(backends=[SDPBackend.MATH]):
        enabled = model.forward_segment(parity_documents[:, :320], fast_a, fast_b)
        disabled = model.forward_segment(
            parity_documents[:, :320], fast_a, fast_b, fast_enabled=False
        )
    disabled_error = float((enabled - disabled).abs().max().item())
    del enabled, disabled, parity_documents
    torch.cuda.empty_cache()

    rows: list[dict[str, Any]] = []
    previous_peak: int | None = None
    previous_batch: int | None = None
    for batch_size in RUNGS:
        projected_peak = None
        if previous_peak is not None and previous_batch is not None:
            projected_peak = int(previous_peak * batch_size / previous_batch)
            if projected_peak > MEMORY_LIMIT_BYTES:
                rows.append(
                    {
                        "batch_size": batch_size,
                        "status": "skipped_projected_over_limit",
                        "projected_peak_cuda_allocated_bytes": projected_peak,
                    }
                )
                break
        indices = data["schedule"][:batch_size]
        documents = select_document_batch(
            data["train_documents"], indices, device=device
        )
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        try:
            with sdpa_kernel(backends=[SDPBackend.MATH]):
                result = model.episode_documents(
                    documents, meta_gradient="exact", update_mode="own"
                )
                result["loss"].backward()
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - started
            peak = int(torch.cuda.max_memory_allocated(device))
            audit = _gradient_audit(model)
            safe = peak <= MEMORY_LIMIT_BYTES and bool(audit["passed"])
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": "safe" if safe else "failed",
                    "seconds": seconds,
                    "peak_cuda_allocated_bytes": peak,
                    "projected_peak_cuda_allocated_bytes": projected_peak,
                    "gradient_audit": audit,
                    "loss": float(result["loss"].detach().item()),
                    "segment_losses": result["segment_losses"].tolist(),
                    "update_norms": result["update_norms"].tolist(),
                }
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            rows.append(
                {
                    "batch_size": batch_size,
                    "status": "failed_cuda_oom",
                    "error": str(error),
                    "projected_peak_cuda_allocated_bytes": projected_peak,
                }
            )
            break
        del documents
        if "result" in locals():
            del result
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        if rows[-1]["status"] != "safe":
            break
        previous_peak = int(rows[-1]["peak_cuda_allocated_bytes"])
        previous_batch = batch_size

    safe_rows = [row for row in rows if row["status"] == "safe"]
    selected = int(safe_rows[-1]["batch_size"]) if safe_rows else 0
    accumulation = EFFECTIVE_BATCH // selected if selected else 0
    exact_effective_step = (
        float(safe_rows[-1]["seconds"]) * accumulation if safe_rows else float("inf")
    )
    projected_total_seconds = exact_effective_step * 256 * 2.5 + 600.0
    feasible = bool(
        disabled_error == 0.0
        and selected
        and EFFECTIVE_BATCH % selected == 0
        and projected_total_seconds <= PROJECTED_TOTAL_LIMIT_SECONDS
    )
    payload = {
        "surface": "marulho_exact_ttt_v76.stage_a1_100m_preflight.v1",
        "passed": feasible,
        "decision": (
            "admit_v76_stage_a1_training"
            if feasible
            else "v76_stage_a1_not_consumer_feasible"
        ),
        "parent": parent_audit,
        "contract_sha256": data["contract_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "source_selections": data["selections"],
        "disabled_max_abs_error": disabled_error,
        "math_sdpa": True,
        "compiled": False,
        "dtype": "torch.bfloat16",
        "effective_batch": EFFECTIVE_BATCH,
        "selected_physical_batch": selected,
        "gradient_accumulation_steps": accumulation,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "projected_total_limit_seconds": PROJECTED_TOTAL_LIMIT_SECONDS,
        "projected_exact_effective_step_seconds": exact_effective_step,
        "projected_three_arm_total_seconds": projected_total_seconds,
        "rungs": rows,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
