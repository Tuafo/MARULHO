"""Bounded CUDA safety ladder for V76 exact meta-gradients."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from marulho.training.language_exact_ttt import V76ExactTTT, make_v76_batch


RUNGS = (8, 16, 32, 64, 128)
EFFECTIVE_BATCH = 128
MEMORY_LIMIT_BYTES = 10 * 1024**3
PROJECTED_SEED_LIMIT_SECONDS = 45 * 60


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


def run(output: Path, seed: int = 7601) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V76 preflight requires observed CUDA execution")
    device = torch.device("cuda")
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
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = V76ExactTTT().to(device).train()
        batch = make_v76_batch(
            model.config,
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(seed + 1_000_000),
            device=device,
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            result = model.episode(batch, meta_gradient="exact")
            result["loss"].backward()
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - started
        peak = int(torch.cuda.max_memory_allocated(device))
        missing = sorted(
            name for name, parameter in model.named_parameters() if parameter.grad is None
        )
        nonfinite = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all().item())
        )
        zero = sorted(
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and not bool(torch.count_nonzero(parameter.grad).item())
        )
        safe = peak <= MEMORY_LIMIT_BYTES and not missing and not nonfinite and not zero
        rows.append(
            {
                "batch_size": batch_size,
                "status": "safe" if safe else "failed",
                "seconds": seconds,
                "peak_cuda_allocated_bytes": peak,
                "projected_peak_cuda_allocated_bytes": projected_peak,
                "missing_gradient_names": missing,
                "nonfinite_gradient_names": nonfinite,
                "zero_gradient_names": zero,
            }
        )
        previous_peak = peak
        previous_batch = batch_size
        del result, batch, model
        torch.cuda.empty_cache()
        if not safe:
            break
    safe_rows = [row for row in rows if row["status"] == "safe"]
    selected = int(safe_rows[-1]["batch_size"]) if safe_rows else 0
    accumulation = EFFECTIVE_BATCH // selected if selected else 0
    exact_step_seconds = float(safe_rows[-1]["seconds"]) * accumulation if safe_rows else float("inf")
    projected_seed_seconds = exact_step_seconds * 800 + 300.0
    feasible = bool(
        selected
        and EFFECTIVE_BATCH % selected == 0
        and projected_seed_seconds <= PROJECTED_SEED_LIMIT_SECONDS
    )
    payload = {
        "surface": "marulho_exact_ttt_v76.cuda_preflight.v1",
        "decision": "admit_v76_seed7601" if feasible else "v76_exact_meta_not_consumer_feasible",
        "passed": feasible,
        "seed": seed,
        "math_sdpa": True,
        "compiled": False,
        "effective_batch": EFFECTIVE_BATCH,
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "projected_seed_limit_seconds": PROJECTED_SEED_LIMIT_SECONDS,
        "selected_physical_batch": selected,
        "gradient_accumulation_steps": accumulation,
        "projected_exact_outer_step_seconds": exact_step_seconds,
        "projected_full_seed_seconds": projected_seed_seconds,
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
