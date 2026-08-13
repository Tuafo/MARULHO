"""Bounded forward benchmark for the direct V64 Triton recurrence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
from uuid import uuid4

import torch
import torch.nn.functional as F

from marulho.training.language_delta_state import gated_delta_state_chunkwise
from marulho.training.language_delta_state_triton import (
    gated_delta_state_recurrent_triton,
    gated_delta_state_recurrent_triton_autograd,
)


SURFACE = "marulho_delta_state_direct_kernel_benchmark.v2"
DEFAULT_OUTPUT = Path(
    "reports/language_scaling/delta-state-v64-direct-kernel-20260813.json"
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inputs(batch: int, *, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, 10, 320, 64)
    query = F.normalize(
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator),
        dim=-1,
    )
    key = F.normalize(
        torch.randn(shape, device="cuda", dtype=torch.float32, generator=generator),
        dim=-1,
    )
    value = torch.randn(
        shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    erase = torch.sigmoid(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    )
    write = torch.sigmoid(
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)
    )
    log_decay = -0.03 * torch.rand(
        shape, device="cuda", dtype=torch.float32, generator=generator
    )
    state = 0.05 * torch.randn(
        (batch, 10, 64, 64),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return query, key, value, erase, write, log_decay, state


def _timed(
    operation: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    *,
    batch: int,
    steps: int,
) -> dict[str, Any]:
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(steps):
        output, state = operation()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    del output, state
    peak = int(torch.cuda.max_memory_allocated())
    positions = batch * 320 * steps
    return {
        "steps": steps,
        "seconds": elapsed,
        "sequence_positions": positions,
        "sequence_positions_per_second": positions / elapsed,
        "head_state_updates_per_second": positions * 10 / elapsed,
        "baseline_cuda_bytes": baseline,
        "peak_cuda_bytes": peak,
        "incremental_peak_cuda_bytes": peak - baseline,
    }


def _benchmark_batch(batch: int, *, seed: int, steps: int) -> dict[str, Any]:
    query, key, value, erase, write, log_decay, initial_state = _inputs(
        batch, seed=seed
    )

    def eager() -> tuple[torch.Tensor, torch.Tensor]:
        return gated_delta_state_chunkwise(
            query,
            key,
            value,
            erase,
            write,
            log_decay,
            initial_state,
            chunk_size=32,
        )

    def direct() -> tuple[torch.Tensor, torch.Tensor]:
        return gated_delta_state_recurrent_triton(
            query, key, value, erase, write, log_decay, initial_state
        )

    expected_output, expected_state = eager()
    torch.cuda.synchronize()
    compile_started = time.perf_counter()
    actual_output, actual_state = direct()
    torch.cuda.synchronize()
    first_call_seconds = time.perf_counter() - compile_started
    output_delta = torch.abs(actual_output - expected_output)
    state_delta = torch.abs(actual_state - expected_state)
    parity = {
        "output_maximum_absolute_delta": float(output_delta.max().item()),
        "output_mean_absolute_delta": float(output_delta.mean().item()),
        "state_maximum_absolute_delta": float(state_delta.max().item()),
        "state_mean_absolute_delta": float(state_delta.mean().item()),
        "within_bf16_mixed_tolerance": bool(
            output_delta.max().item() <= 1.0e-3
            and state_delta.max().item() <= 1.0e-3
        ),
    }
    del expected_output, expected_state, actual_output, actual_state
    del output_delta, state_delta
    eager_timing = _timed(eager, batch=batch, steps=steps)
    direct_timing = _timed(direct, batch=batch, steps=steps)
    ratio = float(direct_timing["sequence_positions_per_second"]) / float(
        eager_timing["sequence_positions_per_second"]
    )
    torch.cuda.empty_cache()
    return {
        "batch_size": batch,
        "heads": 10,
        "time": 320,
        "key_channels": 64,
        "value_channels": 64,
        "first_direct_call_seconds_including_compile_if_uncached": first_call_seconds,
        "parity": parity,
        "eager_compact_wy": eager_timing,
        "direct_recurrent_triton": direct_timing,
        "direct_to_eager_throughput_ratio": ratio,
    }


def _training_operation(
    values: tuple[torch.Tensor, ...], *, direct: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    if direct:
        return gated_delta_state_recurrent_triton_autograd(
            *values[:6], values[6]
        )
    return gated_delta_state_chunkwise(*values[:6], values[6], chunk_size=32)


def _gradient_metrics(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> dict[str, Any]:
    rows = []
    dot = 0.0
    actual_norm = 0.0
    expected_norm = 0.0
    maximum = 0.0
    for index, (observed, reference) in enumerate(zip(actual, expected)):
        observed_float = observed.float()
        reference_float = reference.float()
        delta = torch.abs(observed_float - reference_float)
        local_maximum = float(delta.max().item())
        maximum = max(maximum, local_maximum)
        dot += float(torch.sum(observed_float * reference_float).item())
        actual_norm += float(torch.sum(observed_float.square()).item())
        expected_norm += float(torch.sum(reference_float.square()).item())
        rows.append(
            {
                "input_index": index,
                "maximum_absolute_delta": local_maximum,
                "mean_absolute_delta": float(delta.mean().item()),
                "finite": bool(torch.isfinite(observed).all().item()),
            }
        )
    cosine = dot / max(1.0e-30, (actual_norm * expected_norm) ** 0.5)
    return {
        "rows": rows,
        "global_cosine": cosine,
        "maximum_absolute_element_delta": maximum,
        "passes_frozen_gate": bool(cosine >= 0.999 and maximum <= 0.01)
        and all(row["finite"] for row in rows),
    }


def _timed_training(
    values: tuple[torch.Tensor, ...], *, direct: bool, batch: int, steps: int
) -> dict[str, Any]:
    for value in values:
        value.grad = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = int(torch.cuda.memory_allocated())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    final_loss = 0.0
    for _ in range(steps):
        for value in values:
            value.grad = None
        output, final_state = _training_operation(values, direct=direct)
        loss = output.float().square().mean()
        loss.backward()
        final_loss = float(loss.detach().item())
        del output, final_state, loss
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated())
    positions = batch * 320 * steps
    return {
        "steps": steps,
        "seconds": elapsed,
        "sequence_positions": positions,
        "sequence_positions_per_second": positions / elapsed,
        "head_state_updates_per_second": positions * 10 / elapsed,
        "baseline_cuda_bytes": baseline,
        "peak_cuda_bytes": peak,
        "incremental_peak_cuda_bytes": peak - baseline,
        "final_loss": final_loss,
    }


def _benchmark_training_batch(
    batch: int, *, seed: int, steps: int
) -> dict[str, Any]:
    base = _inputs(batch, seed=seed)
    eager_values = tuple(value.detach().clone().requires_grad_() for value in base)
    direct_values = tuple(value.detach().clone().requires_grad_() for value in base)
    del base

    eager_output, _ = _training_operation(eager_values, direct=False)
    eager_loss = eager_output.float().square().mean()
    eager_loss.backward()
    expected_gradients = tuple(value.grad.detach().clone() for value in eager_values)
    for value in eager_values:
        value.grad = None
    del eager_output, eager_loss

    torch.cuda.synchronize()
    first_call_started = time.perf_counter()
    direct_output, _ = _training_operation(direct_values, direct=True)
    direct_loss = direct_output.float().square().mean()
    direct_loss.backward()
    torch.cuda.synchronize()
    first_call_seconds = time.perf_counter() - first_call_started
    actual_gradients = tuple(value.grad.detach().clone() for value in direct_values)
    parity = _gradient_metrics(actual_gradients, expected_gradients)
    for value in direct_values:
        value.grad = None
    del direct_output, direct_loss, actual_gradients, expected_gradients
    torch.cuda.empty_cache()

    eager_timing = _timed_training(
        eager_values, direct=False, batch=batch, steps=steps
    )
    direct_timing = _timed_training(
        direct_values, direct=True, batch=batch, steps=steps
    )
    ratio = float(direct_timing["sequence_positions_per_second"]) / float(
        eager_timing["sequence_positions_per_second"]
    )
    return {
        "batch_size": batch,
        "heads": 10,
        "time": 320,
        "key_channels": 64,
        "value_channels": 64,
        "first_direct_forward_backward_seconds_including_compile_if_uncached": (
            first_call_seconds
        ),
        "gradient_parity": parity,
        "eager_compact_wy_autograd": eager_timing,
        "direct_checkpoint_replay_triton_autograd": direct_timing,
        "direct_to_eager_training_throughput_ratio": ratio,
    }


def run_benchmark(
    *, output: Path = DEFAULT_OUTPUT, batches: tuple[int, ...] = (2, 32), steps: int = 3
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V64 direct-kernel benchmark requires CUDA")
    rows = [
        _benchmark_batch(batch, seed=64640 + batch, steps=steps)
        for batch in batches
    ]
    training_rows = [
        _benchmark_training_batch(batch, seed=64740 + batch, steps=steps)
        for batch in batches
    ]
    payload = {
        "artifact_kind": "marulho_delta_state_direct_kernel_benchmark",
        "surface": SURFACE,
        "rows": rows,
        "training_rows": training_rows,
        "hardware": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "forward_only": False,
        "training_backend_admitted": all(
            row["gradient_parity"]["passes_frozen_gate"]
            for row in training_rows
        ),
        "torch_compile_used": False,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }
    _atomic_json(output, payload)
    payload["report_path"] = str(output)
    payload["report_sha256"] = _sha256(output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=(2, 32))
    arguments = parser.parse_args()
    report = run_benchmark(
        output=arguments.output,
        batches=tuple(arguments.batches),
        steps=arguments.steps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
