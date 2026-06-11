"""CUDA A/B benchmark for event-driven hypercube binding wake policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import statistics
import time

import torch

from marulho.training.checkpointing import load_trainer_checkpoint


@dataclass(frozen=True)
class BenchmarkArm:
    interval_tokens: int
    samples: int
    median_ms: float
    p95_ms: float
    mean_ms: float
    runtime_bind_count: int
    runtime_idle_skip_count: int
    runtime_active: bool
    binding_usage_sum: float
    binding_state_sum: float
    cuda_allocated_mb: float
    cuda_reserved_mb: float


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction)))
    return ordered[index]


def _run_arm(
    checkpoint: Path,
    patterns: list[torch.Tensor],
    *,
    interval_tokens: int,
    warmup_steps: int,
) -> BenchmarkArm:
    trainer, _ = load_trainer_checkpoint(checkpoint)
    trainer.config.binding_idle_probe_interval_tokens = interval_tokens
    trainer.config.micro_sleep_interval_tokens = 10**9
    trainer.config.deep_sleep_interval_tokens = 10**9
    layer = trainer.model.binding_layer
    if layer is None or not hasattr(layer, "runtime_active"):
        raise RuntimeError("checkpoint does not contain hypercube runtime binding")
    if int(torch.count_nonzero(layer.binding_usage).item()) != 0:
        raise RuntimeError("benchmark requires an inactive binding checkpoint")

    layer.runtime_active = False
    layer.runtime_bind_count = 0
    layer.runtime_idle_skip_count = 0
    layer.last_runtime_execution_mode = "not_run"

    for index in range(warmup_steps):
        trainer.train_step(
            patterns[index],
            raw_window=f"binding wake warmup {index}",
            allow_sleep_maintenance=False,
        )

    timings: list[float] = []
    torch.cuda.synchronize()
    for index, pattern in enumerate(patterns[warmup_steps:], start=warmup_steps):
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        trainer.train_step(
            pattern,
            raw_window=f"binding wake measure {index}",
            allow_sleep_maintenance=False,
        )
        torch.cuda.synchronize()
        timings.append((time.perf_counter_ns() - started) / 1e6)

    return BenchmarkArm(
        interval_tokens=interval_tokens,
        samples=len(timings),
        median_ms=statistics.median(timings),
        p95_ms=_percentile(timings, 0.95),
        mean_ms=statistics.mean(timings),
        runtime_bind_count=int(layer.runtime_bind_count),
        runtime_idle_skip_count=int(layer.runtime_idle_skip_count),
        runtime_active=bool(layer.runtime_active),
        binding_usage_sum=float(layer.binding_usage.sum().item()),
        binding_state_sum=float(layer.binding_state.sum().item()),
        cuda_allocated_mb=torch.cuda.memory_allocated() / 1024**2,
        cuda_reserved_mb=torch.cuda.memory_reserved() / 1024**2,
    )


def run_benchmark(
    checkpoint: Path,
    *,
    samples: int = 120,
    warmup_steps: int = 20,
    idle_probe_interval_tokens: int = 4,
    seed: int = 20260611,
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the binding wake benchmark")
    if samples <= 0 or warmup_steps < 0:
        raise ValueError("samples must be positive and warmup_steps non-negative")
    if idle_probe_interval_tokens <= 1:
        raise ValueError("idle_probe_interval_tokens must be greater than one")

    probe, _ = load_trainer_checkpoint(checkpoint)
    input_dim = probe.config.input_dim
    device = probe.model.device
    if device.type != "cuda":
        raise RuntimeError(f"checkpoint resolved to {device}, expected CUDA")
    del probe

    generator = torch.Generator(device=device).manual_seed(seed)
    patterns = [
        torch.rand(input_dim, generator=generator, device=device)
        for _ in range(samples + warmup_steps)
    ]
    control = _run_arm(
        checkpoint,
        patterns,
        interval_tokens=1,
        warmup_steps=warmup_steps,
    )
    torch.cuda.empty_cache()
    conditional = _run_arm(
        checkpoint,
        patterns,
        interval_tokens=idle_probe_interval_tokens,
        warmup_steps=warmup_steps,
    )

    def improvement(before: float, after: float) -> float:
        return (before - after) / before * 100.0

    return {
        "surface": "binding_wake_benchmark.v1",
        "checkpoint": str(checkpoint),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "seed": seed,
        "control": asdict(control),
        "conditional": asdict(conditional),
        "median_improvement_percent": improvement(
            control.median_ms,
            conditional.median_ms,
        ),
        "p95_improvement_percent": improvement(
            control.p95_ms,
            conditional.p95_ms,
        ),
        "mean_improvement_percent": improvement(
            control.mean_ms,
            conditional.mean_ms,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--idle-probe-interval-tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260611)
    args = parser.parse_args()

    report = run_benchmark(
        args.checkpoint,
        samples=args.samples,
        warmup_steps=args.warmup_steps,
        idle_probe_interval_tokens=args.idle_probe_interval_tokens,
        seed=args.seed,
    )
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
