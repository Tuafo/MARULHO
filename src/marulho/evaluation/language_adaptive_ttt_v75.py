"""Frozen V75 adaptive-retention Stage-A0 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

import torch

from marulho.training.language_adaptive_ttt import (
    RetentionMode,
    V75AdaptiveTTT,
    V75Config,
    make_v75_batch,
)


SEEDS = (7401, 7402, 7403)
TRAIN_STEPS = 800
BATCH_SIZE = 128
EVAL_DOCUMENTS = 4096


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


def _hash_model(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mechanical(seed: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = V75AdaptiveTTT().to(device).train()
    generator = torch.Generator().manual_seed(seed + 500_000)
    batch = make_v75_batch(model.config, batch_size=8, generator=generator, device=device)
    fast_a, fast_b = model.initial_fast_weights(8)
    disabled = model.forward_segment(
        batch.tokens[:, 0, :-1], fast_a, fast_b, fast_enabled=False
    )
    enabled = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
    changed = make_v75_batch(
        model.config,
        batch_size=8,
        generator=torch.Generator().manual_seed(seed + 500_001),
        device=device,
    )
    changed.tokens[:, :2] = batch.tokens[:, :2]
    first = model.episode(batch, mode="adaptive_own")
    second = model.episode(changed, mode="adaptive_own")
    checks = {
        "disabled_max_abs_error": float((enabled - disabled).abs().max().item()),
        "future_perturbation_earlier_loss_max_abs_error": float(
            (first["segment_losses"][:2] - second["segment_losses"][:2]).abs().max().item()
        ),
        "future_perturbation_earlier_update_max_abs_error": float(
            (first["update_norms"][:2] - second["update_norms"][:2]).abs().max().item()
        ),
        "future_perturbation_earlier_gate_max_abs_error": float(
            (first["gates"][:, :2] - second["gates"][:, :2]).abs().max().item()
        ),
        "future_perturbation_earlier_feature_max_abs_error": float(
            (first["gate_features"][:, :2] - second["gate_features"][:, :2])
            .abs()
            .max()
            .item()
        ),
    }
    checks["passed"] = all(value == 0.0 for value in checks.values())
    return checks


def _make_eval_batches(seed: int, device: torch.device) -> list[Any]:
    generator = torch.Generator().manual_seed(seed + 2_000_000)
    return [
        make_v75_batch(
            V75Config(), batch_size=BATCH_SIZE, generator=generator, device=device
        )
        for _ in range(EVAL_DOCUMENTS // BATCH_SIZE)
    ]


def _mean_prequery_gate(model: V75AdaptiveTTT, batches: list[Any]) -> float:
    total = 0.0
    count = 0
    model.eval()
    for batch in batches:
        result = model.episode(batch, mode="adaptive_own")
        values = result["gates"][:, :2].detach()
        total += float(values.sum().item())
        count += int(values.numel())
    return total / count


def _evaluate(
    model: V75AdaptiveTTT,
    *,
    mode: RetentionMode,
    batches: list[Any],
    matched_gate: float,
) -> dict[str, Any]:
    correct = 0
    total = 0
    losses = torch.zeros(3, dtype=torch.float64)
    update_norms = torch.zeros(3, dtype=torch.float64)
    gates: list[torch.Tensor] = []
    accepted: list[torch.Tensor] = []
    features: list[torch.Tensor] = []
    started = time.perf_counter()
    model.eval()
    for batch in batches:
        result = model.episode(batch, mode=mode, matched_gate=matched_gate)
        predictions = result["query_logits"].detach().argmax(dim=-1)
        correct += int((predictions == batch.query_values).sum().item())
        total += int(batch.query_values.numel())
        losses += result["segment_losses"].detach().cpu().double()
        update_norms += result["update_norms"].detach().cpu().double()
        gates.append(result["gates"].detach().cpu())
        accepted.append(result["accepted_gates"].detach().cpu())
        features.append(result["gate_features"].detach().cpu())
    elapsed = time.perf_counter() - started
    all_gates = torch.cat(gates, dim=0)
    all_accepted = torch.cat(accepted, dim=0)
    all_features = torch.cat(features, dim=0)
    quantiles = torch.tensor([0.05, 0.5, 0.95])
    return {
        "query_accuracy": correct / total,
        "query_correct": correct,
        "query_total": total,
        "mean_segment_losses": (losses / len(batches)).tolist(),
        "mean_update_norms": (update_norms / len(batches)).tolist(),
        "mean_raw_gate_by_segment": all_gates.mean(0).tolist(),
        "mean_accepted_gate_by_segment": all_accepted.mean(0).tolist(),
        "prequery_raw_gate_quantiles": torch.quantile(
            all_gates[:, :2].flatten(), quantiles
        ).tolist(),
        "mean_gate_features_by_segment": all_features.mean(0).tolist(),
        "inner_rate": float(torch.nn.functional.softplus(model.inner_log_rate).item()),
        "seconds": elapsed,
        "documents_per_second": EVAL_DOCUMENTS / elapsed,
    }


def run_seed(seed: int, output: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"V75 seed must be one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("V75 requires observed CUDA execution")
    device = torch.device("cuda")
    mechanical = _mechanical(seed, device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = V75AdaptiveTTT().to(device).train()
    initial_hash = _hash_model(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed + 1_000_000)
    schedule = hashlib.sha256()
    observed: set[str] = set()
    nonfinite: set[str] = set()
    final: dict[str, Any] = {}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _step in range(TRAIN_STEPS):
        batch = make_v75_batch(
            model.config, batch_size=BATCH_SIZE, generator=generator, device=device
        )
        schedule.update(batch.tokens.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        result = model.episode(batch, mode="adaptive_own")
        result["loss"].backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all().item()):
                nonfinite.add(name)
            elif bool(torch.count_nonzero(parameter.grad).item()):
                observed.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final = {
            "loss": float(result["loss"].detach().item()),
            "segment_losses": result["segment_losses"].tolist(),
            "update_norms": result["update_norms"].detach().tolist(),
            "mean_gate_by_segment": result["gates"].detach().mean(0).tolist(),
            "inner_rate": float(result["inner_rate"].detach().item()),
        }
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    batches = _make_eval_batches(seed, device)
    matched_gate = _mean_prequery_gate(model, batches)
    arm_order: tuple[RetentionMode, ...] = (
        "adaptive_own",
        "forced_open_own",
        "matched_constant_own",
        "discard_same_compute",
        "adaptive_shuffled",
    )
    arms: list[dict[str, Any]] = []
    for mode in arm_order:
        evaluation = _evaluate(
            model, mode=mode, batches=batches, matched_gate=matched_gate
        )
        arms.append({"arm": mode, "evaluation": evaluation})
        print(
            f"V75 seed={seed} arm={mode} "
            f"accuracy={evaluation['query_accuracy']:.4f} "
            f"docs/s={evaluation['documents_per_second']:.1f}",
            flush=True,
        )
    accuracy = {row["arm"]: row["evaluation"]["query_accuracy"] for row in arms}
    adaptive = float(accuracy["adaptive_own"])
    margins = {name: adaptive - float(value) for name, value in accuracy.items() if name != "adaptive_own"}
    gradients_complete = not (set(dict(model.named_parameters())) - observed) and not nonfinite
    behavioral = (
        adaptive >= 0.80
        and margins["forced_open_own"] >= 0.10
        and margins["matched_constant_own"] >= 0.10
        and margins["discard_same_compute"] >= 0.20
        and margins["adaptive_shuffled"] >= 0.20
    )
    passed = bool(mechanical["passed"] and gradients_complete and behavioral)
    payload = {
        "surface": "marulho_adaptive_ttt_v75.stage_a0_seed.v1",
        "seed": seed,
        "passed": passed,
        "decision": "seed_pass" if passed else "retire_v75_stage_a0_failure",
        "recipe": {
            "config": vars(V75Config()),
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_documents": EVAL_DOCUMENTS,
            "slow_optimizer": "AdamW",
            "slow_learning_rate": 3.0e-4,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "meta_gradient": "first_order_straight_through",
            "gate_features": ["log1p_loss", "log_gradient_rms", "state_alignment", "log_state_rms"],
            "gate_hidden_width": V75Config().gate_width,
            "initial_gate": float(torch.sigmoid(torch.tensor(2.0)).item()),
            "compiled": False,
        },
        "mechanical": mechanical,
        "parameter_count": parameter_count,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": _hash_model(model),
        "schedule_sha256": schedule.hexdigest(),
        "training": {
            "steps": TRAIN_STEPS,
            "documents": TRAIN_STEPS * BATCH_SIZE,
            "seconds": train_seconds,
            "documents_per_second": TRAIN_STEPS * BATCH_SIZE / train_seconds,
            "peak_cuda_allocated_bytes": peak,
            "missing_nonzero_gradient_names": sorted(set(dict(model.named_parameters())) - observed),
            "nonfinite_gradient_names": sorted(nonfinite),
            "complete_finite_gradients": gradients_complete,
            "final": final,
        },
        "matched_constant_gate": matched_gate,
        "adaptive_accuracy": adaptive,
        "adaptive_margins": margins,
        "behavioral_pass": behavioral,
        "arms": arms,
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_seed(args.seed, args.output)
    print(json.dumps({"decision": result["decision"], "passed": result["passed"]}))


if __name__ == "__main__":
    main()
