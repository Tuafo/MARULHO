"""Frozen V74 Stage-A0 end-to-end TTT screen."""

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

from marulho.training.language_end_to_end_ttt import (
    TTTMode,
    V74Config,
    V74EndToEndTTT,
    make_v74_batch,
)


ARMS: tuple[TTTMode, ...] = (
    "persistent_update",
    "no_update_same_compute",
    "shuffled_update",
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
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _mechanical(seed: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    model = V74EndToEndTTT().to(device).train()
    generator = torch.Generator().manual_seed(seed + 500_000)
    batch = make_v74_batch(
        model.config,
        batch_size=8,
        generator=generator,
        device=device,
    )
    fast_a, fast_b = model.initial_fast_weights(8)
    disabled = model.forward_segment(
        batch.tokens[:, 0, :-1], fast_a, fast_b, fast_enabled=False
    )
    enabled = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
    changed = make_v74_batch(
        model.config,
        batch_size=8,
        generator=torch.Generator().manual_seed(seed + 500_001),
        device=device,
    )
    changed.tokens[:, :2] = batch.tokens[:, :2]
    first = model.episode(batch, mode="persistent_update")
    second = model.episode(changed, mode="persistent_update")
    parity_error = float((enabled - disabled).abs().max().item())
    earlier_loss_error = float(
        (first["segment_losses"][:2] - second["segment_losses"][:2]).abs().max().item()
    )
    earlier_update_error = float(
        (first["update_norms"][:2] - second["update_norms"][:2]).abs().max().item()
    )
    return {
        "disabled_max_abs_error": parity_error,
        "future_perturbation_earlier_loss_max_abs_error": earlier_loss_error,
        "future_perturbation_earlier_update_max_abs_error": earlier_update_error,
        "passed": parity_error == 0.0
        and earlier_loss_error == 0.0
        and earlier_update_error == 0.0,
    }


def _evaluate(
    model: V74EndToEndTTT,
    *,
    mode: TTTMode,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + 2_000_000)
    correct = 0
    total = 0
    update_norms = torch.zeros(3, dtype=torch.float64)
    losses = torch.zeros(3, dtype=torch.float64)
    model.eval()
    for _ in range(EVAL_DOCUMENTS // BATCH_SIZE):
        batch = make_v74_batch(
            model.config,
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        )
        result = model.episode(batch, mode=mode)
        predictions = result["query_logits"].detach().argmax(dim=-1)
        correct += int((predictions == batch.query_values).sum().item())
        total += int(batch.query_values.numel())
        update_norms += result["update_norms"].detach().cpu().double()
        losses += result["segment_losses"].detach().cpu().double()
        del result
    batches = EVAL_DOCUMENTS // BATCH_SIZE
    return {
        "query_accuracy": correct / total,
        "query_correct": correct,
        "query_total": total,
        "mean_segment_losses": (losses / batches).tolist(),
        "mean_update_norms": (update_norms / batches).tolist(),
        "inner_rate": float(torch.nn.functional.softplus(model.inner_log_rate).item()),
    }


def _run_arm(seed: int, mode: TTTMode, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = V74EndToEndTTT().to(device).train()
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
        batch = make_v74_batch(
            model.config,
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        )
        schedule.update(batch.tokens.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        result = model.episode(batch, mode=mode)
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
            "inner_rate": float(result["inner_rate"].detach().item()),
        }
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    evaluation = _evaluate(model, mode=mode, seed=seed, device=device)
    missing = sorted(set(dict(model.named_parameters())) - observed)
    return {
        "seed": seed,
        "arm": mode,
        "parameter_count": parameter_count,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": _hash_model(model),
        "schedule_sha256": schedule.hexdigest(),
        "training": {
            "steps": TRAIN_STEPS,
            "documents": TRAIN_STEPS * BATCH_SIZE,
            "seconds": seconds,
            "documents_per_second": TRAIN_STEPS * BATCH_SIZE / seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "missing_nonzero_gradient_names": missing,
            "nonfinite_gradient_names": sorted(nonfinite),
            "complete_finite_gradients": not missing and not nonfinite,
            "final": final,
        },
        "evaluation": evaluation,
    }


def run_seed(seed: int, output: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"V74 seed must be one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("V74 requires observed CUDA execution")
    device = torch.device("cuda")
    mechanical = _mechanical(seed, device)
    arms: list[dict[str, Any]] = []
    for mode in ARMS:
        row = _run_arm(seed, mode, device)
        arms.append(row)
        print(
            f"V74 seed={seed} arm={mode} "
            f"accuracy={row['evaluation']['query_accuracy']:.4f} "
            f"docs/s={row['training']['documents_per_second']:.1f}",
            flush=True,
        )
    persistent = float(arms[0]["evaluation"]["query_accuracy"])
    margins = {
        str(row["arm"]): persistent - float(row["evaluation"]["query_accuracy"])
        for row in arms[1:]
    }
    identity = len({row["initial_parameter_sha256"] for row in arms}) == 1
    schedule = len({row["schedule_sha256"] for row in arms}) == 1
    persistent_gradients = bool(arms[0]["training"]["complete_finite_gradients"])
    behavioral = persistent >= 0.80 and min(margins.values()) >= 0.20
    passed = bool(mechanical["passed"] and identity and schedule and persistent_gradients and behavioral)
    payload = {
        "surface": "marulho_end_to_end_ttt_v74.stage_a0_seed.v1",
        "seed": seed,
        "passed": passed,
        "decision": "seed_pass" if passed else "retire_v74_stage_a0_failure",
        "recipe": {
            "config": vars(V74Config()),
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "evaluation_documents": EVAL_DOCUMENTS,
            "slow_optimizer": "AdamW",
            "slow_learning_rate": 3.0e-4,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "meta_gradient": "first_order_straight_through",
            "compiled": False,
        },
        "mechanical": mechanical,
        "initial_identity_matches": identity,
        "schedule_matches": schedule,
        "persistent_complete_finite_gradients": persistent_gradients,
        "persistent_accuracy": persistent,
        "persistent_margins": margins,
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

