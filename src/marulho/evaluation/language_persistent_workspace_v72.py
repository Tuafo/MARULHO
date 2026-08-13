"""Frozen V72 Stage-A1 delayed-recall screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
from typing import Any

import torch

from marulho.training.language_persistent_workspace import (
    V72PersistentWorkspaceRecall,
    V72RecallConfig,
    WorkspaceMode,
    make_v72_recall_batch,
    v72_recall_loss,
)


ARMS: tuple[WorkspaceMode, ...] = (
    "persistent",
    "reset_each_segment",
    "shuffled_document_state",
    "nonpersistent_same_compute",
)
SEEDS = (7201, 7202, 7203)
TRAIN_STEPS = 600
BATCH_SIZE = 128
LEARNING_RATE = 3e-4
EVALUATION_DOCUMENTS = 4096


def _tensor_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _recipe_hash(config: V72RecallConfig) -> str:
    payload = {
        "arms": ARMS,
        "batch_size": BATCH_SIZE,
        "config": vars(config),
        "evaluation_documents": EVALUATION_DOCUMENTS,
        "learning_rate": LEARNING_RATE,
        "seeds": SEEDS,
        "train_steps": TRAIN_STEPS,
        "loss": {"answer": 1.0, "key": 0.5, "value": 0.5, "gate": 0.1},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _evaluate(
    model: V72PersistentWorkspaceRecall,
    *,
    mode: WorkspaceMode,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator().manual_seed(int(seed) + 2_000_000)
    correct = 0
    total = 0
    gate_sum = torch.zeros(model.config.segments, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for _ in range(EVALUATION_DOCUMENTS // BATCH_SIZE):
            batch = make_v72_recall_batch(
                model.config,
                batch_size=BATCH_SIZE,
                generator=generator,
                device=device,
            )
            result = model(batch.segments, mode=mode)
            predictions = result["segment_logits"][:, -1].argmax(dim=-1)
            correct += int((predictions == batch.query_values).sum().item())
            total += int(batch.query_values.numel())
            gate_sum += torch.sigmoid(result["gate_logits"]).sum(dim=0).cpu().double()
    return {
        "accuracy": correct / total,
        "correct": float(correct),
        "documents": float(total),
        "gate_segment_0": float(gate_sum[0].item() / total),
        "gate_segment_1": float(gate_sum[1].item() / total),
        "gate_segment_2": float(gate_sum[2].item() / total),
    }


def _run_arm(
    *,
    seed: int,
    mode: WorkspaceMode,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    config = V72RecallConfig()
    model = V72PersistentWorkspaceRecall(config).to(device)
    initial_hash = _tensor_hash(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=0.0
    )
    generator = torch.Generator().manual_seed(int(seed) + 1_000_000)
    schedule_digest = hashlib.sha256()
    gradients_seen: set[str] = set()
    all_gradients_finite = True
    final_components: dict[str, float] = {}
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    model.train()
    for _step in range(TRAIN_STEPS):
        batch = make_v72_recall_batch(
            config,
            batch_size=BATCH_SIZE,
            generator=generator,
            device=device,
        )
        schedule_digest.update(batch.segments.detach().cpu().numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        result = model(batch.segments, mode=mode)
        loss, components = v72_recall_loss(result, batch)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            finite = bool(torch.isfinite(parameter.grad).all().item())
            all_gradients_finite = all_gradients_finite and finite
            if finite and bool(torch.count_nonzero(parameter.grad).item()):
                gradients_seen.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_components = {
            "total": float(loss.detach().item()),
            **{name: float(value.detach().item()) for name, value in components.items()},
        }
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    evaluation = _evaluate(model, mode=mode, seed=seed, device=device)
    peak_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    missing_gradients = sorted(
        set(name for name, _ in model.named_parameters()) - gradients_seen
    )
    return {
        "seed": seed,
        "arm": mode,
        "initial_parameter_hash": initial_hash,
        "final_parameter_hash": _tensor_hash(model),
        "schedule_hash": schedule_digest.hexdigest(),
        "parameter_count": parameter_count,
        "train_steps": TRAIN_STEPS,
        "training_documents": TRAIN_STEPS * BATCH_SIZE,
        "training_seconds": elapsed,
        "training_documents_per_second": (TRAIN_STEPS * BATCH_SIZE) / elapsed,
        "peak_cuda_allocated_bytes": peak_bytes,
        "gradients_all_finite": all_gradients_finite,
        "missing_nonzero_gradients": missing_gradients,
        "final_loss_components": final_components,
        "evaluation": evaluation,
    }


def _mechanical_preflight(device: torch.device) -> dict[str, Any]:
    torch.manual_seed(72)
    model = V72PersistentWorkspaceRecall().to(device).eval()
    generator = torch.Generator().manual_seed(72)
    batch = make_v72_recall_batch(
        model.config,
        batch_size=8,
        generator=generator,
        device=device,
    )
    changed = batch.segments.clone()
    changed[:, -1, 8:48] = changed[:, -1, 8:48].roll(1, dims=0)
    with torch.no_grad():
        original = model(batch.segments, mode="persistent")
        perturbed = model(changed, mode="persistent")
        earlier_logit_error = float(
            (original["segment_logits"][:, :2] - perturbed["segment_logits"][:, :2])
            .abs()
            .max()
            .item()
        )
        earlier_state_error = float(
            (original["states"][:, :2] - perturbed["states"][:, :2])
            .abs()
            .max()
            .item()
        )
        random_state = torch.randn_like(original["states"][:, 0])
        reset_error = float(
            (
                model.boundary_state(random_state, "reset_each_segment")
                - model.initial_state(int(random_state.shape[0]))
            )
            .abs()
            .max()
            .item()
        )
    return {
        "future_perturbation_earlier_logit_max_abs_error": earlier_logit_error,
        "future_perturbation_earlier_state_max_abs_error": earlier_state_error,
        "reset_max_abs_error": reset_error,
        "passed": earlier_logit_error == 0.0
        and earlier_state_error == 0.0
        and reset_error == 0.0,
    }


def run(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V72 frozen screen requires observed CUDA execution")
    device = torch.device("cuda")
    config = V72RecallConfig()
    preflight = _mechanical_preflight(device)
    arms: list[dict[str, Any]] = []
    for seed in SEEDS:
        for mode in ARMS:
            arm = _run_arm(seed=seed, mode=mode, device=device)
            arms.append(arm)
            print(
                f"V72 seed={seed} arm={mode} "
                f"accuracy={arm['evaluation']['accuracy']:.4f} "
                f"docs/s={arm['training_documents_per_second']:.1f}",
                flush=True,
            )

    identity_matches = True
    schedule_matches = True
    seed_results: list[dict[str, Any]] = []
    behavioral_pass = True
    for seed in SEEDS:
        selected = [arm for arm in arms if arm["seed"] == seed]
        identity_matches = identity_matches and len(
            {arm["initial_parameter_hash"] for arm in selected}
        ) == 1
        schedule_matches = schedule_matches and len(
            {arm["schedule_hash"] for arm in selected}
        ) == 1
        accuracies = {
            str(arm["arm"]): float(arm["evaluation"]["accuracy"])
            for arm in selected
        }
        persistent = accuracies["persistent"]
        margins = {
            mode: persistent - accuracies[mode]
            for mode in ARMS
            if mode != "persistent"
        }
        seed_pass = persistent >= 0.80 and min(margins.values()) >= 0.20
        behavioral_pass = behavioral_pass and seed_pass
        seed_results.append(
            {
                "seed": seed,
                "accuracies": accuracies,
                "persistent_margins": margins,
                "passed": seed_pass,
            }
        )

    gradients_pass = all(
        arm["gradients_all_finite"] and not arm["missing_nonzero_gradients"]
        for arm in arms
    )
    memory_pass = max(arm["peak_cuda_allocated_bytes"] for arm in arms) < int(
        11.5 * (1024**3)
    )
    mechanical_pass = (
        bool(preflight["passed"])
        and identity_matches
        and schedule_matches
        and gradients_pass
        and memory_pass
    )
    passed = mechanical_pass and behavioral_pass
    report: dict[str, Any] = {
        "schema": "marulho.language_persistent_workspace_v72.stage_a1.v1",
        "created_unix_seconds": time.time(),
        "decision": (
            "advance_v72_to_sequential_real_language"
            if passed
            else "retire_v72_persistent_workspace_stage_a1_failure"
        ),
        "passed": passed,
        "recipe_hash": _recipe_hash(config),
        "recipe": {
            "config": vars(config),
            "seeds": SEEDS,
            "arms": ARMS,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "gradient_norm": 1.0,
            "evaluation_documents": EVALUATION_DOCUMENTS,
            "compiled": False,
            "state_detached_between_segments": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "pid": os.getpid(),
        },
        "mechanical": {
            "preflight": preflight,
            "initial_identity_matches_within_seed": identity_matches,
            "schedule_matches_within_seed": schedule_matches,
            "complete_finite_nonzero_gradients": gradients_pass,
            "peak_memory_below_11_5_gib": memory_pass,
            "passed": mechanical_pass,
        },
        "behavioral": {
            "persistent_accuracy_floor": 0.80,
            "control_margin_floor": 0.20,
            "seeds": seed_results,
            "passed": behavioral_pass,
        },
        "arms": arms,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({"decision": report["decision"], "passed": report["passed"]}))


if __name__ == "__main__":
    main()

