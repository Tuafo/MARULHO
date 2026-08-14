"""Frozen V76 exact versus first-order meta-gradient screen."""

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
from torch.nn.attention import SDPBackend, sdpa_kernel

from marulho.training.language_exact_ttt import (
    MetaGradient,
    UpdateMode,
    V76Config,
    V76ExactTTT,
    make_v76_batch,
)


SEEDS = (7601, 7602, 7603)
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
    model = V76ExactTTT().to(device).train()
    batch = make_v76_batch(
        model.config,
        batch_size=8,
        generator=torch.Generator().manual_seed(seed + 500_000),
        device=device,
    )
    fast_a, fast_b = model.initial_fast_weights(8)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        disabled = model.forward_segment(
            batch.tokens[:, 0, :-1], fast_a, fast_b, fast_enabled=False
        )
        enabled = model.forward_segment(batch.tokens[:, 0, :-1], fast_a, fast_b)
        exact = model.episode(batch, meta_gradient="exact")
        first_order = model.episode(batch, meta_gradient="first_order")
        exact["loss"].backward()
        exact_outer_gradient = model.fast_a0.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        first_order["loss"].backward()
        first_outer_gradient = model.fast_a0.grad.detach().clone()
        model.zero_grad(set_to_none=True)
        changed = make_v76_batch(
            model.config,
            batch_size=8,
            generator=torch.Generator().manual_seed(seed + 500_001),
            device=device,
        )
        changed.tokens[:, :2] = batch.tokens[:, :2]
        future_changed = model.episode(changed, meta_gradient="exact")
    checks = {
        "disabled_max_abs_error": float((enabled - disabled).abs().max().item()),
        "exact_first_order_logit_max_abs_error": float(
            (exact["query_logits"].detach() - first_order["query_logits"].detach())
            .abs()
            .max()
            .item()
        ),
        "exact_first_order_update_max_abs_error": float(
            (exact["update_norms"] - first_order["update_norms"])
            .abs()
            .max()
            .item()
        ),
        "future_perturbation_earlier_loss_max_abs_error": float(
            (exact["segment_losses"][:2] - future_changed["segment_losses"][:2])
            .abs()
            .max()
            .item()
        ),
        "future_perturbation_update_max_abs_error": float(
            (exact["update_norms"] - future_changed["update_norms"])
            .abs()
            .max()
            .item()
        ),
        "outer_gradient_difference_l2": float(
            torch.linalg.vector_norm(exact_outer_gradient - first_outer_gradient).item()
        ),
    }
    checks["passed"] = (
        checks["disabled_max_abs_error"] == 0.0
        and checks["exact_first_order_logit_max_abs_error"] == 0.0
        and checks["exact_first_order_update_max_abs_error"] == 0.0
        and checks["future_perturbation_earlier_loss_max_abs_error"] == 0.0
        and checks["future_perturbation_update_max_abs_error"] == 0.0
        and checks["outer_gradient_difference_l2"] > 0.0
    )
    return checks


def _train(
    seed: int,
    meta_gradient: MetaGradient,
    device: torch.device,
) -> tuple[V76ExactTTT, dict[str, Any]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = V76ExactTTT().to(device).train()
    initial_hash = _hash_model(model)
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
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        for _step in range(TRAIN_STEPS):
            batch = make_v76_batch(
                model.config,
                batch_size=BATCH_SIZE,
                generator=generator,
                device=device,
            )
            schedule.update(batch.tokens.detach().cpu().numpy().tobytes())
            optimizer.zero_grad(set_to_none=True)
            result = model.episode(batch, meta_gradient=meta_gradient)
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
    missing = sorted(set(dict(model.named_parameters())) - observed)
    row = {
        "meta_gradient": meta_gradient,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": _hash_model(model),
        "schedule_sha256": schedule.hexdigest(),
        "steps": TRAIN_STEPS,
        "documents": TRAIN_STEPS * BATCH_SIZE,
        "seconds": seconds,
        "documents_per_second": TRAIN_STEPS * BATCH_SIZE / seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "missing_nonzero_gradient_names": missing,
        "nonfinite_gradient_names": sorted(nonfinite),
        "complete_finite_gradients": not missing and not nonfinite,
        "final": final,
    }
    print(
        f"V76 seed={seed} train={meta_gradient} "
        f"docs/s={row['documents_per_second']:.1f} "
        f"peak_gb={row['peak_cuda_allocated_bytes'] / 1e9:.3f}",
        flush=True,
    )
    return model, row


def _make_eval_batches(seed: int, device: torch.device) -> list[Any]:
    generator = torch.Generator().manual_seed(seed + 2_000_000)
    return [
        make_v76_batch(
            V76Config(), batch_size=BATCH_SIZE, generator=generator, device=device
        )
        for _ in range(EVAL_DOCUMENTS // BATCH_SIZE)
    ]


def _evaluate(
    model: V76ExactTTT,
    *,
    update_mode: UpdateMode,
    batches: list[Any],
) -> dict[str, Any]:
    correct = 0
    total = 0
    losses = torch.zeros(3, dtype=torch.float64)
    update_norms = torch.zeros(2, dtype=torch.float64)
    started = time.perf_counter()
    model.eval()
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        for batch in batches:
            result = model.episode(
                batch, meta_gradient="first_order", update_mode=update_mode
            )
            predictions = result["query_logits"].detach().argmax(dim=-1)
            correct += int((predictions == batch.query_values).sum().item())
            total += int(batch.query_values.numel())
            losses += result["segment_losses"].detach().cpu().double()
            update_norms += result["update_norms"].detach().cpu().double()
    seconds = time.perf_counter() - started
    return {
        "query_accuracy": correct / total,
        "query_correct": correct,
        "query_total": total,
        "mean_segment_losses": (losses / len(batches)).tolist(),
        "mean_update_norms": (update_norms / len(batches)).tolist(),
        "inner_rate": float(torch.nn.functional.softplus(model.inner_log_rate).item()),
        "seconds": seconds,
        "documents_per_second": EVAL_DOCUMENTS / seconds,
    }


def run_seed(seed: int, output: Path) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"V76 seed must be one of {SEEDS}")
    if not torch.cuda.is_available():
        raise RuntimeError("V76 requires observed CUDA execution")
    device = torch.device("cuda")
    mechanical = _mechanical(seed, device)
    exact_model, exact_training = _train(seed, "exact", device)
    first_model, first_training = _train(seed, "first_order", device)
    initial_identity = (
        exact_training["initial_parameter_sha256"]
        == first_training["initial_parameter_sha256"]
    )
    schedule_identity = (
        exact_training["schedule_sha256"] == first_training["schedule_sha256"]
    )
    batches = _make_eval_batches(seed, device)
    arms: list[dict[str, Any]] = []
    for name, model, mode in (
        ("exact_own", exact_model, "own"),
        ("first_order_own", first_model, "own"),
        ("exact_discard", exact_model, "discard"),
        ("exact_shuffled", exact_model, "shuffled"),
    ):
        evaluation = _evaluate(model, update_mode=mode, batches=batches)
        arms.append({"arm": name, "evaluation": evaluation})
        print(
            f"V76 seed={seed} arm={name} "
            f"accuracy={evaluation['query_accuracy']:.4f} "
            f"docs/s={evaluation['documents_per_second']:.1f}",
            flush=True,
        )
    accuracy = {row["arm"]: row["evaluation"]["query_accuracy"] for row in arms}
    exact_accuracy = float(accuracy["exact_own"])
    margins = {
        name: exact_accuracy - float(value)
        for name, value in accuracy.items()
        if name != "exact_own"
    }
    behavioral = (
        exact_accuracy >= 0.80
        and margins["first_order_own"] >= 0.10
        and margins["exact_discard"] >= 0.20
        and margins["exact_shuffled"] >= 0.20
    )
    gradients = bool(
        exact_training["complete_finite_gradients"]
        and first_training["complete_finite_gradients"]
    )
    passed = bool(
        mechanical["passed"]
        and initial_identity
        and schedule_identity
        and gradients
        and behavioral
    )
    payload = {
        "surface": "marulho_exact_ttt_v76.stage_a0_seed.v1",
        "seed": seed,
        "passed": passed,
        "decision": "seed_pass" if passed else "retire_v76_stage_a0_failure",
        "recipe": {
            "config": vars(V76Config()),
            "train_steps": TRAIN_STEPS,
            "physical_batch": BATCH_SIZE,
            "effective_batch": BATCH_SIZE,
            "evaluation_documents": EVAL_DOCUMENTS,
            "slow_optimizer": "AdamW",
            "slow_learning_rate": 3.0e-4,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "attention_backend": "torch_math_sdpa",
            "compiled": False,
        },
        "mechanical": mechanical,
        "initial_identity_matches": initial_identity,
        "schedule_matches": schedule_identity,
        "complete_finite_gradients": gradients,
        "training": [exact_training, first_training],
        "exact_accuracy": exact_accuracy,
        "exact_margins": margins,
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
