"""V73 exact-cortex adaptive-sidecar preflight and quality arm."""

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
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import load_language_tokenizer_state
from marulho.training.language_exact_cortex_sidecar import (
    SidecarMode,
    V73ExactCortexSidecarLanguageModel,
    transfer_v73_transformer_state,
)
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_muon import build_language_muon


ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_CHECKPOINT = (
    ROOT / "reports/language_scaling/v31-general-scaling-qualified-67m-20260711.pt"
)
CONTROL_REPORT = (
    ROOT
    / "reports/language_scaling/persistent-workspace-v72-a2-transformer-20260813.json"
)
CONTROL_PREFLIGHT = (
    ROOT
    / "reports/language_scaling/persistent-workspace-v72-a2-preflight-20260813.json"
)
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
CONTRACT_SHA256 = "eb56d6828e9a89ec7a0a7092663694e5c27c4c1d29dc1104b15ad29d10739d27"
TOKENIZER_SHA256 = "faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715"
DOCUMENT_TOKENS = 961
SEGMENT_LENGTH = 320
SEGMENTS = 3
BATCH_SIZE = 32
TRAIN_STEPS = 256
DATA_SEED = 72121
MODEL_SEED = 72131
LANDMARK_POSITIONS = tuple(range(39, SEGMENT_LENGTH, 40))


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=SEGMENT_LENGTH,
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_exact_cortex_sidecar_v73",
    )


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


def _state_hash(model: torch.nn.Module, *, transformer_only: bool = False) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if transformer_only and name.startswith("sidecar_"):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
    final = "".join(lines).strip()
    if final:
        yield final


def _select(path: Path, count: int, tokenizer: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    selected: list[list[int]] = []
    pending: list[str] = []
    parsed = 0
    eligible = 0

    def flush() -> None:
        nonlocal eligible
        for ids in tokenizer.encode_batch(pending, add_bos=True, add_eos=True):
            if len(ids) >= DOCUMENT_TOKENS:
                eligible += 1
                selected.append(ids[:DOCUMENT_TOKENS])
                if len(selected) >= count:
                    break
        pending.clear()

    for document in _iter_documents(path):
        parsed += 1
        pending.append(document)
        if len(pending) == 128:
            flush()
        if len(selected) >= count:
            break
    if len(selected) < count:
        flush()
    if len(selected) != count:
        raise RuntimeError(f"{path.name} selected {len(selected)} of {count}")
    tensor = torch.tensor(selected, dtype=torch.int32)
    return tensor, {
        "path": str(path),
        "selected": count,
        "parsed": parsed,
        "eligible": eligible,
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
    }


def _prepare_data() -> dict[str, Any]:
    checkpoint = torch.load(TOKENIZER_CHECKPOINT, map_location="cpu")
    tokenizer = load_language_tokenizer_state(checkpoint["tokenizer"])
    if tokenizer.vocabulary_hash() != TOKENIZER_SHA256:
        raise RuntimeError("V73 tokenizer hash mismatch")
    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    selections: dict[str, Any] = {"train": {}, "eval": {}}
    for name, path in zip(SOURCE_NAMES, TRAIN_SOURCES, strict=True):
        tensor, truth = _select(path, 4096, tokenizer)
        train_parts.append(tensor)
        selections["train"][name] = truth
    for name, path in zip(SOURCE_NAMES, EVAL_SOURCES, strict=True):
        tensor, truth = _select(path, 512, tokenizer)
        eval_parts.append(tensor)
        selections["eval"][name] = truth
    train = torch.cat(train_parts)
    evaluation = torch.cat(eval_parts)
    sources = torch.cat(
        [torch.full((512,), index, dtype=torch.long) for index in range(2)]
    )
    schedule = torch.randperm(8192, generator=torch.Generator().manual_seed(DATA_SEED))
    digest = hashlib.sha256()
    digest.update(train.numpy().tobytes())
    digest.update(evaluation.numpy().tobytes())
    digest.update(schedule.numpy().tobytes())
    contract = digest.hexdigest()
    if contract != CONTRACT_SHA256:
        raise RuntimeError(f"V73 immutable data contract mismatch: {contract}")
    return {
        "train": train,
        "eval": evaluation,
        "eval_sources": sources,
        "schedule": schedule,
        "selections": selections,
    }


def _build() -> tuple[V73ExactCortexSidecarLanguageModel, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    control = MarulhoLanguageModel(_config())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    torch.manual_seed(MODEL_SEED + 1)
    candidate = V73ExactCortexSidecarLanguageModel(_config())
    transfer = transfer_v73_transformer_state(control, candidate)
    with torch.no_grad():
        sample = torch.arange(SEGMENT_LENGTH).remainder(8192).unsqueeze(0)
        expected = control(sample, collect_telemetry=False)["logits"]
        actual = candidate.forward_segment(
            sample, candidate.initial_state(1), sidecar_enabled=False
        )["logits"]
    parity = {
        "bit_exact": bool(torch.equal(expected, actual)),
        "max_abs_error": float((expected - actual).abs().max().item()),
    }
    if not parity["bit_exact"]:
        raise RuntimeError(f"V73 disabled parity failed: {parity}")
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    truth = {
        "control_parameter_count": control_count,
        "candidate_parameter_count": candidate_count,
        "parameter_ratio": candidate_count / control_count,
        "disabled_parity": parity,
        "transfer": transfer,
        "initial_transformer_sha256": _state_hash(candidate, transformer_only=True),
    }
    del control
    gc.collect()
    return candidate, truth


def _rate(step: int) -> float:
    if step < 13:
        return 3.0e-5 + 2.7e-4 * ((step + 1) / 13.0)
    progress = (step + 1 - 13) / (TRAIN_STEPS - 13)
    return 3.0e-5 + 0.5 * 2.7e-4 * (1.0 + math.cos(math.pi * progress))


def _segment(
    documents: torch.Tensor,
    indices: torch.Tensor,
    segment: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = documents.index_select(0, indices)
    start = segment * SEGMENT_LENGTH
    return (
        rows[:, start : start + SEGMENT_LENGTH].to(device, dtype=torch.long),
        rows[:, start + 1 : start + SEGMENT_LENGTH + 1].to(device, dtype=torch.long),
    )


def _train(
    model: V73ExactCortexSidecarLanguageModel,
    data: dict[str, Any],
    *,
    mode: SidecarMode,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    model.to(device=device, dtype=torch.bfloat16).train()
    optimizer, optimizer_truth = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
    )
    seen: set[str] = set()
    nonfinite: set[str] = set()
    step_seconds: list[float] = []
    final_losses: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        indices = data["schedule"][step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        for group in optimizer.param_groups:
            group["lr"] = _rate(step)
        optimizer.zero_grad(set_to_none=True)
        state = model.initial_state(BATCH_SIZE)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for segment_index in range(SEGMENTS):
            if segment_index:
                state = model.boundary_state(state, mode)
            inputs, targets = _segment(
                data["train"], indices, segment_index, device
            )
            result = model.forward_segment(inputs, state)
            language = F.cross_entropy(
                result["logits"].flatten(0, 1), targets.flatten()
            )
            landmarks = inputs[:, LANDMARK_POSITIONS]
            auxiliary = F.cross_entropy(
                result["workspace_logits"].flatten(0, 1), landmarks.flatten()
            )
            ((language + 0.1 * auxiliary) / SEGMENTS).backward()
            state = result["next_state"].detach()
            final_losses = {
                "language": float(language.detach().item()),
                "auxiliary": float(auxiliary.detach().item()),
            }
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all().item()):
                nonfinite.add(name)
            elif bool(torch.count_nonzero(parameter.grad).item()):
                seen.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        step_seconds.append(time.perf_counter() - started)
    missing = sorted(set(dict(model.named_parameters())) - seen)
    positions = steps * BATCH_SIZE * SEGMENTS * SEGMENT_LENGTH
    return {
        "steps": steps,
        "positions": positions,
        "seconds": sum(step_seconds),
        "step_seconds": step_seconds,
        "positions_per_second": positions / sum(step_seconds),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "missing_nonzero_gradient_names": missing,
        "nonfinite_gradient_names": sorted(nonfinite),
        "complete_finite_gradients": not missing and not nonfinite,
        "final_losses": final_losses,
        "optimizer": optimizer_truth,
    }


def _evaluate(
    model: V73ExactCortexSidecarLanguageModel,
    data: dict[str, Any],
    *,
    mode: SidecarMode,
    device: torch.device,
) -> dict[str, Any]:
    sums = [0.0, 0.0, 0.0]
    counts = [0, 0, 0]
    source_sums = [0.0, 0.0]
    source_counts = [0, 0]
    correct_state_sum = 0.0
    wrong_state_sum = 0.0
    swap_count = 0
    gates: list[float] = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, 1024, BATCH_SIZE):
            indices = torch.arange(offset, offset + BATCH_SIZE)
            state = model.initial_state(BATCH_SIZE)
            for segment_index in range(SEGMENTS):
                if segment_index:
                    state = model.boundary_state(state, mode)
                incoming = state
                inputs, targets = _segment(data["eval"], indices, segment_index, device)
                result = model.forward_segment(inputs, incoming)
                losses = F.cross_entropy(
                    result["logits"].flatten(0, 1),
                    targets.flatten(),
                    reduction="none",
                ).reshape(BATCH_SIZE, SEGMENT_LENGTH)
                summed = float(losses.sum().item())
                sums[segment_index] += summed
                counts[segment_index] += int(losses.numel())
                gates.append(float(result["content_gate"].mean().item()))
                if segment_index:
                    source_ids = data["eval_sources"].index_select(0, indices)
                    per_document = losses.sum(dim=1)
                    for source in range(2):
                        mask = source_ids == source
                        source_sums[source] += float(per_document[mask].sum().item())
                        source_counts[source] += int(mask.sum().item()) * SEGMENT_LENGTH
                if mode == "persistent" and segment_index:
                    wrong = model.forward_segment(inputs, incoming.roll(1, 0))["logits"]
                    wrong_state_sum += float(
                        F.cross_entropy(
                            wrong.flatten(0, 1), targets.flatten(), reduction="sum"
                        ).item()
                    )
                    correct_state_sum += summed
                    swap_count += int(losses.numel())
                state = result["next_state"].detach()
    segment_losses = [value / count for value, count in zip(sums, counts, strict=True)]
    return {
        "segment_losses": segment_losses,
        "first_segment_loss": segment_losses[0],
        "later_segment_loss": sum(sums[1:]) / sum(counts[1:]),
        "later_loss_by_source": {
            SOURCE_NAMES[index]: source_sums[index] / source_counts[index]
            for index in range(2)
        },
        "state_swap_wrong_minus_correct": (
            (wrong_state_sum - correct_state_sum) / swap_count if swap_count else None
        ),
        "mean_content_gate": sum(gates) / len(gates),
        "read_gates": [float(value) for value in model.sidecar_read_gates.float().cpu()],
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compiled": False,
    }


def run(mode: SidecarMode, output: Path, *, preflight: bool) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V73 requires observed CUDA execution")
    control = json.loads(CONTROL_REPORT.read_text(encoding="utf-8"))
    control_preflight = json.loads(CONTROL_PREFLIGHT.read_text(encoding="utf-8"))
    if control["contract_sha256"] != CONTRACT_SHA256:
        raise RuntimeError("V73 retained control contract mismatch")
    data = _prepare_data()
    model, truth = _build()
    steps = 2 if preflight else TRAIN_STEPS
    device = torch.device("cuda")
    initial_hash = _state_hash(model)
    training = _train(model, data, mode=mode, steps=steps, device=device)
    one_step_control_speed = float(
        control_preflight["rows"][0]["one_step"]["positions_per_second"]
    )
    if preflight:
        evaluation = None
        steady_candidate_speed = (
            BATCH_SIZE * SEGMENTS * SEGMENT_LENGTH / training["step_seconds"][-1]
        )
        checks = {
            "disabled_bit_exact": bool(truth["disabled_parity"]["bit_exact"]),
            "parameter_ratio_at_most_1_02": float(truth["parameter_ratio"]) <= 1.02,
            "complete_finite_gradients": bool(training["complete_finite_gradients"]),
            "throughput_at_least_70_percent": steady_candidate_speed
            / one_step_control_speed
            >= 0.70,
            "peak_below_11_5_gib": int(training["peak_cuda_allocated_bytes"])
            < int(11.5 * 1024**3),
        }
        passed = all(checks.values())
    else:
        evaluation = _evaluate(model, data, mode=mode, device=device)
        checks = None
        passed = None
    payload = {
        "surface": (
            "marulho_exact_cortex_sidecar_v73.preflight.v1"
            if preflight
            else "marulho_exact_cortex_sidecar_v73.quality_arm.v1"
        ),
        "mode": mode,
        "preflight": preflight,
        "passed": passed,
        "checks": checks,
        "contract_sha256": CONTRACT_SHA256,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "source_selections": data["selections"],
        "truth": truth,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_hash(model),
        "training": training,
        "evaluation": evaluation,
        "retained_control": {
            "later_segment_loss": control["evaluation"]["later_segment_loss"],
            "later_loss_by_source": control["evaluation"]["later_loss_by_source"],
            "positions_per_second": control["training"]["positions_per_second"],
            "peak_cuda_allocated_bytes": control["training"][
                "peak_cuda_allocated_bytes"
            ],
            "one_step_positions_per_second": one_step_control_speed,
        },
        "environment": _environment(),
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("persistent", "reset", "shuffled"), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.mode, args.output, preflight=bool(args.preflight))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

