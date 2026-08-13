"""V72 frozen sequential long-document language screen."""

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
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import load_language_tokenizer_state
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_muon import build_language_muon
from marulho.training.language_persistent_workspace import (
    LanguageWorkspaceMode,
    V72PersistentWorkspaceLanguageModel,
    transfer_v72_transformer_common_state,
)


ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_CHECKPOINT = (
    ROOT / "reports/language_scaling/v31-general-scaling-qualified-67m-20260711.pt"
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
TRAIN_DOCUMENTS_PER_SOURCE = 4096
EVAL_DOCUMENTS_PER_SOURCE = 512
DOCUMENT_TOKENS = 961
SEGMENT_LENGTH = 320
SEGMENTS = 3
BATCH_SIZE = 32
TRAIN_STEPS = 256
DATA_SEED = 72121
MODEL_SEED = 72131
LANDMARK_POSITIONS = tuple(range(39, SEGMENT_LENGTH, 40))
Arm = Literal["transformer", "persistent", "reset", "shuffled"]
ARMS: tuple[Arm, ...] = ("transformer", "persistent", "reset", "shuffled")


def _control_config() -> LanguageModelConfig:
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
        active_language_path="marulho_transformer_v72_control",
    )


def _workspace_config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=SEGMENT_LENGTH,
        transformer_mlp_ratio=2768 / 768,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_persistent_workspace_v72",
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


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
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
    document = "".join(lines).strip()
    if document:
        yield document


def _select_documents(
    path: Path,
    *,
    count: int,
    tokenizer: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    selected: list[list[int]] = []
    pending: list[str] = []
    parsed = 0
    eligible = 0

    def flush() -> None:
        nonlocal eligible
        if not pending or len(selected) >= count:
            pending.clear()
            return
        for token_ids in tokenizer.encode_batch(pending, add_bos=True, add_eos=True):
            if len(token_ids) < DOCUMENT_TOKENS:
                continue
            eligible += 1
            selected.append(token_ids[:DOCUMENT_TOKENS])
            if len(selected) >= count:
                break
        pending.clear()

    for document in _iter_documents(path):
        parsed += 1
        pending.append(document)
        if len(pending) >= 128:
            flush()
        if len(selected) >= count:
            break
    flush()
    if len(selected) != count:
        raise RuntimeError(
            f"{path.name} has only {len(selected)} eligible documents; expected {count}"
        )
    tensor = torch.tensor(selected, dtype=torch.int32)
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    return tensor, {
        "path": str(path),
        "requested": count,
        "parsed_before_completion": parsed,
        "eligible_before_completion": eligible,
        "selected_token_sha256": digest,
        "document_tokens": DOCUMENT_TOKENS,
        "selection": "first_eligible_in_file_order",
    }


def _load_tokenizer() -> Any:
    payload = torch.load(TOKENIZER_CHECKPOINT, map_location="cpu")
    tokenizer = load_language_tokenizer_state(payload["tokenizer"])
    if tokenizer.vocab_size != 8192:
        raise RuntimeError("V72 tokenizer vocabulary size changed")
    return tokenizer


def _prepare_data() -> dict[str, Any]:
    tokenizer = _load_tokenizer()
    train_parts: list[torch.Tensor] = []
    eval_parts: list[torch.Tensor] = []
    selections: dict[str, Any] = {"train": {}, "eval": {}}
    for name, path in zip(SOURCE_NAMES, TRAIN_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=TRAIN_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
        )
        train_parts.append(tensor)
        selections["train"][name] = report
    for name, path in zip(SOURCE_NAMES, EVAL_SOURCES, strict=True):
        tensor, report = _select_documents(
            path, count=EVAL_DOCUMENTS_PER_SOURCE, tokenizer=tokenizer
        )
        eval_parts.append(tensor)
        selections["eval"][name] = report
    train_documents = torch.cat(train_parts, dim=0)
    eval_documents = torch.cat(eval_parts, dim=0)
    eval_sources = torch.cat(
        [
            torch.full((EVAL_DOCUMENTS_PER_SOURCE,), index, dtype=torch.long)
            for index in range(len(SOURCE_NAMES))
        ]
    )
    generator = torch.Generator().manual_seed(DATA_SEED)
    schedule = torch.randperm(int(train_documents.shape[0]), generator=generator)
    digest = hashlib.sha256()
    digest.update(train_documents.numpy().tobytes())
    digest.update(eval_documents.numpy().tobytes())
    digest.update(schedule.numpy().tobytes())
    return {
        "tokenizer": tokenizer,
        "train_documents": train_documents,
        "eval_documents": eval_documents,
        "eval_sources": eval_sources,
        "schedule": schedule,
        "contract_sha256": digest.hexdigest(),
        "tokenizer_sha256": tokenizer.vocabulary_hash(),
        "selections": selections,
    }


def _build_model(arm: Arm) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(MODEL_SEED)
    control = MarulhoLanguageModel(_control_config())
    control_count = sum(parameter.numel() for parameter in control.parameters())
    if arm == "transformer":
        return control, {
            "control_parameter_count": control_count,
            "parameter_ratio": 1.0,
            "transfer": None,
        }
    torch.manual_seed(MODEL_SEED)
    candidate = V72PersistentWorkspaceLanguageModel(_workspace_config())
    transfer = transfer_v72_transformer_common_state(control, candidate)
    candidate_count = sum(parameter.numel() for parameter in candidate.parameters())
    del control
    gc.collect()
    return candidate, {
        "control_parameter_count": control_count,
        "parameter_ratio": candidate_count / control_count,
        "transfer": transfer,
    }


def _learning_rate(step: int) -> float:
    if step < 13:
        return 3.0e-5 + (3.0e-4 - 3.0e-5) * ((step + 1) / 13.0)
    progress = (step + 1 - 13) / float(TRAIN_STEPS - 13)
    return 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
        1.0 + math.cos(math.pi * progress)
    )


def _workspace_mode(arm: Arm) -> LanguageWorkspaceMode:
    if arm not in {"persistent", "reset", "shuffled"}:
        raise ValueError(f"{arm} has no workspace mode")
    return arm


def _segment_tensors(
    documents: torch.Tensor,
    indices: torch.Tensor,
    segment: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = int(segment) * SEGMENT_LENGTH
    selected = documents.index_select(0, indices)
    inputs = selected[:, start : start + SEGMENT_LENGTH].to(
        device=device, dtype=torch.long, non_blocking=False
    )
    targets = selected[:, start + 1 : start + SEGMENT_LENGTH + 1].to(
        device=device, dtype=torch.long, non_blocking=False
    )
    return inputs, targets


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


def _train(
    model: torch.nn.Module,
    *,
    arm: Arm,
    data: dict[str, Any],
    device: torch.device,
    steps: int,
) -> dict[str, Any]:
    model.to(device=device, dtype=torch.bfloat16)
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )
    documents: torch.Tensor = data["train_documents"]
    schedule: torch.Tensor = data["schedule"]
    model.train()
    training_seconds = 0.0
    gradient_audit: dict[str, Any] | None = None
    final_losses: dict[str, float] = {}
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(int(steps)):
        indices = schedule[step * BATCH_SIZE : (step + 1) * BATCH_SIZE]
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step)
        optimizer.zero_grad(set_to_none=True)
        workspace: torch.Tensor | None = None
        if arm != "transformer":
            assert isinstance(model, V72PersistentWorkspaceLanguageModel)
            workspace = model.initial_workspace(BATCH_SIZE)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for segment in range(SEGMENTS):
            inputs, targets = _segment_tensors(documents, indices, segment, device)
            if arm == "transformer":
                assert isinstance(model, MarulhoLanguageModel)
                logits = model(inputs, collect_telemetry=False)["logits"]
                language_loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
                )
                auxiliary_loss = torch.zeros((), device=device)
            else:
                assert isinstance(model, V72PersistentWorkspaceLanguageModel)
                assert workspace is not None
                if segment > 0:
                    workspace = model.boundary_workspace(
                        workspace, _workspace_mode(arm)
                    )
                result = model.forward_segment(inputs, workspace)
                language_loss = F.cross_entropy(
                    result["logits"].reshape(-1, result["logits"].shape[-1]),
                    targets.reshape(-1),
                )
                landmark_targets = inputs[:, LANDMARK_POSITIONS]
                auxiliary_loss = F.cross_entropy(
                    result["workspace_logits"].reshape(
                        -1, result["workspace_logits"].shape[-1]
                    ),
                    landmark_targets.reshape(-1),
                )
                workspace = result["next_workspace"]
            ((language_loss + 0.1 * auxiliary_loss) / SEGMENTS).backward()
            if workspace is not None:
                workspace = workspace.detach()
            final_losses = {
                "language": float(language_loss.detach().item()),
                "workspace_reconstruction": float(auxiliary_loss.detach().item()),
            }
        if gradient_audit is None:
            gradient_audit = _gradient_audit(model)
            if not gradient_audit["passed"]:
                raise RuntimeError(f"V72 incomplete gradient audit: {gradient_audit}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        training_seconds += time.perf_counter() - started
    assert gradient_audit is not None
    return {
        "steps": int(steps),
        "positions": int(steps) * BATCH_SIZE * SEGMENTS * SEGMENT_LENGTH,
        "seconds": training_seconds,
        "positions_per_second": (
            int(steps) * BATCH_SIZE * SEGMENTS * SEGMENT_LENGTH / training_seconds
        ),
        "gradient_audit": gradient_audit,
        "final_losses": final_losses,
        "optimizer": optimizer_report,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _evaluate(
    model: torch.nn.Module,
    *,
    arm: Arm,
    data: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    documents: torch.Tensor = data["eval_documents"]
    source_ids: torch.Tensor = data["eval_sources"]
    loss_sums = [0.0, 0.0, 0.0]
    token_counts = [0, 0, 0]
    later_source_sums = [0.0 for _ in SOURCE_NAMES]
    later_source_counts = [0 for _ in SOURCE_NAMES]
    swap_correct_sum = 0.0
    swap_wrong_sum = 0.0
    swap_tokens = 0
    model.eval()
    with torch.no_grad():
        for offset in range(0, int(documents.shape[0]), BATCH_SIZE):
            indices = torch.arange(offset, offset + BATCH_SIZE)
            workspace: torch.Tensor | None = None
            if arm != "transformer":
                assert isinstance(model, V72PersistentWorkspaceLanguageModel)
                workspace = model.initial_workspace(BATCH_SIZE)
            for segment in range(SEGMENTS):
                inputs, targets = _segment_tensors(documents, indices, segment, device)
                incoming: torch.Tensor | None = None
                if arm == "transformer":
                    assert isinstance(model, MarulhoLanguageModel)
                    logits = model(inputs, collect_telemetry=False)["logits"]
                else:
                    assert isinstance(model, V72PersistentWorkspaceLanguageModel)
                    assert workspace is not None
                    if segment > 0:
                        workspace = model.boundary_workspace(
                            workspace, _workspace_mode(arm)
                        )
                    incoming = workspace
                    result = model.forward_segment(inputs, incoming)
                    logits = result["logits"]
                    workspace = result["next_workspace"].detach()
                summed = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="sum",
                )
                count = int(targets.numel())
                loss_sums[segment] += float(summed.item())
                token_counts[segment] += count
                if segment > 0:
                    batch_sources = source_ids.index_select(0, indices)
                    per_token = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="none",
                    ).reshape(BATCH_SIZE, SEGMENT_LENGTH)
                    per_document = per_token.sum(dim=1)
                    for source in range(len(SOURCE_NAMES)):
                        mask = batch_sources == source
                        later_source_sums[source] += float(per_document[mask].sum().item())
                        later_source_counts[source] += int(mask.sum().item()) * SEGMENT_LENGTH
                if arm == "persistent" and segment > 0:
                    assert isinstance(model, V72PersistentWorkspaceLanguageModel)
                    assert incoming is not None
                    wrong = incoming.roll(1, dims=0)
                    wrong_logits = model.forward_segment(inputs, wrong)["logits"]
                    wrong_loss = F.cross_entropy(
                        wrong_logits.reshape(-1, wrong_logits.shape[-1]),
                        targets.reshape(-1),
                        reduction="sum",
                    )
                    swap_correct_sum += float(summed.item())
                    swap_wrong_sum += float(wrong_loss.item())
                    swap_tokens += count
    segment_losses = [
        loss_sum / count for loss_sum, count in zip(loss_sums, token_counts, strict=True)
    ]
    later_sum = sum(loss_sums[1:])
    later_count = sum(token_counts[1:])
    return {
        "segment_losses": segment_losses,
        "first_segment_loss": segment_losses[0],
        "later_segment_loss": later_sum / later_count,
        "later_loss_by_source": {
            name: later_source_sums[index] / later_source_counts[index]
            for index, name in enumerate(SOURCE_NAMES)
        },
        "state_swap": {
            "correct_loss": (
                swap_correct_sum / swap_tokens if swap_tokens else None
            ),
            "wrong_loss": swap_wrong_sum / swap_tokens if swap_tokens else None,
            "wrong_minus_correct": (
                (swap_wrong_sum - swap_correct_sum) / swap_tokens
                if swap_tokens
                else None
            ),
            "tokens": swap_tokens,
        },
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "torch_compile_used": False,
    }


def run_arm(arm: Arm, output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V72 A2 requires observed CUDA execution")
    data = _prepare_data()
    model, truth = _build_model(arm)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if not 0.99 <= float(truth["parameter_ratio"]) <= 1.01:
        raise RuntimeError(f"V72 parameter ratio failed: {truth['parameter_ratio']}")
    initial_hash = _state_hash(model)
    device = torch.device("cuda")
    training = _train(
        model,
        arm=arm,
        data=data,
        device=device,
        steps=TRAIN_STEPS,
    )
    evaluation = _evaluate(model, arm=arm, data=data, device=device)
    payload = {
        "surface": "marulho_persistent_workspace_v72.a2_arm.v1",
        "arm": arm,
        "contract_sha256": data["contract_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "source_selections": data["selections"],
        "parameter_count": parameter_count,
        **truth,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": _state_hash(model),
        "training": training,
        "evaluation": evaluation,
        "recipe": {
            "data_seed": DATA_SEED,
            "model_seed": MODEL_SEED,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "segments": SEGMENTS,
            "segment_length": SEGMENT_LENGTH,
            "processed_positions": TRAIN_STEPS
            * BATCH_SIZE
            * SEGMENTS
            * SEGMENT_LENGTH,
            "compiled": False,
            "workspace_reconstruction_weight": 0.1 if arm != "transformer" else 0.0,
            "state_detached_between_segments": arm != "transformer",
        },
        "environment": _environment(),
    }
    _atomic_json(output, payload)
    return payload


def run_preflight(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V72 A2 preflight requires CUDA")
    device = torch.device("cuda")
    data = _prepare_data()
    rows: list[dict[str, Any]] = []
    for arm in ("transformer", "persistent"):
        model, truth = _build_model(arm)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        training = _train(
            model,
            arm=arm,
            data=data,
            device=device,
            steps=1,
        )
        rows.append(
            {
                "arm": arm,
                "parameter_count": parameter_count,
                **truth,
                "one_step": training,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    control, candidate = rows
    payload = {
        "surface": "marulho_persistent_workspace_v72.a2_preflight.v1",
        "contract_sha256": data["contract_sha256"],
        "tokenizer_sha256": data["tokenizer_sha256"],
        "rows": rows,
        "checks": {
            "parameter_ratio_0_99_to_1_01": 0.99
            <= float(candidate["parameter_ratio"])
            <= 1.01,
            "complete_gradients": all(
                bool(row["one_step"]["gradient_audit"]["passed"]) for row in rows
            ),
            "below_11_5_gib": all(
                int(row["one_step"]["peak_cuda_allocated_bytes"])
                < int(11.5 * 1024**3)
                for row in rows
            ),
            "candidate_throughput_ratio": candidate["one_step"][
                "positions_per_second"
            ]
            / control["one_step"]["positions_per_second"],
        },
        "environment": _environment(),
    }
    payload["passed"] = (
        payload["checks"]["parameter_ratio_0_99_to_1_01"]
        and payload["checks"]["complete_gradients"]
        and payload["checks"]["below_11_5_gib"]
        and payload["checks"]["candidate_throughput_ratio"] >= 0.70
    )
    _atomic_json(output, payload)
    return payload


def aggregate(inputs: list[Path], output: Path) -> dict[str, Any]:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    by_arm = {str(row["arm"]): row for row in rows}
    if set(by_arm) != set(ARMS):
        raise ValueError(f"V72 aggregate requires exactly {ARMS}")
    if len({row["contract_sha256"] for row in rows}) != 1:
        raise RuntimeError("V72 arm data contracts differ")
    persistent = by_arm["persistent"]
    persistent_loss = float(persistent["evaluation"]["later_segment_loss"])
    loss_margins = {
        arm: float(by_arm[arm]["evaluation"]["later_segment_loss"])
        - persistent_loss
        for arm in ("transformer", "reset", "shuffled")
    }
    throughput_ratio = float(persistent["training"]["positions_per_second"]) / float(
        by_arm["transformer"]["training"]["positions_per_second"]
    )
    state_swap_delta = float(
        persistent["evaluation"]["state_swap"]["wrong_minus_correct"]
    )
    peak = int(persistent["training"]["peak_cuda_allocated_bytes"])
    checks = {
        "persistent_beats_every_arm_by_0_02": min(loss_margins.values()) >= 0.02,
        "persistent_throughput_at_least_70_percent": throughput_ratio >= 0.70,
        "persistent_peak_below_11_5_gib": peak < int(11.5 * 1024**3),
        "state_swap_worsens_loss_by_0_02": state_swap_delta >= 0.02,
        "all_gradients_complete": all(
            bool(row["training"]["gradient_audit"]["passed"]) for row in rows
        ),
        "parameter_ratio_0_99_to_1_01": 0.99
        <= float(persistent["parameter_ratio"])
        <= 1.01,
    }
    passed = all(checks.values())
    payload = {
        "surface": "marulho_persistent_workspace_v72.a2_decision.v1",
        "decision": (
            "advance_v72_persistent_workspace_to_stage_b"
            if passed
            else "retire_v72_persistent_workspace_real_language_failure"
        ),
        "passed": passed,
        "checks": checks,
        "later_segment_losses": {
            arm: float(by_arm[arm]["evaluation"]["later_segment_loss"])
            for arm in ARMS
        },
        "persistent_loss_margins": loss_margins,
        "persistent_throughput_ratio": throughput_ratio,
        "persistent_state_swap_delta": state_swap_delta,
        "persistent_peak_cuda_bytes": peak,
        "arm_reports": [str(path) for path in inputs],
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--aggregate", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = sum(
        [bool(args.arm), bool(args.preflight), bool(args.aggregate)]
    )
    if selected != 1:
        raise ValueError("choose exactly one of --arm, --preflight, or --aggregate")
    if args.preflight:
        result = run_preflight(args.output)
    elif args.aggregate:
        result = aggregate(args.aggregate, args.output)
    else:
        result = run_arm(args.arm, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

