"""V70 frozen 512-update macro-cortex quality falsifier."""

from __future__ import annotations

from collections import Counter
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import torch

from marulho.evaluation.language_matched_support import (
    MatchedLanguageDataConfig,
    prepare_matched_language_data,
)
from marulho.training.language_macro_cortex import (
    MarulhoMacroCortexLanguageModel,
    transfer_transformer_common_state,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)
from marulho.training.language_muon import build_language_muon


ROOT = Path(__file__).resolve().parents[3]
TOKENIZER_CHECKPOINT = ROOT / "reports/language_scaling/v31-general-scaling-qualified-67m-20260711.pt"
RELATION_CORPUS = ROOT / "reports/language_curriculum/relation-binding-train-200k-20260710.txt"
RELATION_CASES = ROOT / "reports/language_curriculum/relation-binding-cases-256-20260710.json"
GENERAL_TRAIN = (
    ROOT / "reports/language_curriculum/fineweb-edu-replay-75k-shard2-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-replay-75k-shard4-20260710.txt",
)
GENERAL_EVAL = (
    ROOT / "reports/language_curriculum/fineweb-edu-eval-10k-shard1-20260710.txt",
    ROOT / "reports/language_curriculum/cosmopedia-v2-eval-10k-shard2-20260710.txt",
)


def _config() -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=8192,
        embedding_dim=768,
        state_dim=768,
        state_layers=10,
        attention_heads=12,
        transformer_context_length=320,
        transformer_mlp_ratio=4.0,
        transformer_dropout=0.0,
        tie_embeddings=True,
        active_language_path="marulho_macro_cortex_v70",
    )


def _is_common(name: str) -> bool:
    return not name.endswith(
        (
            ".summary_queries",
            ".start_macro",
            ".query_macro_scale",
            ".output_macro_scale",
        )
    )


def _common_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not _is_common(name):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _learning_rate(step: int) -> float:
    if step < 26:
        return 3.0e-5 + (3.0e-4 - 3.0e-5) * ((step + 1) / 26.0)
    progress = (step + 1 - 26) / float(512 - 26)
    return 3.0e-5 + 0.5 * (3.0e-4 - 3.0e-5) * (
        1.0 + math.cos(math.pi * progress)
    )


def _build_model(arm: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    torch.manual_seed(70131)
    control = MarulhoLanguageModel(_config())
    common_hash = _common_hash(control)
    if arm == "control":
        return control, {"common_parameter_hash": common_hash, "transfer": None}
    torch.manual_seed(70132)
    candidate = MarulhoMacroCortexLanguageModel(_config())
    transfer = transfer_transformer_common_state(control, candidate)
    if _common_hash(candidate) != common_hash:
        raise RuntimeError("V70 quality common initialization mismatch")
    del control
    gc.collect()
    return candidate, {"common_parameter_hash": common_hash, "transfer": transfer}


def run(arm: str, output: Path) -> dict[str, Any]:
    required = (
        TOKENIZER_CHECKPOINT,
        RELATION_CORPUS,
        RELATION_CASES,
        *GENERAL_TRAIN,
        *GENERAL_EVAL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V70 inputs missing: {missing}")
    device = torch.device("cuda")
    prepared = prepare_matched_language_data(
        tokenizer_checkpoint_path=TOKENIZER_CHECKPOINT,
        relation_corpus_path=RELATION_CORPUS,
        relation_cases_path=RELATION_CASES,
        general_train_paths=GENERAL_TRAIN,
        general_eval_paths=GENERAL_EVAL,
        config=MatchedLanguageDataConfig(
            token_budget=5_242_880,
            sequence_length=320,
            batch_size=32,
            eval_batches=16,
            relation_fraction=0.0,
            seed=70121,
            sample_bytes_per_train_source=64 * 1024 * 1024,
            sample_bytes_per_eval_source=32 * 1024 * 1024,
            sample_range_count=16,
            schedule_mode="indexed_host",
        ),
        device=device,
    )
    if prepared.staged.step_count != 512 or prepared.staged.tokens_per_step != 10240:
        raise RuntimeError("V70 prepared schedule does not match frozen 512x10240")
    counts = Counter(kind for kind, _ in prepared.schedule)
    unique = {
        kind: len({index for row_kind, index in prepared.schedule if row_kind == kind})
        for kind in counts
    }
    if counts.get("relation", 0) or any(unique[kind] != count for kind, count in counts.items()):
        raise RuntimeError("V70 schedule contains relation or repeated windows")

    model, truth = _build_model(arm)
    initial_state_hash = _state_hash(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device=device, dtype=torch.bfloat16)
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
        per_head_attention_qkv=False,
    )
    heldout_curve = [{"step": 0, **evaluate_language_model(model, prepared.eval_batches)}]
    observed_gradients: set[str] = set()
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(512):
        batch = prepared.staged.batch(step, device)
        learning_rate = _learning_rate(step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = model.next_token_loss(
            batch.input_ids,
            batch.target_ids,
            collect_telemetry=False,
            return_evidence=False,
        )["loss"]
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"non-finite V70 gradient: {name}")
                observed_gradients.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        if step + 1 in {128, 256, 512}:
            heldout_curve.append(
                {"step": step + 1, **evaluate_language_model(model, prepared.eval_batches)}
            )
    missing_gradients = sorted(set(dict(model.named_parameters())) - observed_gradients)
    final_state_hash = _state_hash(model)
    peak = torch.cuda.max_memory_allocated()
    payload = {
        "artifact_kind": "marulho_macro_cortex_v70_quality_arm",
        "surface": "marulho_macro_cortex_v70_quality_arm.v1",
        "arm": arm,
        "parameter_count": parameter_count,
        **truth,
        "initial_state_hash": initial_state_hash,
        "final_state_hash": final_state_hash,
        "tokenizer_hash": prepared.tokenizer.vocabulary_hash(),
        "schedule_sha256": prepared.schedule_sha256,
        "source_selections": prepared.source_selections,
        "schedule_counts": dict(counts),
        "schedule_unique_counts": unique,
        "processed_steps": 512,
        "processed_positions": 5_242_880,
        "heldout_curve": heldout_curve,
        "training_seconds": training_seconds,
        "positions_per_second": 5_242_880 / training_seconds,
        "peak_cuda_bytes": peak,
        "below_11_5_gib": peak < int(11.5 * 1024**3),
        "observed_gradient_parameter_count": len(observed_gradients),
        "missing_gradient_names": missing_gradients,
        "optimizer": optimizer_report,
        "learning_rate": {
            "peak": 3.0e-4,
            "minimum": 3.0e-5,
            "warmup_steps": 26,
            "schedule": "linear_warmup_cosine",
        },
        "torch_compile_used": False,
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("candidate", "control"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V70 quality screen requires CUDA")
    print(json.dumps(run(args.arm, args.output), indent=2))


if __name__ == "__main__":
    main()
