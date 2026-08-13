"""V71 frozen 512-update periodic-hierarchy quality falsifier."""

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
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    evaluate_language_model,
)
from marulho.training.language_muon import build_language_muon
from marulho.training.language_periodic_hierarchy import (
    MarulhoPeriodicHierarchyLanguageModel,
    transfer_periodic_common_state,
)


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
EXPECTED_COMMON_HASH = "700f403ac0405b11cc25262f87434b9a00174d4ed10bc46198e778b7ad84127a"
EXPECTED_TOKENIZER_HASH = "faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715"
EXPECTED_SCHEDULE_HASH = "8342013bb10d842f136c28338664e24db3132c13f5f160ea1eb94065b99daa07"


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
        active_language_path="marulho_periodic_hierarchy_v71",
    )


def _hash(model: torch.nn.Module, *, common_only: bool) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if common_only and name.endswith(
            (".summary_queries", ".start_macro", ".query_macro_scale", ".output_macro_scale")
        ):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _learning_rate(step: int) -> float:
    if step < 26:
        return 3.0e-5 + 2.7e-4 * ((step + 1) / 26.0)
    progress = (step + 1 - 26) / 486.0
    return 3.0e-5 + 0.5 * 2.7e-4 * (1.0 + math.cos(math.pi * progress))


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


def _model(arm: str) -> tuple[MarulhoPeriodicHierarchyLanguageModel, dict[str, Any]]:
    torch.manual_seed(70131)
    control = MarulhoLanguageModel(_config())
    if _hash(control, common_only=True) != EXPECTED_COMMON_HASH:
        raise RuntimeError("V71 immutable Transformer common hash mismatch")
    torch.manual_seed(71132)
    model = MarulhoPeriodicHierarchyLanguageModel(
        _config(), macro_enabled=arm == "periodic_macro"
    )
    transfer = transfer_periodic_common_state(control, model)
    del control
    gc.collect()
    common_hash = _hash(model, common_only=True)
    if common_hash != EXPECTED_COMMON_HASH:
        raise RuntimeError("V71 arm common hash mismatch")
    return model, {"common_parameter_hash": common_hash, "transfer": transfer}


def run(arm: str, output: Path) -> dict[str, Any]:
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
    if prepared.tokenizer.vocabulary_hash() != EXPECTED_TOKENIZER_HASH:
        raise RuntimeError("V71 tokenizer hash mismatch")
    if prepared.schedule_sha256 != EXPECTED_SCHEDULE_HASH:
        raise RuntimeError("V71 schedule hash mismatch")
    if prepared.staged.step_count != 512 or prepared.staged.tokens_per_step != 10240:
        raise RuntimeError("V71 schedule shape mismatch")
    counts = Counter(kind for kind, _ in prepared.schedule)
    unique = {
        kind: len({index for row_kind, index in prepared.schedule if row_kind == kind})
        for kind in counts
    }
    if counts.get("relation", 0) or unique != dict(counts):
        raise RuntimeError("V71 schedule contains relation or repeated windows")

    model, truth = _model(arm)
    initial_state_hash = _hash(model, common_only=False)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device=device, dtype=torch.bfloat16)
    optimizer, optimizer_report = build_language_muon(
        model,
        learning_rate=3.0e-4,
        weight_decay=0.1,
        compile_orthogonalizer=False,
    )
    curve = [{"step": 0, **evaluate_language_model(model, prepared.eval_batches)}]
    observed: set[str] = set()
    training_seconds = 0.0
    torch.cuda.reset_peak_memory_stats()
    for step in range(512):
        batch = prepared.staged.batch(step, device)
        for group in optimizer.param_groups:
            group["lr"] = _learning_rate(step)
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
                    raise RuntimeError(f"non-finite V71 gradient: {name}")
                observed.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        if step + 1 in {128, 256, 512}:
            curve.append({"step": step + 1, **evaluate_language_model(model, prepared.eval_batches)})
    missing = sorted(set(dict(model.named_parameters())) - observed)
    peak = torch.cuda.max_memory_allocated()
    payload = {
        "artifact_kind": "marulho_periodic_hierarchy_v71_quality_arm",
        "surface": "marulho_periodic_hierarchy_v71_quality_arm.v1",
        "arm": arm,
        "parameter_count": parameter_count,
        **truth,
        "initial_state_hash": initial_state_hash,
        "final_state_hash": _hash(model, common_only=False),
        "tokenizer_hash": prepared.tokenizer.vocabulary_hash(),
        "schedule_sha256": prepared.schedule_sha256,
        "source_selections": prepared.source_selections,
        "schedule_counts": dict(counts),
        "schedule_unique_counts": unique,
        "processed_steps": 512,
        "processed_positions": 5_242_880,
        "heldout_curve": curve,
        "training_seconds": training_seconds,
        "positions_per_second": 5_242_880 / training_seconds,
        "peak_cuda_bytes": peak,
        "below_11_5_gib": peak < int(11.5 * 1024**3),
        "missing_gradient_names": missing,
        "optimizer": optimizer_report,
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
    parser.add_argument("--arm", choices=("periodic_macro", "periodic_local"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("V71 quality screen requires CUDA")
    print(json.dumps(run(args.arm, args.output), indent=2))


if __name__ == "__main__":
    main()
