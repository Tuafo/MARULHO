"""Falsify answer-gated hidden-state episodic memory on the V39 cortex."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_matched_support import _load_cases, sha256_file
from marulho.evaluation.language_relation_binding_experiment import (
    KINDS,
    RelationCase,
    _heldout_signature,
    _random_values,
    _relation_example,
    evaluate_relation_binding_cases_batched,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_hidden_state_memory import (
    HiddenStateEpisodicMemory,
    HiddenStateMemoryConfig,
    HiddenStateMemoryLanguageModel,
    load_hidden_state_memory,
    memory_state_sha256,
    save_hidden_state_memory,
)
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_hidden_state_memory_falsification.v1"
PARENT_SHA256 = "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d"
ANSWER_MARKER_IDS = (1123, 2839, 265, 31)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_documents(path: Path, *, maximum: int) -> list[str]:
    documents: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("### source=") or line == "<|MARULHO_DOCUMENT|>":
                continue
            if "Answer:" not in line:
                continue
            documents.append(line)
            if len(documents) >= int(maximum):
                break
    if len(documents) < int(maximum):
        raise ValueError("relation corpus does not contain enough datastore documents")
    return documents


@torch.no_grad()
def _build_datastore(
    model: MarulhoLanguageModel,
    tokenizer,
    *,
    corpus_path: Path,
    document_count: int,
    maximum_entries: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    documents = _training_documents(corpus_path, maximum=int(document_count))
    marker_ids = torch.tensor(ANSWER_MARKER_IDS, device=model.device, dtype=torch.long)
    grouped: dict[int, list[tuple[list[int], list[int]]]] = {}
    dropped_context = 0
    for document in documents:
        tokens = tokenizer.encode(document, add_bos=True, add_eos=True)
        if len(tokens) - 1 > model.context_length:
            dropped_context += 1
            continue
        grouped.setdefault(len(tokens) - 1, []).append((tokens[:-1], tokens[1:]))
    key_chunks: list[torch.Tensor] = []
    value_chunks: list[torch.Tensor] = []
    selected_documents = 0
    started = time.perf_counter()
    for entries in grouped.values():
        for start in range(0, len(entries), max(1, int(batch_size))):
            chunk = entries[start : start + max(1, int(batch_size))]
            inputs = torch.tensor(
                [entry[0] for entry in chunk],
                device=model.device,
                dtype=torch.long,
            )
            targets = torch.tensor(
                [entry[1] for entry in chunk],
                device=model.device,
                dtype=torch.long,
            )
            with torch.autocast(
                device_type=model.device.type,
                dtype=torch.bfloat16,
                enabled=model.device.type == "cuda",
            ):
                hidden = model._forward_hidden(  # noqa: SLF001
                    inputs,
                    collect_telemetry=False,
                )["hidden"]
            mask = answer_target_mask(
                inputs,
                marker_ids=marker_ids,
                eos_id=tokenizer.eos_id,
            )
            key_chunks.append(hidden[mask].detach().to(torch.float16).cpu())
            value_chunks.append(targets[mask].detach().long().cpu())
            selected_documents += len(chunk)
    keys = torch.cat(key_chunks, dim=0)[: int(maximum_entries)]
    values = torch.cat(value_chunks, dim=0)[: int(maximum_entries)]
    elapsed = time.perf_counter() - started
    return keys, values, {
        "source_path": str(corpus_path),
        "source_sha256": sha256_file(corpus_path),
        "requested_document_count": int(document_count),
        "selected_document_count": selected_documents,
        "dropped_context_document_count": dropped_context,
        "entry_count": int(keys.shape[0]),
        "key_width": int(keys.shape[1]),
        "value_unique_token_count": int(torch.unique(values).numel()),
        "build_elapsed_seconds": elapsed,
        "keys_storage_bytes": int(keys.numel() * keys.element_size()),
        "values_storage_bytes": int(values.numel() * values.element_size()),
    }


def _calibration_cases(
    frozen_cases: Sequence[RelationCase],
    *,
    cases_per_kind: int,
    seed: int,
) -> tuple[RelationCase, ...]:
    used = {case.signature for case in frozen_cases}
    rng = random.Random(int(seed))
    cases: list[RelationCase] = []
    for kind in KINDS:
        kind_count = 0
        while kind_count < int(cases_per_kind):
            values = _random_values(kind, rng)
            signature, prompt, candidates, correct_index = _relation_example(
                kind,
                values,
                evaluation_template=True,
                seed=rng.randrange(2**31),
            )
            if not _heldout_signature(signature) or signature in used:
                continue
            used.add(signature)
            cases.append(
                RelationCase(
                    case_id=f"calibration-{kind}-{kind_count:04d}",
                    kind=kind,
                    signature=signature,
                    prompt=prompt,
                    candidates=candidates,
                    correct_index=correct_index,
                )
            )
            kind_count += 1
    return tuple(cases)


def _compact_relation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "rows"
    }


def _config_grid() -> tuple[HiddenStateMemoryConfig, ...]:
    return tuple(
        HiddenStateMemoryConfig(
            top_k=top_k,
            similarity_threshold=threshold,
            interpolation_weight=weight,
            temperature=0.1,
        )
        for top_k in (1, 8, 32)
        for threshold in (0.70, 0.85)
        for weight in (0.50, 0.80)
    )


def _select_calibration(
    rows: Sequence[dict[str, Any]],
) -> HiddenStateMemoryConfig:
    best = max(
        rows,
        key=lambda row: (
            float(row["relation"]["generation_exact_accuracy"]),
            min(
                float(row["relation"]["generation_kind_accuracy"]["ownership"]),
                float(row["relation"]["generation_kind_accuracy"]["container"]),
            ),
            float(row["relation"]["accuracy"]),
            -float(row["memory_metrics"]["retrieval_seconds"]),
        ),
    )
    return HiddenStateMemoryConfig(**dict(best["config"]))


def _general_windows(tokenizer, paths: Sequence[Path], *, maximum: int) -> tuple[torch.Tensor, dict[str, Any]]:
    tokens: list[int] = []
    sources = []
    for path in paths:
        with path.open("rb") as handle:
            selected = handle.read(2 * 1024 * 1024)
        text = selected.decode("utf-8", errors="replace")
        tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        sources.append(
            {
                "path": str(path),
                "full_sha256": sha256_file(path),
                "selected_bytes": len(selected),
                "selected_sha256": hashlib.sha256(selected).hexdigest(),
            }
        )
    width = 73
    rows = [
        tokens[index : index + width]
        for index in range(0, len(tokens) - width + 1, width)
    ][: int(maximum)]
    return torch.tensor(rows, dtype=torch.long), {"sources": sources, "row_count": len(rows)}


@torch.no_grad()
def _general_retention(
    model: MarulhoLanguageModel,
    memory_model: HiddenStateMemoryLanguageModel,
    tokenizer,
    *,
    corpus_paths: Sequence[Path],
    maximum_windows: int,
    batch_size: int,
) -> dict[str, Any]:
    windows, provenance = _general_windows(
        tokenizer,
        corpus_paths,
        maximum=int(maximum_windows),
    )
    marker_ids = torch.tensor(ANSWER_MARKER_IDS, device=model.device, dtype=torch.long)
    base_loss_sum = 0.0
    memory_loss_sum = 0.0
    token_count = 0
    active_positions = 0
    gate_off_exact = True
    max_gate_off_difference = 0.0
    for start in range(0, int(windows.shape[0]), max(1, int(batch_size))):
        batch = windows[start : start + max(1, int(batch_size))].to(model.device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        base_logits = model.forward(inputs, collect_telemetry=False)["logits"]
        memory_logits = memory_model.forward(inputs, collect_telemetry=False)["logits"]
        mask = answer_target_mask(
            inputs,
            marker_ids=marker_ids,
            eos_id=tokenizer.eos_id,
        )
        gate_off = ~mask
        if bool(gate_off.any().item()):
            difference = (base_logits[gate_off] - memory_logits[gate_off]).abs()
            gate_off_exact = gate_off_exact and torch.equal(
                base_logits[gate_off], memory_logits[gate_off]
            )
            max_gate_off_difference = max(
                max_gate_off_difference,
                float(difference.max().item()),
            )
        base_loss_sum += float(
            F.cross_entropy(base_logits.flatten(0, 1), targets.flatten(), reduction="sum")
            .cpu()
            .item()
        )
        memory_loss_sum += float(
            F.cross_entropy(memory_logits.flatten(0, 1), targets.flatten(), reduction="sum")
            .cpu()
            .item()
        )
        token_count += int(targets.numel())
        active_positions += int(mask.sum().item())
    base_loss = base_loss_sum / max(1, token_count)
    memory_loss = memory_loss_sum / max(1, token_count)
    return {
        **provenance,
        "token_count": token_count,
        "answer_gate_active_position_count": active_positions,
        "answer_gate_active_fraction": float(active_positions) / float(max(1, token_count)),
        "base_loss": base_loss,
        "memory_loss": memory_loss,
        "memory_loss_regression": memory_loss - base_loss,
        "gate_off_logits_bit_exact": gate_off_exact,
        "maximum_gate_off_logit_difference": max_gate_off_difference,
    }


def run_hidden_state_memory_falsification(
    *,
    checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    frozen_cases_path: str | Path,
    general_eval_corpus_paths: Sequence[str | Path],
    output_path: str | Path,
    memory_output_path: str | Path,
    datastore_document_count: int = 8_192,
    maximum_entries: int = 65_536,
    calibration_cases_per_kind: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    relation_corpus = Path(relation_corpus_path)
    cases_file = Path(frozen_cases_path)
    output = Path(output_path)
    memory_output = Path(memory_output_path)
    checkpoint_sha256 = _file_sha256(checkpoint)
    if checkpoint_sha256 != PARENT_SHA256:
        raise ValueError("V41 requires the preregistered V39 checkpoint")
    resolved_device = torch.device(
        "cuda" if str(device) == "auto" and torch.cuda.is_available() else (
            "cpu" if str(device) == "auto" else str(device)
        )
    )
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model = model.to(resolved_device).eval()
    state_hash_before = language_model_state_sha256(model)
    frozen_cases = _load_cases(cases_file)
    calibration_cases = _calibration_cases(
        frozen_cases,
        cases_per_kind=int(calibration_cases_per_kind),
        seed=41041,
    )
    keys, values, datastore_report = _build_datastore(
        model,
        tokenizer,
        corpus_path=relation_corpus,
        document_count=int(datastore_document_count),
        maximum_entries=int(maximum_entries),
        batch_size=256,
    )
    memory = HiddenStateEpisodicMemory(
        keys,
        values,
        config=_config_grid()[0],
        metadata={
            "parent_checkpoint_sha256": checkpoint_sha256,
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "relation_corpus_sha256": datastore_report["source_sha256"],
            "frozen_cases_sha256": sha256_file(cases_file),
        },
    ).to(resolved_device)
    marker_ids = torch.tensor(ANSWER_MARKER_IDS, device=resolved_device)
    memory_model = HiddenStateMemoryLanguageModel(
        model,
        memory,
        answer_marker_ids=marker_ids,
        eos_id=tokenizer.eos_id,
    )
    calibration_rows = []
    for config in _config_grid():
        memory.config = config
        memory.reset_metrics()
        relation = evaluate_relation_binding_cases_batched(
            memory_model,
            tokenizer,
            calibration_cases,
            batch_size=64,
        )
        calibration_rows.append(
            {
                "config": asdict(config),
                "relation": _compact_relation(relation),
                "memory_metrics": memory.metrics(),
            }
        )
    selected_config = _select_calibration(calibration_rows)
    memory.config = selected_config

    base_relation = evaluate_relation_binding_cases_batched(
        model,
        tokenizer,
        frozen_cases,
        batch_size=64,
    )
    memory.reset_metrics()
    true_relation = evaluate_relation_binding_cases_batched(
        memory_model,
        tokenizer,
        frozen_cases,
        batch_size=64,
    )
    true_metrics = memory.metrics()

    shuffle_generator = torch.Generator(device="cpu").manual_seed(41042)
    permutation = torch.randperm(memory.entry_count, generator=shuffle_generator)
    shuffled_memory = HiddenStateEpisodicMemory(
        memory.keys.detach().cpu(),
        memory.values.detach().cpu()[permutation],
        config=selected_config,
        metadata={**memory.metadata, "value_control": "deterministic_permutation_seed41042"},
    ).to(resolved_device)
    shuffled_model = HiddenStateMemoryLanguageModel(
        model,
        shuffled_memory,
        answer_marker_ids=marker_ids,
        eos_id=tokenizer.eos_id,
    )
    shuffled_relation = evaluate_relation_binding_cases_batched(
        shuffled_model,
        tokenizer,
        frozen_cases,
        batch_size=64,
    )
    shuffled_metrics = shuffled_memory.metrics()
    memory.reset_metrics()
    general_retention = _general_retention(
        model,
        memory_model,
        tokenizer,
        corpus_paths=tuple(Path(path) for path in general_eval_corpus_paths),
        maximum_windows=256,
        batch_size=32,
    )
    general_metrics = memory.metrics()

    free_accuracy = float(true_relation["generation_exact_accuracy"])
    shuffled_free_accuracy = float(shuffled_relation["generation_exact_accuracy"])
    free_kinds = true_relation["generation_kind_accuracy"]
    behavioral_pass = all(
        (
            free_accuracy >= 0.65,
            float(true_relation["accuracy"]) >= 0.98,
            float(free_kinds["ownership"]) >= 0.40,
            float(free_kinds["container"]) >= 0.40,
            free_accuracy - shuffled_free_accuracy >= 0.10,
            float(general_retention["memory_loss_regression"]) <= 0.02,
            bool(general_retention["gate_off_logits_bit_exact"]),
        )
    )
    state_hash_after = language_model_state_sha256(model)
    model_state_immutable = state_hash_before == state_hash_after
    checkpoint_saved = False
    checkpoint_fidelity: dict[str, Any] = {
        "attempted": False,
        "exact": False,
        "path": None,
        "sha256": None,
    }
    if behavioral_pass and model_state_immutable:
        save_hidden_state_memory(memory_output, memory)
        saved_hash = _file_sha256(memory_output)
        restored = load_hidden_state_memory(memory_output, device=resolved_device)
        checkpoint_fidelity = {
            "attempted": True,
            "exact": memory_state_sha256(restored) == memory_state_sha256(memory),
            "path": str(memory_output),
            "sha256": saved_hash,
            "state_sha256": memory_state_sha256(restored),
            "tokenizer_hash_matches": restored.metadata.get("tokenizer_hash")
            == tokenizer.vocabulary_hash(),
            "parent_checkpoint_hash_matches": restored.metadata.get(
                "parent_checkpoint_sha256"
            )
            == checkpoint_sha256,
        }
        checkpoint_saved = bool(
            checkpoint_fidelity["exact"]
            and checkpoint_fidelity["tokenizer_hash_matches"]
            and checkpoint_fidelity["parent_checkpoint_hash_matches"]
        )
        if not checkpoint_saved and memory_output.exists():
            memory_output.unlink()
    success = bool(behavioral_pass and model_state_immutable and checkpoint_saved)
    report = {
        "artifact_kind": "marulho_hidden_state_memory_falsification",
        "surface": SURFACE,
        "decision": (
            "advance_v41_hidden_state_memory_to_semantic_gate_and_index"
            if success
            else "retire_v41_hidden_state_memory_no_joint_free_binding_win"
        ),
        "success": success,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "parent_checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_sha256,
            "model_state_sha256_before": state_hash_before,
            "model_state_sha256_after": state_hash_after,
            "model_state_immutable": model_state_immutable,
            "metadata": metadata,
        },
        "datastore": datastore_report,
        "split": {
            "training_signatures_excluded_by_materializer": True,
            "calibration_case_count": len(calibration_cases),
            "frozen_case_count": len(frozen_cases),
            "calibration_frozen_signature_overlap": len(
                {case.signature for case in calibration_cases}
                & {case.signature for case in frozen_cases}
            ),
            "frozen_cases_path": str(cases_file),
            "frozen_cases_sha256": sha256_file(cases_file),
        },
        "calibration": {
            "selection_policy": "free_then_weak_kind_then_rank_then_latency",
            "selected_config": asdict(selected_config),
            "arms": calibration_rows,
        },
        "frozen_evaluation": {
            "base": base_relation,
            "true_memory": true_relation,
            "shuffled_value_memory": shuffled_relation,
            "true_memory_metrics": true_metrics,
            "shuffled_memory_metrics": shuffled_metrics,
            "free_gain_over_base": free_accuracy
            - float(base_relation["generation_exact_accuracy"]),
            "free_gain_over_shuffled": free_accuracy - shuffled_free_accuracy,
        },
        "general_retention": general_retention,
        "general_memory_metrics": general_metrics,
        "gates": {
            "free_accuracy_at_least_0_65": free_accuracy >= 0.65,
            "candidate_accuracy_at_least_0_98": float(true_relation["accuracy"]) >= 0.98,
            "ownership_at_least_0_40": float(free_kinds["ownership"]) >= 0.40,
            "container_at_least_0_40": float(free_kinds["container"]) >= 0.40,
            "beats_shuffled_by_0_10": free_accuracy - shuffled_free_accuracy >= 0.10,
            "general_loss_regression_at_most_0_02": float(
                general_retention["memory_loss_regression"]
            )
            <= 0.02,
            "gate_off_logits_bit_exact": bool(
                general_retention["gate_off_logits_bit_exact"]
            ),
            "model_state_immutable": model_state_immutable,
            "memory_checkpoint_exact": checkpoint_saved,
        },
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_fidelity": checkpoint_fidelity,
        "compute_interpretation": {
            "active_values_are_top_k": True,
            "search_touches_every_key": True,
            "sparse_gpu_compute_claim": False,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--frozen-cases", type=Path, required=True)
    parser.add_argument("--general-eval-corpus", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-output", type=Path, required=True)
    parser.add_argument("--datastore-document-count", type=int, default=8_192)
    parser.add_argument("--maximum-entries", type=int, default=65_536)
    parser.add_argument("--calibration-cases-per-kind", type=int, default=16)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_hidden_state_memory_falsification(
        checkpoint_path=args.checkpoint,
        relation_corpus_path=args.relation_corpus,
        frozen_cases_path=args.frozen_cases,
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        output_path=args.output,
        memory_output_path=args.memory_output,
        datastore_document_count=int(args.datastore_document_count),
        maximum_entries=int(args.maximum_entries),
        calibration_cases_per_kind=int(args.calibration_cases_per_kind),
        device=args.device,
    )
    return 0 if bool(report["success"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
