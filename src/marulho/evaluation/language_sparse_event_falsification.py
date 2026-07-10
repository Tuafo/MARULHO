"""Matched real-language falsification for sparse event-memory v2."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import (
    LANGUAGE_DOCUMENT_SEPARATOR,
    load_language_tokenizer_state,
)
from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _learning_rate,
    _optimizer,
    _precision_context,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
)
from marulho.training.language_sparse_event_memory import (
    MarulhoSparseEventLanguageModel,
    SparseEventMemoryConfig,
)


SURFACE = "marulho_sparse_event_falsification.v1"
ARTIFACT_KIND = "marulho_sparse_event_falsification"


@dataclass(frozen=True)
class SparseEventFalsificationConfig:
    token_budget: int = 16_777_216
    sequence_length: int = 72
    batch_size: int = 144
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    relation_fraction: float = 0.20
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    seed: int = 1337
    model_seed: int = 1337
    sample_bytes_per_train_source: int = 64 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    event_interval: int = 24
    specialist_count: int = 4
    specialist_rank: int = 32
    exploration_rate: float = 0.10
    counterfactual_rate: float = 0.125
    utility_loss_weight: float = 0.05
    compute_cost: float = 1.0e-4
    initial_residual_scale: float = 1.0e-3


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_corpus_ranges(
    path: str | Path, *, byte_budget: int, range_count: int
) -> tuple[str, dict[str, Any]]:
    source = Path(path)
    size = source.stat().st_size
    budget = max(1, min(int(byte_budget), size))
    if budget >= size:
        data = source.read_bytes()
        return data.decode("utf-8"), {
            "path": str(source),
            "source_size_bytes": size,
            "selected_size_bytes": len(data),
            "selected_sha256": hashlib.sha256(data).hexdigest(),
            "ranges": [{"start": 0, "end": size}],
        }
    count = max(1, int(range_count))
    chunk_size = max(1, budget // count)
    maximum_start = max(0, size - chunk_size)
    starts = [
        round(index * maximum_start / max(1, count - 1)) for index in range(count)
    ]
    chunks: list[bytes] = []
    ranges: list[dict[str, int]] = []
    with source.open("rb") as handle:
        for nominal in starts:
            handle.seek(int(nominal))
            if nominal > 0:
                handle.readline()
            start = handle.tell()
            data = handle.read(chunk_size)
            data += handle.readline()
            if data:
                chunks.append(data)
                ranges.append({"start": int(start), "end": int(handle.tell())})
    selected = f"\n{LANGUAGE_DOCUMENT_SEPARATOR}\n".encode("utf-8").join(chunks)
    return selected.decode("utf-8"), {
        "path": str(source),
        "source_size_bytes": size,
        "selected_size_bytes": len(selected),
        "selected_sha256": hashlib.sha256(selected).hexdigest(),
        "ranges": ranges,
    }


def build_matched_schedule(
    *,
    step_count: int,
    relation_fraction: float,
    relation_batch_count: int,
    general_batch_counts: Sequence[int],
    seed: int,
) -> tuple[tuple[str, int], ...]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    orders = {
        "relation": torch.randperm(relation_batch_count, generator=generator).tolist(),
        **{
            f"general_{index}": torch.randperm(count, generator=generator).tolist()
            for index, count in enumerate(general_batch_counts)
        },
    }
    cursors = {name: 0 for name in orders}
    accumulator = 0.0
    source_cursor = 0
    schedule: list[tuple[str, int]] = []
    for _ in range(int(step_count)):
        accumulator += float(relation_fraction)
        if accumulator >= 1.0:
            accumulator -= 1.0
            kind = "relation"
        else:
            kind = f"general_{source_cursor % len(general_batch_counts)}"
            source_cursor += 1
        order = orders[kind]
        cursor = cursors[kind]
        if cursor >= len(order):
            order = torch.randperm(len(order), generator=generator).tolist()
            orders[kind] = order
            cursor = 0
        schedule.append((kind, int(order[cursor])))
        cursors[kind] = cursor + 1
    return tuple(schedule)


def build_probe_schedule(*, steps: int, rate: float, seed: int) -> tuple[bool, ...]:
    count = min(int(steps), max(0, int(round(int(steps) * float(rate)))))
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    selected = set(
        torch.randperm(int(steps), generator=generator)[:count].tolist()
    )
    return tuple(index in selected for index in range(int(steps)))


def _load_tokenizer(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("surface") != "marulho_transformer_language_checkpoint.v2":
        raise ValueError("Tokenizer source must be a Transformer checkpoint")
    return load_language_tokenizer_state(payload["tokenizer"])


def _load_cases(path: Path) -> tuple[RelationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        RelationCase(
            case_id=str(row["case_id"]),
            kind=str(row["kind"]),
            signature=str(row["signature"]),
            prompt=str(row["prompt"]),
            candidates=tuple(str(value) for value in row["candidates"]),
            correct_index=int(row["correct_index"]),
        )
        for row in payload["cases"]
    )


def _build_model(name: str, *, vocab_size: int, config: SparseEventFalsificationConfig):
    common = dict(
        vocab_size=int(vocab_size),
        embedding_dim=512,
        state_dim=512,
        state_layers=4,
        attention_heads=8,
        transformer_context_length=config.sequence_length,
        transformer_mlp_ratio=4.0,
    )
    if name == "exact_only":
        return MarulhoLanguageModel(LanguageModelConfig(**common))
    return MarulhoSparseEventLanguageModel(
        SparseEventMemoryConfig(
            **common,
            selection_mode=name,
            event_interval=config.event_interval,
            specialist_count=config.specialist_count,
            specialist_rank=config.specialist_rank,
            exploration_rate=config.exploration_rate,
            counterfactual_rate=config.counterfactual_rate,
            utility_loss_weight=config.utility_loss_weight,
            compute_cost=config.compute_cost,
            initial_residual_scale=config.initial_residual_scale,
        )
    )


def _run_arm(
    name: str,
    *,
    tokenizer,
    relation_batches: Sequence[LanguageBatch],
    general_batches: Sequence[Sequence[LanguageBatch]],
    eval_batches: Sequence[LanguageBatch],
    cases: Sequence[RelationCase],
    schedule: Sequence[tuple[str, int]],
    probes: Sequence[bool],
    config: SparseEventFalsificationConfig,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    model = _build_model(name, vocab_size=tokenizer.vocab_size, config=config).to(device)
    training_config = LanguageTrainingExperimentConfig(
        learning_rate=config.learning_rate,
        minimum_learning_rate_fraction=config.minimum_learning_rate_fraction,
        warmup_fraction=config.warmup_fraction,
        weight_decay=config.weight_decay,
        precision=config.precision,
    )
    optimizer, fused = _optimizer(model, training_config)
    total_steps = len(schedule)
    warmup_steps = int(round(total_steps * config.warmup_fraction))
    utility_targets: list[float] = []
    processed = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    model.train()
    started = time.perf_counter()
    for step, (kind, index) in enumerate(schedule):
        cpu_batch = (
            relation_batches[index]
            if kind == "relation"
            else general_batches[int(kind.rsplit("_", 1)[1])][index]
        )
        batch = cpu_batch.to(device)
        lr = _learning_rate(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            peak=config.learning_rate,
            minimum_fraction=config.minimum_learning_rate_fraction,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with _precision_context(device, config.precision):
            if name == "utility":
                result = model.next_token_loss(
                    batch.input_ids,
                    batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                    counterfactual_probe=probes[step],
                )
            else:
                result = model.next_token_loss(
                    batch.input_ids,
                    batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                )
            loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        processed += int(batch.target_ids.numel())
        counterfactual = dict(result.get("training_aux") or {}).get(
            "counterfactual", {}
        )
        if bool(counterfactual.get("ran")):
            utility_targets.append(float(counterfactual["mean_target"]))
        if (step + 1) % max(1, total_steps // 10) == 0:
            print(f"[sparse-v2] {name} {step + 1}/{total_steps}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    heldout = evaluate_language_model(model, eval_batches)
    relation = evaluate_relation_binding_cases_batched(
        model, tokenizer, cases, batch_size=config.relation_eval_batch_size
    )
    sample = general_batches[0][0].to(device)
    with torch.no_grad(), _precision_context(device, config.precision):
        telemetry = model(sample.input_ids)["telemetry"]
    parameters = sum(parameter.numel() for parameter in model.parameters())
    gradients = sum(
        parameter.numel() for parameter in model.parameters() if parameter.grad is not None
    )
    return {
        "name": name,
        "parameters": parameters,
        "parameters_with_final_gradient": gradients,
        "optimizer_fused": bool(fused),
        "processed_tokens": processed,
        "training_seconds": elapsed,
        "training_tokens_per_second": processed / max(elapsed, 1.0e-9),
        "peak_cuda_memory_bytes": peak,
        "heldout": heldout,
        "relation": relation,
        "utility": {
            "probe_count": len(utility_targets),
            "mean_target": (
                sum(utility_targets) / len(utility_targets)
                if utility_targets
                else 0.0
            ),
        },
        "telemetry": telemetry,
    }


def sparse_event_decision(
    arms: Sequence[Mapping[str, Any]], *, minimum_tokens: int = 16_777_216
) -> str:
    rows = {str(row["name"]): row for row in arms}
    exact, random, utility = rows["exact_only"], rows["random"], rows["utility"]
    processed = min(int(row.get("processed_tokens") or 0) for row in rows.values())
    if processed < int(minimum_tokens):
        return "incomplete_mechanism_smoke"
    utility_loss = float(utility["heldout"]["heldout_loss"])
    random_loss = float(random["heldout"]["heldout_loss"])
    exact_loss = float(exact["heldout"]["heldout_loss"])
    utility_free = float(utility["relation"]["generation_exact_accuracy"])
    random_free = float(random["relation"]["generation_exact_accuracy"])
    if utility_loss > exact_loss + 0.02:
        return "retire_v2_sidecar_harms_exact_stream"
    if utility_loss >= random_loss - 0.005 and utility_free < random_free + 0.02:
        return "retire_v2_utility_selector_not_better_than_random"
    return "continue_v2_to_64m_and_unseen_generation"


def run_sparse_event_falsification(
    *,
    tokenizer_checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: SparseEventFalsificationConfig = SparseEventFalsificationConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("Exactly two general train and two eval sources are required")
    resolved = _resolve_device(device)
    tokenizer = _load_tokenizer(Path(tokenizer_checkpoint_path))
    cases = _load_cases(Path(relation_cases_path))
    steps = math.ceil(config.token_budget / (config.batch_size * config.sequence_length))
    relation_steps = max(1, int(round(steps * config.relation_fraction)))
    general_steps = max(1, math.ceil((steps - relation_steps) / 2))
    relation_text, relation_selection = sample_corpus_ranges(
        relation_corpus_path,
        byte_budget=config.sample_bytes_per_train_source,
        range_count=config.sample_range_count,
    )
    train_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=config.sample_bytes_per_train_source,
            range_count=config.sample_range_count,
        )
        for path in general_train_paths
    ]
    eval_samples = [
        sample_corpus_ranges(
            path,
            byte_budget=config.sample_bytes_per_eval_source,
            range_count=config.sample_range_count,
        )
        for path in general_eval_paths
    ]
    relation_split = build_language_model_splits(
        [relation_text], tokenizer,
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        max_train_batches=relation_steps,
        max_eval_batches=1,
    )
    general_splits = [
        build_language_model_splits(
            [text], tokenizer,
            sequence_length=config.sequence_length,
            stride=config.sequence_length,
            batch_size=config.batch_size,
            max_train_batches=general_steps,
            max_eval_batches=1,
        )
        for text, _selection in train_samples
    ]
    eval_split = build_language_model_splits(
        [text for text, _selection in train_samples], tokenizer,
        eval_texts=[text for text, _selection in eval_samples],
        sequence_length=config.sequence_length,
        stride=config.sequence_length,
        batch_size=config.batch_size,
        max_train_batches=1,
        max_eval_batches=config.eval_batches,
    )
    schedule = build_matched_schedule(
        step_count=steps,
        relation_fraction=config.relation_fraction,
        relation_batch_count=len(relation_split.train),
        general_batch_counts=[len(split.train) for split in general_splits],
        seed=config.seed,
    )
    probes = build_probe_schedule(
        steps=steps, rate=config.counterfactual_rate, seed=config.seed + 91_337
    )
    arms = []
    for name in ("exact_only", "dense", "random", "utility"):
        print(f"[sparse-v2] starting {name}", flush=True)
        arms.append(_run_arm(
            name,
            tokenizer=tokenizer,
            relation_batches=relation_split.train,
            general_batches=[split.train for split in general_splits],
            eval_batches=eval_split.eval,
            cases=cases,
            schedule=schedule,
            probes=probes,
            config=config,
            device=resolved,
        ))
        print(
            f"[sparse-v2] completed {name}: loss "
            f"{arms[-1]['heldout']['heldout_loss']:.4f}, free "
            f"{arms[-1]['relation']['generation_exact_accuracy']:.3f}",
            flush=True,
        )
        if resolved.type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "tokenizer": {
            "checkpoint": str(tokenizer_checkpoint_path),
            "vocab_size": tokenizer.vocab_size,
            "hash": tokenizer.vocabulary_hash(),
        },
        "source_selections": {
            "relation": relation_selection,
            "general_train": [row for _text, row in train_samples],
            "general_eval": [row for _text, row in eval_samples],
        },
        "schedule": {
            "steps": steps,
            "processed_tokens": steps * config.batch_size * config.sequence_length,
            "relation_steps": sum(kind == "relation" for kind, _ in schedule),
            "probe_steps": sum(probes),
            "identical_for_all_arms": True,
        },
        "arms": arms,
        "decision": sparse_event_decision(arms),
        "quality_boundary": {
            "promotes_runtime_installation": False,
            "promotes_unseen_generation": False,
        },
    }
    write_json_report_with_readme(
        Path(output_path), report,
        title="MARULHO Sparse Event-Memory v2 Falsification",
    )
    print(f"[sparse-v2] decision {report['decision']}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-corpus", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--general-train", action="append", type=Path, required=True)
    parser.add_argument("--general-eval", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-budget", type=int, default=16_777_216)
    parser.add_argument("--train-sample-mib", type=int, default=64)
    parser.add_argument("--eval-sample-mib", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_sparse_event_falsification(
        tokenizer_checkpoint_path=args.tokenizer_checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        output_path=args.output,
        config=SparseEventFalsificationConfig(
            token_budget=args.token_budget,
            sample_bytes_per_train_source=max(1, args.train_sample_mib) * 1024 * 1024,
            sample_bytes_per_eval_source=max(1, args.eval_sample_mib) * 1024 * 1024,
        ),
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
