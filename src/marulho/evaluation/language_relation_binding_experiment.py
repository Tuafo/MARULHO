"""Falsify entity/event relation binding from a MARULHO language checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import LANGUAGE_DOCUMENT_SEPARATOR
from marulho.evaluation.language_scaling_experiment import (
    LanguageScalingExperimentConfig,
    ScalingArmConfig,
    run_language_scaling_experiment,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_language_relation_binding_falsification.v1"
ARTIFACT_KIND = "marulho_language_relation_binding_falsification"

ENTITIES = ("Ava", "Ben", "Cora", "Dax", "Eli", "Mara", "Nora", "Owen")
COLORS = ("red", "blue", "green", "yellow", "silver", "orange")
OBJECTS = ("coin", "key", "notebook", "token", "ring", "card")
CONTAINERS = ("cup", "box", "basket", "drawer", "jar", "case")
PLACES = ("shelf", "desk", "garden", "studio", "hall", "porch")
KINDS = ("container", "ownership", "property", "event_order")


@dataclass(frozen=True)
class RelationCase:
    case_id: str
    kind: str
    signature: str
    prompt: str
    candidates: tuple[str, ...]
    correct_index: int


def _heldout_signature(signature: str) -> bool:
    digest = hashlib.sha256(str(signature).encode("utf-8")).digest()
    return int(digest[0]) % 5 == 0


def _signature(kind: str, *parts: str) -> str:
    return "|".join((str(kind), *(str(part) for part in parts)))


def _shuffled_candidates(
    correct: str,
    distractors: Sequence[str],
    *,
    seed: int,
) -> tuple[tuple[str, ...], int]:
    values = [str(correct), *(str(value) for value in distractors)]
    random.Random(int(seed)).shuffle(values)
    return tuple(values), int(values.index(str(correct)))


def _relation_example(
    kind: str,
    values: Sequence[str],
    *,
    evaluation_template: bool,
    seed: int,
) -> tuple[str, str, tuple[str, ...], int]:
    if kind == "container":
        entity, color, item, container, place = values
        signature = _signature(kind, entity, color, item, container, place)
        answer = f"The {color} {item} remains inside the {container}."
        distractors = [
            f"The {color} {item} is inside the {other}."
            for other in CONTAINERS
            if other != container
        ][:3]
        if evaluation_template:
            prompt = (
                f"{entity} put a {color} {item} in a {container}. Later, "
                f"{entity} carried the {container} to the {place}. Where is "
                f"the {color} {item} now? Answer:"
            )
        else:
            prompt = (
                f"{entity} placed the {color} {item} inside the {container}. "
                f"Then {entity} moved the {container} to the {place}. "
                f"Question: What contains the {color} {item} now? Answer:"
            )
    elif kind == "ownership":
        giver, receiver, color, item, place = values
        signature = _signature(kind, giver, receiver, color, item, place)
        answer = f"{receiver} has the {color} {item}."
        distractors = [
            f"{entity} has the {color} {item}."
            for entity in ENTITIES
            if entity != receiver
        ][:3]
        if evaluation_template:
            prompt = (
                f"{giver} handed a {color} {item} to {receiver}. Afterward, "
                f"{receiver} walked to the {place}. Who possesses the "
                f"{color} {item}? Answer:"
            )
        else:
            prompt = (
                f"{giver} gave the {color} {item} to {receiver}. {receiver} "
                f"then went to the {place}. Question: Who has the {color} "
                f"{item} now? Answer:"
            )
    elif kind == "property":
        entity, color, item, container, place = values
        signature = _signature(kind, entity, color, item, container, place)
        answer = f"The {item} is still {color}."
        distractors = [
            f"The {item} is now {other}." for other in COLORS if other != color
        ][:3]
        if evaluation_template:
            prompt = (
                f"{entity} stored a {color} {item} in a {container} and moved "
                f"the {container} to the {place}. What color is the {item} "
                "after the move? Answer:"
            )
        else:
            prompt = (
                f"A {color} {item} was inside {entity}'s {container}. "
                f"{entity} carried the {container} to the {place}. Question: "
                f"Did the {item}'s color change? Answer:"
            )
    elif kind == "event_order":
        first_event, place, entity, item = values
        signature = _signature(kind, first_event, place, entity, item)
        reaches = first_event == "pump_started"
        answer = (
            f"Some water reaches the {place}."
            if reaches
            else f"No water reaches the {place}."
        )
        distractors = [
            f"No water reaches the {place}."
            if reaches
            else f"Some water reaches the {place}.",
            "The pump remains off.",
            "The valve stays open.",
        ]
        if first_event == "pump_started":
            events = "The pump started. After that, the valve closed."
        else:
            events = "The valve closed. After that, the pump started."
        if evaluation_template:
            prompt = (
                f"Water can enter the {place} only while the pump runs and "
                f"the valve is open. During {entity}'s {item} test, {events} "
                f"Did any water enter the "
                f"{place}? Answer:"
            )
        else:
            prompt = (
                f"Rule: the pump sends water to the {place} only when the "
                f"valve is open. {entity} runs a {item} test. {events} "
                f"Question: Does water reach the "
                f"{place}? Answer:"
            )
    else:  # pragma: no cover
        raise ValueError(f"Unknown relation kind: {kind}")
    candidates, correct_index = _shuffled_candidates(
        answer,
        distractors,
        seed=seed,
    )
    return signature, prompt, candidates, correct_index


def _random_values(kind: str, rng: random.Random) -> tuple[str, ...]:
    if kind == "ownership":
        giver, receiver = rng.sample(ENTITIES, 2)
        return giver, receiver, rng.choice(COLORS), rng.choice(OBJECTS), rng.choice(PLACES)
    if kind == "event_order":
        return (
            rng.choice(("pump_started", "valve_closed")),
            rng.choice(PLACES),
            rng.choice(ENTITIES),
            rng.choice(OBJECTS),
        )
    return (
        rng.choice(ENTITIES),
        rng.choice(COLORS),
        rng.choice(OBJECTS),
        rng.choice(CONTAINERS),
        rng.choice(PLACES),
    )


def materialize_relation_binding_benchmark(
    *,
    corpus_path: str | Path,
    cases_path: str | Path,
    train_document_count: int = 200_000,
    eval_cases_per_kind: int = 64,
    seed: int = 20260710,
) -> tuple[Path, Path, tuple[RelationCase, ...]]:
    corpus = Path(corpus_path)
    cases_output = Path(cases_path)
    corpus.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    documents: list[str] = []
    kind_counts: Counter[str] = Counter()
    while len(documents) < max(1, int(train_document_count)):
        kind = KINDS[len(documents) % len(KINDS)]
        values = _random_values(kind, rng)
        signature, prompt, candidates, correct_index = _relation_example(
            kind,
            values,
            evaluation_template=False,
            seed=rng.randrange(2**31),
        )
        if _heldout_signature(signature):
            continue
        documents.append(f"{prompt} {candidates[correct_index]}")
        kind_counts[kind] += 1
    header = (
        "### source=marulho/procedural-relation-binding split=train "
        "role=relation_binding_falsification"
    )
    corpus.write_text(
        header
        + "".join(
            f"{LANGUAGE_DOCUMENT_SEPARATOR}{document}" for document in documents
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    cases: list[RelationCase] = []
    seen: set[str] = set()
    eval_rng = random.Random(int(seed) + 1)
    for kind in KINDS:
        attempts = 0
        while sum(case.kind == kind for case in cases) < int(eval_cases_per_kind):
            attempts += 1
            if attempts > 1_000_000:  # pragma: no cover
                raise RuntimeError(f"Unable to construct heldout cases for {kind}")
            values = _random_values(kind, eval_rng)
            signature, prompt, candidates, correct_index = _relation_example(
                kind,
                values,
                evaluation_template=True,
                seed=eval_rng.randrange(2**31),
            )
            if not _heldout_signature(signature) or signature in seen:
                continue
            seen.add(signature)
            cases.append(
                RelationCase(
                    case_id=f"{kind}-{sum(case.kind == kind for case in cases):04d}",
                    kind=kind,
                    signature=signature,
                    prompt=prompt,
                    candidates=candidates,
                    correct_index=correct_index,
                )
            )
    payload = {
        "surface": "marulho_relation_binding_benchmark.v1",
        "seed": int(seed),
        "train_document_count": len(documents),
        "train_kind_counts": dict(kind_counts),
        "split_policy": "sha256_signature_mod5_holdout",
        "evaluation_template_disjoint": True,
        "prediction_label_policy": "correct_index_metrics_only",
        "cases": [asdict(case) for case in cases],
    }
    cases_output.parent.mkdir(parents=True, exist_ok=True)
    cases_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return corpus, cases_output, tuple(cases)


@torch.no_grad()
def evaluate_relation_binding_cases(
    model,
    tokenizer,
    cases: Sequence[RelationCase],
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for case in cases:
        prompt_ids = tokenizer.encode(case.prompt, add_bos=True, add_eos=False)
        candidate_losses: list[float] = []
        for candidate in case.candidates:
            candidate_ids = tokenizer.encode(
                f" {candidate}",
                add_bos=False,
                add_eos=True,
            )
            combined = prompt_ids + candidate_ids
            if len(combined) > int(model.config.transformer_context_length):
                raise ValueError("Relation case exceeds checkpoint context length")
            input_ids = torch.tensor(
                [combined[:-1]],
                dtype=torch.long,
                device=model.device,
            )
            targets = torch.tensor(
                [combined[1:]],
                dtype=torch.long,
                device=model.device,
            )
            logits = model.forward(input_ids, collect_telemetry=False)["logits"]
            start = len(prompt_ids) - 1
            candidate_logits = logits[:, start:, :].reshape(-1, logits.shape[-1])
            candidate_targets = targets[:, start:].reshape(-1)
            loss = F.cross_entropy(candidate_logits, candidate_targets)
            candidate_losses.append(float(loss.detach().cpu().item()))
        predicted_index = min(
            range(len(candidate_losses)),
            key=candidate_losses.__getitem__,
        )
        rows.append(
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "candidate_losses": candidate_losses,
                "predicted_index": int(predicted_index),
                "correct": bool(predicted_index == int(case.correct_index)),
                "label_used_for_prediction": False,
            }
        )
    kind_accuracy = {
        kind: sum(row["correct"] for row in rows if row["kind"] == kind)
        / max(1, sum(row["kind"] == kind for row in rows))
        for kind in KINDS
    }
    return {
        "surface": "marulho_relation_binding_evaluation.v1",
        "case_count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / max(1, len(rows)),
        "kind_accuracy": kind_accuracy,
        "prediction_uses_correct_index": False,
        "correct_index_metrics_only": True,
        "rows": rows,
    }


def relation_binding_branch_decision(
    *,
    accuracy_before: float,
    accuracy_after: float,
    general_loss_delta: float,
) -> str:
    relation_gain = float(accuracy_after) - float(accuracy_before)
    if relation_gain >= 0.25 and float(general_loss_delta) > 0.15:
        return "relation_learned_but_catastrophic_forgetting_test_replay"
    if float(accuracy_after) >= 0.80 and float(general_loss_delta) <= 0.15:
        return "transformer_learns_relations_redesign_curriculum"
    if relation_gain >= 0.25:
        return "extend_relation_curriculum_before_memory"
    return "test_episodic_binding_or_larger_capacity"


def run_relation_binding_falsification(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    corpus_path: str | Path,
    cases_path: str | Path,
    general_eval_corpus_paths: Sequence[str | Path],
    train_document_count: int = 200_000,
    eval_cases_per_kind: int = 64,
    token_budgets: Sequence[int] = (4_194_304, 8_388_608, 16_777_216),
    seed: int = 20260710,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    corpus, cases_output, cases = materialize_relation_binding_benchmark(
        corpus_path=corpus_path,
        cases_path=cases_path,
        train_document_count=int(train_document_count),
        eval_cases_per_kind=int(eval_cases_per_kind),
        seed=int(seed),
    )
    resolved_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if str(device) == "auto" else torch.device(device)
    base_model, tokenizer, _metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model_config = base_model.config
    base_model = base_model.to(resolved_device)
    before = evaluate_relation_binding_cases(base_model, tokenizer, cases)
    del base_model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    training_output = output.with_name(f"{output.stem}-training.json")
    sequence_length = 256
    batch_size = 16
    maximum_budget = max(int(value) for value in token_budgets)
    max_train_batches = math.ceil(maximum_budget / (sequence_length * batch_size)) + 128
    scaling_report = run_language_scaling_experiment(
        output_path=training_output,
        corpus_paths=(corpus,),
        eval_corpus_paths=tuple(general_eval_corpus_paths),
        prompts=tuple(case.prompt for case in cases[:8]),
        config=LanguageScalingExperimentConfig(
            tokenizer_vocab_size=int(tokenizer.vocab_size),
            sequence_length=sequence_length,
            stride=sequence_length,
            batch_size=batch_size,
            max_train_batches=max_train_batches,
            max_eval_batches=256,
            token_budgets=tuple(int(value) for value in token_budgets),
            arms=(
                ScalingArmConfig(
                    "relation-branch",
                    width=int(model_config.embedding_dim),
                    layers=int(model_config.state_layers),
                    heads=int(model_config.attention_heads),
                    mlp_ratio=float(model_config.transformer_mlp_ratio),
                ),
            ),
            transformer_context_length=int(
                model_config.transformer_context_length
            ),
            learning_rate=1.0e-4,
            precision="bfloat16" if resolved_device.type == "cuda" else "float32",
            generation_tokens=64,
            seed=int(seed),
            device=str(resolved_device),
            resume_checkpoint_path=str(checkpoint),
        ),
    )
    candidate_checkpoint = Path(
        scaling_report["selection"]["selected_checkpoint"]
    )
    candidate_model, candidate_tokenizer, _candidate_metadata = (
        load_language_model_checkpoint(candidate_checkpoint, map_location="cpu")
    )
    candidate_model = candidate_model.to(resolved_device)
    after = evaluate_relation_binding_cases(
        candidate_model,
        candidate_tokenizer,
        cases,
    )
    del candidate_model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    arm = scaling_report["arms"][0]
    general_loss_before = float(arm["eval_before"]["heldout_loss"])
    general_loss_after = float(arm["final_heldout_loss"])
    relation_gain = float(after["accuracy"]) - float(before["accuracy"])
    general_loss_delta = general_loss_after - general_loss_before
    decision = relation_binding_branch_decision(
        accuracy_before=float(before["accuracy"]),
        accuracy_after=float(after["accuracy"]),
        general_loss_delta=general_loss_delta,
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "checkpoint_path": str(checkpoint),
        "candidate_checkpoint_path": str(candidate_checkpoint),
        "benchmark_corpus_path": str(corpus),
        "benchmark_cases_path": str(cases_output),
        "training_report_path": str(training_output),
        "benchmark": {
            "train_document_count": int(train_document_count),
            "eval_cases_per_kind": int(eval_cases_per_kind),
            "case_count": len(cases),
            "split_policy": "sha256_signature_mod5_holdout",
            "evaluation_template_disjoint": True,
            "correct_index_metrics_only": True,
        },
        "relation_before": before,
        "relation_after": after,
        "relation_accuracy_gain": relation_gain,
        "general_holdout": {
            "source_count": len(general_eval_corpus_paths),
            "loss_before": general_loss_before,
            "loss_after": general_loss_after,
            "loss_delta": general_loss_delta,
        },
        "decision": decision,
        "quality_boundary": {
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "candidate_promotable": False,
            "reason": "controlled_relation_falsification_not_base_language_qualification",
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Relation-Binding Falsification",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, required=True)
    parser.add_argument("--cases-output", type=Path, required=True)
    parser.add_argument("--general-eval-corpus", action="append", type=Path, required=True)
    parser.add_argument("--train-documents", type=int, default=200_000)
    parser.add_argument("--eval-cases-per-kind", type=int, default=64)
    parser.add_argument("--token-budget", action="append", type=int, default=[])
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_relation_binding_falsification(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        corpus_path=args.corpus_output,
        cases_path=args.cases_output,
        general_eval_corpus_paths=tuple(args.general_eval_corpus),
        train_document_count=max(1, int(args.train_documents)),
        eval_cases_per_kind=max(1, int(args.eval_cases_per_kind)),
        token_budgets=tuple(args.token_budget) or (4_194_304, 8_388_608, 16_777_216),
        seed=int(args.seed),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
