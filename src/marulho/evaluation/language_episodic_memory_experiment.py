"""Compare bounded episodic write policies on MARULHO relation binding."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_relation_binding_experiment import (
    KINDS,
    RelationCase,
    evaluate_relation_binding_cases,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_selective_episodic_relation_memory.v1"
ARTIFACT_KIND = "marulho_selective_episodic_relation_memory"
POLICIES = ("no_memory", "random", "recency", "surprise", "full", "oracle")

DISTRACTORS = (
    "The weather report mentioned a quiet afternoon.",
    "A nearby library opened at nine in the morning.",
    "The hallway clock made a soft sound every hour.",
    "Several students discussed a painting after lunch.",
    "A delivery truck stopped beside the old building.",
    "Someone watered the flowers near the front door.",
    "The local newspaper printed a story about music.",
    "A small lamp remained on beside the window.",
    "Two neighbors planned a walk for the weekend.",
    "The kitchen table had four empty chairs.",
    "A radio played quietly in another room.",
    "The park path curved around a group of trees.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _prompt_episodes_and_question(prompt: str) -> tuple[tuple[str, ...], str]:
    parts = tuple(
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+", str(prompt).strip())
        if part.strip()
    )
    if len(parts) < 2:
        raise ValueError("Episodic prompt requires evidence plus a question")
    if parts[-1].casefold().startswith("answer") and len(parts) >= 3:
        return parts[:-2], f"{parts[-2]} {parts[-1]}"
    return parts[:-1], parts[-1]


def _stressed_prompt(
    case: RelationCase,
    *,
    distractor_count: int,
    seed: int,
) -> tuple[tuple[str, ...], str, str]:
    evidence, question = _prompt_episodes_and_question(case.prompt)
    rng = random.Random(int(seed))
    distractors = rng.sample(
        DISTRACTORS,
        min(max(0, int(distractor_count)), len(DISTRACTORS)),
    )
    episodes = (*evidence, *distractors)
    return episodes, question, " ".join((*episodes, question))


@torch.no_grad()
def _episode_key(model, tokenizer, text: str) -> torch.Tensor:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    token_tensor = torch.tensor(ids, dtype=torch.long, device=model.device)
    key_ids = token_tensor[1:] if int(token_tensor.numel()) > 1 else token_tensor
    return F.normalize(model.token_embedding(key_ids).mean(dim=0), dim=0).detach()


@torch.no_grad()
def _episode_surprise(model, tokenizer, text: str) -> float:
    ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    token_tensor = torch.tensor(ids, dtype=torch.long, device=model.device)
    if len(ids) < 2:
        return 0.0
    output = model.forward(token_tensor[:-1].unsqueeze(0), collect_telemetry=False)
    targets = token_tensor[1:]
    loss = F.cross_entropy(
        output["logits"].reshape(-1, output["logits"].shape[-1]),
        targets,
    )
    return float(loss.detach().cpu().item())


@torch.no_grad()
def _query_key(model, tokenizer, question: str) -> torch.Tensor:
    ids = tokenizer.encode(question, add_bos=False, add_eos=False)
    token_tensor = torch.tensor(ids, dtype=torch.long, device=model.device)
    return F.normalize(model.token_embedding(token_tensor).mean(dim=0), dim=0)


def select_episode_indices(
    *,
    policy: str,
    surprise_scores: Sequence[float],
    oracle_overlap_scores: Sequence[int],
    slot_budget: int,
    seed: int,
) -> tuple[int, ...]:
    count = len(surprise_scores)
    budget = min(max(0, int(slot_budget)), count)
    if policy == "no_memory" or budget <= 0:
        return tuple()
    if policy == "full":
        return tuple(range(count))
    if policy == "random":
        return tuple(sorted(random.Random(int(seed)).sample(range(count), budget)))
    if policy == "recency":
        return tuple(range(count - budget, count))
    if policy == "surprise":
        ranked = sorted(
            range(count),
            key=lambda index: (-float(surprise_scores[index]), index),
        )
        return tuple(sorted(ranked[:budget]))
    if policy == "oracle":
        ranked = sorted(
            range(count),
            key=lambda index: (-int(oracle_overlap_scores[index]), index),
        )
        return tuple(sorted(ranked[:budget]))
    raise ValueError(f"Unknown episodic policy: {policy}")


def _content_terms(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
        "after", "now", "still", "some", "no", "has", "answer",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).casefold())
        if token not in stop
    }


def _memory_augmented_case(
    case: RelationCase,
    *,
    policy: str,
    model,
    tokenizer,
    slot_budget: int,
    read_budget: int,
    distractor_count: int,
    seed: int,
) -> tuple[RelationCase, dict[str, Any]]:
    started = time.perf_counter()
    episodes, question, stressed = _stressed_prompt(
        case,
        distractor_count=distractor_count,
        seed=seed,
    )
    if policy == "no_memory":
        return replace(case, prompt=stressed), {
            "case_id": case.case_id,
            "stored_indices": [],
            "retrieved_indices": [],
            "stored_bytes": 0,
            "write_count": 0,
            "read_count": 0,
            "selection_latency_seconds": time.perf_counter() - started,
            "label_used_for_memory_selection": False,
        }
    surprise_scores = (
        [_episode_surprise(model, tokenizer, episode) for episode in episodes]
        if policy == "surprise"
        else [0.0 for _episode in episodes]
    )
    answer_terms = _content_terms(case.candidates[int(case.correct_index)])
    overlap_scores = [
        len(_content_terms(episode) & answer_terms) for episode in episodes
    ]
    stored = select_episode_indices(
        policy=policy,
        surprise_scores=surprise_scores,
        oracle_overlap_scores=overlap_scores,
        slot_budget=slot_budget,
        seed=seed,
    )
    query = _query_key(model, tokenizer, question)
    stored_keys = {
        index: _episode_key(model, tokenizer, episodes[index]) for index in stored
    }
    ranked_reads = sorted(
        stored,
        key=lambda index: (
            -float(torch.dot(stored_keys[index], query).detach().cpu().item()),
            index,
        ),
    )
    retrieved = tuple(sorted(ranked_reads[: max(0, int(read_budget))]))
    memory_text = " ".join(episodes[index] for index in retrieved)
    augmented = f"Relevant earlier events: {memory_text} {question}"
    stored_bytes = sum(len(episodes[index].encode("utf-8")) for index in stored)
    return replace(case, prompt=augmented), {
        "case_id": case.case_id,
        "stored_indices": list(stored),
        "retrieved_indices": list(retrieved),
        "stored_bytes": int(stored_bytes),
        "write_count": len(stored),
        "read_count": len(retrieved),
        "surprise_scores": surprise_scores,
        "selection_latency_seconds": time.perf_counter() - started,
        "label_used_for_memory_selection": policy == "oracle",
    }


def run_episodic_memory_experiment(
    *,
    checkpoint_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    policies: Sequence[str] = POLICIES,
    slot_budget: int = 2,
    read_budget: int = 2,
    distractor_count: int = 8,
    seed: int = 20260710,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    cases_file = Path(cases_path)
    cases = _load_cases(cases_file)
    resolved_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if str(device) == "auto" else torch.device(device)
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model = model.to(resolved_device).eval()
    policy_reports: list[dict[str, Any]] = []
    for policy_index, policy in enumerate(policies):
        if policy not in POLICIES:
            raise ValueError(f"Unknown episodic policy: {policy}")
        augmented_cases: list[RelationCase] = []
        memory_rows: list[dict[str, Any]] = []
        policy_started = time.perf_counter()
        for case_index, case in enumerate(cases):
            augmented, memory = _memory_augmented_case(
                case,
                policy=policy,
                model=model,
                tokenizer=tokenizer,
                slot_budget=slot_budget,
                read_budget=read_budget,
                distractor_count=distractor_count,
                seed=int(seed) + policy_index * 100_000 + case_index,
            )
            augmented_cases.append(augmented)
            memory_rows.append(memory)
        evaluation = evaluate_relation_binding_cases(
            model,
            tokenizer,
            augmented_cases,
        )
        elapsed = time.perf_counter() - policy_started
        total_bytes = sum(int(row["stored_bytes"]) for row in memory_rows)
        total_writes = sum(int(row["write_count"]) for row in memory_rows)
        total_reads = sum(int(row["read_count"]) for row in memory_rows)
        policy_reports.append(
            {
                "policy": policy,
                "promotable_policy": policy not in {"full", "oracle"},
                "label_used_for_memory_selection": policy == "oracle",
                "evaluation": evaluation,
                "memory": {
                    "slot_budget": int(slot_budget),
                    "read_budget": int(read_budget),
                    "total_stored_bytes": total_bytes,
                    "mean_stored_bytes_per_case": total_bytes / max(1, len(cases)),
                    "max_stored_bytes_per_case": max(
                        (int(row["stored_bytes"]) for row in memory_rows),
                        default=0,
                    ),
                    "write_count": total_writes,
                    "read_count": total_reads,
                    "write_rate_per_case": total_writes / max(1, len(cases)),
                    "read_rate_per_case": total_reads / max(1, len(cases)),
                    "selection_latency_seconds": sum(
                        float(row["selection_latency_seconds"])
                        for row in memory_rows
                    ),
                    "rows": memory_rows,
                },
                "elapsed_seconds": elapsed,
                "cases_per_second": len(cases) / max(elapsed, 1.0e-9),
            }
        )
    by_policy = {row["policy"]: row for row in policy_reports}
    no_memory_free = float(
        by_policy.get("no_memory", {"evaluation": {}})["evaluation"].get(
            "generation_exact_accuracy", 0.0
        )
    )
    surprise_free = float(
        by_policy.get("surprise", {"evaluation": {}})["evaluation"].get(
            "generation_exact_accuracy", 0.0
        )
    )
    control_free = max(
        (
            float(by_policy[name]["evaluation"]["generation_exact_accuracy"])
            for name in ("random", "recency")
            if name in by_policy
        ),
        default=0.0,
    )
    if surprise_free >= no_memory_free + 0.15 and surprise_free >= control_free + 0.10:
        decision = "surprise_selected_memory_promising"
    elif surprise_free > no_memory_free:
        decision = "memory_helps_but_surprise_policy_not_distinct"
    else:
        decision = "prompt_level_episodic_memory_falsified"
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "cumulative_update_tokens": metadata.get("cumulative_update_tokens"),
            "cumulative_optimizer_steps": metadata.get(
                "cumulative_optimizer_steps"
            ),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
        },
        "cases": {
            "path": str(cases_file),
            "sha256": _sha256_file(cases_file),
            "case_count": len(cases),
            "kind_count": len(KINDS),
            "distractor_count": int(distractor_count),
        },
        "policies": policy_reports,
        "decision": decision,
        "quality_boundary": {
            "oracle_non_promotable": True,
            "full_store_non_promotable": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "requires_general_language_retention_pairing": True,
        },
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO Selective Episodic Memory Experiment",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--slot-budget", type=int, default=2)
    parser.add_argument("--read-budget", type=int, default=2)
    parser.add_argument("--distractors", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_episodic_memory_experiment(
        checkpoint_path=args.checkpoint,
        cases_path=args.cases,
        output_path=args.output,
        policies=tuple(args.policy) or POLICIES,
        slot_budget=max(0, int(args.slot_budget)),
        read_budget=max(0, int(args.read_budget)),
        distractor_count=max(0, int(args.distractors)),
        seed=int(args.seed),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
