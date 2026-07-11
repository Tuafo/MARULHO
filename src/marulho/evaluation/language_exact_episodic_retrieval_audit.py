"""Audit label-safe keys for bounded retrieval of exact source episodes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import sha256_file
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_exact_episodic_retrieval_audit.v1"
ARTIFACT_KIND = "marulho_exact_episodic_retrieval_audit"
CANDIDATE_POLICIES = ("lexical_tfidf", "frozen_last", "frozen_mean")
ADVANCE_DECISION = "advance_v20_exact_episodic_retrieval_to_language_screen"


@dataclass(frozen=True)
class RetrievalAuditConfig:
    facts_per_query: int = 4
    source_length: int = 48
    query_length: int = 40
    feature_batch_size: int = 64
    precision: str = "bfloat16"
    data_seed: int = 9501
    minimum_recall_at_1: float = 0.75
    minimum_recall_at_2: float = 0.90
    minimum_macro_query_recall_at_1: float = 0.70
    minimum_pair_both_targets_selected: float = 0.60
    minimum_recall_gain_over_temporal: float = 0.30


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    kind: str
    source: str
    query_prefix: str


@dataclass(frozen=True)
class EncodedTextBank:
    ids: torch.Tensor
    mask: torch.Tensor


def split_relation_case_prompt(prompt: str) -> tuple[str, str]:
    text = str(prompt).strip()
    if not text.endswith("Answer:"):
        raise ValueError("relation case prompt must end with Answer:")
    without_answer = text[: -len("Answer:")].rstrip()
    if ". " not in without_answer:
        raise ValueError("relation case prompt has no source/question boundary")
    source, question = without_answer.rsplit(". ", 1)
    source = f"{source.strip()}."
    question = question.strip()
    if not question.endswith("?"):
        raise ValueError("relation case query must end with a question mark")
    return source, f"Question: {question} Answer: "


def load_retrieval_cases(path: str | Path) -> tuple[RetrievalCase, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("relation case payload contains no cases")
    cases = []
    for row in rows:
        source, query_prefix = split_relation_case_prompt(str(row["prompt"]))
        cases.append(
            RetrievalCase(
                case_id=str(row["case_id"]),
                kind=str(row["kind"]),
                source=source,
                query_prefix=query_prefix,
            )
        )
    return tuple(cases)


def build_evaluation_groups(
    *,
    case_count: int,
    facts_per_query: int,
    seed: int,
    case_labels: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(case_labels) != int(case_count):
        raise ValueError("case_labels length must equal case_count")
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(case_labels):
        buckets.setdefault(str(label), []).append(index)
    if len(buckets) < int(facts_per_query):
        raise ValueError("not enough distinct query identities for retrieval groups")
    generator = random.Random(int(seed))
    plans: dict[str, tuple[list[int], int]] = {}
    for target_label in sorted(buckets):
        distractor_labels = generator.sample(
            [label for label in buckets if label != target_label],
            int(facts_per_query) - 1,
        )
        distractors = [generator.choice(buckets[label]) for label in distractor_labels]
        plans[target_label] = (
            distractors,
            generator.randrange(int(facts_per_query)),
        )
    rows = []
    slots = []
    for target in range(int(case_count)):
        distractors, target_slot = plans[str(case_labels[target])]
        row = list(distractors)
        row.insert(int(target_slot), target)
        rows.append(row)
        slots.append(int(target_slot))
    return torch.tensor(rows, dtype=torch.long), torch.tensor(slots, dtype=torch.long)


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def encode_text_bank(
    tokenizer,
    texts: Sequence[str],
    *,
    length: int,
    add_eos: bool,
) -> EncodedTextBank:
    ids = torch.full(
        (len(texts), int(length)), int(tokenizer.pad_id), dtype=torch.long
    )
    mask = torch.zeros((len(texts), int(length)), dtype=torch.bool)
    for index, text in enumerate(texts):
        encoded = tokenizer.encode(text, add_bos=True, add_eos=bool(add_eos))
        if len(encoded) > int(length):
            raise ValueError(
                f"encoded text length {len(encoded)} exceeds retrieval budget {length}"
            )
        ids[index, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
        mask[index, : len(encoded)] = True
    return EncodedTextBank(ids=ids, mask=mask)


def _precision_context(device: torch.device, precision: str):
    if device.type != "cuda":
        return nullcontext()
    if precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "float32":
        return nullcontext()
    raise ValueError("precision must be float32, float16, or bfloat16")


@torch.no_grad()
def frozen_feature_keys(
    model: MarulhoHashedMicroExpertLanguageModel,
    bank: EncodedTextBank,
    *,
    batch_size: int,
    precision: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    last_rows = []
    mean_rows = []
    was_training = model.training
    model.eval()
    try:
        for start in range(0, int(bank.ids.shape[0]), int(batch_size)):
            end = min(int(bank.ids.shape[0]), start + int(batch_size))
            ids = bank.ids[start:end].to(model.device)
            mask = bank.mask[start:end].to(model.device)
            with _precision_context(model.device, str(precision)):
                hidden = model._forward_hidden(
                    ids, collect_telemetry=False
                )["hidden"]
            lengths = mask.sum(dim=1).long()
            rows = torch.arange(end - start, device=model.device)
            last = hidden[rows, lengths - 1]
            mean = (hidden * mask.unsqueeze(-1)).sum(dim=1) / lengths.unsqueeze(1)
            last_rows.append(F.normalize(last.float(), dim=-1).cpu())
            mean_rows.append(F.normalize(mean.float(), dim=-1).cpu())
    finally:
        model.train(was_training)
    return torch.cat(last_rows), torch.cat(mean_rows)


def lexical_tfidf_scores(
    source_bank: EncodedTextBank,
    query_bank: EncodedTextBank,
    group_indices: torch.Tensor,
    *,
    excluded_token_ids: Sequence[int],
) -> torch.Tensor:
    excluded = {int(value) for value in excluded_token_ids}
    source_documents = []
    document_frequency: dict[int, int] = {}
    for ids, mask in zip(source_bank.ids, source_bank.mask):
        counts: dict[int, int] = {}
        for raw in ids[mask].tolist():
            token_id = int(raw)
            if token_id not in excluded:
                counts[token_id] = counts.get(token_id, 0) + 1
        source_documents.append(counts)
        for token_id in counts:
            document_frequency[token_id] = document_frequency.get(token_id, 0) + 1
    count = len(source_documents)
    idf = {
        token_id: math.log((count + 1.0) / (frequency + 1.0)) + 1.0
        for token_id, frequency in document_frequency.items()
    }

    def vector(ids: torch.Tensor, mask: torch.Tensor) -> dict[int, float]:
        counts: dict[int, int] = {}
        for raw in ids[mask].tolist():
            token_id = int(raw)
            if token_id not in excluded:
                counts[token_id] = counts.get(token_id, 0) + 1
        return {
            token_id: float(frequency) * idf.get(token_id, 1.0)
            for token_id, frequency in counts.items()
        }

    source_vectors = [
        {token_id: float(value) * idf[token_id] for token_id, value in counts.items()}
        for counts in source_documents
    ]
    source_norms = [
        math.sqrt(sum(value * value for value in values.values()))
        for values in source_vectors
    ]
    query_vectors = [
        vector(ids, mask) for ids, mask in zip(query_bank.ids, query_bank.mask)
    ]
    scores = torch.zeros(group_indices.shape, dtype=torch.float32)
    for query_index, group in enumerate(group_indices):
        query = query_vectors[query_index]
        query_norm = math.sqrt(sum(value * value for value in query.values()))
        for slot, source_index in enumerate(group.tolist()):
            source = source_vectors[int(source_index)]
            dot = sum(value * source.get(token_id, 0.0) for token_id, value in query.items())
            scores[query_index, slot] = dot / max(
                query_norm * source_norms[int(source_index)], 1.0e-12
            )
    return scores


def cosine_group_scores(
    source_keys: torch.Tensor,
    query_keys: torch.Tensor,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    grouped = source_keys.index_select(0, group_indices.reshape(-1)).reshape(
        int(group_indices.shape[0]), int(group_indices.shape[1]), -1
    )
    return torch.einsum("bfd,bd->bf", grouped, query_keys)


def rankings_from_scores(scores: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("retrieval scores must be [query,fact]")
    return torch.argsort(scores, dim=1, descending=True, stable=True)


def counterfactual_retrieval_metrics(
    *,
    cases: Sequence[RetrievalCase],
    group_indices: torch.Tensor,
    target_slots: torch.Tensor,
    rankings: torch.Tensor,
) -> dict[str, Any]:
    selected_slots = rankings[:, 0]
    selected_sources = group_indices.gather(1, selected_slots.unsqueeze(1)).squeeze(1)
    target_selected = selected_slots == target_slots
    grouped: dict[str, list[int]] = {}
    for index, case in enumerate(cases):
        grouped.setdefault(case.query_prefix, []).append(index)
    usable = [indices for indices in grouped.values() if len(indices) > 1]
    eligible = [index for indices in usable for index in indices]
    pairs = []
    for indices in usable:
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                pairs.append((left, right))
    return {
        "query_group_count": len(usable),
        "case_count": len(eligible),
        "different_target_pair_count": len(pairs),
        "target_following_recall_at_1": (
            sum(bool(target_selected[index]) for index in eligible)
            / max(1, len(eligible))
        ),
        "selected_source_change_rate_when_target_changes": (
            sum(
                int(selected_sources[left]) != int(selected_sources[right])
                for left, right in pairs
            )
            / max(1, len(pairs))
        ),
        "both_targets_selected_pair_rate": (
            sum(
                bool(target_selected[left]) and bool(target_selected[right])
                for left, right in pairs
            )
            / max(1, len(pairs))
        ),
        "identical_query_and_distractors_with_target_source_swap": True,
        "promotion_metric": True,
    }


def retrieval_metrics(
    *,
    name: str,
    scores: torch.Tensor,
    cases: Sequence[RetrievalCase],
    group_indices: torch.Tensor,
    target_slots: torch.Tensor,
    promotable: bool,
) -> dict[str, Any]:
    rankings = rankings_from_scores(scores)
    target_positions = (rankings == target_slots.unsqueeze(1)).nonzero()[:, 1]
    recall_at_1 = float((target_positions < 1).float().mean())
    recall_at_2 = float((target_positions < 2).float().mean())
    reciprocal_rank = float((1.0 / (target_positions.float() + 1.0)).mean())
    kind_rows: dict[str, list[bool]] = {}
    for case, position in zip(cases, target_positions.tolist()):
        kind_rows.setdefault(case.kind, []).append(int(position) == 0)
    query_groups: dict[str, list[bool]] = {}
    for case, position in zip(cases, target_positions.tolist()):
        query_groups.setdefault(case.query_prefix, []).append(int(position) == 0)
    return {
        "policy": name,
        "promotable": bool(promotable),
        "recall_at_1": recall_at_1,
        "recall_at_2": recall_at_2,
        "mean_reciprocal_rank": reciprocal_rank,
        "macro_query_recall_at_1": sum(
            sum(values) / len(values) for values in query_groups.values()
        )
        / len(query_groups),
        "kind_recall_at_1": {
            kind: sum(values) / len(values) for kind, values in kind_rows.items()
        },
        "mean_selected_score": float(scores.gather(1, rankings[:, :1]).mean()),
        "counterfactual": counterfactual_retrieval_metrics(
            cases=cases,
            group_indices=group_indices,
            target_slots=target_slots,
            rankings=rankings,
        ),
        "rankings_use_target_index": False,
        "target_index_metrics_only": True,
    }


def retrieval_audit_decision(
    policies: Mapping[str, Mapping[str, Any]],
    *,
    config: RetrievalAuditConfig,
) -> tuple[str, str | None]:
    missing = [name for name in CANDIDATE_POLICIES if name not in policies]
    if missing:
        return "incomplete_v20_missing_candidate_policy", None
    temporal = max(
        float(policies[name]["recall_at_1"]) for name in ("random", "recency")
    )
    ordered = sorted(
        CANDIDATE_POLICIES,
        key=lambda name: (
            float(policies[name]["recall_at_1"]),
            float(policies[name]["mean_reciprocal_rank"]),
        ),
        reverse=True,
    )
    for name in ordered:
        row = policies[name]
        paired = row["counterfactual"]
        if (
            float(row["recall_at_1"]) >= float(config.minimum_recall_at_1)
            and float(row["recall_at_2"]) >= float(config.minimum_recall_at_2)
            and float(row["macro_query_recall_at_1"])
            >= float(config.minimum_macro_query_recall_at_1)
            and float(paired["both_targets_selected_pair_rate"])
            >= float(config.minimum_pair_both_targets_selected)
            and float(row["recall_at_1"]) - temporal
            >= float(config.minimum_recall_gain_over_temporal)
        ):
            return ADVANCE_DECISION, name
    return "redesign_v20_no_fixed_key_retrieves_exact_episode", None


def run_exact_episodic_retrieval_audit(
    *,
    parent_checkpoint_path: str | Path,
    relation_cases_path: str | Path,
    output_path: str | Path,
    config: RetrievalAuditConfig = RetrievalAuditConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.facts_per_query) < 2:
        raise ValueError("V20 facts_per_query must be at least two")
    if int(config.source_length) < 2 or int(config.query_length) < 2:
        raise ValueError("V20 source and query lengths must be at least two")
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V20 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    cases_path = Path(relation_cases_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V20 requires the one-billion-token V11 parent")
    if max(int(config.source_length), int(config.query_length)) > int(
        model.hashed_config.context_length
    ):
        raise ValueError("V20 encoded source/query exceeds the parent context")
    cases = load_retrieval_cases(cases_path)
    groups, target_slots = build_evaluation_groups(
        case_count=len(cases),
        facts_per_query=int(config.facts_per_query),
        seed=int(config.data_seed),
        case_labels=[case.query_prefix for case in cases],
    )
    if any(
        int(groups[index, target_slots[index]]) != index
        for index in range(len(cases))
    ):
        raise RuntimeError("V20 evaluation group lost its target source")
    source_bank = encode_text_bank(
        tokenizer,
        [case.source for case in cases],
        length=int(config.source_length),
        add_eos=True,
    )
    query_bank = encode_text_bank(
        tokenizer,
        [case.query_prefix for case in cases],
        length=int(config.query_length),
        add_eos=False,
    )
    excluded = (
        tokenizer.pad_id,
        tokenizer.bos_id,
        tokenizer.eos_id,
        tokenizer.unk_id,
        tokenizer.checkpoint_id,
        tokenizer.replay_id,
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        groups,
        excluded_token_ids=excluded,
    )
    model = model.to(resolved).eval()
    print("[exact-episodic-v20] extracting frozen source/query keys", flush=True)
    source_last, source_mean = frozen_feature_keys(
        model,
        source_bank,
        batch_size=int(config.feature_batch_size),
        precision=str(config.precision),
    )
    query_last, query_mean = frozen_feature_keys(
        model,
        query_bank,
        batch_size=int(config.feature_batch_size),
        precision=str(config.precision),
    )
    frozen_last_scores = cosine_group_scores(source_last, query_last, groups)
    frozen_mean_scores = cosine_group_scores(source_mean, query_mean, groups)
    generator = torch.Generator(device="cpu").manual_seed(int(config.data_seed) + 1)
    random_scores = torch.rand(groups.shape, generator=generator)
    recency_scores = torch.arange(
        int(config.facts_per_query), dtype=torch.float32
    ).unsqueeze(0).expand(len(cases), -1)
    oracle_scores = torch.zeros(groups.shape, dtype=torch.float32)
    oracle_scores.scatter_(1, target_slots.unsqueeze(1), 1.0)
    score_rows = {
        "random": random_scores,
        "recency": recency_scores,
        "lexical_tfidf": lexical_scores,
        "frozen_last": frozen_last_scores,
        "frozen_mean": frozen_mean_scores,
        "oracle": oracle_scores,
    }
    policies = {
        name: {
            **retrieval_metrics(
                name=name,
                scores=scores,
                cases=cases,
                group_indices=groups,
                target_slots=target_slots,
                promotable=name in CANDIDATE_POLICIES,
            ),
            "score_sha256": _tensor_sha256(scores),
            "active_source_count_at_1": 1,
            "active_source_tokens_at_1": int(config.source_length),
            "active_source_count_at_2": 2,
            "active_source_tokens_at_2": 2 * int(config.source_length),
        }
        for name, scores in score_rows.items()
    }
    policies["oracle"]["rankings_use_target_index"] = True
    policies["oracle"]["target_index_metrics_only"] = True
    policies["oracle"]["promotable"] = False
    decision, selected_policy = retrieval_audit_decision(policies, config=config)
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "configuration": asdict(config),
        "parent": {
            "path": str(parent_path),
            "sha256": sha256_file(parent_path),
            "processed_tokens": parent_tokens,
            "decision": parent_metadata.get("decision"),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
            "parameters_frozen": True,
            "parameter_gradients_enabled": False,
        },
        "source": {
            "relation_cases_path": str(cases_path),
            "relation_cases_sha256": sha256_file(cases_path),
            "case_count": len(cases),
            "distinct_query_identities": len(
                {case.query_prefix for case in cases}
            ),
            "source_tokens_are_stored_exactly": True,
            "source_write_sees_question": False,
            "source_write_sees_answer": False,
            "source_write_sees_candidates": False,
        },
        "groups": {
            "sha256": _tensor_sha256(groups, target_slots),
            "facts_per_query": int(config.facts_per_query),
            "target_slot_metrics_only": True,
            "paired_cases_hold_query_distractors_and_positions_fixed": True,
        },
        "anti_cheat": {
            "write_input": "source_tokens_only",
            "read_input": "question_prefix_without_answer",
            "candidate_policies_use_answer": False,
            "candidate_policies_use_candidates": False,
            "candidate_policies_use_target_slot": False,
            "target_slot_metrics_only": True,
            "oracle_uses_target_slot": True,
            "oracle_promotable": False,
        },
        "key_interfaces": {
            "lexical_tfidf": "checkpoint_bpe_counts_with_corpus_idf",
            "frozen_last": "cosine_of_final_causal_v11_states",
            "frozen_mean": "cosine_of_masked_mean_v11_states",
            "stable_tie_break": "earlier_source_position",
            "learned_selector": False,
        },
        "policies": policies,
        "all_history_upper_bound": {
            "target_inclusion": 1.0,
            "active_source_count": int(config.facts_per_query),
            "active_source_tokens": (
                int(config.facts_per_query) * int(config.source_length)
            ),
            "promotable_selector": False,
        },
        "decision": decision,
        "selected_policy": selected_policy,
        "promotion_boundary": {
            "advance_to_language_screen": decision == ADVANCE_DECISION,
            "retrieval_is_language_quality": False,
            "base_quality_promoted": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
        },
        "hardware": {
            "device": str(resolved),
            "cuda_device_name": (
                torch.cuda.get_device_name(resolved)
                if resolved.type == "cuda"
                else None
            ),
            "torch_version": torch.__version__,
        },
        "experiment_wall_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V20 Exact Episodic Retrieval Audit",
    )
    print(
        f"[exact-episodic-v20] decision {decision} selected={selected_policy}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--relation-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--facts-per-query", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--data-seed", type=int, default=9501)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = RetrievalAuditConfig(
        facts_per_query=int(args.facts_per_query),
        feature_batch_size=int(args.feature_batch_size),
        data_seed=int(args.data_seed),
        precision=str(args.precision),
    )
    run_exact_episodic_retrieval_audit(
        parent_checkpoint_path=args.parent_checkpoint,
        relation_cases_path=args.relation_cases,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
