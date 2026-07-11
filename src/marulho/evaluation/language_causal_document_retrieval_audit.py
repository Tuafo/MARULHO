"""Audit causal exact-episode retrieval on disjoint general-language documents."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import iter_language_corpus_documents
from marulho.evaluation.language_exact_episodic_retrieval_audit import (
    EncodedTextBank,
    frozen_feature_keys,
    lexical_tfidf_scores,
    rankings_from_scores,
)
from marulho.evaluation.language_hashed_micro_expert_continuation import (
    _validate_parent,
)
from marulho.evaluation.language_matched_support import (
    sample_corpus_ranges,
    sha256_file,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_hashed_micro_experts import (
    MarulhoHashedMicroExpertLanguageModel,
    load_hashed_micro_expert_checkpoint,
)


SURFACE = "marulho_causal_document_retrieval_audit.v1"
ARTIFACT_KIND = "marulho_causal_document_retrieval_audit"
ARM_NAMES = (
    "off",
    "all4",
    "random1",
    "random2",
    "recency1",
    "recency2",
    "lexical1",
    "lexical2",
    "frozen_last1",
    "frozen_last2",
    "frozen_mean1",
    "frozen_mean2",
    "oracle1",
)
CANDIDATE_ARMS = (
    "lexical1",
    "lexical2",
    "frozen_last1",
    "frozen_last2",
    "frozen_mean1",
    "frozen_mean2",
)
ADVANCE_DECISION = "advance_v22_causal_document_retrieval_to_joint_training"


@dataclass(frozen=True)
class CausalDocumentRetrievalConfig:
    case_count_per_source: int = 128
    facts_per_query: int = 4
    source_length: int = 48
    prefix_length: int = 48
    target_length: int = 16
    minimum_gap_tokens: int = 48
    maximum_gap_tokens: int = 192
    eval_batch_size: int = 16
    feature_batch_size: int = 64
    sample_bytes: int = 8 * 1024 * 1024
    sample_range_count: int = 8
    precision: str = "bfloat16"
    data_seed: int = 9701
    bootstrap_samples: int = 4096
    minimum_oracle_loss_gain: float = 0.005
    minimum_candidate_loss_gain: float = 0.005
    minimum_control_loss_gain: float = 0.0025
    minimum_target_recall_at_1: float = 0.50
    minimum_target_recall_at_2: float = 0.70
    minimum_recall_gain_over_control: float = 0.20
    maximum_source_loss_regression: float = 0.0
    maximum_regret_to_all_history: float = 0.01


@dataclass(frozen=True)
class DocumentContinuationCase:
    case_id: str
    source_index: int
    source_name: str
    document_sha256: str
    document_token_count: int
    source_start: int
    source_end: int
    prefix_start: int
    prefix_end: int
    target_start: int
    target_end: int
    source_ids: tuple[int, ...]
    prefix_ids: tuple[int, ...]
    target_ids: tuple[int, ...]


@dataclass(frozen=True)
class EncodedDocumentContinuations:
    source_ids: torch.Tensor
    prefix_ids: torch.Tensor
    query_input_ids: torch.Tensor
    query_target_ids: torch.Tensor
    query_loss_mask: torch.Tensor

    @property
    def case_count(self) -> int:
        return int(self.source_ids.shape[0])


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


def _tensor_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().contiguous().cpu()
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _special_token_ids(tokenizer) -> tuple[int, ...]:
    return (
        int(tokenizer.pad_id),
        int(tokenizer.bos_id),
        int(tokenizer.eos_id),
        int(tokenizer.unk_id),
        int(tokenizer.checkpoint_id),
        int(tokenizer.replay_id),
    )


def build_document_cases(
    tokenizer,
    text: str,
    *,
    source_index: int,
    source_name: str,
    config: CausalDocumentRetrievalConfig,
    seed: int,
) -> tuple[tuple[DocumentContinuationCase, ...], dict[str, Any]]:
    """Reservoir-sample long documents and cut strictly ordered spans."""

    if int(config.case_count_per_source) < 1:
        raise ValueError("case_count_per_source must be positive")
    if int(config.minimum_gap_tokens) < 0:
        raise ValueError("minimum_gap_tokens cannot be negative")
    if int(config.maximum_gap_tokens) < int(config.minimum_gap_tokens):
        raise ValueError("maximum_gap_tokens must be at least minimum_gap_tokens")
    documents = tuple(iter_language_corpus_documents([text]))
    generator = random.Random(int(seed))
    reservoir: list[tuple[str, list[int], int]] = []
    eligible_count = 0
    duplicate_count = 0
    seen_hashes: set[str] = set()
    encode_batch_size = 512
    for start in range(0, len(documents), encode_batch_size):
        chunk = documents[start : start + encode_batch_size]
        encoded = tokenizer.encode_batch(chunk, add_bos=True, add_eos=True)
        for document, token_ids in zip(chunk, encoded):
            document_hash = hashlib.sha256(document.encode("utf-8")).hexdigest()
            if document_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(document_hash)
            minimum_query_start = (
                int(config.source_length) + int(config.minimum_gap_tokens)
            )
            latest_query_start = (
                len(token_ids)
                - int(config.prefix_length)
                - int(config.target_length)
            )
            if latest_query_start < minimum_query_start:
                continue
            eligible_count += 1
            maximum_query_start = min(
                latest_query_start,
                int(config.source_length) + int(config.maximum_gap_tokens),
            )
            query_start = generator.randint(minimum_query_start, maximum_query_start)
            candidate = (document_hash, token_ids, query_start)
            if len(reservoir) < int(config.case_count_per_source):
                reservoir.append(candidate)
                continue
            replacement = generator.randrange(eligible_count)
            if replacement < int(config.case_count_per_source):
                reservoir[replacement] = candidate
    if len(reservoir) < int(config.case_count_per_source):
        raise ValueError(
            f"{source_name} has only {len(reservoir)} eligible unique documents; "
            f"requested {config.case_count_per_source}"
        )
    generator.shuffle(reservoir)
    cases = []
    for local_index, (document_hash, token_ids, query_start) in enumerate(reservoir):
        source_end = int(config.source_length)
        prefix_end = query_start + int(config.prefix_length)
        target_end = prefix_end + int(config.target_length)
        cases.append(
            DocumentContinuationCase(
                case_id=f"{source_index}-{local_index:04d}",
                source_index=int(source_index),
                source_name=str(source_name),
                document_sha256=document_hash,
                document_token_count=len(token_ids),
                source_start=0,
                source_end=source_end,
                prefix_start=query_start,
                prefix_end=prefix_end,
                target_start=prefix_end,
                target_end=target_end,
                source_ids=tuple(int(value) for value in token_ids[:source_end]),
                prefix_ids=tuple(
                    int(value) for value in token_ids[query_start:prefix_end]
                ),
                target_ids=tuple(
                    int(value) for value in token_ids[prefix_end:target_end]
                ),
            )
        )
    return tuple(cases), {
        "source_index": int(source_index),
        "source_name": str(source_name),
        "parsed_document_count": len(documents),
        "unique_document_count": len(seen_hashes),
        "duplicate_document_count": duplicate_count,
        "eligible_document_count": eligible_count,
        "selected_case_count": len(cases),
        "selection_seed": int(seed),
    }


def encode_document_cases(
    cases: Sequence[DocumentContinuationCase],
    *,
    config: CausalDocumentRetrievalConfig,
) -> EncodedDocumentContinuations:
    if not cases:
        raise ValueError("document continuation cases cannot be empty")
    source_ids = torch.tensor([case.source_ids for case in cases], dtype=torch.long)
    prefix_ids = torch.tensor([case.prefix_ids for case in cases], dtype=torch.long)
    query_inputs = []
    query_targets = []
    loss_masks = []
    for case in cases:
        joined = [*case.prefix_ids, *case.target_ids]
        query_inputs.append(joined[:-1])
        query_targets.append(joined[1:])
        mask = [False] * (len(joined) - 1)
        mask[int(config.prefix_length) - 1 :] = [True] * int(config.target_length)
        loss_masks.append(mask)
    return EncodedDocumentContinuations(
        source_ids=source_ids,
        prefix_ids=prefix_ids,
        query_input_ids=torch.tensor(query_inputs, dtype=torch.long),
        query_target_ids=torch.tensor(query_targets, dtype=torch.long),
        query_loss_mask=torch.tensor(loss_masks, dtype=torch.bool),
    )


def build_archive_groups(
    cases: Sequence[DocumentContinuationCase],
    *,
    facts_per_query: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if int(facts_per_query) < 2:
        raise ValueError("facts_per_query must be at least two")
    source_buckets: dict[int, list[int]] = {}
    for index, case in enumerate(cases):
        source_buckets.setdefault(int(case.source_index), []).append(index)
    if any(len(values) < int(facts_per_query) for values in source_buckets.values()):
        raise ValueError("each corpus needs at least facts_per_query cases")
    generator = random.Random(int(seed))
    groups = []
    target_slots = []
    for target, case in enumerate(cases):
        distractors = generator.sample(
            [
                index
                for index in source_buckets[int(case.source_index)]
                if index != target
            ],
            int(facts_per_query) - 1,
        )
        row = [target, *distractors]
        generator.shuffle(row)
        groups.append(row)
        target_slots.append(row.index(target))
    return torch.tensor(groups, dtype=torch.long), torch.tensor(
        target_slots, dtype=torch.long
    )


def build_policy_rankings(
    *,
    lexical_scores: torch.Tensor,
    frozen_last_scores: torch.Tensor,
    frozen_mean_scores: torch.Tensor,
    target_slots: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    shape = lexical_scores.shape
    if frozen_last_scores.shape != shape or frozen_mean_scores.shape != shape:
        raise ValueError("retrieval score shapes must match")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random_scores = torch.rand(shape, generator=generator)
    recency_scores = torch.arange(shape[1], dtype=torch.float32).unsqueeze(0).expand(
        shape[0], -1
    )
    oracle_scores = torch.zeros(shape, dtype=torch.float32)
    oracle_scores.scatter_(1, target_slots.unsqueeze(1), 1.0)
    return {
        "random1": rankings_from_scores(random_scores),
        "random2": rankings_from_scores(random_scores),
        "recency1": rankings_from_scores(recency_scores),
        "recency2": rankings_from_scores(recency_scores),
        "lexical1": rankings_from_scores(lexical_scores),
        "lexical2": rankings_from_scores(lexical_scores),
        "frozen_last1": rankings_from_scores(frozen_last_scores),
        "frozen_last2": rankings_from_scores(frozen_last_scores),
        "frozen_mean1": rankings_from_scores(frozen_mean_scores),
        "frozen_mean2": rankings_from_scores(frozen_mean_scores),
        "oracle1": rankings_from_scores(oracle_scores),
    }


def selected_slots_for_arm(
    arm: str,
    *,
    policy_rankings: Mapping[str, torch.Tensor],
    facts_per_query: int,
) -> torch.Tensor | None:
    if arm == "off":
        return None
    if arm == "all4":
        sample = next(iter(policy_rankings.values()))
        return torch.arange(int(facts_per_query), dtype=torch.long).unsqueeze(0).expand(
            sample.shape[0], -1
        )
    if arm in {
        "random1",
        "recency1",
        "lexical1",
        "frozen_last1",
        "frozen_mean1",
        "oracle1",
    }:
        return policy_rankings[arm][:, :1]
    if arm in {
        "random2",
        "recency2",
        "lexical2",
        "frozen_last2",
        "frozen_mean2",
    }:
        return policy_rankings[arm][:, :2]
    raise ValueError(f"unknown V22 arm: {arm}")


def gather_retrieved_episodes(
    bank: EncodedDocumentContinuations,
    groups: torch.Tensor,
    selected_slots: torch.Tensor | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    if selected_slots is None:
        return torch.empty(groups.shape[0], 0, dtype=torch.long, device=device)
    selected_cases = groups.gather(1, selected_slots)
    return bank.source_ids.index_select(0, selected_cases.reshape(-1)).reshape(
        groups.shape[0], -1
    ).to(device)


def retrieval_metrics_for_arm(
    arm: str,
    *,
    selected_slots: torch.Tensor | None,
    target_slots: torch.Tensor,
    cases: Sequence[DocumentContinuationCase],
    source_length: int,
) -> dict[str, Any]:
    count = 0 if selected_slots is None else int(selected_slots.shape[1])
    if selected_slots is None:
        included = torch.zeros(len(cases), dtype=torch.bool)
    else:
        included = (selected_slots == target_slots.unsqueeze(1)).any(dim=1)
    per_source: dict[str, dict[str, float | int]] = {}
    for source_name in sorted({case.source_name for case in cases}):
        indices = [index for index, case in enumerate(cases) if case.source_name == source_name]
        per_source[source_name] = {
            "case_count": len(indices),
            "target_inclusion": float(included[indices].float().mean()),
        }
    return {
        "arm": arm,
        "selected_source_count": count,
        "active_source_tokens": count * int(source_length),
        "target_inclusion": float(included.float().mean()),
        "per_source": per_source,
        "target_inclusion_mask": [bool(value) for value in included],
        "target_document_identity_used_by_selector": arm == "oracle1",
        "target_slot_metrics_only": True,
    }


def retrieval_confidence_curves(
    scores: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    coverages: Sequence[float] = (0.25, 0.50, 0.75, 1.0),
) -> dict[str, Any]:
    """Metrics-only precision curves for deciding whether a gate is plausible."""

    if scores.ndim != 2 or scores.shape[0] != target_slots.shape[0]:
        raise ValueError("confidence scores and target slots must align")
    ordered_scores, ordered_slots = torch.sort(
        scores, dim=1, descending=True, stable=True
    )
    correct = ordered_slots[:, 0] == target_slots
    margin = ordered_scores[:, 0] - ordered_scores[:, 1]
    maximum = ordered_scores[:, 0]

    def curve(signal: torch.Tensor) -> list[dict[str, Any]]:
        ranking = torch.argsort(signal, descending=True, stable=True)
        rows = []
        for coverage in coverages:
            selected_count = max(
                1, min(len(correct), round(float(coverage) * len(correct)))
            )
            selected = ranking[:selected_count]
            selected_correct = int(correct.index_select(0, selected).sum())
            rows.append(
                {
                    "requested_coverage": float(coverage),
                    "selected_count": selected_count,
                    "actual_coverage": selected_count / len(correct),
                    "precision": selected_correct / selected_count,
                    "effective_correct_retrieval_rate": selected_correct / len(correct),
                    "minimum_selected_signal": float(
                        signal.index_select(0, selected).min()
                    ),
                }
            )
        return rows

    return {
        "margin_top1_minus_top2": curve(margin),
        "absolute_top1_score": curve(maximum),
        "target_identity_used_only_for_metrics": True,
        "promotable": False,
    }


@torch.no_grad()
def evaluate_document_arm(
    model: MarulhoHashedMicroExpertLanguageModel,
    bank: EncodedDocumentContinuations,
    groups: torch.Tensor,
    selected_slots: torch.Tensor | None,
    cases: Sequence[DocumentContinuationCase],
    *,
    batch_size: int,
    precision: str,
) -> dict[str, Any]:
    model.eval()
    losses = []
    accuracies = []
    active_positions = 0
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    for start in range(0, bank.case_count, int(batch_size)):
        end = min(bank.case_count, start + int(batch_size))
        batch_groups = groups[start:end]
        batch_selected = (
            None if selected_slots is None else selected_slots[start:end]
        )
        retrieved = gather_retrieved_episodes(
            bank, batch_groups, batch_selected, device=model.device
        )
        query_input = bank.query_input_ids[start:end].to(model.device)
        targets = bank.query_target_ids[start:end].to(model.device)
        mask = bank.query_loss_mask[start:end].to(model.device)
        combined = torch.cat((retrieved, query_input), dim=1)
        if int(combined.shape[1]) > int(model.hashed_config.context_length):
            raise ValueError("V22 retrieved context exceeds the cortex window")
        with _precision_context(model.device, str(precision)):
            hidden = model._forward_hidden(
                combined, collect_telemetry=False
            )["hidden"][:, -int(query_input.shape[1]) :]
            logits = model.lm_head(hidden)
        token_losses = F.cross_entropy(
            logits.float().transpose(1, 2), targets, reduction="none"
        )
        masked_losses = (token_losses * mask).sum(dim=1) / mask.sum(dim=1)
        predictions = logits.argmax(dim=-1)
        masked_accuracy = ((predictions == targets) & mask).sum(dim=1) / mask.sum(dim=1)
        losses.append(masked_losses.cpu())
        accuracies.append(masked_accuracy.float().cpu())
        active_positions += int(combined.numel())
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = time.perf_counter() - started
    case_losses = torch.cat(losses)
    case_accuracies = torch.cat(accuracies)
    per_source = {}
    for source_name in sorted({case.source_name for case in cases}):
        indices = torch.tensor(
            [index for index, case in enumerate(cases) if case.source_name == source_name],
            dtype=torch.long,
        )
        per_source[source_name] = {
            "case_count": int(indices.numel()),
            "heldout_loss": float(case_losses.index_select(0, indices).mean()),
            "next_token_accuracy": float(
                case_accuracies.index_select(0, indices).mean()
            ),
        }
    return {
        "heldout_loss": float(case_losses.mean()),
        "next_token_accuracy": float(case_accuracies.mean()),
        "case_count": len(cases),
        "target_token_count": len(cases) * int(bank.query_loss_mask[0].sum()),
        "per_source": per_source,
        "case_losses": [float(value) for value in case_losses],
        "case_next_token_accuracy": [float(value) for value in case_accuracies],
        "elapsed_seconds": elapsed,
        "active_input_positions": active_positions,
        "input_positions_per_second": active_positions / max(elapsed, 1.0e-12),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda"
            else 0
        ),
        "parameters_frozen": True,
    }


def paired_bootstrap_gain(
    baseline_losses: Sequence[float],
    candidate_losses: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = torch.tensor(baseline_losses, dtype=torch.float64)
    candidate = torch.tensor(candidate_losses, dtype=torch.float64)
    if baseline.shape != candidate.shape or baseline.ndim != 1:
        raise ValueError("paired bootstrap losses must be same-length vectors")
    gain = baseline - candidate
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = torch.randint(
        0,
        int(gain.numel()),
        (int(samples), int(gain.numel())),
        generator=generator,
    )
    means = gain.index_select(0, indices.reshape(-1)).reshape(indices.shape).mean(1)
    return {
        "mean_loss_gain": float(gain.mean()),
        "median_case_loss_gain": float(gain.median()),
        "case_win_rate": float((gain > 0).float().mean()),
        "bootstrap_samples": int(samples),
        "bootstrap_95_ci": [
            float(torch.quantile(means, 0.025)),
            float(torch.quantile(means, 0.975)),
        ],
        "positive_mean_probability": float((means > 0).float().mean()),
    }


def attach_paired_evidence(
    arms: dict[str, dict[str, Any]],
    cases: Sequence[DocumentContinuationCase],
    *,
    config: CausalDocumentRetrievalConfig,
) -> None:
    baseline = arms["off"]["language"]["case_losses"]
    for arm_index, arm in enumerate(ARM_NAMES):
        language = arms[arm]["language"]
        language["paired_to_off"] = paired_bootstrap_gain(
            baseline,
            language["case_losses"],
            samples=int(config.bootstrap_samples),
            seed=int(config.data_seed) + 100 + arm_index,
        )
        inclusion_mask = arms[arm]["retrieval"]["target_inclusion_mask"]
        conditional = {}
        for label, expected in (("target_included", True), ("target_absent", False)):
            indices = [
                index
                for index, included in enumerate(inclusion_mask)
                if bool(included) is expected
            ]
            conditional[label] = (
                None
                if not indices
                else {
                    "case_count": len(indices),
                    **paired_bootstrap_gain(
                        [baseline[index] for index in indices],
                        [language["case_losses"][index] for index in indices],
                        samples=int(config.bootstrap_samples),
                        seed=int(config.data_seed) + 500 + 19 * arm_index,
                    ),
                }
            )
        language["paired_by_target_inclusion"] = conditional
        for source_index, source_name in enumerate(
            sorted({case.source_name for case in cases})
        ):
            indices = [
                index for index, case in enumerate(cases) if case.source_name == source_name
            ]
            language["per_source"][source_name]["paired_to_off"] = paired_bootstrap_gain(
                [baseline[index] for index in indices],
                [language["case_losses"][index] for index in indices],
                samples=int(config.bootstrap_samples),
                seed=int(config.data_seed) + 1000 + 17 * arm_index + source_index,
            )


def document_retrieval_decision(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    config: CausalDocumentRetrievalConfig,
) -> tuple[str, str | None]:
    missing = [name for name in ARM_NAMES if name not in arms]
    if missing:
        return "incomplete_v22_missing_arm", None
    oracle = arms["oracle1"]["language"]["paired_to_off"]
    if (
        float(oracle["mean_loss_gain"]) < float(config.minimum_oracle_loss_gain)
        or float(oracle["bootstrap_95_ci"][0]) <= 0.0
    ):
        return "redesign_v22_prior_episode_not_predictively_useful", None
    off_loss = float(arms["off"]["language"]["heldout_loss"])
    all_history_loss = float(arms["all4"]["language"]["heldout_loss"])
    ordered = sorted(
        CANDIDATE_ARMS,
        key=lambda name: float(arms[name]["language"]["heldout_loss"]),
    )
    for arm in ordered:
        row = arms[arm]
        language = row["language"]
        paired = language["paired_to_off"]
        source_gains = [
            float(value["paired_to_off"]["mean_loss_gain"])
            for value in language["per_source"].values()
        ]
        candidate_loss = float(language["heldout_loss"])
        selected_count = int(row["retrieval"]["selected_source_count"])
        suffix = "1" if selected_count == 1 else "2"
        random_row = arms[f"random{suffix}"]
        recency_row = arms[f"recency{suffix}"]
        random_loss = float(random_row["language"]["heldout_loss"])
        recency_loss = float(recency_row["language"]["heldout_loss"])
        target_inclusion = float(row["retrieval"]["target_inclusion"])
        control_inclusion = max(
            float(random_row["retrieval"]["target_inclusion"]),
            float(recency_row["retrieval"]["target_inclusion"]),
        )
        minimum_recall = (
            float(config.minimum_target_recall_at_1)
            if selected_count == 1
            else float(config.minimum_target_recall_at_2)
        )
        if (
            target_inclusion >= minimum_recall
            and target_inclusion - control_inclusion
            >= float(config.minimum_recall_gain_over_control)
            and off_loss - candidate_loss
            >= float(config.minimum_candidate_loss_gain)
            and random_loss - candidate_loss
            >= float(config.minimum_control_loss_gain)
            and recency_loss - candidate_loss
            >= float(config.minimum_control_loss_gain)
            and candidate_loss - all_history_loss
            <= float(config.maximum_regret_to_all_history)
            and float(paired["bootstrap_95_ci"][0]) > 0.0
            and min(source_gains) >= -float(config.maximum_source_loss_regression)
        ):
            return ADVANCE_DECISION, arm
    return "redesign_v22_addressing_does_not_recover_useful_episode", None


def _case_examples(
    cases: Sequence[DocumentContinuationCase],
    tokenizer,
    arms: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    chosen = []
    for source_name in sorted({case.source_name for case in cases}):
        source_indices = [
            index
            for index, case in enumerate(cases)
            if case.source_name == source_name
        ]
        chosen.extend(source_indices[:2])
    rows = []
    for index in chosen:
        case = cases[index]
        rows.append(
            {
                "case_id": case.case_id,
                "source_name": case.source_name,
                "document_sha256": case.document_sha256,
                "archive_episode": tokenizer.decode(case.source_ids),
                "visible_prefix": tokenizer.decode(case.prefix_ids),
                "hidden_target": tokenizer.decode(case.target_ids),
                "losses": {
                    arm: float(arms[arm]["language"]["case_losses"][index])
                    for arm in ARM_NAMES
                },
            }
        )
    return rows


def run_causal_document_retrieval_audit(
    *,
    parent_checkpoint_path: str | Path,
    eval_paths: Sequence[str | Path],
    output_path: str | Path,
    config: CausalDocumentRetrievalConfig = CausalDocumentRetrievalConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if len(eval_paths) != 2:
        raise ValueError("V22 requires exactly two document-disjoint eval sources")
    if int(config.facts_per_query) != 4:
        raise ValueError("V22 currently requires four archive candidates")
    maximum_sequence = (
        int(config.facts_per_query) * int(config.source_length)
        + int(config.prefix_length)
        + int(config.target_length)
        - 1
    )
    resolved = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for V22 but unavailable")
    started = time.perf_counter()
    parent_path = Path(parent_checkpoint_path)
    model, tokenizer, parent_metadata = load_hashed_micro_expert_checkpoint(
        parent_path, map_location="cpu"
    )
    parent_tokens = _validate_parent(model, parent_metadata)
    if parent_tokens < 1_000_000_000:
        raise ValueError("V22 requires the one-billion-token V11 parent")
    if maximum_sequence > int(model.hashed_config.context_length):
        raise ValueError(
            f"V22 all-history sequence {maximum_sequence} exceeds parent context "
            f"{model.hashed_config.context_length}"
        )

    all_cases = []
    source_reports = []
    for source_index, raw_path in enumerate(eval_paths):
        path = Path(raw_path)
        text, sample_report = sample_corpus_ranges(
            path,
            byte_budget=int(config.sample_bytes),
            range_count=int(config.sample_range_count),
        )
        cases, selection_report = build_document_cases(
            tokenizer,
            text,
            source_index=source_index,
            source_name=path.stem,
            config=config,
            seed=int(config.data_seed) + source_index,
        )
        all_cases.extend(cases)
        source_reports.append(
            {
                **sample_report,
                **selection_report,
                "file_sha256": sha256_file(path),
                "declared_role": "document_disjoint_evaluation",
            }
        )
    cases = tuple(all_cases)
    bank = encode_document_cases(cases, config=config)
    groups, target_slots = build_archive_groups(
        cases,
        facts_per_query=int(config.facts_per_query),
        seed=int(config.data_seed) + 10,
    )
    if any(int(groups[index, target_slots[index]]) != index for index in range(len(cases))):
        raise RuntimeError("V22 archive group lost its same-document target")

    source_bank = EncodedTextBank(
        ids=bank.source_ids,
        mask=torch.ones_like(bank.source_ids, dtype=torch.bool),
    )
    query_bank = EncodedTextBank(
        ids=bank.prefix_ids,
        mask=torch.ones_like(bank.prefix_ids, dtype=torch.bool),
    )
    lexical_scores = lexical_tfidf_scores(
        source_bank,
        query_bank,
        groups,
        excluded_token_ids=_special_token_ids(tokenizer),
    )
    model = model.to(resolved).eval()
    print("[causal-document-v22] extracting frozen V11 keys", flush=True)
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
    grouped_last = source_last.index_select(0, groups.reshape(-1)).reshape(
        len(cases), int(config.facts_per_query), -1
    )
    grouped_mean = source_mean.index_select(0, groups.reshape(-1)).reshape(
        len(cases), int(config.facts_per_query), -1
    )
    frozen_last_scores = torch.einsum("bfd,bd->bf", grouped_last, query_last)
    frozen_mean_scores = torch.einsum("bfd,bd->bf", grouped_mean, query_mean)
    policy_rankings = build_policy_rankings(
        lexical_scores=lexical_scores,
        frozen_last_scores=frozen_last_scores,
        frozen_mean_scores=frozen_mean_scores,
        target_slots=target_slots,
        seed=int(config.data_seed) + 20,
    )
    confidence_diagnostics = {
        "lexical": retrieval_confidence_curves(lexical_scores, target_slots),
        "frozen_last": retrieval_confidence_curves(
            frozen_last_scores, target_slots
        ),
        "frozen_mean": retrieval_confidence_curves(
            frozen_mean_scores, target_slots
        ),
        "purpose": "non_promotable_gate_feasibility_only",
    }

    arms: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        selected = selected_slots_for_arm(
            arm,
            policy_rankings=policy_rankings,
            facts_per_query=int(config.facts_per_query),
        )
        retrieval = retrieval_metrics_for_arm(
            arm,
            selected_slots=selected,
            target_slots=target_slots,
            cases=cases,
            source_length=int(config.source_length),
        )
        print(
            f"[causal-document-v22] evaluating {arm} "
            f"inclusion={retrieval['target_inclusion']:.3f}",
            flush=True,
        )
        arms[arm] = {
            "retrieval": retrieval,
            "language": evaluate_document_arm(
                model,
                bank,
                groups,
                selected,
                cases,
                batch_size=int(config.eval_batch_size),
                precision=str(config.precision),
            ),
        }
    attach_paired_evidence(arms, cases, config=config)
    decision, selected_arm = document_retrieval_decision(arms, config=config)
    examples = _case_examples(cases, tokenizer, arms)
    for arm in ARM_NAMES:
        arms[arm]["language"].pop("case_losses")
        arms[arm]["language"].pop("case_next_token_accuracy")
        arms[arm]["retrieval"].pop("target_inclusion_mask")

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
        "sources": source_reports,
        "archive": {
            "case_count": len(cases),
            "facts_per_query": int(config.facts_per_query),
            "episode_tokens": int(config.source_length),
            "group_sha256": _tensor_sha256(groups, target_slots),
            "source_tensor_sha256": _tensor_sha256(bank.source_ids),
            "prefix_tensor_sha256": _tensor_sha256(bank.prefix_ids),
            "target_tensor_sha256": _tensor_sha256(bank.query_target_ids),
            "content": "exact_prior_document_token_spans",
            "distractors_from_same_corpus": True,
            "write_order_randomized_before_query": True,
        },
        "anti_cheat": {
            "archive_write_input": "first_prior_48_document_tokens_only",
            "retrieval_query_input": "later_visible_48_token_prefix_only",
            "scored_target_input_to_selector": False,
            "minimum_unseen_gap_tokens": int(config.minimum_gap_tokens),
            "source_ends_before_prefix": all(
                case.source_end <= case.prefix_start for case in cases
            ),
            "prefix_ends_before_target": all(
                case.prefix_end == case.target_start for case in cases
            ),
            "candidate_selectors_use_document_identity": False,
            "target_slot_metrics_only": True,
            "oracle_uses_document_identity": True,
            "oracle_promotable": False,
            "teacher_forcing_target_used_only_for_language_scoring": True,
        },
        "key_interfaces": {
            "lexical": "checkpoint_bpe_tfidf_visible_prefix_to_prior_episode",
            "frozen_last": "cosine_of_final_frozen_v11_states",
            "frozen_mean": "cosine_of_masked_mean_frozen_v11_states",
            "random": "seeded_equal_token_control",
            "recency": "last_written_equal_token_control",
            "learned_selector": False,
        },
        "confidence_diagnostics": confidence_diagnostics,
        "arms": arms,
        "examples": examples,
        "decision": decision,
        "selected_arm": selected_arm,
        "promotion_boundary": {
            "advance_to_joint_training": decision == ADVANCE_DECISION,
            "frozen_audit_is_language_quality": False,
            "base_quality_promoted": False,
            "checkpoint_saved": False,
            "runtime_install_allowed": False,
            "continual_learning_claimed": False,
            "speed_claimed": False,
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
        title="MARULHO V22 Causal Document Retrieval Audit",
    )
    print(
        f"[causal-document-v22] decision {decision} selected={selected_arm}",
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--eval-source", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-count-per-source", type=int, default=128)
    parser.add_argument("--sample-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--sample-range-count", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=4096)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--data-seed", type=int, default=9701)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = CausalDocumentRetrievalConfig(
        case_count_per_source=int(args.case_count_per_source),
        sample_bytes=int(args.sample_bytes),
        sample_range_count=int(args.sample_range_count),
        eval_batch_size=int(args.eval_batch_size),
        feature_batch_size=int(args.feature_batch_size),
        bootstrap_samples=int(args.bootstrap_samples),
        precision=str(args.precision),
        data_seed=int(args.data_seed),
    )
    run_causal_document_retrieval_audit(
        parent_checkpoint_path=args.parent_checkpoint,
        eval_paths=args.eval_source,
        output_path=args.output,
        config=config,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
