from __future__ import annotations

import argparse
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hecsn.config.presets import (
    autonomy_acquisition_preset_names,
    get_autonomy_acquisition_preset,
)
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.gap_planner import bank_semantic_relevance_score
from hecsn.reporting.autonomy import plot_acquisition_summary
from hecsn.reporting.io import write_json_file
from hecsn.semantics.frontier import (
    candidate_semantic_signature,
    current_context_signature,
    frontier_semantic_plan,
)
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.autonomy_runner import (
    ProbeGapMetrics,
    SourceBank,
    describe_source_banks,
    load_source_banks,
    make_config,
    probe_gap,
    selection_metric,
    select_active_source,
    source_bank_protocol,
    summarize_training_metrics,
)
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


ACQUISITION_ABSOLUTE_IMPROVEMENT_TARGET = 0.01
ACQUISITION_RELATIVE_IMPROVEMENT_FRACTION = 0.03


def acquisition_candidate_improvement_target(reference_gap: float) -> float:
    positive_reference = max(0.0, float(reference_gap))
    return min(
        ACQUISITION_ABSOLUTE_IMPROVEMENT_TARGET,
        ACQUISITION_RELATIVE_IMPROVEMENT_FRACTION * positive_reference,
    )


def _candidate_gap_better_than(delta: float, reference_gap: float) -> bool:
    return float(delta) <= -acquisition_candidate_improvement_target(reference_gap)


def acquisition_gate_from_comparison(comparison: dict[str, float]) -> dict[str, Any]:
    mean_target = acquisition_candidate_improvement_target(comparison["round_robin_final_mean_candidate_gap"])
    max_target = acquisition_candidate_improvement_target(comparison["round_robin_final_max_candidate_gap"])
    return {
        "pass": bool(
            comparison["active_minus_round_robin_mean_candidate_gap"] <= -mean_target
            and comparison["active_minus_round_robin_max_candidate_gap"] <= -max_target
        ),
        "active_mean_candidate_gap_better_than_round_robin": bool(
            comparison["active_minus_round_robin_mean_candidate_gap"] <= -mean_target
        ),
        "active_max_candidate_gap_better_than_round_robin": bool(
            comparison["active_minus_round_robin_max_candidate_gap"] <= -max_target
        ),
        "thresholds": {
            "absolute_improvement_cap": ACQUISITION_ABSOLUTE_IMPROVEMENT_TARGET,
            "relative_improvement_fraction": ACQUISITION_RELATIVE_IMPROVEMENT_FRACTION,
            "active_minus_round_robin_mean_candidate_gap_max": -mean_target,
            "active_minus_round_robin_max_candidate_gap_max": -max_target,
        },
    }


def _candidate_has_semantic_signal(bank: SourceBank) -> bool:
    metadata = bank.metadata or {}
    semantic_relevance = float(metadata.get("semantic_relevance") or 0.0)
    query_text = str(metadata.get("query_text") or "").strip().lower()
    return semantic_relevance > 0.0 or query_text not in {"", "none"}


def candidate_gap_snapshot(
    trainer: HECSNTrainer,
    candidates: list[SourceBank],
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
) -> dict[str, ProbeGapMetrics]:
    frontier_plan = frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)
    snapshot = {
        bank.name: probe_gap(
            trainer,
            bank,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        for bank in candidates
    }
    for bank in candidates:
        metrics = snapshot[bank.name]
        frontier_relevance = bank_semantic_relevance_score(bank, frontier_plan)
        metrics["frontier_semantic_relevance"] = float(frontier_relevance)
        metrics["frontier_follow_up_questions"] = list(frontier_plan.get("follow_up_questions") or [])
        metrics["semantic_action_score"] = float(
            float(metrics.get("semantic_action_score", metrics["diagnostic_gap_score"]))
            + 0.20 * float(frontier_relevance)
        )
    return snapshot


def next_candidate_round_robin(candidates: list[SourceBank], start_idx: int) -> tuple[SourceBank, int]:
    total = len(candidates)
    for offset in range(total):
        idx = (start_idx + offset) % total
        if candidates[idx].remaining() > 0:
            return candidates[idx], (idx + 1) % total
    raise RuntimeError("No candidate source has remaining tokens")


def train_source_chunk(trainer: HECSNTrainer, bank: SourceBank, n_tokens: int, phase: str, metrics_rows: list[dict[str, Any]]) -> int:
    chunk = bank.next_chunk(n_tokens)
    for raw_window, pattern in chunk:
        row = trainer.train_step(pattern, raw_window=raw_window)
        row["source_name"] = bank.name
        row["phase"] = phase
        metrics_rows.append(row)
    return len(chunk)


def preview_source_chunk(bank: SourceBank, n_tokens: int, *, offset_tokens: int = 0) -> list[tuple[str, torch.Tensor]]:
    start = min(len(bank.train_patterns), bank.cursor + max(0, int(offset_tokens)))
    end = min(len(bank.train_patterns), start + max(0, int(n_tokens)))
    return list(zip(bank.train_raw_windows[start:end], bank.train_patterns[start:end]))


def consume_previewed_chunk(bank: SourceBank, preview_chunk: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    if not preview_chunk:
        return []

    committed_chunk = bank.next_chunk(len(preview_chunk))
    if len(committed_chunk) != len(preview_chunk):
        raise RuntimeError("Committed chunk length diverged from preview chunk length")

    for (expected_raw, expected_pattern), (actual_raw, actual_pattern) in zip(preview_chunk, committed_chunk):
        if expected_raw != actual_raw or not torch.equal(expected_pattern, actual_pattern):
            raise RuntimeError("Committed chunk diverged from preview chunk contents")

    return committed_chunk


def replay_source_chunk(
    trainer: HECSNTrainer,
    bank: SourceBank,
    chunk: list[tuple[str, torch.Tensor]],
    phase: str,
    metrics_rows: list[dict[str, Any]] | None,
) -> int:
    for raw_window, pattern in chunk:
        row = trainer.train_step(pattern, raw_window=raw_window)
        if metrics_rows is not None:
            row["source_name"] = bank.name
            row["phase"] = phase
            metrics_rows.append(row)
    return len(chunk)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def record_trial_chunk_consumption(bank: SourceBank, chunk: list[tuple[str, torch.Tensor]]) -> None:
    if not chunk:
        return
    bank.cursor = min(len(bank.train_patterns), bank.cursor + len(chunk))
    bank.visits += 1


def serialize_candidate_snapshot(snapshot: dict[str, ProbeGapMetrics]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "recon_error": float(values["recon_error"]),
            "gap_score": float(values["gap_score"]),
            "diagnostic_gap_score": float(values["diagnostic_gap_score"]),
            "semantic_action_score": float(values.get("semantic_action_score", values["diagnostic_gap_score"])),
            "concept_novelty": float(values.get("concept_novelty", 0.0)),
            "concept_uncertainty": float(values.get("concept_uncertainty", 0.0)),
            "concept_support": float(values.get("concept_support", 0.0)),
            "frontier_semantic_relevance": float(values.get("frontier_semantic_relevance", 0.0)),
            "info_gain_score": float(values.get("info_gain_score", values["diagnostic_gap_score"])),
            "winner_switch_rate": float(values["winner_switch_rate"]),
            "mean_top1_margin": float(values["mean_top1_margin"]),
        }
        for name, values in snapshot.items()
    }


def summarize_candidate_frontier(snapshot: dict[str, ProbeGapMetrics]) -> dict[str, Any]:
    final_candidate_gap_by_source = {
        name: float(values["recon_error"])
        for name, values in snapshot.items()
    }
    final_candidate_diagnostic_gap_by_source = {
        name: float(values["diagnostic_gap_score"])
        for name, values in snapshot.items()
    }
    final_candidate_info_gain_by_source = {
        name: float(values["info_gain_score"])
        for name, values in snapshot.items()
    }
    final_candidate_semantic_action_by_source = {
        name: float(values.get("semantic_action_score", values["diagnostic_gap_score"]))
        for name, values in snapshot.items()
    }

    gap_values = list(final_candidate_gap_by_source.values())
    diagnostic_gap_values = list(final_candidate_diagnostic_gap_by_source.values())
    info_gain_values = list(final_candidate_info_gain_by_source.values())
    semantic_action_values = list(final_candidate_semantic_action_by_source.values())
    return {
        "final_candidate_gap_by_source": final_candidate_gap_by_source,
        "final_candidate_diagnostic_gap_by_source": final_candidate_diagnostic_gap_by_source,
        "final_candidate_info_gain_by_source": final_candidate_info_gain_by_source,
        "final_candidate_semantic_action_by_source": final_candidate_semantic_action_by_source,
        "final_mean_candidate_gap": float(np.mean(gap_values)) if gap_values else float("nan"),
        "final_max_candidate_gap": float(max(gap_values)) if gap_values else float("nan"),
        "final_mean_candidate_diagnostic_gap": float(np.mean(diagnostic_gap_values)) if diagnostic_gap_values else float("nan"),
        "final_max_candidate_diagnostic_gap": float(max(diagnostic_gap_values)) if diagnostic_gap_values else float("nan"),
        "final_mean_candidate_info_gain": float(np.mean(info_gain_values)) if info_gain_values else float("nan"),
        "final_max_candidate_info_gain": float(max(info_gain_values)) if info_gain_values else float("nan"),
        "final_mean_candidate_semantic_action": float(np.mean(semantic_action_values)) if semantic_action_values else float("nan"),
        "final_max_candidate_semantic_action": float(max(semantic_action_values)) if semantic_action_values else float("nan"),
    }


def project_candidate_frontier(
    trainer: HECSNTrainer,
    bank: SourceBank,
    preview_chunk: list[tuple[str, torch.Tensor]],
    projection_candidates: list[SourceBank],
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    *,
    followup_tokens: int = 0,
    baseline_metrics: ProbeGapMetrics | None = None,
) -> dict[str, Any]:
    before = baseline_metrics or probe_gap(
        trainer,
        bank,
        gap_exploration_bonus,
        gap_ambiguity_weight,
        gap_switch_weight,
        gap_margin_reference,
    )
    gap_before = float(before["recon_error"])
    diagnostic_gap_before = float(before["diagnostic_gap_score"])

    trial_trainer = deepcopy(trainer)
    trial_candidates = deepcopy(projection_candidates)
    trial_lookup = {candidate.name: candidate for candidate in trial_candidates}
    trial_bank = trial_lookup.get(bank.name)
    if trial_bank is None:
        trial_bank = deepcopy(bank)
        trial_candidates.append(trial_bank)

    replay_source_chunk(trial_trainer, trial_bank, preview_chunk, "lookahead_preview", None)
    record_trial_chunk_consumption(trial_bank, preview_chunk)
    after_preview = probe_gap(
        trial_trainer,
        trial_bank,
        gap_exploration_bonus,
        gap_ambiguity_weight,
        gap_switch_weight,
        gap_margin_reference,
    )

    followup_chunk = preview_source_chunk(trial_bank, followup_tokens)
    if followup_chunk:
        replay_source_chunk(trial_trainer, trial_bank, followup_chunk, "lookahead_followup", None)
        record_trial_chunk_consumption(trial_bank, followup_chunk)

    projected_snapshot = candidate_gap_snapshot(
        trial_trainer,
        trial_candidates,
        gap_exploration_bonus,
        gap_ambiguity_weight,
        gap_switch_weight,
        gap_margin_reference,
    )
    projected_frontier = summarize_candidate_frontier(projected_snapshot)
    return {
        "source": bank.name,
        "tokens_trained": int(len(preview_chunk)),
        "gap_before": gap_before,
        "gap_after": float(after_preview["recon_error"]),
        "gap_reduction": gap_before - float(after_preview["recon_error"]),
        "diagnostic_gap_before": diagnostic_gap_before,
        "diagnostic_gap_after": float(after_preview["diagnostic_gap_score"]),
        "diagnostic_gap_reduction": diagnostic_gap_before - float(after_preview["diagnostic_gap_score"]),
        "projected_commit_tokens": int(len(followup_chunk)),
        "projected_final_mean_candidate_gap": float(projected_frontier["final_mean_candidate_gap"]),
        "projected_final_max_candidate_gap": float(projected_frontier["final_max_candidate_gap"]),
        "projected_final_mean_candidate_diagnostic_gap": float(projected_frontier["final_mean_candidate_diagnostic_gap"]),
        "projected_final_max_candidate_diagnostic_gap": float(projected_frontier["final_max_candidate_diagnostic_gap"]),
    }


def evaluate_projected_candidates(
    trainer: HECSNTrainer,
    available: list[SourceBank],
    preview_chunks: dict[str, list[tuple[str, torch.Tensor]]],
    projection_candidates: list[SourceBank],
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    *,
    followup_tokens: int = 0,
    followup_tokens_by_source: dict[str, int] | None = None,
    snapshot: dict[str, ProbeGapMetrics] | None = None,
) -> list[dict[str, Any]]:
    projected_rows: list[dict[str, Any]] = []
    base_rng_state = capture_rng_state()
    try:
        for bank in available:
            preview_chunk = preview_chunks.get(bank.name, [])
            if not preview_chunk:
                continue
            restore_rng_state(base_rng_state)
            projected_rows.append(
                project_candidate_frontier(
                    trainer,
                    bank,
                    preview_chunk,
                    projection_candidates,
                    gap_exploration_bonus,
                    gap_ambiguity_weight,
                    gap_switch_weight,
                    gap_margin_reference,
                    followup_tokens=int(followup_tokens if followup_tokens_by_source is None else followup_tokens_by_source.get(bank.name, followup_tokens)),
                    baseline_metrics=None if snapshot is None else snapshot.get(bank.name),
                )
            )
    finally:
        restore_rng_state(base_rng_state)
    return projected_rows


def select_projected_commit_target(
    candidates: list[SourceBank],
    projected_rows: list[dict[str, Any]],
    snapshot: dict[str, ProbeGapMetrics],
    coverage_balance_penalty: float,
    gap_focus_margin: float,
) -> tuple[SourceBank, dict[str, float]]:
    projected_lookup = {
        str(row["source"]): row
        for row in projected_rows
        if np.isfinite(float(row.get("projected_final_mean_candidate_gap", float("nan"))))
    }
    if projected_lookup:
        projected_candidates = [bank for bank in candidates if bank.name in projected_lookup]
        if projected_candidates:
            current_gap_values = [float(values["recon_error"]) for values in snapshot.values()]
            current_mean_gap = float(np.mean(current_gap_values)) if current_gap_values else 0.0
            current_max_gap = float(max(current_gap_values)) if current_gap_values else 0.0
            projected_scores = {
                bank.name: float(
                    (current_mean_gap - float(projected_lookup[bank.name]["projected_final_mean_candidate_gap"]))
                    + 0.5 * (current_max_gap - float(projected_lookup[bank.name]["projected_final_max_candidate_gap"]))
                )
                for bank in projected_candidates
            }
            selected = min(
                projected_candidates,
                key=lambda bank: (
                    float(projected_lookup[bank.name]["projected_final_mean_candidate_gap"]),
                    float(projected_lookup[bank.name]["projected_final_max_candidate_gap"]),
                    float(projected_lookup[bank.name].get("projected_final_mean_candidate_diagnostic_gap", float("inf"))),
                    float(projected_lookup[bank.name].get("projected_final_max_candidate_diagnostic_gap", float("inf"))),
                    -float(selection_metric(snapshot.get(bank.name, {}))),
                    -max(0.0, float(projected_lookup[bank.name].get("diagnostic_gap_reduction", 0.0))),
                    -max(0.0, float(projected_lookup[bank.name].get("gap_reduction", 0.0))),
                ),
            )
            return selected, projected_scores

    selected, selection_scores = select_active_source(
        candidates,
        snapshot,
        coverage_balance_penalty,
        gap_focus_margin,
    )
    return selected, {bank.name: float(selection_scores[bank.name]) for bank in candidates}


def select_scout_commit_target(
    shortlist: list[SourceBank],
    scout_rows: list[dict[str, Any]],
    snapshot: dict[str, ProbeGapMetrics],
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    commit_tokens: int,
    scout_commit_tokens: int,
) -> tuple[SourceBank, dict[str, float]]:
    commit_candidates = shortlist
    if scout_rows:
        scouted_names = {str(row["source"]) for row in scout_rows}
        commit_candidates = [bank for bank in shortlist if bank.name in scouted_names]
        if not commit_candidates:
            commit_candidates = shortlist

    projected_rows = [row for row in scout_rows if "projected_final_mean_candidate_gap" in row]
    if projected_rows:
        return select_projected_commit_target(
            commit_candidates,
            projected_rows,
            snapshot,
            coverage_balance_penalty,
            gap_focus_margin,
        )

    selected, selection_scores = select_active_source(
        commit_candidates,
        snapshot,
        coverage_balance_penalty,
        gap_focus_margin,
    )
    if not scout_rows or int(scout_commit_tokens) <= 0 or int(commit_tokens) <= 0:
        return selected, {bank.name: float(selection_scores[bank.name]) for bank in commit_candidates}

    gain_scale = float(commit_tokens) / max(1.0, float(scout_commit_tokens))
    scout_gain = {
        str(row["source"]): max(0.0, float(row["diagnostic_gap_reduction"])) * gain_scale
        for row in scout_rows
    }
    commit_scores = {
        bank.name: float(selection_scores[bank.name] + scout_gain.get(bank.name, 0.0))
        for bank in commit_candidates
    }
    selected = max(commit_candidates, key=lambda bank: commit_scores[bank.name])
    return selected, commit_scores


def semantic_shortlist(
    trainer: HECSNTrainer | None,
    available: list[SourceBank],
    gap_snapshot: dict[str, ProbeGapMetrics],
    shortlist_size: int,
    gap_weight: float,
    affinity_weight: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
) -> tuple[list[SourceBank], dict[str, float]]:
    if shortlist_size <= 0 or shortlist_size >= len(available):
        _, selection_scores = select_active_source(
            available,
            gap_snapshot,
            coverage_balance_penalty,
            gap_focus_margin,
        )
        ranked = sorted(available, key=lambda bank: float(selection_scores[bank.name]), reverse=True)
        return ranked, {bank.name: float(selection_scores[bank.name]) for bank in ranked}

    context_signature = current_context_signature(trainer)
    signatures = {bank.name: candidate_semantic_signature(trainer, bank) for bank in available}
    frontier_plan = frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)
    gap_values = [float(selection_metric(gap_snapshot[bank.name])) for bank in available]
    gap_min = min(gap_values)
    gap_max = max(gap_values)
    scores: dict[str, float] = {}
    for bank in available:
        gap_score = float(selection_metric(gap_snapshot[bank.name]))
        if gap_max - gap_min <= 1e-8:
            gap_norm = 0.0
        else:
            gap_norm = (gap_score - gap_min) / (gap_max - gap_min)
        bank_signature = signatures[bank.name]
        context_affinity = 0.0
        if context_signature is not None and bank_signature is not None:
            context_affinity = float(torch.dot(context_signature, bank_signature).item())
        context_affinity_norm = max(0.0, 0.5 * (context_affinity + 1.0))
        planner_relevance = bank_semantic_relevance_score(bank, frontier_plan)
        if frontier_plan.get("gap_terms") and context_signature is not None:
            affinity_norm = 0.5 * context_affinity_norm + 0.5 * planner_relevance
        elif frontier_plan.get("gap_terms"):
            affinity_norm = planner_relevance
        else:
            affinity_norm = context_affinity_norm
        scores[bank.name] = float(gap_weight * gap_norm + affinity_weight * affinity_norm)

    ranked = sorted(available, key=lambda bank: scores[bank.name], reverse=True)
    return ranked[:shortlist_size], scores


def initialize_acquisition_trainer(
    seed_banks: list[SourceBank],
    candidate_banks: list[SourceBank],
    cfg: Any,
    seed: int,
    seed_train_tokens: int,
    warmup_rounds: int,
) -> tuple[HECSNTrainer, list[SourceBank], list[dict[str, Any]]]:
    set_seed(seed)
    model = HECSNModelLite(cfg)
    trainer = HECSNTrainer(model, cfg)
    seed_bank_state = deepcopy(seed_banks)
    candidate_state = deepcopy(candidate_banks)
    metrics_rows: list[dict[str, Any]] = []

    for _ in range(max(0, int(warmup_rounds))):
        for bank in seed_bank_state:
            train_source_chunk(trainer, bank, seed_train_tokens, "seed_warmup", metrics_rows)

    return trainer, candidate_state, metrics_rows


def build_candidate_discovery_plan(
    seed_banks: list[SourceBank],
    cfg: Any,
    seed: int,
    seed_train_tokens: int,
    warmup_rounds: int,
) -> dict[str, Any]:
    trainer, _, _ = initialize_acquisition_trainer(
        seed_banks=seed_banks,
        candidate_banks=[],
        cfg=cfg,
        seed=seed,
        seed_train_tokens=seed_train_tokens,
        warmup_rounds=warmup_rounds,
    )
    return frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)


def _merge_exclusions(*groups: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for group in groups:
        for raw_value in group:
            value = str(raw_value).strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def refresh_candidate_catalog_state(
    *,
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    candidate_bank_specs: list[dict[str, Any]],
    candidate_train_tokens: int,
    probe_tokens: int,
    excluded_names: list[str] | None = None,
    excluded_sources: list[str] | None = None,
) -> tuple[list[SourceBank], dict[str, Any]]:
    candidate_discovery_plan = frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)
    resolved_specs: list[dict[str, Any]] = []
    for raw_spec in candidate_bank_specs:
        spec = dict(raw_spec)
        if spec.get("catalog_mode"):
            spec["catalog_exclude_names"] = _merge_exclusions(
                list(spec.get("catalog_exclude_names") or []),
                list(excluded_names or []),
            )
            spec["catalog_exclude_sources"] = _merge_exclusions(
                list(spec.get("catalog_exclude_sources") or []),
                list(excluded_sources or []),
            )
        resolved_specs.append(spec)

    candidate_state = load_source_banks(
        resolved_specs,
        encoder,
        trainer.config.window_size,
        probe_tokens,
        candidate_train_tokens,
        semantic_plan=candidate_discovery_plan,
    )
    return candidate_state, candidate_discovery_plan


def build_acquisition_history_row(
    *,
    slot_idx: int,
    selected: SourceBank,
    tokens_trained: int,
    selection_score: float,
    gap_before: float,
    gap_after: float,
    diagnostic_gap_before: float,
    diagnostic_gap_after: float,
    snapshot: dict[str, ProbeGapMetrics],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "slot": slot_idx + 1,
        "selected_source": selected.name,
        "tokens_trained": int(tokens_trained),
        "selected_gap_before": gap_before,
        "selected_gap_after": gap_after,
        "selected_gap_reduction": gap_before - gap_after,
        "selected_selection_score": float(selection_score),
        "selected_diagnostic_gap_before": diagnostic_gap_before,
        "selected_diagnostic_gap_after": diagnostic_gap_after,
        "selected_diagnostic_gap_reduction": diagnostic_gap_before - diagnostic_gap_after,
        "candidate_snapshot": serialize_candidate_snapshot(snapshot),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def select_projected_active_source(
    trainer: HECSNTrainer,
    available: list[SourceBank],
    projection_candidates: list[SourceBank],
    snapshot: dict[str, ProbeGapMetrics],
    acquisition_tokens: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
) -> tuple[SourceBank, dict[str, float], list[dict[str, Any]], dict[str, list[tuple[str, torch.Tensor]]]]:
    projected_chunks = {
        bank.name: preview_source_chunk(bank, acquisition_tokens)
        for bank in available
    }
    projected_rows = evaluate_projected_candidates(
        trainer,
        available,
        projected_chunks,
        projection_candidates,
        gap_exploration_bonus,
        gap_ambiguity_weight,
        gap_switch_weight,
        gap_margin_reference,
        snapshot=snapshot,
    )

    selected, selection_scores = select_projected_commit_target(
        available,
        projected_rows,
        snapshot,
        coverage_balance_penalty,
        gap_focus_margin,
    )
    return selected, selection_scores, projected_rows, projected_chunks


def execute_acquisition_policy(
    *,
    trainer: HECSNTrainer,
    candidate_state: list[SourceBank],
    candidate_bank_specs: list[dict[str, Any]] | None = None,
    encoder: RTFEncoder | None = None,
    candidate_train_tokens: int = 0,
    probe_tokens: int = 0,
    metrics_rows: list[dict[str, Any]],
    policy_name: str,
    acquisition_tokens: int,
    acquisition_slots: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    acquisition_phase: str,
    scout_phase: str = "scout",
    commit_phase: str = "commit",
    scout_commit_tokens: int = 0,
    scout_top_k: int = 1,
    semantic_shortlist_size: int = 0,
    semantic_shortlist_gap_weight: float = 0.5,
    semantic_shortlist_affinity_weight: float = 0.5,
) -> dict[str, Any]:
    if policy_name not in {"active", "round_robin", "scout_commit"}:
        raise ValueError(f"Unknown acquisition policy: {policy_name}")
    if policy_name == "scout_commit" and int(scout_commit_tokens) <= 0:
        raise ValueError("scout_commit_tokens must be positive when policy=scout_commit")

    acquisition_history: list[dict[str, Any]] = []
    rr_index = 0
    token_count_before = int(trainer.token_count)
    dynamic_catalog = bool(candidate_bank_specs) and any(spec.get("catalog_mode") for spec in candidate_bank_specs or [])
    if dynamic_catalog and encoder is None:
        raise ValueError("encoder is required when candidate_bank_specs use catalog_mode")
    current_candidate_state = candidate_state
    acquired_names: list[str] = []
    acquired_sources: list[str] = []
    candidate_discovery_history: list[dict[str, Any]] = []
    discovered_candidate_sources: dict[str, dict[str, Any]] = {
        str(item["name"]): item
        for item in describe_source_banks(current_candidate_state)
    }
    last_candidate_discovery_plan: dict[str, Any] | None = None

    for slot_idx in range(max(0, int(acquisition_slots))):
        current_discovery_plan: dict[str, Any] | None = None
        if dynamic_catalog:
            current_candidate_state, current_discovery_plan = refresh_candidate_catalog_state(
                trainer=trainer,
                encoder=encoder,
                candidate_bank_specs=list(candidate_bank_specs or []),
                candidate_train_tokens=int(candidate_train_tokens),
                probe_tokens=int(probe_tokens),
                excluded_names=list(acquired_names),
                excluded_sources=list(acquired_sources),
            )
            last_candidate_discovery_plan = current_discovery_plan
            discovery_sources = describe_source_banks(current_candidate_state)
            candidate_discovery_history.append(
                {
                    "slot": slot_idx + 1,
                    "plan": current_discovery_plan,
                    "candidate_sources": discovery_sources,
                }
            )
            for item in discovery_sources:
                discovered_candidate_sources[str(item["name"])] = item

        available = [bank for bank in current_candidate_state if bank.remaining() > 0]
        if not available:
            break

        snapshot = candidate_gap_snapshot(
            trainer,
            available,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        shortlist = available
        shortlist_scores: dict[str, float] = {}
        if (
            policy_name == "active"
            and int(semantic_shortlist_size) > 0
            and len(available) > 1
            and any(_candidate_has_semantic_signal(bank) for bank in available)
        ):
            shortlist, shortlist_scores = semantic_shortlist(
                trainer,
                available,
                snapshot,
                int(semantic_shortlist_size),
                float(semantic_shortlist_gap_weight),
                float(semantic_shortlist_affinity_weight),
                coverage_balance_penalty,
                gap_focus_margin,
            )
            if not shortlist:
                shortlist = available

        if policy_name == "active":
            selected, selection_scores, projected_rows, projected_chunks = select_projected_active_source(
                trainer,
                shortlist,
                current_candidate_state,
                snapshot,
                acquisition_tokens,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
                coverage_balance_penalty,
                gap_focus_margin,
            )
            gap_before = float(snapshot[selected.name]["recon_error"])
            diagnostic_gap_before = float(snapshot[selected.name]["diagnostic_gap_score"])
            preview_chunk = projected_chunks.get(selected.name, [])
            if preview_chunk:
                committed_chunk = consume_previewed_chunk(selected, preview_chunk)
                tokens_trained = replay_source_chunk(trainer, selected, committed_chunk, acquisition_phase, metrics_rows)
            else:
                tokens_trained = train_source_chunk(trainer, selected, acquisition_tokens, acquisition_phase, metrics_rows)
            after = probe_gap(
                trainer,
                selected,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )
            acquisition_history.append(
                build_acquisition_history_row(
                    slot_idx=slot_idx,
                    selected=selected,
                    tokens_trained=tokens_trained,
                    selection_score=float(selection_scores[selected.name]),
                    gap_before=gap_before,
                    gap_after=float(after["recon_error"]),
                    diagnostic_gap_before=diagnostic_gap_before,
                    diagnostic_gap_after=float(after["diagnostic_gap_score"]),
                    snapshot=snapshot,
                    extra_fields={
                        "candidate_shortlist": [bank.name for bank in shortlist],
                        "candidate_shortlist_scores": {name: float(score) for name, score in shortlist_scores.items()},
                        "selection_actions": projected_rows,
                        "candidate_discovery_plan": current_discovery_plan,
                        "candidate_catalog_candidates": [bank.name for bank in current_candidate_state],
                    },
                )
            )
            if dynamic_catalog:
                acquired_names.append(selected.name)
                acquired_sources.append(selected.source)
            continue

        if policy_name == "round_robin":
            selected, rr_index = next_candidate_round_robin(current_candidate_state, rr_index)
            gap_before = float(snapshot[selected.name]["recon_error"])
            diagnostic_gap_before = float(snapshot[selected.name]["diagnostic_gap_score"])
            selection_scores = {bank.name: float(selection_metric(snapshot[bank.name])) for bank in available}
            tokens_trained = train_source_chunk(trainer, selected, acquisition_tokens, acquisition_phase, metrics_rows)
            after = probe_gap(
                trainer,
                selected,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )
            acquisition_history.append(
                build_acquisition_history_row(
                    slot_idx=slot_idx,
                    selected=selected,
                    tokens_trained=tokens_trained,
                    selection_score=float(selection_scores[selected.name]),
                    gap_before=gap_before,
                    gap_after=float(after["recon_error"]),
                    diagnostic_gap_before=diagnostic_gap_before,
                    diagnostic_gap_after=float(after["diagnostic_gap_score"]),
                    snapshot=snapshot,
                    extra_fields={
                        "candidate_discovery_plan": current_discovery_plan,
                        "candidate_catalog_candidates": [bank.name for bank in current_candidate_state],
                    },
                )
            )
            if dynamic_catalog:
                acquired_names.append(selected.name)
                acquired_sources.append(selected.source)
            continue

        frontier_plan = frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)
        shortlist, shortlist_scores = semantic_shortlist(
            trainer,
            available,
            snapshot,
            int(semantic_shortlist_size),
            float(semantic_shortlist_gap_weight),
            float(semantic_shortlist_affinity_weight),
            coverage_balance_penalty,
            gap_focus_margin,
        )
        scout_chunks: dict[str, list[tuple[str, torch.Tensor]]] = {}
        spent_tokens = 0
        scout_rows: list[dict[str, Any]] = []
        scout_candidates = shortlist[: min(max(1, scout_top_k), len(shortlist))]

        if int(scout_top_k) == 1:
            scout_candidates = shortlist if int(semantic_shortlist_size) > 0 else available
            scout_followup_tokens: dict[str, int] = {}
            for bank in scout_candidates:
                scout_tokens = min(int(scout_commit_tokens), bank.remaining(), int(acquisition_tokens))
                if scout_tokens <= 0:
                    continue
                scout_chunk = preview_source_chunk(bank, scout_tokens)
                if not scout_chunk:
                    continue
                scout_chunks[bank.name] = scout_chunk
                scout_followup_tokens[bank.name] = max(0, int(acquisition_tokens) - len(scout_chunk))

            scout_rows = evaluate_projected_candidates(
                trainer,
                scout_candidates,
                scout_chunks,
                current_candidate_state,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
                followup_tokens_by_source=scout_followup_tokens,
                snapshot=snapshot,
            )
            commit_target, commit_scores = select_scout_commit_target(
                shortlist=scout_candidates,
                scout_rows=scout_rows,
                snapshot=snapshot,
                coverage_balance_penalty=coverage_balance_penalty,
                gap_focus_margin=gap_focus_margin,
                commit_tokens=max(scout_followup_tokens.values(), default=max(0, int(acquisition_tokens) - int(scout_commit_tokens))),
                scout_commit_tokens=int(scout_commit_tokens),
            )
            ranked_candidates = [bank for bank in scout_candidates if bank.name in commit_scores]
            if ranked_candidates:
                shortlist = sorted(ranked_candidates, key=lambda bank: float(commit_scores[bank.name]), reverse=True)
                shortlist_scores = {bank.name: float(commit_scores[bank.name]) for bank in shortlist}
            spent_tokens = len(scout_chunks.get(commit_target.name, []))
        else:
            for bank in scout_candidates:
                remaining_budget = int(acquisition_tokens) - spent_tokens
                if remaining_budget <= 0:
                    break
                scout_tokens = min(int(scout_commit_tokens), bank.remaining(), remaining_budget)
                if scout_tokens <= 0:
                    continue
                scout_chunk = preview_source_chunk(bank, scout_tokens)
                if not scout_chunk:
                    continue
                spent_tokens += len(scout_chunk)
                scout_chunks[bank.name] = scout_chunk

            commit_tokens = max(0, int(acquisition_tokens) - spent_tokens)
            scout_rows = evaluate_projected_candidates(
                trainer,
                scout_candidates,
                scout_chunks,
                current_candidate_state,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
                followup_tokens=commit_tokens,
                snapshot=snapshot,
            )

            commit_target, commit_scores = select_scout_commit_target(
                shortlist=scout_candidates,
                scout_rows=scout_rows,
                snapshot=snapshot,
                coverage_balance_penalty=coverage_balance_penalty,
                gap_focus_margin=gap_focus_margin,
                commit_tokens=commit_tokens,
                scout_commit_tokens=int(scout_commit_tokens),
            )
        committed_scout_chunk = consume_previewed_chunk(commit_target, scout_chunks.get(commit_target.name, []))
        replay_source_chunk(trainer, commit_target, committed_scout_chunk, scout_phase, metrics_rows)
        post_scout_snapshot = candidate_gap_snapshot(
            trainer,
            current_candidate_state,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        commit_gap_before = float(post_scout_snapshot[commit_target.name]["recon_error"])
        commit_diagnostic_gap_before = float(post_scout_snapshot[commit_target.name]["diagnostic_gap_score"])
        commit_tokens = max(0, int(acquisition_tokens) - spent_tokens)
        commit_trained = train_source_chunk(trainer, commit_target, commit_tokens, commit_phase, metrics_rows)
        commit_after = probe_gap(
            trainer,
            commit_target,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        acquisition_history.append(
            build_acquisition_history_row(
                slot_idx=slot_idx,
                selected=commit_target,
                tokens_trained=int(spent_tokens + commit_trained),
                selection_score=float(commit_scores[commit_target.name]),
                gap_before=commit_gap_before,
                gap_after=float(commit_after["recon_error"]),
                diagnostic_gap_before=commit_diagnostic_gap_before,
                diagnostic_gap_after=float(commit_after["diagnostic_gap_score"]),
                snapshot=post_scout_snapshot,
                extra_fields={
                    "semantic_gap_plan": frontier_plan,
                    "semantic_shortlist": [bank.name for bank in shortlist],
                    "semantic_shortlist_scores": {name: float(score) for name, score in shortlist_scores.items()},
                    "scout_actions": scout_rows,
                    "commit_tokens": int(commit_trained),
                    "scout_budget_spent": int(spent_tokens),
                    "candidate_discovery_plan": current_discovery_plan,
                    "candidate_catalog_candidates": [bank.name for bank in current_candidate_state],
                },
            )
        )
        if dynamic_catalog:
            acquired_names.append(commit_target.name)
            acquired_sources.append(commit_target.source)

    if dynamic_catalog:
        try:
            current_candidate_state, last_candidate_discovery_plan = refresh_candidate_catalog_state(
                trainer=trainer,
                encoder=encoder,
                candidate_bank_specs=list(candidate_bank_specs or []),
                candidate_train_tokens=int(candidate_train_tokens),
                probe_tokens=int(probe_tokens),
                excluded_names=list(acquired_names),
                excluded_sources=list(acquired_sources),
            )
        except ValueError:
            current_candidate_state = []

    final_snapshot = (
        candidate_gap_snapshot(
            trainer,
            current_candidate_state,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        if current_candidate_state
        else {}
    )
    tokens_trained_total = sum(int(row["tokens_trained"]) for row in acquisition_history)
    result = {
        "policy": policy_name,
        "candidate_sources": (
            describe_source_banks(current_candidate_state)
            if not dynamic_catalog
            else list(discovered_candidate_sources.values())
        ),
        "acquired_sources": [row["selected_source"] for row in acquisition_history],
        "acquisition_history": acquisition_history,
        "candidate_discovery_history": candidate_discovery_history,
        "candidate_discovery_plan": last_candidate_discovery_plan,
        "training_diagnostics": summarize_training_metrics(metrics_rows),
        "runtime_scope": trainer.model.runtime_scope_report(),
        "token_count_before": token_count_before,
        "token_count_after": int(trainer.token_count),
        "tokens_trained_total": int(tokens_trained_total),
        "scout_commit_tokens": int(scout_commit_tokens),
        "scout_top_k": int(scout_top_k),
        "semantic_shortlist_size": int(semantic_shortlist_size),
        "semantic_shortlist_gap_weight": float(semantic_shortlist_gap_weight),
        "semantic_shortlist_affinity_weight": float(semantic_shortlist_affinity_weight),
    }
    result.update(summarize_candidate_frontier(final_snapshot))
    return result


def run_acquisition_policy(
    policy_name: str,
    seed_banks: list[SourceBank],
    candidate_banks: list[SourceBank],
    candidate_bank_specs: list[dict[str, Any]],
    encoder: RTFEncoder,
    candidate_train_tokens: int,
    probe_tokens: int,
    cfg: Any,
    seed: int,
    seed_train_tokens: int,
    acquisition_tokens: int,
    warmup_rounds: int,
    acquisition_slots: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    scout_commit_tokens: int = 0,
    scout_top_k: int = 1,
    semantic_shortlist_size: int = 0,
    semantic_shortlist_gap_weight: float = 0.5,
    semantic_shortlist_affinity_weight: float = 0.5,
) -> tuple[dict[str, Any], HECSNTrainer]:
    trainer, candidate_state, metrics_rows = initialize_acquisition_trainer(
        seed_banks,
        candidate_banks,
        cfg,
        seed,
        seed_train_tokens,
        warmup_rounds,
    )
    result = execute_acquisition_policy(
        trainer=trainer,
        candidate_state=candidate_state,
        candidate_bank_specs=candidate_bank_specs,
        encoder=encoder,
        candidate_train_tokens=candidate_train_tokens,
        probe_tokens=probe_tokens,
        metrics_rows=metrics_rows,
        policy_name=policy_name,
        acquisition_tokens=acquisition_tokens,
        acquisition_slots=acquisition_slots,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
        acquisition_phase="acquisition",
        scout_phase="scout",
        commit_phase="commit",
        scout_commit_tokens=scout_commit_tokens,
        scout_top_k=scout_top_k,
        semantic_shortlist_size=semantic_shortlist_size,
        semantic_shortlist_gap_weight=semantic_shortlist_gap_weight,
        semantic_shortlist_affinity_weight=semantic_shortlist_affinity_weight,
    )
    return result, trainer


def run_live_acquisition(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    *,
    candidate_bank_specs: list[dict[str, Any]],
    candidate_train_tokens: int,
    probe_tokens: int,
    acquisition_tokens: int,
    acquisition_slots: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    policy_name: str = "scout_commit",
    scout_commit_tokens: int = 0,
    scout_top_k: int = 1,
    semantic_shortlist_size: int = 0,
    semantic_shortlist_gap_weight: float = 0.5,
    semantic_shortlist_affinity_weight: float = 0.5,
) -> dict[str, Any]:
    candidate_discovery_plan = frontier_semantic_plan(trainer, max_terms=8, max_queries=4, max_questions=4)
    candidate_state = load_source_banks(
        candidate_bank_specs,
        encoder,
        trainer.config.window_size,
        probe_tokens,
        candidate_train_tokens,
        semantic_plan=candidate_discovery_plan,
    )
    result = execute_acquisition_policy(
        trainer=trainer,
        candidate_state=candidate_state,
        candidate_bank_specs=list(candidate_bank_specs),
        encoder=encoder,
        candidate_train_tokens=candidate_train_tokens,
        probe_tokens=probe_tokens,
        metrics_rows=[],
        policy_name=policy_name,
        acquisition_tokens=acquisition_tokens,
        acquisition_slots=acquisition_slots,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
        acquisition_phase="service_acquisition",
        scout_phase="service_scout",
        commit_phase="service_commit",
        scout_commit_tokens=scout_commit_tokens,
        scout_top_k=scout_top_k,
        semantic_shortlist_size=semantic_shortlist_size,
        semantic_shortlist_gap_weight=semantic_shortlist_gap_weight,
        semantic_shortlist_affinity_weight=semantic_shortlist_affinity_weight,
    )
    result.setdefault("candidate_discovery_plan", candidate_discovery_plan)
    result["initial_candidate_discovery_plan"] = candidate_discovery_plan
    return result


def run_acquisition_benchmark(
    seed_bank: list[dict[str, Any]],
    candidate_bank: list[dict[str, Any]],
    seed_train_tokens: int,
    candidate_train_tokens: int,
    probe_tokens: int,
    acquisition_tokens: int,
    warmup_rounds: int,
    acquisition_slots: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    scout_commit_tokens: int,
    scout_top_k: int,
    semantic_shortlist_size: int,
    semantic_shortlist_gap_weight: float,
    semantic_shortlist_affinity_weight: float,
    output_dir: Path,
    checkpoint_out: Path | None,
    save_plots: bool,
    **kwargs: Any,
) -> None:
    cfg = make_config(kwargs)
    encoder = RTFEncoder.from_config(cfg)
    seed_banks = load_source_banks(seed_bank, encoder, cfg.window_size, probe_tokens, seed_train_tokens)
    candidate_discovery_plan = build_candidate_discovery_plan(
        seed_banks=seed_banks,
        cfg=cfg,
        seed=int(kwargs["seed"]),
        seed_train_tokens=seed_train_tokens,
        warmup_rounds=warmup_rounds,
    )
    candidate_banks = load_source_banks(
        candidate_bank,
        encoder,
        cfg.window_size,
        probe_tokens,
        candidate_train_tokens,
        semantic_plan=candidate_discovery_plan,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    active, active_trainer = run_acquisition_policy(
        policy_name="active",
        seed_banks=seed_banks,
        candidate_banks=candidate_banks,
        candidate_bank_specs=list(candidate_bank),
        encoder=encoder,
        candidate_train_tokens=candidate_train_tokens,
        probe_tokens=probe_tokens,
        cfg=cfg,
        seed=int(kwargs["seed"]),
        seed_train_tokens=seed_train_tokens,
        acquisition_tokens=acquisition_tokens,
        warmup_rounds=warmup_rounds,
        acquisition_slots=acquisition_slots,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
        semantic_shortlist_size=semantic_shortlist_size,
        semantic_shortlist_gap_weight=semantic_shortlist_gap_weight,
        semantic_shortlist_affinity_weight=semantic_shortlist_affinity_weight,
    )
    round_robin, _ = run_acquisition_policy(
        policy_name="round_robin",
        seed_banks=seed_banks,
        candidate_banks=candidate_banks,
        candidate_bank_specs=list(candidate_bank),
        encoder=encoder,
        candidate_train_tokens=candidate_train_tokens,
        probe_tokens=probe_tokens,
        cfg=cfg,
        seed=int(kwargs["seed"]),
        seed_train_tokens=seed_train_tokens,
        acquisition_tokens=acquisition_tokens,
        warmup_rounds=warmup_rounds,
        acquisition_slots=acquisition_slots,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
        semantic_shortlist_size=semantic_shortlist_size,
        semantic_shortlist_gap_weight=semantic_shortlist_gap_weight,
        semantic_shortlist_affinity_weight=semantic_shortlist_affinity_weight,
    )
    scout_commit = None
    checkpoint_trainer = active_trainer
    checkpoint_policy = "active"
    if int(scout_commit_tokens) > 0:
        scout_commit, scout_commit_trainer = run_acquisition_policy(
            policy_name="scout_commit",
            seed_banks=seed_banks,
            candidate_banks=candidate_banks,
            candidate_bank_specs=list(candidate_bank),
            encoder=encoder,
            candidate_train_tokens=candidate_train_tokens,
            probe_tokens=probe_tokens,
            cfg=cfg,
            seed=int(kwargs["seed"]),
            seed_train_tokens=seed_train_tokens,
            acquisition_tokens=acquisition_tokens,
            warmup_rounds=warmup_rounds,
            acquisition_slots=acquisition_slots,
            gap_exploration_bonus=gap_exploration_bonus,
            gap_ambiguity_weight=gap_ambiguity_weight,
            gap_switch_weight=gap_switch_weight,
            gap_margin_reference=gap_margin_reference,
            coverage_balance_penalty=coverage_balance_penalty,
            gap_focus_margin=gap_focus_margin,
            scout_commit_tokens=scout_commit_tokens,
            scout_top_k=scout_top_k,
            semantic_shortlist_size=semantic_shortlist_size,
            semantic_shortlist_gap_weight=semantic_shortlist_gap_weight,
            semantic_shortlist_affinity_weight=semantic_shortlist_affinity_weight,
        )
        checkpoint_trainer = scout_commit_trainer
        checkpoint_policy = "scout_commit"

    comparison = {
        "active_final_mean_candidate_gap": float(active["final_mean_candidate_gap"]),
        "round_robin_final_mean_candidate_gap": float(round_robin["final_mean_candidate_gap"]),
        "active_final_max_candidate_gap": float(active["final_max_candidate_gap"]),
        "round_robin_final_max_candidate_gap": float(round_robin["final_max_candidate_gap"]),
        "active_final_mean_candidate_diagnostic_gap": float(active["final_mean_candidate_diagnostic_gap"]),
        "round_robin_final_mean_candidate_diagnostic_gap": float(round_robin["final_mean_candidate_diagnostic_gap"]),
        "active_minus_round_robin_mean_candidate_gap": float(active["final_mean_candidate_gap"] - round_robin["final_mean_candidate_gap"]),
        "active_minus_round_robin_max_candidate_gap": float(active["final_max_candidate_gap"] - round_robin["final_max_candidate_gap"]),
        "active_minus_round_robin_mean_candidate_diagnostic_gap": float(
            active["final_mean_candidate_diagnostic_gap"] - round_robin["final_mean_candidate_diagnostic_gap"]
        ),
    }
    if scout_commit is not None:
        comparison.update(
            {
                "scout_commit_final_mean_candidate_gap": float(scout_commit["final_mean_candidate_gap"]),
                "scout_commit_final_max_candidate_gap": float(scout_commit["final_max_candidate_gap"]),
                "scout_commit_final_mean_candidate_diagnostic_gap": float(scout_commit["final_mean_candidate_diagnostic_gap"]),
                "scout_commit_minus_round_robin_mean_candidate_gap": float(
                    scout_commit["final_mean_candidate_gap"] - round_robin["final_mean_candidate_gap"]
                ),
                "scout_commit_minus_round_robin_max_candidate_gap": float(
                    scout_commit["final_max_candidate_gap"] - round_robin["final_max_candidate_gap"]
                ),
                "scout_commit_minus_active_mean_candidate_gap": float(
                    scout_commit["final_mean_candidate_gap"] - active["final_mean_candidate_gap"]
                ),
                "scout_commit_minus_active_max_candidate_gap": float(
                    scout_commit["final_max_candidate_gap"] - active["final_max_candidate_gap"]
                ),
            }
        )
    acquisition_gate = acquisition_gate_from_comparison(comparison)
    scout_commit_gate = None
    if scout_commit is not None:
        scout_rr_mean_target = acquisition_candidate_improvement_target(comparison["round_robin_final_mean_candidate_gap"])
        scout_rr_max_target = acquisition_candidate_improvement_target(comparison["round_robin_final_max_candidate_gap"])
        scout_active_mean_target = acquisition_candidate_improvement_target(comparison["active_final_mean_candidate_gap"])
        scout_active_max_target = acquisition_candidate_improvement_target(comparison["active_final_max_candidate_gap"])
        scout_commit_gate = {
            "pass": bool(
                comparison["scout_commit_minus_round_robin_mean_candidate_gap"] <= -scout_rr_mean_target
                and comparison["scout_commit_minus_round_robin_max_candidate_gap"] <= -scout_rr_max_target
                and comparison["scout_commit_minus_active_mean_candidate_gap"] <= -scout_active_mean_target
                and comparison["scout_commit_minus_active_max_candidate_gap"] <= -scout_active_max_target
            ),
            "scout_commit_mean_candidate_gap_better_than_round_robin": bool(
                comparison["scout_commit_minus_round_robin_mean_candidate_gap"] <= -scout_rr_mean_target
            ),
            "scout_commit_max_candidate_gap_better_than_round_robin": bool(
                comparison["scout_commit_minus_round_robin_max_candidate_gap"] <= -scout_rr_max_target
            ),
            "scout_commit_mean_candidate_gap_better_than_active": bool(
                comparison["scout_commit_minus_active_mean_candidate_gap"] <= -scout_active_mean_target
            ),
            "scout_commit_max_candidate_gap_better_than_active": bool(
                comparison["scout_commit_minus_active_max_candidate_gap"] <= -scout_active_max_target
            ),
            "thresholds": {
                "absolute_improvement_cap": ACQUISITION_ABSOLUTE_IMPROVEMENT_TARGET,
                "relative_improvement_fraction": ACQUISITION_RELATIVE_IMPROVEMENT_FRACTION,
                "scout_commit_minus_round_robin_mean_candidate_gap_max": -scout_rr_mean_target,
                "scout_commit_minus_round_robin_max_candidate_gap_max": -scout_rr_max_target,
                "scout_commit_minus_active_mean_candidate_gap_max": -scout_active_mean_target,
                "scout_commit_minus_active_max_candidate_gap_max": -scout_active_max_target,
            },
        }

    summary = {
        "protocol": source_bank_protocol("autonomy_source_acquisition", [*seed_banks, *candidate_banks]),
        "seed_names": [bank.name for bank in seed_banks],
        "candidate_names": [bank.name for bank in candidate_banks],
        "seed_source_types": sorted({bank.source_type for bank in seed_banks}),
        "candidate_source_types": sorted({bank.source_type for bank in candidate_banks}),
        "seed_bank": describe_source_banks(seed_banks),
        "candidate_bank": describe_source_banks(candidate_banks),
        "candidate_discovery_plan": candidate_discovery_plan,
        "acquisition_setup": {
            "seed_train_tokens": int(seed_train_tokens),
            "candidate_train_tokens": int(candidate_train_tokens),
            "probe_tokens": int(probe_tokens),
            "acquisition_tokens": int(acquisition_tokens),
            "warmup_rounds": int(warmup_rounds),
            "acquisition_slots": int(acquisition_slots),
            "gap_exploration_bonus": float(gap_exploration_bonus),
            "gap_ambiguity_weight": float(gap_ambiguity_weight),
            "gap_switch_weight": float(gap_switch_weight),
            "gap_margin_reference": float(gap_margin_reference),
            "coverage_balance_penalty": float(coverage_balance_penalty),
            "gap_focus_margin": float(gap_focus_margin),
            "scout_commit_tokens": int(scout_commit_tokens),
            "scout_top_k": int(scout_top_k),
            "semantic_shortlist_size": int(semantic_shortlist_size),
            "semantic_shortlist_gap_weight": float(semantic_shortlist_gap_weight),
            "semantic_shortlist_affinity_weight": float(semantic_shortlist_affinity_weight),
        },
        "policy_results": {
            "active": active,
            "round_robin": round_robin,
        },
        "comparison": comparison,
        "acquisition_gate": acquisition_gate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if scout_commit is not None:
        summary["policy_results"]["scout_commit"] = scout_commit
        summary["scout_commit_gate"] = scout_commit_gate

    checkpoint_path: Path | None = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            checkpoint_trainer,
            metadata={
                "protocol": source_bank_protocol("autonomy_source_acquisition", [*seed_banks, *candidate_banks]),
                "exported_policy": checkpoint_policy,
                "seed_names": [bank.name for bank in seed_banks],
                "candidate_names": [bank.name for bank in candidate_banks],
                "seed_source_types": sorted({bank.source_type for bank in seed_banks}),
                "candidate_source_types": sorted({bank.source_type for bank in candidate_banks}),
                "seed_train_tokens": int(seed_train_tokens),
                "candidate_train_tokens": int(candidate_train_tokens),
                "probe_tokens": int(probe_tokens),
                "acquisition_tokens": int(acquisition_tokens),
                "acquisition_slots": int(acquisition_slots),
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)
    write_json_file(output_dir / "summary.json", summary)
    if save_plots:
        plot_acquisition_summary(output_dir, summary)

    print("Autonomy acquisition summary")
    print(f"active_final_mean_candidate_gap={comparison['active_final_mean_candidate_gap']:.6f}")
    print(f"round_robin_final_mean_candidate_gap={comparison['round_robin_final_mean_candidate_gap']:.6f}")
    print(f"active_final_max_candidate_gap={comparison['active_final_max_candidate_gap']:.6f}")
    print(f"round_robin_final_max_candidate_gap={comparison['round_robin_final_max_candidate_gap']:.6f}")
    print(f"acquisition_gate_pass={acquisition_gate['pass']}")
    if scout_commit is not None and scout_commit_gate is not None:
        print(f"scout_commit_final_mean_candidate_gap={comparison['scout_commit_final_mean_candidate_gap']:.6f}")
        print(f"scout_commit_final_max_candidate_gap={comparison['scout_commit_final_max_candidate_gap']:.6f}")
        print(f"scout_commit_gate_pass={scout_commit_gate['pass']}")
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(f"acquisition_plot={output_dir / 'autonomy_acquisition_diagnostics.png'}")


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=autonomy_acquisition_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_autonomy_acquisition_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Run HECSN autonomy acquisition benchmark across source banks")
    parser.add_argument("--preset", choices=autonomy_acquisition_preset_names(), default=preset_args.preset)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed-train-tokens", type=int, default=preset_defaults.get("seed_train_tokens", 3000))
    parser.add_argument("--candidate-train-tokens", type=int, default=preset_defaults.get("candidate_train_tokens", 3000))
    parser.add_argument("--probe-tokens", type=int, default=preset_defaults.get("probe_tokens", 128))
    parser.add_argument("--acquisition-tokens", type=int, default=preset_defaults.get("acquisition_tokens", 1000))
    parser.add_argument("--warmup-rounds", type=int, default=preset_defaults.get("warmup_rounds", 1))
    parser.add_argument("--acquisition-slots", type=int, default=preset_defaults.get("acquisition_slots", 1))
    parser.add_argument("--gap-exploration-bonus", type=float, default=preset_defaults.get("gap_exploration_bonus", 0.02))
    parser.add_argument("--gap-ambiguity-weight", type=float, default=preset_defaults.get("gap_ambiguity_weight", 0.08))
    parser.add_argument("--gap-switch-weight", type=float, default=preset_defaults.get("gap_switch_weight", 0.08))
    parser.add_argument("--gap-margin-reference", type=float, default=preset_defaults.get("gap_margin_reference", 0.12))
    parser.add_argument("--coverage-balance-penalty", type=float, default=preset_defaults.get("coverage_balance_penalty", 0.02))
    parser.add_argument("--gap-focus-margin", type=float, default=preset_defaults.get("gap_focus_margin", 0.02))
    parser.add_argument("--scout-commit-tokens", type=int, default=preset_defaults.get("scout_commit_tokens", 0))
    parser.add_argument("--scout-top-k", type=int, default=preset_defaults.get("scout_top_k", 2))
    parser.add_argument("--semantic-shortlist-size", type=int, default=preset_defaults.get("semantic_shortlist_size", 0))
    parser.add_argument("--semantic-shortlist-gap-weight", type=float, default=preset_defaults.get("semantic_shortlist_gap_weight", 0.5))
    parser.add_argument("--semantic-shortlist-affinity-weight", type=float, default=preset_defaults.get("semantic_shortlist_affinity_weight", 0.5))
    parser.add_argument("--seed", type=int, default=preset_defaults.get("seed", 7))
    parser.add_argument("--n-columns", type=int, default=preset_defaults.get("n_columns", 100))
    parser.add_argument("--column-latent-dim", type=int, default=preset_defaults.get("column_latent_dim", 256))
    parser.add_argument("--memory-capacity", type=int, default=preset_defaults.get("memory_capacity", 1000))
    parser.add_argument("--input-weight-blend", type=float, default=preset_defaults.get("input_weight_blend", 0.02))
    parser.add_argument("--input-synapse-ltp", type=float, default=preset_defaults.get("input_synapse_ltp", 0.02))
    parser.add_argument("--input-synapse-ltd", type=float, default=preset_defaults.get("input_synapse_ltd", 0.01))
    parser.add_argument("--input-weight-row-target", type=float, default=preset_defaults.get("input_weight_row_target", 1.0))
    parser.add_argument("--homeostasis-beta", type=float, default=preset_defaults.get("homeostasis_beta", 0.01))
    parser.add_argument("--homeostasis-lr", type=float, default=preset_defaults.get("homeostasis_lr", 0.2))
    parser.add_argument("--slow-mean-decay", type=float, default=preset_defaults.get("slow_mean_decay", 0.9999))
    parser.add_argument("--use-winner-local-drift", action="store_true", default=bool(preset_defaults.get("use_winner_local_drift", True)))
    parser.add_argument("--no-winner-local-drift", action="store_true")
    parser.add_argument("--drift-threshold", type=float, default=preset_defaults.get("drift_threshold", 0.02))
    parser.add_argument("--micro-sleep-interval-tokens", type=int, default=preset_defaults.get("micro_sleep_interval_tokens", 200))
    parser.add_argument("--micro-sleep-replay-steps", type=int, default=preset_defaults.get("micro_sleep_replay_steps", 10))
    parser.add_argument("--micro-sleep-candidate-pool", type=int, default=preset_defaults.get("micro_sleep_candidate_pool", 5))
    parser.add_argument("--micro-sleep-memory-blend", type=float, default=preset_defaults.get("micro_sleep_memory_blend", 0.05))
    parser.add_argument("--deep-sleep-interval-tokens", type=int, default=preset_defaults.get("deep_sleep_interval_tokens", 2500))
    parser.add_argument("--deep-sleep-replay-steps", type=int, default=preset_defaults.get("deep_sleep_replay_steps", 200))
    parser.add_argument("--deep-sleep-candidate-pool", type=int, default=preset_defaults.get("deep_sleep_candidate_pool", 100))
    parser.add_argument("--deep-sleep-memory-blend", type=float, default=preset_defaults.get("deep_sleep_memory_blend", 0.20))
    parser.add_argument("--deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--emergency-deep-sleep-cooldown-tokens", type=int, default=preset_defaults.get("emergency_deep_sleep_cooldown_tokens", 1000))
    parser.add_argument("--drift-floor-history-tokens", type=int, default=preset_defaults.get("drift_floor_history_tokens", 1000))
    parser.add_argument("--drift-floor-check-interval-tokens", type=int, default=preset_defaults.get("drift_floor_check_interval_tokens", 200))
    parser.add_argument("--drift-floor-window-tokens", type=int, default=preset_defaults.get("drift_floor_window_tokens", 10000))
    parser.add_argument("--drift-floor-trigger-min-tokens", type=int, default=preset_defaults.get("drift_floor_trigger_min_tokens", 1000))
    parser.add_argument("--drift-floor-rise-tolerance", type=float, default=preset_defaults.get("drift_floor_rise_tolerance", 0.0))
    parser.add_argument("--prototype-momentum", type=float, default=preset_defaults.get("prototype_momentum", 0.85))
    parser.add_argument("--context-decay", type=float, default=preset_defaults.get("context_decay", 0.92))
    parser.add_argument("--context-transition-lr", type=float, default=preset_defaults.get("context_transition_lr", 0.05))
    parser.add_argument("--context-modulation-strength", type=float, default=preset_defaults.get("context_modulation_strength", 0.60))
    parser.add_argument("--binding-threshold", type=float, default=preset_defaults.get("binding_threshold", 0.02))
    parser.add_argument("--binding-association-lr", type=float, default=preset_defaults.get("binding_association_lr", 0.20))
    parser.add_argument("--binding-association-decay", type=float, default=preset_defaults.get("binding_association_decay", 0.995))
    parser.add_argument("--binding-gain-strength", type=float, default=preset_defaults.get("binding_gain_strength", 0.80))
    parser.add_argument("--checkpoint-out", type=Path, default=None)
    args = parser.parse_args()

    if args.use_winner_local_drift and args.no_winner_local_drift:
        raise ValueError("Choose at most one of --use-winner-local-drift or --no-winner-local-drift")
    use_winner_local_drift = not args.no_winner_local_drift

    preset_data = get_autonomy_acquisition_preset(args.preset)
    if not preset_data:
        raise ValueError("Acquisition runner currently requires a preset-defined seed and candidate bank")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / (f"{args.preset}_{stamp}" if args.preset else f"autonomy_acquisition_{stamp}")
    else:
        output_dir = args.output_dir

    run_acquisition_benchmark(
        seed_bank=deepcopy(preset_data["seed_bank"]),
        candidate_bank=deepcopy(preset_data["candidate_bank"]),
        seed_train_tokens=args.seed_train_tokens,
        candidate_train_tokens=args.candidate_train_tokens,
        probe_tokens=args.probe_tokens,
        acquisition_tokens=args.acquisition_tokens,
        warmup_rounds=args.warmup_rounds,
        acquisition_slots=args.acquisition_slots,
        gap_exploration_bonus=args.gap_exploration_bonus,
        gap_ambiguity_weight=args.gap_ambiguity_weight,
        gap_switch_weight=args.gap_switch_weight,
        gap_margin_reference=args.gap_margin_reference,
        coverage_balance_penalty=args.coverage_balance_penalty,
        gap_focus_margin=args.gap_focus_margin,
        scout_commit_tokens=args.scout_commit_tokens,
        scout_top_k=args.scout_top_k,
        semantic_shortlist_size=args.semantic_shortlist_size,
        semantic_shortlist_gap_weight=args.semantic_shortlist_gap_weight,
        semantic_shortlist_affinity_weight=args.semantic_shortlist_affinity_weight,
        output_dir=output_dir,
        checkpoint_out=args.checkpoint_out,
        save_plots=not args.no_plots,
        seed=args.seed,
        n_columns=args.n_columns,
        column_latent_dim=args.column_latent_dim,
        memory_capacity=args.memory_capacity,
        input_weight_blend=args.input_weight_blend,
        input_synapse_ltp=args.input_synapse_ltp,
        input_synapse_ltd=args.input_synapse_ltd,
        input_weight_row_target=args.input_weight_row_target,
        homeostasis_beta=args.homeostasis_beta,
        homeostasis_lr=args.homeostasis_lr,
        slow_mean_decay=args.slow_mean_decay,
        use_winner_local_drift=use_winner_local_drift,
        drift_threshold=args.drift_threshold,
        micro_sleep_interval_tokens=args.micro_sleep_interval_tokens,
        micro_sleep_replay_steps=args.micro_sleep_replay_steps,
        micro_sleep_candidate_pool=args.micro_sleep_candidate_pool,
        micro_sleep_memory_blend=args.micro_sleep_memory_blend,
        deep_sleep_interval_tokens=args.deep_sleep_interval_tokens,
        deep_sleep_replay_steps=args.deep_sleep_replay_steps,
        deep_sleep_candidate_pool=args.deep_sleep_candidate_pool,
        deep_sleep_memory_blend=args.deep_sleep_memory_blend,
        deep_sleep_cooldown_tokens=args.deep_sleep_cooldown_tokens,
        emergency_deep_sleep_cooldown_tokens=args.emergency_deep_sleep_cooldown_tokens,
        drift_floor_history_tokens=args.drift_floor_history_tokens,
        drift_floor_check_interval_tokens=args.drift_floor_check_interval_tokens,
        drift_floor_window_tokens=args.drift_floor_window_tokens,
        drift_floor_trigger_min_tokens=args.drift_floor_trigger_min_tokens,
        drift_floor_rise_tolerance=args.drift_floor_rise_tolerance,
        prototype_momentum=args.prototype_momentum,
        context_decay=args.context_decay,
        context_transition_lr=args.context_transition_lr,
        context_modulation_strength=args.context_modulation_strength,
        binding_threshold=args.binding_threshold,
        binding_association_lr=args.binding_association_lr,
        binding_association_decay=args.binding_association_decay,
        binding_gain_strength=args.binding_gain_strength,
    )


if __name__ == "__main__":
    main()
