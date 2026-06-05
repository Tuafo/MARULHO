from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Mapping, Optional, Sequence, TypedDict, cast

import numpy as np
import torch

from marulho.config.model_config import MarulhoConfig
from marulho.config.presets import autonomy_preset_names, get_autonomy_preset
from marulho.data import expand_source_bank_specs
from marulho.data.corpus_loader import SourceType
from marulho.data.pattern_loader import load_probe_train_examples
from marulho.data.rtf_encoder import RTFEncoder
from marulho.gap_planner import bank_semantic_relevance_score
from marulho.reporting.io import write_json_file
from marulho.semantics.grounding_text import match_terms, salient_query_terms, split_sentences
from marulho.semantics.frontier import bank_gap_plan
from marulho.training.checkpointing import save_trainer_checkpoint
from marulho.training.runner_utils import set_seed
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


class ProbeDiagnostics(TypedDict):
    recon_error: float
    winner_entropy_bits: float
    winner_switch_rate: float
    mean_top1_margin: float
    winners: list[int]


class ProbeGapMetrics(TypedDict):
    recon_error: float
    winner_entropy_bits: float
    winner_switch_rate: float
    mean_top1_margin: float
    ambiguity: float
    exploration_bonus: float
    gap_score: float
    diagnostic_gap_score: float
    concept_novelty: float
    concept_uncertainty: float
    concept_support: float
    info_gain_score: float
    semantic_grounding_gap: float
    semantic_unsupported_ratio: float
    semantic_weak_concept_pressure: float
    semantic_answerability: float
    semantic_priority: float
    semantic_action_score: float
    semantic_gap_terms: list[dict[str, Any]]
    semantic_follow_up_questions: list[str]


class SourceSelectionFeedback(TypedDict):
    episodes: int
    gap_reduction_ema: float
    gap_score_reduction_ema: float


AUTONOMY_ABSOLUTE_IMPROVEMENT_TARGET = 0.01
AUTONOMY_RELATIVE_IMPROVEMENT_FRACTION = 0.03
AUTONOMY_SELECTION_FEEDBACK_ALPHA = 0.65
AUTONOMY_SELECTION_FEEDBACK_WEIGHT = 0.08
AUTONOMY_SELECTION_UNSEEN_BONUS = 0.04
AUTONOMY_SELECTION_VISIT_PENALTY_EXPONENT = 0.5


@dataclass
class SourceBank:
    name: str
    source: str
    source_type: str
    hf_config: Optional[str]
    text_field: str
    probe_patterns: list[torch.Tensor]
    probe_raw_windows: list[str]
    train_patterns: list[torch.Tensor]
    train_raw_windows: list[str]
    cursor: int = 0
    visits: int = 0
    metadata: dict[str, Any] | None = None

    def remaining(self) -> int:
        return max(0, len(self.train_patterns) - self.cursor)

    def next_chunk(self, n_tokens: int) -> list[tuple[str, torch.Tensor]]:
        end = min(len(self.train_patterns), self.cursor + max(0, int(n_tokens)))
        chunk = list(zip(self.train_raw_windows[self.cursor:end], self.train_patterns[self.cursor:end]))
        self.cursor = end
        if chunk:
            self.visits += 1
        return chunk


def mean_reconstruction_error(trainer: MarulhoTrainer, patterns: list[torch.Tensor]) -> float:
    if not patterns:
        return float("nan")
    values = [trainer.reconstruction_error(pattern) for pattern in patterns]
    return float(sum(values) / len(values))


def entropy_bits(counts: list[int]) -> float:
    if not counts:
        return 0.0
    values = np.asarray(counts, dtype=np.float64)
    total = values.sum()
    if total <= 0.0:
        return 0.0
    probs = values / total
    probs = probs[probs > 0.0]
    return float(-(probs * np.log2(probs)).sum())


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def autonomy_gap_improvement_target(round_robin_gap: float) -> float:
    baseline_gap = float(max(0.0, round_robin_gap))
    return float(
        min(
            AUTONOMY_ABSOLUTE_IMPROVEMENT_TARGET,
            AUTONOMY_RELATIVE_IMPROVEMENT_FRACTION * baseline_gap,
        )
    )


def relative_improvement(before: float, after: float) -> float:
    baseline = max(1e-8, abs(float(before)))
    return clamp01(max(0.0, float(before) - float(after)) / baseline)


def empirical_feedback_value(record: Mapping[str, object] | None) -> float:
    if not record:
        return 0.0
    gap_value = float(record.get("gap_reduction_ema", 0.0))
    gap_score_value = float(record.get("gap_score_reduction_ema", 0.0))
    return clamp01(0.65 * gap_value + 0.35 * gap_score_value)


def feedback_selection_bonus(
    bank: "SourceBank",
    feedback_state: Mapping[str, SourceSelectionFeedback] | None,
) -> float:
    if feedback_state is None:
        return 0.0

    record = feedback_state.get(bank.name)
    if not record or int(record.get("episodes", 0)) <= 0:
        return float(AUTONOMY_SELECTION_UNSEEN_BONUS / np.sqrt(1.0 + max(0, bank.visits)))

    value = empirical_feedback_value(record)
    uncertainty = float(1.0 / np.sqrt(1.0 + max(0, int(record.get("episodes", 0)))))
    return float(
        AUTONOMY_SELECTION_FEEDBACK_WEIGHT * value
        + 0.5 * AUTONOMY_SELECTION_UNSEEN_BONUS * uncertainty
    )


def update_source_feedback(
    feedback_state: dict[str, SourceSelectionFeedback],
    *,
    source_name: str,
    gap_before: float,
    gap_after: float,
    gap_score_before: float,
    gap_score_after: float,
) -> SourceSelectionFeedback:
    observed_gap_reduction = relative_improvement(gap_before, gap_after)
    observed_gap_score_reduction = relative_improvement(gap_score_before, gap_score_after)
    previous = feedback_state.get(source_name)
    if previous is None:
        updated: SourceSelectionFeedback = {
            "episodes": 1,
            "gap_reduction_ema": observed_gap_reduction,
            "gap_score_reduction_ema": observed_gap_score_reduction,
        }
    else:
        alpha = float(AUTONOMY_SELECTION_FEEDBACK_ALPHA)
        updated = {
            "episodes": int(previous.get("episodes", 0)) + 1,
            "gap_reduction_ema": float(
                alpha * observed_gap_reduction
                + (1.0 - alpha) * float(previous.get("gap_reduction_ema", 0.0))
            ),
            "gap_score_reduction_ema": float(
                alpha * observed_gap_score_reduction
                + (1.0 - alpha) * float(previous.get("gap_score_reduction_ema", 0.0))
            ),
        }
    feedback_state[source_name] = updated
    return updated


def _mean_signature(vectors: Sequence[torch.Tensor]) -> torch.Tensor | None:
    if not vectors:
        return None
    normalized = []
    for value in vectors:
        vector = value.detach().cpu().float().reshape(-1)
        if int(vector.numel()) <= 0 or float(vector.norm().item()) <= 1e-8:
            continue
        normalized.append(torch.nn.functional.normalize(vector, dim=0))
    if not normalized:
        return None
    mean = torch.stack(normalized, dim=0).mean(dim=0)
    if float(mean.norm().item()) <= 1e-8:
        return None
    return torch.nn.functional.normalize(mean, dim=0)


def concept_frontier_metrics(trainer: MarulhoTrainer, bank: SourceBank) -> tuple[float, float, float]:
    bank_signature = _mean_signature([trainer.routing_key_for_pattern(pattern).detach().cpu() for pattern in bank.probe_patterns])
    if bank_signature is None:
        return 1.0, 1.0, 0.0

    store = trainer.model.memory_store
    memory_keys = [
        key.detach().cpu().float().reshape(-1)
        for key in getattr(store, "slow_routing_keys", [])
        if isinstance(key, torch.Tensor) and int(key.numel()) > 0 and float(key.norm().item()) > 1e-8
    ]
    if not memory_keys:
        return 1.0, 1.0, 0.0

    normalized_keys = [torch.nn.functional.normalize(key, dim=0) for key in memory_keys]
    similarities = torch.tensor([float(torch.dot(bank_signature, key).item()) for key in normalized_keys], dtype=torch.float32)
    if int(similarities.numel()) <= 0:
        return 1.0, 1.0, 0.0

    top_k = min(8, int(similarities.numel()))
    top_values, top_indices = torch.topk(similarities, k=top_k)
    shifted = torch.clamp(top_values + 1.0, min=1e-6)
    weights = shifted / (shifted.sum() + 1e-8)

    effective_captures = torch.tensor(
        [store._effective_capture_strength(int(idx.item()), trainer.token_count) for idx in top_indices],
        dtype=torch.float32,
    )
    consolidations = torch.tensor(
        [float(store.slow_consolidation_level[int(idx.item())]) for idx in top_indices],
        dtype=torch.float32,
    )
    novelty = clamp01(1.0 - max(0.0, float(top_values.max().item())))
    uncertainty_pressure = torch.clamp(effective_captures - consolidations, min=0.0) + 0.5 * torch.clamp(1.0 - consolidations, min=0.0)
    uncertainty = clamp01(float(torch.dot(weights, uncertainty_pressure).item()))
    support = clamp01(float(torch.dot(weights, consolidations).item()))
    return novelty, uncertainty, support


def selection_metric(metrics: Mapping[str, object]) -> float:
    """Return the score used for active source selection.

    The maintained path now prefers ``semantic_action_score`` when present.
    That score starts from the observed diagnostic gap and then augments it
    with explicit probe-derived semantic grounding pressure and answerability.
    Older report surfaces may still log ``info_gain_score`` for continuity,
    but selection no longer relies on novelty/uncertainty alone.
    """
    def read_value(key: str, default: float = 0.0) -> float:
        value = metrics.get(key, default)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default

    if "semantic_action_score" in metrics:
        return read_value("semantic_action_score")
    if "diagnostic_gap_score" in metrics:
        return read_value("diagnostic_gap_score")
    return read_value("gap_score")


def probe_diagnostics(trainer: MarulhoTrainer, patterns: list[torch.Tensor]) -> ProbeDiagnostics:
    if not patterns:
        return {
            "recon_error": float("nan"),
            "winner_entropy_bits": 0.0,
            "winner_switch_rate": 0.0,
            "mean_top1_margin": 0.0,
            "winners": [],
        }

    recon_errors: list[float] = []
    winners: list[int] = []
    top1_margins: list[float] = []
    with torch.no_grad():
        prototypes = trainer.model.competitive.prototypes
        for pattern in patterns:
            routing_key = trainer.routing_key_for_pattern(pattern).to(trainer.model.device)
            sims = torch.mv(prototypes, routing_key)
            topk = torch.topk(sims, k=min(2, sims.numel()))
            top1 = float(topk.values[0].item())
            top2 = float(topk.values[1].item()) if int(topk.values.numel()) > 1 else top1
            recon_errors.append(max(0.0, float(1.0 - top1)))
            winners.append(int(topk.indices[0].item()))
            top1_margins.append(float(top1 - top2))

    winner_hist = np.bincount(winners, minlength=trainer.config.n_columns).tolist() if winners else []
    winner_entropy = entropy_bits(winner_hist)
    switch_count = sum(1 for left, right in zip(winners, winners[1:]) if left != right)
    switch_rate = float(switch_count / max(1, len(winners) - 1))
    return {
        "recon_error": float(np.mean(recon_errors)),
        "winner_entropy_bits": winner_entropy,
        "winner_switch_rate": switch_rate,
        "mean_top1_margin": float(np.mean(top1_margins)),
        "winners": winners,
    }


def probe_gap(
    trainer: MarulhoTrainer,
    bank: SourceBank,
    exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
) -> ProbeGapMetrics:
    diagnostics = probe_diagnostics(trainer, bank.probe_patterns)
    recon_error = float(diagnostics["recon_error"])
    winner_entropy = float(diagnostics["winner_entropy_bits"])
    winner_switch_rate = float(diagnostics["winner_switch_rate"])
    mean_top1_margin = float(diagnostics["mean_top1_margin"])
    ambiguity = clamp01(1.0 - (mean_top1_margin / max(1e-6, gap_margin_reference)))
    bonus = float(exploration_bonus / np.sqrt(1.0 + bank.visits))
    concept_novelty, concept_uncertainty, concept_support = concept_frontier_metrics(trainer, bank)
    semantic_plan = bank_gap_plan(trainer, bank)
    grounding_gap = float(semantic_plan["grounding_gap"])
    unsupported_ratio = float(semantic_plan["unsupported_ratio"])
    weak_pressure = float(semantic_plan["weak_concept_pressure"])
    answerability = float(semantic_plan["answerability"])
    semantic_priority = float(semantic_plan["semantic_priority"])
    score = float(
        recon_error
        + gap_ambiguity_weight * ambiguity
        + gap_switch_weight * winner_switch_rate
        + bonus
    )
    info_gain_score = float(
        score
        + float(trainer.config.acquisition_concept_novelty_weight) * concept_novelty
        + float(trainer.config.acquisition_concept_uncertainty_weight) * concept_uncertainty
    )
    semantic_action_score = float(
        score
        + float(trainer.config.acquisition_concept_novelty_weight) * concept_novelty
        + float(trainer.config.acquisition_concept_uncertainty_weight) * concept_uncertainty
        + 0.35 * semantic_priority
    )
    return {
        "recon_error": recon_error,
        "winner_entropy_bits": winner_entropy,
        "winner_switch_rate": winner_switch_rate,
        "mean_top1_margin": mean_top1_margin,
        "ambiguity": ambiguity,
        "exploration_bonus": bonus,
        "gap_score": float(recon_error + bonus),
        "diagnostic_gap_score": score,
        "concept_novelty": concept_novelty,
        "concept_uncertainty": concept_uncertainty,
        "concept_support": concept_support,
        "info_gain_score": info_gain_score,
        "semantic_grounding_gap": grounding_gap,
        "semantic_unsupported_ratio": unsupported_ratio,
        "semantic_weak_concept_pressure": weak_pressure,
        "semantic_answerability": answerability,
        "semantic_priority": semantic_priority,
        "semantic_action_score": semantic_action_score,
        "semantic_gap_terms": list(semantic_plan["gap_plan"].get("gap_terms") or []),
        "semantic_follow_up_questions": list(semantic_plan["gap_plan"].get("follow_up_questions") or []),
    }


def active_selection_score(
    bank: SourceBank,
    gap_snapshot: Mapping[str, ProbeGapMetrics],
    min_visits: int,
    coverage_balance_penalty: float,
    feedback_state: Mapping[str, SourceSelectionFeedback] | None = None,
) -> float:
    coverage_penalty = float(
        np.power(max(0, bank.visits - min_visits), AUTONOMY_SELECTION_VISIT_PENALTY_EXPONENT)
        * coverage_balance_penalty
    )
    return float(
        selection_metric(gap_snapshot[bank.name])
        + feedback_selection_bonus(bank, feedback_state)
        - coverage_penalty
    )


def select_active_source(
    available: list[SourceBank],
    gap_snapshot: Mapping[str, ProbeGapMetrics],
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    feedback_state: Mapping[str, SourceSelectionFeedback] | None = None,
) -> tuple[SourceBank, dict[str, float]]:
    diagnostic_scores = {
        bank.name: float(selection_metric(gap_snapshot[bank.name]))
        for bank in available
    }
    ranked = sorted(diagnostic_scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        raise RuntimeError("No sources available for active selection")

    if len(ranked) == 1 or float(ranked[0][1] - ranked[1][1]) > gap_focus_margin:
        selected_name = ranked[0][0]
        selected = next(bank for bank in available if bank.name == selected_name)
        return selected, diagnostic_scores

    min_visits = min(bank.visits for bank in available)
    selection_scores = {
        bank.name: active_selection_score(
            bank=bank,
            gap_snapshot=gap_snapshot,
            min_visits=min_visits,
            coverage_balance_penalty=coverage_balance_penalty,
            feedback_state=feedback_state,
        )
        for bank in available
    }
    selected = max(available, key=lambda bank: selection_scores[bank.name])
    return selected, selection_scores


def summarize_training_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "mean_dopamine": float("nan"),
            "mean_serotonin": float("nan"),
            "mean_acetylcholine": float("nan"),
            "mean_norepinephrine": float("nan"),
            "mean_binding_strength": float("nan"),
            "mean_context_gain": float("nan"),
        }
    return {
        "mean_dopamine": float(np.mean([float(row["dopamine"]) for row in rows])),
        "mean_serotonin": float(np.mean([float(row.get("serotonin", float("nan"))) for row in rows])),
        "mean_acetylcholine": float(np.mean([float(row["acetylcholine"]) for row in rows])),
        "mean_norepinephrine": float(np.mean([float(row["norepinephrine"]) for row in rows])),
        "mean_binding_strength": float(np.mean([float(row["binding_strength"]) for row in rows])),
        "mean_context_gain": float(np.mean([float(row["context_gain_mean"]) for row in rows])),
    }


def autonomy_gate_from_comparison(comparison: Mapping[str, float]) -> dict[str, Any]:
    mean_target = autonomy_gap_improvement_target(float(comparison["round_robin_final_mean_gap"]))
    max_target = autonomy_gap_improvement_target(float(comparison["round_robin_final_max_gap"]))
    mean_delta = float(comparison["active_minus_round_robin_mean_gap"])
    max_delta = float(comparison["active_minus_round_robin_max_gap"])
    return {
        "pass": bool(
            mean_delta <= -mean_target
            and max_delta <= -max_target
        ),
        "active_mean_gap_better_than_round_robin": bool(mean_delta <= -mean_target),
        "active_max_gap_better_than_round_robin": bool(max_delta <= -max_target),
        "thresholds": {
            "absolute_improvement_cap": float(AUTONOMY_ABSOLUTE_IMPROVEMENT_TARGET),
            "relative_improvement_fraction": float(AUTONOMY_RELATIVE_IMPROVEMENT_FRACTION),
            "active_minus_round_robin_mean_gap_max": float(-mean_target),
            "active_minus_round_robin_max_gap_max": float(-max_target),
        },
    }


def source_bank_protocol(prefix: str, banks: Sequence[SourceBank]) -> str:
    source_types = sorted({str(bank.source_type or "auto") for bank in banks})
    if source_types == ["hf"]:
        suffix = "hf"
    elif source_types == ["web"]:
        suffix = "open_web"
    elif source_types == ["file"]:
        suffix = "local_files"
    else:
        suffix = "mixed_" + "_".join(source_types)
    return f"{prefix}_{suffix}"


def describe_source_banks(banks: Sequence[SourceBank]) -> list[dict[str, Any]]:
    return [
        {
            "name": bank.name,
            "source": bank.source,
            "source_type": bank.source_type,
            "hf_config": bank.hf_config,
            "text_field": bank.text_field,
            "probe_tokens": len(bank.probe_patterns),
            "probe_windows": len(bank.probe_raw_windows),
            "train_tokens": len(bank.train_patterns),
            "metadata": deepcopy(bank.metadata) if bank.metadata else None,
        }
        for bank in banks
    ]


def make_config(args: dict[str, Any]) -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=int(args["n_columns"]),
        column_latent_dim=int(args["column_latent_dim"]),
        memory_capacity=int(args["memory_capacity"]),
        input_weight_blend=float(args["input_weight_blend"]),
        input_synapse_ltp=float(args["input_synapse_ltp"]),
        input_synapse_ltd=float(args["input_synapse_ltd"]),
        input_weight_row_target=float(args["input_weight_row_target"]),
        homeostasis_beta=float(args["homeostasis_beta"]),
        homeostasis_lr=float(args["homeostasis_lr"]),
        slow_mean_decay=float(args["slow_mean_decay"]),
        use_winner_local_drift=bool(args["use_winner_local_drift"]),
        drift_threshold=float(args["drift_threshold"]),
        micro_sleep_interval_tokens=int(args["micro_sleep_interval_tokens"]),
        micro_sleep_replay_steps=int(args["micro_sleep_replay_steps"]),
        micro_sleep_candidate_pool=int(args["micro_sleep_candidate_pool"]),
        micro_sleep_memory_blend=float(args["micro_sleep_memory_blend"]),
        deep_sleep_interval_tokens=int(args["deep_sleep_interval_tokens"]),
        deep_sleep_replay_steps=int(args["deep_sleep_replay_steps"]),
        deep_sleep_candidate_pool=int(args["deep_sleep_candidate_pool"]),
        deep_sleep_memory_blend=float(args["deep_sleep_memory_blend"]),
        deep_sleep_cooldown_tokens=int(args["deep_sleep_cooldown_tokens"]),
        emergency_deep_sleep_cooldown_tokens=int(args["emergency_deep_sleep_cooldown_tokens"]),
        drift_floor_history_tokens=int(args["drift_floor_history_tokens"]),
        drift_floor_check_interval_tokens=int(args["drift_floor_check_interval_tokens"]),
        drift_floor_window_tokens=int(args["drift_floor_window_tokens"]),
        drift_floor_trigger_min_tokens=int(args["drift_floor_trigger_min_tokens"]),
        drift_floor_rise_tolerance=float(args["drift_floor_rise_tolerance"]),
        prototype_momentum=float(args["prototype_momentum"]),
        enable_context_layer=True,
        context_decay=float(args["context_decay"]),
        context_transition_lr=float(args["context_transition_lr"]),
        context_modulation_strength=float(args["context_modulation_strength"]),
        enable_binding_layer=True,
        binding_threshold=float(args["binding_threshold"]),
        binding_association_lr=float(args["binding_association_lr"]),
        binding_association_decay=float(args["binding_association_decay"]),
        binding_gain_strength=float(args["binding_gain_strength"]),
    )


def load_source_banks(
    source_bank_specs: list[dict[str, Any]],
    encoder: RTFEncoder,
    window_size: int,
    probe_tokens: int,
    source_train_tokens: int,
    *,
    semantic_plan: Mapping[str, Any] | None = None,
    metadata_prefilter: bool = False,
) -> list[SourceBank]:
    resolved_specs = expand_source_bank_specs(
        source_bank_specs,
        semantic_plan=semantic_plan,
        metadata_prefilter=metadata_prefilter,
    )
    banks: list[SourceBank] = []
    for spec in resolved_specs:
        source_type = cast(SourceType, spec.get("source_type", "auto"))
        prefix_text = _catalog_metadata_prefix_text(spec)
        probe_patterns, probe_raw_windows, train_patterns, train_raw_windows = load_probe_train_examples(
            source=str(spec["source"]),
            source_type=source_type,
            hf_config=spec.get("hf_config"),
            text_field=str(spec.get("text_field", "text")),
            encoder=encoder,
            window_size=window_size,
            probe_tokens=probe_tokens,
            train_tokens=source_train_tokens,
            prefix_text=prefix_text,
        )
        if not probe_patterns or not train_patterns:
            metadata = spec.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("catalog_mode"):
                continue
            raise ValueError(f"Source bank {spec['name']} did not produce enough patterns")
        bank = SourceBank(
            name=str(spec["name"]),
            source=str(spec["source"]),
            source_type=str(source_type),
            hf_config=spec.get("hf_config"),
            text_field=str(spec.get("text_field", "text")),
            probe_patterns=probe_patterns,
            probe_raw_windows=probe_raw_windows,
            train_patterns=train_patterns,
            train_raw_windows=train_raw_windows,
            metadata=dict(spec.get("metadata") or {}) or None,
        )
        _refresh_loaded_bank_metadata(bank, semantic_plan)
        banks.append(bank)
    if not banks:
        raise ValueError("Source bank expansion did not yield any usable sources")
    return banks


def _normalize_optional_metadata_text(value: Any) -> str:
    if value is None:
        return ""
    normalized = " ".join(str(value).split()).strip()
    return "" if normalized.lower() == "none" else normalized


def _refresh_loaded_bank_metadata(
    bank: SourceBank,
    semantic_plan: Mapping[str, Any] | None,
) -> None:
    metadata = dict(bank.metadata or {})
    if not metadata:
        return
    for field in ("provider", "query_text"):
        if field in metadata:
            metadata[field] = _normalize_optional_metadata_text(metadata.get(field))
    if semantic_plan:
        metadata["semantic_relevance"] = max(
            float(metadata.get("semantic_relevance") or 0.0),
            float(bank_semantic_relevance_score(bank, semantic_plan)),
        )
    bank.metadata = metadata or None


def _catalog_metadata_prefix_text(spec: Mapping[str, Any]) -> str:
    metadata = spec.get("metadata")
    if not isinstance(metadata, Mapping):
        return ""

    title = _normalize_optional_metadata_text(metadata.get("catalog_title"))
    content_preview = _normalize_optional_metadata_text(metadata.get("catalog_content_preview"))
    summary = _normalize_optional_metadata_text(metadata.get("catalog_summary"))
    topic_terms = [
        _normalize_optional_metadata_text(term)
        for term in list(metadata.get("catalog_terms") or [])
    ]
    topic_terms = [term for term in topic_terms if term]

    parts: list[str] = []
    focus_terms = _catalog_metadata_focus_terms(metadata)
    focus_term_text = _catalog_metadata_focus_term_text(
        title=title,
        content_preview=content_preview,
        summary=summary,
        topic_terms=topic_terms,
        focus_terms=focus_terms,
    )
    primary_summary = content_preview or summary
    focused_summary = _catalog_metadata_focused_summary(primary_summary, focus_terms, title=title)
    if focused_summary:
        parts.append(focused_summary)
    if focus_term_text:
        parts.append(focus_term_text)
    elif title:
        parts.append(title)
    if content_preview and content_preview.lower() not in " ".join(parts).lower():
        parts.append(content_preview)
    if summary and summary.lower() not in " ".join(parts).lower():
        parts.append(summary)
    if topic_terms:
        parts.append(f"Topics: {', '.join(topic_terms[:6])}.")
    return " ".join(parts).strip()


def _catalog_metadata_focus_terms(metadata: Mapping[str, Any]) -> list[str]:
    query_texts = [
        _normalize_optional_metadata_text(item)
        for item in list(metadata.get("query_texts") or [])
    ]
    query_text = _normalize_optional_metadata_text(metadata.get("query_text"))
    if query_text:
        query_texts.append(query_text)

    ordered_terms: list[str] = []
    seen: set[str] = set()
    for text in query_texts:
        for term in salient_query_terms(text):
            normalized = term.strip().lower()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered_terms.append(normalized)
    return ordered_terms[:6]


def _catalog_metadata_focus_term_text(
    *,
    title: str,
    content_preview: str,
    summary: str,
    topic_terms: Sequence[str],
    focus_terms: Sequence[str],
) -> str:
    if not focus_terms:
        return ""
    metadata_text = " ".join([title, content_preview, summary, *topic_terms]).strip()
    matched_terms = match_terms(list(focus_terms), metadata_text)
    if not matched_terms:
        return ""
    return f"Terms: {', '.join(matched_terms[:6])}."


def _catalog_metadata_focused_summary(
    summary: str,
    focus_terms: Sequence[str],
    *,
    title: str = "",
) -> str:
    if not summary:
        return title
    if not focus_terms:
        return f"{title}. {summary}".strip(". ") if title else summary

    title_matches = set(match_terms(list(focus_terms), title))
    fragments: list[str] = []
    seen_fragments: set[str] = set()
    for sentence in split_sentences(summary):
        clauses = [
            _normalize_optional_metadata_text(fragment)
            for fragment in re.split(r"[,;:()]", sentence)
        ]
        clauses = [fragment for fragment in clauses if fragment]
        if not clauses:
            continue
        candidate_windows = [" ".join(clauses)]
        for index, clause in enumerate(clauses):
            candidate_windows.append(clause)
            if index + 1 < len(clauses):
                candidate_windows.append(f"{clause}, {clauses[index + 1]}")
        for candidate in candidate_windows:
            normalized = _normalize_optional_metadata_text(candidate)
            compact = normalized.lower()
            if not normalized or compact in seen_fragments:
                continue
            seen_fragments.add(compact)
            fragments.append(normalized)

    ranked_fragments: list[tuple[str, list[str], int, int, int]] = []
    for fragment in fragments:
        matches = match_terms(list(focus_terms), fragment)
        if not matches:
            continue
        unseen_matches = [term for term in matches if term not in title_matches]
        ranked_fragments.append(
            (
                fragment,
                matches,
                len(unseen_matches),
                len(matches),
                len(fragment),
            )
        )
    ranked_fragments.sort(
        key=lambda item: (
            item[2],
            -item[4],
            item[3],
        ),
        reverse=True,
    )
    if not ranked_fragments:
        return f"{title}. {summary}".strip(". ") if title else summary

    selected_fragments: list[str] = []
    covered_terms = set(title_matches)
    for fragment, matches, _unseen_count, _match_count, _length in ranked_fragments:
        new_terms = [term for term in matches if term not in covered_terms]
        if selected_fragments and not new_terms:
            continue
        selected_fragments.append(fragment)
        covered_terms.update(matches)
        if len(selected_fragments) >= 2 or covered_terms.issuperset(focus_terms):
            break
    if not selected_fragments:
        selected_fragments.append(ranked_fragments[0][0])

    fragment_text = ". ".join(selected_fragments).strip()
    if title:
        return f"{title}: {fragment_text}."
    return f"{fragment_text}."


def next_round_robin_source(banks: list[SourceBank], start_idx: int) -> tuple[SourceBank, int]:
    total = len(banks)
    for offset in range(total):
        idx = (start_idx + offset) % total
        if banks[idx].remaining() > 0:
            return banks[idx], (idx + 1) % total
    raise RuntimeError("No source has remaining tokens")


def run_policy(
    policy_name: str,
    base_banks: list[SourceBank],
    cfg: MarulhoConfig,
    seed: int,
    episode_tokens: int,
    warmup_rounds: int,
    seek_episodes: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
) -> tuple[dict[str, Any], MarulhoTrainer]:
    set_seed(seed)
    model = MarulhoModel(cfg)
    trainer = MarulhoTrainer(model, cfg)
    banks = deepcopy(base_banks)
    metrics_rows: list[dict[str, Any]] = []
    episode_history: list[dict[str, Any]] = []
    rr_index = 0

    for warmup_round in range(max(0, int(warmup_rounds))):
        for bank in banks:
            chunk = bank.next_chunk(episode_tokens)
            if not chunk:
                continue
            for raw_window, pattern in chunk:
                row = trainer.train_step(pattern, raw_window=raw_window)
                row["source_name"] = bank.name
                row["phase"] = "warmup"
                metrics_rows.append(row)
            episode_history.append(
                {
                    "phase": "warmup",
                    "episode_index": len(episode_history) + 1,
                    "warmup_round": warmup_round + 1,
                    "selected_source": bank.name,
                    "tokens_trained": len(chunk),
                }
            )

    selection_feedback: dict[str, SourceSelectionFeedback] = {}
    for seek_idx in range(max(0, int(seek_episodes))):
        available = [bank for bank in banks if bank.remaining() > 0]
        if not available:
            break

        gap_snapshot = {
            bank.name: probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )
            for bank in available
        }
        if policy_name == "active":
            selected, selection_scores = select_active_source(
                available,
                gap_snapshot,
                coverage_balance_penalty,
                gap_focus_margin,
                feedback_state=selection_feedback,
            )
        elif policy_name == "round_robin":
            selected, rr_index = next_round_robin_source(banks, rr_index)
            gap_snapshot = {
                bank.name: probe_gap(
                    trainer,
                    bank,
                    gap_exploration_bonus,
                    gap_ambiguity_weight,
                    gap_switch_weight,
                    gap_margin_reference,
                )
                for bank in available
            }
            selection_scores = {bank.name: float(selection_metric(gap_snapshot[bank.name])) for bank in available}
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        gap_before = float(gap_snapshot[selected.name]["recon_error"])
        selected_gap_score_before = float(gap_snapshot[selected.name]["gap_score"])
        selected_diagnostic_gap_score_before = float(gap_snapshot[selected.name]["diagnostic_gap_score"])
        selected_empirical_value_before = empirical_feedback_value(selection_feedback.get(selected.name))
        chunk = selected.next_chunk(episode_tokens)
        for raw_window, pattern in chunk:
            row = trainer.train_step(pattern, raw_window=raw_window)
            row["source_name"] = selected.name
            row["phase"] = "seek"
            metrics_rows.append(row)
        selected_after_metrics = probe_gap(
            trainer,
            selected,
            gap_exploration_bonus,
            gap_ambiguity_weight,
            gap_switch_weight,
            gap_margin_reference,
        )
        gap_after = float(selected_after_metrics["recon_error"])
        selected_gap_score_after = float(selected_after_metrics["gap_score"])
        selected_diagnostic_gap_score_after = float(selected_after_metrics["diagnostic_gap_score"])
        updated_feedback = update_source_feedback(
            selection_feedback,
            source_name=selected.name,
            gap_before=gap_before,
            gap_after=gap_after,
            gap_score_before=selected_gap_score_before,
            gap_score_after=selected_gap_score_after,
        )

        episode_history.append(
            {
                "phase": "seek",
                "episode_index": len(episode_history) + 1,
                "seek_episode": seek_idx + 1,
                "selected_source": selected.name,
                "tokens_trained": len(chunk),
                "selected_gap_before": gap_before,
                "selected_gap_after": gap_after,
                "selected_gap_reduction": gap_before - gap_after,
                "selected_gap_score_before": selected_gap_score_before,
                "selected_gap_score_after": selected_gap_score_after,
                "selected_gap_score_reduction": selected_gap_score_before - selected_gap_score_after,
                "selected_diagnostic_gap_score_before": selected_diagnostic_gap_score_before,
                "selected_diagnostic_gap_score_after": selected_diagnostic_gap_score_after,
                "selected_diagnostic_gap_score_reduction": selected_diagnostic_gap_score_before - selected_diagnostic_gap_score_after,
                "selected_empirical_value_before": selected_empirical_value_before,
                "selected_empirical_value_after": empirical_feedback_value(updated_feedback),
                "gap_snapshot": {
                    name: {
                        "recon_error": float(values["recon_error"]),
                        "winner_entropy_bits": float(values["winner_entropy_bits"]),
                        "winner_switch_rate": float(values["winner_switch_rate"]),
                        "mean_top1_margin": float(values["mean_top1_margin"]),
                        "ambiguity": float(values["ambiguity"]),
                        "exploration_bonus": float(values["exploration_bonus"]),
                        "gap_score": float(values["gap_score"]),
                        "diagnostic_gap_score": float(values["diagnostic_gap_score"]),
                        "concept_novelty": float(values.get("concept_novelty", 0.0)),
                        "concept_uncertainty": float(values.get("concept_uncertainty", 0.0)),
                        "concept_support": float(values.get("concept_support", 0.0)),
                        "info_gain_score": float(values.get("info_gain_score", values["diagnostic_gap_score"])),
                        "semantic_grounding_gap": float(values.get("semantic_grounding_gap", 0.0)),
                        "semantic_unsupported_ratio": float(values.get("semantic_unsupported_ratio", 0.0)),
                        "semantic_weak_concept_pressure": float(values.get("semantic_weak_concept_pressure", 0.0)),
                        "semantic_answerability": float(values.get("semantic_answerability", 0.0)),
                        "semantic_priority": float(values.get("semantic_priority", 0.0)),
                        "semantic_action_score": float(values.get("semantic_action_score", values["diagnostic_gap_score"])),
                        "semantic_gap_terms": list(values.get("semantic_gap_terms", [])),
                        "semantic_follow_up_questions": list(values.get("semantic_follow_up_questions", [])),
                        "selection_score": float(selection_scores[name]),
                    }
                    for name, values in gap_snapshot.items()
                },
            }
        )

    final_gap_by_source = {
        bank.name: float(
            probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )["recon_error"]
        )
        for bank in banks
    }
    final_gap_score_by_source = {
        bank.name: float(
            probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )["gap_score"]
        )
        for bank in banks
    }
    final_diagnostic_gap_score_by_source = {
        bank.name: float(
            probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )["diagnostic_gap_score"]
        )
        for bank in banks
    }
    final_info_gain_by_source = {
        bank.name: float(
            probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )["info_gain_score"]
        )
        for bank in banks
    }
    final_semantic_action_by_source = {
        bank.name: float(
            probe_gap(
                trainer,
                bank,
                gap_exploration_bonus,
                gap_ambiguity_weight,
                gap_switch_weight,
                gap_margin_reference,
            )["semantic_action_score"]
        )
        for bank in banks
    }
    selected_reductions = [
        float(row["selected_gap_reduction"])
        for row in episode_history
        if row["phase"] == "seek"
    ]
    selected_gap_score_reductions = [
        float(row["selected_gap_score_reduction"])
        for row in episode_history
        if row["phase"] == "seek"
    ]
    return {
        "policy": policy_name,
        "final_gap_by_source": final_gap_by_source,
        "final_gap_score_by_source": final_gap_score_by_source,
        "final_diagnostic_gap_score_by_source": final_diagnostic_gap_score_by_source,
        "final_info_gain_by_source": final_info_gain_by_source,
        "final_semantic_action_by_source": final_semantic_action_by_source,
        "final_mean_gap": float(np.mean(list(final_gap_by_source.values()))),
        "final_max_gap": float(max(final_gap_by_source.values())),
        "final_mean_gap_score": float(np.mean(list(final_gap_score_by_source.values()))),
        "final_max_gap_score": float(max(final_gap_score_by_source.values())),
        "final_mean_diagnostic_gap_score": float(np.mean(list(final_diagnostic_gap_score_by_source.values()))),
        "final_max_diagnostic_gap_score": float(max(final_diagnostic_gap_score_by_source.values())),
        "final_mean_info_gain": float(np.mean(list(final_info_gain_by_source.values()))),
        "final_max_info_gain": float(max(final_info_gain_by_source.values())),
        "final_mean_semantic_action": float(np.mean(list(final_semantic_action_by_source.values()))),
        "final_max_semantic_action": float(max(final_semantic_action_by_source.values())),
        "source_visit_counts": {bank.name: int(bank.visits) for bank in banks},
        "source_empirical_value_by_source": {
            bank.name: empirical_feedback_value(selection_feedback.get(bank.name))
            for bank in banks
        },
        "source_feedback_observations_by_source": {
            bank.name: int(selection_feedback.get(bank.name, {}).get("episodes", 0))
            for bank in banks
        },
        "mean_selected_gap_reduction": float(np.mean(selected_reductions)) if selected_reductions else float("nan"),
        "mean_selected_gap_score_reduction": float(np.mean(selected_gap_score_reductions)) if selected_gap_score_reductions else float("nan"),
        "episode_history": episode_history,
        "training_diagnostics": summarize_training_metrics(metrics_rows),
        "runtime_scope": model.runtime_scope_report(),
    }, trainer


def run_autonomy(
    source_bank: list[dict[str, Any]],
    source_train_tokens: int,
    probe_tokens: int,
    episode_tokens: int,
    warmup_rounds: int,
    seek_episodes: int,
    gap_exploration_bonus: float,
    gap_ambiguity_weight: float,
    gap_switch_weight: float,
    gap_margin_reference: float,
    coverage_balance_penalty: float,
    gap_focus_margin: float,
    output_dir: Path,
    checkpoint_out: Optional[Path],
    save_plots: bool,
    **kwargs: Any,
) -> None:
    cfg = make_config(kwargs)
    encoder = RTFEncoder.from_config(cfg)
    base_banks = load_source_banks(
        source_bank_specs=source_bank,
        encoder=encoder,
        window_size=cfg.window_size,
        probe_tokens=probe_tokens,
        source_train_tokens=source_train_tokens,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    active, active_trainer = run_policy(
        policy_name="active",
        base_banks=base_banks,
        cfg=cfg,
        seed=int(kwargs["seed"]),
        episode_tokens=episode_tokens,
        warmup_rounds=warmup_rounds,
        seek_episodes=seek_episodes,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
    )
    round_robin, _ = run_policy(
        policy_name="round_robin",
        base_banks=base_banks,
        cfg=cfg,
        seed=int(kwargs["seed"]),
        episode_tokens=episode_tokens,
        warmup_rounds=warmup_rounds,
        seek_episodes=seek_episodes,
        gap_exploration_bonus=gap_exploration_bonus,
        gap_ambiguity_weight=gap_ambiguity_weight,
        gap_switch_weight=gap_switch_weight,
        gap_margin_reference=gap_margin_reference,
        coverage_balance_penalty=coverage_balance_penalty,
        gap_focus_margin=gap_focus_margin,
    )

    comparison = {
        "active_final_mean_gap": float(active["final_mean_gap"]),
        "round_robin_final_mean_gap": float(round_robin["final_mean_gap"]),
        "active_final_max_gap": float(active["final_max_gap"]),
        "round_robin_final_max_gap": float(round_robin["final_max_gap"]),
        "active_final_mean_gap_score": float(active["final_mean_gap_score"]),
        "round_robin_final_mean_gap_score": float(round_robin["final_mean_gap_score"]),
        "active_final_max_gap_score": float(active["final_max_gap_score"]),
        "round_robin_final_max_gap_score": float(round_robin["final_max_gap_score"]),
        "active_final_mean_diagnostic_gap_score": float(active["final_mean_diagnostic_gap_score"]),
        "round_robin_final_mean_diagnostic_gap_score": float(round_robin["final_mean_diagnostic_gap_score"]),
        "active_final_mean_info_gain": float(active.get("final_mean_info_gain", active["final_mean_diagnostic_gap_score"])),
        "round_robin_final_mean_info_gain": float(round_robin.get("final_mean_info_gain", round_robin["final_mean_diagnostic_gap_score"])),
        "active_final_mean_semantic_action": float(active.get("final_mean_semantic_action", active["final_mean_diagnostic_gap_score"])),
        "round_robin_final_mean_semantic_action": float(
            round_robin.get("final_mean_semantic_action", round_robin["final_mean_diagnostic_gap_score"])
        ),
        "active_minus_round_robin_mean_gap": float(active["final_mean_gap"] - round_robin["final_mean_gap"]),
        "active_minus_round_robin_max_gap": float(active["final_max_gap"] - round_robin["final_max_gap"]),
        "active_minus_round_robin_mean_gap_score": float(active["final_mean_gap_score"] - round_robin["final_mean_gap_score"]),
        "active_minus_round_robin_max_gap_score": float(active["final_max_gap_score"] - round_robin["final_max_gap_score"]),
        "active_minus_round_robin_mean_diagnostic_gap_score": float(active["final_mean_diagnostic_gap_score"] - round_robin["final_mean_diagnostic_gap_score"]),
        "active_minus_round_robin_mean_info_gain": float(
            active.get("final_mean_info_gain", active["final_mean_diagnostic_gap_score"])
            - round_robin.get("final_mean_info_gain", round_robin["final_mean_diagnostic_gap_score"])
        ),
        "active_minus_round_robin_mean_semantic_action": float(
            active.get("final_mean_semantic_action", active["final_mean_diagnostic_gap_score"])
            - round_robin.get("final_mean_semantic_action", round_robin["final_mean_diagnostic_gap_score"])
        ),
        "active_mean_selected_gap_reduction": float(active["mean_selected_gap_reduction"]),
        "round_robin_mean_selected_gap_reduction": float(round_robin["mean_selected_gap_reduction"]),
        "active_mean_selected_gap_score_reduction": float(active["mean_selected_gap_score_reduction"]),
        "round_robin_mean_selected_gap_score_reduction": float(round_robin["mean_selected_gap_score_reduction"]),
    }
    autonomy_gate = autonomy_gate_from_comparison(comparison)

    summary = {
        "protocol": source_bank_protocol("autonomy_gap_seeking", base_banks),
        "source_names": [bank.name for bank in base_banks],
        "source_types": sorted({bank.source_type for bank in base_banks}),
        "source_bank": describe_source_banks(base_banks),
        "autonomy_setup": {
            "source_train_tokens": int(source_train_tokens),
            "probe_tokens": int(probe_tokens),
            "episode_tokens": int(episode_tokens),
            "warmup_rounds": int(warmup_rounds),
            "seek_episodes": int(seek_episodes),
            "gap_exploration_bonus": float(gap_exploration_bonus),
            "gap_ambiguity_weight": float(gap_ambiguity_weight),
            "gap_switch_weight": float(gap_switch_weight),
            "gap_margin_reference": float(gap_margin_reference),
            "coverage_balance_penalty": float(coverage_balance_penalty),
            "gap_focus_margin": float(gap_focus_margin),
        },
        "policy_results": {
            "active": active,
            "round_robin": round_robin,
        },
        "comparison": comparison,
        "autonomy_gate": autonomy_gate,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    checkpoint_path: Optional[Path] = None
    if checkpoint_out is not None:
        checkpoint_path = save_trainer_checkpoint(
            checkpoint_out,
            active_trainer,
            metadata={
                "protocol": source_bank_protocol("autonomy_gap_seeking", base_banks),
                "exported_policy": "active",
                "source_names": [bank.name for bank in base_banks],
                "source_types": sorted({bank.source_type for bank in base_banks}),
                "source_train_tokens": int(source_train_tokens),
                "probe_tokens": int(probe_tokens),
                "episode_tokens": int(episode_tokens),
                "warmup_rounds": int(warmup_rounds),
                "seek_episodes": int(seek_episodes),
            },
        )
        summary["checkpoint_path"] = str(checkpoint_path)
    write_json_file(output_dir / "summary.json", summary)
    if save_plots:
        from marulho.reporting.autonomy import plot_autonomy_summary
        plot_autonomy_summary(output_dir, summary)

    print("Autonomy gap-seeking summary")
    print(f"active_final_mean_gap={comparison['active_final_mean_gap']:.6f}")
    print(f"round_robin_final_mean_gap={comparison['round_robin_final_mean_gap']:.6f}")
    print(f"active_final_max_gap={comparison['active_final_max_gap']:.6f}")
    print(f"round_robin_final_max_gap={comparison['round_robin_final_max_gap']:.6f}")
    print(f"active_final_mean_gap_score={comparison['active_final_mean_gap_score']:.6f}")
    print(f"round_robin_final_mean_gap_score={comparison['round_robin_final_mean_gap_score']:.6f}")
    print(f"active_mean_selected_gap_reduction={comparison['active_mean_selected_gap_reduction']:.6f}")
    print(f"round_robin_mean_selected_gap_reduction={comparison['round_robin_mean_selected_gap_reduction']:.6f}")
    print(f"autonomy_gate_pass={autonomy_gate['pass']}")
    print(f"summary_json={output_dir / 'summary.json'}")
    if checkpoint_path is not None:
        print(f"checkpoint_path={checkpoint_path}")
    if save_plots:
        print(f"autonomy_plot={output_dir / 'autonomy_diagnostics.png'}")


def main() -> None:
    preset_parser = argparse.ArgumentParser(add_help=False)
    preset_parser.add_argument("--preset", choices=autonomy_preset_names(), default=None)
    preset_args, _ = preset_parser.parse_known_args()
    preset_defaults = get_autonomy_preset(preset_args.preset)

    parser = argparse.ArgumentParser(description="Run MARULHO autonomy benchmark with gap-driven source selection")
    parser.add_argument("--preset", choices=autonomy_preset_names(), default=preset_args.preset)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--source-train-tokens", type=int, default=preset_defaults.get("source_train_tokens", 4000))
    parser.add_argument("--probe-tokens", type=int, default=preset_defaults.get("probe_tokens", 128))
    parser.add_argument("--episode-tokens", type=int, default=preset_defaults.get("episode_tokens", 500))
    parser.add_argument("--warmup-rounds", type=int, default=preset_defaults.get("warmup_rounds", 1))
    parser.add_argument("--seek-episodes", type=int, default=preset_defaults.get("seek_episodes", 6))
    parser.add_argument("--gap-exploration-bonus", type=float, default=preset_defaults.get("gap_exploration_bonus", 0.02))
    parser.add_argument("--gap-ambiguity-weight", type=float, default=preset_defaults.get("gap_ambiguity_weight", 0.08))
    parser.add_argument("--gap-switch-weight", type=float, default=preset_defaults.get("gap_switch_weight", 0.08))
    parser.add_argument("--gap-margin-reference", type=float, default=preset_defaults.get("gap_margin_reference", 0.12))
    parser.add_argument("--coverage-balance-penalty", type=float, default=preset_defaults.get("coverage_balance_penalty", 0.03))
    parser.add_argument("--gap-focus-margin", type=float, default=preset_defaults.get("gap_focus_margin", 0.03))
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

    preset_data = get_autonomy_preset(args.preset)
    if not preset_data:
        raise ValueError("Autonomy runner currently requires a preset-defined source bank")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("reports") / (f"{args.preset}_{stamp}" if args.preset else f"autonomy_{stamp}")
    else:
        output_dir = args.output_dir

    run_autonomy(
        source_bank=deepcopy(preset_data["source_bank"]),
        source_train_tokens=args.source_train_tokens,
        probe_tokens=args.probe_tokens,
        episode_tokens=args.episode_tokens,
        warmup_rounds=args.warmup_rounds,
        seek_episodes=args.seek_episodes,
        gap_exploration_bonus=args.gap_exploration_bonus,
        gap_ambiguity_weight=args.gap_ambiguity_weight,
        gap_switch_weight=args.gap_switch_weight,
        gap_margin_reference=args.gap_margin_reference,
        coverage_balance_penalty=args.coverage_balance_penalty,
        gap_focus_margin=args.gap_focus_margin,
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
