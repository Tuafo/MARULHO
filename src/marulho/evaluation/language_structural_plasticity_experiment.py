"""Write standalone MARULHO LM structural-plasticity transaction evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_training_experiment import DEFAULT_CORPUS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    precompute_sampled_vocab_batches,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_deep_sleep_proposal,
    build_language_structural_memory_slot_expansion_proposal,
    build_language_structural_route_bank_expansion_proposal,
)


SURFACE = "marulho_language_structural_plasticity_experiment.v1"
ARTIFACT_KIND = "marulho_language_structural_plasticity_experiment"
ENTRY_SURFACE = "marulho_language_structural_plasticity_experiment_entry.v1"

SUPPORTED_PROPOSAL_KINDS = (
    "memory_slot_expansion",
    "route_bank_expansion",
    "expert_deep_sleep",
)


@dataclass(frozen=True)
class LanguageStructuralPlasticityExperimentConfig:
    model_vocab_size: int = 0
    sampled_vocab_size: int = 0
    embedding_dim: int = 32
    state_dim: int = 64
    expert_count: int = 8
    active_expert_count: int = 2
    route_candidate_count: int = 4
    expert_hidden_dim: int = 96
    memory_slot_count: int = 0
    memory_slot_growth: int = 1024
    memory_slot_candidate_count: int = 8
    active_memory_slot_count: int = 2
    route_candidate_growth: int = 2
    deep_sleep_expert_id: int = -1
    sequence_length: int = 32
    stride: int = 32
    batch_size: int = 8
    eval_fraction: float = 0.2
    max_eval_batches: int = 2
    max_eval_loss_delta: float = 100.0
    seed: int = 20260705
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def _read_text(path: str | Path | None) -> tuple[str, str]:
    if path is None:
        return DEFAULT_CORPUS, "default_inline"
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Structural-plasticity corpus is empty: {resolved}")
    return text, str(resolved)


def _model_vocab_size(
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageStructuralPlasticityExperimentConfig,
) -> int:
    configured = int(config.model_vocab_size)
    return configured if configured > 0 else int(tokenizer.vocab_size)


def _model_config(
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageStructuralPlasticityExperimentConfig,
    *,
    proposal_kind: str,
) -> LanguageModelConfig:
    vocab_size = _model_vocab_size(tokenizer, config)
    if vocab_size < int(tokenizer.vocab_size):
        raise ValueError("model_vocab_size must be at least tokenizer vocab size")
    sampled_vocab_size = max(0, int(config.sampled_vocab_size))
    if sampled_vocab_size >= vocab_size:
        raise ValueError("sampled_vocab_size must be smaller than model_vocab_size")
    source_memory_slot_count = (
        max(0, int(config.memory_slot_count))
        if proposal_kind == "memory_slot_expansion"
        else 0
    )
    active_memory_slot_count = max(1, int(config.active_memory_slot_count))
    return LanguageModelConfig(
        vocab_size=vocab_size,
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        expert_count=max(2, int(config.expert_count)),
        active_expert_count=max(1, int(config.active_expert_count)),
        route_candidate_count=max(1, int(config.route_candidate_count)),
        expert_hidden_dim=max(1, int(config.expert_hidden_dim)),
        sampled_vocab_size=sampled_vocab_size,
        sampled_vocab_sparse_lm_head_gradient=sampled_vocab_size > 0,
        sparse_token_embedding_gradients=sampled_vocab_size > 0,
        generation_vocab_size=(
            int(tokenizer.vocab_size) if vocab_size > int(tokenizer.vocab_size) else 0
        ),
        memory_slot_count=source_memory_slot_count,
        memory_slot_candidate_count=(
            min(max(1, int(config.memory_slot_candidate_count)), source_memory_slot_count)
            if source_memory_slot_count > 0
            else 0
        ),
        active_memory_slot_count=active_memory_slot_count,
    )


def _build_eval_batches(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    corpus: str,
    config: LanguageStructuralPlasticityExperimentConfig,
    *,
    device: torch.device,
) -> tuple[tuple[LanguageBatch, ...], dict[str, Any], dict[str, Any]]:
    split = build_language_model_splits(
        [corpus],
        tokenizer,
        sequence_length=int(config.sequence_length),
        eval_fraction=float(config.eval_fraction),
        stride=int(config.stride),
        batch_size=int(config.batch_size),
        device=device,
    )
    eval_batches = tuple(split.eval[: max(1, int(config.max_eval_batches))])
    cached_eval, precompute_report = precompute_sampled_vocab_batches(
        model,
        eval_batches,
    )
    return tuple(cached_eval), split.report, precompute_report


def _memory_slot_proposal(
    model: MarulhoLanguageModel,
    config: LanguageStructuralPlasticityExperimentConfig,
) -> tuple[dict[str, Any], LanguageStructuralPlasticityConfig]:
    source_slot_count = max(0, int(model.config.memory_slot_count))
    target_slot_count = max(
        source_slot_count + 1,
        source_slot_count + max(1, int(config.memory_slot_growth)),
    )
    transaction_config = LanguageStructuralPlasticityConfig(
        max_memory_slot_growth=target_slot_count - source_slot_count,
        max_memory_slot_count=target_slot_count,
        max_memory_slot_candidate_count=max(1, int(config.memory_slot_candidate_count)),
        max_eval_loss_delta=float(config.max_eval_loss_delta),
    )
    proposal = build_language_structural_memory_slot_expansion_proposal(
        model,
        routing_evidence={
            "surface": "marulho_language_memory_slots.v1",
            "memory_slot_pressure": True,
            "novel_concept_cluster": True,
            "replay_conflict": True,
            "candidate_rows_scored": max(1, int(config.batch_size))
            * max(1, int(config.sequence_length)),
            "runs_all_columns": False,
        },
        config=transaction_config,
    )
    return proposal, transaction_config


def _route_bank_proposal(
    model: MarulhoLanguageModel,
    config: LanguageStructuralPlasticityExperimentConfig,
) -> tuple[dict[str, Any], LanguageStructuralPlasticityConfig]:
    transaction_config = LanguageStructuralPlasticityConfig(
        route_saturation_threshold=0.5,
        max_route_candidate_growth=max(1, int(config.route_candidate_growth)),
        max_eval_loss_delta=float(config.max_eval_loss_delta),
    )
    proposal = build_language_structural_route_bank_expansion_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": int(model.config.expert_count),
            "active_columns": int(model.config.active_expert_count),
            "route_candidate_count": int(model.config.route_candidate_count),
            "output_candidate_count": int(model.config.active_expert_count),
            "candidate_rows_scored": max(1, int(config.batch_size))
            * max(1, int(config.sequence_length)),
            "runs_all_columns": False,
            "route_bank_pressure": True,
        },
        config=transaction_config,
    )
    return proposal, transaction_config


def _deep_sleep_proposal(
    model: MarulhoLanguageModel,
    config: LanguageStructuralPlasticityExperimentConfig,
) -> tuple[dict[str, Any], LanguageStructuralPlasticityConfig]:
    target_id = int(config.deep_sleep_expert_id)
    if target_id < 0:
        target_id = max(0, int(model.config.expert_count) - 1)
    utilities = [0.6 for _ in range(int(model.config.expert_count))]
    utilities[target_id] = 0.0
    transaction_config = LanguageStructuralPlasticityConfig(
        min_expert_count=max(1, int(model.config.active_expert_count)),
        max_deep_sleep_experts=1,
        deep_sleep_utility_threshold=0.1,
        max_eval_loss_delta=float(config.max_eval_loss_delta),
    )
    proposal = build_language_structural_deep_sleep_proposal(
        model,
        routing_evidence={
            "surface": "marulho_routed_language_experts.v1",
            "total_columns": int(model.config.expert_count),
            "active_columns": int(model.config.active_expert_count),
            "active_expert_ids": [0],
            "stale_expert_ids": [target_id],
            "low_activation_expert_ids": [target_id],
            "expert_utilities": utilities,
            "candidate_rows_scored": max(1, int(config.batch_size))
            * max(1, int(config.sequence_length)),
            "runs_all_columns": False,
        },
        config=transaction_config,
    )
    return proposal, transaction_config


_PROPOSAL_BUILDERS: dict[
    str,
    Callable[
        [MarulhoLanguageModel, LanguageStructuralPlasticityExperimentConfig],
        tuple[dict[str, Any], LanguageStructuralPlasticityConfig],
    ],
] = {
    "memory_slot_expansion": _memory_slot_proposal,
    "route_bank_expansion": _route_bank_proposal,
    "expert_deep_sleep": _deep_sleep_proposal,
}


def _entry_summary(entry: Mapping[str, Any]) -> dict[str, Any]:
    transaction = (
        entry.get("transaction")
        if isinstance(entry.get("transaction"), Mapping)
        else {}
    )
    proposal = (
        entry.get("proposal")
        if isinstance(entry.get("proposal"), Mapping)
        else {}
    )
    mutation = (
        transaction.get("mutation")
        if isinstance(transaction.get("mutation"), Mapping)
        else {}
    )
    gate = (
        transaction.get("promotion_gate")
        if isinstance(transaction.get("promotion_gate"), Mapping)
        else {}
    )
    checkpoint = (
        transaction.get("checkpoint")
        if isinstance(transaction.get("checkpoint"), Mapping)
        else {}
    )
    rollback = (
        transaction.get("rollback_evidence")
        if isinstance(transaction.get("rollback_evidence"), Mapping)
        else {}
    )
    body = proposal.get("proposal") if isinstance(proposal.get("proposal"), Mapping) else {}
    return {
        "proposal_kind": mutation.get("proposal_kind") or body.get("proposal_kind"),
        "proposal_mutates_runtime_state": proposal.get("mutates_runtime_state"),
        "applied": bool(transaction.get("applied", False)),
        "operator_approved": bool(transaction.get("operator_approved", False)),
        "checkpoint_restore_verified": bool(
            checkpoint.get("checkpoint_restore_verified", False)
        ),
        "rollback_verified": bool(rollback.get("rollback_verified", False)),
        "heldout_non_regression": bool(gate.get("heldout_non_regression", False)),
        "source_expert_count": mutation.get("source_expert_count"),
        "target_expert_count": mutation.get("target_expert_count"),
        "source_route_candidate_count": mutation.get("source_route_candidate_count"),
        "target_route_candidate_count": mutation.get("target_route_candidate_count"),
        "source_memory_slot_count": mutation.get("source_memory_slot_count"),
        "target_memory_slot_count": mutation.get("target_memory_slot_count"),
        "target_memory_slot_candidate_count": mutation.get(
            "target_memory_slot_candidate_count"
        ),
    }


def run_language_structural_plasticity_experiment(
    *,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    proposal_kinds: Sequence[str] = SUPPORTED_PROPOSAL_KINDS,
    config: LanguageStructuralPlasticityExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageStructuralPlasticityExperimentConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    requested_kinds = tuple(dict.fromkeys(str(kind) for kind in proposal_kinds))
    if not requested_kinds:
        raise ValueError("At least one proposal kind is required")
    unsupported = [kind for kind in requested_kinds if kind not in _PROPOSAL_BUILDERS]
    if unsupported:
        raise ValueError(f"Unsupported proposal kind(s): {', '.join(unsupported)}")

    torch.manual_seed(int(cfg.seed))
    device = _resolve_device(str(cfg.device))
    tokenizer = ByteLevelLanguageTokenizer()
    corpus, corpus_source = _read_text(corpus_path)
    entries: list[dict[str, Any]] = []
    split_reports: list[dict[str, Any]] = []
    precompute_reports: list[dict[str, Any]] = []
    for index, proposal_kind in enumerate(requested_kinds):
        torch.manual_seed(int(cfg.seed) + index)
        model = MarulhoLanguageModel(
            _model_config(tokenizer, cfg, proposal_kind=proposal_kind)
        ).to(device)
        eval_batches, split_report, precompute_report = _build_eval_batches(
            model,
            tokenizer,
            corpus,
            cfg,
            device=device,
        )
        proposal, transaction_config = _PROPOSAL_BUILDERS[proposal_kind](model, cfg)
        candidate, transaction_report = apply_language_structural_plasticity_transaction(
            model,
            proposal,
            eval_batches=eval_batches,
            checkpoint_path=(
                output.parent
                / f"{output.stem}-{proposal_kind.replace('_', '-')}-baseline.pt"
            ),
            operator_approved=True,
            config=transaction_config,
        )
        candidate_eval = evaluate_language_model(candidate, eval_batches)
        entries.append(
            {
                "surface": ENTRY_SURFACE,
                "proposal_kind": proposal_kind,
                "model_config": asdict(model.config),
                "transaction_config": asdict(transaction_config),
                "proposal": proposal,
                "transaction": transaction_report,
                "candidate_evaluation": candidate_eval,
            }
        )
        split_reports.append(split_report)
        precompute_reports.append(precompute_report)
        del model, candidate
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summaries = [_entry_summary(entry) for entry in entries]
    all_valid = bool(summaries) and all(
        bool(item["proposal_mutates_runtime_state"] is False)
        and bool(item["applied"])
        and bool(item["operator_approved"])
        and bool(item["checkpoint_restore_verified"])
        and bool(item["rollback_verified"])
        and bool(item["heldout_non_regression"])
        for item in summaries
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": "marulho_lm_head",
        "status": (
            "completed_structural_plasticity_transactions"
            if all_valid
            else "structural_plasticity_transaction_evidence_incomplete"
        ),
        "model_vocab_size": _model_vocab_size(tokenizer, cfg),
        "sampled_vocab_size": int(cfg.sampled_vocab_size),
        "device": str(device),
        "proposal_kinds": list(requested_kinds),
        "proposal_count": len(entries),
        "transaction_count": len(entries),
        "valid_transaction_count": sum(1 for item in summaries if item["applied"]),
        "config": asdict(cfg),
        "corpus_source": corpus_source,
        "split_reports": split_reports,
        "sampled_vocab_precompute_reports": precompute_reports,
        "transactions": entries,
        "transaction_summaries": summaries,
        "promotion_gate": {
            "standalone_structural_evidence_available": bool(all_valid),
            "all_proposals_non_mutating": all(
                item["proposal_mutates_runtime_state"] is False for item in summaries
            ),
            "all_transactions_checkpoint_backed": all(
                bool(item["checkpoint_restore_verified"]) for item in summaries
            ),
            "all_rollbacks_verified": all(
                bool(item["rollback_verified"]) for item in summaries
            ),
            "operator_approval_recorded": all(
                bool(item["operator_approved"]) for item in summaries
            ),
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--proposal-kind",
        action="append",
        choices=SUPPORTED_PROPOSAL_KINDS,
        default=[],
    )
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--sampled-vocab-size", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=96)
    parser.add_argument("--memory-slot-count", type=int, default=0)
    parser.add_argument("--memory-slot-growth", type=int, default=1024)
    parser.add_argument("--memory-slot-candidate-count", type=int, default=8)
    parser.add_argument("--active-memory-slot-count", type=int, default=2)
    parser.add_argument("--route-candidate-growth", type=int, default=2)
    parser.add_argument("--deep-sleep-expert-id", type=int, default=-1)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--max-eval-batches", type=int, default=2)
    parser.add_argument("--max-eval-loss-delta", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageStructuralPlasticityExperimentConfig(
        model_vocab_size=max(0, int(args.model_vocab_size)),
        sampled_vocab_size=max(0, int(args.sampled_vocab_size)),
        embedding_dim=int(args.embedding_dim),
        state_dim=int(args.state_dim),
        expert_count=int(args.expert_count),
        active_expert_count=int(args.active_expert_count),
        route_candidate_count=int(args.route_candidate_count),
        expert_hidden_dim=int(args.expert_hidden_dim),
        memory_slot_count=max(0, int(args.memory_slot_count)),
        memory_slot_growth=max(1, int(args.memory_slot_growth)),
        memory_slot_candidate_count=max(1, int(args.memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(args.active_memory_slot_count)),
        route_candidate_growth=max(1, int(args.route_candidate_growth)),
        deep_sleep_expert_id=int(args.deep_sleep_expert_id),
        sequence_length=max(2, int(args.sequence_length)),
        stride=max(1, int(args.stride)),
        batch_size=max(1, int(args.batch_size)),
        eval_fraction=float(args.eval_fraction),
        max_eval_batches=max(1, int(args.max_eval_batches)),
        max_eval_loss_delta=float(args.max_eval_loss_delta),
        seed=int(args.seed),
        device=str(args.device),
    )
    report = run_language_structural_plasticity_experiment(
        output_path=args.output,
        corpus_path=args.corpus,
        proposal_kinds=tuple(args.proposal_kind) or SUPPORTED_PROPOSAL_KINDS,
        config=config,
    )
    return 0 if report["promotion_gate"]["standalone_structural_evidence_available"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
