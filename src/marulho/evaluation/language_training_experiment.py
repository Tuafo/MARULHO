"""Fast mutable training experiment runner for the MARULHO LM head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.evaluation.language_sustained_runtime_evidence import (
    run_language_sustained_runtime_evidence,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    save_language_model_checkpoint,
)


SURFACE = "marulho_language_training_experiment.v1"
ARTIFACT_KIND = "marulho_language_training_experiment"


DEFAULT_CORPUS = (
    "MARULHO learns runtime evidence from local source windows. "
    "The language model routes sparse experts, records checkpoints, and keeps "
    "external LLM generation disabled. "
    "Replay protects old evidence while new source text updates the LM head. "
    "Structural pressure can grow, prune, merge, or sleep experts under review. "
    "Long sustained runs measure token throughput, spike health, and fallback "
    "truth before any runtime claim. "
) * 24


@dataclass(frozen=True)
class LanguageTrainingExperimentConfig:
    embedding_dim: int = 32
    state_dim: int = 64
    expert_count: int = 8
    active_expert_count: int = 2
    route_candidate_count: int = 4
    expert_hidden_dim: int = 96
    adaptive_timestep_budget: int = 1
    sequence_length: int = 32
    stride: int = 16
    max_train_batches: int = 64
    train_epochs: int = 2
    learning_rate: float = 2e-3
    max_grad_norm: float = 1.0
    generation_tokens: int = 48
    sustained_target_tokens: int = 512
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 120.0
    device: str = "auto"


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device was requested but torch.cuda.is_available() is false")
    return resolved


def _read_corpus(corpus_path: str | Path | None) -> str:
    if corpus_path is None:
        return DEFAULT_CORPUS
    path = Path(corpus_path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Language experiment corpus is empty: {path}")
    return text


def _model_config(
    tokenizer: ByteLevelLanguageTokenizer,
    config: LanguageTrainingExperimentConfig,
) -> LanguageModelConfig:
    return LanguageModelConfig(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        adaptive_timestep_budget=int(config.adaptive_timestep_budget),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
    )


def _trim_batches(
    batches: Sequence[LanguageBatch],
    *,
    limit: int,
) -> tuple[LanguageBatch, ...]:
    if int(limit) <= 0:
        return tuple(batches)
    return tuple(batches[: int(limit)])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _decoded_generation(
    model: MarulhoLanguageModel,
    tokenizer: ByteLevelLanguageTokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt, add_eos=False),
        dtype=torch.long,
    )
    generation = model.generate(
        prompt_ids,
        max_new_tokens=max(0, int(max_new_tokens)),
        eos_id=tokenizer.eos_id,
    )
    generated_ids = [
        int(token_id)
        for token_id in generation["generated_ids"].detach().cpu().reshape(-1).tolist()
    ]
    prompt_count = int(prompt_ids.numel())
    continuation_ids = generated_ids[prompt_count:]
    return {
        "surface": generation["surface"],
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated_ids),
        "continuation_text": tokenizer.decode(continuation_ids),
        "prompt_token_count": prompt_count,
        "generated_token_count": len(generated_ids),
        "new_token_count": int(generation["new_token_count"]),
        "sequence_hash": tokenizer.sequence_hash(generated_ids),
        "continuation_sequence_hash": tokenizer.sequence_hash(continuation_ids),
        "active_language_path": generation["active_language_path"],
        "external_llm_used": bool(generation["external_llm_used"]),
        "owned_by_marulho": bool(generation["owned_by_marulho"]),
    }


def _train_language_model(
    model: MarulhoLanguageModel,
    batches: Sequence[LanguageBatch],
    *,
    config: LanguageTrainingExperimentConfig,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one train batch is required")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
    model.train()
    token_count = 0
    losses: list[float] = []
    grad_norms: list[float] = []
    started = time.perf_counter()
    for _epoch in range(max(1, int(config.train_epochs))):
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            result = model.next_token_loss(
                batch.input_ids.to(model.device),
                batch.target_ids.to(model.device),
            )
            loss = result["loss"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(config.max_grad_norm),
            )
            optimizer.step()
            token_count += int(batch.target_ids.numel())
            losses.append(float(loss.detach().cpu().item()))
            grad_norms.append(float(grad_norm.detach().cpu().item()))
    elapsed = max(0.0, time.perf_counter() - started)
    return {
        "surface": "marulho_language_training_experiment_update.v1",
        "train_batch_count": len(batches),
        "train_epochs": max(1, int(config.train_epochs)),
        "optimizer": "AdamW",
        "learning_rate": float(config.learning_rate),
        "token_count": int(token_count),
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(token_count) / elapsed if elapsed > 0.0 else 0.0,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_delta": (losses[-1] - losses[0]) if len(losses) >= 2 else 0.0,
        "mean_loss_first_8": _mean(losses[:8]),
        "mean_loss_last_8": _mean(losses[-8:]),
        "max_gradient_norm": max(grad_norms) if grad_norms else 0.0,
        "device": str(model.device),
    }


def run_language_training_experiment(
    *,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    prompts: Sequence[str] = ("MARULHO", "runtime truth"),
    config: LanguageTrainingExperimentConfig | None = None,
) -> dict[str, Any]:
    """Train the LM head, generate text, run sustained inference, and report it."""

    cfg = config or LanguageTrainingExperimentConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = ByteLevelLanguageTokenizer()
    corpus = _read_corpus(corpus_path)
    split = build_language_model_splits(
        [corpus],
        tokenizer,
        sequence_length=int(cfg.sequence_length),
        eval_fraction=0.20,
        stride=int(cfg.stride),
    )
    train_batches = _trim_batches(split.train, limit=int(cfg.max_train_batches))
    device = _resolve_device(str(cfg.device))
    model = MarulhoLanguageModel(_model_config(tokenizer, cfg)).to(device)

    before_eval = evaluate_language_model(model, split.eval)
    before_generations = [
        _decoded_generation(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=min(16, int(cfg.generation_tokens)),
        )
        for prompt in prompts
    ]
    training = _train_language_model(model, train_batches, config=cfg)
    after_eval = evaluate_language_model(model, split.eval)
    after_generations = [
        _decoded_generation(
            model,
            tokenizer,
            prompt=prompt,
            max_new_tokens=int(cfg.generation_tokens),
        )
        for prompt in prompts
    ]

    checkpoint_path = save_language_model_checkpoint(
        output.with_name(f"{output.stem}-checkpoint.pt"),
        model,
        tokenizer,
        metadata={
            "experiment_report": str(output),
            "split": split.report,
            "config": asdict(cfg),
        },
    )
    sustained_path = output.with_name(f"{output.stem}-sustained.json")
    sustained_report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=sustained_path,
        target_tokens=max(1, int(cfg.sustained_target_tokens)),
        checkpoint_path=checkpoint_path,
        checkpoint_metadata={"experiment_report": str(output)},
        prompt=prompts[0] if prompts else "MARULHO",
        tick_tokens=int(cfg.sustained_tick_tokens),
        quantum_tokens=int(cfg.sustained_quantum_tokens),
        timeout_seconds=float(cfg.sustained_timeout_seconds),
        collect_environment=False,
    )

    loss_delta = float(after_eval["heldout_loss"]) - float(before_eval["heldout_loss"])
    perplexity_delta = (
        float(after_eval["heldout_perplexity"])
        - float(before_eval["heldout_perplexity"])
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "corpus": {
            "source": "default_inline" if corpus_path is None else str(corpus_path),
            "character_count": len(corpus),
        },
        "config": asdict(cfg),
        "model_config": asdict(model.config),
        "split": split.report,
        "training": training,
        "eval_before": before_eval,
        "eval_after": after_eval,
        "language_delta": {
            "heldout_loss_delta": loss_delta,
            "heldout_perplexity_delta": perplexity_delta,
            "heldout_loss_improved": loss_delta < 0.0,
            "heldout_perplexity_improved": perplexity_delta < 0.0,
        },
        "generation_before": before_generations,
        "generation_after": after_generations,
        "checkpoint_path": str(checkpoint_path),
        "sustained_report_path": str(sustained_path),
        "sustained_summary": {
            "report_status": sustained_report["report_status"],
            "success": bool(sustained_report["success"]),
            "target_tokens": int(sustained_report["target_tokens"]),
            "token_delta": int(sustained_report["token_delta"]),
            "tokens_per_second": sustained_report["tokens_per_second"],
            "device_backend": sustained_report["device_backend"],
            "fallback_counts": sustained_report["fallback_counts"],
        },
        "experiment_review": {
            "fast_mutable_experiment": True,
            "records_actual_training": training["token_count"] > 0,
            "records_actual_generation": bool(after_generations),
            "records_sustained_inference": int(sustained_report["token_delta"]) > 0,
            "promotes_runtime_claim": False,
            "promotes_generation_quality_claim": False,
            "next_experiment": (
                "increase corpus/train tokens, compare checkpoints, and run "
                "8192/131072-token sustained LM evidence"
            ),
        },
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=96)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--generation-tokens", type=int, default=48)
    parser.add_argument("--sustained-target-tokens", type=int, default=512)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageTrainingExperimentConfig(
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        sequence_length=args.sequence_length,
        stride=args.stride,
        max_train_batches=args.max_train_batches,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        generation_tokens=args.generation_tokens,
        sustained_target_tokens=args.sustained_target_tokens,
        sustained_timeout_seconds=args.sustained_timeout_seconds,
        device=args.device,
    )
    report = run_language_training_experiment(
        output_path=args.output,
        corpus_path=args.corpus,
        prompts=tuple(args.prompt) or ("MARULHO", "runtime truth"),
        config=config,
    )
    return 0 if report["sustained_summary"]["success"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
