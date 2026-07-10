"""Train and evaluate the active Transformer-only MARULHO language model."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from marulho.data.language_tokenizer import (
    BPE_TRAINING_CHUNK_CHARACTERS,
    ByteLevelLanguageTokenizer,
    BytePairLanguageTokenizer,
    LanguageTokenizer,
)
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


SURFACE = "marulho_transformer_training_experiment.v4"
ARTIFACT_KIND = "marulho_transformer_training_experiment"

DEFAULT_CORPUS = (
    "MARULHO owns a causal language model and learns from local source windows. "
    "Checkpoints preserve the trained tokenizer and every durable parameter. "
    "Heldout language quality decides whether an architecture survives.\n"
) * 64


@dataclass(frozen=True)
class LanguageTrainingExperimentConfig:
    tokenizer_kind: str = "bpe"
    tokenizer_vocab_size: int = 4096
    tokenizer_min_frequency: int = 2
    embedding_dim: int = 256
    state_dim: int = 256
    state_core: str = "transformer"
    state_layers: int = 4
    attention_heads: int = 8
    transformer_context_length: int = 512
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    tie_embeddings: bool = True
    sequence_length: int = 256
    stride: int = 128
    batch_size: int = 8
    max_train_batches: int = 512
    max_eval_batches: int = 128
    window_selection: str = "stratified"
    train_epochs: int = 1
    learning_rate: float = 5e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    max_grad_norm: float = 1.0
    precision: str = "float32"
    generation_tokens: int = 128
    generation_repetition_penalty: float = 1.1
    generation_no_repeat_ngram_size: int = 3
    sustained_target_tokens: int = 0
    sustained_timeout_seconds: float = 600.0
    cuda_allow_tf32: bool = True
    device: str = "auto"


def _resolve_device(name: str) -> torch.device:
    if str(name) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def _read_corpus(path: str | Path | None) -> str:
    if path is None:
        return DEFAULT_CORPUS
    corpus = Path(path).read_text(encoding="utf-8")
    if not corpus.strip():
        raise ValueError(f"Language corpus is empty: {path}")
    return corpus


def _build_tokenizer(
    corpus: str | Sequence[str],
    config: LanguageTrainingExperimentConfig,
) -> LanguageTokenizer:
    texts = [corpus] if isinstance(corpus, str) else [str(text) for text in corpus]
    kind = str(config.tokenizer_kind).strip().lower()
    if kind == "byte":
        return ByteLevelLanguageTokenizer()
    if kind == "bpe":
        return BytePairLanguageTokenizer.train(
            texts,
            vocab_size=max(512, int(config.tokenizer_vocab_size)),
            min_frequency=max(1, int(config.tokenizer_min_frequency)),
        )
    raise ValueError("tokenizer_kind must be 'byte' or 'bpe'")


def _model_config(
    tokenizer: LanguageTokenizer,
    config: LanguageTrainingExperimentConfig,
) -> LanguageModelConfig:
    if str(config.state_core).strip().lower() != "transformer":
        raise ValueError("The active language experiment supports only Transformer models")
    return LanguageModelConfig(
        vocab_size=int(tokenizer.vocab_size),
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        state_core="transformer",
        state_layers=int(config.state_layers),
        attention_heads=int(config.attention_heads),
        transformer_context_length=int(config.transformer_context_length),
        transformer_mlp_ratio=float(config.transformer_mlp_ratio),
        transformer_dropout=float(config.transformer_dropout),
        tie_embeddings=bool(config.tie_embeddings),
    )


def _precision_context(
    device: torch.device,
    precision: str,
):
    kind = str(precision).strip().lower()
    if kind == "float32" or device.type != "cuda":
        return nullcontext()
    if kind == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if kind == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("precision must be float32, bfloat16, or float16")


def _optimizer(model: MarulhoLanguageModel, config: LanguageTrainingExperimentConfig):
    kwargs = {
        "lr": float(config.learning_rate),
        "betas": (float(config.adam_beta1), float(config.adam_beta2)),
        "weight_decay": float(config.weight_decay),
    }
    if model.device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


def _learning_rate(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    peak: float,
    minimum_fraction: float,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return peak * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps - 1))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak * (minimum_fraction + (1.0 - minimum_fraction) * cosine)


def _train(
    model: MarulhoLanguageModel,
    batches: Sequence[LanguageBatch],
    config: LanguageTrainingExperimentConfig,
) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one train batch is required")
    optimizer, fused = _optimizer(model, config)
    epochs = max(1, int(config.train_epochs))
    total_steps = epochs * len(batches)
    warmup_steps = int(round(total_steps * max(0.0, float(config.warmup_fraction))))
    use_scaler = model.device.type == "cuda" and str(config.precision).lower() == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    losses: list[torch.Tensor] = []
    gradient_norms: list[torch.Tensor] = []
    token_count = 0
    model.train()
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    step = 0
    for _epoch in range(epochs):
        for batch in batches:
            device_batch = batch.to(model.device)
            lr = _learning_rate(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak=float(config.learning_rate),
                minimum_fraction=max(0.0, float(config.minimum_learning_rate_fraction)),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(model.device, config.precision):
                result = model.next_token_loss(
                    device_batch.input_ids,
                    device_batch.target_ids,
                    collect_telemetry=False,
                    return_evidence=False,
                )
                loss = result["loss"]
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max(0.0, float(config.max_grad_norm)),
            )
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            losses.append(loss.detach().float())
            gradient_norms.append(gradient_norm.detach().float())
            token_count += int(device_batch.target_ids.numel())
            step += 1
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    loss_values = (
        torch.stack(losses).cpu().tolist()
        if losses
        else []
    )
    gradient_norm_max = (
        float(torch.stack(gradient_norms).max().cpu().item())
        if gradient_norms
        else 0.0
    )
    return {
        "surface": "marulho_transformer_training_update.v3",
        "optimizer": "AdamW",
        "fused_optimizer": fused,
        "precision": str(config.precision),
        "batch_transfer_policy": "cpu_split_per_batch_to_model_device",
        "train_epochs": epochs,
        "optimizer_step_count": total_steps,
        "warmup_steps": warmup_steps,
        "learning_rate_peak": float(config.learning_rate),
        "learning_rate_final": float(optimizer.param_groups[0]["lr"]),
        "weight_decay": float(config.weight_decay),
        "token_count": token_count,
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(token_count) / elapsed,
        "loss_start": loss_values[0] if loss_values else None,
        "loss_end": loss_values[-1] if loss_values else None,
        "mean_loss_first_8": (
            sum(loss_values[:8]) / max(1, len(loss_values[:8]))
        ),
        "mean_loss_last_8": (
            sum(loss_values[-8:]) / max(1, len(loss_values[-8:]))
        ),
        "max_gradient_norm": gradient_norm_max,
        "loss_record_count": len(loss_values),
        "per_step_host_metric_readback": False,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(model.device))
            if model.device.type == "cuda"
            else 0
        ),
        "external_llm_used": False,
    }


def _parameter_inventory(model: MarulhoLanguageModel) -> dict[str, int]:
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "embedding_parameters": model.token_embedding.weight.numel(),
        "transformer_parameters": sum(
            parameter.numel() for parameter in model.state_block.parameters()
        ),
        "tied_embedding_head": int(
            model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
        ),
    }


def _decoded_generation(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    config: LanguageTrainingExperimentConfig,
) -> dict[str, Any]:
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt, add_bos=True, add_eos=False),
        dtype=torch.long,
        device=model.device,
    )
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        repetition_penalty=max(1.0, float(config.generation_repetition_penalty)),
        no_repeat_ngram_size=max(0, int(config.generation_no_repeat_ngram_size)),
    )
    ids = [int(value) for value in generated["generated_ids"][0].cpu().tolist()]
    continuation = ids[int(prompt_ids.numel()) :]
    return {
        "prompt": prompt,
        "generated_text": tokenizer.decode(ids),
        "continuation_text": tokenizer.decode(continuation),
        "prompt_token_count": int(prompt_ids.numel()),
        "continuation_token_count": len(continuation),
        "sequence_hash": tokenizer.sequence_hash(ids),
        "external_llm_used": False,
        "owned_by_marulho": True,
        "generation_decode": generated["generation_decode"],
    }


def run_language_training_experiment(
    *,
    output_path: str | Path,
    corpus_path: str | Path | None = None,
    prompts: Sequence[str] = ("MARULHO", "The system"),
    config: LanguageTrainingExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageTrainingExperimentConfig()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    corpus = _read_corpus(corpus_path)
    tokenizer = _build_tokenizer(corpus, cfg)
    device = _resolve_device(cfg.device)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.cuda_allow_tf32)
    try:
        if int(cfg.sequence_length) > int(cfg.transformer_context_length):
            raise ValueError("sequence_length cannot exceed transformer_context_length")
        split = build_language_model_splits(
            [corpus],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=0.20,
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device="cpu",
            max_train_batches=int(cfg.max_train_batches),
            max_eval_batches=int(cfg.max_eval_batches),
            window_selection=str(cfg.window_selection),
        )
        model = MarulhoLanguageModel(_model_config(tokenizer, cfg)).to(device)
        eval_before = evaluate_language_model(model, split.eval)
        training = _train(model, split.train, cfg)
        eval_after = evaluate_language_model(model, split.eval)
        generations = [
            _decoded_generation(
                model,
                tokenizer,
                prompt=str(prompt),
                max_new_tokens=max(0, int(cfg.generation_tokens)),
                config=cfg,
            )
            for prompt in prompts
        ]
        checkpoint = save_language_model_checkpoint(
            output.with_name(f"{output.stem}-checkpoint.pt"),
            model,
            tokenizer,
            metadata={
                "experiment_report": str(output),
                "split": split.report,
                "config": asdict(cfg),
            },
        )
        sustained: dict[str, Any] | None = None
        sustained_path: Path | None = None
        if int(cfg.sustained_target_tokens) > 0:
            sustained_path = output.with_name(f"{output.stem}-sustained.json")
            sustained = run_language_sustained_runtime_evidence(
                model,
                tokenizer,
                output_path=sustained_path,
                target_tokens=int(cfg.sustained_target_tokens),
                checkpoint_path=checkpoint,
                prompt=str(prompts[0] if prompts else "MARULHO"),
                timeout_seconds=float(cfg.sustained_timeout_seconds),
                generation_repetition_penalty=float(cfg.generation_repetition_penalty),
                generation_no_repeat_ngram_size=int(cfg.generation_no_repeat_ngram_size),
                collect_environment=False,
            )
        loss_delta = float(eval_after["heldout_loss"]) - float(eval_before["heldout_loss"])
        corpus_token_count = int(split.report["train_text_token_count"])
        report = {
            "artifact_kind": ARTIFACT_KIND,
            "surface": SURFACE,
            "output_path": str(output),
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "active_language_path": model.config.active_language_path,
            "state_core": "transformer",
            "parameter_inventory": _parameter_inventory(model),
            "model_vocab_size": int(model.config.vocab_size),
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "tokenizer": {
                "surface": tokenizer.state_dict().get("surface"),
                "vocabulary_hash": tokenizer.vocabulary_hash(),
                "vocab_size": int(tokenizer.vocab_size),
                "corpus_token_count": corpus_token_count,
                "corpus_utf8_byte_count": len(corpus.encode("utf-8")),
                "bytes_per_token": len(corpus.encode("utf-8"))
                / max(1, corpus_token_count),
                "vocabulary_trained_by_marulho": bool(
                    tokenizer.state_dict().get("vocabulary_trained_by_marulho", False)
                ),
                "training_chunk_characters": BPE_TRAINING_CHUNK_CHARACTERS,
            },
            "corpus": {
                "source": "default_inline" if corpus_path is None else str(corpus_path),
                "character_count": len(corpus),
            },
            "config": asdict(cfg),
            "model_config": asdict(model.config),
            "split": split.report,
            "training": training,
            "eval_before": eval_before,
            "eval_after": eval_after,
            "language_delta": {
                "heldout_loss_delta": loss_delta,
                "heldout_loss_improved": loss_delta < 0.0,
            },
            "generation_after": generations,
            "checkpoint_path": str(checkpoint),
            "sustained_report_path": None if sustained_path is None else str(sustained_path),
            "sustained_summary": None
            if sustained is None
            else {
                "success": bool(sustained["success"]),
                "target_tokens": int(sustained["target_tokens"]),
                "token_delta": int(sustained["token_delta"]),
                "tokens_per_second": float(sustained["tokens_per_second"]),
            },
            "experiment_review": {
                "base_quality_first": True,
                "recurrent_language_path_present": False,
                "routing_present": False,
                "sampled_padded_vocab_present": False,
                "promotes_generation_quality_claim": False,
                "promotes_runtime_claim": False,
            },
        }
        write_json_report_with_readme(output, report)
        return report
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--tokenizer-kind", choices=("byte", "bpe"), default="bpe")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--state-layers", type=int, default=4)
    parser.add_argument("--attention-heads", type=int, default=8)
    parser.add_argument("--transformer-context-length", type=int, default=512)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--untie-embeddings", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=512)
    parser.add_argument("--max-eval-batches", type=int, default=128)
    parser.add_argument("--window-selection", choices=("stratified", "prefix"), default="stratified")
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--minimum-learning-rate-fraction", type=float, default=0.10)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--precision", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--sustained-target-tokens", type=int, default=0)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageTrainingExperimentConfig(
        tokenizer_kind=args.tokenizer_kind,
        tokenizer_vocab_size=max(512, int(args.tokenizer_vocab_size)),
        tokenizer_min_frequency=max(1, int(args.tokenizer_min_frequency)),
        embedding_dim=int(args.embedding_dim),
        state_dim=int(args.state_dim),
        state_layers=max(1, int(args.state_layers)),
        attention_heads=max(1, int(args.attention_heads)),
        transformer_context_length=max(2, int(args.transformer_context_length)),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        transformer_dropout=float(args.transformer_dropout),
        tie_embeddings=not bool(args.untie_embeddings),
        sequence_length=int(args.sequence_length),
        stride=int(args.stride),
        batch_size=max(1, int(args.batch_size)),
        max_train_batches=max(1, int(args.max_train_batches)),
        max_eval_batches=max(1, int(args.max_eval_batches)),
        window_selection=args.window_selection,
        train_epochs=max(1, int(args.train_epochs)),
        learning_rate=float(args.learning_rate),
        minimum_learning_rate_fraction=float(args.minimum_learning_rate_fraction),
        warmup_fraction=float(args.warmup_fraction),
        weight_decay=float(args.weight_decay),
        max_grad_norm=float(args.max_grad_norm),
        precision=args.precision,
        generation_tokens=max(0, int(args.generation_tokens)),
        generation_repetition_penalty=max(1.0, float(args.generation_repetition_penalty)),
        generation_no_repeat_ngram_size=max(0, int(args.generation_no_repeat_ngram_size)),
        sustained_target_tokens=max(0, int(args.sustained_target_tokens)),
        sustained_timeout_seconds=float(args.sustained_timeout_seconds),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        device=args.device,
    )
    report = run_language_training_experiment(
        output_path=args.output,
        corpus_path=args.corpus,
        prompts=tuple(args.prompt) or ("MARULHO", "The system"),
        config=config,
    )
    return 0 if bool(report["language_delta"]["heldout_loss_improved"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
