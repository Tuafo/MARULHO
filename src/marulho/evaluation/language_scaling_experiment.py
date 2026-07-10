"""Measure MARULHO Transformer quality across model-size and token budgets."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from marulho.data.language_tokenizer import BPE_TRAINING_CHUNK_CHARACTERS

from marulho.evaluation.language_training_experiment import (
    LanguageTrainingExperimentConfig,
    _build_tokenizer,
    _decoded_generation,
    _learning_rate,
    _model_config,
    _optimizer,
    _parameter_inventory,
    _precision_context,
    _read_corpus,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    LanguageBatch,
    MarulhoLanguageModel,
    build_language_model_splits,
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)


SURFACE = "marulho_transformer_scaling_experiment.v9"
ARTIFACT_KIND = "marulho_transformer_scaling_experiment"

DEFAULT_PROMPTS = (
    "A researcher places a red key inside a blue box. After moving the box",
    "The program failed because the cache",
    "When rain reaches dry soil, the first change is",
    "A continual learner should remember old skills while",
)


@dataclass(frozen=True)
class ScalingArmConfig:
    name: str
    width: int
    layers: int
    heads: int
    mlp_ratio: float = 4.0
    token_budget_multiplier: float = 1.0


DEFAULT_ARMS = (
    ScalingArmConfig("5m", width=256, layers=4, heads=8),
    ScalingArmConfig("19m", width=512, layers=4, heads=8),
    ScalingArmConfig("60m", width=768, layers=6, heads=12),
)


@dataclass(frozen=True)
class LanguageScalingExperimentConfig:
    tokenizer_vocab_size: int = 4096
    tokenizer_min_frequency: int = 2
    sequence_length: int = 256
    stride: int = 256
    batch_size: int = 4
    max_train_batches: int = 4096
    max_eval_batches: int = 128
    eval_fraction: float = 0.20
    token_budgets: tuple[int, ...] = (1_048_576, 2_097_152, 4_194_304)
    budget_basis: str = "equal_update_tokens"
    arms: tuple[ScalingArmConfig, ...] = DEFAULT_ARMS
    transformer_context_length: int = 512
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    max_grad_norm: float = 1.0
    precision: str = "bfloat16"
    generation_tokens: int = 96
    generation_repetition_penalty: float = 1.1
    generation_no_repeat_ngram_size: int = 3
    seed: int = 1337
    cuda_allow_tf32: bool = True
    retain_best_checkpoint_only: bool = True
    device: str = "auto"
    resume_checkpoint_path: str | None = None


def _seed_everything(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _arm_training_config(
    arm: ScalingArmConfig,
    config: LanguageScalingExperimentConfig,
) -> LanguageTrainingExperimentConfig:
    return LanguageTrainingExperimentConfig(
        tokenizer_kind="bpe",
        tokenizer_vocab_size=int(config.tokenizer_vocab_size),
        tokenizer_min_frequency=int(config.tokenizer_min_frequency),
        embedding_dim=int(arm.width),
        state_dim=int(arm.width),
        state_layers=int(arm.layers),
        attention_heads=int(arm.heads),
        transformer_context_length=int(config.transformer_context_length),
        transformer_mlp_ratio=float(arm.mlp_ratio),
        sequence_length=int(config.sequence_length),
        stride=int(config.stride),
        batch_size=int(config.batch_size),
        max_train_batches=int(config.max_train_batches),
        max_eval_batches=int(config.max_eval_batches),
        train_epochs=1,
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(
            config.minimum_learning_rate_fraction
        ),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        max_grad_norm=float(config.max_grad_norm),
        precision=str(config.precision),
        generation_tokens=int(config.generation_tokens),
        generation_repetition_penalty=float(
            config.generation_repetition_penalty
        ),
        generation_no_repeat_ngram_size=int(
            config.generation_no_repeat_ngram_size
        ),
        cuda_allow_tf32=bool(config.cuda_allow_tf32),
        device=str(config.device),
    )


def _inverse_softplus(value: float) -> float:
    bounded = max(float(value), 1.0e-6)
    return math.log(math.expm1(bounded))


def fit_language_scaling_law(points: Sequence[dict[str, float]]) -> dict[str, Any]:
    sizes = sorted({int(point["non_embedding_parameters"]) for point in points})
    budgets = sorted({int(point["update_tokens"]) for point in points})
    if len(points) < 9 or len(sizes) < 3 or len(budgets) < 3:
        return {
            "available": False,
            "reason": "requires_at_least_three_model_sizes_and_three_token_budgets",
            "point_count": len(points),
            "model_size_count": len(sizes),
            "token_budget_count": len(budgets),
        }

    n = torch.tensor(
        [point["non_embedding_parameters"] for point in points],
        dtype=torch.float64,
    )
    d = torch.tensor(
        [point["update_tokens"] for point in points],
        dtype=torch.float64,
    )
    observed = torch.tensor(
        [point["heldout_loss"] for point in points],
        dtype=torch.float64,
    )
    n_reference = float(torch.exp(torch.log(n).mean()).item())
    d_reference = float(torch.exp(torch.log(d).mean()).item())
    normalized_n = n / n_reference
    normalized_d = d / d_reference
    minimum_loss = float(observed.min().item())
    loss_span = max(float((observed.max() - observed.min()).item()), 0.1)

    best: tuple[float, torch.Tensor] | None = None
    for restart in range(4):
        _seed_everything(91_000 + restart)
        raw = torch.nn.Parameter(
            torch.tensor(
                [
                    1.0 + 0.1 * restart,
                    _inverse_softplus(loss_span * 0.5),
                    -0.5 + 0.1 * restart,
                    _inverse_softplus(loss_span * 0.5),
                    -0.5 - 0.1 * restart,
                ],
                dtype=torch.float64,
            )
        )
        optimizer = torch.optim.Adam([raw], lr=0.03)
        for _ in range(1200):
            optimizer.zero_grad(set_to_none=True)
            irreducible = minimum_loss * torch.sigmoid(raw[0])
            size_coefficient = F.softplus(raw[1])
            size_exponent = 0.01 + 1.49 * torch.sigmoid(raw[2])
            data_coefficient = F.softplus(raw[3])
            data_exponent = 0.01 + 1.49 * torch.sigmoid(raw[4])
            predicted = (
                irreducible
                + size_coefficient * normalized_n.pow(-size_exponent)
                + data_coefficient * normalized_d.pow(-data_exponent)
            )
            loss = (predicted - observed).pow(2).mean()
            loss.backward()
            optimizer.step()
        score = float(loss.detach().item())
        if best is None or score < best[0]:
            best = (score, raw.detach().clone())

    assert best is not None
    raw = best[1]
    irreducible = minimum_loss * torch.sigmoid(raw[0])
    size_coefficient = F.softplus(raw[1])
    size_exponent = 0.01 + 1.49 * torch.sigmoid(raw[2])
    data_coefficient = F.softplus(raw[3])
    data_exponent = 0.01 + 1.49 * torch.sigmoid(raw[4])
    predicted = (
        irreducible
        + size_coefficient * normalized_n.pow(-size_exponent)
        + data_coefficient * normalized_d.pow(-data_exponent)
    )
    residual = predicted - observed
    rmse = float(residual.pow(2).mean().sqrt().item())
    total_variance = float(
        (observed - observed.mean()).pow(2).sum().item()
    )
    residual_variance = float(residual.pow(2).sum().item())
    r_squared = (
        1.0 - residual_variance / total_variance
        if total_variance > 0.0
        else 0.0
    )
    return {
        "available": True,
        "formula": (
            "L=E+A*(N/N_reference)^(-alpha)"
            "+B*(D/D_reference)^(-beta)"
        ),
        "fit_variable_N": "non_embedding_transformer_parameters",
        "fit_variable_D": "cumulative_optimizer_update_tokens",
        "E": float(irreducible.item()),
        "A": float(size_coefficient.item()),
        "alpha": float(size_exponent.item()),
        "B": float(data_coefficient.item()),
        "beta": float(data_exponent.item()),
        "N_reference": n_reference,
        "D_reference": d_reference,
        "rmse": rmse,
        "r_squared": r_squared,
        "point_count": len(points),
        "provisional": True,
        "limitations": [
            "single_corpus",
            "single_seed_per_size",
            "small_model_range",
            "repeated_training_windows_after_one_selected_pass",
            "unknown_true_irreducible_loss",
        ],
        "predictions": [
            {
                **dict(point),
                "predicted_heldout_loss": float(predicted[index].item()),
                "residual": float(residual[index].item()),
            }
            for index, point in enumerate(points)
        ],
    }


def _next_batch_order(
    batch_count: int,
    *,
    generator: torch.Generator,
) -> list[int]:
    return torch.randperm(int(batch_count), generator=generator).tolist()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prior_training_progress(metadata: Mapping[str, Any]) -> tuple[int, int]:
    cumulative_tokens = metadata.get("cumulative_update_tokens")
    cumulative_steps = metadata.get("cumulative_optimizer_steps")
    curve = metadata.get("curve")
    if isinstance(curve, Sequence) and curve and isinstance(curve[-1], Mapping):
        final = curve[-1]
        if cumulative_tokens is None:
            cumulative_tokens = final.get(
                "cumulative_update_tokens",
                final.get("update_tokens", 0),
            )
        if cumulative_steps is None:
            cumulative_steps = final.get(
                "cumulative_optimizer_steps",
                final.get("optimizer_steps", 0),
            )
    return int(cumulative_tokens or 0), int(cumulative_steps or 0)


def _arm_token_budgets(
    arm: ScalingArmConfig,
    config: LanguageScalingExperimentConfig,
) -> tuple[int, ...]:
    multiplier = float(arm.token_budget_multiplier)
    if not math.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError("token_budget_multiplier must be finite and positive")
    return tuple(
        sorted(
            {
                max(1, int(round(int(value) * multiplier)))
                for value in config.token_budgets
            }
        )
    )


def _run_arm(
    arm: ScalingArmConfig,
    *,
    tokenizer,
    split,
    prompts: Sequence[str],
    output: Path,
    config: LanguageScalingExperimentConfig,
    device: torch.device,
    initial_model: MarulhoLanguageModel | None = None,
    resume_metadata: Mapping[str, Any] | None = None,
    resume_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    _seed_everything(config.seed)
    arm_config = _arm_training_config(arm, config)
    expected_model_config = _model_config(tokenizer, arm_config)
    if initial_model is None:
        model = MarulhoLanguageModel(expected_model_config).to(device)
    else:
        if asdict(initial_model.config) != asdict(expected_model_config):
            raise ValueError(
                "Resume checkpoint model config does not match the selected arm"
            )
        model = initial_model.to(device)
    parameter_inventory = _parameter_inventory(model)
    optimizer, fused = _optimizer(model, arm_config)
    checkpoint_metadata = dict(resume_metadata or {})
    prior_update_tokens, prior_optimizer_steps = _prior_training_progress(
        checkpoint_metadata
    )
    training_state = checkpoint_metadata.get("training_state")
    optimizer_state_restored = False
    if isinstance(training_state, Mapping) and isinstance(
        training_state.get("optimizer_state"), Mapping
    ):
        optimizer.load_state_dict(dict(training_state["optimizer_state"]))
        optimizer_state_restored = True
    budgets = _arm_token_budgets(arm, config)
    nominal_batch_tokens = max(
        1,
        int(split.train[0].target_ids.numel()),
    )
    selected_train_tokens_per_epoch = sum(
        int(batch.target_ids.numel()) for batch in split.train
    )
    total_steps = math.ceil(max(budgets) / nominal_batch_tokens)
    warmup_steps = int(round(total_steps * max(0.0, config.warmup_fraction)))
    use_scaler = device.type == "cuda" and config.precision.lower() == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if (
        optimizer_state_restored
        and isinstance(training_state, Mapping)
        and isinstance(training_state.get("scaler_state"), Mapping)
    ):
        scaler.load_state_dict(dict(training_state["scaler_state"]))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.seed))
    order = _next_batch_order(len(split.train), generator=generator)
    order_cursor = 0
    batch_order_state_restored = False
    if (
        optimizer_state_restored
        and isinstance(training_state, Mapping)
        and str(training_state.get("split_train_hash", ""))
        == str(split.report["train_split_hash"])
        and isinstance(training_state.get("generator_state"), torch.Tensor)
        and isinstance(training_state.get("batch_order"), Sequence)
    ):
        restored_order = [int(index) for index in training_state["batch_order"]]
        restored_cursor = int(training_state.get("batch_order_cursor", 0))
        if (
            len(restored_order) == len(split.train)
            and all(0 <= index < len(split.train) for index in restored_order)
            and 0 <= restored_cursor <= len(restored_order)
        ):
            generator.set_state(training_state["generator_state"].cpu())
            order = restored_order
            order_cursor = restored_cursor
            batch_order_state_restored = True
    update_tokens = 0
    step = 0
    curve: list[dict[str, Any]] = []
    loss_scalars: list[torch.Tensor] = []
    gradient_scalars: list[torch.Tensor] = []
    training_elapsed = 0.0
    wall_started = time.perf_counter()
    eval_before = evaluate_language_model(model, split.eval)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    segment_started = time.perf_counter()
    model.train()

    for budget in budgets:
        while update_tokens < budget:
            if order_cursor >= len(order):
                order = _next_batch_order(len(split.train), generator=generator)
                order_cursor = 0
            batch: LanguageBatch = split.train[order[order_cursor]]
            order_cursor += 1
            device_batch = batch.to(device)
            lr = _learning_rate(
                step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                peak=float(config.learning_rate),
                minimum_fraction=max(
                    0.0,
                    float(config.minimum_learning_rate_fraction),
                ),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with _precision_context(device, config.precision):
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
            loss_scalars.append(loss.detach().float())
            gradient_scalars.append(gradient_norm.detach().float())
            update_tokens += int(device_batch.target_ids.numel())
            step += 1

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_elapsed += time.perf_counter() - segment_started
        evaluation = evaluate_language_model(model, split.eval)
        losses = torch.stack(loss_scalars).cpu()
        gradients = torch.stack(gradient_scalars).cpu()
        curve.append(
            {
                "base_requested_update_tokens": int(
                    min(
                        config.token_budgets,
                        key=lambda value: abs(
                            int(round(int(value) * arm.token_budget_multiplier))
                            - int(budget)
                        ),
                    )
                ),
                "requested_update_tokens": int(budget),
                "token_budget_multiplier": float(arm.token_budget_multiplier),
                "budget_basis": str(config.budget_basis),
                "update_tokens": int(update_tokens),
                "prior_update_tokens": int(prior_update_tokens),
                "cumulative_update_tokens": int(
                    prior_update_tokens + update_tokens
                ),
                "unique_selected_update_tokens": min(
                    int(update_tokens),
                    int(selected_train_tokens_per_epoch),
                ),
                "repeated_selected_update_tokens": max(
                    0,
                    int(update_tokens) - int(selected_train_tokens_per_epoch),
                ),
                "selected_train_token_epochs": float(update_tokens)
                / max(1, selected_train_tokens_per_epoch),
                "optimizer_steps": int(step),
                "prior_optimizer_steps": int(prior_optimizer_steps),
                "cumulative_optimizer_steps": int(
                    prior_optimizer_steps + step
                ),
                "heldout_loss": float(evaluation["heldout_loss"]),
                "heldout_perplexity": float(evaluation["heldout_perplexity"]),
                "eval_token_count": int(evaluation["token_count"]),
                "mean_training_loss": float(losses.mean().item()),
                "recent_training_loss": float(
                    losses[-min(32, int(losses.numel())) :].mean().item()
                ),
                "max_gradient_norm": float(gradients.max().item()),
                "training_elapsed_seconds": float(training_elapsed),
                "training_tokens_per_second": (
                    float(update_tokens) / max(training_elapsed, 1.0e-9)
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        segment_started = time.perf_counter()

    final_evaluation = curve[-1]
    generations = [
        _decoded_generation(
            model,
            tokenizer,
            prompt=str(prompt),
            max_new_tokens=max(0, int(config.generation_tokens)),
            config=arm_config,
        )
        for prompt in prompts
    ]
    checkpoint = output.with_name(f"{output.stem}-{arm.name}-checkpoint.pt")
    save_language_model_checkpoint(
        checkpoint,
        model,
        tokenizer,
        metadata={
            "scaling_experiment_report": str(output),
            "arm": asdict(arm),
            "curve": curve,
            "split": split.report,
            "resume_checkpoint_path": (
                None
                if resume_checkpoint_path is None
                else str(resume_checkpoint_path)
            ),
            "cumulative_update_tokens": int(prior_update_tokens + update_tokens),
            "cumulative_optimizer_steps": int(prior_optimizer_steps + step),
            "training_state": {
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "generator_state": generator.get_state(),
                "batch_order": order,
                "batch_order_cursor": int(order_cursor),
                "split_train_hash": str(split.report["train_split_hash"]),
            },
        },
    )
    return {
        "name": arm.name,
        "status": "completed",
        "arm": asdict(arm),
        "model_config": asdict(model.config),
        "parameter_inventory": parameter_inventory,
        "optimizer": {
            "kind": "AdamW",
            "fused": bool(fused),
            "precision": config.precision,
            "warmup_steps": warmup_steps,
            "total_planned_steps": total_steps,
            "per_step_host_metric_readback": False,
            "batch_transfer_policy": "cpu_split_per_batch_to_model_device",
            "state_restored": bool(optimizer_state_restored),
            "batch_order_state_restored": bool(batch_order_state_restored),
            "schedule_policy": "new_phase_cosine",
        },
        "continuation": {
            "resume_checkpoint_path": (
                None
                if resume_checkpoint_path is None
                else str(resume_checkpoint_path)
            ),
            "prior_update_tokens": int(prior_update_tokens),
            "prior_optimizer_steps": int(prior_optimizer_steps),
            "optimizer_state_restored": bool(optimizer_state_restored),
            "batch_order_state_restored": bool(batch_order_state_restored),
            "cumulative_update_tokens": int(prior_update_tokens + update_tokens),
            "cumulative_optimizer_steps": int(prior_optimizer_steps + step),
        },
        "eval_before": eval_before,
        "curve": curve,
        "final_heldout_loss": float(final_evaluation["heldout_loss"]),
        "final_heldout_perplexity": float(final_evaluation["heldout_perplexity"]),
        "generations": generations,
        "checkpoint_path": str(checkpoint),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "wall_elapsed_seconds": time.perf_counter() - wall_started,
        "owned_by_marulho": True,
        "external_llm_used": False,
    }


def _failed_arm(arm: ScalingArmConfig, exc: BaseException) -> dict[str, Any]:
    message = f"{type(exc).__name__}: {exc}"
    return {
        "name": arm.name,
        "status": "failed",
        "arm": asdict(arm),
        "failure_reason": message,
        "cuda_out_of_memory": "out of memory" in message.lower(),
        "owned_by_marulho": True,
        "external_llm_used": False,
    }


def run_language_scaling_experiment(
    *,
    output_path: str | Path,
    corpus_paths: Sequence[str | Path],
    eval_corpus_paths: Sequence[str | Path] = (),
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    config: LanguageScalingExperimentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LanguageScalingExperimentConfig()
    budget_basis = str(cfg.budget_basis).strip().lower()
    if budget_basis not in {"equal_update_tokens", "empirical_wall_clock"}:
        raise ValueError(
            "budget_basis must be 'equal_update_tokens' or "
            "'empirical_wall_clock'"
        )
    multipliers = [float(arm.token_budget_multiplier) for arm in cfg.arms]
    if budget_basis == "equal_update_tokens" and any(
        not math.isclose(multiplier, 1.0) for multiplier in multipliers
    ):
        raise ValueError(
            "equal_update_tokens requires token_budget_multiplier=1 for every arm"
        )
    if budget_basis == "empirical_wall_clock" and len(set(multipliers)) == 1:
        raise ValueError(
            "empirical_wall_clock requires different per-arm token budget multipliers"
        )
    resume_checkpoint = (
        None
        if cfg.resume_checkpoint_path is None
        else Path(cfg.resume_checkpoint_path)
    )
    if resume_checkpoint is not None and len(cfg.arms) != 1:
        raise ValueError("Checkpoint continuation requires exactly one scaling arm")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = tuple(Path(path) for path in corpus_paths)
    if not source_paths:
        raise ValueError("At least one training corpus path is required")
    corpora = tuple(_read_corpus(path) for path in source_paths)
    eval_corpus_files = tuple(Path(path) for path in eval_corpus_paths)
    eval_corpora = tuple(_read_corpus(path) for path in eval_corpus_files)
    device = _resolve_device(cfg.device)
    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.cuda_allow_tf32)
        torch.set_float32_matmul_precision("high")
    tokenizer_config = LanguageTrainingExperimentConfig(
        tokenizer_kind="bpe",
        tokenizer_vocab_size=int(cfg.tokenizer_vocab_size),
        tokenizer_min_frequency=int(cfg.tokenizer_min_frequency),
    )
    resumed_model: MarulhoLanguageModel | None = None
    resume_metadata: dict[str, Any] = {}
    if resume_checkpoint is None:
        tokenizer = _build_tokenizer(corpora, tokenizer_config)
    else:
        resumed_model, tokenizer, resume_metadata = load_language_model_checkpoint(
            resume_checkpoint,
            map_location="cpu",
        )
    split = build_language_model_splits(
        corpora,
        tokenizer,
        eval_texts=None if not eval_corpora else list(eval_corpora),
        sequence_length=int(cfg.sequence_length),
        eval_fraction=float(cfg.eval_fraction),
        stride=int(cfg.stride),
        batch_size=int(cfg.batch_size),
        device="cpu",
        max_train_batches=int(cfg.max_train_batches),
        max_eval_batches=int(cfg.max_eval_batches),
        window_selection="stratified",
    )
    corpus_token_count = int(split.report["train_text_token_count"])
    prompt_rows = [
        {
            "prompt": str(prompt),
            "exact_prompt_absent_from_corpus": all(
                str(prompt).lower() not in corpus.lower() for corpus in corpora
            ),
        }
        for prompt in prompts
    ]
    arms: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for arm_index, arm in enumerate(cfg.arms):
            try:
                arms.append(
                    _run_arm(
                        arm,
                        tokenizer=tokenizer,
                        split=split,
                        prompts=prompts,
                        output=output,
                        config=cfg,
                        device=device,
                        initial_model=(
                            resumed_model
                            if resume_checkpoint is not None and arm_index == 0
                            else None
                        ),
                        resume_metadata=resume_metadata,
                        resume_checkpoint_path=resume_checkpoint,
                    )
                )
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                arms.append(_failed_arm(arm, exc))
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    completed = [arm for arm in arms if arm["status"] == "completed"]
    if not completed:
        raise RuntimeError("No scaling arm completed")
    best = min(
        completed,
        key=lambda arm: (
            float(arm["final_heldout_loss"]),
            int(arm["parameter_inventory"]["total_parameters"]),
        ),
    )
    deleted_checkpoints: list[str] = []
    if cfg.retain_best_checkpoint_only:
        best_path = Path(str(best["checkpoint_path"])).resolve()
        for arm in completed:
            checkpoint = Path(str(arm["checkpoint_path"]))
            if checkpoint.resolve() != best_path and checkpoint.is_file():
                checkpoint.unlink()
                deleted_checkpoints.append(str(checkpoint))
                arm["checkpoint_retained"] = False
            else:
                arm["checkpoint_retained"] = True
    else:
        for arm in completed:
            arm["checkpoint_retained"] = True

    fit_points = [
        {
            "arm": str(arm["name"]),
            "total_parameters": float(
                arm["parameter_inventory"]["total_parameters"]
            ),
            "non_embedding_parameters": float(
                arm["parameter_inventory"]["transformer_parameters"]
            ),
            "update_tokens": float(point["update_tokens"]),
            "heldout_loss": float(point["heldout_loss"]),
        }
        for arm in completed
        for point in arm["curve"]
    ]
    scaling_law = (
        fit_language_scaling_law(fit_points)
        if budget_basis == "equal_update_tokens"
        else {
            "available": False,
            "reason": "requires_equal_update_token_grid",
            "point_count": len(fit_points),
            "model_size_count": len(
                {point["non_embedding_parameters"] for point in fit_points}
            ),
            "token_budget_count": len(
                {point["update_tokens"] for point in fit_points}
            ),
        }
    )
    completed_by_size = sorted(
        completed,
        key=lambda arm: int(arm["parameter_inventory"]["total_parameters"]),
    )
    final_losses = [float(arm["final_heldout_loss"]) for arm in completed_by_size]
    size_quality_monotonic = all(
        right < left for left, right in zip(final_losses, final_losses[1:])
    )
    largest_completed = completed_by_size[-1]
    best_is_largest = best["name"] == largest_completed["name"]
    smallest_completed = completed_by_size[0]
    final_size_loss_gain = (
        float(smallest_completed["final_heldout_loss"])
        - float(largest_completed["final_heldout_loss"])
    )
    largest_curve = list(largest_completed["curve"])
    largest_data_loss_gain = (
        float(largest_curve[0]["heldout_loss"])
        - float(largest_curve[-1]["heldout_loss"])
        if len(largest_curve) >= 2
        else 0.0
    )
    data_to_size_gain_ratio = (
        largest_data_loss_gain / max(final_size_loss_gain, 1.0e-9)
        if budget_basis == "equal_update_tokens" and final_size_loss_gain > 0.0
        else None
    )
    training_elapsed_by_arm = {
        str(arm["name"]): float(arm["curve"][-1]["training_elapsed_seconds"])
        for arm in completed_by_size
    }
    positive_training_elapsed = [
        elapsed for elapsed in training_elapsed_by_arm.values() if elapsed > 0.0
    ]
    training_elapsed_ratio = (
        max(positive_training_elapsed) / min(positive_training_elapsed)
        if len(positive_training_elapsed) >= 2
        else 1.0
    )
    wall_clock_comparison_accepted = (
        budget_basis != "empirical_wall_clock" or training_elapsed_ratio <= 1.15
    )
    if len(completed_by_size) == 1:
        branch_decision = "continue_data_scaling_at_selected_model_size"
    elif budget_basis == "empirical_wall_clock":
        if not wall_clock_comparison_accepted:
            branch_decision = "recalibrate_empirical_wall_clock_budgets"
        elif best["name"] == smallest_completed["name"]:
            branch_decision = "scale_data_at_compute_optimal_smaller_model"
        elif best_is_largest:
            branch_decision = "scale_data_at_compute_optimal_larger_model"
        else:
            branch_decision = "scale_data_at_compute_optimal_selected_model"
    elif best_is_largest and size_quality_monotonic:
        branch_decision = (
            "scale_data_at_selected_model_size"
            if data_to_size_gain_ratio is not None
            and data_to_size_gain_ratio >= 2.0
            else "scale_transformer_data_and_model"
        )
    else:
        branch_decision = "redesign_scaling_recipe_before_larger_run"
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "continuation": {
            "enabled": resume_checkpoint is not None,
            "checkpoint_path": (
                None if resume_checkpoint is None else str(resume_checkpoint)
            ),
            "checkpoint_sha256": (
                None
                if resume_checkpoint is None
                else _sha256_file(resume_checkpoint)
            ),
            "checkpoint_owned_by_marulho": resume_checkpoint is not None,
            "optimizer_state_available": isinstance(
                resume_metadata.get("training_state"), Mapping
            ),
        },
        "device": str(device),
        "config": {
            **asdict(cfg),
            "arms": [asdict(arm) for arm in cfg.arms],
        },
        "corpus": {
            "sources": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "utf8_bytes": len(corpus.encode("utf-8")),
                    "row_provenance_report_expected": str(
                        path.with_suffix(".json")
                    ),
                }
                for path, corpus in zip(source_paths, corpora)
            ],
            "source_count": len(source_paths),
            "utf8_bytes": sum(len(corpus.encode("utf-8")) for corpus in corpora),
            "bpe_tokens": corpus_token_count,
            "explicit_eval_sources": [
                {
                    "path": str(path),
                    "sha256": _sha256_file(path),
                    "utf8_bytes": len(corpus.encode("utf-8")),
                    "row_provenance_report_expected": str(
                        path.with_suffix(".json")
                    ),
                }
                for path, corpus in zip(eval_corpus_files, eval_corpora)
            ],
            "explicit_eval_source_count": len(eval_corpus_files),
            "explicit_eval_utf8_bytes": sum(
                len(corpus.encode("utf-8")) for corpus in eval_corpora
            ),
        },
        "tokenizer": {
            "surface": tokenizer.state_dict().get("surface"),
            "vocab_size": int(tokenizer.vocab_size),
            "vocabulary_hash": tokenizer.vocabulary_hash(),
            "vocabulary_trained_by_marulho": True,
            "source": (
                "checkpoint_owned"
                if resume_checkpoint is not None
                else "trained_from_current_corpus"
            ),
            "training_chunk_characters": BPE_TRAINING_CHUNK_CHARACTERS,
        },
        "split": split.report,
        "prompts": prompt_rows,
        "arms": arms,
        "completed_arm_count": len(completed),
        "failed_arm_count": len(arms) - len(completed),
        "selection": {
            "metric": (
                "final_heldout_loss"
                if budget_basis == "equal_update_tokens"
                else "final_heldout_loss_under_empirical_wall_clock_budget"
            ),
            "selected_arm": best["name"],
            "selected_checkpoint": best["checkpoint_path"],
            "deleted_nonselected_checkpoints": deleted_checkpoints,
            "best_is_largest_completed_arm": best_is_largest,
            "size_quality_monotonic": size_quality_monotonic,
        },
        "scaling_law": scaling_law,
        "effect_sizes": {
            "final_loss_difference_smallest_minus_largest": final_size_loss_gain,
            "largest_arm_first_to_final_data_loss_gain": largest_data_loss_gain,
            "data_to_size_gain_ratio": data_to_size_gain_ratio,
        },
        "budget_normalization": {
            "basis": budget_basis,
            "base_token_budgets": [int(value) for value in cfg.token_budgets],
            "arm_token_budget_multipliers": {
                str(arm.name): float(arm.token_budget_multiplier)
                for arm in cfg.arms
            },
            "training_elapsed_seconds_by_arm": training_elapsed_by_arm,
            "full_corpus_token_epochs_by_arm": {
                str(arm["name"]): float(arm["curve"][-1]["update_tokens"])
                / max(1, corpus_token_count)
                for arm in completed_by_size
            },
            "max_to_min_training_elapsed_ratio": training_elapsed_ratio,
            "comparison_tolerance_ratio": 1.15,
            "comparison_accepted": wall_clock_comparison_accepted,
            "method": (
                "shared_update_token_budgets"
                if budget_basis == "equal_update_tokens"
                else "multipliers_from_prior_observed_training_throughput"
            ),
        },
        "branch_decision": branch_decision,
        "quality_boundary": {
            "coherent_unseen_multisentence_generation_manually_verified": False,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Transformer Scaling Experiment",
    )
    return report


def _parse_arm(value: str) -> ScalingArmConfig:
    parts = str(value).split(":")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError(
            "arm must be name:width:layers:heads[:token_budget_multiplier]"
        )
    name, width, layers, heads = parts[:4]
    multiplier = 1.0 if len(parts) == 4 else float(parts[4])
    return ScalingArmConfig(
        name=name,
        width=int(width),
        layers=int(layers),
        heads=int(heads),
        token_budget_multiplier=multiplier,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", action="append", type=Path, required=True)
    parser.add_argument("--eval-corpus", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_parse_arm, default=[])
    parser.add_argument("--token-budget", action="append", type=int, default=[])
    parser.add_argument(
        "--budget-basis",
        choices=("equal_update_tokens", "empirical_wall_clock"),
        default="equal_update_tokens",
    )
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-train-batches", type=int, default=4096)
    parser.add_argument("--max-eval-batches", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
    )
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--retain-all-checkpoints", action="store_true")
    args = parser.parse_args()
    config = LanguageScalingExperimentConfig(
        tokenizer_vocab_size=max(512, int(args.tokenizer_vocab_size)),
        sequence_length=max(2, int(args.sequence_length)),
        stride=max(1, int(args.stride)),
        batch_size=max(1, int(args.batch_size)),
        max_train_batches=max(1, int(args.max_train_batches)),
        max_eval_batches=max(1, int(args.max_eval_batches)),
        token_budgets=tuple(args.token_budget)
        or LanguageScalingExperimentConfig.token_budgets,
        budget_basis=str(args.budget_basis),
        arms=tuple(args.arm) or DEFAULT_ARMS,
        transformer_context_length=max(2, int(args.context_length)),
        learning_rate=float(args.learning_rate),
        precision=str(args.precision),
        generation_tokens=max(0, int(args.generation_tokens)),
        seed=int(args.seed),
        retain_best_checkpoint_only=not bool(args.retain_all_checkpoints),
        device=str(args.device),
        resume_checkpoint_path=(
            None
            if args.resume_checkpoint is None
            else str(args.resume_checkpoint)
        ),
    )
    report = run_language_scaling_experiment(
        output_path=args.output,
        corpus_paths=tuple(args.corpus),
        eval_corpus_paths=tuple(args.eval_corpus),
        prompts=tuple(args.prompt) or DEFAULT_PROMPTS,
        config=config,
    )
    return 0 if int(report["completed_arm_count"]) >= 1 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
