"""Fast mutable training experiment runner for the MARULHO LM head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Sequence

import torch

from marulho.core.language_plif_triton import (
    language_plif_triton_stats,
    language_plif_triton_stats_delta,
)
from marulho.data.language_tokenizer import (
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
    precompute_sampled_vocab_batches,
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
    tokenizer_kind: str = "byte"
    tokenizer_vocab_size: int = 4096
    tokenizer_min_frequency: int = 2
    model_vocab_size: int = 0
    sampled_vocab_size: int = 0
    sparse_vocab_optimizer: bool = True
    embedding_dim: int = 32
    state_dim: int = 64
    state_core: str = "selective_spiking"
    state_layers: int = 1
    attention_heads: int = 4
    transformer_context_length: int = 256
    transformer_mlp_ratio: float = 4.0
    transformer_dropout: float = 0.0
    tie_embeddings: bool = False
    expert_count: int = 8
    active_expert_count: int = 2
    route_candidate_count: int = 4
    expert_hidden_dim: int = 96
    adaptive_timestep_budget: int = 1
    recurrent_gradient_horizon: int = 0
    memory_slot_count: int = 0
    memory_slot_candidate_count: int = 0
    active_memory_slot_count: int = 1
    memory_slot_init_std: float = 0.02
    sequence_length: int = 32
    stride: int = 16
    batch_size: int = 8
    max_train_batches: int = 64
    max_eval_batches: int = 64
    window_selection: str = "stratified"
    train_epochs: int = 2
    learning_rate: float = 2e-3
    max_grad_norm: float = 1.0
    gradient_clip_interval: int = 1
    generation_tokens: int = 48
    sustained_target_tokens: int = 512
    sustained_tick_tokens: int = 128
    sustained_quantum_tokens: int = 16
    sustained_timeout_seconds: float = 120.0
    profile_training_stages: bool = False
    cuda_allow_tf32: bool = True
    cuda_float32_matmul_precision: str = "high"
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
    tokenizer: LanguageTokenizer,
    config: LanguageTrainingExperimentConfig,
) -> LanguageModelConfig:
    model_vocab_size = (
        int(config.model_vocab_size)
        if int(config.model_vocab_size) > 0
        else int(tokenizer.vocab_size)
    )
    if model_vocab_size < int(tokenizer.vocab_size):
        raise ValueError("model_vocab_size must be at least the tokenizer vocab size")
    sampled_vocab_size = max(0, int(config.sampled_vocab_size))
    if sampled_vocab_size >= model_vocab_size:
        raise ValueError("sampled_vocab_size must be smaller than model_vocab_size")
    sparse_vocab_gradients = bool(config.sparse_vocab_optimizer and sampled_vocab_size > 0)
    if bool(config.tie_embeddings) and sparse_vocab_gradients:
        raise ValueError("tie_embeddings is incompatible with sparse sampled-vocab training")
    return LanguageModelConfig(
        vocab_size=model_vocab_size,
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        state_core=str(config.state_core),
        state_layers=max(1, int(config.state_layers)),
        attention_heads=max(1, int(config.attention_heads)),
        transformer_context_length=max(2, int(config.transformer_context_length)),
        transformer_mlp_ratio=float(config.transformer_mlp_ratio),
        transformer_dropout=float(config.transformer_dropout),
        tie_embeddings=bool(config.tie_embeddings),
        adaptive_timestep_budget=int(config.adaptive_timestep_budget),
        recurrent_gradient_horizon=max(0, int(config.recurrent_gradient_horizon)),
        expert_count=int(config.expert_count),
        active_expert_count=int(config.active_expert_count),
        route_candidate_count=int(config.route_candidate_count),
        expert_hidden_dim=int(config.expert_hidden_dim),
        sampled_vocab_size=sampled_vocab_size,
        sampled_vocab_sparse_lm_head_gradient=sparse_vocab_gradients,
        sparse_token_embedding_gradients=sparse_vocab_gradients,
        generation_vocab_size=(
            int(tokenizer.vocab_size)
            if model_vocab_size > int(tokenizer.vocab_size)
            else 0
        ),
        memory_slot_count=max(0, int(config.memory_slot_count)),
        memory_slot_candidate_count=max(0, int(config.memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(config.active_memory_slot_count)),
        memory_slot_init_std=float(config.memory_slot_init_std),
    )


def _build_tokenizer(
    corpus: str,
    config: LanguageTrainingExperimentConfig,
) -> LanguageTokenizer:
    kind = str(config.tokenizer_kind).strip().lower()
    if kind == "byte":
        return ByteLevelLanguageTokenizer()
    if kind == "bpe":
        return BytePairLanguageTokenizer.train(
            [corpus],
            vocab_size=max(512, int(config.tokenizer_vocab_size)),
            min_frequency=max(1, int(config.tokenizer_min_frequency)),
        )
    raise ValueError("tokenizer_kind must be 'byte' or 'bpe'")


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


def _synchronize_if_cuda(device: torch.device | str) -> bool:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return False
    torch.cuda.synchronize(resolved)
    return True


def _cuda_math_policy_snapshot() -> dict[str, Any]:
    precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else "unavailable"
    )
    return {
        "surface": "marulho_cuda_math_policy.v1",
        "cuda_available": bool(torch.cuda.is_available()),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": str(precision),
    }


def _apply_cuda_math_policy(
    device: torch.device,
    config: LanguageTrainingExperimentConfig,
) -> dict[str, Any]:
    requested_precision = str(config.cuda_float32_matmul_precision)
    if requested_precision not in {"highest", "high", "medium"}:
        raise ValueError(
            "cuda_float32_matmul_precision must be one of: highest, high, medium"
        )
    before = _cuda_math_policy_snapshot()
    applied = bool(device.type == "cuda" and torch.cuda.is_available())
    if applied:
        torch.backends.cuda.matmul.allow_tf32 = bool(config.cuda_allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(config.cuda_allow_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(requested_precision)
    active = _cuda_math_policy_snapshot()
    return {
        "surface": "marulho_cuda_math_policy_application.v1",
        "device": str(device),
        "applied": applied,
        "requested_matmul_allow_tf32": bool(config.cuda_allow_tf32),
        "requested_cudnn_allow_tf32": bool(config.cuda_allow_tf32),
        "requested_float32_matmul_precision": requested_precision,
        "before": before,
        "active": active,
    }


def _restore_cuda_math_policy(snapshot: dict[str, Any]) -> None:
    torch.backends.cuda.matmul.allow_tf32 = bool(
        snapshot.get("matmul_allow_tf32", False)
    )
    torch.backends.cudnn.allow_tf32 = bool(snapshot.get("cudnn_allow_tf32", True))
    precision = snapshot.get("float32_matmul_precision")
    if isinstance(precision, str) and hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(precision)
        except ValueError:
            pass


def _scalar_tensor_to_float(value: torch.Tensor | None) -> float | None:
    if value is None:
        return None
    return float(value.detach().cpu().item())


def _detached_scalar_on_device(
    value: torch.Tensor | float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        return tensor.to(device=device) if tensor.device != device else tensor
    return torch.tensor(float(value), device=device)


class _TrainingStageProfiler:
    def __init__(self, device: torch.device, *, enabled: bool) -> None:
        self.device = device
        self.enabled = bool(enabled)
        self.cuda_events = bool(self.enabled and device.type == "cuda")
        self._records: list[dict[str, Any]] = []

    def start(self) -> torch.cuda.Event | float | None:
        if not self.enabled:
            return None
        if self.cuda_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    def record_elapsed(
        self,
        name: str,
        token_count: int,
        marker: torch.cuda.Event | float | None,
    ) -> None:
        if not self.enabled or marker is None:
            return
        if self.cuda_events:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            self._records.append(
                {
                    "stage": str(name),
                    "token_count": int(token_count),
                    "start_event": marker,
                    "end_event": end_event,
                }
            )
            return
        elapsed_ms = (time.perf_counter() - float(marker)) * 1000.0
        self._records.append(
            {
                "stage": str(name),
                "token_count": int(token_count),
                "elapsed_ms": float(elapsed_ms),
            }
        )

    def report(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "surface": "marulho_language_training_stage_profile.v1",
                "enabled": False,
            }
        if self.cuda_events:
            torch.cuda.synchronize(self.device)
        per_stage: dict[str, dict[str, float]] = {}
        total_ms = 0.0
        for record in self._records:
            stage = str(record["stage"])
            if self.cuda_events:
                elapsed_ms = float(
                    record["start_event"].elapsed_time(record["end_event"])
                )
            else:
                elapsed_ms = float(record["elapsed_ms"])
            token_count = int(record.get("token_count", 0) or 0)
            summary = per_stage.setdefault(
                stage,
                {
                    "count": 0.0,
                    "total_ms": 0.0,
                    "token_count": 0.0,
                },
            )
            summary["count"] += 1.0
            summary["total_ms"] += elapsed_ms
            summary["token_count"] += float(token_count)
            total_ms += elapsed_ms
        formatted: dict[str, dict[str, float]] = {}
        for stage, summary in sorted(per_stage.items()):
            count = max(1.0, float(summary["count"]))
            token_count = max(1.0, float(summary["token_count"]))
            total_stage_ms = float(summary["total_ms"])
            formatted[stage] = {
                "count": int(summary["count"]),
                "total_ms": total_stage_ms,
                "mean_ms": total_stage_ms / count,
                "mean_ms_per_token": total_stage_ms / token_count,
            }
        top_stage_mean_ms_per_token = sorted(
            (
                {
                    "stage": stage,
                    "mean_ms_per_token": values["mean_ms_per_token"],
                    "total_ms": values["total_ms"],
                }
                for stage, values in formatted.items()
                if stage != "batch_total"
            ),
            key=lambda item: float(item["mean_ms_per_token"]),
            reverse=True,
        )
        return {
            "surface": "marulho_language_training_stage_profile.v1",
            "enabled": True,
            "device": str(self.device),
            "measurement": (
                "cuda_event_no_per_stage_sync"
                if self.cuda_events
                else "host_perf_counter"
            ),
            "record_count": len(self._records),
            "total_recorded_ms": total_ms,
            "per_stage": formatted,
            "top_stage_mean_ms_per_token": top_stage_mean_ms_per_token,
            "profile_scope": "training_update_window_only",
        }


def _all_trainable_parameters(model: MarulhoLanguageModel) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _parameter_inventory(model: MarulhoLanguageModel) -> dict[str, Any]:
    def count(module: torch.nn.Module) -> int:
        return sum(int(parameter.numel()) for parameter in module.parameters())

    total = count(model)
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    state_core = count(model.state_block)
    routed_experts = count(model.routed_experts)
    token_embedding = count(model.token_embedding)
    lm_head = count(model.lm_head)
    memory = sum(
        int(parameter.numel())
        for name, parameter in model.named_parameters()
        if name.startswith("memory_slot")
    )
    return {
        "surface": "marulho_language_model_parameter_inventory.v1",
        "state_core": str(model.config.state_core),
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "state_core_parameters": int(state_core),
        "routed_expert_parameters": int(routed_experts),
        "token_embedding_parameters": int(token_embedding),
        "lm_head_parameters": int(lm_head),
        "memory_parameters": int(memory),
    }


def _optimizer_policy(
    model: MarulhoLanguageModel,
    *,
    config: LanguageTrainingExperimentConfig,
) -> tuple[list[torch.optim.Optimizer], str]:
    sparse_vocab_optimizer = bool(
        config.sparse_vocab_optimizer
        and int(model.config.sampled_vocab_size) > 0
        and bool(model.config.sampled_vocab_sparse_lm_head_gradient)
        and bool(model.config.sparse_token_embedding_gradients)
    )
    if not sparse_vocab_optimizer:
        return [
            torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate))
        ], "AdamW_all_parameters"

    sparse_names = {"token_embedding.weight", "lm_head.weight"}
    sparse_params: list[torch.nn.Parameter] = []
    dense_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name in sparse_names:
            sparse_params.append(parameter)
        else:
            dense_params.append(parameter)
    optimizers: list[torch.optim.Optimizer] = []
    if dense_params:
        optimizers.append(torch.optim.AdamW(dense_params, lr=float(config.learning_rate)))
    if sparse_params:
        optimizers.append(torch.optim.SparseAdam(sparse_params, lr=float(config.learning_rate)))
    return optimizers, "AdamW_dense_core_plus_SparseAdam_vocab_rows"


def _clip_grad_norm_sparse_aware(
    parameters: Sequence[torch.nn.Parameter],
    *,
    max_norm: float,
    device: torch.device,
) -> torch.Tensor:
    total_sq = torch.zeros((), device=device, dtype=torch.float32)
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        values = grad.coalesce().values() if grad.is_sparse else grad
        total_sq = total_sq + values.detach().float().pow(2).sum()
    total_norm = torch.sqrt(total_sq)
    limit = float(max_norm)
    if limit > 0.0:
        clip_coef = torch.clamp(
            torch.tensor(limit, device=device, dtype=torch.float32)
            / (total_norm + 1e-6),
            max=1.0,
        )
        for parameter in parameters:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                grad = grad.coalesce()
                grad.values().mul_(clip_coef)
                parameter.grad = grad
            else:
                grad.mul_(clip_coef)
    return total_norm


def _common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    for index in range(limit):
        if left[index] != right[index]:
            return index
    return limit


def _printable_fraction(text: str) -> float:
    if not text:
        return 0.0
    printable = sum(1 for char in text if char.isprintable() or char in "\n\t")
    return float(printable) / float(len(text))


def _distinct_bigram_fraction(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 2:
        return 1.0 if token_ids else 0.0
    bigrams = list(zip(token_ids, token_ids[1:]))
    return float(len(set(bigrams))) / float(len(bigrams))


def _max_token_run_length(token_ids: Sequence[int]) -> int:
    if not token_ids:
        return 0
    longest = 1
    current = 1
    previous = int(token_ids[0])
    for token_id in token_ids[1:]:
        token_id = int(token_id)
        if token_id == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
            previous = token_id
    return max(longest, current)


def _source_continuation_review(
    *,
    prompt: str,
    continuation_text: str,
    corpus: str,
    max_chars: int,
) -> dict[str, Any]:
    source_index = corpus.find(prompt)
    if source_index < 0:
        return {
            "source_prompt_found": False,
            "expected_source_continuation": "",
            "prefix_match_chars": 0,
            "prefix_match_fraction": 0.0,
            "next_character_matches_source": False,
        }
    expected = corpus[
        source_index + len(prompt) : source_index + len(prompt) + max(0, int(max_chars))
    ]
    prefix_match_chars = _common_prefix_length(expected, continuation_text)
    return {
        "source_prompt_found": True,
        "expected_source_continuation": expected,
        "prefix_match_chars": int(prefix_match_chars),
        "prefix_match_fraction": (
            float(prefix_match_chars) / float(len(expected)) if expected else 0.0
        ),
        "next_character_matches_source": bool(
            expected and continuation_text and expected[0] == continuation_text[0]
        ),
    }


def _decoded_generation(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    prompt: str,
    max_new_tokens: int,
    corpus: str | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> dict[str, Any]:
    prompt_ids = torch.tensor(
        tokenizer.encode(prompt, add_eos=False),
        dtype=torch.long,
    )
    generation = model.generate(
        prompt_ids,
        max_new_tokens=max(0, int(max_new_tokens)),
        eos_id=tokenizer.eos_id,
        repetition_penalty=max(1.0, float(repetition_penalty)),
        no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
    )
    generated_ids = [
        int(token_id)
        for token_id in generation["generated_ids"].detach().cpu().reshape(-1).tolist()
    ]
    prompt_count = int(prompt_ids.numel())
    continuation_ids = generated_ids[prompt_count:]
    continuation_text = tokenizer.decode(continuation_ids)
    source_review = _source_continuation_review(
        prompt=prompt,
        continuation_text=continuation_text,
        corpus=corpus or "",
        max_chars=len(continuation_text),
    )
    return {
        "surface": generation["surface"],
        "prompt": prompt,
        "generated_text": tokenizer.decode(generated_ids),
        "continuation_text": continuation_text,
        "prompt_token_count": prompt_count,
        "generated_token_count": len(generated_ids),
        "continuation_token_count": len(continuation_ids),
        "new_token_count": int(generation["new_token_count"]),
        "sequence_hash": tokenizer.sequence_hash(generated_ids),
        "continuation_sequence_hash": tokenizer.sequence_hash(continuation_ids),
        "quality_probe": {
            "printable_fraction": _printable_fraction(continuation_text),
            "distinct_bigram_fraction": _distinct_bigram_fraction(continuation_ids),
            "max_token_run_length": _max_token_run_length(continuation_ids),
            "source_continuation": source_review,
        },
        "active_language_path": generation["active_language_path"],
        "external_llm_used": bool(generation["external_llm_used"]),
        "owned_by_marulho": bool(generation["owned_by_marulho"]),
        "generation_decode": dict(generation.get("generation_decode") or {}),
    }


def _generation_quality_summary(
    generations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not generations:
        return {
            "surface": "marulho_language_generation_quality_summary.v1",
            "generation_count": 0,
            "mean_printable_fraction": 0.0,
            "mean_distinct_bigram_fraction": 0.0,
            "mean_source_prefix_match_chars": 0.0,
            "next_character_match_rate": 0.0,
        }
    printable: list[float] = []
    distinct_bigrams: list[float] = []
    prefix_matches: list[float] = []
    next_matches = 0
    source_found = 0
    for generation in generations:
        probe = generation.get("quality_probe") if isinstance(generation, dict) else {}
        if not isinstance(probe, dict):
            continue
        printable.append(float(probe.get("printable_fraction", 0.0) or 0.0))
        distinct_bigrams.append(
            float(probe.get("distinct_bigram_fraction", 0.0) or 0.0)
        )
        source = probe.get("source_continuation")
        if isinstance(source, dict) and bool(source.get("source_prompt_found")):
            source_found += 1
            prefix_matches.append(float(source.get("prefix_match_chars", 0.0) or 0.0))
            if bool(source.get("next_character_matches_source")):
                next_matches += 1
    return {
        "surface": "marulho_language_generation_quality_summary.v1",
        "generation_count": len(generations),
        "source_prompt_found_count": int(source_found),
        "mean_printable_fraction": _mean(printable),
        "mean_distinct_bigram_fraction": _mean(distinct_bigrams),
        "mean_source_prefix_match_chars": _mean(prefix_matches),
        "next_character_match_rate": (
            float(next_matches) / float(source_found) if source_found else 0.0
        ),
        "review_kind": "source_continuation_probe_not_human_quality_review",
        "promotes_generation_quality_claim": False,
    }


def _train_language_model(
    model: MarulhoLanguageModel,
    batches: Sequence[LanguageBatch],
    *,
    config: LanguageTrainingExperimentConfig,
    cuda_math_policy: dict[str, Any],
) -> dict[str, Any]:
    if not batches:
        raise ValueError("At least one train batch is required")
    optimizers, optimizer_policy = _optimizer_policy(model, config=config)
    trainable_parameters = _all_trainable_parameters(model)
    assume_no_sleeping = (
        model.routed_experts.enabled
        and not bool(model.routed_experts.sleeping_expert_mask.detach().any().cpu().item())
    )
    batches, sampled_vocab_precompute = precompute_sampled_vocab_batches(
        model,
        batches,
        assume_no_sleeping_experts=assume_no_sleeping,
    )
    model.train()
    token_count = 0
    optimizer_step_count = 0
    gradient_clip_applied_step_count = 0
    gradient_clip_skipped_step_count = 0
    gradient_clip_interval = max(0, int(config.gradient_clip_interval))
    loss_records: list[torch.Tensor] = []
    max_grad_norm_record: torch.Tensor | None = None
    last_loss_kind = "unknown"
    last_loss_evidence: dict[str, Any] = {}
    last_state_block_telemetry: dict[str, Any] = {}
    last_batch_for_probe: LanguageBatch | None = None
    stage_profiler = _TrainingStageProfiler(
        model.device,
        enabled=bool(config.profile_training_stages),
    )
    plif_stats_before = language_plif_triton_stats()
    cuda_synchronized_before_timing_start = _synchronize_if_cuda(model.device)
    started = time.perf_counter()
    for _epoch in range(max(1, int(config.train_epochs))):
        for batch in batches:
            optimizer_step_count += 1
            last_batch_for_probe = batch
            batch_token_count = int(batch.target_ids.numel())
            batch_marker = stage_profiler.start()
            stage_marker = stage_profiler.start()
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            stage_profiler.record_elapsed(
                "zero_grad",
                batch_token_count,
                stage_marker,
            )
            stage_marker = stage_profiler.start()
            result = model.next_token_loss(
                batch.input_ids.to(model.device),
                batch.target_ids.to(model.device),
                collect_telemetry=False,
                assume_no_sleeping_experts=assume_no_sleeping,
                sampled_vocab_ids=batch.sampled_vocab_ids,
                sampled_target_positions=batch.sampled_target_positions,
                memory_candidate_ids=batch.memory_candidate_ids,
                route_candidate_ids=batch.route_candidate_ids,
                return_evidence=False,
            )
            stage_profiler.record_elapsed(
                "forward_loss",
                batch_token_count,
                stage_marker,
            )
            loss = result["loss"]
            stage_marker = stage_profiler.start()
            loss.backward()
            stage_profiler.record_elapsed(
                "backward",
                batch_token_count,
                stage_marker,
            )
            stage_marker = stage_profiler.start()
            should_clip_gradients = bool(
                float(config.max_grad_norm) > 0.0
                and gradient_clip_interval > 0
                and optimizer_step_count % gradient_clip_interval == 0
            )
            if should_clip_gradients:
                grad_norm = _clip_grad_norm_sparse_aware(
                    trainable_parameters,
                    max_norm=float(config.max_grad_norm),
                    device=model.device,
                )
                gradient_clip_applied_step_count += 1
            else:
                grad_norm = torch.zeros((), device=model.device, dtype=torch.float32)
                gradient_clip_skipped_step_count += 1
            stage_profiler.record_elapsed(
                "gradient_clip",
                batch_token_count,
                stage_marker,
            )
            stage_marker = stage_profiler.start()
            for optimizer in optimizers:
                optimizer.step()
            stage_profiler.record_elapsed(
                "optimizer_step",
                batch_token_count,
                stage_marker,
            )
            stage_profiler.record_elapsed(
                "batch_total",
                batch_token_count,
                batch_marker,
            )
            token_count += batch_token_count
            loss_records.append(loss.detach())
            last_loss_kind = str(result.get("loss_kind", "unknown"))
            if should_clip_gradients:
                grad_norm_record = _detached_scalar_on_device(
                    grad_norm,
                    device=model.device,
                )
                max_grad_norm_record = (
                    grad_norm_record
                    if max_grad_norm_record is None
                    else torch.maximum(max_grad_norm_record, grad_norm_record)
                )
    cuda_synchronized_before_timing_stop = _synchronize_if_cuda(model.device)
    elapsed = max(0.0, time.perf_counter() - started)
    telemetry_probe_outside_measured_window = False
    post_window_telemetry_probe_batch_tokens = 0
    if last_batch_for_probe is not None:
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        probe_result = model.next_token_loss(
            last_batch_for_probe.input_ids.to(model.device),
            last_batch_for_probe.target_ids.to(model.device),
            collect_telemetry=False,
            assume_no_sleeping_experts=assume_no_sleeping,
            sampled_vocab_ids=last_batch_for_probe.sampled_vocab_ids,
            sampled_target_positions=last_batch_for_probe.sampled_target_positions,
            memory_candidate_ids=last_batch_for_probe.memory_candidate_ids,
            route_candidate_ids=last_batch_for_probe.route_candidate_ids,
            return_evidence=True,
        )
        telemetry_probe_outside_measured_window = True
        post_window_telemetry_probe_batch_tokens = int(
            last_batch_for_probe.target_ids.numel()
        )
        last_loss_kind = str(probe_result.get("loss_kind", last_loss_kind))
        last_loss_evidence = dict(probe_result.get("loss_evidence") or {})
        telemetry = probe_result.get("telemetry")
        last_state_block_telemetry = (
            dict(telemetry) if isinstance(telemetry, dict) else {}
        )
        del probe_result
    loss_values = (
        torch.stack([loss.to(model.device) for loss in loss_records]).float()
        if loss_records
        else torch.empty(0, device=model.device)
    )
    loss_start = _scalar_tensor_to_float(loss_values[0] if loss_values.numel() else None)
    loss_end = _scalar_tensor_to_float(loss_values[-1] if loss_values.numel() else None)
    mean_loss_first_8 = _scalar_tensor_to_float(
        loss_values[:8].mean() if loss_values.numel() else None
    )
    mean_loss_last_8 = _scalar_tensor_to_float(
        loss_values[-8:].mean() if loss_values.numel() else None
    )
    max_gradient_norm = _scalar_tensor_to_float(max_grad_norm_record) or 0.0
    plif_stats_delta = language_plif_triton_stats_delta(
        plif_stats_before,
        language_plif_triton_stats(),
    )
    routing_telemetry = last_state_block_telemetry.get("routing")
    if not isinstance(routing_telemetry, dict):
        routing_telemetry = {}
    memory_telemetry = last_state_block_telemetry.get("memory")
    if not isinstance(memory_telemetry, dict):
        memory_telemetry = {}
    return {
        "surface": "marulho_language_training_experiment_update.v1",
        "train_batch_count": len(batches),
        "train_epochs": max(1, int(config.train_epochs)),
        "batch_size": int(config.batch_size),
        "max_tokens_per_optimizer_step": max(
            (int(batch.target_ids.numel()) for batch in batches),
            default=0,
        ),
        "optimizer": optimizer_policy,
        "optimizer_policy": optimizer_policy,
        "learning_rate": float(config.learning_rate),
        "optimizer_step_count": int(optimizer_step_count),
        "recurrent_gradient_horizon": int(model.config.recurrent_gradient_horizon),
        "truncated_recurrent_bptt": bool(
            int(model.config.recurrent_gradient_horizon) > 0
        ),
        "gradient_horizon_policy": (
            "bounded_recurrent_state_detach"
            if int(model.config.recurrent_gradient_horizon) > 0
            else "full_sequence_bptt"
        ),
        "truncated_bptt_boundary_count_per_batch": int(
            last_state_block_telemetry.get("truncated_bptt_boundary_count", 0) or 0
        ),
        "state_block_gradient_horizon_policy": str(
            last_state_block_telemetry.get(
                "gradient_horizon_policy",
                (
                    "bounded_recurrent_state_detach"
                    if int(model.config.recurrent_gradient_horizon) > 0
                    else "full_sequence_bptt"
                ),
            )
        ),
        "state_block_projection_mode": str(
            last_state_block_telemetry.get("state_block_projection_mode", "unknown")
        ),
        "state_output_projection_batched": bool(
            last_state_block_telemetry.get("state_block_projection_mode")
            == "batched_token_and_state_output_projection_recurrent_loop"
        ),
        "expert_dispatch_backend": str(
            routing_telemetry.get("expert_dispatch_backend", "unknown")
        ),
        "expert_training_dispatch_batched_matmul": bool(
            routing_telemetry.get("expert_dispatch_backend")
            == "torch_selected_expert_batched_matmul_dispatch"
        ),
        "memory_enabled": bool(memory_telemetry.get("enabled", False)),
        "memory_total_slots": int(memory_telemetry.get("total_slots", 0) or 0),
        "memory_candidate_slot_count": int(
            memory_telemetry.get("candidate_slot_count", 0) or 0
        ),
        "memory_active_slots_per_token": int(
            memory_telemetry.get("active_slots_per_token", 0) or 0
        ),
        "memory_candidate_slots_scored": int(
            memory_telemetry.get("candidate_slots_scored", 0) or 0
        ),
        "memory_runs_all_slots": bool(memory_telemetry.get("runs_all_slots", False)),
        "memory_candidate_id_source": memory_telemetry.get("candidate_id_source"),
        "memory_slot_retrieval_backend": memory_telemetry.get(
            "memory_slot_retrieval_backend"
        ),
        "memory_slot_triton_stats_delta": memory_telemetry.get(
            "memory_slot_triton_stats_delta"
        ),
        "memory_gate_readback": bool(
            memory_telemetry.get("memory_gate_readback", False)
        ),
        "memory_slot_initialization": memory_telemetry.get(
            "memory_slot_initialization"
        ),
        "memory_slot_init_std": memory_telemetry.get("memory_slot_init_std"),
        "token_count": int(token_count),
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(token_count) / elapsed if elapsed > 0.0 else 0.0,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "loss_delta": (
            float(loss_end) - float(loss_start)
            if loss_start is not None and loss_end is not None
            else 0.0
        ),
        "mean_loss_first_8": mean_loss_first_8 or 0.0,
        "mean_loss_last_8": mean_loss_last_8 or 0.0,
        "max_gradient_norm": max_gradient_norm,
        "device": str(model.device),
        "metric_readback_mode": "deferred_gpu_scalar_aggregation",
        "per_batch_metric_cpu_sync": False,
        "hot_update_evidence_mode": "post_window_telemetry_probe",
        "per_step_evidence_dict_build": False,
        "telemetry_probe_outside_measured_window": bool(
            telemetry_probe_outside_measured_window
        ),
        "post_window_telemetry_probe_batch_tokens": int(
            post_window_telemetry_probe_batch_tokens
        ),
        "cuda_math_policy": cuda_math_policy,
        "training_stage_profile": stage_profiler.report(),
        "gradient_clip_mode": (
            "disabled"
            if float(config.max_grad_norm) <= 0.0 or gradient_clip_interval <= 0
            else (
                "sparse_aware_device_norm_every_step"
                if gradient_clip_interval == 1
                else "sparse_aware_device_norm_every_n_steps"
            )
        ),
        "gradient_clip_max_norm": float(config.max_grad_norm),
        "gradient_clip_interval": int(gradient_clip_interval),
        "gradient_clip_applied_step_count": int(gradient_clip_applied_step_count),
        "gradient_clip_skipped_step_count": int(gradient_clip_skipped_step_count),
        "gradient_norm_observed_step_count": int(gradient_clip_applied_step_count),
        "loss_kind": last_loss_kind,
        "loss_evidence": last_loss_evidence,
        "sampled_vocab_precompute": sampled_vocab_precompute,
        "memory_candidate_precompute": sampled_vocab_precompute.get(
            "memory_candidate_precompute",
            {
                "surface": "marulho_language_memory_candidate_batch_precompute.v1",
                "enabled": False,
                "reason": "precompute_report_missing",
                "batch_count": 0,
                "device": str(model.device),
            },
        ),
        "route_candidate_precompute": sampled_vocab_precompute.get(
            "route_candidate_precompute",
            {
                "surface": "marulho_language_route_candidate_batch_precompute.v1",
                "enabled": False,
                "reason": "precompute_report_missing",
                "batch_count": 0,
                "device": str(model.device),
            },
        ),
        "route_candidate_id_source": routing_telemetry.get("candidate_id_source"),
        "route_precomputed_candidate_ids_used": bool(
            routing_telemetry.get("precomputed_candidate_ids_used", False)
        ),
        "sampled_vocab_training": bool(
            last_loss_evidence.get("sampled_vocab_training", False)
        ),
        "full_vocab_logits_materialized": bool(
            last_loss_evidence.get("full_vocab_logits_materialized", True)
        ),
        "loss_record_count": len(loss_records),
        "cuda_synchronized_before_timing_start": bool(
            cuda_synchronized_before_timing_start
        ),
        "cuda_synchronized_before_timing_stop": bool(
            cuda_synchronized_before_timing_stop
        ),
        "language_plif_triton": plif_stats_delta,
        "plif_surrogate_triton_used": bool(
            int(plif_stats_delta.get("triton_backward_calls", 0) or 0) > 0
        ),
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
    corpus = _read_corpus(corpus_path)
    tokenizer = _build_tokenizer(corpus, cfg)
    device = _resolve_device(str(cfg.device))
    cuda_math_policy = _apply_cuda_math_policy(device, cfg)
    try:
        split = build_language_model_splits(
            [corpus],
            tokenizer,
            sequence_length=int(cfg.sequence_length),
            eval_fraction=0.20,
            stride=int(cfg.stride),
            batch_size=int(cfg.batch_size),
            device=device,
            max_train_batches=int(cfg.max_train_batches),
            max_eval_batches=int(cfg.max_eval_batches),
            window_selection=str(cfg.window_selection),
        )
        train_batches = _trim_batches(split.train, limit=int(cfg.max_train_batches))
        model = MarulhoLanguageModel(_model_config(tokenizer, cfg)).to(device)

        before_eval = evaluate_language_model(model, split.eval)
        before_generations = [
            _decoded_generation(
                model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=min(16, int(cfg.generation_tokens)),
                corpus=corpus,
            )
            for prompt in prompts
        ]
        training = _train_language_model(
            model,
            train_batches,
            config=cfg,
            cuda_math_policy=cuda_math_policy,
        )
        after_eval = evaluate_language_model(model, split.eval)
        after_generations = [
            _decoded_generation(
                model,
                tokenizer,
                prompt=prompt,
                max_new_tokens=int(cfg.generation_tokens),
                corpus=corpus,
            )
            for prompt in prompts
        ]
        before_generation_quality = _generation_quality_summary(before_generations)
        after_generation_quality = _generation_quality_summary(after_generations)

        checkpoint_path = save_language_model_checkpoint(
            output.with_name(f"{output.stem}-checkpoint.pt"),
            model,
            tokenizer,
            metadata={
                "experiment_report": str(output),
                "split": split.report,
                "config": asdict(cfg),
                "cuda_math_policy": cuda_math_policy,
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
            "state_core": str(model.config.state_core),
            "parameter_inventory": _parameter_inventory(model),
            "model_vocab_size": int(model.config.vocab_size),
            "tokenizer_vocab_size": int(tokenizer.vocab_size),
            "tokenizer": {
                "surface": tokenizer.state_dict().get("surface"),
                "vocabulary_hash": tokenizer.vocabulary_hash(),
                "vocab_size": int(tokenizer.vocab_size),
                "corpus_token_count": len(
                    tokenizer.encode(corpus, add_bos=False, add_eos=False)
                ),
                "corpus_utf8_byte_count": len(corpus.encode("utf-8")),
                "vocabulary_trained_by_marulho": bool(
                    tokenizer.state_dict().get("vocabulary_trained_by_marulho", False)
                ),
                "loads_external_checkpoint": False,
            },
            "generation_vocab_size": int(model.generation_vocab_size),
            "padded_vocab_rows": max(
                0,
                int(model.config.vocab_size) - int(tokenizer.vocab_size),
            ),
            "generation_decode": model.generation_decode_policy(),
            "cuda_math_policy": cuda_math_policy,
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
            "generation_quality_before": before_generation_quality,
            "generation_quality_after": after_generation_quality,
            "generation_quality_delta": {
                "mean_source_prefix_match_chars_delta": (
                    after_generation_quality["mean_source_prefix_match_chars"]
                    - before_generation_quality["mean_source_prefix_match_chars"]
                ),
                "next_character_match_rate_delta": (
                    after_generation_quality["next_character_match_rate"]
                    - before_generation_quality["next_character_match_rate"]
                ),
                "mean_distinct_bigram_fraction_delta": (
                    after_generation_quality["mean_distinct_bigram_fraction"]
                    - before_generation_quality["mean_distinct_bigram_fraction"]
                ),
            },
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
                "model_vocab_size": sustained_report.get("model_vocab_size"),
                "tokenizer_vocab_size": sustained_report.get("tokenizer_vocab_size"),
                "generation_vocab_size": sustained_report.get("generation_vocab_size"),
                "padded_vocab_rows": sustained_report.get("padded_vocab_rows"),
                "generation_decode": sustained_report.get("generation_decode"),
                "memory_slots": sustained_report.get("memory_slots"),
            },
            "experiment_review": {
                "fast_mutable_experiment": True,
                "records_actual_training": training["token_count"] > 0,
                "records_actual_generation": bool(after_generations),
                "records_generation_quality_probe": bool(after_generations),
                "records_sustained_inference": int(sustained_report["token_delta"]) > 0,
                "records_sampled_vocab_training": bool(
                    training.get("sampled_vocab_training", False)
                ),
                "records_padded_vocab_decode_policy": bool(
                    max(0, int(model.config.vocab_size) - int(tokenizer.vocab_size)) > 0
                ),
                "records_memory_slot_path": bool(
                    training.get("memory_enabled", False)
                    and isinstance(sustained_report.get("memory_slots"), dict)
                    and sustained_report.get("memory_slots", {}).get("enabled") is True
                ),
                "records_bounded_memory_slot_path": bool(
                    training.get("memory_enabled", False)
                    and not bool(training.get("memory_runs_all_slots", False))
                    and isinstance(sustained_report.get("memory_slots"), dict)
                    and sustained_report.get("memory_slots", {}).get("enabled") is True
                    and not bool(
                        sustained_report.get("memory_slots", {}).get(
                            "runs_all_slots",
                            False,
                        )
                    )
                ),
                "records_cuda_math_policy": bool(cuda_math_policy.get("applied", False)),
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
    finally:
        before_policy = cuda_math_policy.get("before")
        if isinstance(before_policy, dict):
            _restore_cuda_math_policy(before_policy)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--tokenizer-kind", choices=("byte", "bpe"), default="byte")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=4096)
    parser.add_argument("--tokenizer-min-frequency", type=int, default=2)
    parser.add_argument("--model-vocab-size", type=int, default=0)
    parser.add_argument("--sampled-vocab-size", type=int, default=0)
    parser.add_argument("--disable-sparse-vocab-optimizer", action="store_true")
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument(
        "--state-core",
        choices=("selective_spiking", "selective_continuous", "gru", "transformer"),
        default="selective_spiking",
    )
    parser.add_argument("--state-layers", type=int, default=1)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--transformer-context-length", type=int, default=256)
    parser.add_argument("--transformer-mlp-ratio", type=float, default=4.0)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--tie-embeddings", action="store_true")
    parser.add_argument("--expert-count", type=int, default=8)
    parser.add_argument("--active-expert-count", type=int, default=2)
    parser.add_argument("--route-candidate-count", type=int, default=4)
    parser.add_argument("--expert-hidden-dim", type=int, default=96)
    parser.add_argument("--recurrent-gradient-horizon", type=int, default=0)
    parser.add_argument("--memory-slot-count", type=int, default=0)
    parser.add_argument("--memory-slot-candidate-count", type=int, default=0)
    parser.add_argument("--active-memory-slot-count", type=int, default=1)
    parser.add_argument("--memory-slot-init-std", type=float, default=0.02)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--max-eval-batches", type=int, default=64)
    parser.add_argument(
        "--window-selection",
        choices=("stratified", "prefix"),
        default="stratified",
    )
    parser.add_argument("--train-epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-clip-interval", type=int, default=1)
    parser.add_argument("--generation-tokens", type=int, default=48)
    parser.add_argument("--sustained-target-tokens", type=int, default=512)
    parser.add_argument("--sustained-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--profile-training-stages", action="store_true")
    parser.add_argument("--disable-cuda-tf32", action="store_true")
    parser.add_argument(
        "--cuda-float32-matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = LanguageTrainingExperimentConfig(
        tokenizer_kind=args.tokenizer_kind,
        tokenizer_vocab_size=max(512, int(args.tokenizer_vocab_size)),
        tokenizer_min_frequency=max(1, int(args.tokenizer_min_frequency)),
        model_vocab_size=args.model_vocab_size,
        sampled_vocab_size=args.sampled_vocab_size,
        sparse_vocab_optimizer=not bool(args.disable_sparse_vocab_optimizer),
        embedding_dim=args.embedding_dim,
        state_dim=args.state_dim,
        state_core=args.state_core,
        state_layers=max(1, int(args.state_layers)),
        attention_heads=max(1, int(args.attention_heads)),
        transformer_context_length=max(2, int(args.transformer_context_length)),
        transformer_mlp_ratio=float(args.transformer_mlp_ratio),
        transformer_dropout=float(args.transformer_dropout),
        tie_embeddings=bool(args.tie_embeddings),
        expert_count=args.expert_count,
        active_expert_count=args.active_expert_count,
        route_candidate_count=args.route_candidate_count,
        expert_hidden_dim=args.expert_hidden_dim,
        recurrent_gradient_horizon=max(0, int(args.recurrent_gradient_horizon)),
        memory_slot_count=max(0, int(args.memory_slot_count)),
        memory_slot_candidate_count=max(0, int(args.memory_slot_candidate_count)),
        active_memory_slot_count=max(1, int(args.active_memory_slot_count)),
        memory_slot_init_std=float(args.memory_slot_init_std),
        sequence_length=args.sequence_length,
        stride=args.stride,
        batch_size=args.batch_size,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        window_selection=args.window_selection,
        train_epochs=args.train_epochs,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        gradient_clip_interval=max(0, int(args.gradient_clip_interval)),
        generation_tokens=args.generation_tokens,
        sustained_target_tokens=args.sustained_target_tokens,
        sustained_timeout_seconds=args.sustained_timeout_seconds,
        profile_training_stages=bool(args.profile_training_stages),
        cuda_allow_tf32=not bool(args.disable_cuda_tf32),
        cuda_float32_matmul_precision=args.cuda_float32_matmul_precision,
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
