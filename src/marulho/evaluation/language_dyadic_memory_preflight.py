"""Matched causal dyadic-memory preflight before any V15 language run."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from marulho.reporting.readme_reports import write_json_report_with_readme


SURFACE = "marulho_dyadic_memory_preflight.v1"
ARTIFACT_KIND = "marulho_dyadic_memory_preflight"
DYADIC_MODES = ("raw_average", "shuffled_contrast", "haar")
ARM_NAMES = ("flat_gru", *DYADIC_MODES)
EVALUATION_PROFILES = (
    "heldout_128",
    "long_256",
    "long_512",
    "overwrite_256",
)
ADMIT_DECISION = "admit_v15_dyadic_state_to_language_falsifier"


@dataclass(frozen=True)
class RecallTaskConfig:
    key_count: int = 16
    value_count: int = 16
    query_count: int = 8
    train_sequence_length: int = 128
    embedding_dim: int = 32
    dyadic_levels: int = 7
    bank_width: int = 16

    @property
    def total_state_width(self) -> int:
        return int(self.dyadic_levels) * int(self.bank_width)


@dataclass(frozen=True)
class DyadicPreflightConfig:
    train_steps: int = 2400
    batch_size: int = 128
    eval_batches: int = 8
    learning_rate: float = 2.0e-3
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    data_seed: int = 5101
    model_seeds: tuple[int, ...] = (5201, 5202, 5203)
    minimum_long_accuracy_gain: float = 0.03
    maximum_guard_accuracy_regret: float = 0.01


@dataclass(frozen=True)
class RecallBatch:
    roles: torch.Tensor
    keys: torch.Tensor
    values: torch.Tensor
    targets: torch.Tensor
    delays: torch.Tensor
    profile: str

    @property
    def sequence_length(self) -> int:
        return int(self.roles.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.roles.shape[0])

    @property
    def query_count(self) -> int:
        return int((self.targets >= 0).sum().item())

    def to(self, device: torch.device) -> RecallBatch:
        return RecallBatch(
            roles=self.roles.to(device),
            keys=self.keys.to(device),
            values=self.values.to(device),
            targets=self.targets.to(device),
            delays=self.delays.to(device),
            profile=self.profile,
        )


def _profile_windows(
    profile: str,
) -> tuple[int, tuple[int, int], tuple[int, int], tuple[int, int] | None]:
    if profile in {"train_128", "heldout_128"}:
        return 128, (4, 52), (72, 124), None
    if profile == "long_256":
        return 256, (4, 84), (164, 252), None
    if profile == "long_512":
        return 512, (4, 132), (380, 508), None
    if profile == "overwrite_256":
        return 256, (4, 68), (196, 252), (112, 176)
    raise ValueError(f"Unknown recall profile: {profile}")


def _slotted_positions(
    *,
    batch_size: int,
    count: int,
    window: tuple[int, int],
    generator: torch.Generator,
) -> torch.Tensor:
    start, end = window
    width = (int(end) - int(start)) // int(count)
    if width < 1:
        raise ValueError("Recall position window is too small")
    bases = int(start) + torch.arange(int(count), dtype=torch.long) * width
    jitter = torch.randint(
        0,
        width,
        (int(batch_size), int(count)),
        generator=generator,
    )
    return bases.unsqueeze(0) + jitter


def generate_recall_batch(
    task: RecallTaskConfig,
    *,
    profile: str,
    batch_size: int,
    seed: int,
) -> RecallBatch:
    """Generate label-safe writes, distractors, and later key-only queries."""

    if int(task.query_count) > int(task.key_count):
        raise ValueError("query_count cannot exceed key_count")
    sequence_length, write_window, query_window, overwrite_window = (
        _profile_windows(profile)
    )
    if profile == "train_128" and sequence_length != task.train_sequence_length:
        raise ValueError("train profile length differs from task configuration")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    batch = int(batch_size)
    queries = int(task.query_count)
    roles = torch.full((batch, sequence_length), 3, dtype=torch.long)
    keys = torch.randint(
        0,
        int(task.key_count),
        (batch, sequence_length),
        generator=generator,
    )
    values = torch.randint(
        0,
        int(task.value_count),
        (batch, sequence_length),
        generator=generator,
    )
    targets = torch.full((batch, sequence_length), -100, dtype=torch.long)
    delays = torch.full((batch, sequence_length), -1, dtype=torch.long)
    selected_keys = torch.rand(
        batch,
        int(task.key_count),
        generator=generator,
    ).argsort(dim=1)[:, :queries]
    target_values = torch.randint(
        0,
        int(task.value_count),
        (batch, queries),
        generator=generator,
    )
    write_positions = _slotted_positions(
        batch_size=batch,
        count=queries,
        window=write_window,
        generator=generator,
    )
    query_positions_ordered = _slotted_positions(
        batch_size=batch,
        count=queries,
        window=query_window,
        generator=generator,
    )
    query_permutation = torch.rand(
        batch,
        queries,
        generator=generator,
    ).argsort(dim=1)
    query_positions = query_positions_ordered.gather(1, query_permutation)
    rows = torch.arange(batch, dtype=torch.long).unsqueeze(1).expand(-1, queries)

    roles[rows, write_positions] = 1
    keys[rows, write_positions] = selected_keys
    values[rows, write_positions] = target_values
    latest_write_positions = write_positions
    if overwrite_window is not None:
        replacement_positions = _slotted_positions(
            batch_size=batch,
            count=queries,
            window=overwrite_window,
            generator=generator,
        )
        old_offset = torch.randint(
            1,
            int(task.value_count),
            (batch, queries),
            generator=generator,
        )
        old_values = (target_values + old_offset) % int(task.value_count)
        values[rows, write_positions] = old_values
        roles[rows, replacement_positions] = 1
        keys[rows, replacement_positions] = selected_keys
        values[rows, replacement_positions] = target_values
        latest_write_positions = replacement_positions

    roles[rows, query_positions] = 2
    keys[rows, query_positions] = selected_keys
    values[rows, query_positions] = int(task.value_count)
    targets[rows, query_positions] = target_values
    delays[rows, query_positions] = query_positions - latest_write_positions
    if bool((delays[rows, query_positions] <= 0).any()):
        raise RuntimeError("Recall query does not follow its latest write")
    return RecallBatch(
        roles=roles,
        keys=keys,
        values=values,
        targets=targets,
        delays=delays,
        profile=profile,
    )


def oracle_recall_accuracy(batch: RecallBatch, task: RecallTaskConfig) -> float:
    memory = torch.full(
        (batch.batch_size, int(task.key_count)),
        -1,
        dtype=torch.long,
    )
    correct = 0
    total = 0
    rows = torch.arange(batch.batch_size, dtype=torch.long)
    for index in range(batch.sequence_length):
        write = batch.roles[:, index] == 1
        if bool(write.any()):
            memory[rows[write], batch.keys[write, index]] = batch.values[
                write, index
            ]
        query = batch.roles[:, index] == 2
        if bool(query.any()):
            observed = memory[rows[query], batch.keys[query, index]]
            expected = batch.targets[query, index]
            correct += int((observed == expected).sum())
            total += int(query.sum())
    return correct / max(1, total)


class RecallInputEmbedding(nn.Module):
    def __init__(self, task: RecallTaskConfig) -> None:
        super().__init__()
        width = int(task.embedding_dim)
        self.role = nn.Embedding(4, width)
        self.key = nn.Embedding(int(task.key_count), width)
        self.value = nn.Embedding(int(task.value_count) + 1, width)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        roles: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        return self.norm(self.role(roles) + self.key(keys) + self.value(values))


class FlatGatedRecallModel(nn.Module):
    surface = "marulho_flat_gated_recall_control.v1"

    def __init__(self, task: RecallTaskConfig) -> None:
        super().__init__()
        self.task = task
        self.embedding = RecallInputEmbedding(task)
        self.recurrence = nn.GRU(
            input_size=int(task.embedding_dim),
            hidden_size=int(task.total_state_width),
            batch_first=True,
        )
        self.readout = nn.Linear(
            int(task.embedding_dim) + int(task.total_state_width),
            int(task.value_count),
        )

    def forward(
        self,
        roles: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = self.embedding(roles, keys, values)
        state, _final = self.recurrence(embedded)
        return {
            "logits": self.readout(torch.cat((embedded, state), dim=-1)),
            "state": state,
        }

    def state_bytes(self, batch_size: int, *, element_size: int = 4) -> int:
        return int(batch_size) * int(self.task.total_state_width) * int(element_size)

    def recurrent_updates_per_sequence(self, sequence_length: int) -> int:
        return int(sequence_length)


class DyadicRecallModel(nn.Module):
    surface = "marulho_dyadic_recall_candidate.v1"

    def __init__(
        self,
        task: RecallTaskConfig,
        *,
        mode: str = "haar",
        contrast_seed: int = 4301,
    ) -> None:
        super().__init__()
        if mode not in DYADIC_MODES:
            raise ValueError(f"mode must be one of {DYADIC_MODES}")
        self.task = task
        self._mode_name = str(mode)
        self.embedding = RecallInputEmbedding(task)
        self.banks = nn.ModuleList(
            nn.GRU(
                input_size=2 * int(task.embedding_dim),
                hidden_size=int(task.bank_width),
                batch_first=True,
            )
            for _ in range(int(task.dyadic_levels))
        )
        self.readout = nn.Linear(
            int(task.embedding_dim) + int(task.total_state_width),
            int(task.value_count),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(contrast_seed))
        for level in range(int(task.dyadic_levels)):
            block = 2 ** (level + 1)
            haar = torch.cat(
                (
                    torch.ones(block // 2, dtype=torch.float32),
                    -torch.ones(block // 2, dtype=torch.float32),
                )
            )
            shuffled = haar[torch.randperm(block, generator=generator)]
            self.register_buffer(f"haar_contrast_{level}", haar)
            self.register_buffer(f"shuffled_contrast_{level}", shuffled)

    def set_mode(self, mode: str) -> None:
        if mode not in DYADIC_MODES:
            raise ValueError(f"mode must be one of {DYADIC_MODES}")
        self._mode_name = str(mode)

    def _block_features(
        self,
        embedded: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        block = 2 ** (int(level) + 1)
        usable = (int(embedded.shape[1]) // block) * block
        if usable < block:
            return embedded.new_empty(
                int(embedded.shape[0]),
                0,
                2 * int(embedded.shape[-1]),
            )
        chunks = embedded[:, :usable].reshape(
            int(embedded.shape[0]),
            usable // block,
            block,
            int(embedded.shape[-1]),
        )
        approximation = chunks.sum(dim=2) / math.sqrt(float(block))
        if self._mode_name == "raw_average":
            contrast = approximation
        else:
            name = (
                f"haar_contrast_{level}"
                if self._mode_name == "haar"
                else f"shuffled_contrast_{level}"
            )
            signs = getattr(self, name).to(device=embedded.device, dtype=embedded.dtype)
            contrast = torch.einsum("bnse,s->bne", chunks, signs) / math.sqrt(
                float(block)
            )
        return torch.cat((approximation, contrast), dim=-1)

    def forward(
        self,
        roles: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        embedded = self.embedding(roles, keys, values)
        batch_size, time_steps, _ = embedded.shape
        timelines: list[torch.Tensor] = []
        positions = torch.arange(int(time_steps), device=embedded.device)
        for level, bank in enumerate(self.banks):
            block = 2 ** (int(level) + 1)
            features = self._block_features(embedded, level)
            if int(features.shape[1]) == 0:
                timelines.append(
                    embedded.new_zeros(
                        int(batch_size), int(time_steps), int(self.task.bank_width)
                    )
                )
                continue
            completed, _final = bank(features)
            padded = torch.cat(
                (
                    completed.new_zeros(
                        int(batch_size), 1, int(self.task.bank_width)
                    ),
                    completed,
                ),
                dim=1,
            )
            completed_count = torch.div(
                positions + 1,
                block,
                rounding_mode="floor",
            ).clamp_max(int(completed.shape[1]))
            timelines.append(padded.index_select(1, completed_count))
        state = torch.cat(timelines, dim=-1)
        return {
            "logits": self.readout(torch.cat((embedded, state), dim=-1)),
            "state": state,
        }

    def state_bytes(self, batch_size: int, *, element_size: int = 4) -> int:
        return int(batch_size) * int(self.task.total_state_width) * int(element_size)

    def recurrent_updates_per_sequence(self, sequence_length: int) -> int:
        return sum(
            int(sequence_length) // (2 ** (level + 1))
            for level in range(int(self.task.dyadic_levels))
        )


def _masked_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    selected = targets >= 0
    if not bool(selected.any()):
        raise ValueError("Recall batch contains no query targets")
    return F.cross_entropy(logits[selected], targets[selected])


def _state_geometry(states: torch.Tensor) -> dict[str, Any]:
    matrix = states.detach().float().cpu()
    if int(matrix.shape[0]) < 2:
        raise ValueError("State geometry requires at least two samples")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    probability = variance / variance.sum().clamp_min(1.0e-12)
    participation = variance.sum().square() / variance.square().sum().clamp_min(
        1.0e-12
    )
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1.0e-12).log()).sum()
    )
    return {
        "surface": "marulho_recall_state_geometry.v1",
        "sample_count": int(matrix.shape[0]),
        "ambient_dimension": int(matrix.shape[1]),
        "matrix_rank": int(torch.linalg.matrix_rank(centered)),
        "participation_ratio": float(participation),
        "effective_rank": float(effective_rank),
        "mean_state_norm": float(matrix.norm(dim=-1).mean()),
        "labels_used_for_geometry_only": True,
        "promotion_metric": False,
    }


@torch.no_grad()
def evaluate_recall_model(
    model: nn.Module,
    batches: Sequence[RecallBatch],
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    query_count = 0
    delay_values: list[torch.Tensor] = []
    state_values: list[torch.Tensor] = []
    started = time.perf_counter()
    for batch in batches:
        active = batch.to(device)
        output = model(active.roles, active.keys, active.values)
        selected = active.targets >= 0
        selected_logits = output["logits"][selected]
        selected_targets = active.targets[selected]
        loss_sum += float(
            F.cross_entropy(
                selected_logits,
                selected_targets,
                reduction="sum",
            ).cpu()
        )
        correct += int((selected_logits.argmax(dim=-1) == selected_targets).sum())
        query_count += int(selected.sum())
        delay_values.append(active.delays[selected].detach().cpu())
        state_values.append(output["state"][selected].detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    delays = torch.cat(delay_values)
    states = torch.cat(state_values)
    return {
        "surface": "marulho_recall_evaluation.v1",
        "profile": batches[0].profile,
        "accuracy": correct / max(1, query_count),
        "loss": loss_sum / max(1, query_count),
        "query_count": query_count,
        "chance_accuracy": 1.0 / int(model.task.value_count),
        "minimum_delay": int(delays.min()),
        "mean_delay": float(delays.float().mean()),
        "maximum_delay": int(delays.max()),
        "queries_per_second": query_count / max(elapsed, 1.0e-12),
        "state_geometry": _state_geometry(states),
        "targets_used_for_metrics_only": True,
        "external_llm_used": False,
    }


def _all_parameters_received_gradient(model: nn.Module) -> bool:
    return all(parameter.grad is not None for parameter in model.parameters())


def _train_arm(
    name: str,
    model: nn.Module,
    *,
    task: RecallTaskConfig,
    config: DyadicPreflightConfig,
    schedule_seeds: Sequence[int],
    eval_batches: Mapping[str, Sequence[RecallBatch]],
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    trace_steps = {
        max(0, math.ceil(len(schedule_seeds) * fraction / 10) - 1)
        for fraction in range(1, 11)
    }
    trace: list[dict[str, Any]] = []
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_loss: torch.Tensor | None = None
    for step, seed in enumerate(schedule_seeds):
        batch = generate_recall_batch(
            task,
            profile="train_128",
            batch_size=int(config.batch_size),
            seed=int(seed),
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.roles, batch.keys, batch.values)
        final_loss = _masked_loss(output["logits"], batch.targets)
        final_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.gradient_clip))
        optimizer.step()
        if step in trace_steps:
            trace.append(
                {
                    "step": step + 1,
                    "loss": float(final_loss.detach().cpu()),
                }
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if final_loss is None:
        raise RuntimeError("Dyadic preflight training produced no steps")
    gradient_complete = _all_parameters_received_gradient(model)
    profiles = {
        profile: evaluate_recall_model(model, batches, device=device)
        for profile, batches in eval_batches.items()
    }
    parameter_count = sum(int(value.numel()) for value in model.parameters())
    processed_tokens = (
        len(schedule_seeds)
        * int(config.batch_size)
        * int(task.train_sequence_length)
    )
    return {
        "name": name,
        "surface": str(model.surface),
        "parameters": parameter_count,
        "state_width": int(task.total_state_width),
        "state_bytes_for_training_batch": model.state_bytes(
            int(config.batch_size)
        ),
        "recurrent_updates_per_training_sequence": (
            model.recurrent_updates_per_sequence(int(task.train_sequence_length))
        ),
        "processed_tokens": processed_tokens,
        "training_seconds": elapsed,
        "tokens_per_second": processed_tokens / max(elapsed, 1.0e-12),
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "final_batch_loss": float(final_loss.detach().cpu()),
        "loss_trace": trace,
        "all_parameters_received_final_gradient": gradient_complete,
        "profiles": profiles,
        "target_labels_enter_state_updates": False,
        "external_llm_used": False,
    }


def _schedule_sha256(seeds: Sequence[int]) -> str:
    payload = json.dumps([int(seed) for seed in seeds], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean_accuracy(
    replicates: Sequence[Mapping[str, Any]],
    arm: str,
    profile: str,
) -> float:
    return sum(
        float(replicate["arms"][arm]["profiles"][profile]["accuracy"])
        for replicate in replicates
    ) / len(replicates)


def dyadic_preflight_decision(
    replicates: Sequence[Mapping[str, Any]],
    *,
    requested_replicates: int,
    minimum_gain: float = 0.03,
    maximum_guard_regret: float = 0.01,
) -> str:
    if len(replicates) != int(requested_replicates):
        return "incomplete_v15_preflight_replicates"
    if any(set(replicate.get("arms") or {}) != set(ARM_NAMES) for replicate in replicates):
        return "incomplete_v15_preflight_arms"
    long_profiles = ("long_256", "long_512")
    controls = ("flat_gru", "raw_average", "shuffled_contrast")
    for profile in long_profiles:
        haar = _mean_accuracy(replicates, "haar", profile)
        if any(
            haar - _mean_accuracy(replicates, control, profile)
            < float(minimum_gain)
            for control in controls
        ):
            break
    else:
        guard_pass = all(
            _mean_accuracy(replicates, "haar", profile)
            >= _mean_accuracy(replicates, "flat_gru", profile)
            - float(maximum_guard_regret)
            for profile in ("heldout_128", "overwrite_256")
        )
        replicated = all(
            sum(
                float(replicate["arms"]["haar"]["profiles"][profile]["accuracy"])
                > float(
                    replicate["arms"]["flat_gru"]["profiles"][profile][
                        "accuracy"
                    ]
                )
                for replicate in replicates
            )
            >= math.ceil(len(replicates) * 2 / 3)
            for profile in long_profiles
        )
        if guard_pass and replicated:
            return ADMIT_DECISION
    if all(
        _mean_accuracy(replicates, "raw_average", profile)
        - _mean_accuracy(replicates, "flat_gru", profile)
        >= float(minimum_gain)
        for profile in long_profiles
    ):
        return "redesign_v15_retain_multiscale_clocks_reject_haar_ordering"
    return "retire_v15_dyadic_preflight_no_gated_recurrence_gain"


def _resolve_device(requested: str) -> torch.device:
    normalized = str(requested).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(normalized)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for dyadic preflight but unavailable")
    return resolved


def run_dyadic_memory_preflight(
    *,
    output_path: str | Path,
    task: RecallTaskConfig = RecallTaskConfig(),
    config: DyadicPreflightConfig = DyadicPreflightConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    if int(config.train_steps) < 1 or int(config.batch_size) < 1:
        raise ValueError("Dyadic preflight train_steps and batch_size must be positive")
    if int(config.eval_batches) < 1 or not config.model_seeds:
        raise ValueError("Dyadic preflight requires eval batches and model seeds")
    if int(task.dyadic_levels) < 1 or int(task.bank_width) < 1:
        raise ValueError("Dyadic state dimensions must be positive")
    if 2 ** int(task.dyadic_levels) > int(task.train_sequence_length):
        raise ValueError("Largest dyadic block must fit the training sequence")
    resolved = _resolve_device(device)
    started = time.perf_counter()
    schedule_seeds = tuple(
        int(config.data_seed) + step for step in range(int(config.train_steps))
    )
    eval_sets = {
        profile: tuple(
            generate_recall_batch(
                task,
                profile=profile,
                batch_size=int(config.batch_size),
                seed=int(config.data_seed)
                + 100_000
                + profile_index * 10_000
                + batch_index,
            )
            for batch_index in range(int(config.eval_batches))
        )
        for profile_index, profile in enumerate(EVALUATION_PROFILES)
    }
    oracle = {
        profile: min(oracle_recall_accuracy(batch, task) for batch in batches)
        for profile, batches in eval_sets.items()
    }
    if any(value != 1.0 for value in oracle.values()):
        raise RuntimeError(f"Recall task oracle failed: {oracle}")

    previous_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    previous_matmul_precision = torch.get_float32_matmul_precision()
    if resolved.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    replicates: list[dict[str, Any]] = []
    try:
        for replicate_index, model_seed in enumerate(config.model_seeds):
            torch.manual_seed(int(model_seed))
            candidate = DyadicRecallModel(task, mode="haar")
            candidate_initial = {
                name: value.detach().clone()
                for name, value in candidate.state_dict().items()
            }
            torch.manual_seed(int(model_seed) + 1)
            flat = FlatGatedRecallModel(task)
            flat.embedding.load_state_dict(candidate.embedding.state_dict(), strict=True)
            flat.readout.load_state_dict(candidate.readout.state_dict(), strict=True)
            flat_initial = {
                name: value.detach().clone() for name, value in flat.state_dict().items()
            }
            arms: dict[str, dict[str, Any]] = {}
            arms["flat_gru"] = _train_arm(
                "flat_gru",
                flat,
                task=task,
                config=config,
                schedule_seeds=schedule_seeds,
                eval_batches=eval_sets,
                device=resolved,
            )
            for mode in DYADIC_MODES:
                active = DyadicRecallModel(task, mode=mode)
                active.load_state_dict(candidate_initial, strict=True)
                active.set_mode(mode)
                arms[mode] = _train_arm(
                    mode,
                    active,
                    task=task,
                    config=config,
                    schedule_seeds=schedule_seeds,
                    eval_batches=eval_sets,
                    device=resolved,
                )
            replicates.append(
                {
                    "replicate_index": replicate_index,
                    "model_seed": int(model_seed),
                    "schedule_sha256": _schedule_sha256(schedule_seeds),
                    "candidate_modes_exact_initial_state": True,
                    "flat_and_candidate_embedding_exact_initial_state": all(
                        torch.equal(
                            flat_initial[f"embedding.{name}"],
                            candidate_initial[f"embedding.{name}"],
                        )
                        for name in candidate.embedding.state_dict()
                    ),
                    "flat_and_candidate_readout_exact_initial_state": all(
                        torch.equal(
                            flat_initial[f"readout.{name}"],
                            candidate_initial[f"readout.{name}"],
                        )
                        for name in candidate.readout.state_dict()
                    ),
                    "arms": arms,
                }
            )
            print(
                f"[dyadic-preflight] replicate {replicate_index + 1}/"
                f"{len(config.model_seeds)}: "
                + ", ".join(
                    f"{name}="
                    f"{row['profiles']['long_512']['accuracy']:.3f}"
                    for name, row in arms.items()
                ),
                flush=True,
            )
    finally:
        if resolved.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            torch.set_float32_matmul_precision(previous_matmul_precision)

    decision = dyadic_preflight_decision(
        replicates,
        requested_replicates=len(config.model_seeds),
        minimum_gain=float(config.minimum_long_accuracy_gain),
        maximum_guard_regret=float(config.maximum_guard_accuracy_regret),
    )
    means = {
        arm: {
            profile: _mean_accuracy(replicates, arm, profile)
            for profile in EVALUATION_PROFILES
        }
        for arm in ARM_NAMES
    }
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "task": asdict(task),
        "configuration": asdict(config),
        "device": str(resolved),
        "training_schedule": {
            "seed_count": len(schedule_seeds),
            "sha256": _schedule_sha256(schedule_seeds),
            "same_schedule_for_every_arm_and_replicate": True,
        },
        "anti_shortcut_contract": {
            "query_input_value_is_sentinel": int(task.value_count),
            "target_value_absent_from_query_input": True,
            "distractors_have_random_keys_and_values": True,
            "prediction_path_reads_targets": False,
            "state_updates_read_targets": False,
            "oracle_metrics_only": True,
        },
        "oracle_accuracy": oracle,
        "chance_accuracy": 1.0 / int(task.value_count),
        "research_basis": [
            "https://arxiv.org/abs/2312.04927",
            "https://arxiv.org/abs/2506.07920",
            "https://arxiv.org/abs/2507.00449",
            "https://arxiv.org/abs/2505.15105",
        ],
        "frozen_evidence_boundary": {
            "mechanical_smoke_data_seed": 4101,
            "mechanical_smoke_model_seed": 4201,
            "smoke_reports_deleted": True,
            "training_steps_selected_from_smoke_convergence": 2400,
            "architecture_changed_after_smoke": False,
            "thresholds_changed_after_smoke": False,
            "final_data_and_evaluation_seeds_fresh": True,
            "final_model_seeds_fresh": True,
        },
        "replicates": replicates,
        "mean_accuracies": means,
        "decision": decision,
        "promotion_boundary": {
            "language_architecture_admitted": decision == ADMIT_DECISION,
            "base_language_quality_promoted": False,
            "runtime_install_allowed": False,
            "synthetic_result_is_language_evidence": False,
        },
        "experiment_wall_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO V15 Dyadic Memory Preflight",
    )
    print(f"[dyadic-preflight] decision {decision}", flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=2400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--model-seed", action="append", type=int)
    parser.add_argument("--data-seed", type=int, default=5101)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    seeds = (
        tuple(int(value) for value in args.model_seed)
        if args.model_seed
        else DyadicPreflightConfig().model_seeds
    )
    config = DyadicPreflightConfig(
        train_steps=int(args.train_steps),
        batch_size=int(args.batch_size),
        eval_batches=int(args.eval_batches),
        data_seed=int(args.data_seed),
        model_seeds=seeds,
    )
    run_dyadic_memory_preflight(
        output_path=args.output,
        config=config,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
