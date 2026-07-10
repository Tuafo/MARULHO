"""Full-strength Transformer with a causal, utility-routed sparse event sidecar."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


@dataclass(frozen=True)
class SparseEventMemoryConfig(LanguageModelConfig):
    event_interval: int = 24
    specialist_count: int = 4
    specialist_rank: int = 32
    selection_mode: str = "utility"
    exploration_rate: float = 0.10
    counterfactual_rate: float = 0.125
    utility_loss_weight: float = 0.05
    compute_cost: float = 1.0e-4
    initial_residual_scale: float = 1.0e-3
    active_language_path: str = "marulho_sparse_event_memory_v2"


def _validate_sparse_config(config: SparseEventMemoryConfig) -> None:
    if int(config.event_interval) < 1:
        raise ValueError("event_interval must be positive")
    if int(config.specialist_count) < 2:
        raise ValueError("specialist_count must be at least two")
    if int(config.specialist_rank) < 1:
        raise ValueError("specialist_rank must be positive")
    if str(config.selection_mode) not in {"utility", "random", "dense"}:
        raise ValueError("selection_mode must be utility, random, or dense")
    for name, value in (
        ("exploration_rate", config.exploration_rate),
        ("counterfactual_rate", config.counterfactual_rate),
    ):
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(float(config.utility_loss_weight)) or float(
        config.utility_loss_weight
    ) < 0.0:
        raise ValueError("utility_loss_weight must be finite and non-negative")
    if not math.isfinite(float(config.compute_cost)) or float(config.compute_cost) < 0:
        raise ValueError("compute_cost must be finite and non-negative")
    if not math.isfinite(float(config.initial_residual_scale)):
        raise ValueError("initial_residual_scale must be finite")


class SparseEventMemorySidecar(nn.Module):
    """Runs one gathered specialist per completed event, or all for dense control."""

    def __init__(self, config: SparseEventMemoryConfig) -> None:
        super().__init__()
        self.width = int(config.state_dim)
        self.interval = int(config.event_interval)
        self.specialists = int(config.specialist_count)
        self.rank = int(config.specialist_rank)
        self.selection_mode = str(config.selection_mode)
        self.exploration_rate = float(config.exploration_rate)
        self.router = nn.Linear(self.width, self.specialists, bias=False)
        self.down = nn.Parameter(
            torch.empty(self.specialists, self.width, self.rank)
        )
        self.up = nn.Parameter(torch.empty(self.specialists, self.rank, self.width))
        self.residual_scale = nn.Parameter(
            torch.full(
                (self.specialists,), float(config.initial_residual_scale)
            )
        )
        nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.down, mean=0.0, std=0.02)
        nn.init.normal_(self.up, mean=0.0, std=0.02)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        return {
            "event_pending_sum": torch.zeros(
                int(batch_size), self.width, device=device, dtype=dtype
            ),
            "event_pending_count": torch.zeros((), device=device, dtype=torch.long),
            "event_current_residual": torch.zeros(
                int(batch_size), self.width, device=device, dtype=dtype
            ),
            "event_current_score": torch.zeros(
                int(batch_size), device=device, dtype=dtype
            ),
            "event_current_active": torch.zeros(
                int(batch_size), device=device, dtype=torch.bool
            ),
        }

    def _select(self, scores: torch.Tensor) -> torch.Tensor:
        batch_size = int(scores.shape[0])
        if self.selection_mode == "random":
            return torch.randint(
                self.specialists, (batch_size,), device=scores.device
            )
        selected = torch.argmax(scores, dim=-1)
        if self.training and self.exploration_rate > 0.0:
            explore = torch.rand(batch_size, device=scores.device) < self.exploration_rate
            random_selected = torch.randint(
                self.specialists, (batch_size,), device=scores.device
            )
            selected = torch.where(explore, random_selected, selected)
        return selected

    def _sparse_residual(
        self, summary: torch.Tensor, selected: torch.Tensor
    ) -> torch.Tensor:
        down = self.down.index_select(0, selected)
        up = self.up.index_select(0, selected)
        latent = torch.bmm(summary.unsqueeze(1), down).squeeze(1)
        output = torch.bmm(F.silu(latent).unsqueeze(1), up).squeeze(1)
        scale = self.residual_scale.index_select(0, selected).unsqueeze(-1)
        return torch.tanh(output) * scale

    def _dense_residual(self, summary: torch.Tensor) -> torch.Tensor:
        return self._all_residuals(summary).mean(dim=1)

    def _all_residuals(self, summary: torch.Tensor) -> torch.Tensor:
        latent = torch.einsum("bw,nwr->bnr", summary, self.down)
        outputs = torch.einsum("bnr,nrw->bnw", F.silu(latent), self.up)
        outputs = torch.tanh(outputs) * self.residual_scale.view(1, -1, 1)
        return outputs

    def forward(
        self,
        hidden: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None,
        *,
        collect_evidence: bool,
        collect_telemetry: bool,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        batch_size, time_steps, _ = hidden.shape
        fresh_state = state is None
        current = (
            self.initial_state(
                int(batch_size), device=hidden.device, dtype=hidden.dtype
            )
            if state is None
            else state
        )
        pending_sum = current["event_pending_sum"].to(hidden)
        pending_count = (
            0
            if fresh_state
            else int(current["event_pending_count"].detach().cpu().item())
        )
        current_residual = current["event_current_residual"].to(hidden)
        current_score = current["event_current_score"].to(hidden)
        current_active = current["event_current_active"].to(device=hidden.device)
        outputs: list[torch.Tensor] = []
        score_rows: list[torch.Tensor] = []
        active_rows: list[torch.Tensor] = []
        candidate_rows: list[torch.Tensor] = []
        router_rows: list[torch.Tensor] = []
        current_candidates = torch.zeros(
            int(batch_size), self.specialists, self.width,
            device=hidden.device, dtype=hidden.dtype,
        )
        current_router_scores = torch.zeros(
            int(batch_size), self.specialists,
            device=hidden.device, dtype=hidden.dtype,
        )
        selected_counts = torch.zeros(
            self.specialists, device=hidden.device, dtype=torch.long
        )
        completed_events = 0
        executed_specialists = 0
        start = 0
        while start < int(time_steps):
            take = min(self.interval - pending_count, int(time_steps) - start)
            end = start + take
            segment = hidden[:, start:end]
            outputs.append(segment + current_residual.unsqueeze(1))
            if collect_evidence:
                score_rows.append(current_score.unsqueeze(1).expand(-1, take))
                active_rows.append(current_active.unsqueeze(1).expand(-1, take))
                candidate_rows.append(
                    current_candidates.unsqueeze(1).expand(-1, take, -1, -1)
                )
                router_rows.append(
                    current_router_scores.unsqueeze(1).expand(-1, take, -1)
                )
            pending_sum = pending_sum + segment.sum(dim=1)
            pending_count += take
            if pending_count == self.interval:
                summary = pending_sum / float(self.interval)
                scores = self.router(summary)
                all_residuals = (
                    self._all_residuals(summary) if collect_evidence else None
                )
                if self.selection_mode == "dense":
                    current_residual = (
                        all_residuals.mean(dim=1)
                        if all_residuals is not None
                        else self._dense_residual(summary)
                    )
                    current_score = scores.mean(dim=-1)
                    current_active = torch.ones(
                        int(batch_size), device=hidden.device, dtype=torch.bool
                    )
                    selected_counts += int(batch_size)
                    executed_specialists += int(batch_size) * self.specialists
                else:
                    selected = self._select(scores)
                    current_residual = (
                        all_residuals.gather(
                            1,
                            selected.view(-1, 1, 1).expand(-1, 1, self.width),
                        ).squeeze(1)
                        if all_residuals is not None
                        else self._sparse_residual(summary, selected)
                    )
                    current_score = scores.gather(1, selected.unsqueeze(1)).squeeze(1)
                    current_active = torch.ones(
                        int(batch_size), device=hidden.device, dtype=torch.bool
                    )
                    selected_counts += torch.bincount(
                        selected, minlength=self.specialists
                    )
                    executed_specialists += int(batch_size)
                if collect_evidence:
                    assert all_residuals is not None
                    current_candidates = all_residuals
                    current_router_scores = scores
                completed_events += int(batch_size)
                pending_sum = torch.zeros_like(pending_sum)
                pending_count = 0
            start = end
        evidence = (
            {
                "predicted_utility": torch.cat(score_rows, dim=1),
                "residual_active": torch.cat(active_rows, dim=1),
                "candidate_residuals": torch.cat(candidate_rows, dim=1),
                "router_scores": torch.cat(router_rows, dim=1),
            }
            if collect_evidence
            else {}
        )
        next_state = {
            "event_pending_sum": pending_sum.detach(),
            "event_pending_count": torch.tensor(
                pending_count, device=hidden.device, dtype=torch.long
            ),
            "event_current_residual": current_residual.detach(),
            "event_current_score": current_score.detach(),
            "event_current_active": current_active.detach(),
        }
        possible = max(1, completed_events * self.specialists)
        telemetry = {
            "sidecar_surface": "marulho_sparse_event_memory_sidecar.v1",
            "selection_mode": self.selection_mode,
            "event_interval": self.interval,
            "specialist_count": self.specialists,
            "specialist_rank": self.rank,
            "completed_event_batch_count": completed_events,
            "executed_specialist_count": executed_specialists,
            "possible_specialist_count": possible,
            "active_compute_fraction": executed_specialists / possible,
            "selected_counts": (
                selected_counts.detach().cpu().tolist()
                if collect_telemetry
                else None
            ),
            "actual_sparse_execution": self.selection_mode != "dense",
        }
        return torch.cat(outputs, dim=1), next_state, evidence, telemetry


class MarulhoSparseEventLanguageModel(MarulhoLanguageModel):
    """Transformer language model whose exact stream is untouched by the sidecar."""

    surface = "marulho_sparse_event_language_model.v1"

    def __init__(self, config: SparseEventMemoryConfig) -> None:
        _validate_sparse_config(config)
        super().__init__(config)
        self.config = config
        self.event_memory = SparseEventMemorySidecar(config)

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        collect_sidecar_evidence: bool = False,
    ) -> dict[str, Any]:
        exact = super()._forward_hidden(
            input_ids, state, collect_telemetry=collect_telemetry
        )
        augmented, memory_state, evidence, memory_telemetry = self.event_memory(
            exact["hidden"],
            state,
            collect_evidence=collect_sidecar_evidence,
            collect_telemetry=collect_telemetry,
        )
        return {
            "hidden": augmented,
            "exact_hidden": exact["hidden"],
            "state": {**exact["state"], **memory_state},
            "sidecar_evidence": evidence,
            "telemetry": {
                **exact["telemetry"],
                **memory_telemetry,
                "active_language_path": self.config.active_language_path,
            },
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        if input_ids.ndim == 1:
            step_ids = input_ids.unsqueeze(1)
        elif input_ids.ndim == 2 and int(input_ids.shape[1]) == 1:
            step_ids = input_ids
        else:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        return self.forward(
            step_ids,
            state,
            collect_telemetry=collect_telemetry,
            decode_vocab_only=decode_vocab_only,
        )

    def next_token_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        *,
        collect_telemetry: bool = True,
        return_evidence: bool = True,
        counterfactual_probe: bool | None = None,
    ) -> dict[str, Any]:
        targets = target_ids.to(device=self.device, dtype=torch.long)
        eligible = bool(
            self.training
            and float(self.config.utility_loss_weight) > 0.0
            and int(targets.shape[1]) > self.config.event_interval
            and self.config.selection_mode == "utility"
        )
        should_probe = bool(
            eligible
            and (
                bool(counterfactual_probe)
                if counterfactual_probe is not None
                else (
                    float(self.config.counterfactual_rate) > 0.0
                    and float(torch.rand(()).item())
                    < float(self.config.counterfactual_rate)
                )
            )
        )
        result = self._forward_hidden(
            input_ids,
            collect_telemetry=collect_telemetry,
            collect_sidecar_evidence=should_probe,
        )
        logits = self.lm_head(result["hidden"])
        language_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
        )
        utility_loss = language_loss.new_zeros(())
        mean_target = 0.0
        mean_advantage_spread = 0.0
        if should_probe:
            exact_logits = self.lm_head(result["exact_hidden"])
            exact_losses = F.cross_entropy(
                exact_logits.reshape(-1, exact_logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).view_as(targets)
            evidence = result["sidecar_evidence"]
            active = evidence["residual_active"]
            candidate_targets: list[torch.Tensor] = []
            candidates = evidence["candidate_residuals"]
            for specialist_index in range(int(self.config.specialist_count)):
                candidate_logits = self.lm_head(
                    result["exact_hidden"] + candidates[:, :, specialist_index]
                )
                candidate_losses = F.cross_entropy(
                    candidate_logits.reshape(-1, candidate_logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                ).view_as(targets)
                candidate_targets.append(
                    exact_losses
                    - candidate_losses
                    - float(self.config.compute_cost)
                )
            utility_target = torch.stack(candidate_targets, dim=-1).detach()
            if bool(active.any()):
                predicted = evidence["router_scores"][active]
                target = utility_target[active]
                centered_prediction = predicted - predicted.mean(
                    dim=-1, keepdim=True
                )
                centered_target = target - target.mean(dim=-1, keepdim=True)
                utility_loss = F.mse_loss(centered_prediction, centered_target)
                mean_target = float(target.mean().detach().cpu())
                mean_advantage_spread = float(
                    (target.max(dim=-1).values - target.min(dim=-1).values)
                    .mean()
                    .detach()
                    .cpu()
                )
        loss = language_loss + float(self.config.utility_loss_weight) * utility_loss
        counterfactual = {
            "ran": should_probe,
            "mean_target": mean_target,
            "mean_advantage_spread": mean_advantage_spread,
            "compute_cost": float(self.config.compute_cost),
        }
        loss_evidence = {
            "surface": "marulho_sparse_event_loss.v2",
            "language_loss": float(language_loss.detach().cpu()),
            "utility_loss": float(utility_loss.detach().cpu()),
            "counterfactual": counterfactual,
            "review_oracle_used_for_prediction": False,
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy_plus_sparse_event_utility",
            "loss_evidence": loss_evidence if return_evidence else {},
            "training_aux": {"counterfactual": counterfactual},
            "state": result["state"],
            "telemetry": result["telemetry"],
        }
