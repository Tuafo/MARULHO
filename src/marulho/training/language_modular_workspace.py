"""Depth-preserving modular latent workspace language candidate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel
from marulho.training.language_transformer import (
    MarulhoCausalTransformerStateBlock,
    TransformerRMSNorm,
)


@dataclass(frozen=True)
class ModularWorkspaceConfig:
    vocab_size: int
    shared_width: int = 368
    shared_layers_per_stage: int = 2
    shared_attention_heads: int = 8
    cell_count: int = 4
    cell_width: int = 256
    cell_layers_per_stage: int = 1
    cell_attention_heads: int = 8
    workspace_width: int = 64
    context_length: int = 72
    mlp_ratio: float = 4.0
    mode: str = "real"
    tie_embeddings: bool = True
    active_language_path: str = "marulho_depth_preserving_modular_workspace_v4"


def _validate_workspace_config(config: ModularWorkspaceConfig) -> None:
    if int(config.vocab_size) <= 1:
        raise ValueError("vocab_size must exceed one")
    if int(config.cell_count) < 2:
        raise ValueError("cell_count must be at least two")
    if int(config.context_length) < 2:
        raise ValueError("context_length must be at least two")
    if int(config.shared_layers_per_stage) < 1:
        raise ValueError("shared_layers_per_stage must be positive")
    if int(config.cell_layers_per_stage) < 1:
        raise ValueError("cell_layers_per_stage must be positive")
    if int(config.workspace_width) < 1:
        raise ValueError("workspace_width must be positive")
    for width, heads, label in (
        (
            int(config.shared_width),
            int(config.shared_attention_heads),
            "shared",
        ),
        (int(config.cell_width), int(config.cell_attention_heads), "cell"),
    ):
        if width <= 0 or heads <= 0 or width % heads != 0:
            raise ValueError(f"{label} width must be divisible by its head count")
        if (width // heads) % 2 != 0:
            raise ValueError(f"{label} attention head dimension must be even")
    if str(config.mode) not in {"no_exchange", "shuffled", "real"}:
        raise ValueError("mode must be no_exchange, shuffled, or real")


def _substate(
    state: Mapping[str, torch.Tensor] | None,
    prefix: str,
) -> dict[str, torch.Tensor] | None:
    if state is None:
        return None
    selected = {
        key[len(prefix) :]: value
        for key, value in state.items()
        if key.startswith(prefix)
    }
    return selected or None


def _store_state(
    target: dict[str, torch.Tensor],
    prefix: str,
    state: Mapping[str, torch.Tensor],
) -> None:
    for key, value in state.items():
        target[f"{prefix}{key}"] = value.detach()


class _WorkspaceCell(nn.Module):
    def __init__(self, config: ModularWorkspaceConfig) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            config.shared_width,
            config.cell_width,
            bias=False,
        )
        self.input_norm = TransformerRMSNorm(config.cell_width)
        self.before_exchange = MarulhoCausalTransformerStateBlock(
            config.cell_width,
            config.cell_width,
            state_layers=config.cell_layers_per_stage,
            attention_heads=config.cell_attention_heads,
            context_length=config.context_length,
            mlp_ratio=config.mlp_ratio,
            dropout=0.0,
        )
        self.message_out = nn.Linear(
            config.cell_width,
            config.workspace_width,
            bias=False,
        )
        self.message_in = nn.Linear(
            config.workspace_width,
            config.cell_width,
            bias=False,
        )
        self.after_exchange = MarulhoCausalTransformerStateBlock(
            config.cell_width,
            config.cell_width,
            state_layers=config.cell_layers_per_stage,
            attention_heads=config.cell_attention_heads,
            context_length=config.context_length,
            mlp_ratio=config.mlp_ratio,
            dropout=0.0,
        )


class MarulhoModularWorkspaceLanguageModel(MarulhoLanguageModel):
    """One language interface with parallel internal cells and a narrow workspace."""

    surface = "marulho_modular_workspace_language_model.v1"

    def __init__(self, workspace_config: ModularWorkspaceConfig) -> None:
        nn.Module.__init__(self)
        _validate_workspace_config(workspace_config)
        self.workspace_config = workspace_config
        path_layers = (
            (2 * int(workspace_config.shared_layers_per_stage))
            + (2 * int(workspace_config.cell_layers_per_stage))
        )
        self.config = LanguageModelConfig(
            vocab_size=int(workspace_config.vocab_size),
            embedding_dim=int(workspace_config.shared_width),
            state_dim=int(workspace_config.shared_width),
            state_layers=path_layers,
            attention_heads=int(workspace_config.shared_attention_heads),
            transformer_context_length=int(workspace_config.context_length),
            transformer_mlp_ratio=float(workspace_config.mlp_ratio),
            tie_embeddings=bool(workspace_config.tie_embeddings),
            active_language_path=str(workspace_config.active_language_path),
        )
        self.token_embedding = nn.Embedding(
            workspace_config.vocab_size,
            workspace_config.shared_width,
        )
        self.shared_before = MarulhoCausalTransformerStateBlock(
            workspace_config.shared_width,
            workspace_config.shared_width,
            state_layers=workspace_config.shared_layers_per_stage,
            attention_heads=workspace_config.shared_attention_heads,
            context_length=workspace_config.context_length,
            mlp_ratio=workspace_config.mlp_ratio,
            dropout=0.0,
        )
        self.cells = nn.ModuleList(
            _WorkspaceCell(workspace_config)
            for _ in range(workspace_config.cell_count)
        )
        combined_cell_width = workspace_config.cell_count * workspace_config.cell_width
        self.cell_gate = nn.Linear(
            combined_cell_width,
            workspace_config.cell_count,
            bias=False,
        )
        self.merge = nn.Linear(
            combined_cell_width,
            workspace_config.shared_width,
            bias=False,
        )
        self.merge_norm = TransformerRMSNorm(workspace_config.shared_width)
        self.shared_after = MarulhoCausalTransformerStateBlock(
            workspace_config.shared_width,
            workspace_config.shared_width,
            state_layers=workspace_config.shared_layers_per_stage,
            attention_heads=workspace_config.shared_attention_heads,
            context_length=workspace_config.context_length,
            mlp_ratio=workspace_config.mlp_ratio,
            dropout=0.0,
        )
        self.lm_head = nn.Linear(
            workspace_config.shared_width,
            workspace_config.vocab_size,
            bias=False,
        )
        if bool(workspace_config.tie_embeddings):
            self.lm_head.weight = self.token_embedding.weight
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _workspace_inputs(
        self,
        messages: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        mode = self.workspace_config.mode
        if mode == "shuffled":
            usable = (
                [torch.roll(message, shifts=1, dims=0) for message in messages]
                if int(messages[0].shape[0]) > 1
                else [message * 0.0 for message in messages]
            )
        else:
            usable = messages
        total = torch.stack(usable, dim=0).sum(dim=0)
        incoming = [
            (total - usable[index]) / float(self.workspace_config.cell_count - 1)
            for index in range(self.workspace_config.cell_count)
        ]
        if mode == "no_exchange":
            incoming = [message * 0.0 for message in incoming]
        return incoming

    def _forward_hidden(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
    ) -> dict[str, Any]:
        if input_ids.ndim != 2:
            raise ValueError("Modular workspace expects input_ids shaped [batch, time]")
        ids = input_ids.to(device=self.device, dtype=torch.long)
        if int(ids.shape[1]) < 1:
            raise ValueError("Modular workspace expects at least one token")
        shared, before_state, _ = self.shared_before(
            self.token_embedding(ids),
            _substate(state, "shared_before_"),
            collect_telemetry=False,
        )
        before_rows: list[torch.Tensor] = []
        before_states: list[dict[str, torch.Tensor]] = []
        messages: list[torch.Tensor] = []
        for index, cell in enumerate(self.cells):
            projected = cell.input_norm(cell.input_projection(shared))
            hidden, next_state, _ = cell.before_exchange(
                projected,
                _substate(state, f"cell_{index}_before_"),
                collect_telemetry=False,
            )
            before_rows.append(hidden)
            before_states.append(next_state)
            messages.append(cell.message_out(hidden))
        incoming = self._workspace_inputs(messages)
        after_rows: list[torch.Tensor] = []
        after_states: list[dict[str, torch.Tensor]] = []
        for index, cell in enumerate(self.cells):
            communicated = before_rows[index] + cell.message_in(incoming[index])
            hidden, next_state, _ = cell.after_exchange(
                communicated,
                _substate(state, f"cell_{index}_after_"),
                collect_telemetry=False,
            )
            after_rows.append(hidden)
            after_states.append(next_state)
        concatenated = torch.cat(after_rows, dim=-1)
        gate = torch.softmax(self.cell_gate(concatenated), dim=-1)
        weighted = torch.cat(
            [
                hidden * gate[..., index : index + 1]
                for index, hidden in enumerate(after_rows)
            ],
            dim=-1,
        )
        merged = self.merge_norm(shared + self.merge(weighted))
        hidden, after_state, transformer = self.shared_after(
            merged,
            _substate(state, "shared_after_"),
            collect_telemetry=collect_telemetry,
        )
        next_state: dict[str, torch.Tensor] = {
            "position": after_state["position"].detach(),
        }
        _store_state(next_state, "shared_before_", before_state)
        for index in range(self.workspace_config.cell_count):
            _store_state(
                next_state,
                f"cell_{index}_before_",
                before_states[index],
            )
            _store_state(
                next_state,
                f"cell_{index}_after_",
                after_states[index],
            )
        _store_state(next_state, "shared_after_", after_state)
        telemetry = {
            **transformer,
            "surface": self.surface,
            "active_language_path": self.workspace_config.active_language_path,
            "owned_by_marulho": True,
            "external_llm_used": False,
            "effective_path_layers": int(self.config.state_layers),
            "shared_width": int(self.workspace_config.shared_width),
            "cell_count": int(self.workspace_config.cell_count),
            "cell_width": int(self.workspace_config.cell_width),
            "workspace_width": int(self.workspace_config.workspace_width),
            "workspace_mode": str(self.workspace_config.mode),
            "workspace_exchange_active": self.workspace_config.mode
            in {"shuffled", "real"},
            "full_context_gradient_path": True,
            "mean_cell_gate_entropy": (
                float(
                    (-(gate.float() * gate.float().clamp_min(1.0e-9).log()).sum(-1))
                    .mean()
                    .detach()
                    .cpu()
                )
                if collect_telemetry
                else None
            ),
        }
        return {
            "hidden": hidden,
            "state": next_state,
            "telemetry": telemetry,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        del decode_vocab_only
        result = self._forward_hidden(
            input_ids,
            state,
            collect_telemetry=collect_telemetry,
        )
        return {
            "logits": self.lm_head(result["hidden"]),
            "state": result["state"],
            "telemetry": result["telemetry"],
        }

    def forward_step(
        self,
        input_ids: torch.Tensor,
        state: Mapping[str, torch.Tensor] | None = None,
        *,
        collect_telemetry: bool = True,
        decode_vocab_only: bool = False,
    ) -> dict[str, Any]:
        ids = input_ids.unsqueeze(1) if input_ids.ndim == 1 else input_ids
        if ids.ndim != 2 or int(ids.shape[1]) != 1:
            raise ValueError("forward_step expects [batch] or [batch, 1] token ids")
        return self.forward(
            ids,
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
    ) -> dict[str, Any]:
        output = self.forward(input_ids, collect_telemetry=collect_telemetry)
        targets = target_ids.to(device=self.device, dtype=torch.long)
        logits = output["logits"]
        if logits.shape[:2] != targets.shape:
            raise ValueError("target_ids must match input batch/time dimensions")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
        )
        evidence = {
            "surface": "marulho_modular_workspace_cross_entropy.v1",
            "sampled_vocab_training": False,
            "full_vocab_logits_materialized": True,
            "target_token_count": int(targets.numel()),
            "external_llm_used": False,
        }
        return {
            "loss": loss,
            "loss_kind": "full_vocab_cross_entropy",
            "loss_evidence": evidence if return_evidence else {},
            "state": output["state"],
            "telemetry": output["telemetry"],
        }

    def generation_decode_policy(
        self,
        *,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, Any]:
        policy = super().generation_decode_policy(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )
        return {
            **policy,
            "surface": "marulho_modular_workspace_decode_policy.v1",
            "kv_cache": "bounded_shared_and_per_cell_layers",
        }
