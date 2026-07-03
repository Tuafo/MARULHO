from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_checkpoint_evolution import (
    LanguageCheckpointEvolutionConfig,
    run_language_checkpoint_evolution,
)
from marulho.training.language_continual_learning import (
    LanguageContinualLearningConfig,
    run_language_continual_learning_window,
)
from marulho.training.language_model import (
    LanguageBatch,
    LanguageModelConfig,
    MarulhoLanguageModel,
)
from marulho.training.language_structural_plasticity import (
    LanguageStructuralPlasticityConfig,
    apply_language_structural_plasticity_transaction,
    build_language_structural_prune_proposal,
    build_language_structural_plasticity_proposal,
)


class BrainLanguageModelRuntime:
    """Brain-owned adapter for the training-owned MARULHO LM head."""

    surface = "marulho_brain_language_model_runtime.v1"

    def __init__(
        self,
        model: MarulhoLanguageModel,
        tokenizer: ByteLevelLanguageTokenizer,
        *,
        evaluation_report: Mapping[str, Any] | None = None,
        learning_reports: Sequence[Mapping[str, Any]] = (),
        structural_reports: Sequence[Mapping[str, Any]] = (),
        checkpoint_evolution_reports: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if int(model.config.vocab_size) != int(tokenizer.vocab_size):
            raise ValueError("Language model vocab size must match tokenizer state")
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_report = dict(evaluation_report or {})
        self.learning_reports = [dict(report) for report in learning_reports][-16:]
        self.structural_reports = [dict(report) for report in structural_reports][-16:]
        self.checkpoint_evolution_reports = [
            dict(report) for report in checkpoint_evolution_reports
        ][-16:]
        self.model.eval()

    @property
    def active_language_path(self) -> str:
        return str(self.model.config.active_language_path)

    @property
    def device(self) -> torch.device:
        return self.model.device

    def to_device(self, device: torch.device | str) -> None:
        self.model.to(torch.device(device))
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompt: str | None, *, max_tokens: int) -> dict[str, Any]:
        prompt_text = str(prompt or "")
        limit = max(0, int(max_tokens))
        prompt_ids = torch.tensor(
            self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False),
            dtype=torch.long,
            device=self.device,
        )
        model_generation = self.model.generate(
            prompt_ids,
            max_new_tokens=limit,
            eos_id=self.tokenizer.eos_id,
        )
        generated_ids = [
            int(token_id)
            for token_id in model_generation["generated_ids"][0].detach().cpu().tolist()
        ]
        prompt_token_count = int(prompt_ids.numel())
        continuation_ids = generated_ids[prompt_token_count:]
        text = self.tokenizer.decode(generated_ids)
        continuation_text = self.tokenizer.decode(continuation_ids)
        return {
            "surface": "marulho_brain_language_model_generation.v1",
            "language_model_surface": model_generation["surface"],
            "text": text,
            "continuation_text": continuation_text,
            "available": bool(generated_ids),
            "prompt": prompt,
            "prompt_token_count": prompt_token_count,
            "generated_token_count": len(generated_ids),
            "emitted_tokens": int(model_generation["new_token_count"]),
            "max_tokens": limit,
            "generated_token_ids": generated_ids,
            "continuation_token_ids": continuation_ids,
            "active_language_path": self.active_language_path,
            "transition_readout_fallback_used": False,
            "fallback_language_path": "local_transition_readout",
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "thought_loop_used": False,
            "cortex_used": False,
            "loads_external_checkpoint": False,
            "checkpointed_language_components": True,
            "tokenizer_hash": self.tokenizer.vocabulary_hash(),
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device),
            "heldout_loss": self.evaluation_report.get("heldout_loss"),
            "heldout_perplexity": self.evaluation_report.get("heldout_perplexity"),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "available": True,
            "active_language_path": self.active_language_path,
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "checkpointed_language_components": True,
            "tokenizer_hash": self.tokenizer.vocabulary_hash(),
            "vocab_size": self.tokenizer.vocab_size,
            "device": str(self.device),
            "heldout_evaluation_available": bool(self.evaluation_report),
            "heldout_loss": self.evaluation_report.get("heldout_loss"),
            "heldout_perplexity": self.evaluation_report.get("heldout_perplexity"),
            "continual_learning_window_count": len(self.learning_reports),
            "last_continual_learning": (
                dict(self.learning_reports[-1]) if self.learning_reports else None
            ),
            "structural_transaction_count": len(self.structural_reports),
            "last_structural_transaction": (
                dict(self.structural_reports[-1]) if self.structural_reports else None
            ),
            "checkpoint_evolution_count": len(self.checkpoint_evolution_reports),
            "last_checkpoint_evolution": (
                dict(self.checkpoint_evolution_reports[-1])
                if self.checkpoint_evolution_reports
                else None
            ),
        }

    @classmethod
    def empty_summary(cls) -> dict[str, Any]:
        return {
            "surface": cls.surface,
            "available": False,
            "active_language_path": "local_transition_readout",
            "owned_by_marulho": True,
            "external_dependency": False,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "checkpointed_language_components": False,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "active_language_path": self.active_language_path,
            "tokenizer": self.tokenizer.state_dict(),
            "tokenizer_hash": self.tokenizer.vocabulary_hash(),
            "config": asdict(self.model.config),
            "model_state": {
                key: value.detach().cpu()
                for key, value in self.model.state_dict().items()
            },
            "evaluation_report": dict(self.evaluation_report),
            "learning_reports": [dict(report) for report in self.learning_reports],
            "structural_reports": [dict(report) for report in self.structural_reports],
            "checkpoint_evolution_reports": [
                dict(report) for report in self.checkpoint_evolution_reports
            ],
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
        }

    def learn_continual_window(
        self,
        *,
        new_batches: Sequence[LanguageBatch],
        old_eval_batches: Sequence[LanguageBatch],
        new_eval_batches: Sequence[LanguageBatch],
        replay_batches: Sequence[LanguageBatch] = (),
        config: LanguageContinualLearningConfig | None = None,
    ) -> dict[str, Any]:
        report = run_language_continual_learning_window(
            self.model,
            new_batches=new_batches,
            old_eval_batches=old_eval_batches,
            new_eval_batches=new_eval_batches,
            replay_batches=replay_batches,
            config=config,
        )
        self.evaluation_report = dict(report["new_domain_after"])
        self.learning_reports.append(dict(report))
        self.learning_reports = self.learning_reports[-16:]
        return report

    def propose_structural_plasticity(
        self,
        *,
        routing_evidence: Mapping[str, Any],
        learning_evidence: Mapping[str, Any] | None = None,
        config: LanguageStructuralPlasticityConfig | None = None,
        mutation_kind: str = "growth",
    ) -> dict[str, Any]:
        if str(mutation_kind) == "prune":
            return build_language_structural_prune_proposal(
                self.model,
                routing_evidence=routing_evidence,
                learning_evidence=learning_evidence,
                config=config,
            )
        return build_language_structural_plasticity_proposal(
            self.model,
            routing_evidence=routing_evidence,
            learning_evidence=learning_evidence,
            config=config,
        )

    def apply_structural_plasticity(
        self,
        proposal: Mapping[str, Any],
        *,
        eval_batches: Sequence[LanguageBatch],
        checkpoint_path: str,
        operator_approved: bool,
        config: LanguageStructuralPlasticityConfig | None = None,
    ) -> dict[str, Any]:
        candidate, report = apply_language_structural_plasticity_transaction(
            self.model,
            proposal,
            eval_batches=eval_batches,
            checkpoint_path=checkpoint_path,
            operator_approved=operator_approved,
            config=config,
        )
        if bool(report.get("applied")):
            self.model = candidate.to(self.device)
            self.model.eval()
            self.evaluation_report = dict(report["evaluation"]["candidate"])
        self.structural_reports.append(dict(report))
        self.structural_reports = self.structural_reports[-16:]
        return report

    def evolve_checkpoint(
        self,
        *,
        eval_batches: Sequence[LanguageBatch],
        child_train_batches: Sequence[LanguageBatch],
        child_new_eval_batches: Sequence[LanguageBatch],
        checkpoint_dir: str,
        replay_batches: Sequence[LanguageBatch] = (),
        config: LanguageCheckpointEvolutionConfig | None = None,
        learning_config: LanguageContinualLearningConfig | None = None,
        structural_config: LanguageStructuralPlasticityConfig | None = None,
    ) -> dict[str, Any]:
        _child, report = run_language_checkpoint_evolution(
            self.model,
            self.tokenizer,
            eval_batches=eval_batches,
            child_train_batches=child_train_batches,
            child_new_eval_batches=child_new_eval_batches,
            replay_batches=replay_batches,
            checkpoint_dir=checkpoint_dir,
            config=config,
            learning_config=learning_config,
            structural_config=structural_config,
        )
        self.checkpoint_evolution_reports.append(dict(report))
        self.checkpoint_evolution_reports = self.checkpoint_evolution_reports[-16:]
        return report

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        device: torch.device | str,
    ) -> "BrainLanguageModelRuntime":
        tokenizer = ByteLevelLanguageTokenizer.load_state_dict(state["tokenizer"])
        config = LanguageModelConfig(**dict(state["config"]))
        model = MarulhoLanguageModel(config)
        model.load_state_dict(state["model_state"])
        runtime = cls(
            model,
            tokenizer,
            evaluation_report=(
                state.get("evaluation_report")
                if isinstance(state.get("evaluation_report"), Mapping)
                else None
            ),
            learning_reports=[
                item
                for item in list(state.get("learning_reports") or [])
                if isinstance(item, Mapping)
            ],
            structural_reports=[
                item
                for item in list(state.get("structural_reports") or [])
                if isinstance(item, Mapping)
            ],
            checkpoint_evolution_reports=[
                item
                for item in list(state.get("checkpoint_evolution_reports") or [])
                if isinstance(item, Mapping)
            ],
        )
        runtime.to_device(device)
        return runtime
