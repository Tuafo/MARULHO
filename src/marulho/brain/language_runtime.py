from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import torch

from marulho.data.language_tokenizer import ByteLevelLanguageTokenizer
from marulho.training.language_model import LanguageModelConfig, MarulhoLanguageModel


class BrainLanguageModelRuntime:
    """Brain-owned adapter for the training-owned MARULHO LM head."""

    surface = "marulho_brain_language_model_runtime.v1"

    def __init__(
        self,
        model: MarulhoLanguageModel,
        tokenizer: ByteLevelLanguageTokenizer,
        *,
        evaluation_report: Mapping[str, Any] | None = None,
    ) -> None:
        if int(model.config.vocab_size) != int(tokenizer.vocab_size):
            raise ValueError("Language model vocab size must match tokenizer state")
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_report = dict(evaluation_report or {})
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
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
        }

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
        )
        runtime.to_device(device)
        return runtime
