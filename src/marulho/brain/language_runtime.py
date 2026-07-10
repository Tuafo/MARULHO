"""Brain-owned adapter for the active MARULHO Transformer language model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import (
    LanguageTokenizer,
    load_language_tokenizer_state,
)
from marulho.evaluation.language_sustained_runtime_evidence import (
    run_language_sustained_runtime_evidence,
)
from marulho.training.language_model import (
    LanguageModelConfig,
    MarulhoLanguageModel,
    load_language_model_state,
)


class BrainLanguageModelRuntime:
    """Own one Transformer checkpoint inside the brain lifecycle."""

    surface = "marulho_brain_transformer_runtime.v2"

    def __init__(
        self,
        model: MarulhoLanguageModel,
        tokenizer: LanguageTokenizer,
        *,
        evaluation_report: Mapping[str, Any] | None = None,
        checkpoint_installation_reports: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        if int(model.config.vocab_size) != int(tokenizer.vocab_size):
            raise ValueError("Brain language model and tokenizer vocabularies must match")
        self.model = model
        self.tokenizer = tokenizer
        self.evaluation_report = dict(evaluation_report or {})
        self.checkpoint_installation_reports = [
            dict(report) for report in checkpoint_installation_reports
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
    def generate(
        self,
        prompt: str | None,
        *,
        max_tokens: int,
        generation_repetition_penalty: float = 1.1,
        generation_no_repeat_ngram_size: int = 3,
    ) -> dict[str, Any]:
        prompt_text = str(prompt or "")
        prompt_ids = torch.tensor(
            self.tokenizer.encode(prompt_text, add_bos=True, add_eos=False),
            dtype=torch.long,
            device=self.device,
        )
        result = self.model.generate(
            prompt_ids,
            max_new_tokens=max(0, int(max_tokens)),
            eos_id=self.tokenizer.eos_id,
            repetition_penalty=max(1.0, float(generation_repetition_penalty)),
            no_repeat_ngram_size=max(0, int(generation_no_repeat_ngram_size)),
        )
        ids = [int(value) for value in result["generated_ids"][0].cpu().tolist()]
        prompt_count = int(prompt_ids.numel())
        continuation = ids[prompt_count:]
        return {
            "surface": "marulho_brain_transformer_generation.v2",
            "language_model_surface": result["surface"],
            "text": self.tokenizer.decode(ids),
            "continuation_text": self.tokenizer.decode(continuation),
            "available": bool(ids),
            "prompt": prompt_text,
            "prompt_token_count": prompt_count,
            "generated_token_count": len(ids),
            "emitted_tokens": int(result["new_token_count"]),
            "generated_token_ids": ids,
            "continuation_token_ids": continuation,
            "generation_decode": result["generation_decode"],
            "active_language_path": self.active_language_path,
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "tokenizer_hash": self.tokenizer.vocabulary_hash(),
            "vocab_size": int(self.tokenizer.vocab_size),
            "model_vocab_size": int(self.model.config.vocab_size),
            "state_core": "transformer",
        }

    def generate_sustained(
        self,
        *,
        output_path: str | Path,
        target_tokens: int,
        checkpoint_path: str | Path | None = None,
        prompt: str = "MARULHO",
        timeout_seconds: float = 600.0,
        generation_repetition_penalty: float = 1.1,
        generation_no_repeat_ngram_size: int = 3,
    ) -> dict[str, Any]:
        return run_language_sustained_runtime_evidence(
            self.model,
            self.tokenizer,
            output_path=output_path,
            target_tokens=max(0, int(target_tokens)),
            checkpoint_path=checkpoint_path,
            prompt=prompt,
            timeout_seconds=float(timeout_seconds),
            generation_repetition_penalty=float(generation_repetition_penalty),
            generation_no_repeat_ngram_size=int(generation_no_repeat_ngram_size),
        )

    def record_checkpoint_installation(self, report: Mapping[str, Any]) -> None:
        self.checkpoint_installation_reports.append(dict(report))
        self.checkpoint_installation_reports = self.checkpoint_installation_reports[-16:]

    def summary(self) -> dict[str, Any]:
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        return {
            "surface": self.surface,
            "installed": True,
            "active_language_path": self.active_language_path,
            "state_core": "transformer",
            "model_config": asdict(self.model.config),
            "parameter_count": parameters,
            "tokenizer_surface": self.tokenizer.state_dict().get("surface"),
            "tokenizer_hash": self.tokenizer.vocabulary_hash(),
            "vocab_size": int(self.tokenizer.vocab_size),
            "evaluation_report": dict(self.evaluation_report),
            "checkpoint_installation_reports": list(
                self.checkpoint_installation_reports
            ),
            "continual_learning_enabled": False,
            "structural_plasticity_enabled": False,
            "owned_by_marulho": True,
            "external_llm_used": False,
            "loads_external_checkpoint": False,
            "device": str(self.device),
        }

    @classmethod
    def empty_summary(cls) -> dict[str, Any]:
        return {
            "surface": cls.surface,
            "installed": False,
            "active_language_path": None,
            "state_core": None,
            "continual_learning_enabled": False,
            "structural_plasticity_enabled": False,
            "owned_by_marulho": True,
            "external_llm_used": False,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "config": asdict(self.model.config),
            "model_state": {
                key: value.detach().cpu() for key, value in self.model.state_dict().items()
            },
            "tokenizer": self.tokenizer.state_dict(),
            "evaluation_report": dict(self.evaluation_report),
            "checkpoint_installation_reports": list(
                self.checkpoint_installation_reports
            ),
            "owned_by_marulho": True,
            "external_llm_used": False,
        }

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        device: torch.device | str = "cpu",
    ) -> "BrainLanguageModelRuntime":
        if state.get("surface") != cls.surface:
            raise ValueError("Rejected legacy brain language runtime state")
        tokenizer = load_language_tokenizer_state(state["tokenizer"])
        model = MarulhoLanguageModel(LanguageModelConfig(**dict(state["config"])))
        load_language_model_state(model, state["model_state"])
        runtime = cls(
            model,
            tokenizer,
            evaluation_report=state.get("evaluation_report"),
            checkpoint_installation_reports=state.get(
                "checkpoint_installation_reports",
                (),
            ),
        )
        runtime.to_device(device)
        return runtime
