"""Source-absent semantic generation audit for a MARULHO organism checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Sequence

import torch

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_organism import (
    load_distributed_language_checkpoint,
)


SURFACE = "marulho_organism_generation_audit.v1"
ARTIFACT_KIND = "marulho_organism_generation_audit"


@dataclass(frozen=True)
class GenerationAuditPrompt:
    prompt_id: str
    capability: str
    text: str
    review_question: str


DEFAULT_PROMPTS = (
    GenerationAuditPrompt(
        prompt_id="controlled_science",
        capability="causal_explanation",
        text="A careful scientist changes one variable at a time because",
        review_question=(
            "Does the continuation explain causal isolation or experimental control?"
        ),
    ),
    GenerationAuditPrompt(
        prompt_id="quiet_island",
        capability="narrative_continuation",
        text="When the red boat reached the quiet island,",
        review_question=(
            "Does it continue a stable scene with compatible entities and events?"
        ),
    ),
    GenerationAuditPrompt(
        prompt_id="memory_understanding",
        capability="abstraction",
        text="The difference between remembering and understanding is",
        review_question=(
            "Does it express a coherent distinction rather than repeat the prompt?"
        ),
    ),
    GenerationAuditPrompt(
        prompt_id="conflict_replacement",
        capability="state_update",
        text=(
            "Mira placed the silver key inside the wooden drawer. Later she moved "
            "it into the glass jar. The silver key is now"
        ),
        review_question=(
            "Does it preserve the silver key and answer that its current location is "
            "the glass jar, without reverting to the drawer?"
        ),
    ),
    GenerationAuditPrompt(
        prompt_id="shattered_glass",
        capability="physical_causality",
        text=(
            "A drinking glass fell from a high table onto a stone floor. It shattered "
            "because"
        ),
        review_question=(
            "Does it give a physically compatible explanation involving impact, "
            "height, hardness, or brittleness?"
        ),
    ),
    GenerationAuditPrompt(
        prompt_id="battery_test",
        capability="procedural_planning",
        text=(
            "Lena wants to learn whether a flashlight battery is dead without "
            "damaging the flashlight. First, she should"
        ),
        review_question=(
            "Does it propose a safe, relevant diagnostic sequence and maintain the "
            "goal across multiple sentences?"
        ),
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prompt_source_matches(
    paths: Sequence[Path], prompts: Sequence[GenerationAuditPrompt]
) -> dict[str, list[str]]:
    normalized = {prompt.prompt_id: prompt.text.casefold() for prompt in prompts}
    longest = max(len(text) for text in normalized.values())
    matches = {prompt.prompt_id: [] for prompt in prompts}
    for path in paths:
        remaining = set(normalized)
        carry = ""
        with path.open("r", encoding="utf-8") as handle:
            while remaining:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                searchable = (carry + chunk).casefold()
                found = {
                    prompt_id
                    for prompt_id in remaining
                    if normalized[prompt_id] in searchable
                }
                for prompt_id in found:
                    matches[prompt_id].append(str(path))
                remaining.difference_update(found)
                carry = (carry + chunk)[-max(0, longest - 1) :]
    return matches


def _text_metrics(text: str) -> dict[str, Any]:
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    bigrams = list(zip(words, words[1:]))
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_terminator_count": sum(text.count(mark) for mark in ".!?"),
        "printable_fraction": (
            sum(character.isprintable() or character in "\n\t" for character in text)
            / len(text)
            if text
            else 0.0
        ),
        "replacement_character_count": text.count("�"),
        "distinct_bigram_fraction": (
            len(set(bigrams)) / len(bigrams) if bigrams else 0.0
        ),
    }


def run_organism_generation_audit(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    prompts: Sequence[GenerationAuditPrompt] = DEFAULT_PROMPTS,
    source_paths: Sequence[str | Path] = (),
    max_new_tokens: int = 96,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    prompt_rows = tuple(prompts)
    if not prompt_rows:
        raise ValueError("At least one unseen prompt is required")
    if len({prompt.prompt_id for prompt in prompt_rows}) != len(prompt_rows):
        raise ValueError("Generation audit prompt IDs must be unique")
    sources = tuple(Path(path) for path in source_paths)
    source_matches = _prompt_source_matches(sources, prompt_rows) if sources else {
        prompt.prompt_id: [] for prompt in prompt_rows
    }
    model, tokenizer, metadata, runtime_state = load_distributed_language_checkpoint(
        checkpoint
    )
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    model = model.to(resolved_device).eval()
    modes = (
        ("greedy", {"temperature": 0.0, "top_p": 1.0, "seed": None}),
        (
            "sample_t0.8_p0.9",
            {"temperature": 0.8, "top_p": 0.9, "seed": 17},
        ),
    )
    generations: list[dict[str, Any]] = []
    for prompt in prompt_rows:
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt.text, add_bos=True, add_eos=False),
            dtype=torch.long,
            device=resolved_device,
        )
        for mode, decode in modes:
            generation = model.generate(
                prompt_ids,
                max_new_tokens=max(1, int(max_new_tokens)),
                eos_id=tokenizer.eos_id,
                repetition_penalty=1.08,
                no_repeat_ngram_size=3,
                **decode,
            )
            generated = generation["generated_ids"][0].detach().cpu().tolist()
            continuation_ids = generated[int(prompt_ids.numel()) :]
            continuation = tokenizer.decode(continuation_ids)
            generations.append(
                {
                    "prompt_id": prompt.prompt_id,
                    "capability": prompt.capability,
                    "prompt": prompt.text,
                    "review_question_metrics_only": prompt.review_question,
                    "prompt_exactly_present_in_sources": bool(
                        source_matches[prompt.prompt_id]
                    ),
                    "matching_source_paths": source_matches[prompt.prompt_id],
                    "decode_mode": mode,
                    "continuation": continuation,
                    "continuation_token_count": len(continuation_ids),
                    "sequence_hash": tokenizer.sequence_hash(generated),
                    "metrics": _text_metrics(continuation),
                    "generation_decode": generation["generation_decode"],
                    "owned_by_marulho": bool(generation["owned_by_marulho"]),
                    "external_llm_used": bool(generation["external_llm_used"]),
                    "review_question_used_for_generation": False,
                }
            )
    all_unseen = all(not paths for paths in source_matches.values())
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": all(
            bool(row["owned_by_marulho"]) for row in generations
        ),
        "external_llm_used": any(
            bool(row["external_llm_used"]) for row in generations
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "cumulative_update_tokens": int(
                metadata.get("cumulative_update_tokens") or 0
            ),
            "optimizer_steps": int(metadata.get("optimizer_steps") or 0),
            "runtime_state_present": runtime_state is not None,
        },
        "sources": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in sources
        ],
        "prompt_contract": {
            "prompts": [asdict(prompt) for prompt in prompt_rows],
            "prompt_count": len(prompt_rows),
            "decode_modes": [mode for mode, _decode in modes],
            "exact_prompt_absence_verified": bool(sources) and all_unseen,
            "source_search_streaming": True,
            "review_questions_metrics_only": True,
            "human_semantic_review_required": True,
        },
        "generations": generations,
        "decision_boundary": {
            "automated_surface_metrics_prove_coherence": False,
            "promotes_unseen_generation_quality": False,
            "promotes_runtime_installation": False,
            "branch_decision_requires_human_semantic_review": True,
        },
    }
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Organism Source-Absent Generation Audit",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_organism_generation_audit(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        source_paths=tuple(args.source),
        max_new_tokens=max(1, int(args.max_new_tokens)),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
