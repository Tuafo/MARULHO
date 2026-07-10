"""Unseen-prompt generation audit for a MARULHO delta checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
from typing import Any, Sequence

import torch

from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_delta import load_delta_language_checkpoint


SURFACE = "marulho_delta_generation_audit.v1"
ARTIFACT_KIND = "marulho_delta_generation_audit"
DEFAULT_PROMPTS = (
    "A careful scientist changes one variable at a time because",
    "When the red boat reached the quiet island,",
    "The difference between remembering and understanding is",
    (
        "Mira placed the silver key inside the wooden drawer. Later she moved "
        "it into the glass jar. The silver key is now"
    ),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_metrics(text: str) -> dict[str, Any]:
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)
    bigrams = list(zip(words, words[1:]))
    distinct_bigram_fraction = (
        len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    )
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    replacement_count = text.count("�")
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_terminator_count": sum(text.count(mark) for mark in ".!?"),
        "printable_fraction": printable / len(text) if text else 0.0,
        "replacement_character_count": replacement_count,
        "distinct_bigram_fraction": distinct_bigram_fraction,
    }


def run_delta_generation_audit(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    source_paths: Sequence[str | Path] = (),
    max_new_tokens: int = 64,
    device: str = "cuda",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if not prompts:
        raise ValueError("At least one unseen prompt is required")
    source_files = tuple(Path(path) for path in source_paths)
    source_texts = tuple(path.read_text(encoding="utf-8") for path in source_files)
    normalized_sources = tuple(text.casefold() for text in source_texts)
    model, tokenizer, metadata, _runtime_state = load_delta_language_checkpoint(
        checkpoint
    )
    resolved_device = torch.device(
        "cuda" if device == "auto" and torch.cuda.is_available() else device
    )
    model = model.to(resolved_device).eval()
    rows: list[dict[str, Any]] = []
    modes = (
        (
            "greedy",
            {
                "temperature": 0.0,
                "top_p": 1.0,
                "seed": None,
            },
        ),
        (
            "sample_t0.8_p0.9",
            {
                "temperature": 0.8,
                "top_p": 0.9,
                "seed": 17,
            },
        ),
    )
    for prompt in prompts:
        prompt_text = str(prompt)
        prompt_present = any(
            prompt_text.casefold() in source for source in normalized_sources
        )
        prompt_ids = torch.tensor(
            tokenizer.encode(prompt_text, add_bos=True, add_eos=False),
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
            rows.append(
                {
                    "prompt": prompt_text,
                    "prompt_exactly_present_in_sources": prompt_present,
                    "decode_mode": mode,
                    "continuation": continuation,
                    "continuation_token_count": len(continuation_ids),
                    "sequence_hash": tokenizer.sequence_hash(generated),
                    "metrics": _text_metrics(continuation),
                    "generation_decode": generation["generation_decode"],
                    "owned_by_marulho": bool(generation["owned_by_marulho"]),
                    "external_llm_used": bool(generation["external_llm_used"]),
                }
            )
    all_unseen = all(
        not bool(row["prompt_exactly_present_in_sources"]) for row in rows
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": all(bool(row["owned_by_marulho"]) for row in rows),
        "external_llm_used": any(bool(row["external_llm_used"]) for row in rows),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "cumulative_update_tokens": int(
                metadata.get("cumulative_update_tokens") or 0
            ),
            "optimizer_steps": int(metadata.get("optimizer_steps") or 0),
        },
        "sources": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in source_files
        ],
        "prompt_contract": {
            "prompt_count": len(prompts),
            "decode_modes": [mode for mode, _decode in modes],
            "exact_prompt_absence_verified": bool(source_files) and all_unseen,
            "human_semantic_review_required": True,
        },
        "generations": rows,
        "decision_boundary": {
            "automated_surface_metrics_prove_coherence": False,
            "promotes_unseen_generation_quality": False,
            "promotes_runtime_installation": False,
        },
    }
    write_json_report_with_readme(
        output, report, title="MARULHO Delta Unseen-Generation Audit"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, default=[])
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_delta_generation_audit(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        prompts=tuple(args.prompt) or DEFAULT_PROMPTS,
        source_paths=tuple(args.source),
        max_new_tokens=max(1, int(args.max_new_tokens)),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
