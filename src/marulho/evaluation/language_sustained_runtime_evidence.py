"""Measure sustained generation from the active MARULHO Transformer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
from typing import Any

import torch

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    load_language_model_checkpoint,
)


SURFACE = "marulho_transformer_sustained_generation.v2"


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_language_sustained_runtime_evidence(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    output_path: str | Path,
    target_tokens: int,
    checkpoint_path: str | Path | None = None,
    prompt: str = "MARULHO",
    timeout_seconds: float = 600.0,
    generation_repetition_penalty: float = 1.1,
    generation_no_repeat_ngram_size: int = 3,
    collect_environment: bool = False,
) -> dict[str, Any]:
    del collect_environment
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = max(0, int(target_tokens))
    prompt_ids = torch.tensor(
        tokenizer.encode(str(prompt), add_bos=True, add_eos=False),
        dtype=torch.long,
        device=model.device,
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    generated = model.generate(
        prompt_ids,
        max_new_tokens=target,
        eos_id=None,
        repetition_penalty=max(1.0, float(generation_repetition_penalty)),
        no_repeat_ngram_size=max(0, int(generation_no_repeat_ngram_size)),
    )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    token_delta = int(generated["new_token_count"])
    timed_out = elapsed > float(timeout_seconds)
    ids = [int(value) for value in generated["generated_ids"][0].cpu().tolist()]
    continuation = ids[int(prompt_ids.numel()) :]
    report = {
        "artifact_kind": "marulho_transformer_sustained_generation",
        "surface": SURFACE,
        "report_status": "accepted" if token_delta == target and not timed_out else "rejected",
        "success": token_delta == target and not timed_out,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "target_tokens": target,
        "token_delta": token_delta,
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(token_delta) / elapsed,
        "timeout_seconds": float(timeout_seconds),
        "timed_out": timed_out,
        "device_backend": str(model.device),
        "model_vocab_size": int(model.config.vocab_size),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "generation_vocab_size": int(model.generation_vocab_size),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "prompt": str(prompt),
        "prompt_token_count": int(prompt_ids.numel()),
        "generated_text": tokenizer.decode(ids),
        "continuation_text": tokenizer.decode(continuation),
        "generation_decode": generated["generation_decode"],
        "runtime": {
            "state_core": "transformer",
            "bounded_kv_cache": True,
            "context_length": int(model.config.transformer_context_length),
            "routing_present": False,
            "spiking_present": False,
            "sampled_padded_vocab_present": False,
        },
        "promotes_runtime_claim": False,
        "promotes_generation_quality_claim": False,
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--prompt", default="MARULHO")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--map-location", default="auto")
    args = parser.parse_args()
    target_device = (
        "cuda"
        if args.map_location == "auto" and torch.cuda.is_available()
        else ("cpu" if args.map_location == "auto" else args.map_location)
    )
    model, tokenizer, _metadata = load_language_model_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    model = model.to(target_device)
    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=args.output,
        target_tokens=max(0, int(args.target_tokens)),
        checkpoint_path=args.checkpoint,
        prompt=args.prompt,
        timeout_seconds=float(args.timeout_seconds),
        generation_repetition_penalty=float(args.generation_repetition_penalty),
        generation_no_repeat_ngram_size=int(args.generation_no_repeat_ngram_size),
    )
    return 0 if bool(report["success"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
