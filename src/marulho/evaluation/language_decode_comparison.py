"""Compare MARULHO-owned greedy and sampled decoding from one checkpoint."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from marulho.evaluation.language_scaling_experiment import DEFAULT_PROMPTS
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_language_decode_comparison.v1"
ARTIFACT_KIND = "marulho_language_decode_comparison"


@dataclass(frozen=True)
class DecodePolicy:
    name: str
    temperature: float
    top_p: float
    seed: int = 1337


DEFAULT_POLICIES = (
    DecodePolicy("greedy", temperature=0.0, top_p=1.0),
    DecodePolicy("nucleus_0p8_0p9", temperature=0.8, top_p=0.9),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_policy(value: str) -> DecodePolicy:
    parts = str(value).split(":")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "policy must be name:temperature:top_p:seed"
        )
    name, temperature, top_p, seed = parts
    return DecodePolicy(
        name=str(name),
        temperature=float(temperature),
        top_p=float(top_p),
        seed=int(seed),
    )


def run_language_decode_comparison(
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    policies: Sequence[DecodePolicy] = DEFAULT_POLICIES,
    max_new_tokens: int = 192,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    if not prompts:
        raise ValueError("At least one prompt is required")
    if not policies:
        raise ValueError("At least one decode policy is required")
    resolved_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if str(device) == "auto" else torch.device(device)
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model = model.to(resolved_device)
    model.eval()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for prompt_index, prompt in enumerate(prompts):
            effective_seed = int(policy.seed) + int(prompt_index)
            prompt_ids = torch.tensor(
                tokenizer.encode(str(prompt), add_bos=True, add_eos=False),
                dtype=torch.long,
                device=resolved_device,
            )
            generated = model.generate(
                prompt_ids,
                max_new_tokens=max(0, int(max_new_tokens)),
                eos_id=tokenizer.eos_id,
                repetition_penalty=max(1.0, float(repetition_penalty)),
                no_repeat_ngram_size=max(0, int(no_repeat_ngram_size)),
                temperature=float(policy.temperature),
                top_p=float(policy.top_p),
                seed=effective_seed,
            )
            ids = [
                int(value)
                for value in generated["generated_ids"][0].detach().cpu().tolist()
            ]
            continuation_ids = ids[int(prompt_ids.numel()) :]
            rows.append(
                {
                    "policy": str(policy.name),
                    "prompt": str(prompt),
                    "effective_seed": effective_seed,
                    "prompt_token_count": int(prompt_ids.numel()),
                    "continuation_token_count": len(continuation_ids),
                    "generated_text": tokenizer.decode(ids),
                    "continuation_text": tokenizer.decode(continuation_ids),
                    "sequence_hash": tokenizer.sequence_hash(ids),
                    "generation_decode": dict(generated["generation_decode"]),
                    "owned_by_marulho": True,
                    "external_llm_used": False,
                }
            )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "cumulative_update_tokens": metadata.get("cumulative_update_tokens"),
            "cumulative_optimizer_steps": metadata.get(
                "cumulative_optimizer_steps"
            ),
            "tokenizer_hash": tokenizer.vocabulary_hash(),
        },
        "device": str(resolved_device),
        "policies": [asdict(policy) for policy in policies],
        "prompt_count": len(prompts),
        "generation_count": len(rows),
        "max_new_tokens": max(0, int(max_new_tokens)),
        "repetition_penalty": max(1.0, float(repetition_penalty)),
        "no_repeat_ngram_size": max(0, int(no_repeat_ngram_size)),
        "generations": rows,
        "quality_boundary": {
            "human_review_required": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
            "claim": "decode_policy_ablation_only",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_report_with_readme(
        output,
        report,
        title="MARULHO Decode Policy Comparison",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--policy", action="append", type=_parse_policy, default=[])
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_language_decode_comparison(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        prompts=tuple(args.prompt) or DEFAULT_PROMPTS,
        policies=tuple(args.policy) or DEFAULT_POLICIES,
        max_new_tokens=max(0, int(args.max_new_tokens)),
        repetition_penalty=max(1.0, float(args.repetition_penalty)),
        no_repeat_ngram_size=max(0, int(args.no_repeat_ngram_size)),
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
