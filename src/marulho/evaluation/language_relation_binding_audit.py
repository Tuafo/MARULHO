"""Audit candidate ranking and free-form relation binding for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from marulho.evaluation.language_relation_binding_experiment import (
    RelationCase,
    evaluate_relation_binding_cases,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_language_relation_binding_checkpoint_audit.v1"
ARTIFACT_KIND = "marulho_language_relation_binding_checkpoint_audit"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_relation_binding_checkpoint_audit(
    *,
    checkpoint_path: str | Path,
    cases_path: str | Path,
    output_path: str | Path,
    device: str = "auto",
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    cases_file = Path(cases_path)
    payload = json.loads(cases_file.read_text(encoding="utf-8"))
    cases = tuple(
        RelationCase(
            case_id=str(row["case_id"]),
            kind=str(row["kind"]),
            signature=str(row["signature"]),
            prompt=str(row["prompt"]),
            candidates=tuple(str(value) for value in row["candidates"]),
            correct_index=int(row["correct_index"]),
        )
        for row in payload["cases"]
    )
    resolved_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if str(device) == "auto" else torch.device(device)
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    model = model.to(resolved_device)
    evaluation = evaluate_relation_binding_cases(model, tokenizer, cases)
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
        "cases": {
            "path": str(cases_file),
            "sha256": _sha256_file(cases_file),
            "case_count": len(cases),
            "split_policy": payload.get("split_policy"),
            "correct_index_metrics_only": True,
        },
        "evaluation": evaluation,
        "quality_boundary": {
            "human_review_required": True,
            "promotes_generation_quality_claim": False,
            "promotes_runtime_claim": False,
        },
    }
    write_json_report_with_readme(
        output_path,
        report,
        title="MARULHO Relation-Binding Checkpoint Audit",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_relation_binding_checkpoint_audit(
        checkpoint_path=args.checkpoint,
        cases_path=args.cases,
        output_path=args.output,
        device=str(args.device),
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
