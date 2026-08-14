"""Run and decide V80's frozen unseen-language generation review."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_billion_continuation import (
    EVAL_ARTIFACT,
    EVAL_ARTIFACT_SHA256,
    EVAL_DOCUMENTS_PER_SOURCE,
    EVAL_TOKEN_SHA256,
    TARGET_CUMULATIVE_POSITIONS,
)
from marulho.evaluation.language_generation_coherence import (
    LanguageGenerationPromptCase,
    prompt_case_metadata,
    run_language_generation_coherence_report,
)
from marulho.evaluation.language_quality_continuation import (
    DOCUMENT_MARKER,
    DOCUMENT_TOKENS,
    ROOT,
    TOKENIZER_SHA256,
    _atomic_json,
    file_sha256,
)
from marulho.evaluation.language_scale_corpus_materialization import (
    _token_tensor_sha256,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import load_language_model_checkpoint


SURFACE = "marulho_language_billion_generation.v80"
DECISION_SURFACE = "marulho_language_billion_generation.v80_decision"
TRAINING_REPORT = (
    ROOT
    / "reports/language_scaling/"
    "v80-billion-position-continuation-20260814.json"
)
CHECKPOINT = (
    ROOT
    / "reports/language_scaling/"
    "v80-three-source-qualified-100m-1264m-20260814.pt"
)
FINEWEB_SOURCE = (
    ROOT
    / "reports/language_curriculum/"
    "fineweb-edu-eval-10k-shard1-20260710.txt"
)
FINEWEB_SOURCE_SHA256 = (
    "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a"
)
COSMOPEDIA_SOURCE = (
    ROOT
    / "reports/language_curriculum/"
    "cosmopedia-v2-eval-10k-shard2-20260710.txt"
)
COSMOPEDIA_SOURCE_SHA256 = (
    "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491"
)
DCLM_SOURCE = (
    ROOT
    / "reports/language_curriculum/"
    "v79-dclm-edu-selected-16896-20260814.txt"
)
DCLM_SOURCE_SHA256 = (
    "22f89f163967c7ef957b419410ea9e77402b9e7670350a3886ca40706c7ca0d7"
)

FINEWEB_PROMPTS = (
    "The Heilwood Company",
    "Heilwood Company store",
    "Company store was",
    "Being a rural,",
)
COSMOPEDIA_PROMPTS = (
    "In today's digital",
    "today's digital age,",
    "digital age, computers",
    "One critical aspect",
)
DCLM_ROWS_AND_PROMPTS = (
    (0, "Numerous industries utilize"),
    (2, "Summer in the"),
    (4, "Prevent Direct Execution"),
    (6, "In just one"),
)

V78_BASELINE_REPORTS = {
    "fineweb_greedy": (
        ROOT / "reports/language_scaling/v78-unseen-fineweb-greedy-20260814.json",
        "ea8ca8651f82f50943dd3a35c0472952811c2c26da25a448917395868b594928",
    ),
    "cosmopedia_greedy": (
        ROOT
        / "reports/language_scaling/v78-unseen-cosmopedia-greedy-20260814.json",
        "3ef5aa4d7f8bfa9d9ec6918f8a4b61e1b7d7e0b147f9d4afde046098ffe91e29",
    ),
    "cosmopedia_controlled": (
        ROOT
        / "reports/language_scaling/"
        "v78-unseen-cosmopedia-controlled-20260814.json",
        "0c01a4a45a9a23cf85f26b8ed175937bbfe53d70391c8468967616e828664930",
    ),
}
V80_PANEL_REPORTS = {
    "fineweb_greedy": (
        ROOT / "reports/language_scaling/v80-unseen-fineweb-greedy-20260814.json"
    ),
    "cosmopedia_greedy": (
        ROOT / "reports/language_scaling/v80-unseen-cosmopedia-greedy-20260814.json"
    ),
    "cosmopedia_controlled": (
        ROOT
        / "reports/language_scaling/"
        "v80-unseen-cosmopedia-controlled-20260814.json"
    ),
    "dclm_greedy": (
        ROOT / "reports/language_scaling/v80-unseen-dclm-greedy-20260814.json"
    ),
    "dclm_controlled": (
        ROOT / "reports/language_scaling/v80-unseen-dclm-controlled-20260814.json"
    ),
}
REVIEW_PATH = (
    ROOT / "reports/language_scaling/v80-unseen-generation-review-20260814.md"
)
DECISION_REPORT = (
    ROOT / "reports/language_scaling/v80-unseen-generation-decision-20260814.json"
)


@dataclass(frozen=True)
class PanelSpec:
    name: str
    source_path: Path
    source_sha256: str
    prompts: tuple[str, ...]
    dclm_rows: tuple[int, ...] = ()
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0


PANEL_SPECS = (
    PanelSpec(
        name="fineweb_greedy",
        source_path=FINEWEB_SOURCE,
        source_sha256=FINEWEB_SOURCE_SHA256,
        prompts=FINEWEB_PROMPTS,
    ),
    PanelSpec(
        name="cosmopedia_greedy",
        source_path=COSMOPEDIA_SOURCE,
        source_sha256=COSMOPEDIA_SOURCE_SHA256,
        prompts=COSMOPEDIA_PROMPTS,
    ),
    PanelSpec(
        name="cosmopedia_controlled",
        source_path=COSMOPEDIA_SOURCE,
        source_sha256=COSMOPEDIA_SOURCE_SHA256,
        prompts=COSMOPEDIA_PROMPTS,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    ),
    PanelSpec(
        name="dclm_greedy",
        source_path=DCLM_SOURCE,
        source_sha256=DCLM_SOURCE_SHA256,
        prompts=tuple(prompt for _, prompt in DCLM_ROWS_AND_PROMPTS),
        dclm_rows=tuple(row for row, _ in DCLM_ROWS_AND_PROMPTS),
    ),
    PanelSpec(
        name="dclm_controlled",
        source_path=DCLM_SOURCE,
        source_sha256=DCLM_SOURCE_SHA256,
        prompts=tuple(prompt for _, prompt in DCLM_ROWS_AND_PROMPTS),
        dclm_rows=tuple(row for row, _ in DCLM_ROWS_AND_PROMPTS),
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_marked_documents(path: Path, rows: Sequence[int]) -> dict[int, str]:
    requested = {int(row) for row in rows}
    if not requested or min(requested) < 0:
        raise ValueError("DCLM rows must be non-negative and non-empty")
    documents: dict[int, str] = {}
    document_index = -1
    current: list[str] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.rstrip("\r\n")
            if stripped == DOCUMENT_MARKER:
                if current is not None and document_index in requested:
                    documents[document_index] = "\n".join(current).strip()
                document_index += 1
                if len(documents) == len(requested):
                    break
                current = [] if document_index in requested else None
            elif current is not None:
                current.append(stripped)
        else:
            if current is not None and document_index in requested:
                documents[document_index] = "\n".join(current).strip()
    missing = requested.difference(documents)
    if missing:
        raise RuntimeError(f"DCLM source is missing frozen rows: {sorted(missing)}")
    return documents


def _source_texts() -> dict[str, dict[int, str] | str]:
    checks = {
        FINEWEB_SOURCE: FINEWEB_SOURCE_SHA256,
        COSMOPEDIA_SOURCE: COSMOPEDIA_SOURCE_SHA256,
        DCLM_SOURCE: DCLM_SOURCE_SHA256,
    }
    for path, expected in checks.items():
        actual = file_sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen generation source changed: {path}: {actual}")
    return {
        "fineweb": FINEWEB_SOURCE.read_text(encoding="utf-8"),
        "cosmopedia": COSMOPEDIA_SOURCE.read_text(encoding="utf-8"),
        "dclm": _read_marked_documents(
            DCLM_SOURCE,
            [row for row, _ in DCLM_ROWS_AND_PROMPTS],
        ),
    }


def _cases_for_spec(
    spec: PanelSpec,
    sources: Mapping[str, dict[int, str] | str],
) -> tuple[LanguageGenerationPromptCase, ...]:
    if spec.dclm_rows:
        dclm = sources["dclm"]
        if not isinstance(dclm, Mapping):
            raise TypeError("DCLM source must be indexed by frozen row")
        texts = [str(dclm[row]) for row in spec.dclm_rows]
    else:
        key = "fineweb" if spec.name.startswith("fineweb") else "cosmopedia"
        source = sources[key]
        if not isinstance(source, str):
            raise TypeError(f"{key} source must be text")
        texts = [source] * len(spec.prompts)
    cases = tuple(
        LanguageGenerationPromptCase(prompt_text=prompt, source_text=source_text)
        for prompt, source_text in zip(spec.prompts, texts, strict=True)
    )
    missing = [case.prompt_text for case in cases if case.prompt_text not in case.source_text]
    if missing:
        raise RuntimeError(f"Frozen prompts missing from their source: {missing}")
    return cases


def _validate_dclm_eval_rows(tokenizer: Any, sources: Mapping[str, Any]) -> dict[str, Any]:
    if file_sha256(EVAL_ARTIFACT) != EVAL_ARTIFACT_SHA256:
        raise RuntimeError("V80 evaluation artifact hash changed")
    artifact = torch.load(EVAL_ARTIFACT, map_location="cpu", weights_only=False)
    tokens = artifact["tokens"].to(dtype=torch.int32)
    checks = {
        "token_hash_exact": _token_tensor_sha256(tokens) == EVAL_TOKEN_SHA256,
        "shape_exact": tuple(tokens.shape)
        == (3 * EVAL_DOCUMENTS_PER_SOURCE, DOCUMENT_TOKENS),
    }
    dclm = sources["dclm"]
    for row, prompt in DCLM_ROWS_AND_PROMPTS:
        source_text = str(dclm[row])
        encoded = torch.tensor(
            tokenizer.encode(source_text, add_bos=True, add_eos=True)[:DOCUMENT_TOKENS],
            dtype=torch.int32,
        )
        target = tokens[2 * EVAL_DOCUMENTS_PER_SOURCE + row]
        checks[f"row_{row}_token_exact"] = bool(torch.equal(encoded, target))
        checks[f"row_{row}_prompt_present"] = prompt in source_text
    if not all(checks.values()):
        raise RuntimeError(f"Frozen DCLM generation rows failed eval binding: {checks}")
    return {
        "path": str(EVAL_ARTIFACT),
        "sha256": EVAL_ARTIFACT_SHA256,
        "token_sha256": EVAL_TOKEN_SHA256,
        "checks": checks,
    }


def _training_admission(
    training_report_path: Path,
    checkpoint_path: Path,
) -> tuple[dict[str, Any], str]:
    report = _load_json(training_report_path)
    checkpoint = dict(report.get("checkpoint") or {})
    checks = {
        "surface_exact": report.get("surface")
        == "marulho_language_billion_continuation.v80",
        "training_passed": report.get("passed") is True,
        "decision_exact": report.get("decision")
        == "admit_v80_checkpoint_to_unseen_generation",
        "quality_checks_all_pass": bool(report.get("quality_checks"))
        and all(bool(value) for value in report["quality_checks"].values()),
        "checkpoint_saved": checkpoint.get("saved") is True,
        "checkpoint_fidelity_passed": dict(checkpoint.get("fidelity") or {}).get(
            "passed"
        )
        is True,
        "cumulative_positions_exact": int(
            report.get("configuration", {}).get(
                "target_cumulative_processed_positions", -1
            )
        )
        == TARGET_CUMULATIVE_POSITIONS,
        "owned_by_marulho": report.get("owned_by_marulho") is True,
        "external_llm_absent": report.get("external_llm_used") is False,
        "checkpoint_exists": checkpoint_path.is_file(),
    }
    expected_hash = str(checkpoint.get("sha256") or "")
    if checkpoint_path.is_file():
        checks["checkpoint_hash_exact"] = file_sha256(checkpoint_path) == expected_hash
    else:
        checks["checkpoint_hash_exact"] = False
    if not all(checks.values()):
        raise RuntimeError(f"V80 generation is not admitted: {checks}")
    return {
        "path": str(training_report_path),
        "sha256": file_sha256(training_report_path),
        "checks": checks,
        "final_evaluation": report["final_evaluation"],
        "checkpoint_fidelity": checkpoint["fidelity"],
    }, expected_hash


def _render_review(reports: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "# V80 unseen-generation review",
        "",
        "Judge visible language, not automatic prefix counts. A coherent pass requires "
        "multi-sentence text that remains on the prompt's topic without template or "
        "repetition collapse.",
        "",
    ]
    for panel_name, report in reports.items():
        lines.extend([f"## {panel_name}", ""])
        for index, case in enumerate(report["cases"], start=1):
            lines.extend(
                [
                    f"### Case {index}: {case['prompt_text']}",
                    "",
                    "Generated:",
                    "",
                    str(case["generated_text"]),
                    "",
                    "Source continuation:",
                    "",
                    str(case["expected_source_continuation"]),
                    "",
                    "---",
                    "",
                ]
            )
    return "\n".join(lines)


def run_v80_generation_panels(
    *,
    training_report_path: Path = TRAINING_REPORT,
    checkpoint_path: Path = CHECKPOINT,
    output_paths: Mapping[str, Path] = V80_PANEL_REPORTS,
    review_path: Path = REVIEW_PATH,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("V80 generation review requires observed CUDA execution")
    requested_paths = [Path(output_paths[spec.name]) for spec in PANEL_SPECS]
    if any(path.exists() for path in (*requested_paths, review_path)):
        raise ValueError("V80 generation output already exists")
    admission, expected_checkpoint_hash = _training_admission(
        training_report_path,
        checkpoint_path,
    )
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    checkpoint_checks = {
        "tokenizer_exact": tokenizer.vocabulary_hash() == TOKENIZER_SHA256,
        "language_path_exact": model.config.active_language_path
        == "marulho_transformer",
        "metadata_decision_exact": metadata.get("decision")
        == "save_v80_billion_checkpoint_for_unseen_generation",
        "metadata_cumulative_positions_exact": int(
            metadata.get("cumulative_processed_tokens", -1)
        )
        == TARGET_CUMULATIVE_POSITIONS,
        "external_llm_absent": metadata.get("external_llm_used") is False,
    }
    if not all(checkpoint_checks.values()):
        raise RuntimeError(f"V80 checkpoint metadata failed: {checkpoint_checks}")
    sources = _source_texts()
    dclm_eval_binding = _validate_dclm_eval_rows(tokenizer, sources)
    device = torch.device("cuda")
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    torch.cuda.reset_peak_memory_stats(device)
    reports: dict[str, dict[str, Any]] = {}
    for spec in PANEL_SPECS:
        cases = _cases_for_spec(spec, sources)
        reports[spec.name] = run_language_generation_coherence_report(
            model,
            tokenizer,
            prompt_cases=cases,
            min_case_pass_rate=1.0,
            checkpoint_path=checkpoint_path,
            checkpoint_kind="transformer",
            checkpoint_metadata=metadata,
            source_path=spec.source_path,
            generation_repetition_penalty=spec.repetition_penalty,
            generation_no_repeat_ngram_size=spec.no_repeat_ngram_size,
        )
        reports[spec.name]["v80_generation_contract"] = {
            "surface": SURFACE,
            "training_admission": admission,
            "checkpoint_checks": checkpoint_checks,
            "dclm_eval_binding": dclm_eval_binding,
            "run_peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "expected_checkpoint_sha256": expected_checkpoint_hash,
        }
    for spec in PANEL_SPECS:
        write_json_report_with_readme(
            output_paths[spec.name],
            reports[spec.name],
            title=f"V80 Unseen Generation: {spec.name}",
        )
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(_render_review(reports), encoding="utf-8")
    return {
        "surface": SURFACE,
        "decision": "await_human_review",
        "checkpoint_sha256": expected_checkpoint_hash,
        "panel_paths": {name: str(path) for name, path in output_paths.items()},
        "review_path": str(review_path),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def _expected_decode_controls(spec: PanelSpec) -> dict[str, Any]:
    return {
        "repetition_penalty": float(spec.repetition_penalty),
        "repetition_penalty_applied": spec.repetition_penalty > 1.0,
        "no_repeat_ngram_size": int(spec.no_repeat_ngram_size),
        "no_repeat_ngram_applied": spec.no_repeat_ngram_size > 0,
        "decode_controls_requested": (
            spec.repetition_penalty > 1.0 or spec.no_repeat_ngram_size > 0
        ),
    }


def _panel_checks(
    report: Mapping[str, Any],
    *,
    spec: PanelSpec,
    cases: Sequence[LanguageGenerationPromptCase],
    checkpoint_sha256: str,
) -> dict[str, bool]:
    return {
        "prompt_contract_exact": report.get("prompt_suite", {}).get(
            "prompt_cases"
        )
        == [prompt_case_metadata(case) for case in cases],
        "decode_controls_exact": report.get("prompt_suite", {}).get(
            "generation_decode_controls"
        )
        == _expected_decode_controls(spec),
        "source_sha256_exact": report.get("source", {}).get("sha256")
        == spec.source_sha256,
        "checkpoint_sha256_exact": report.get("checkpoint", {}).get("sha256")
        == checkpoint_sha256,
        "tokenizer_sha256_exact": report.get("checkpoint", {}).get(
            "tokenizer_hash"
        )
        == TOKENIZER_SHA256,
        "marulho_owned": report.get("owned_by_marulho") is True,
        "external_llm_absent": report.get("external_llm_used") is False,
        "four_cases_complete": len(report.get("cases", ())) == 4,
    }


def generation_decision(*, validity_passed: bool, human_review_coherent: bool) -> str:
    if not validity_passed:
        return "reject_v80_unseen_generation_invalid_evidence"
    if human_review_coherent:
        return "advance_v80_to_continual_and_grounded_self_challenge_validation"
    return "redesign_v80_objective_or_tokenizer_after_scale_failure"


def aggregate_v80_generation(
    *,
    output_path: Path = DECISION_REPORT,
    training_report_path: Path = TRAINING_REPORT,
    checkpoint_path: Path = CHECKPOINT,
    panel_paths: Mapping[str, Path] = V80_PANEL_REPORTS,
    human_review_coherent: bool,
    human_review_observations: Sequence[str],
) -> dict[str, Any]:
    if output_path.exists():
        raise ValueError(f"V80 generation decision already exists: {output_path}")
    if not human_review_observations:
        raise ValueError("Human review requires at least one concrete observation")
    admission, checkpoint_sha256 = _training_admission(
        training_report_path,
        checkpoint_path,
    )
    sources = _source_texts()
    specs = {spec.name: spec for spec in PANEL_SPECS}
    panels: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    total_passed = 0
    total_cases = 0
    for name, spec in specs.items():
        candidate_path = Path(panel_paths[name])
        candidate = _load_json(candidate_path)
        cases = _cases_for_spec(spec, sources)
        panel_checks = _panel_checks(
            candidate,
            spec=spec,
            cases=cases,
            checkpoint_sha256=checkpoint_sha256,
        )
        for check_name, passed in panel_checks.items():
            checks[f"{name}_{check_name}"] = bool(passed)
        summary = dict(candidate["summary"])
        total_passed += int(summary["passed_case_count"])
        total_cases += int(summary["case_count"])
        panel: dict[str, Any] = {
            "candidate_report": {
                "path": str(candidate_path),
                "sha256": file_sha256(candidate_path),
            },
            "checks": panel_checks,
            "candidate_summary": summary,
        }
        if name in V78_BASELINE_REPORTS:
            baseline_path, baseline_hash = V78_BASELINE_REPORTS[name]
            actual_baseline_hash = file_sha256(baseline_path)
            if actual_baseline_hash != baseline_hash:
                raise RuntimeError(f"Frozen V78 generation report changed: {baseline_path}")
            baseline = _load_json(baseline_path)
            baseline_summary = dict(baseline["summary"])
            panel["v78_baseline_report"] = {
                "path": str(baseline_path),
                "sha256": baseline_hash,
            }
            panel["v78_baseline_summary"] = baseline_summary
            panel["deltas_v80_minus_v78"] = {
                field: float(summary[field]) - float(baseline_summary[field])
                for field in (
                    "mean_source_continuation_loss",
                    "mean_prefix_match_chars",
                    "mean_distinct_bigram_fraction",
                )
            }
        panels[name] = panel
    validity_passed = all(checks.values())
    decision = generation_decision(
        validity_passed=validity_passed,
        human_review_coherent=human_review_coherent,
    )
    payload = {
        "surface": DECISION_SURFACE,
        "artifact_kind": "marulho_language_unseen_generation_decision",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "validity_passed": validity_passed,
        "checks": checks,
        "training_admission": admission,
        "checkpoint_sha256": checkpoint_sha256,
        "panels": panels,
        "aggregate": {
            "passed_cases": total_passed,
            "case_count": total_cases,
            "case_pass_rate": total_passed / total_cases if total_cases else 0.0,
        },
        "human_review": {
            "review_kind": "direct_visible_text_human_review",
            "coherent_multi_sentence_generation": human_review_coherent,
            "requires_topic_stability": True,
            "requires_no_template_or_repetition_collapse": True,
            "observations": list(human_review_observations),
        },
        "decision": decision,
        "checkpoint_boundary": {
            "retained_as_strongest_quantitative_base": validity_passed,
            "coherent_generation_claimed": bool(
                validity_passed and human_review_coherent
            ),
            "continual_learning_admitted": decision
            == "advance_v80_to_continual_and_grounded_self_challenge_validation",
            "runtime_install_allowed": False,
        },
    }
    _atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-panels", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--training-report", type=Path, default=TRAINING_REPORT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--human-review-verdict",
        choices=("coherent", "not-coherent"),
    )
    parser.add_argument("--human-review-observation", action="append", default=[])
    args = parser.parse_args()
    if args.run_panels == args.aggregate:
        parser.error("choose exactly one of --run-panels or --aggregate")
    if args.run_panels:
        result = run_v80_generation_panels(
            training_report_path=args.training_report,
            checkpoint_path=args.checkpoint,
        )
    else:
        if args.human_review_verdict is None:
            parser.error("--aggregate requires --human-review-verdict")
        result = aggregate_v80_generation(
            training_report_path=args.training_report,
            checkpoint_path=args.checkpoint,
            human_review_coherent=args.human_review_verdict == "coherent",
            human_review_observations=args.human_review_observation,
        )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":  # pragma: no cover
    main()
