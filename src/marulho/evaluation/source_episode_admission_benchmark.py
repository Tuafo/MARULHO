"""Quality and cost gate for bounded explicit-feed source episode admission."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from marulho.gap_planner import plan_query_gaps
from marulho.interaction import EvidenceResponder
from marulho.reporting.io import write_json_file
from marulho.semantics import ConceptStore
from marulho.training.meaning_grounding_runner import (
    meaning_grounding_benchmark_config,
    meaning_grounding_scenario_payload,
)
from marulho.training.model import MarulhoModel
from marulho.training.query_runner import build_query_result, feed_text
from marulho.training.runner_utils import set_seed
from marulho.training.trainer import MarulhoTrainer
from marulho.data.rtf_encoder import RTFEncoder


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        return float(psutil.Process().memory_info().rss / (1024.0 * 1024.0))
    except Exception:
        return None


def _cuda_memory_mb() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "allocated_mb": None,
            "reserved_mb": None,
            "max_allocated_mb": None,
            "max_reserved_mb": None,
        }
    return {
        "allocated_mb": float(torch.cuda.memory_allocated() / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved() / (1024.0 * 1024.0)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024.0 * 1024.0)),
    }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))
    return float(ordered[index])


def _top_concept_terms(concept_summary: Mapping[str, Any]) -> list[str]:
    concepts = list(concept_summary.get("concepts") or [])
    if not concepts:
        return []
    return [str(term).lower() for term in list(concepts[0].get("top_terms") or [])]


def _evaluate_case(case: Mapping[str, Any], query_summary: Mapping[str, Any], response: Mapping[str, Any], concept_summary: Mapping[str, Any]) -> dict[str, Any]:
    evidence_text = " ".join(
        str(item.get("text", "")) for item in response.get("selected_evidence") or []
    ).lower()
    response_text = str(response.get("response_text", "")).lower()
    expected_terms = [str(term).lower() for term in case.get("expected_terms") or []]
    expected_phrases = [str(term).lower() for term in case.get("expected_phrases") or []]
    expected_concept_terms = [
        str(term).lower() for term in case.get("expected_concept_terms") or []
    ]
    forbidden_concept_terms = [
        str(term).lower() for term in case.get("forbidden_concept_terms") or []
    ]
    forbidden_unsupported_terms = [
        str(term).lower() for term in case.get("forbidden_unsupported_terms") or []
    ]
    expected_mode = str(case.get("expected_mode", "")).strip()
    grounded_terms = [
        term for term in expected_terms if term in evidence_text or term in response_text
    ]
    grounded_phrases = [
        term for term in expected_phrases if term in evidence_text or term in response_text
    ]
    episodes = list(query_summary.get("memory_episodes") or [])
    top_terms = _top_concept_terms(concept_summary)
    concept_pass = (
        not expected_concept_terms
        or any(term in top_terms for term in expected_concept_terms)
    ) and all(term not in top_terms for term in forbidden_concept_terms)
    unsupported_terms = [
        str(term).lower() for term in response.get("unsupported_terms") or []
    ]
    unsupported_pass = all(
        term not in unsupported_terms for term in forbidden_unsupported_terms
    )
    episode_evidence_present = any(
        len(str(item.get("text", "")).strip())
        > len(str(item.get("raw_window", "")).strip()) + 8
        for item in episodes
    )
    if bool(case.get("expect_response")):
        passed = (
            response.get("response_mode") != "insufficient_evidence"
            and bool(grounded_terms)
            and len(grounded_phrases) == len(expected_phrases)
            and episode_evidence_present
            and any(
                not bool(item.get("fragmentary"))
                for item in response.get("selected_evidence") or []
            )
            and concept_pass
            and unsupported_pass
        )
    else:
        passed = (
            response.get("response_mode") == (expected_mode or "insufficient_evidence")
            and not grounded_terms
            and unsupported_pass
        )
    if expected_mode:
        passed = bool(passed and response.get("response_mode") == expected_mode)
    return {
        "name": str(case.get("name", "")),
        "pass": bool(passed),
        "response_mode": response.get("response_mode"),
        "grounded_terms": grounded_terms,
        "grounded_phrases": grounded_phrases,
        "top_concept_terms": top_terms,
        "concept_pass": bool(concept_pass),
        "unsupported_pass": bool(unsupported_pass),
        "episode_evidence_present": bool(episode_evidence_present),
        "selected_evidence": [
            {
                "memory_index": item.get("memory_index"),
                "text": item.get("text"),
                "matching_terms": list(item.get("matching_terms") or []),
                "source_type": item.get("source_type"),
            }
            for item in list(response.get("selected_evidence") or [])
        ],
        "memory_match_report": dict(query_summary.get("memory_match_report") or {}),
        "memory_episode_report": dict(query_summary.get("memory_episode_report") or {}),
    }


def _run_arm(*, scenario: str, seed: int, admit_source_episodes: bool) -> dict[str, Any]:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    payload = meaning_grounding_scenario_payload(scenario)
    cfg = meaning_grounding_benchmark_config()
    trainer = MarulhoTrainer(MarulhoModel(cfg), cfg)
    encoder = RTFEncoder.from_config(cfg)
    concept_store = ConceptStore()
    responder = EvidenceResponder(min_similarity=0.0, min_token_coverage=0.25)

    rss_before = _process_rss_mb()
    cuda_before = _cuda_memory_mb()
    started = time.perf_counter()
    feed_summary = feed_text(
        trainer,
        encoder,
        str(payload["feed_text"]),
        admit_source_episodes=admit_source_episodes,
    )
    feed_latency_ms = (time.perf_counter() - started) * 1000.0

    query_latencies: list[float] = []
    cases: list[dict[str, Any]] = []
    for case in payload["queries"]:
        query_started = time.perf_counter()
        query_result = build_query_result(
            trainer=trainer,
            checkpoint=Path(f"benchmark://{scenario}"),
            metadata={
                "benchmark": "source_episode_admission",
                "scenario": scenario,
                "admit_source_episodes": bool(admit_source_episodes),
            },
            encoder=encoder,
            query_text_resolved=str(case["query"]),
            feed_text_resolved=None,
            context_text=None,
            top_k_candidates=6,
            top_k_memories=8,
            top_chars=8,
            compare_context_a=None,
            compare_context_b=None,
        )
        query_latencies.append((time.perf_counter() - query_started) * 1000.0)
        query_summary = dict(query_result.get("query_summary") or {})
        concept_summary = concept_store.observe(
            query_text=str(case["query"]),
            memory_matches=list(query_summary.get("memory_matches") or []),
            memory_episodes=list(query_summary.get("memory_episodes") or []),
            memory_store=trainer.model.memory_store,
        )
        response = responder.build_response(
            str(case["query"]),
            query_summary,
            concept_summary=concept_summary,
            max_evidence_items=3,
        )
        gap_plan = plan_query_gaps(
            query_text=str(case["query"]),
            query_summary=query_summary,
            concept_summary=concept_summary,
        )
        case_result = _evaluate_case(case, query_summary, response, concept_summary)
        case_result["gap_plan"] = {
            "grounded_fraction": float(gap_plan.get("grounded_fraction", 0.0)),
            "unsupported_terms": list(gap_plan.get("unsupported_terms") or []),
        }
        cases.append(case_result)

    pass_count = sum(1 for case in cases if bool(case["pass"]))
    query_count = len(cases)
    rss_after = _process_rss_mb()
    device_report = trainer.model.memory_store.device_report()
    return {
        "admit_source_episodes": bool(admit_source_episodes),
        "pass": bool(query_count and pass_count / query_count >= 0.80),
        "query_pass_count": int(pass_count),
        "query_total_count": int(query_count),
        "query_pass_rate": float(pass_count / max(1, query_count)),
        "feed_latency_ms": float(feed_latency_ms),
        "query_latency_ms_total": float(sum(query_latencies)),
        "query_latency_ms_mean": float(statistics.mean(query_latencies)) if query_latencies else 0.0,
        "query_latency_ms_p95": _p95(query_latencies),
        "tokens_processed": int(feed_summary.get("tokens_processed", 0)),
        "memory_buffer_size": int(feed_summary.get("memory_buffer_size", 0)),
        "source_memory_admission_report": dict(
            feed_summary.get("source_memory_admission_report") or {}
        ),
        "cases": cases,
        "device_report": device_report,
        "process_rss_mb_before": rss_before,
        "process_rss_mb_after": rss_after,
        "process_rss_mb_delta": (
            None if rss_before is None or rss_after is None else float(rss_after - rss_before)
        ),
        "cuda_memory_mb_before": cuda_before,
        "cuda_memory_mb_after": _cuda_memory_mb(),
    }


def run_source_episode_admission_benchmark(
    *,
    output: Path,
    scenario: str = "simple_animals",
    seed: int = 7,
) -> dict[str, Any]:
    baseline = _run_arm(
        scenario=scenario,
        seed=seed,
        admit_source_episodes=False,
    )
    bounded = _run_arm(
        scenario=scenario,
        seed=seed,
        admit_source_episodes=True,
    )
    baseline_rate = float(baseline["query_pass_rate"])
    bounded_rate = float(bounded["query_pass_rate"])
    report = {
        "artifact_kind": "marulho_source_episode_admission_benchmark",
        "surface": "bounded_feed_source_episode_admission_benchmark.v1",
        "scenario": str(scenario),
        "seed": int(seed),
        "quality_metric": "meaning_grounding_pass_rate_and_expected_term_recovery",
        "latency_metric": "explicit_feed_admission_latency_plus_query_readout_latency",
        "selection_criteria": "deduped explicit feed sentence units with bounded source episode budget",
        "memory_budget": {
            "candidate_episode_budget_entries": 32,
            "source_episode_max_chars": 240,
            "source_payload_char_budget": 32 * 240,
        },
        "runtime_truth": {
            "runs_live_tick": False,
            "runs_every_token": False,
            "global_candidate_scan": False,
            "global_score_scan": False,
            "language_reasoning": False,
            "hidden_language_reasoning": False,
            "archival_storage_device": "cpu",
            "active_computation_device": bounded["source_memory_admission_report"].get(
                "active_computation_device"
            ),
            "all_archival_tensors_cpu": bool(
                bounded["device_report"].get("all_archival_tensors_cpu")
            ),
        },
        "baseline": baseline,
        "bounded": bounded,
        "comparison": {
            "baseline_query_pass_rate": baseline_rate,
            "bounded_query_pass_rate": bounded_rate,
            "query_pass_rate_delta": float(bounded_rate - baseline_rate),
            "baseline_query_pass_count": int(baseline["query_pass_count"]),
            "bounded_query_pass_count": int(bounded["query_pass_count"]),
            "feed_latency_ms_delta": float(
                bounded["feed_latency_ms"] - baseline["feed_latency_ms"]
            ),
            "query_latency_ms_mean_delta": float(
                bounded["query_latency_ms_mean"] - baseline["query_latency_ms_mean"]
            ),
            "bounded_improves_quality": bool(bounded_rate > baseline_rate),
        },
    }
    report["pass"] = bool(
        report["comparison"]["bounded_improves_quality"]
        and bounded_rate >= 0.80
        and not report["runtime_truth"]["runs_live_tick"]
        and not report["runtime_truth"]["runs_every_token"]
        and not report["runtime_truth"]["global_candidate_scan"]
        and bool(report["runtime_truth"]["all_archival_tensors_cpu"])
    )
    write_json_file(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", default="simple_animals")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    report = run_source_episode_admission_benchmark(
        output=args.output,
        scenario=args.scenario,
        seed=args.seed,
    )
    print(
        f"[source_episode_admission] scenario={report['scenario']} "
        f"pass={report['pass']} "
        f"baseline={report['comparison']['baseline_query_pass_rate']:.3f} "
        f"bounded={report['comparison']['bounded_query_pass_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
