from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import socket
import tempfile
import threading
from typing import Any
from unittest.mock import patch

from hecsn.config.model_config import HECSNConfig
from hecsn.reporting.io import write_json_file
from hecsn.service.manager import HECSNServiceManager
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def _benchmark_config() -> HECSNConfig:
    return HECSNConfig(
        n_columns=4,
        column_latent_dim=8,
        bootstrap_tokens=0,
        memory_capacity=96,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        enable_context_layer=True,
        enable_binding_layer=True,
    )


def _build_checkpoint(root: Path, *, test_case: str) -> Path:
    cfg = _benchmark_config()
    trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
    return save_trainer_checkpoint(
        root / "initial.pt",
        trainer,
        metadata={"benchmark": "terminus_long_horizon_autonomy", "test_case": test_case},
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _SilentSimpleHTTPRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:  # pragma: no cover - benchmark noise
        return None


def _scenario_steps() -> list[dict[str, Any]]:
    return [
        {
            "topic": "submarine",
            "query": "What reduces submarine buoyancy?",
            "critical_terms": ["ballast"],
            "topic_anchor_terms": ["submarine", "ballast"],
            "expected_provider": "wikipedia",
            "expected_source_prefix": "wikipedia_",
        },
        {
            "topic": "octopus",
            "query": "What do octopuses open?",
            "critical_terms": ["jars"],
            "topic_anchor_terms": ["octopus", "octopuses", "jars"],
            "expected_provider": "openalex",
            "expected_source_prefix": "openalex_",
        },
        {
            "topic": "submarine",
            "query": "What corrects submarine trim?",
            "critical_terms": ["trim"],
            "topic_anchor_terms": ["submarine", "trim", "ballast"],
            "expected_provider": "wikipedia",
            "expected_source_prefix": "wikipedia_",
        },
        {
            "topic": "octopus",
            "query": "What do octopuses use to open jars?",
            "critical_terms": ["arms"],
            "topic_anchor_terms": ["octopus", "octopuses", "arms", "jars"],
            "expected_provider": "openalex",
            "expected_source_prefix": "openalex_",
        },
    ]


def _write_scenario_files(root: Path) -> None:
    files = {
        "background.txt": "neutral background signal " * 40,
        "ballast_tank.txt": (
            "ballast tanks reduce submarine buoyancy and correct trim. "
            "submarines regulate buoyancy with ballast water inside ballast tanks. "
        )
        * 24,
        "library.txt": (
            "libraries lend books and provide quiet reading rooms. "
            "a library offers books, reading rooms, and quiet study space. "
        )
        * 24,
        "octopus.txt": (
            "octopuses solve puzzles and open jars with flexible arms. "
            "an octopus opens jars and manipulates objects with flexible arms. "
        )
        * 24,
        "volcano.txt": (
            "volcanoes release ash and lava during eruptions. "
            "an eruption sends ash and lava from a volcano. "
        )
        * 24,
        "garden.txt": "garden tomatoes need soil sunlight and watering. " * 24,
        "astronomy.txt": "astronomy studies planets observatories and telescope images. " * 24,
        "cable.txt": "submarine cables carry internet traffic between continents. " * 24,
    }
    for name, text in files.items():
        (root / name).write_text(text, encoding="utf-8")


def _topic_for_query(query: str) -> str | None:
    normalized = " ".join(str(query).lower().split())
    if any(term in normalized for term in ("submarine", "ballast", "buoyancy", "trim")):
        return "submarine"
    if any(term in normalized for term in ("octopus", "octopuses", "jar", "jars", "puzzle", "puzzles", "arms")):
        return "octopus"
    if any(term in normalized for term in ("library", "libraries", "books", "reading room", "reading rooms")):
        return "library"
    if any(term in normalized for term in ("volcano", "volcanoes", "ash", "lava", "eruption", "eruptions")):
        return "volcano"
    return None


def _search_payloads(content_port: int, *, provider: str, query: str) -> list[dict[str, object]]:
    topic = _topic_for_query(query)

    def _page(name: str, filename: str, summary: str, priority: float) -> dict[str, object]:
        return {
            "name": name,
            "source": f"http://127.0.0.1:{content_port}/{filename}",
            "source_type": "web",
            "summary": summary,
            "query_text": query,
            "catalog_priority": float(priority),
            "provider": provider,
        }

    distractors = [
        _page(
            f"{provider}_garden_source",
            "garden.txt",
            "garden tomatoes soil sunlight watering",
            0.35,
        ),
        _page(
            f"{provider}_astronomy_source",
            "astronomy.txt",
            "astronomy planets observatory telescope orbit",
            0.30,
        ),
    ]

    if provider == "wikipedia":
        if topic == "submarine":
            return [
                _page(
                    "wikipedia_ballast_tank",
                    "ballast_tank.txt",
                    "submarine buoyancy ballast tanks reduce buoyancy and correct trim",
                    0.95,
                ),
                *distractors,
            ]
        return [
            _page(
                "wikipedia_submarine_cable",
                "cable.txt",
                "submarine cables carry internet traffic",
                0.50,
            ),
            *distractors,
        ]

    if provider == "openalex":
        if topic == "octopus":
            return [
                _page(
                    "openalex_octopus_cognition",
                    "octopus.txt",
                    "octopuses solve puzzles and open jars with flexible arms",
                    0.96,
                ),
                *distractors,
            ]
        if topic == "volcano":
            return [
                _page(
                    "openalex_volcano_eruption",
                    "volcano.txt",
                    "volcanoes release ash and lava during eruptions",
                    0.96,
                ),
                *distractors,
            ]
        return [
            _page(
                "marine_archives",
                "astronomy.txt",
                "scientific archives and observatory records",
                0.45,
            ),
            *distractors,
        ]

    return [
        _page(
            "generic_report",
            "astronomy.txt",
            "technical report archive with broad scientific summaries",
            0.40,
        ),
        *distractors,
    ]


def _search_remote_provider(
    content_port: int,
    provider: str,
    query: str,
    *,
    result_limit: int,
    timeout_seconds: float,
) -> list[dict[str, object]]:
    del timeout_seconds
    return _search_payloads(content_port, provider=provider, query=query)[:result_limit]


def _top_concept_terms(result: dict[str, Any]) -> list[str]:
    concepts = list((result.get("concept_summary") or {}).get("concepts") or [])
    if not concepts:
        return []
    return [
        str(term).lower()
        for term in list(concepts[0].get("top_terms") or [])
        if str(term).strip()
    ]


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = {str(item).strip().lower() for item in left if str(item).strip()}
    right_set = {str(item).strip().lower() for item in right if str(item).strip()}
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set)) / float(len(union))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(item) for item in values)) / float(len(values))


def run_terminus_long_horizon_benchmark(*, output_dir: Path, seed: int = 7) -> dict[str, Any]:
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_scenario_files(root)
        checkpoint_path = _build_checkpoint(root, test_case="terminus_long_horizon_benchmark")
        content_port = _free_port()
        content_server = ThreadingHTTPServer(
            ("127.0.0.1", content_port),
            partial(_SilentSimpleHTTPRequestHandler, directory=str(root)),
        )
        content_thread = threading.Thread(target=content_server.serve_forever, daemon=True)
        content_thread.start()

        manager = HECSNServiceManager(
            checkpoint_path=checkpoint_path,
            trace_history_limit=200,
            trace_dir=root / "traces",
        )
        steps: list[dict[str, Any]] = []
        first_topic_results: dict[str, dict[str, Any]] = {}
        try:
            background_path = root / "background.txt"
            manager.configure_terminus(
                source_bank=[
                    {
                        "name": "background",
                        "source": str(background_path),
                        "source_type": "file",
                    }
                ],
                tick_tokens=16,
                sleep_interval_seconds=0.01,
                repeat_sources=True,
                autonomy={
                    "enabled": True,
                    "policy": "active",
                    "trigger_interval_tokens": 1,
                    "candidate_train_tokens": 64,
                    "probe_tokens": 32,
                    "acquisition_tokens": 96,
                    "acquisition_slots": 1,
                },
            )

            with patch(
                "hecsn.data.source_catalog._search_remote_provider",
                side_effect=partial(_search_remote_provider, content_port),
            ):
                for step_index, case in enumerate(_scenario_steps(), start=1):
                    manager._brain_recent_query_gaps.clear()
                    query_result = manager.query(
                        query_text=str(case["query"]),
                        top_k_memories=6,
                    )
                    tick_result = manager.terminus_tick(steps=1)
                    respond_result = manager.respond(
                        query_text=str(case["query"]),
                        top_k_memories=6,
                        max_evidence_items=3,
                        learn_mode="none",
                    )

                    pre_gap = dict(query_result.get("gap_plan") or {})
                    response = dict(respond_result.get("response") or {})
                    post_query_result = dict(respond_result.get("query_result") or {})
                    selected_evidence = list(response.get("selected_evidence") or [])
                    evidence_text = " ".join(str(item.get("text", "")) for item in selected_evidence).lower()
                    response_text = str(response.get("response_text", "")).lower()
                    critical_terms = [str(item).lower() for item in list(case.get("critical_terms") or [])]
                    topic_anchor_terms = [str(item).lower() for item in list(case.get("topic_anchor_terms") or [])]
                    expected_phrases = [str(item).lower() for item in list(case.get("expected_phrases") or [])]
                    expected_provider = str(case.get("expected_provider", "")).strip().lower()
                    expected_source_prefix = str(case.get("expected_source_prefix", "")).strip().lower()
                    grounded_terms = [term for term in critical_terms if term in evidence_text or term in response_text]
                    grounded_phrases = [term for term in expected_phrases if term in evidence_text or term in response_text]
                    query_terms = [
                        str(term).lower()
                        for term in list(pre_gap.get("query_terms") or [])
                        if str(term).strip()
                    ]
                    unsupported_terms = [str(term).lower() for term in list(response.get("unsupported_terms") or [])]
                    pre_answerability = max(0.0, min(1.0, float(pre_gap.get("grounded_fraction", 0.0))))
                    denominator = max(1, len(query_terms) or len(critical_terms))
                    post_answerability = max(
                        0.0,
                        min(1.0, 1.0 - float(len(unsupported_terms)) / float(denominator)),
                    )
                    autonomy = dict((tick_result.get("terminus_runtime") or {}).get("autonomy") or {})
                    acquisition = dict(autonomy.get("last_acquisition_summary") or {})
                    provider_curriculum = dict(autonomy.get("provider_curriculum") or {})
                    ranked_providers = list(provider_curriculum.get("ranked_providers") or [])
                    top_provider = str(ranked_providers[0]["provider"]) if ranked_providers else ""
                    concept_terms = _top_concept_terms(post_query_result)
                    acquired_sources = [str(source) for source in list(acquisition.get("acquired_sources") or [])]
                    retained_source_hit = bool(
                        expected_source_prefix
                        and any(
                            str(item.get("text", "")).strip().lower().startswith(expected_source_prefix)
                            for item in selected_evidence
                        )
                    )
                    anchor_present = bool(
                        any(term in evidence_text or term in response_text for term in topic_anchor_terms)
                        or any(term in concept_terms for term in topic_anchor_terms)
                        or any(source.lower().startswith(expected_source_prefix) for source in acquired_sources)
                        or retained_source_hit
                    )
                    critical_supported = all(term not in unsupported_terms for term in critical_terms)
                    supported = bool(
                        response.get("response_mode") != "insufficient_evidence"
                        and anchor_present
                        and critical_supported
                        and len(grounded_phrases) == len(expected_phrases)
                    )
                    record = {
                        "step_index": int(step_index),
                        "topic": str(case["topic"]),
                        "query": str(case["query"]),
                        "expected_provider": str(case["expected_provider"]),
                        "critical_terms": critical_terms,
                        "topic_anchor_terms": topic_anchor_terms,
                        "expected_phrases": expected_phrases,
                        "grounded_terms": grounded_terms,
                        "grounded_phrases": grounded_phrases,
                        "response_mode": str(response.get("response_mode", "")),
                        "response_text": str(response.get("response_text", "")),
                        "unsupported_terms": unsupported_terms,
                        "pre_grounded_fraction": float(pre_answerability),
                        "post_answerability": float(post_answerability),
                        "answerability_growth": float(post_answerability - pre_answerability),
                        "anchor_present": bool(anchor_present),
                        "critical_supported": bool(critical_supported),
                        "supported": bool(supported),
                        "tokens_trained_total": int(acquisition.get("tokens_trained_total", 0)),
                        "acquired_sources": acquired_sources,
                        "provider_curriculum_top": top_provider,
                        "provider_curriculum_snapshot": provider_curriculum,
                        "concept_terms": concept_terms,
                    }
                    if str(case["topic"]) in first_topic_results:
                        baseline = first_topic_results[str(case["topic"])]
                        record["revisit"] = True
                        record["concept_stability"] = float(
                            _jaccard(concept_terms, list(baseline.get("concept_terms") or []))
                        )
                        record["answerability_delta_vs_first"] = float(
                            post_answerability - float(baseline.get("post_answerability", 0.0))
                        )
                        record["provider_hit"] = bool(
                            top_provider == expected_provider
                            or any(source.lower().startswith(expected_source_prefix) for source in acquired_sources)
                            or retained_source_hit
                        )
                    else:
                        record["revisit"] = False
                        first_topic_results[str(case["topic"])] = {
                            "concept_terms": concept_terms,
                            "post_answerability": float(post_answerability),
                        }
                    steps.append(record)

            final_runtime = manager.terminus_status()["terminus_runtime"]
        finally:
            manager.close()
            content_server.shutdown()
            content_server.server_close()

    supported_steps = [step for step in steps if bool(step.get("supported"))]
    revisit_steps = [step for step in steps if bool(step.get("revisit"))]
    unique_topics = {str(step["topic"]) for step in steps}
    supported_topics = {str(step["topic"]) for step in supported_steps}
    total_tokens_trained = int(sum(int(step.get("tokens_trained_total", 0)) for step in steps))
    unique_acquired_sources = sorted(
        {
            str(source)
            for step in steps
            for source in list(step.get("acquired_sources") or [])
            if str(source).strip()
        }
    )
    metrics = {
        "step_count": int(len(steps)),
        "supported_step_count": int(len(supported_steps)),
        "supported_topic_coverage": float(
            0.0 if not unique_topics else len(supported_topics) / float(len(unique_topics))
        ),
        "answerability_growth_mean": float(
            _mean([float(step.get("answerability_growth", 0.0)) for step in steps])
        ),
        "concept_stability_mean": float(
            _mean([float(step.get("concept_stability", 0.0)) for step in revisit_steps])
        ),
        "revisit_retention_rate": float(
            _mean([1.0 if bool(step.get("supported")) else 0.0 for step in revisit_steps])
        ),
        "revisit_answerability_delta_mean": float(
            _mean([float(step.get("answerability_delta_vs_first", 0.0)) for step in revisit_steps])
        ),
        "revisit_provider_hit_rate": float(
            _mean([1.0 if bool(step.get("provider_hit")) else 0.0 for step in revisit_steps])
        ),
        "total_tokens_trained": int(total_tokens_trained),
        "mean_tokens_per_supported_step": float(
            0.0 if not supported_steps else total_tokens_trained / float(len(supported_steps))
        ),
        "tokens_per_supported_topic": float(
            0.0 if not supported_topics else total_tokens_trained / float(len(supported_topics))
        ),
        "topics_supported_per_96_tokens": float(
            0.0 if total_tokens_trained <= 0 else len(supported_topics) / float(total_tokens_trained / 96.0)
        ),
        "unique_acquired_source_count": int(len(unique_acquired_sources)),
    }
    long_horizon_gate = {
        "pass": bool(
            metrics["supported_topic_coverage"] >= 1.0
            and metrics["answerability_growth_mean"] >= 0.25
            and metrics["concept_stability_mean"] >= 0.20
            and metrics["revisit_retention_rate"] >= 1.0
            and metrics["revisit_answerability_delta_mean"] >= -0.25
            and metrics["revisit_provider_hit_rate"] >= 1.0
            and metrics["tokens_per_supported_topic"] <= 160.0
            and metrics["topics_supported_per_96_tokens"] >= 0.50
        ),
        "thresholds": {
            "supported_topic_coverage_min": 1.0,
            "answerability_growth_mean_min": 0.25,
            "concept_stability_mean_min": 0.20,
            "revisit_retention_rate_min": 1.0,
            "revisit_answerability_delta_mean_min": -0.25,
            "revisit_provider_hit_rate_min": 1.0,
            "tokens_per_supported_topic_max": 160.0,
            "topics_supported_per_96_tokens_min": 0.50,
        },
    }
    summary = {
        "benchmark": "terminus_long_horizon_autonomy",
        "seed": int(seed),
        "runtime_setup": {
            "tick_tokens": 16,
            "trigger_interval_tokens": 1,
            "candidate_train_tokens": 64,
            "probe_tokens": 32,
            "acquisition_tokens": 96,
            "acquisition_slots": 1,
            "default_candidate_bank": True,
            "default_catalog_providers": ["wikipedia", "arxiv", "openalex"],
        },
        "steps": steps,
        "metrics": metrics,
        "unique_acquired_sources": unique_acquired_sources,
        "final_runtime": final_runtime,
        "long_horizon_gate": long_horizon_gate,
    }
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the maintained long-horizon Terminus autonomy benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary.json should be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic runs.",
    )
    args = parser.parse_args()
    summary = run_terminus_long_horizon_benchmark(
        output_dir=args.output_dir,
        seed=int(args.seed),
    )
    print(
        f"[terminus_long_horizon] pass={summary['long_horizon_gate']['pass']} "
        f"supported_topic_coverage={summary['metrics']['supported_topic_coverage']:.3f} "
        f"revisit_provider_hit_rate={summary['metrics']['revisit_provider_hit_rate']:.3f}"
    )
    print(f"summary_json={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
