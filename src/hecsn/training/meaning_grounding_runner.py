from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from hecsn.config.model_config import HECSNConfig
from hecsn.gap_planner import plan_query_gaps
from hecsn.interaction import EvidenceResponder
from hecsn.reporting.io import write_json_file
from hecsn.semantics import ConceptStore
from hecsn.training.query_runner import build_query_result
from hecsn.training.runner_utils import set_seed
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer
from hecsn.data.rtf_encoder import RTFEncoder


def _benchmark_config() -> HECSNConfig:
    return HECSNConfig(
        n_columns=24,
        column_latent_dim=32,
        bootstrap_tokens=0,
        memory_capacity=192,
        eta_competitive=0.05,
        eta_decay=0.0,
        input_weight_blend=0.0,
        micro_sleep_interval_tokens=10**9,
        deep_sleep_interval_tokens=10**9,
        enable_context_layer=True,
        enable_abstraction_layer=True,
        enable_binding_layer=True,
        learned_chunk_feature_mode="concat",
    )


def meaning_grounding_benchmark_config() -> HECSNConfig:
    return _benchmark_config()


def _scenario_payload(name: str) -> dict[str, Any]:
    if name == "simple_animals":
        corpus = "\n".join(
            [
                "a cat purrs when it feels safe.",
                "cats rest indoors.",
                "cats chase mice at night.",
                "a dog guards the house and barks at strangers.",
                "dogs wag their tails when familiar people arrive home.",
            ]
        )
        return {
            "scenario": name,
            "feed_text": "\n".join([corpus] * 14),
            "queries": [
                {
                    "name": "cat-purrs",
                    "query": "What purrs when it feels safe?",
                    "expected_terms": ["cat"],
                    "expected_concept_terms": ["cat", "cats"],
                    "forbidden_concept_terms": ["dog", "dogs"],
                    "expect_response": True,
                },
                {
                    "name": "dog-guards",
                    "query": "What guards the house and barks at strangers?",
                    "expected_terms": ["dog"],
                    "expected_concept_terms": ["dog", "dogs"],
                    "forbidden_concept_terms": ["cat", "cats"],
                    "expect_response": True,
                },
                {
                    "name": "cat-composition",
                    "query": "Where do cats rest and what do they chase at night?",
                    "expected_terms": ["indoors", "mice"],
                    "expected_concept_terms": ["cat", "cats"],
                    "forbidden_concept_terms": ["dog", "dogs"],
                    "expect_response": True,
                },
                {
                    "name": "unsupported-ocean",
                    "query": "What hums beneath the ocean floor?",
                    "expected_terms": [],
                    "expected_mode": "insufficient_evidence",
                    "expect_response": False,
                },
            ],
        }

    if name == "mixed_world":
        corpus = "\n".join(
            [
                "mercury is the closest planet to the sun.",
                "volcanoes release ash and lava during eruptions.",
                "octopuses solve puzzles and open jars.",
                "rainbows form when sunlight passes through water droplets.",
                "libraries lend books and provide quiet reading rooms.",
                "moss grows on damp forest stones.",
                "a violin produces music when a bow moves across its strings.",
            ]
        )
        return {
            "scenario": name,
            "feed_text": "\n".join([corpus] * 16),
            "queries": [
                {
                    "name": "octopus-tools",
                    "query": "What opens jars and solves puzzles?",
                    "expected_terms": ["octopuses"],
                    "expected_concept_terms": ["octopuses"],
                    "forbidden_unsupported_terms": ["opens", "solves"],
                    "expect_response": True,
                },
                {
                    "name": "rainbow-formation",
                    "query": "What forms when sunlight passes through water droplets?",
                    "expected_terms": ["rainbows"],
                    "expected_concept_terms": ["rainbows"],
                    "forbidden_unsupported_terms": ["forms"],
                    "expect_response": True,
                },
                {
                    "name": "library-composition",
                    "query": "Which place lends books and what kind of rooms does it provide?",
                    "expected_terms": ["books"],
                    "expected_phrases": ["reading rooms"],
                    "expected_concept_terms": ["libraries", "library"],
                    "forbidden_unsupported_terms": ["place", "kind", "lends", "provide"],
                    "expect_response": True,
                },
                {
                    "name": "planet-volcano-composition",
                    "query": "What is closest to the sun and what do volcanoes release?",
                    "expected_terms": ["mercury", "ash", "lava"],
                    "expected_mode": "grounded_synthesis",
                    "forbidden_unsupported_terms": ["closest", "volcanoes", "release", "sun"],
                    "expect_response": True,
                },
                {
                    "name": "unsupported-submarine",
                    "query": "What powers the submarine engine?",
                    "expected_terms": [],
                    "expected_mode": "insufficient_evidence",
                    "expect_response": False,
                },
            ],
        }

    raise ValueError(f"Unsupported meaning benchmark scenario: {name}")


def meaning_grounding_scenario_payload(name: str) -> dict[str, Any]:
    return _scenario_payload(name)


def run_meaning_grounding_benchmark(
    *,
    output_dir: Path,
    scenario: str = "simple_animals",
    seed: int = 7,
) -> dict[str, Any]:
    set_seed(seed)
    payload = _scenario_payload(scenario)
    cfg = _benchmark_config()
    trainer = HECSNTrainer(HECSNModelLite(cfg), cfg)
    encoder = RTFEncoder.from_config(cfg)
    concept_store = ConceptStore()
    responder = EvidenceResponder(min_similarity=0.0, min_token_coverage=0.25)

    result = build_query_result(
        trainer=trainer,
        checkpoint=Path(f"benchmark://{scenario}"),
        metadata={"benchmark": "meaning_grounding", "scenario": scenario},
        encoder=encoder,
        query_text_resolved=None,
        feed_text_resolved=str(payload["feed_text"]),
        context_text=None,
        top_k_candidates=6,
        top_k_memories=8,
        top_chars=8,
        compare_context_a=None,
        compare_context_b=None,
    )

    queries: list[dict[str, Any]] = []
    passed = True
    for case in payload["queries"]:
        query_result = build_query_result(
            trainer=trainer,
            checkpoint=Path(f"benchmark://{scenario}"),
            metadata={"benchmark": "meaning_grounding", "scenario": scenario},
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

        evidence_text = " ".join(
            str(item.get("text", ""))
            for item in response.get("selected_evidence") or []
        ).lower()
        response_text = str(response.get("response_text", "")).lower()
        expected_terms = [str(term).lower() for term in case.get("expected_terms") or []]
        expected_phrases = [str(term).lower() for term in case.get("expected_phrases") or []]
        expected_concept_terms = [str(term).lower() for term in case.get("expected_concept_terms") or []]
        forbidden_concept_terms = [str(term).lower() for term in case.get("forbidden_concept_terms") or []]
        forbidden_unsupported_terms = [str(term).lower() for term in case.get("forbidden_unsupported_terms") or []]
        expected_mode = str(case.get("expected_mode", "")).strip()
        grounded_terms = [term for term in expected_terms if term in evidence_text or term in response_text]
        grounded_phrases = [term for term in expected_phrases if term in evidence_text or term in response_text]
        episodes = list(query_summary.get("memory_episodes") or [])
        top_concept_terms = [
            str(term).lower()
            for term in (
                ((concept_summary.get("concepts") or [{}])[0].get("top_terms") or [])
                if concept_summary.get("concepts")
                else []
            )
        ]
        concept_pass = (
            (not expected_concept_terms or any(term in top_concept_terms for term in expected_concept_terms))
            and all(term not in top_concept_terms for term in forbidden_concept_terms)
        )
        unsupported_response_terms = [str(term).lower() for term in response.get("unsupported_terms") or []]
        unsupported_pass = all(term not in unsupported_response_terms for term in forbidden_unsupported_terms)
        episode_evidence_present = any(
            len(str(item.get("text", "")).strip()) > len(str(item.get("raw_window", "")).strip()) + 8
            for item in episodes
        )

        if bool(case.get("expect_response")):
            case_pass = (
                response.get("response_mode") != "insufficient_evidence"
                and bool(grounded_terms)
                and len(grounded_phrases) == len(expected_phrases)
                and episode_evidence_present
                and any(not bool(item.get("fragmentary")) for item in response.get("selected_evidence") or [])
                and concept_pass
                and unsupported_pass
                and (not expected_mode or str(response.get("response_mode")) == expected_mode)
            )
        else:
            case_pass = response.get("response_mode") == "insufficient_evidence"
            if expected_mode:
                case_pass = case_pass and str(response.get("response_mode")) == expected_mode
            concept_pass = True
            unsupported_pass = True

        passed = passed and bool(case_pass)
        queries.append(
            {
                "name": str(case["name"]),
                "query": str(case["query"]),
                "expected_terms": expected_terms,
                "expected_phrases": expected_phrases,
                "expected_concept_terms": expected_concept_terms,
                "forbidden_concept_terms": forbidden_concept_terms,
                "forbidden_unsupported_terms": forbidden_unsupported_terms,
                "expected_mode": expected_mode or None,
                "grounded_terms": grounded_terms,
                "grounded_phrases": grounded_phrases,
                "top_concept_terms": top_concept_terms,
                "concept_pass": bool(concept_pass),
                "unsupported_pass": bool(unsupported_pass),
                "pass": bool(case_pass),
                "response": response,
                "query_summary": query_summary,
                "concept_summary": concept_summary,
                "gap_plan": gap_plan,
                "episode_evidence_present": bool(episode_evidence_present),
            }
        )

    concept_snapshot = concept_store.snapshot()
    concept_separation_pass = bool(
        concept_snapshot.get("concept_count", 0) >= 2
        and all(bool(item.get("concept_pass")) for item in queries if bool(item.get("expected_concept_terms")))
    )
    query_pass_count = sum(1 for q in queries if q.get("pass"))
    query_pass_rate = query_pass_count / max(len(queries), 1)
    summary = {
        "benchmark": "meaning_grounding",
        "scenario": scenario,
        "seed": int(seed),
        "token_count": int(trainer.token_count),
        "memory_buffer_size": int(len(trainer.model.memory_store.slow_buffer)),
        "feed_summary": result.get("feed_summary"),
        "queries": queries,
        "concept_snapshot": concept_snapshot,
        "concept_separation_gate": {
            "pass": concept_separation_pass,
            "threshold": "grounded queries keep distinct concept evidence while the store retains at least two concepts",
        },
        "meaning_grounding_gate": {
            "pass": bool(query_pass_rate >= 0.80 and concept_separation_pass),
            "threshold": ">=80% of benchmark queries pass grounded-response and concept-separation expectations",
            "query_pass_rate": float(query_pass_rate),
            "query_pass_count": int(query_pass_count),
            "query_total_count": int(len(queries)),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_file(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the maintained meaning-grounding benchmark.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where summary.json should be written.",
    )
    parser.add_argument(
        "--scenario",
        default="simple_animals",
        help="Maintained benchmark scenario to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic runs.",
    )
    args = parser.parse_args()
    summary = run_meaning_grounding_benchmark(
        output_dir=args.output_dir,
        scenario=str(args.scenario),
        seed=int(args.seed),
    )
    print(
        f"[meaning_grounding] scenario={summary['scenario']} "
        f"pass={summary['meaning_grounding_gate']['pass']} "
        f"token_count={summary['token_count']}"
    )


if __name__ == "__main__":
    main()
