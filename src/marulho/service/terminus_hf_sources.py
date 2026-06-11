"""Recommended Hugging Face datasets for the live Terminus runtime.

The current runtime uses a Hugging Face mixture for both background text and
real multimodal grounding:
- OpenStax open textbooks for instructional grounding
- S2ORC ArXiv abstracts for dense scientific language
- FineWeb-Edu for broader educational coverage
- S1-MMAlign for real scientific image grounding
- AudioCaps for real audio grounding

Targeted learning should now prefer maintained real-source autonomy acquisition
over synthetic curriculum text generation. The maintained runtime now retunes
that acquisition cadence and budget under strong focus pressure/provider
alignment, routes passive background sources through the same focus-aware
selection discipline, persists source/provider utility on that maintained
path, calibrates that utility against grounded answer/action outcomes, credits
selected response evidence back to the source/provider provenance that
produced it, tracks delayed multi-turn consequences tied to later query
improvement on that same maintained path, applies contradiction/decay-aware
long-horizon utility penalties when later evidence regresses or is
contradicted, explicitly schedules recovery/forgiveness when later mixed
evidence repairs those earlier penalties, now cools/ages long-horizon
consequence state explicitly instead of relying only on bounded record limits,
and compactly aggregates repeated long-horizon consequence families instead of
keeping them only as separate near-duplicate records, now tracks
trajectory-sensitive family summaries instead of relying only on bounded
maxima, can split mixed long-horizon consequence families when their query
branches diverge, can remerge those split lineages when later evidence
re-aligns them, and now calibrates long-horizon consequence utility against
that grounded family summary instead of relying only on bounded family-state
scalars. Multimodal grounding lives on the maintained real Hugging Face
visual/audio streams.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any


CURRENT_TERMINUS_HF_SOURCE_BANK: tuple[dict[str, Any], ...] = (
    {
        "name": "open_textbooks",
        "source": "izumi-lab/open-text-books",
        "source_type": "hf",
        "text_field": "text",
        "topic_terms": [
            "open textbook peer reviewed education",
            "chemistry physics biology mathematics",
            "worked examples definitions exercises",
        ],
        "metadata": {
            "role": "open_textbook_grounding",
            "label": "OpenStax open textbooks",
            "url": "https://huggingface.co/datasets/izumi-lab/open-text-books",
            "why": (
                "Human-written open textbook prose is more instructional and "
                "worked-example dense than raw encyclopedic articles while "
                "remaining simple to stream."
            ),
        },
    },
    {
        "name": "s2orc_arxiv_abstracts",
        "source": "AlgorithmicResearchGroup/s2orc_arxiv",
        "source_type": "hf",
        "text_field": "abstract",
        "topic_terms": [
            "science research paper abstract",
            "technical engineering mathematics",
            "methods results analysis hypothesis",
        ],
        "metadata": {
            "role": "scientific_technical_depth",
            "label": "S2ORC ArXiv abstracts",
            "url": "https://huggingface.co/datasets/AlgorithmicResearchGroup/s2orc_arxiv",
            "why": (
                "Abstract-level scientific content is denser and cheaper to stream "
                "than full papers while still providing strong technical vocabulary."
            ),
        },
    },
    {
        "name": "fineweb_edu",
        "source": "HuggingFaceFW/fineweb-edu",
        "source_type": "hf",
        "hf_config": "sample-10BT",
        "text_field": "text",
        "topic_terms": [
            "education tutorial lesson explanation",
            "broad curriculum learning study",
            "general knowledge practice examples",
        ],
        "metadata": {
            "role": "broad_educational_background",
            "label": "FineWeb-Edu sample-10BT",
            "url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
            "why": (
                "Large public educational web corpus with strong diversity and a "
                "practical sample config for always-on runtime use."
            ),
        },
    },
)


CURRENT_TERMINUS_SENSORY_SOURCE_BANK: tuple[dict[str, Any], ...] = (
    {
        "name": "science_figures",
        "adapter": "s1_mmalign",
        "source": "ScienceOne-AI/S1-MMAlign",
        "split": "train",
        "year_prefixes": ["07", "08", "09"],
        "max_text_chars": 480,
        "topic_terms": [
            "scientific figure",
            "diagram plot graph chart",
            "spatial geometry contour lattice",
            "microscope molecule structure",
            "equation phase map",
        ],
        "metadata": {
            "role": "real_scientific_visual_grounding",
            "label": "S1-MMAlign early-year scientific figures",
            "url": "https://huggingface.co/datasets/ScienceOne-AI/S1-MMAlign",
            "why": (
                "Streamable real scientific figure images paired with recaptions; "
                "better aligned with Terminus than synthetic visual hints alone."
            ),
        },
    },
    {
        "name": "environmental_audio",
        "adapter": "audiocaps",
        "source": "OpenSound/AudioCaps",
        "split": "train",
        "sample_rate": 16000,
        "n_fft": 512,
        "max_text_chars": 240,
        "audio_candidates_per_item": 6,
        "topic_terms": [
            "audio sound acoustic noise",
            "environment ambient event vibration",
            "speech voice music rhythm",
            "water wind birds engine footsteps",
        ],
        "metadata": {
            "role": "real_audio_grounding",
            "label": "AudioCaps",
            "url": "https://huggingface.co/datasets/OpenSound/AudioCaps",
            "why": (
                "Natural audio-caption pairs provide real environmental sound grounding "
                "instead of narrow spoken-digit benchmarks."
            ),
        },
    },
)


CURRENT_TERMINUS_RUNTIME_DATASETS: tuple[dict[str, Any], ...] = tuple(
    [
        {
            "name": str(item["name"]),
            "type": "text",
            "path": f"hf://{item['source']}",
            "exists": True,
            "file_count": None,
            "hf_config": item.get("hf_config"),
            "text_field": item.get("text_field", "text"),
            "role": deepcopy(item.get("metadata", {})).get("role"),
            "description": deepcopy(item.get("metadata", {})).get("why"),
            "url": deepcopy(item.get("metadata", {})).get("url"),
        }
        for item in CURRENT_TERMINUS_HF_SOURCE_BANK
    ]
    + [
        {
            "name": str(item["name"]),
            "type": "image+text" if str(item.get("adapter")) == "s1_mmalign" else "audio+text",
            "path": f"hf://{item['source']}",
            "exists": True,
            "file_count": None,
            "role": deepcopy(item.get("metadata", {})).get("role"),
            "description": deepcopy(item.get("metadata", {})).get("why"),
            "url": deepcopy(item.get("metadata", {})).get("url"),
            "adapter": item.get("adapter"),
        }
        for item in CURRENT_TERMINUS_SENSORY_SOURCE_BANK
    ]
)


def current_runtime_source_bank() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in CURRENT_TERMINUS_HF_SOURCE_BANK]


def _runtime_autonomy_terms(*values: str) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).replace("_", " ").lower()
        for term in re.findall(r"[a-zA-Z][a-zA-Z'-]+", cleaned):
            token = term.strip().lower()
            if len(token) < 4 or token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return ordered


def current_runtime_autonomy_config() -> dict[str, Any]:
    catalog_entries: list[dict[str, Any]] = []
    for item in CURRENT_TERMINUS_HF_SOURCE_BANK:
        metadata = deepcopy(item.get("metadata", {}))
        summary_parts = [
            str(metadata.get("role", "")).replace("_", " ").strip(),
            str(metadata.get("label", "")).strip(),
            str(metadata.get("why", "")).strip(),
        ]
        summary = " ".join(part for part in summary_parts if part).strip()
        catalog_entries.append(
            {
                "name": str(item["name"]),
                "source": str(item["source"]),
                "source_type": "hf",
                "hf_config": item.get("hf_config"),
                "text_field": str(item.get("text_field", "text")),
                "summary": summary,
                "terms": _runtime_autonomy_terms(
                    str(item.get("name", "")),
                    str(metadata.get("role", "")),
                    str(metadata.get("label", "")),
                    *[str(term) for term in list(item.get("topic_terms") or [])],
                ),
            }
        )
    return {
        "enabled": True,
        "policy": "active",
        "candidate_bank": [
            {
                "name": "runtime_hf_registry",
                "catalog_mode": "semantic_registry",
                "catalog_limit": len(catalog_entries),
                "catalog_probe_pool_limit": len(catalog_entries),
                "catalog_entries": catalog_entries,
            }
        ],
        "trigger_interval_tokens": 1024,
        "candidate_train_tokens": 768,
        "probe_tokens": 96,
        "acquisition_tokens": 512,
    }


def current_runtime_sensory_config() -> dict[str, Any]:
    return {
        "enabled": True,
        "source_bank": [deepcopy(item) for item in CURRENT_TERMINUS_SENSORY_SOURCE_BANK],
        "episode_interval_tokens": 1536,
        "items_per_episode": 2,
        "base_windows_per_item": 4,
        "max_windows_per_item": 10,
        "confidence_window_gain": 3.0,
        "semantic_window_gain": 3.0,
        "item_retrieval_lookahead": 6,
        "item_retrieval_semantic_weight": 0.72,
        "modality_target_confidence": 0.70,
        "observation_salience": 0.82,
        "cooldown_seconds": 8.0,
        "repeat_sources": True,
    }


def current_runtime_datasets() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in CURRENT_TERMINUS_RUNTIME_DATASETS]


def future_multimodal_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "scicap",
            "source": "CrowdAILab/scicap",
            "modality": "image+caption+paragraph",
            "url": "https://huggingface.co/datasets/CrowdAILab/scicap",
            "why": "Smaller scientific figure-caption corpus that remains a useful fallback / benchmark.",
        }
    ]
