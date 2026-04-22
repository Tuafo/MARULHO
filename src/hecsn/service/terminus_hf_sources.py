"""Recommended Hugging Face datasets for the live Terminus runtime.

The current runtime uses a Hugging Face mixture for both background text and
real multimodal grounding:
- Wikipedia for fast encyclopedic grounding
- S2ORC ArXiv abstracts for dense scientific language
- FineWeb-Edu for broader educational coverage
- S1-MMAlign for real scientific image grounding
- AudioCaps for real audio grounding

NIM curriculum generation remains the targeted, gap-driven source of active
learning episodes. Curriculum-derived sensory hints still exist, but they are
now complemented by real Hugging Face visual/audio streams.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


CURRENT_TERMINUS_HF_SOURCE_BANK: tuple[dict[str, Any], ...] = (
    {
        "name": "wikipedia_en",
        "source": "wikimedia/wikipedia",
        "source_type": "hf",
        "hf_config": "20231101.en",
        "text_field": "text",
        "metadata": {
            "role": "encyclopedic_grounding",
            "label": "Wikipedia English 20231101",
            "url": "https://huggingface.co/datasets/wikimedia/wikipedia",
            "why": (
                "Stable factual prose gives Terminus a fast, structured background "
                "stream complementary to open-web educational text."
            ),
        },
    },
    {
        "name": "s2orc_arxiv_abstracts",
        "source": "AlgorithmicResearchGroup/s2orc_arxiv",
        "source_type": "hf",
        "text_field": "abstract",
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
