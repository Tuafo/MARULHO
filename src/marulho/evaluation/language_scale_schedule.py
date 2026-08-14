"""Materialize V80's balanced billion-position source/row schedule."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from marulho.evaluation.language_quality_continuation import (
    ROOT,
    _atomic_json,
    file_sha256,
)
from marulho.evaluation.language_scale_corpus_materialization import (
    SURFACE as CORPUS_SURFACE,
    _atomic_torch_save,
    _token_tensor_sha256,
)


SURFACE = "marulho_language_scale_schedule.v80"
SCHEDULE_SEED = 80_121
SOURCE_ROW_SEEDS = {
    "fineweb_edu": 80_122,
    "cosmopedia_v2": 80_123,
    "dclm_edu": 80_124,
}
SOURCE_NAMES = ("fineweb_edu", "cosmopedia_v2", "dclm_edu")
SOURCE_SLOT_COUNTS = {
    "fineweb_edu": 419_430,
    "cosmopedia_v2": 314_573,
    "dclm_edu": 314_573,
}
TOTAL_SLOTS = 1_048_576
POSITIONS_PER_DOCUMENT = 960
TOTAL_POSITIONS = TOTAL_SLOTS * POSITIONS_PER_DOCUMENT
SOURCE_ARTIFACTS = {
    "fineweb_edu": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-fineweb-edu-train-20260814.pt",
        "dc182d9d8da5bcf70d727cc64ef239269ef46d0498478a9546b209543cbce73b",
        "203750d238058d93426db243d0e3ee02b466a719d2392ce72464ce5b70017e8f",
        58_999,
    ),
    "cosmopedia_v2": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-cosmopedia-v2-train-20260814.pt",
        "24d000c88f65a554ca2d6d38a0826b2146ac63fad7fd24ebb218d41e86ed3871",
        "a13f2d07d9a284dd4332fdd9066f156ffa723da3151d294746341cf834fd5573",
        62_298,
    ),
    "dclm_edu": (
        ROOT
        / "reports/language_curriculum/"
        "v80-scale-dclm-edu-train-20260814.pt",
        "72cbb6ba6c0e9723e7b52b27b380bf50041709b9a946e3ac68e8069956a07a99",
        "71730172e0d74d277efac157a8062cb21074c75a943282ed444d4d00bed6971f",
        150_910,
    ),
}


def _balanced_rows(*, documents: int, slots: int, seed: int) -> torch.Tensor:
    if documents < 1 or slots < 1:
        raise ValueError("V80 balanced rows require positive counts")
    generator = torch.Generator().manual_seed(seed)
    chunks: list[torch.Tensor] = []
    full_cycles, remainder = divmod(slots, documents)
    for _ in range(full_cycles):
        chunks.append(torch.randperm(documents, generator=generator))
    if remainder:
        chunks.append(torch.randperm(documents, generator=generator)[:remainder])
    return torch.cat(chunks).to(dtype=torch.int32)


def _schedule_sha256(source_ids: torch.Tensor, row_ids: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(source_ids.contiguous().view(torch.uint8).numpy().tobytes())
    digest.update(row_ids.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_source_audits() -> dict[str, Any]:
    audits: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        path, expected_artifact_hash, expected_token_hash, expected_documents = (
            SOURCE_ARTIFACTS[name]
        )
        artifact_hash = file_sha256(path)
        if artifact_hash != expected_artifact_hash:
            raise RuntimeError(f"V80 {name} artifact changed: {artifact_hash}")
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        tokens: torch.Tensor = artifact["tokens"]
        checks = {
            "surface_exact": artifact.get("surface") == CORPUS_SURFACE,
            "source_exact": artifact.get("source_name") == name,
            "documents_exact": int(tokens.shape[0]) == expected_documents,
            "token_hash_exact": _token_tensor_sha256(tokens) == expected_token_hash,
            "artifact_token_hash_exact": artifact.get("token_sha256")
            == expected_token_hash,
            "external_llm_absent": artifact.get("external_llm_used") is False,
        }
        if not all(checks.values()):
            raise RuntimeError(f"V80 {name} source validation failed: {checks}")
        audits[name] = {
            "path": str(path),
            "sha256": artifact_hash,
            "token_sha256": expected_token_hash,
            "documents": expected_documents,
            "checks": checks,
        }
        del artifact, tokens
    return audits


def _exposure_report(rows: torch.Tensor, *, documents: int) -> dict[str, Any]:
    counts = torch.bincount(rows.long(), minlength=documents)
    histogram = Counter(int(value) for value in counts.tolist())
    return {
        "documents": documents,
        "slots": int(rows.numel()),
        "minimum_exposures": int(counts.min().item()),
        "maximum_exposures": int(counts.max().item()),
        "exposure_count_histogram": {
            str(key): value for key, value in sorted(histogram.items())
        },
        "balanced_within_one": int(counts.max().item() - counts.min().item()) <= 1,
    }


def materialize_schedule(*, schedule_output: Path, report_output: Path) -> dict[str, Any]:
    if schedule_output.exists() or report_output.exists():
        raise ValueError("V80 schedule output already exists")
    if sum(SOURCE_SLOT_COUNTS.values()) != TOTAL_SLOTS:
        raise RuntimeError("V80 source slot counts do not sum to the total")
    source_audits = _load_source_audits()
    base_source_ids = torch.cat(
        [
            torch.full(
                (SOURCE_SLOT_COUNTS[name],),
                index,
                dtype=torch.int8,
            )
            for index, name in enumerate(SOURCE_NAMES)
        ]
    )
    source_permutation = torch.randperm(
        TOTAL_SLOTS,
        generator=torch.Generator().manual_seed(SCHEDULE_SEED),
    )
    source_ids = base_source_ids[source_permutation].contiguous()
    row_ids = torch.empty(TOTAL_SLOTS, dtype=torch.int32)
    source_rows: dict[str, torch.Tensor] = {}
    exposures: dict[str, Any] = {}
    for index, name in enumerate(SOURCE_NAMES):
        rows = _balanced_rows(
            documents=int(source_audits[name]["documents"]),
            slots=SOURCE_SLOT_COUNTS[name],
            seed=SOURCE_ROW_SEEDS[name],
        )
        positions = torch.nonzero(source_ids == index, as_tuple=False).flatten()
        if int(positions.numel()) != SOURCE_SLOT_COUNTS[name]:
            raise RuntimeError(f"V80 {name} source-slot count changed")
        row_ids[positions] = rows
        source_rows[name] = rows
        exposures[name] = _exposure_report(
            rows,
            documents=int(source_audits[name]["documents"]),
        )
    schedule_hash = _schedule_sha256(source_ids, row_ids)
    first_hash = _schedule_sha256(source_ids[:1024], row_ids[:1024])
    last_hash = _schedule_sha256(source_ids[-1024:], row_ids[-1024:])
    payload = {
        "surface": SURFACE,
        "artifact_kind": "marulho_billion_position_schedule",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "source_names": SOURCE_NAMES,
        "source_ids": source_ids,
        "row_ids": row_ids,
        "schedule_sha256": schedule_hash,
        "first_1024_sha256": first_hash,
        "last_1024_sha256": last_hash,
        "source_slot_counts": SOURCE_SLOT_COUNTS,
        "source_artifact_sha256": {
            name: audit["sha256"] for name, audit in source_audits.items()
        },
        "schedule_seed": SCHEDULE_SEED,
        "source_row_seeds": SOURCE_ROW_SEEDS,
        "positions_per_document": POSITIONS_PER_DOCUMENT,
        "total_positions": TOTAL_POSITIONS,
    }
    _atomic_torch_save(schedule_output, payload)
    artifact_hash = file_sha256(schedule_output)
    restored = torch.load(schedule_output, map_location="cpu", weights_only=False)
    verification = {
        "surface_exact": restored.get("surface") == SURFACE,
        "source_names_exact": tuple(restored.get("source_names", ())) == SOURCE_NAMES,
        "source_ids_exact": torch.equal(restored["source_ids"], source_ids),
        "row_ids_exact": torch.equal(restored["row_ids"], row_ids),
        "schedule_hash_exact": restored.get("schedule_sha256") == schedule_hash,
        "first_hash_exact": restored.get("first_1024_sha256") == first_hash,
        "last_hash_exact": restored.get("last_1024_sha256") == last_hash,
        "slot_counts_exact": all(
            int(torch.count_nonzero(source_ids == index).item())
            == SOURCE_SLOT_COUNTS[name]
            for index, name in enumerate(SOURCE_NAMES)
        ),
        "row_bounds_exact": all(
            int(row_ids[source_ids == index].min().item()) >= 0
            and int(row_ids[source_ids == index].max().item())
            < int(source_audits[name]["documents"])
            for index, name in enumerate(SOURCE_NAMES)
        ),
        "exposures_balanced": all(
            bool(report["balanced_within_one"]) for report in exposures.values()
        ),
    }
    verification["passed"] = all(verification.values())
    if not verification["passed"]:
        raise RuntimeError(f"V80 schedule verification failed: {verification}")
    report = {
        "surface": SURFACE,
        "artifact_kind": "marulho_billion_position_schedule_report",
        "owned_by_marulho": True,
        "external_llm_used": False,
        "source_audits": source_audits,
        "configuration": {
            "source_names": list(SOURCE_NAMES),
            "source_slot_counts": SOURCE_SLOT_COUNTS,
            "total_slots": TOTAL_SLOTS,
            "positions_per_document": POSITIONS_PER_DOCUMENT,
            "total_positions": TOTAL_POSITIONS,
            "schedule_seed": SCHEDULE_SEED,
            "source_row_seeds": SOURCE_ROW_SEEDS,
            "source_slot_shuffle": "one_global_randperm",
            "row_schedule": "independent_full_randperm_cycles_then_remainder",
        },
        "schedule": {
            "path": str(schedule_output),
            "size_bytes": schedule_output.stat().st_size,
            "sha256": artifact_hash,
            "schedule_sha256": schedule_hash,
            "first_1024_sha256": first_hash,
            "last_1024_sha256": last_hash,
        },
        "exposures": exposures,
        "verification": verification,
        "decision": "freeze_v80_billion_position_schedule",
    }
    _atomic_json(report_output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = materialize_schedule(
        schedule_output=args.schedule,
        report_output=args.report,
    )
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "schedule_sha256": report["schedule"]["schedule_sha256"],
                "total_positions": report["configuration"]["total_positions"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
