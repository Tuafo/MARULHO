"""Source-visible heldout QA grounding audit for MARULHO language checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from uuid import uuid4

import torch

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.data.language_tokenizer import LANGUAGE_DOCUMENT_SEPARATOR
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_source_visible_grounding.v1"
MANIFEST_SURFACE = "marulho_squad_grounding_manifest.v1"
DATASET = "rajpurkar/squad"
CONFIG = "plain_text"
DEFAULT_SPLIT = "validation"
DATASET_SERVER = "https://datasets-server.huggingface.co"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _contains_answer(text: str, answers: Sequence[str]) -> bool:
    normalized = _normalized(text)
    return any(_normalized(answer) in normalized for answer in answers if _normalized(answer))


def _prompt(source: str, question: str) -> str:
    return f"Context: {source}\nQuestion: {question}\nAnswer:"


def _sentence_around_answer(context: str, answer_start: int, answer: str) -> str:
    start = max(0, int(answer_start))
    stop = min(len(context), start + len(answer))
    left = max(
        context.rfind(".", 0, start),
        context.rfind("?", 0, start),
        context.rfind("!", 0, start),
        context.rfind("\n", 0, start),
    )
    right_candidates = [
        position
        for marker in (".", "?", "!", "\n")
        if (position := context.find(marker, stop)) >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(context)
    return context[left + 1 : right].strip()


def _bounded_source_text(
    tokenizer: LanguageTokenizer,
    *,
    context: str,
    question: str,
    answer_start: int,
    answer: str,
    maximum_prompt_tokens: int,
) -> tuple[str, int] | None:
    sentence = _sentence_around_answer(context, answer_start, answer)
    local_answer = sentence.casefold().find(answer.casefold())
    if local_answer < 0:
        return None
    answer_stop = local_answer + len(answer)
    for radius in (512, 384, 256, 192, 160, 128, 96, 72, 56, 40, 28, 20, 12):
        left = max(0, local_answer - radius)
        right = min(len(sentence), answer_stop + radius)
        if left > 0:
            boundary = sentence.find(" ", left)
            left = boundary + 1 if 0 <= boundary < local_answer else left
        if right < len(sentence):
            boundary = sentence.rfind(" ", answer_stop, right)
            right = boundary if boundary > answer_stop else right
        source = sentence[left:right].strip()
        prompt = _prompt(source, question)
        token_count = len(tokenizer.encode(prompt, add_eos=False))
        if answer.casefold() in source.casefold() and token_count <= int(
            maximum_prompt_tokens
        ):
            return source, token_count
    return None


def build_squad_grounding_cases(
    rows: Sequence[Mapping[str, Any]],
    tokenizer: LanguageTokenizer,
    *,
    case_count: int,
    maximum_prompt_tokens: int,
    maximum_answer_tokens: int = 8,
    maximum_cases_per_title: int = 6,
) -> tuple[dict[str, Any], ...]:
    selected: list[dict[str, Any]] = []
    title_counts: dict[str, int] = {}
    for envelope in sorted(rows, key=lambda value: int(value["row_idx"])):
        row = dict(envelope["row"])
        title = str(row["title"])
        if title_counts.get(title, 0) >= int(maximum_cases_per_title):
            continue
        answers_payload = dict(row["answers"])
        answers = tuple(dict.fromkeys(str(value) for value in answers_payload["text"]))
        starts = tuple(int(value) for value in answers_payload["answer_start"])
        if not answers or not starts or not answers[0].strip():
            continue
        question = str(row["question"]).strip()
        if _contains_answer(question, answers):
            continue
        answer_ids = tokenizer.encode(answers[0], add_bos=False, add_eos=False)
        if not 1 <= len(answer_ids) <= int(maximum_answer_tokens):
            continue
        bounded = _bounded_source_text(
            tokenizer,
            context=str(row["context"]),
            question=question,
            answer_start=starts[0],
            answer=answers[0],
            maximum_prompt_tokens=int(maximum_prompt_tokens),
        )
        if bounded is None:
            continue
        source, prompt_token_count = bounded
        visible_answers = tuple(
            answer for answer in answers if _normalized(answer) in _normalized(source)
        )
        if not visible_answers:
            continue
        selected.append(
            {
                "case_id": str(row["id"]),
                "row_idx": int(envelope["row_idx"]),
                "title": title,
                "source_text": source,
                "question": question,
                "answers": list(visible_answers),
                "prompt": _prompt(source, question),
                "prompt_token_count": int(prompt_token_count),
                "question_only_prompt": f"Question: {question}\nAnswer:",
                "question_only_prompt_token_count": len(
                    tokenizer.encode(
                        f"Question: {question}\nAnswer:",
                        add_eos=False,
                    )
                ),
            }
        )
        title_counts[title] = title_counts.get(title, 0) + 1
        if len(selected) >= int(case_count):
            break
    if len(selected) < int(case_count):
        raise ValueError(
            f"only {len(selected)} valid grounding cases found; requested {case_count}"
        )
    for index, case in enumerate(selected):
        mismatch = None
        for offset in range(1, len(selected)):
            candidate = selected[(index + offset) % len(selected)]["source_text"]
            candidate_prompt = _prompt(str(candidate), str(case["question"]))
            candidate_token_count = len(
                tokenizer.encode(candidate_prompt, add_eos=False)
            )
            if (
                not _contains_answer(str(candidate), tuple(case["answers"]))
                and candidate_token_count <= int(maximum_prompt_tokens)
            ):
                mismatch = (str(candidate), candidate_prompt, candidate_token_count)
                break
        if mismatch is None:
            raise ValueError("could not construct answer-absent mismatched source")
        mismatch_source, mismatch_prompt, mismatch_token_count = mismatch
        case["mismatched_source_text"] = mismatch_source
        case["mismatched_prompt"] = mismatch_prompt
        case["mismatched_prompt_token_count"] = int(mismatch_token_count)
    return tuple(selected)


def _fetch_page(
    offset: int,
    length: int,
    *,
    split: str,
) -> tuple[int, dict[str, Any], str, str]:
    query = urlencode(
        {
            "dataset": DATASET,
            "config": CONFIG,
            "split": str(split),
            "offset": int(offset),
            "length": int(length),
        }
    )
    url = f"{DATASET_SERVER}/rows?{query}"
    request = Request(url, headers={"User-Agent": "MARULHO-source-grounding/1"})
    with urlopen(request, timeout=60) as response:
        raw = response.read()
    return int(offset), json.loads(raw), hashlib.sha256(raw).hexdigest(), url


def fetch_squad_rows(
    *,
    row_count: int,
    split: str = DEFAULT_SPLIT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    count = max(1, int(row_count))
    offsets = list(range(0, count, 100))
    with ThreadPoolExecutor(max_workers=min(4, len(offsets))) as executor:
        pages = list(
            executor.map(
                lambda offset: _fetch_page(
                    offset,
                    min(100, count - offset),
                    split=str(split),
                ),
                offsets,
            )
        )
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for offset, payload, sha256, url in sorted(pages):
        rows.extend(dict(value) for value in payload["rows"])
        provenance.append(
            {
                "offset": int(offset),
                "length": int(len(payload["rows"])),
                "response_sha256": sha256,
                "url": url,
            }
        )
    return rows, provenance


def materialize_squad_grounding_manifest(
    tokenizer: LanguageTokenizer,
    *,
    output_path: str | Path,
    case_count: int = 64,
    fetch_row_count: int = 2_000,
    maximum_prompt_tokens: int = 64,
    split: str = DEFAULT_SPLIT,
    maximum_cases_per_title: int = 6,
) -> dict[str, Any]:
    rows, pages = fetch_squad_rows(
        row_count=int(fetch_row_count),
        split=str(split),
    )
    cases = build_squad_grounding_cases(
        rows,
        tokenizer,
        case_count=int(case_count),
        maximum_prompt_tokens=int(maximum_prompt_tokens),
        maximum_cases_per_title=int(maximum_cases_per_title),
    )
    manifest = {
        "surface": MANIFEST_SURFACE,
        "dataset": DATASET,
        "config": CONFIG,
        "split": str(split),
        "dataset_server": DATASET_SERVER,
        "external_text_data": True,
        "external_model_used": False,
        "fetched_row_count": int(len(rows)),
        "case_count": int(len(cases)),
        "maximum_prompt_tokens": int(maximum_prompt_tokens),
        "maximum_cases_per_title": int(maximum_cases_per_title),
        "tokenizer_hash": tokenizer.vocabulary_hash(),
        "pages": pages,
        "cases": list(cases),
    }
    manifest["contract_sha256"] = _canonical_sha256(manifest)
    _write_json_atomic(output_path, manifest)
    return manifest


def materialize_squad_training_corpus(
    manifest: Mapping[str, Any],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write manifest prompts and first answers as separator-delimited documents."""

    cases = tuple(dict(value) for value in manifest["cases"])
    if not cases:
        raise ValueError("grounding training manifest contains no cases")
    documents = []
    for case in cases:
        answers = tuple(str(value).strip() for value in case["answers"])
        if not answers or not answers[0]:
            raise ValueError("grounding training case lacks a first answer")
        documents.append(f'{str(case["prompt"]).rstrip()} {answers[0]}')
    text = LANGUAGE_DOCUMENT_SEPARATOR.join(documents)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(output),
        "sha256": _sha256_file(output),
        "document_count": len(documents),
        "size_bytes": output.stat().st_size,
        "manifest_contract_sha256": str(manifest["contract_sha256"]),
        "separator": LANGUAGE_DOCUMENT_SEPARATOR,
    }


def load_squad_grounding_manifest(
    path: str | Path,
    tokenizer: LanguageTokenizer,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("surface") != MANIFEST_SURFACE:
        raise ValueError("grounding manifest surface differs")
    if payload.get("tokenizer_hash") != tokenizer.vocabulary_hash():
        raise ValueError("grounding manifest tokenizer differs")
    contract = str(payload.pop("contract_sha256"))
    if _canonical_sha256(payload) != contract:
        raise ValueError("grounding manifest contract hash differs")
    payload["contract_sha256"] = contract
    return payload


def _generate_prompts(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> list[str]:
    encoded = [tokenizer.encode(prompt, add_eos=False) for prompt in prompts]
    groups: dict[int, list[int]] = {}
    for index, token_ids in enumerate(encoded):
        groups.setdefault(len(token_ids), []).append(index)
    continuations = [""] * len(prompts)
    for prompt_length, indices in groups.items():
        prompt_batch = torch.tensor(
            [encoded[index] for index in indices],
            dtype=torch.long,
            device=model.device,
        )
        generated = model.generate(
            prompt_batch,
            max_new_tokens=int(max_new_tokens),
            eos_id=tokenizer.eos_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )["generated_ids"].detach().cpu()
        for row_index, case_index in enumerate(indices):
            continuations[case_index] = tokenizer.decode(
                [int(value) for value in generated[row_index, prompt_length:].tolist()]
            )
    return continuations


def _condition_report(
    cases: Sequence[Mapping[str, Any]],
    continuations: Sequence[str],
) -> dict[str, Any]:
    rows = []
    for case, continuation in zip(cases, continuations):
        answers = tuple(str(value) for value in case["answers"])
        exact = _contains_answer(continuation, answers)
        rows.append(
            {
                "case_id": str(case["case_id"]),
                "answers": list(answers),
                "continuation": str(continuation),
                "exact_answer_match": bool(exact),
            }
        )
    return {
        "case_count": len(rows),
        "exact_answer_count": sum(bool(row["exact_answer_match"]) for row in rows),
        "exact_answer_accuracy": (
            sum(bool(row["exact_answer_match"]) for row in rows) / max(1, len(rows))
        ),
        "rows": rows,
    }


def evaluate_source_grounding(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    manifest: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    output_path: str | Path,
    max_new_tokens: int = 16,
) -> dict[str, Any]:
    cases = tuple(dict(value) for value in manifest["cases"])
    if not cases:
        raise ValueError("grounding manifest contains no cases")
    state_before = language_model_state_sha256(model)
    model.eval()
    intact = _condition_report(
        cases,
        _generate_prompts(
            model,
            tokenizer,
            [str(case["prompt"]) for case in cases],
            max_new_tokens=int(max_new_tokens),
        ),
    )
    question_only = _condition_report(
        cases,
        _generate_prompts(
            model,
            tokenizer,
            [str(case["question_only_prompt"]) for case in cases],
            max_new_tokens=int(max_new_tokens),
        ),
    )
    mismatched = _condition_report(
        cases,
        _generate_prompts(
            model,
            tokenizer,
            [str(case["mismatched_prompt"]) for case in cases],
            max_new_tokens=int(max_new_tokens),
        ),
    )
    state_after = language_model_state_sha256(model)
    intact_accuracy = float(intact["exact_answer_accuracy"])
    stronger_control = max(
        float(question_only["exact_answer_accuracy"]),
        float(mismatched["exact_answer_accuracy"]),
    )
    source_gain = intact_accuracy - stronger_control
    validity = {
        "all_intact_prompts_fit_manifest_bound": all(
            int(case["prompt_token_count"]) <= int(manifest["maximum_prompt_tokens"])
            for case in cases
        ),
        "all_control_prompts_fit_manifest_bound": all(
            int(case["question_only_prompt_token_count"])
            <= int(manifest["maximum_prompt_tokens"])
            and int(case["mismatched_prompt_token_count"])
            <= int(manifest["maximum_prompt_tokens"])
            for case in cases
        ),
        "all_answers_visible_in_intact_source": all(
            _contains_answer(str(case["source_text"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_absent_from_question": all(
            not _contains_answer(str(case["question"]), tuple(case["answers"]))
            for case in cases
        ),
        "all_answers_absent_from_mismatched_source": all(
            not _contains_answer(
                str(case["mismatched_source_text"]), tuple(case["answers"])
            )
            for case in cases
        ),
        "model_state_immutable": state_before == state_after,
    }
    valid = all(bool(value) for value in validity.values())
    if not valid:
        decision = "invalid_v47_source_grounding_audit"
    elif intact_accuracy >= 0.25 and source_gain >= 0.10:
        decision = "v39_uses_visible_source_advance_to_continual_grounding"
    elif source_gain >= 0.05:
        decision = "weak_v39_source_use_advance_to_continual_grounding"
    else:
        decision = "v39_no_visible_source_use_train_grounding_with_replay"
    report = {
        "surface": SURFACE,
        "decision": decision,
        "valid": valid,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "external_text_data": True,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "manifest_path": str(manifest.get("path") or ""),
        "manifest_contract_sha256": str(manifest["contract_sha256"]),
        "case_count": len(cases),
        "max_new_tokens": int(max_new_tokens),
        "decode_control_scope": "generated_continuation_only",
        "validity": validity,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "intact_source": intact,
        "question_only": question_only,
        "mismatched_source": mismatched,
        "intact_gain_over_stronger_control": source_gain,
        "promotion_gate": {
            "minimum_intact_accuracy": 0.25,
            "minimum_gain_over_stronger_control": 0.10,
            "passed": bool(intact_accuracy >= 0.25 and source_gain >= 0.10),
        },
        "boundary": (
            "This audit measures source-visible extractive QA on heldout SQuAD. "
            "It does not prove long-term memory, open-domain factuality, or general reasoning."
        ),
    }
    write_json_report_with_readme(output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-count", type=int, default=64)
    parser.add_argument("--fetch-row-count", type=int, default=2_000)
    parser.add_argument("--maximum-prompt-tokens", type=int, default=64)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--maximum-cases-per-title", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--map-location", default="cuda")
    args = parser.parse_args()
    model, tokenizer, _metadata = load_language_model_checkpoint(
        args.checkpoint,
        map_location=args.map_location,
    )
    model = model.to(torch.device(args.map_location))
    if args.manifest.is_file():
        manifest = load_squad_grounding_manifest(args.manifest, tokenizer)
    else:
        manifest = materialize_squad_grounding_manifest(
            tokenizer,
            output_path=args.manifest,
            case_count=int(args.case_count),
            fetch_row_count=int(args.fetch_row_count),
            maximum_prompt_tokens=int(args.maximum_prompt_tokens),
            split=str(args.split),
            maximum_cases_per_title=int(args.maximum_cases_per_title),
        )
    manifest["path"] = str(args.manifest)
    report = evaluate_source_grounding(
        model,
        tokenizer,
        manifest,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_new_tokens=int(args.max_new_tokens),
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
