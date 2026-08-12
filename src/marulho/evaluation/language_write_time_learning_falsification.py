"""V59 source-native write-time learning capacity falsifier."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.evaluation.artifact_io import sha256_json
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_source_native_write_time_learning_falsification.v1"
ARTIFACT_KIND = "marulho_source_native_write_time_learning_falsification"
ADVANCE_DECISION = "advance_v59_write_time_learning_to_compact_meta_learner"
RETIRE_DECISION = "retire_v59_naive_source_only_gradient_memory"
DEFAULT_CHECKPOINT = Path(
    "reports/language_scaling/v39-answer-objective-qualified-100m-218m-20260810.pt"
)
DEFAULT_MANIFEST = Path(
    "reports/language_curriculum/squad-v57-native-validation-256-20260812.json"
)
DEFAULT_OUTPUT = Path(
    "reports/language_scaling/write-time-learning-v59-20260812.json"
)
EXPECTED_PANEL_SHA256 = (
    "185a9963bd28d53f04d075cc54937e0d6ca75ffc7719ac5979359ca1ee84e94f"
)


@dataclass(frozen=True)
class V59Config:
    panel_case_count: int = 64
    expected_panel_title_count: int = 22
    context_length: int = 72
    write_epochs: int = 4
    learning_rate: float = 1.0e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    generation_tokens: int = 16
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3
    minimum_true_exact_answers: int = 16
    minimum_true_control_margin: int = 12
    maximum_mismatched_exact_answers: int = 8
    minimum_oracle_exact_answers: int = 24
    minimum_true_loss_improvement_fraction: float = 0.90
    maximum_total_wall_seconds: float = 2400.0
    precision: str = "bfloat16"
    execution_backend: str = "pytorch_eager"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}")
    return dict(payload)


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).casefold()))


def select_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    case_count: int,
) -> tuple[tuple[dict[str, Any], ...], str]:
    by_title: dict[str, list[dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        by_title.setdefault(str(row["title"]), []).append(row)
    for title_rows in by_title.values():
        title_rows.sort(key=lambda row: str(row["case_id"]))
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < int(case_count):
        added = 0
        for title in sorted(by_title):
            title_rows = by_title[title]
            if depth < len(title_rows):
                selected.append(title_rows[depth])
                added += 1
                if len(selected) == int(case_count):
                    break
        if added == 0:
            raise ValueError("Manifest cannot fill the requested round-robin panel")
        depth += 1
    case_ids = [str(row["case_id"]) for row in selected]
    digest = hashlib.sha256(
        json.dumps(case_ids, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return tuple(selected), digest


def source_windows(
    source_text: str,
    tokenizer: LanguageTokenizer,
    *,
    context_length: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    ids = tokenizer.encode(
        f"Context: {source_text}",
        add_bos=True,
        add_eos=True,
    )
    window = int(context_length)
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for start in range(0, len(ids) - 1, window):
        segment = ids[start : start + window + 1]
        if len(segment) < 2:
            continue
        rows.append(
            (
                torch.tensor(segment[:-1], dtype=torch.long).unsqueeze(0),
                torch.tensor(segment[1:], dtype=torch.long).unsqueeze(0),
            )
        )
    if not rows:
        raise ValueError("Source produces no next-token write window")
    return tuple(rows)


def _question_prompt(row: Mapping[str, Any]) -> str:
    return f"Question: {str(row['question']).strip()}\nAnswer: "


@torch.no_grad()
def _generate_answer(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
    config: V59Config,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    prompt = torch.tensor(prompt_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    output = model.generate(
        prompt,
        max_new_tokens=config.generation_tokens,
        eos_id=tokenizer.eos_id,
        repetition_penalty=config.repetition_penalty,
        no_repeat_ngram_size=config.no_repeat_ngram_size,
    )
    generated = output["generated_ids"].detach().cpu().reshape(-1).tolist()
    continuation_ids = generated[len(prompt_ids) :]
    continuation = tokenizer.decode(continuation_ids)
    accepted = {_normalized(value) for value in row["answers"]}
    exact = _normalized(continuation) in accepted
    return {
        "case_id": str(row["case_id"]),
        "title": str(row["title"]),
        "question": str(row["question"]),
        "answers": [str(value) for value in row["answers"]],
        "continuation_text": continuation,
        "continuation_ids": [int(value) for value in continuation_ids],
        "exact_answer_match": exact,
        "generation_new_token_count": int(output["new_token_count"]),
    }


def _state_exact(
    model: MarulhoLanguageModel,
    baseline: Mapping[str, torch.Tensor],
) -> bool:
    observed = model.state_dict()
    mismatches = [
        torch.count_nonzero(observed[name] != expected)
        for name, expected in baseline.items()
    ]
    return int(torch.stack(mismatches).sum().item()) == 0


def _gradient_audit(model: MarulhoLanguageModel) -> dict[str, Any]:
    by_parameter: dict[str, bool] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        by_parameter[name] = bool(
            gradient is not None and torch.count_nonzero(gradient).item() > 0
        )
    return {
        "tensor_count": len(by_parameter),
        "nonzero_tensor_count": sum(by_parameter.values()),
        "all_trainable_tensors_nonzero": bool(by_parameter)
        and all(by_parameter.values()),
        "by_parameter": by_parameter,
    }


def _source_loss(
    model: MarulhoLanguageModel,
    windows: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, int]:
    weighted: list[torch.Tensor] = []
    positions = 0
    for inputs, targets in windows:
        device_inputs = inputs.to(model.device)
        device_targets = targets.to(model.device)
        result = model.next_token_loss(
            device_inputs,
            device_targets,
            collect_telemetry=False,
            return_evidence=False,
        )
        count = int(device_targets.numel())
        weighted.append(result["loss"] * count)
        positions += count
    return torch.stack(weighted).sum() / max(1, positions), positions


def _adapt_one_case(
    model: MarulhoLanguageModel,
    *,
    row: Mapping[str, Any],
    source_field: str,
    tokenizer: LanguageTokenizer,
    config: V59Config,
    collect_gradient_audit: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    source = str(row[source_field])
    windows = source_windows(
        source,
        tokenizer,
        context_length=config.context_length,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=config.adamw_betas,
        weight_decay=config.weight_decay,
        fused=model.device.type == "cuda",
    )
    epoch_losses: list[float] = []
    total_positions = 0
    optimizer_steps = 0
    model.train()
    started = time.perf_counter()
    for _epoch in range(config.write_epochs):
        weighted_loss = 0.0
        epoch_positions = 0
        for inputs, targets in windows:
            optimizer.zero_grad(set_to_none=True)
            device_inputs = inputs.to(model.device)
            device_targets = targets.to(model.device)
            result = model.next_token_loss(
                device_inputs,
                device_targets,
                collect_telemetry=False,
                return_evidence=False,
            )
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            count = int(device_targets.numel())
            weighted_loss += float(loss.detach()) * count
            epoch_positions += count
            optimizer_steps += 1
        epoch_losses.append(weighted_loss / max(1, epoch_positions))
        total_positions += epoch_positions
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    gradients = _gradient_audit(model) if collect_gradient_audit else None
    del optimizer
    return (
        {
            "source_field": source_field,
            "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "source_character_count": len(source),
            "source_token_count": sum(int(target.numel()) for _, target in windows),
            "write_window_count": len(windows),
            "write_epochs": config.write_epochs,
            "optimizer_steps": optimizer_steps,
            "processed_positions": total_positions,
            "epoch_losses": epoch_losses,
            "loss_improved": epoch_losses[-1] < epoch_losses[0],
            "write_seconds": elapsed,
            "positions_per_second": total_positions / elapsed,
        },
        gradients,
    )


def _evaluate_no_write(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    panel: Sequence[Mapping[str, Any]],
    config: V59Config,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = [_generate_answer(model, tokenizer, row, config) for row in panel]
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    exact = sum(bool(row["exact_answer_match"]) for row in rows)
    return {
        "arm_name": "no_write",
        "case_count": len(rows),
        "exact_answer_count": exact,
        "exact_answer_accuracy": exact / max(1, len(rows)),
        "wall_seconds": elapsed,
        "rows": rows,
    }


def _evaluate_write_arm(
    model: MarulhoLanguageModel,
    *,
    baseline_state: Mapping[str, torch.Tensor],
    tokenizer: LanguageTokenizer,
    panel: Sequence[Mapping[str, Any]],
    source_field: str,
    arm_name: str,
    config: V59Config,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    reset_exact_count = 0
    final_gradients: dict[str, Any] | None = None
    total_positions = 0
    total_steps = 0
    write_seconds = 0.0
    for index, row in enumerate(panel):
        model.load_state_dict(dict(baseline_state), strict=True)
        reset_exact = _state_exact(model, baseline_state)
        reset_exact_count += int(reset_exact)
        adaptation, gradients = _adapt_one_case(
            model,
            row=row,
            source_field=source_field,
            tokenizer=tokenizer,
            config=config,
            collect_gradient_audit=index + 1 == len(panel),
        )
        if gradients is not None:
            final_gradients = gradients
        generation = _generate_answer(model, tokenizer, row, config)
        rows.append(
            {
                **generation,
                "reset_exact": reset_exact,
                "write": adaptation,
            }
        )
        total_positions += int(adaptation["processed_positions"])
        total_steps += int(adaptation["optimizer_steps"])
        write_seconds += float(adaptation["write_seconds"])
        if (index + 1) % 8 == 0:
            exact_so_far = sum(bool(item["exact_answer_match"]) for item in rows)
            print(
                f"[v59] {arm_name} {index + 1}/{len(panel)} "
                f"exact={exact_so_far}",
                flush=True,
            )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    exact = sum(bool(row["exact_answer_match"]) for row in rows)
    improved = sum(bool(row["write"]["loss_improved"]) for row in rows)
    return {
        "arm_name": arm_name,
        "source_field": source_field,
        "case_count": len(rows),
        "exact_answer_count": exact,
        "exact_answer_accuracy": exact / max(1, len(rows)),
        "source_loss_improved_count": improved,
        "source_loss_improved_fraction": improved / max(1, len(rows)),
        "reset_exact_count": reset_exact_count,
        "all_resets_exact": reset_exact_count == len(rows),
        "optimizer_steps": total_steps,
        "processed_positions": total_positions,
        "write_seconds": write_seconds,
        "wall_seconds": elapsed,
        "write_positions_per_second": total_positions / max(write_seconds, 1.0e-9),
        "final_gradients": final_gradients,
        "rows": rows,
    }


def _parent_probe(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    row: Mapping[str, Any],
) -> torch.Tensor:
    ids = tokenizer.encode(_question_prompt(row), add_eos=False)
    probe = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        return model(probe, collect_telemetry=False)["logits"].detach().cpu()


def run_v59(
    *,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_path: str | Path = DEFAULT_OUTPUT,
    device: str = "cuda",
) -> dict[str, Any]:
    config = V59Config()
    runtime_device = torch.device(device)
    if runtime_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("V59 is frozen as a CUDA experiment")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    total_started = time.perf_counter()

    checkpoint = Path(checkpoint_path)
    manifest_file = Path(manifest_path)
    checkpoint_sha_before = _sha256_file(checkpoint)
    manifest = _load_json(manifest_file)
    panel, panel_sha = select_panel(
        tuple(manifest["cases"]),
        case_count=config.panel_case_count,
    )
    if panel_sha != EXPECTED_PANEL_SHA256:
        raise ValueError(f"V59 panel hash mismatch: {panel_sha}")
    title_count = len({str(row["title"]) for row in panel})
    if title_count != config.expected_panel_title_count:
        raise ValueError(f"V59 panel title count is {title_count}")

    parent, tokenizer, parent_payload = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if int(parent.context_length) != config.context_length:
        raise ValueError("V59 requires the exact context-72 V39 parent")
    parent.eval()
    parent_state_sha_before = language_model_state_sha256(parent)
    tokenizer_hash_before = tokenizer.vocabulary_hash()
    parent_logits_before = _parent_probe(parent, tokenizer, panel[0])

    adaptive, adaptive_tokenizer, _ = load_language_model_checkpoint(
        checkpoint,
        map_location="cpu",
    )
    if adaptive_tokenizer.vocabulary_hash() != tokenizer_hash_before:
        raise ValueError("Adaptive copy tokenizer differs from V39")
    adaptive = adaptive.to(device=runtime_device, dtype=torch.bfloat16)
    baseline_state = {
        name: value.detach().clone()
        for name, value in adaptive.state_dict().items()
    }
    if not _state_exact(adaptive, baseline_state):
        raise RuntimeError("Initial adaptive BF16 state is not self-exact")
    torch.cuda.reset_peak_memory_stats(runtime_device)

    print("[v59] evaluating exact no-write baseline", flush=True)
    no_write = _evaluate_no_write(adaptive, tokenizer, panel, config)
    print("[v59] evaluating mismatched-source writes", flush=True)
    mismatched = _evaluate_write_arm(
        adaptive,
        baseline_state=baseline_state,
        tokenizer=tokenizer,
        panel=panel,
        source_field="mismatched_source_text",
        arm_name="mismatched_write",
        config=config,
    )
    print("[v59] evaluating true-source writes", flush=True)
    true_write = _evaluate_write_arm(
        adaptive,
        baseline_state=baseline_state,
        tokenizer=tokenizer,
        panel=panel,
        source_field="source_text",
        arm_name="true_write",
        config=config,
    )
    print("[v59] evaluating oracle-short writes", flush=True)
    oracle = _evaluate_write_arm(
        adaptive,
        baseline_state=baseline_state,
        tokenizer=tokenizer,
        panel=panel,
        source_field="oracle_source_text",
        arm_name="oracle_short_write",
        config=config,
    )
    adaptive.load_state_dict(dict(baseline_state), strict=True)
    final_reset_exact = _state_exact(adaptive, baseline_state)
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated(runtime_device))
    del adaptive, baseline_state
    torch.cuda.empty_cache()

    checkpoint_sha_after = _sha256_file(checkpoint)
    parent_state_sha_after = language_model_state_sha256(parent)
    tokenizer_hash_after = tokenizer.vocabulary_hash()
    parent_logits_after = _parent_probe(parent, tokenizer, panel[0])
    parent_checks = {
        "checkpoint_file_exact": checkpoint_sha_before == checkpoint_sha_after,
        "state_exact": parent_state_sha_before == parent_state_sha_after,
        "tokenizer_exact": tokenizer_hash_before == tokenizer_hash_after,
        "sample_logits_exact": torch.equal(parent_logits_before, parent_logits_after),
        "final_adaptive_reset_exact": final_reset_exact,
    }
    total_wall_seconds = time.perf_counter() - total_started
    true_exact = int(true_write["exact_answer_count"])
    no_write_exact = int(no_write["exact_answer_count"])
    mismatched_exact = int(mismatched["exact_answer_count"])
    oracle_exact = int(oracle["exact_answer_count"])
    control_exact = max(no_write_exact, mismatched_exact)
    checks = {
        "minimum_true_exact_answers": true_exact
        >= config.minimum_true_exact_answers,
        "minimum_true_control_margin": true_exact - control_exact
        >= config.minimum_true_control_margin,
        "maximum_mismatched_exact_answers": mismatched_exact
        <= config.maximum_mismatched_exact_answers,
        "minimum_oracle_exact_answers": oracle_exact
        >= config.minimum_oracle_exact_answers,
        "minimum_true_loss_improvement_fraction": float(
            true_write["source_loss_improved_fraction"]
        )
        >= config.minimum_true_loss_improvement_fraction,
        "all_resets_exact": all(
            bool(arm["all_resets_exact"])
            for arm in (mismatched, true_write, oracle)
        ),
        "complete_final_gradients": all(
            bool(arm["final_gradients"]["all_trainable_tensors_nonzero"])
            for arm in (mismatched, true_write, oracle)
        ),
        "maximum_total_wall_seconds": total_wall_seconds
        <= config.maximum_total_wall_seconds,
        "parent_fidelity": all(parent_checks.values()),
        "panel_hash_exact": panel_sha == EXPECTED_PANEL_SHA256,
    }
    passed = all(checks.values())
    decision = ADVANCE_DECISION if passed else RETIRE_DECISION
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "owned_by_marulho": True,
        "external_llm_used": False,
        "decision": decision,
        "configuration": asdict(config),
        "experiment_contract_sha256": sha256_json(
            {
                "surface": SURFACE,
                "configuration": asdict(config),
                "checkpoint_sha256": checkpoint_sha_before,
                "manifest_contract_sha256": manifest["contract_sha256"],
                "panel_sha256": panel_sha,
            }
        ),
        "data": {
            "manifest_path": str(manifest_file),
            "manifest_sha256": _sha256_file(manifest_file),
            "manifest_contract_sha256": manifest["contract_sha256"],
            "panel_case_count": len(panel),
            "panel_title_count": title_count,
            "panel_case_ids_sha256": panel_sha,
            "panel_case_ids": [str(row["case_id"]) for row in panel],
            "write_inputs_exclude_question": True,
            "write_inputs_exclude_answer": True,
            "write_inputs_exclude_span": True,
            "write_inputs_exclude_labels": True,
        },
        "parent": {
            "path": str(checkpoint),
            "checkpoint_sha256_before": checkpoint_sha_before,
            "checkpoint_sha256_after": checkpoint_sha_after,
            "state_sha256_before": parent_state_sha_before,
            "state_sha256_after": parent_state_sha_after,
            "tokenizer_hash_before": tokenizer_hash_before,
            "tokenizer_hash_after": tokenizer_hash_after,
            "metadata": parent_payload.get("metadata", {}),
            "checks": parent_checks,
        },
        "arms": {
            "no_write": no_write,
            "mismatched_write": mismatched,
            "true_write": true_write,
            "oracle_short_write": oracle,
        },
        "runtime": {
            "total_wall_seconds": total_wall_seconds,
            "peak_cuda_bytes": peak_cuda_bytes,
        },
        "gate": {
            "passed": passed,
            "checks": checks,
            "observed": {
                "no_write_exact_answers": no_write_exact,
                "mismatched_write_exact_answers": mismatched_exact,
                "true_write_exact_answers": true_exact,
                "oracle_short_write_exact_answers": oracle_exact,
                "true_control_margin": true_exact - control_exact,
                "true_loss_improvement_fraction": true_write[
                    "source_loss_improved_fraction"
                ],
            },
            "thresholds": asdict(config),
        },
        "checkpoint": {
            "transient_case_states_saved": False,
            "durable_candidate_saved": False,
            "policy": "reset_per_case_then_discard",
        },
    }
    write_json_report_with_readme(output_path, report)
    print(
        f"[v59] decision={decision} no={no_write_exact} mismatch={mismatched_exact} "
        f"true={true_exact} oracle={oracle_exact}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run_v59(
        checkpoint_path=arguments.checkpoint,
        manifest_path=arguments.manifest,
        output_path=arguments.output,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
