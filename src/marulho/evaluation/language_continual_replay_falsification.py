"""V38 continual relation learning with matched general-language replay."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_general_context_falsification import (
    GeneralContextFalsificationConfig,
    _prepare_data,
    _training_config,
    model_state_sha256,
)
from marulho.evaluation.language_matched_support import (
    grouped_staged_batch,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    evaluate_relation_binding_cases_batched,
)
from marulho.evaluation.language_training_experiment import (
    _prepare_language_loss_backend,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    evaluate_language_model,
    load_language_model_checkpoint,
    save_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon


SURFACE = "marulho_continual_replay_falsification.v1"
ARTIFACT_KIND = "marulho_continual_replay_falsification"
FOCUSED_ARM = "relation100"
REPLAY_ARMS = ("relation50_replay50", "relation20_replay80")
ARM_FRACTIONS = {FOCUSED_ARM: 1.0, REPLAY_ARMS[0]: 0.5, REPLAY_ARMS[1]: 0.2}
ADVANCE_DECISION = "advance_v38_continual_replay_checkpoint"
RETIRE_REPLAY_DECISION = "retire_v38_replay_test_parameter_isolation"
REDESIGN_DOMAIN_DECISION = "redesign_v38_relation_objective_no_free_learning"
INVALID_DECISION = "invalid_v38_continual_replay_evidence"
V35R_CHECKPOINT_SHA256 = (
    "48bfe82a70d9c537f10dc6d898c3cf18906716bd90acfefb7089ccd30477d9df"
)


@dataclass(frozen=True)
class ContinualReplayConfig:
    token_budget: int = 16_773_120
    sequence_length: int = 72
    microbatch_size: int = 32
    microbatches_per_optimizer_step: int = 8
    eval_batches: int = 16
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 38_121
    model_seed: int = 38_131
    sample_bytes_per_train_source: int = 128 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "inductor"
    compile_loss_tolerance: float = 1.0e-3
    minimum_free_relation_accuracy: float = 0.50
    minimum_candidate_relation_accuracy: float = 0.80
    maximum_general_loss_regression: float = 0.10

    @property
    def physical_batch_size(self) -> int:
        return int(self.microbatch_size) * int(self.microbatches_per_optimizer_step)


def select_continual_replay(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    initial_relation: Mapping[str, Any],
    initial_general_loss: float,
    config: ContinualReplayConfig,
) -> tuple[str | None, str]:
    if set(arms) != set(ARM_FRACTIONS):
        return None, INVALID_DECISION
    if not all(bool(row["all_parameters_received_final_gradient"]) for row in arms.values()):
        return None, INVALID_DECISION
    qualified: list[tuple[float, float, str]] = []
    for name in REPLAY_ARMS:
        row = arms[name]
        free = float(row["relation"]["generation_exact_accuracy"])
        ranked = float(row["relation"]["accuracy"])
        regression = float(row["heldout"]["heldout_loss"]) - float(
            initial_general_loss
        )
        if (
            free >= float(config.minimum_free_relation_accuracy)
            and ranked >= float(config.minimum_candidate_relation_accuracy)
            and regression <= float(config.maximum_general_loss_regression)
        ):
            qualified.append((free, -regression, name))
    if qualified:
        return max(qualified)[2], ADVANCE_DECISION
    focused = arms[FOCUSED_ARM]["relation"]
    initial_free = float(initial_relation["generation_exact_accuracy"])
    if (
        float(focused["generation_exact_accuracy"])
        >= float(config.minimum_free_relation_accuracy)
        and float(focused["generation_exact_accuracy"]) > initial_free
    ):
        return None, RETIRE_REPLAY_DECISION
    return None, REDESIGN_DOMAIN_DECISION


def _data_config(
    config: ContinualReplayConfig,
    *,
    relation_fraction: float,
) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.microbatch_size),
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=8,
        relation_case_limit=0,
        relation_fraction=float(relation_fraction),
        learning_rate=float(config.learning_rate),
        minimum_learning_rate_fraction=float(config.minimum_learning_rate_fraction),
        warmup_fraction=float(config.warmup_fraction),
        weight_decay=float(config.weight_decay),
        gradient_clip=float(config.gradient_clip),
        precision=str(config.precision),
        data_seed=int(config.data_seed),
        model_seed=int(config.model_seed),
        sample_bytes_per_train_source=int(config.sample_bytes_per_train_source),
        sample_bytes_per_eval_source=int(config.sample_bytes_per_eval_source),
        sample_range_count=int(config.sample_range_count),
        schedule_mode=str(config.schedule_mode),
        execution_backend=str(config.execution_backend),
        compile_loss_tolerance=float(config.compile_loss_tolerance),
        width=768,
        layers=10,
        heads=12,
        mlp_ratio=4.0,
    )


def _optimizer_builder(model, config):
    return build_language_muon(
        model,
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        adamw_betas=(float(config.adam_beta1), float(config.adam_beta2)),
        per_head_attention_qkv=False,
    )


def run_continual_replay_falsification(
    *,
    checkpoint_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    report_output_path: str | Path,
    checkpoint_output_path: str | Path,
    config: ContinualReplayConfig = ContinualReplayConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda" or int(config.physical_batch_size) != 256:
        raise ValueError("V38 requires CUDA and the V36 physical batch of 256")
    if int(config.token_budget) % (256 * int(config.sequence_length)) != 0:
        raise ValueError("V38 token budget must divide into complete optimizer steps")
    checkpoint = Path(checkpoint_path)
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("V38 requires exactly two replay and two eval sources")
    expected = {
        checkpoint: V35R_CHECKPOINT_SHA256,
        Path(relation_corpus_path): "2cb66bc468ad244d1c0846f0b1b06e1ec2c2d3ad754bab2483dda51b12d7bdc3",
        Path(relation_cases_path): "5506af0f51a2e0744b237bcebf2c90862a84c9cc4a69d74814909d2c466ce85a",
        Path(general_train_paths[0]): "034a3a00ea86ec097b913f6002485a6081c6adb2b66c14ddc82be7d57b13751c",
        Path(general_train_paths[1]): "7b6f41e3b3d2c1871d0124dc19f212713e3c8136e9f66cb462c845354e267aa7",
        Path(general_eval_paths[0]): "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a",
        Path(general_eval_paths[1]): "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491",
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V38 source hash differs for {path.name}")

    prepared_by_arm = {}
    for name, fraction in ARM_FRACTIONS.items():
        print(f"[V38] preparing {name}", flush=True)
        prepared = _prepare_data(
            tokenizer_checkpoint_path=checkpoint,
            relation_corpus_path=relation_corpus_path,
            relation_cases_path=relation_cases_path,
            general_train_paths=general_train_paths,
            general_eval_paths=general_eval_paths,
            sequence_length=int(config.sequence_length),
            batch_size=int(config.microbatch_size),
            config=_data_config(config, relation_fraction=fraction),
            device=resolved,
        )
        if len(prepared.schedule) != len(set(prepared.schedule)):
            raise ValueError(f"V38 {name} schedule repeats a source batch")
        prepared_by_arm[name] = prepared

    model, tokenizer, metadata = load_language_model_checkpoint(checkpoint, map_location="cpu")
    model = model.to(resolved)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_hash = model_state_sha256(model)
    reference_prepared = prepared_by_arm[FOCUSED_ARM]
    initial_heldout = evaluate_language_model(model, reference_prepared.eval_batches)
    initial_relation = evaluate_relation_binding_cases_batched(
        model, tokenizer, reference_prepared.cases, batch_size=8
    )
    example = grouped_staged_batch(
        reference_prepared.staged,
        start=0,
        count=int(config.microbatches_per_optimizer_step),
        device=resolved,
    )
    training_config = _training_config(
        _data_config(config, relation_fraction=1.0),
        sequence_length=int(config.sequence_length),
        batch_size=int(config.physical_batch_size),
    )
    print("[V38] compiling shared Transformer graph", flush=True)
    training_loss, execution = _prepare_language_loss_backend(model, example, training_config)

    rows: dict[str, dict[str, Any]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name, fraction in ARM_FRACTIONS.items():
        print(f"[V38] training {name}", flush=True)
        rows[name] = run_matched_training_arm(
            name,
            architecture="v35r_causal_transformer_continual_replay",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution=execution,
            allocated_compile_seconds=float(execution["compile_seconds"]) / len(ARM_FRACTIONS),
            prepared=prepared_by_arm[name],
            training_config=training_config,
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=8,
            model_seed=int(config.model_seed),
            device=resolved,
            progress_prefix="V38",
            optimizer_builder=_optimizer_builder,
            optimizer_warmup_steps=3,
            microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
            extra_row={"relation_fraction": fraction, "initial_heldout": initial_heldout},
        )
        if name in REPLAY_ARMS:
            states[name] = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    selected, decision = select_continual_replay(
        rows,
        initial_relation=initial_relation,
        initial_general_loss=float(initial_heldout["heldout_loss"]),
        config=config,
    )
    for row in rows.values():
        row["general_loss_regression"] = float(row["heldout"]["heldout_loss"]) - float(
            initial_heldout["heldout_loss"]
        )
        row["free_relation_gain"] = float(row["relation"]["generation_exact_accuracy"]) - float(
            initial_relation["generation_exact_accuracy"]
        )

    fidelity: dict[str, Any] = {"performed": False}
    checkpoint_saved = False
    if selected is not None:
        model.load_state_dict(states[selected], strict=True)
        selected_hash = model_state_sha256(model)
        output = save_language_model_checkpoint(
            checkpoint_output_path,
            model,
            tokenizer,
            metadata={
                **metadata,
                "architecture": "v35r_causal_transformer_continual_replay_v38",
                "parent_checkpoint_sha256": V35R_CHECKPOINT_SHA256,
                "selected_arm": selected,
                "cumulative_tokens": 201_335_040 + int(config.token_budget),
                "external_llm_used": False,
            },
        )
        restored, restored_tokenizer, _restored_metadata = load_language_model_checkpoint(
            output, map_location="cpu"
        )
        fidelity = {
            "performed": True,
            "checkpoint_sha256": sha256_file(output),
            "model_state_sha256_before_save": selected_hash,
            "model_state_sha256_after_reload": model_state_sha256(restored),
            "model_state_exact": model_state_sha256(restored) == selected_hash,
            "tokenizer_exact": restored_tokenizer.vocabulary_hash() == tokenizer.vocabulary_hash(),
        }
        checkpoint_saved = bool(fidelity["model_state_exact"] and fidelity["tokenizer_exact"])
        if not checkpoint_saved:
            raise RuntimeError("V38 checkpoint fidelity failed")

    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "created_at_unix": time.time(),
        "external_llm_used": False,
        "parent_checkpoint": {
            "path": str(checkpoint),
            "sha256": V35R_CHECKPOINT_SHA256,
            "initial_state_sha256": initial_hash,
        },
        "config": asdict(config),
        "initial_heldout": initial_heldout,
        "initial_relation": initial_relation,
        "arms": rows,
        "selected_arm": selected,
        "decision": decision,
        "checkpoint_saved": checkpoint_saved,
        "checkpoint_fidelity": fidelity,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path, report, title="MARULHO V38 Continual Replay Falsification"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--relation-corpus", required=True)
    parser.add_argument("--relation-cases", required=True)
    parser.add_argument("--general-train", action="append", required=True)
    parser.add_argument("--general-eval", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--checkpoint-output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = run_continual_replay_falsification(
        checkpoint_path=args.checkpoint,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        report_output_path=args.report,
        checkpoint_output_path=args.checkpoint_output,
        device=args.device,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
