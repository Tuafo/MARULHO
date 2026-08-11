"""V42 tokenizer-trie role-contrastive continual-learning pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import torch

from marulho.evaluation.language_general_context_falsification import (
    GeneralContextFalsificationConfig,
    _prepare_data,
    _training_config,
)
from marulho.evaluation.language_matched_support import (
    grouped_staged_batch,
    run_matched_training_arm,
    sha256_file,
)
from marulho.evaluation.language_relation_binding_experiment import (
    COLORS,
    CONTAINERS,
    ENTITIES,
)
from marulho.evaluation.language_training_experiment import (
    _precision_context,
    _resolve_device,
)
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_answer_objective import answer_target_mask
from marulho.training.language_model import (
    language_model_state_sha256,
    load_language_model_checkpoint,
)
from marulho.training.language_muon import build_language_muon
from marulho.training.language_role_contrastive import (
    build_role_contrastive_branches,
    role_contrastive_answer_loss,
    prepare_role_contrastive_lookup,
    role_contrastive_lookup_active_mask,
)


SURFACE = "marulho_role_contrastive_falsification.v1"
ARTIFACT_KIND = "marulho_role_contrastive_falsification"
ARM_WEIGHTS = {
    "answer4_control": 0.0,
    "role_contrastive_025": 0.25,
    "role_contrastive_1": 1.0,
}
ROLE_GROUPS = {
    "entity": ENTITIES,
    "container": CONTAINERS,
    "color": COLORS,
    "event_polarity": ("Some", "No"),
}
ADVANCE_DECISION = "advance_v42_role_contrastive_full_confirmation"
RETIRE_DECISION = "retire_v42_role_contrastive_no_weak_binding_gain"
INVALID_DECISION = "invalid_v42_role_contrastive_pilot"
V39_CHECKPOINT_SHA256 = (
    "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d"
)
V39_REPORT_SHA256 = (
    "3b64d702ed2db458587c78316d34fe826138bef8d4d72b8093dc861d11289127"
)
SOURCE_SHA256 = {
    "relation_corpus": "2cb66bc468ad244d1c0846f0b1b06e1ec2c2d3ad754bab2483dda51b12d7bdc3",
    "relation_cases": "5506af0f51a2e0744b237bcebf2c90862a84c9cc4a69d74814909d2c466ce85a",
    "general_train_0": "034a3a00ea86ec097b913f6002485a6081c6adb2b66c14ddc82be7d57b13751c",
    "general_train_1": "7b6f41e3b3d2c1871d0124dc19f212713e3c8136e9f66cb462c845354e267aa7",
    "general_eval_0": "a4e00212ab6101ebb4e269068fae414d53a16bca063ba37038331c10e3cda64a",
    "general_eval_1": "e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491",
}


@dataclass(frozen=True)
class RoleContrastivePilotConfig:
    token_budget: int = 2_359_296
    sequence_length: int = 72
    microbatch_size: int = 32
    microbatches_per_optimizer_step: int = 8
    eval_batches: int = 16
    relation_eval_batch_size: int = 64
    relation_fraction: float = 0.50
    learning_rate: float = 3.0e-4
    minimum_learning_rate_fraction: float = 0.10
    warmup_fraction: float = 0.05
    weight_decay: float = 0.10
    gradient_clip: float = 1.0
    precision: str = "bfloat16"
    data_seed: int = 42_121
    model_seed: int = 42_131
    sample_bytes_per_train_source: int = 128 * 1024 * 1024
    sample_bytes_per_eval_source: int = 32 * 1024 * 1024
    sample_range_count: int = 16
    schedule_mode: str = "indexed_host"
    execution_backend: str = "eager"
    compile_loss_tolerance: float = 1.0e-3
    answer_weight: float = 4.0
    minimum_free_gain: float = 0.05
    minimum_weak_kind_gain: float = 0.05
    maximum_general_loss_regression: float = 0.03

    @property
    def physical_batch_size(self) -> int:
        return int(self.microbatch_size) * int(self.microbatches_per_optimizer_step)


def _data_config(config: RoleContrastivePilotConfig) -> GeneralContextFalsificationConfig:
    return GeneralContextFalsificationConfig(
        token_budget=int(config.token_budget),
        common_sequence_length=int(config.sequence_length),
        common_batch_size=int(config.microbatch_size),
        eval_batches=int(config.eval_batches),
        relation_eval_batch_size=int(config.relation_eval_batch_size),
        relation_case_limit=0,
        relation_fraction=float(config.relation_fraction),
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


def select_role_contrastive_candidate(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    config: RoleContrastivePilotConfig,
) -> tuple[str | None, str, dict[str, dict[str, float]]]:
    """Apply the frozen pilot gate relative to its trained control."""

    if set(arms) != set(ARM_WEIGHTS):
        return None, INVALID_DECISION, {}
    if not all(bool(row["all_parameters_received_final_gradient"]) for row in arms.values()):
        return None, INVALID_DECISION, {}
    control = arms["answer4_control"]
    control_free = float(control["relation"]["generation_exact_accuracy"])
    control_kinds = control["relation"]["generation_kind_accuracy"]
    control_loss = float(control["heldout"]["heldout_loss"])
    deltas: dict[str, dict[str, float]] = {}
    qualified: list[tuple[float, float, float, str]] = []
    for name in ("role_contrastive_025", "role_contrastive_1"):
        row = arms[name]
        kinds = row["relation"]["generation_kind_accuracy"]
        free_gain = float(row["relation"]["generation_exact_accuracy"]) - control_free
        ownership_gain = float(kinds["ownership"]) - float(control_kinds["ownership"])
        container_gain = float(kinds["container"]) - float(control_kinds["container"])
        weak_gain = max(ownership_gain, container_gain)
        general_regression = float(row["heldout"]["heldout_loss"]) - control_loss
        deltas[name] = {
            "free_accuracy_gain_over_control": free_gain,
            "ownership_accuracy_gain_over_control": ownership_gain,
            "container_accuracy_gain_over_control": container_gain,
            "maximum_weak_kind_gain_over_control": weak_gain,
            "general_loss_regression_over_control": general_regression,
        }
        if (
            free_gain >= float(config.minimum_free_gain)
            and weak_gain >= float(config.minimum_weak_kind_gain)
            and general_regression <= float(config.maximum_general_loss_regression)
        ):
            qualified.append((free_gain, weak_gain, -general_regression, name))
    if qualified:
        return max(qualified)[3], ADVANCE_DECISION, deltas
    return None, RETIRE_DECISION, deltas


def _prepare_shared_eager_loss(
    model,
    example_batch,
    *,
    marker_ids: torch.Tensor,
    eos_id: int,
    role_lookup,
    config: RoleContrastivePilotConfig,
):
    """Audit one eager function whose scalar weight keeps arms compute-matched."""

    def eager_loss(
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        contrastive_weight: torch.Tensor,
    ) -> torch.Tensor:
        return role_contrastive_answer_loss(
            model,
            input_ids,
            target_ids,
            contrastive_weight,
            marker_ids=marker_ids,
            eos_id=int(eos_id),
            answer_weight=float(config.answer_weight),
            lookup=role_lookup,
        )

    device_batch = example_batch.to(model.device)
    audit_weight = torch.tensor(1.0, dtype=torch.float32, device=model.device)
    torch.cuda.reset_peak_memory_stats(model.device)
    torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with _precision_context(model.device, config.precision):
        observed = eager_loss(
            device_batch.input_ids, device_batch.target_ids, audit_weight
        )
    observed.backward()
    torch.cuda.synchronize(model.device)
    warmup_seconds = time.perf_counter() - started
    if not bool(torch.isfinite(observed.detach())):
        raise RuntimeError("V42 eager role loss is non-finite")
    if not all(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("V42 eager role loss misses model gradients")
    model.zero_grad(set_to_none=True)
    execution = {
        "requested_backend": "eager",
        "effective_backend": "eager",
        "ordinary_step_backend": "eager_shared_loss",
        "compile_fullgraph": False,
        "compile_dynamic_shapes": False,
        "shared_across_all_arms": True,
        "scalar_weight_is_function_input": True,
        "compile_seconds": 0.0,
        "warmup_seconds": warmup_seconds,
        "warmup_peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(model.device)),
        "warmup_loss_parity": {
            "performed": False,
            "reason": "pilot intentionally uses eager execution",
            "contrastive_weight": 1.0,
            "eager_loss": float(observed.detach().float().cpu()),
        },
    }
    return eager_loss, execution


def run_role_contrastive_pilot(
    *,
    checkpoint_path: str | Path,
    v39_report_path: str | Path,
    relation_corpus_path: str | Path,
    relation_cases_path: str | Path,
    general_train_paths: Sequence[str | Path],
    general_eval_paths: Sequence[str | Path],
    report_output_path: str | Path,
    config: RoleContrastivePilotConfig = RoleContrastivePilotConfig(),
    device: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    resolved = _resolve_device(device)
    if resolved.type != "cuda" or int(config.physical_batch_size) != 256:
        raise ValueError("V42 requires CUDA and physical batch 256")
    if int(config.token_budget) % (256 * int(config.sequence_length)) != 0:
        raise ValueError("V42 token budget must divide into complete optimizer steps")
    if len(general_train_paths) != 2 or len(general_eval_paths) != 2:
        raise ValueError("V42 requires exactly two replay and two eval sources")
    pinned_paths = {
        Path(checkpoint_path): V39_CHECKPOINT_SHA256,
        Path(v39_report_path): V39_REPORT_SHA256,
        Path(relation_corpus_path): SOURCE_SHA256["relation_corpus"],
        Path(relation_cases_path): SOURCE_SHA256["relation_cases"],
        Path(general_train_paths[0]): SOURCE_SHA256["general_train_0"],
        Path(general_train_paths[1]): SOURCE_SHA256["general_train_1"],
        Path(general_eval_paths[0]): SOURCE_SHA256["general_eval_0"],
        Path(general_eval_paths[1]): SOURCE_SHA256["general_eval_1"],
    }
    for path, digest in pinned_paths.items():
        if sha256_file(path) != digest:
            raise ValueError(f"V42 source hash differs for {path.name}")

    data_config = _data_config(config)
    prepared = _prepare_data(
        tokenizer_checkpoint_path=checkpoint_path,
        relation_corpus_path=relation_corpus_path,
        relation_cases_path=relation_cases_path,
        general_train_paths=general_train_paths,
        general_eval_paths=general_eval_paths,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.microbatch_size),
        config=data_config,
        device=resolved,
    )
    if len(prepared.schedule) != len(set(prepared.schedule)):
        raise ValueError("V42 pilot schedule repeats a source batch")
    model, tokenizer, metadata = load_language_model_checkpoint(
        checkpoint_path, map_location="cpu"
    )
    model = model.to(resolved)
    initial_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    initial_hash = language_model_state_sha256(model)
    v39_report = json.loads(Path(v39_report_path).read_text(encoding="utf-8"))
    v39_selected = str(v39_report["selected_arm"])
    if v39_selected != "answer_weight4":
        raise ValueError("V42 parent report does not select the frozen V39 arm")
    parent_reference = {
        "selected_arm": v39_selected,
        "heldout": v39_report["arms"][v39_selected]["heldout"],
        "relation": v39_report["arms"][v39_selected]["relation"],
        "live_recomputed": False,
        "gate_dependency": False,
        "reason": "hash-pinned V39 evidence; V42 gate compares trained arms",
    }
    marker_values = tokenizer.encode(" Answer:", add_bos=False, add_eos=False)
    marker_ids = torch.tensor(marker_values, dtype=torch.long, device=resolved)
    branches = build_role_contrastive_branches(tokenizer, ROLE_GROUPS)
    role_lookup = prepare_role_contrastive_lookup(
        branches, vocab_size=tokenizer.vocab_size, device=resolved
    )
    example = grouped_staged_batch(
        prepared.staged,
        start=0,
        count=int(config.microbatches_per_optimizer_step),
        device=resolved,
    )
    answer_mask = answer_target_mask(
        example.input_ids, marker_ids=marker_ids, eos_id=tokenizer.eos_id
    )
    mask_fraction = float(answer_mask.float().mean().cpu())
    active_count = int(
        role_contrastive_lookup_active_mask(
            example.target_ids, answer_mask, role_lookup
        ).sum().cpu()
    )
    if not 0.0 < mask_fraction < 1.0 or active_count < 1:
        raise ValueError("V42 example batch has invalid objective coverage")
    training_config = _training_config(
        data_config,
        sequence_length=int(config.sequence_length),
        batch_size=int(config.physical_batch_size),
    )
    print("[V42] auditing shared eager role-contrastive loss", flush=True)
    loss_backend, execution = _prepare_shared_eager_loss(
        model,
        example,
        marker_ids=marker_ids,
        eos_id=tokenizer.eos_id,
        role_lookup=role_lookup,
        config=config,
    )

    rows: dict[str, dict[str, Any]] = {}
    weight_tensors = {
        name: torch.tensor(weight, dtype=torch.float32, device=resolved)
        for name, weight in ARM_WEIGHTS.items()
    }
    for name, weight in ARM_WEIGHTS.items():
        arm_weight = weight_tensors[name]

        def training_loss(
            input_ids: torch.Tensor,
            target_ids: torch.Tensor,
            fixed_weight: torch.Tensor = arm_weight,
        ) -> torch.Tensor:
            return loss_backend(input_ids, target_ids, fixed_weight)

        print(f"[V42] training {name}", flush=True)
        rows[name] = run_matched_training_arm(
            name,
            architecture="v39_transformer_role_contrastive_pilot",
            model=model,
            initial_state=initial_state,
            training_loss=training_loss,
            execution=execution,
            allocated_compile_seconds=float(execution["compile_seconds"]) / len(ARM_WEIGHTS),
            prepared=prepared,
            training_config=training_config,
            gradient_clip=float(config.gradient_clip),
            precision=str(config.precision),
            relation_eval_batch_size=int(config.relation_eval_batch_size),
            model_seed=int(config.model_seed),
            device=resolved,
            progress_prefix="V42",
            optimizer_builder=_optimizer_builder,
            optimizer_warmup_steps=3,
            microbatches_per_optimizer_step=int(config.microbatches_per_optimizer_step),
            extra_row={
                "contrastive_weight": weight,
                "answer_weight": float(config.answer_weight),
                "answer_mask_fraction_in_example": mask_fraction,
                "active_trie_branches_in_example": active_count,
            },
        )

    selected, decision, deltas = select_role_contrastive_candidate(
        rows, config=config
    )
    report = {
        "artifact_kind": ARTIFACT_KIND,
        "surface": SURFACE,
        "created_at_unix": time.time(),
        "external_llm_used": False,
        "parent_checkpoint": {
            "path": str(Path(checkpoint_path)),
            "sha256": V39_CHECKPOINT_SHA256,
            "initial_state_sha256": initial_hash,
            "metadata": metadata,
        },
        "v39_report": {
            "path": str(Path(v39_report_path)),
            "sha256": V39_REPORT_SHA256,
        },
        "config": asdict(config),
        "arms": rows,
        "candidate_deltas": deltas,
        "selected_arm": selected,
        "decision": decision,
        "checkpoint_saved": False,
        "pilot_only": True,
        "parent_reference": parent_reference,
        "objective": {
            "kind": "tokenizer_trie_role_unlikelihood",
            "role_groups": {key: list(values) for key, values in ROLE_GROUPS.items()},
            "branch_count": len(branches),
            "branches": [asdict(branch) for branch in branches],
            "answer_marker_ids": marker_values,
            "answer_mask_fraction_in_example": mask_fraction,
            "active_trie_branches_in_example": active_count,
            "shared_loss_function_all_arms": True,
        },
        "data": {
            "schedule_sha256": prepared.schedule_sha256,
            "source_selections": prepared.source_selections,
            "schedule_has_no_repeated_source_batch": True,
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report_with_readme(
        report_output_path,
        report,
        title="MARULHO V42 Role-Contrastive Pilot",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--v39-report", required=True)
    parser.add_argument("--relation-corpus", required=True)
    parser.add_argument("--relation-cases", required=True)
    parser.add_argument("--general-train", action="append", required=True)
    parser.add_argument("--general-eval", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    report = run_role_contrastive_pilot(
        checkpoint_path=args.checkpoint,
        v39_report_path=args.v39_report,
        relation_corpus_path=args.relation_corpus,
        relation_cases_path=args.relation_cases,
        general_train_paths=args.general_train,
        general_eval_paths=args.general_eval,
        report_output_path=args.report,
        device=args.device,
    )
    print(report["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
