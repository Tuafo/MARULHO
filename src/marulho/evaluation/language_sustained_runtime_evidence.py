"""Qualify sustained generation from one exact MARULHO Transformer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import time
from typing import Any

import torch

from marulho.data.language_tokenizer import LanguageTokenizer
from marulho.reporting.readme_reports import write_json_report_with_readme
from marulho.training.language_model import (
    MarulhoLanguageModel,
    language_model_state_sha256,
    load_language_model_checkpoint,
)


SURFACE = "marulho_transformer_sustained_generation.v4"


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None or not Path(path).is_file():
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _build_prompt_batch(
    tokenizer: LanguageTokenizer,
    *,
    prompt: str,
    stream_count: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    texts = (
        [str(prompt)]
        if int(stream_count) == 1
        else [f"{prompt} Stream {index:04d}:" for index in range(int(stream_count))]
    )
    encoded = []
    filler = tokenizer.encode(
        " Continue this independent language sequence coherently.",
        add_bos=False,
        add_eos=False,
    )
    for text in texts:
        seed = tokenizer.encode(text, add_bos=True, add_eos=False)
        if int(stream_count) > 1:
            repeats = 1 + max(0, int(context_length) - len(seed)) // max(1, len(filler))
            seed = (filler * repeats) + seed
        encoded.append(seed[-int(context_length) :])
    width = max(len(row) for row in encoded)
    prompt_ids = torch.full(
        (int(stream_count), width),
        int(tokenizer.bos_id),
        dtype=torch.long,
        device=device,
    )
    for index, row in enumerate(encoded):
        prompt_ids[index, -len(row) :] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
    return prompt_ids, texts


def _observed_active_compute(
    model: MarulhoLanguageModel,
    prompt_ids: torch.Tensor,
    *,
    use_bfloat16: bool,
) -> dict[str, Any]:
    executed_names: set[str] = set()
    executed_parameter_ids: set[int] = set()
    handles = []

    def hook(name: str, module: torch.nn.Module):
        def record(_module, _inputs, _output) -> None:
            executed_names.add(name)
            executed_parameter_ids.update(
                id(parameter) for parameter in module.parameters(recurse=False)
            )

        return record

    named_modules = [(name, module) for name, module in model.named_modules() if name]
    try:
        for name, module in named_modules:
            handles.append(module.register_forward_hook(hook(name, module)))
        with torch.inference_mode(), torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=bool(use_bfloat16),
        ):
            audit = model.forward(prompt_ids[:1], collect_telemetry=False)
            raw_nonfinite = int(
                torch.count_nonzero(~torch.isfinite(audit["logits"])).item()
            )
    finally:
        for handle in handles:
            handle.remove()

    unique_parameters = {id(parameter): parameter for parameter in model.parameters()}
    total_parameter_count = sum(
        int(parameter.numel()) for parameter in unique_parameters.values()
    )
    executed_parameter_count = sum(
        int(unique_parameters[parameter_id].numel())
        for parameter_id in executed_parameter_ids
    )
    layer_indices = sorted(
        {
            int(name.split(".")[2])
            for name in executed_names
            if name.startswith("state_block.layers.")
            and len(name.split(".")) >= 3
            and name.split(".")[2].isdigit()
        }
    )
    attention_modules = sorted(
        name for name in executed_names if name.endswith(".attention")
    )
    mlp_gate_modules = sorted(
        name for name in executed_names if name.endswith(".gate_up")
    )
    fraction = float(executed_parameter_count) / float(max(1, total_parameter_count))
    return {
        "measurement": "observed_forward_hooks_plus_dense_architecture",
        "audit_stream_count": 1,
        "raw_nonfinite_logit_count": raw_nonfinite,
        "parameter_owning_module_count": sum(
            1 for _name, module in named_modules if list(module.parameters(recurse=False))
        ),
        "executed_module_count": len(executed_names),
        "executed_parameter_count": executed_parameter_count,
        "total_parameter_count": total_parameter_count,
        "executed_parameter_fraction": fraction,
        "unexecuted_parameter_fraction": 1.0 - fraction,
        "executed_transformer_layer_indices": layer_indices,
        "executed_attention_module_count": len(attention_modules),
        "executed_mlp_gate_module_count": len(mlp_gate_modules),
        "configured_attention_heads_per_layer": int(model.config.attention_heads),
        "conditional_parameter_routing_present": False,
        "structural_sparsity_observed": False,
        "structural_sparsity_fraction": 0.0,
        "interpretation": "the maintained V39 Transformer executes densely",
    }


def run_language_sustained_runtime_evidence(
    model: MarulhoLanguageModel,
    tokenizer: LanguageTokenizer,
    *,
    output_path: str | Path,
    target_tokens: int,
    checkpoint_path: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    prompt: str = "MARULHO sustained runtime.",
    stream_count: int = 1,
    timeout_seconds: float = 600.0,
    generation_repetition_penalty: float = 1.1,
    generation_no_repeat_ngram_size: int = 3,
    inference_dtype: str = "float32",
    require_cuda: bool = False,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    target = max(0, int(target_tokens))
    streams = max(1, int(stream_count))
    if target % streams != 0:
        raise ValueError("target_tokens must be divisible by stream_count")
    tokens_per_stream = target // streams
    dtype_name = str(inference_dtype).strip().lower()
    if dtype_name not in {"float32", "bfloat16"}:
        raise ValueError("inference_dtype must be float32 or bfloat16")
    use_bfloat16 = dtype_name == "bfloat16"
    if use_bfloat16 and model.device.type != "cuda":
        raise ValueError("bfloat16 sustained inference currently requires CUDA")

    checkpoint_sha256 = _file_sha256(checkpoint_path)
    expected_sha256 = (
        None
        if expected_checkpoint_sha256 is None
        else str(expected_checkpoint_sha256).strip().lower()
    )
    if expected_sha256 is not None and checkpoint_sha256 != expected_sha256:
        raise ValueError("checkpoint SHA-256 does not match the preregistered parent")

    prompt_ids, prompt_texts = _build_prompt_batch(
        tokenizer,
        prompt=str(prompt),
        stream_count=streams,
        context_length=model.context_length,
        device=model.device,
    )
    state_hash_before = language_model_state_sha256(model)
    active_compute = _observed_active_compute(
        model,
        prompt_ids,
        use_bfloat16=use_bfloat16,
    )
    if model.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(model.device)
        torch.cuda.synchronize(model.device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=model.device.type,
        dtype=torch.bfloat16,
        enabled=use_bfloat16,
    ):
        generated = model.generate(
            prompt_ids,
            max_new_tokens=tokens_per_stream,
            eos_id=None,
            repetition_penalty=max(1.0, float(generation_repetition_penalty)),
            no_repeat_ngram_size=max(0, int(generation_no_repeat_ngram_size)),
            decode_control_window=model.context_length,
        )
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)
    elapsed = max(time.perf_counter() - started, 1.0e-9)
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(model.device))
        if model.device.type == "cuda"
        else 0
    )

    prompt_width = int(prompt_ids.shape[1])
    generated_ids = generated["generated_ids"].detach().cpu()
    continuation_ids = generated_ids[:, prompt_width:]
    token_delta = int(generated["total_new_token_count"])
    out_of_vocab_count = int(
        ((continuation_ids < 0) | (continuation_ids >= tokenizer.vocab_size))
        .sum()
        .item()
    )
    stream_hashes = [_tensor_sha256(row) for row in continuation_ids]
    preview_indices = sorted({0, min(1, streams - 1), streams - 1})
    previews = []
    for index in preview_indices:
        row = continuation_ids[index]
        head = row[: min(256, int(row.numel()))].tolist()
        tail = row[-min(64, int(row.numel())) :].tolist()
        previews.append(
            {
                "stream_index": index,
                "prompt": prompt_texts[index],
                "continuation_sha256": stream_hashes[index],
                "head_token_count": len(head),
                "head_text": tokenizer.decode(head),
                "tail_token_count": len(tail),
                "tail_text": tokenizer.decode(tail),
            }
        )
    final_state = generated["state"]
    final_kv_lengths = [
        int(value.shape[2])
        for key, value in final_state.items()
        if key.endswith("_key")
    ]
    max_final_kv_tokens = max(final_kv_lengths, default=0)
    state_hash_after = language_model_state_sha256(model)
    state_immutable = state_hash_before == state_hash_after
    timed_out = elapsed > float(timeout_seconds)
    cuda_satisfied = model.device.type == "cuda" or not bool(require_cuda)
    success = all(
        (
            token_delta == target,
            int(generated["new_token_count_per_stream"]) == tokens_per_stream,
            not timed_out,
            cuda_satisfied,
            int(generated["nonfinite_logit_count"]) == 0,
            int(active_compute["raw_nonfinite_logit_count"]) == 0,
            out_of_vocab_count == 0,
            max_final_kv_tokens <= model.context_length,
            state_immutable,
        )
    )
    qualifies_sustained_runtime_contract = bool(
        success
        and require_cuda
        and target == 524_288
        and streams == 256
        and tokens_per_stream == 2_048
        and expected_sha256 is not None
    )
    first_preview = previews[0]["head_text"] if previews else ""
    report = {
        "artifact_kind": "marulho_transformer_sustained_generation",
        "surface": SURFACE,
        "report_status": "accepted" if success else "rejected",
        "decision": (
            "qualify_same_checkpoint_sustained_runtime"
            if qualifies_sustained_runtime_contract
            else ("accept_sustained_diagnostic" if success else "reject_sustained_runtime")
        ),
        "success": success,
        "qualifies_sustained_runtime_contract": qualifies_sustained_runtime_contract,
        "output_path": str(output),
        "owned_by_marulho": True,
        "external_llm_used": False,
        "loads_external_checkpoint": False,
        "active_language_path": model.config.active_language_path,
        "target_tokens": target,
        "token_delta": token_delta,
        "stream_count": streams,
        "tokens_per_stream": tokens_per_stream,
        "single_stream_524288": bool(streams == 1 and tokens_per_stream == 524_288),
        "elapsed_seconds": elapsed,
        "tokens_per_second": float(token_delta) / elapsed,
        "timeout_seconds": float(timeout_seconds),
        "timed_out": timed_out,
        "device_backend": str(model.device),
        "inference_dtype": dtype_name,
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "model_vocab_size": int(model.config.vocab_size),
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "generation_vocab_size": int(model.generation_vocab_size),
        "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "expected_checkpoint_sha256": expected_sha256,
        "checkpoint_hash_matches_expected": (
            None if expected_sha256 is None else checkpoint_sha256 == expected_sha256
        ),
        "model_state_sha256_before": state_hash_before,
        "model_state_sha256_after": state_hash_after,
        "model_state_immutable": state_immutable,
        "prompt": str(prompt),
        "prompt_batch_sha256": _tensor_sha256(prompt_ids),
        "prompt_token_width": prompt_width,
        "continuation_token_sha256": _tensor_sha256(continuation_ids),
        "unique_stream_continuation_hash_count": len(set(stream_hashes)),
        "stream_continuation_sha256": stream_hashes,
        "output_previews": previews,
        "generated_text": first_preview,
        "continuation_text": first_preview,
        "raw_nonfinite_logit_count": int(generated["nonfinite_logit_count"]),
        "out_of_vocabulary_token_count": out_of_vocab_count,
        "generation_decode": generated["generation_decode"],
        "decode_control_totals": generated["decode_control_totals"],
        "active_compute": active_compute,
        "runtime": {
            "state_core": "transformer",
            "bounded_kv_cache": True,
            "context_length": int(model.context_length),
            "max_final_kv_tokens": max_final_kv_tokens,
            "decode_control_history_tokens": int(model.context_length),
            "output_storage": "aggregate_hashes_and_bounded_previews",
            "routing_present": False,
            "spiking_present": False,
            "sampled_padded_vocab_present": False,
        },
        "promotes_runtime_claim": qualifies_sustained_runtime_contract,
        "promotes_generation_quality_claim": False,
    }
    write_json_report_with_readme(output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, required=True)
    parser.add_argument("--stream-count", type=int, default=1)
    parser.add_argument("--prompt", default="MARULHO sustained runtime.")
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.1)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument(
        "--inference-dtype",
        choices=("float32", "bfloat16"),
        default="float32",
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--map-location", default="auto")
    args = parser.parse_args()
    target_device = (
        "cuda"
        if args.map_location == "auto" and torch.cuda.is_available()
        else ("cpu" if args.map_location == "auto" else args.map_location)
    )
    model, tokenizer, _metadata = load_language_model_checkpoint(
        args.checkpoint,
        map_location="cpu",
    )
    model = model.to(target_device)
    report = run_language_sustained_runtime_evidence(
        model,
        tokenizer,
        output_path=args.output,
        target_tokens=max(0, int(args.target_tokens)),
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        prompt=args.prompt,
        stream_count=max(1, int(args.stream_count)),
        timeout_seconds=float(args.timeout_seconds),
        generation_repetition_penalty=float(args.generation_repetition_penalty),
        generation_no_repeat_ngram_size=int(args.generation_no_repeat_ngram_size),
        inference_dtype=args.inference_dtype,
        require_cuda=bool(args.require_cuda),
    )
    return 0 if bool(report["success"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
