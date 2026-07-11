"""Read-only representation geometry for MARULHO language experiments."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


@torch.no_grad()
def transformer_depth_geometry_report(
    model,
    input_ids: torch.Tensor,
    *,
    max_samples: int = 4096,
) -> dict[str, Any]:
    """Measure depth geometry without changing model state or selecting quality."""

    if int(max_samples) < 2:
        raise ValueError("max_samples must be at least two")
    state_block = getattr(model, "state_block", None)
    if state_block is None or not hasattr(state_block, "layers"):
        raise TypeError("model must expose a Transformer-compatible state_block")
    histories: list[torch.Tensor] = []

    def capture_projection(_module, _inputs, output) -> None:
        histories.append(output.detach())

    def capture_layer(_module, _inputs, output) -> None:
        histories.append(output[0].detach())

    handles = [
        state_block.input_projection.register_forward_hook(capture_projection),
        *[
            layer.register_forward_hook(capture_layer)
            for layer in state_block.layers
        ],
    ]
    was_training = model.training
    try:
        model.eval()
        model(input_ids, collect_telemetry=False)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if len(histories) != len(state_block.layers) + 1:
        raise RuntimeError("Depth geometry did not capture every Transformer depth")

    rows: list[dict[str, Any]] = []
    previous: torch.Tensor | None = None
    for depth, hidden in enumerate(histories):
        flat = hidden.float().reshape(-1, hidden.shape[-1])
        if int(flat.shape[0]) > int(max_samples):
            indices = torch.linspace(
                0,
                int(flat.shape[0]) - 1,
                steps=int(max_samples),
                device=flat.device,
            ).long()
            flat = flat.index_select(0, indices)
        with torch.autocast(device_type=flat.device.type, enabled=False):
            centered = flat - flat.mean(dim=0, keepdim=True)
            covariance = centered.T @ centered / float(
                max(1, int(flat.shape[0]) - 1)
            )
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            total = eigenvalues.sum().clamp_min(1.0e-12)
            probabilities = eigenvalues / total
            participation = total.square() / eigenvalues.square().sum().clamp_min(
                1.0e-12
            )
            effective_rank = torch.exp(
                -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
            )
            adjacent_cosine = None
            if previous is not None:
                adjacent_cosine = float(
                    F.cosine_similarity(flat, previous, dim=-1).mean().cpu()
                )
        rows.append(
            {
                "depth": depth,
                "sample_count": int(flat.shape[0]),
                "participation_ratio": float(participation.cpu()),
                "effective_rank": float(effective_rank.cpu()),
                "rms": float(flat.square().mean().sqrt().cpu()),
                "mean_vector_norm": float(flat.mean(dim=0).norm().cpu()),
                "adjacent_cosine": adjacent_cosine,
            }
        )
        previous = flat
    return {
        "surface": "marulho_transformer_depth_geometry.v1",
        "promotion_metric": False,
        "rows": rows,
        "external_llm_used": False,
    }
