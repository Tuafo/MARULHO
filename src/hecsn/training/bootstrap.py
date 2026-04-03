from __future__ import annotations

import torch
import torch.nn.functional as F


class PredictiveBootstrap:
    """Lightweight next-step predictor used during cold start."""

    def __init__(self, device: torch.device, input_dim: int, lr: float = 0.01) -> None:
        dim = int(input_dim)
        if dim <= 0:
            raise ValueError("input_dim must be positive")
        self.W = torch.zeros((dim, dim), device=device)
        self.lr = float(lr)
        self.prev_pattern: torch.Tensor | None = None

    def update(self, current_pattern: torch.Tensor) -> float:
        cur = current_pattern.to(self.W.device)
        cur = cur / (cur.sum() + 1e-8)
        if self.prev_pattern is None:
            self.prev_pattern = cur.detach().clone()
            return 0.0

        pred = F.softmax(torch.mv(self.W, self.prev_pattern), dim=0)
        err = F.kl_div(torch.log(pred + 1e-8), cur, reduction="sum").item()

        delta = (cur - pred).unsqueeze(1)
        self.W += self.lr * torch.mm(delta, self.prev_pattern.unsqueeze(0))
        self.W = torch.clamp(self.W, -1.0, 1.0)

        self.prev_pattern = cur.detach().clone()
        return float(err)
