"""Cross-modal grounding layer (§5.1–§5.3).

Maintains four cross-modal weight matrices (W_tv, W_vt, W_ta, W_at) updated
by Hebbian STDP when text and sensory spikes co-occur within a temporal
binding window.  Also tracks per-dimension grounding confidence.

Design decisions:
* A_plus < A_minus (0.010 vs 0.012) — anti-Hebbian drift prevents runaway
  potentiation and stabilises associations over time.
* Grounding confidence is an EMA of prediction accuracy — it is NOT a
  binary flag but a continuous measure of cross-modal reliability.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class CrossModalGroundingLayer:
    """Cross-modal grounding via temporal co-occurrence STDP.

    Args:
        dim_text: Dimensionality of text assembly vectors.
        dim_visual: Dimensionality of visual spike vectors.
        dim_audio: Dimensionality of audio spike vectors.
        A_plus: LTP amplitude for Hebbian potentiation (default 0.010).
        A_minus: LTD amplitude for anti-Hebbian decay (default 0.012).
        tau_trace: Trace decay time constant in functional ticks (default 10.0).
        confidence_alpha: EMA rate for grounding confidence (default 0.01).
        device: Torch device.
    """

    def __init__(
        self,
        dim_text: int,
        dim_visual: int,
        dim_audio: int = 64,
        A_plus: float = 0.010,
        A_minus: float = 0.012,
        tau_trace: float = 10.0,
        confidence_alpha: float = 0.01,
        device: torch.device | None = None,
    ) -> None:
        self.dim_text = int(dim_text)
        self.dim_visual = int(dim_visual)
        self.dim_audio = int(dim_audio)
        self.A_plus = float(A_plus)
        self.A_minus = float(A_minus)
        self.tau_trace = float(tau_trace)
        self.confidence_alpha = float(confidence_alpha)
        self.device = device or torch.device("cpu")

        # Four cross-modal weight matrices
        self.W_tv = torch.randn(dim_text, dim_visual, device=self.device) * 0.01
        self.W_vt = torch.randn(dim_visual, dim_text, device=self.device) * 0.01
        self.W_ta = torch.randn(dim_text, dim_audio, device=self.device) * 0.01
        self.W_at = torch.randn(dim_audio, dim_text, device=self.device) * 0.01

        # Exponential traces
        self.text_trace = torch.zeros(dim_text, device=self.device)
        self.visual_trace = torch.zeros(dim_visual, device=self.device)
        self.audio_trace = torch.zeros(dim_audio, device=self.device)

        # Per-dimension grounding confidence
        self.visual_confidence = torch.zeros(dim_text, device=self.device)
        self.audio_confidence = torch.zeros(dim_text, device=self.device)

    # -- trace decay --------------------------------------------------------

    def _decay_traces(self) -> None:
        """Apply exponential decay to all traces."""
        factor = torch.exp(torch.tensor(-1.0 / max(self.tau_trace, 0.01), device=self.device))
        self.text_trace *= factor
        self.visual_trace *= factor
        self.audio_trace *= factor

    # -- spike events -------------------------------------------------------

    def on_text_spike(self, text_assembly: torch.Tensor) -> None:
        """Process a text spike event — update traces and cross-modal weights.

        STDP: potentiate W_tv where visual_trace is active (co-occurrence),
        depress W_tv where visual_trace is inactive (text-alone).
        """
        t = text_assembly.to(self.device).float()
        if t.dim() > 1:
            t = t.squeeze(0)

        # Update text trace
        self.text_trace += t

        # LTP: text × visual_trace
        self.W_tv += self.A_plus * torch.outer(t, self.visual_trace)
        self.W_ta += self.A_plus * torch.outer(t, self.audio_trace)

        # LTD: anti-Hebbian decay where sensory trace is low
        visual_inactive = 1.0 - torch.clamp(self.visual_trace, 0, 1)
        audio_inactive = 1.0 - torch.clamp(self.audio_trace, 0, 1)
        self.W_tv -= self.A_minus * torch.outer(t, visual_inactive) * self.W_tv.abs()
        self.W_ta -= self.A_minus * torch.outer(t, audio_inactive) * self.W_ta.abs()

        # Update grounding confidence
        self._update_visual_confidence(t)
        self._update_audio_confidence(t)

        self._decay_traces()

    def on_visual_spike(self, visual_spikes: torch.Tensor) -> None:
        """Process a visual spike event — update traces and W_vt."""
        v = visual_spikes.to(self.device).float()
        if v.dim() > 1:
            v = v.squeeze(0)

        self.visual_trace += v

        # LTP: visual × text_trace
        self.W_vt += self.A_plus * torch.outer(v, self.text_trace)

        # LTD: text_trace inactive
        text_inactive = 1.0 - torch.clamp(self.text_trace, 0, 1)
        self.W_vt -= self.A_minus * torch.outer(v, text_inactive) * self.W_vt.abs()

        self._decay_traces()

    def on_audio_spike(self, audio_spikes: torch.Tensor) -> None:
        """Process an audio spike event — update traces and W_at."""
        a = audio_spikes.to(self.device).float()
        if a.dim() > 1:
            a = a.squeeze(0)

        self.audio_trace += a

        # LTP: audio × text_trace
        self.W_at += self.A_plus * torch.outer(a, self.text_trace)

        # LTD
        text_inactive = 1.0 - torch.clamp(self.text_trace, 0, 1)
        self.W_at -= self.A_minus * torch.outer(a, text_inactive) * self.W_at.abs()

        self._decay_traces()

    # -- grounding confidence -----------------------------------------------

    def _update_visual_confidence(self, text_assembly: torch.Tensor) -> None:
        """Update visual grounding confidence via prediction error.

        Applies exponential decay before the update so that confidence
        reflects recent prediction quality, not total accumulated exposure.
        Clamped to [0, 1].
        """
        # Exponential decay: confidence slowly forgets without reinforcement
        self.visual_confidence = self.visual_confidence * (1.0 - self.confidence_alpha * 0.1)

        predicted_visual = torch.mv(self.W_tv.T, text_assembly) if text_assembly.sum() > 0.01 else torch.zeros(self.dim_visual, device=self.device)
        # Prediction error is norm of predicted vs actual trace
        if self.visual_trace.sum() > 0.01 and predicted_visual.norm() > 1e-6:
            pn = F.normalize(predicted_visual, dim=0)
            vn = F.normalize(self.visual_trace, dim=0)
            error = 1.0 - F.cosine_similarity(pn.unsqueeze(0), vn.unsqueeze(0)).item()
        else:
            error = 1.0

        quality = max(0.0, 1.0 - error)
        update = self.confidence_alpha * text_assembly * quality
        self.visual_confidence = (self.visual_confidence + update).clamp(0.0, 1.0)

    def _update_audio_confidence(self, text_assembly: torch.Tensor) -> None:
        """Update audio grounding confidence via prediction error.

        Same decay + clamp as visual confidence.
        """
        self.audio_confidence = self.audio_confidence * (1.0 - self.confidence_alpha * 0.1)

        predicted_audio = torch.mv(self.W_ta.T, text_assembly) if text_assembly.sum() > 0.01 else torch.zeros(self.dim_audio, device=self.device)
        if self.audio_trace.sum() > 0.01 and predicted_audio.norm() > 1e-6:
            pn = F.normalize(predicted_audio, dim=0)
            an = F.normalize(self.audio_trace, dim=0)
            error = 1.0 - F.cosine_similarity(pn.unsqueeze(0), an.unsqueeze(0)).item()
        else:
            error = 1.0

        quality = max(0.0, 1.0 - error)
        update = self.confidence_alpha * text_assembly * quality
        self.audio_confidence = (self.audio_confidence + update).clamp(0.0, 1.0)

    # -- query API ----------------------------------------------------------

    def grounding_confidence(self) -> torch.Tensor:
        """Combined grounding confidence (visual + audio)."""
        return self.visual_confidence + self.audio_confidence

    def predict_visual(self, text_assembly: torch.Tensor) -> torch.Tensor:
        """Predict visual pattern from text assembly."""
        t = text_assembly.to(self.device).float()
        if t.dim() > 1:
            t = t.squeeze(0)
        return torch.mv(self.W_tv.T, t)

    def predict_text_from_visual(self, visual_spikes: torch.Tensor) -> torch.Tensor:
        """Predict text assembly from visual spikes."""
        v = visual_spikes.to(self.device).float()
        if v.dim() > 1:
            v = v.squeeze(0)
        return torch.mv(self.W_vt.T, v)

    def predict_audio(self, text_assembly: torch.Tensor) -> torch.Tensor:
        """Predict audio pattern from text assembly."""
        t = text_assembly.to(self.device).float()
        if t.dim() > 1:
            t = t.squeeze(0)
        return torch.mv(self.W_ta.T, t)

    def predict_text_from_audio(self, audio_spikes: torch.Tensor) -> torch.Tensor:
        """Predict text assembly from audio spikes."""
        a = audio_spikes.to(self.device).float()
        if a.dim() > 1:
            a = a.squeeze(0)
        return torch.mv(self.W_at.T, a)

    # -- alignment filter (§5.3) -------------------------------------------

    def alignment_gate(
        self,
        text_assembly: torch.Tensor,
        visual_spikes: torch.Tensor,
        threshold: float = 0.4,
    ) -> tuple[bool, float]:
        """Alignment filter — should we update cross-modal weights?

        Uses grounding confidence to mask ungrounded text dimensions,
        then checks cosine similarity between predicted and actual visual.
        """
        t = text_assembly.to(self.device).float()
        v = visual_spikes.to(self.device).float()
        if t.dim() > 1:
            t = t.squeeze(0)
        if v.dim() > 1:
            v = v.squeeze(0)

        conf_mask = (self.visual_confidence > 0.2).float()
        masked = t * conf_mask

        if masked.sum() < 0.01:
            return False, 0.0

        predicted = torch.mv(self.W_tv.T, masked)
        if predicted.norm() < 0.01:
            return False, 0.0

        p_norm = F.normalize(predicted, dim=0)
        v_norm = F.normalize(v, dim=0)
        score = float(F.cosine_similarity(p_norm.unsqueeze(0), v_norm.unsqueeze(0)).item())
        return score > threshold, max(0.0, score)

    def alignment_gate_audio(
        self,
        text_assembly: torch.Tensor,
        audio_spikes: torch.Tensor,
        threshold: float = 0.4,
    ) -> tuple[bool, float]:
        """Audio alignment filter — same logic as visual but for audio path."""
        t = text_assembly.to(self.device).float()
        a = audio_spikes.to(self.device).float()
        if t.dim() > 1:
            t = t.squeeze(0)
        if a.dim() > 1:
            a = a.squeeze(0)

        conf_mask = (self.audio_confidence > 0.2).float()
        masked = t * conf_mask

        if masked.sum() < 0.01:
            return False, 0.0

        predicted = torch.mv(self.W_ta.T, masked)
        if predicted.norm() < 0.01:
            return False, 0.0

        p_norm = F.normalize(predicted, dim=0)
        a_norm = F.normalize(a, dim=0)
        score = float(F.cosine_similarity(p_norm.unsqueeze(0), a_norm.unsqueeze(0)).item())
        return score > threshold, max(0.0, score)

    def reset(self) -> None:
        """Clear traces (keep learned weights and confidence)."""
        self.text_trace.zero_()
        self.visual_trace.zero_()
        self.audio_trace.zero_()

    # -- §7.4 self-criticism loop -------------------------------------------

    def run_self_criticism(
        self,
        recent_visual_frames: list[torch.Tensor],
        confidence_threshold: float = 0.7,
        alignment_floor: float = 0.2,
        penalty: float = 0.10,
        blacklist: dict[int, int] | None = None,
        blacklist_strikes: int = 2,
    ) -> dict[str, Any]:
        """Visual self-criticism loop (§7.4): verify high-confidence visual groundings.

        Modality-specific: only affects W_tv/W_vt and visual_confidence.
        Audio associations (W_ta/W_at/audio_confidence) are untouched — a wrong
        visual association does not imply a wrong audio association.
        """
        if blacklist is None:
            blacklist = {}

        high_conf = (self.visual_confidence > confidence_threshold).nonzero(as_tuple=True)[0]
        checked = 0
        penalised = 0
        blacklisted_count = 0

        for idx in high_conf:
            i = int(idx.item())
            w_row = self.W_tv[i]
            if w_row.norm() < 1e-6:
                continue

            pred_visual = F.normalize(w_row, dim=0)

            best_score = 0.0
            for frame in recent_visual_frames:
                if frame.norm() < 1e-6:
                    continue
                frame_norm = F.normalize(frame.flatten()[:pred_visual.shape[0]], dim=0)
                sim = float(F.cosine_similarity(
                    pred_visual.unsqueeze(0), frame_norm.unsqueeze(0)
                ).item())
                best_score = max(best_score, sim)

            checked += 1
            if best_score < alignment_floor:
                self.visual_confidence[i] = max(
                    0.0, float(self.visual_confidence[i]) - penalty
                )
                penalised += 1

                blacklist[i] = blacklist.get(i, 0) + 1
                if blacklist[i] >= blacklist_strikes:
                    # Zero ONLY visual association weights
                    self.W_tv[i].zero_()
                    self.W_vt[:, i].zero_()
                    self.visual_confidence[i] = 0.0
                    blacklisted_count += 1

        return {
            "checked": checked,
            "penalised": penalised,
            "blacklisted": blacklisted_count,
            "blacklist_state": blacklist,
        }

    def run_self_criticism_audio(
        self,
        recent_audio_frames: list[torch.Tensor],
        confidence_threshold: float = 0.7,
        alignment_floor: float = 0.2,
        penalty: float = 0.10,
        blacklist: dict[int, int] | None = None,
        blacklist_strikes: int = 2,
    ) -> dict[str, Any]:
        """Audio self-criticism loop: verify high-confidence audio groundings.

        Modality-specific: only affects W_ta/W_at and audio_confidence.
        Visual associations are untouched.
        """
        if blacklist is None:
            blacklist = {}

        high_conf = (self.audio_confidence > confidence_threshold).nonzero(as_tuple=True)[0]
        checked = 0
        penalised = 0
        blacklisted_count = 0

        for idx in high_conf:
            i = int(idx.item())
            w_row = self.W_ta[i]
            if w_row.norm() < 1e-6:
                continue

            pred_audio = F.normalize(w_row, dim=0)

            best_score = 0.0
            for frame in recent_audio_frames:
                if frame.norm() < 1e-6:
                    continue
                frame_norm = F.normalize(frame.flatten()[:pred_audio.shape[0]], dim=0)
                sim = float(F.cosine_similarity(
                    pred_audio.unsqueeze(0), frame_norm.unsqueeze(0)
                ).item())
                best_score = max(best_score, sim)

            checked += 1
            if best_score < alignment_floor:
                self.audio_confidence[i] = max(
                    0.0, float(self.audio_confidence[i]) - penalty
                )
                penalised += 1

                blacklist[i] = blacklist.get(i, 0) + 1
                if blacklist[i] >= blacklist_strikes:
                    # Zero ONLY audio association weights
                    self.W_ta[i].zero_()
                    self.W_at[:, i].zero_()
                    self.audio_confidence[i] = 0.0
                    blacklisted_count += 1

        return {
            "checked": checked,
            "penalised": penalised,
            "blacklisted": blacklisted_count,
            "blacklist_state": blacklist,
        }

    # -- serialization ------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "W_tv": self.W_tv.cpu(),
            "W_vt": self.W_vt.cpu(),
            "W_ta": self.W_ta.cpu(),
            "W_at": self.W_at.cpu(),
            "visual_confidence": self.visual_confidence.cpu(),
            "audio_confidence": self.audio_confidence.cpu(),
            "text_trace": self.text_trace.cpu(),
            "visual_trace": self.visual_trace.cpu(),
            "audio_trace": self.audio_trace.cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for key in ("W_tv", "W_vt", "W_ta", "W_at",
                     "visual_confidence", "audio_confidence",
                     "text_trace", "visual_trace", "audio_trace"):
            val = state.get(key)
            if val is not None:
                setattr(self, key, val.to(self.device))
