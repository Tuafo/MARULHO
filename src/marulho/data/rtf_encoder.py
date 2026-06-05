from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Literal, Sequence
import torch

if TYPE_CHECKING:
    from marulho.config.model_config import MarulhoConfig


RepresentationMode = Literal["order_weighted_ascii", "unigram_ascii", "hashed_ngram"]


def _normalize(vector: torch.Tensor) -> torch.Tensor:
    total = float(torch.norm(vector.float(), p=2).item())
    if total <= 0.0:
        return torch.zeros_like(vector, dtype=torch.float32)
    return vector.float() / (torch.norm(vector.float(), p=2) + 1e-8)


def _ascii_code(ch: str) -> int:
    code = ord(ch)
    return int(code if code < 128 else 0)


class LearnedChunkingLayer:
    def __init__(
        self,
        *,
        n_detectors: int,
        min_chunk_len: int,
        max_chunk_len: int,
        similarity_floor: float,
        boundary_threshold: float,
        update_lr: float,
        association_blend: float,
        association_lr: float,
        association_decay: float,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.n_detectors = int(n_detectors)
        self.min_chunk_len = int(min_chunk_len)
        self.max_chunk_len = int(max_chunk_len)
        self.similarity_floor = float(similarity_floor)
        self.boundary_threshold = float(boundary_threshold)
        self._base_boundary_threshold = float(boundary_threshold)
        self._abstraction_bias: float = 0.0  # top-down bias from Abstraction Layer
        self.update_lr = float(update_lr)
        self.association_blend = float(association_blend)
        self.association_lr = float(association_lr)
        self.association_decay = float(association_decay)
        self.prototype_dim = (self.max_chunk_len * 8) + 32
        self.prototypes = torch.zeros(self.n_detectors, self.prototype_dim, dtype=torch.float32, device=self.device)
        self.confidence = torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        self.usage = torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        self.associations = torch.zeros(
            self.n_detectors,
            self.n_detectors,
            dtype=torch.float32,
            device=self.device,
        )
        self._bit_shifts = torch.arange(8, dtype=torch.int64, device=self.device)

    def device_report(self) -> dict[str, object]:
        return {
            "device": str(self.device),
            "prototypes_device": str(self.prototypes.device),
            "confidence_device": str(self.confidence.device),
            "usage_device": str(self.usage.device),
            "associations_device": str(self.associations.device),
            "bit_shifts_device": str(self._bit_shifts.device),
            "cuda": self.device.type == "cuda",
        }

    def set_abstraction_bias(self, mean_certainty: float, max_gap_score: float) -> None:
        """Top-down boundary bias from Abstraction Layer (§3.1).

        High uncertainty (low certainty, high gap) → lower threshold → finer chunks.
        High certainty (low gap) → higher threshold → coarser chunks.
        Bias range: ±30% of base threshold.
        """
        # Certainty in [0,1]; gap_score unbounded but typically [0,2]
        certainty = max(0.0, min(1.0, float(mean_certainty)))
        gap = max(0.0, min(2.0, float(max_gap_score)))
        # Positive bias → raise threshold (coarser), negative → lower (finer)
        self._abstraction_bias = 0.3 * (certainty - 0.5 * gap)
        self.boundary_threshold = self._base_boundary_threshold * (1.0 + self._abstraction_bias)

    @staticmethod
    def is_separator(code: int) -> bool:
        if not 0 <= int(code) < 128:
            return True
        ch = chr(int(code))
        return ch.isspace() or ch in ",.;:!?()[]{}<>\""

    @staticmethod
    def is_hard_boundary(code: int) -> bool:
        if not 0 <= int(code) < 128:
            return True
        return chr(int(code)) in ".!?\n\r"

    @staticmethod
    def _byte_weight(code: int) -> float:
        if not 0 <= int(code) < 128:
            return 0.0
        ch = chr(int(code))
        if ch.isalpha():
            return 1.0
        if ch.isdigit():
            return 0.85
        if ch in "'-_/":
            return 0.65
        if ch.isspace():
            return 0.2
        return 0.5

    def state_dict(self) -> dict[str, Any]:
        return {
            "prototypes": self.prototypes.detach().clone().cpu(),
            "confidence": self.confidence.detach().clone().cpu(),
            "usage": self.usage.detach().clone().cpu(),
            "associations": self.associations.detach().clone().cpu(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        prototypes = state.get("prototypes")
        confidence = state.get("confidence")
        usage = state.get("usage")
        associations = state.get("associations")
        if isinstance(prototypes, torch.Tensor) and prototypes.shape == self.prototypes.shape:
            self.prototypes = prototypes.detach().clone().to(self.device).float()
        if isinstance(confidence, torch.Tensor) and confidence.shape == self.confidence.shape:
            self.confidence = confidence.detach().clone().to(self.device).float()
        if isinstance(usage, torch.Tensor) and usage.shape == self.usage.shape:
            self.usage = usage.detach().clone().to(self.device).float()
        if isinstance(associations, torch.Tensor) and associations.shape == self.associations.shape:
            self.associations = associations.detach().clone().to(self.device).float()

    def encode_chunk(self, chars: Sequence[int]) -> torch.Tensor:
        window: list[int] = [int(code) for code in list(chars)[-self.max_chunk_len :] if 0 <= int(code) < 128]
        vector = torch.zeros(self.prototype_dim, dtype=torch.float32, device=self.device)
        if not window:
            return vector

        n = len(window)
        codes_t = torch.tensor(window, dtype=torch.int64, device=self.device)
        bit_masks = ((codes_t.unsqueeze(1) >> self._bit_shifts) & 1).float()  # (n, 8)
        for pos in range(n):
            w = self._byte_weight(window[pos]) * max(0.35, 1.0 - 0.04 * pos)
            base = pos * 8
            vector[base : base + 8] = bit_masks[pos] * w

        offset = self.max_chunk_len * 8
        if n >= 2:
            left = codes_t[:-1]
            right = codes_t[1:]
            buckets = (((left + 17) * 131 + (right + 1) * 31) % 32).tolist()
            for b in buckets:
                vector[offset + b] += 1.0

        return _normalize(vector)

    def _similarities_from_encoding(self, encoding: torch.Tensor) -> torch.Tensor:
        encoding = encoding.to(self.device)
        if float(encoding.sum().item()) <= 0.0:
            return torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        active = self.usage > 0.0
        if not bool(active.any().item()):
            return torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        sims = torch.mv(self.prototypes, encoding.float())
        sims = torch.clamp(sims, min=0.0) * active.float()
        sims = sims * torch.clamp(self.confidence, min=0.25, max=1.0)
        return sims

    def _best_detector(self, chars: Sequence[int]) -> tuple[int | None, float]:
        similarities = self._similarities_from_encoding(self.encode_chunk(chars))
        if similarities.numel() == 0:
            return None, 0.0
        best = float(similarities.max().item()) if int(similarities.numel()) > 0 else 0.0
        if best <= 0.0:
            return None, 0.0
        return int(torch.argmax(similarities).item()), best

    def detector_activations(self, chars: Sequence[int]) -> torch.Tensor:
        similarities = self._similarities_from_encoding(self.encode_chunk(chars))
        if float(similarities.max().item()) <= 0.0:
            return torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        active_count = max(1, int((self.usage > 0.0).sum().item()))
        top_k = min(8, active_count)
        values, indices = torch.topk(similarities, k=top_k)
        result = torch.zeros_like(similarities)
        result[indices] = values
        result = _normalize(result)
        if self.association_blend <= 0.0:
            return result
        associated = self._association_context(result)
        if float(associated.sum().item()) <= 0.0:
            return result
        blended = ((1.0 - self.association_blend) * result) + (self.association_blend * associated)
        return _normalize(blended)

    def _allocate_detector_index(self) -> int:
        inactive = torch.nonzero(self.usage <= 0.0, as_tuple=False).flatten()
        if inactive.numel() > 0:
            return int(inactive[0].item())
        scores = self.usage + (self.confidence * 8.0)
        return int(torch.argmin(scores).item())

    def _creation_floor(self, chars: Sequence[int]) -> float:
        length = min(len(chars), self.max_chunk_len)
        return max(self.similarity_floor, 0.55 - (0.02 * float(length)))

    def _association_context(self, activations: torch.Tensor) -> torch.Tensor:
        activations = activations.to(self.device)
        if activations.numel() == 0 or float(activations.sum().item()) <= 0.0:
            return torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        associated = torch.mv(self.associations, activations.float())
        if float(associated.sum().item()) <= 0.0:
            return torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        associated = associated * (self.usage > 0.0).float()
        return _normalize(torch.clamp(associated, min=0.0))

    def _update_associations(self, current: torch.Tensor, context: torch.Tensor | None) -> None:
        if self.association_lr <= 0.0 or context is None:
            return
        if context.dim() != 1 or int(context.numel()) != self.n_detectors:
            return
        current_vec = _normalize(torch.clamp(current.float(), min=0.0))
        context_vec = _normalize(torch.clamp(context.to(self.device).float(), min=0.0))
        if float(current_vec.sum().item()) <= 0.0 or float(context_vec.sum().item()) <= 0.0:
            return
        pair = torch.outer(context_vec, current_vec) + torch.outer(current_vec, context_vec)
        updated = (self.association_decay * self.associations) + (self.association_lr * pair)
        updated.fill_diagonal_(0.0)
        self.associations = torch.clamp(updated, min=0.0, max=1.0)

    def learn_chunk(self, chars: Sequence[int], *, context: torch.Tensor | None = None) -> None:
        encoding = self.encode_chunk(chars)
        if float(encoding.sum().item()) <= 0.0:
            return
        similarities = self._similarities_from_encoding(encoding)
        best_score = float(similarities.max().item()) if int(similarities.numel()) > 0 else 0.0
        if best_score < self._creation_floor(chars):
            index = self._allocate_detector_index()
            self.prototypes[index] = encoding
            self.confidence[index] = max(0.5, best_score)
            self.usage[index] = self.usage[index] + 1.0
            current = torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
            current[index] = 1.0
            self._update_associations(current, context)
            return

        index = int(torch.argmax(similarities).item())
        lr = self.update_lr * max(0.25, 1.0 - float(self.confidence[index].item()))
        updated = ((1.0 - lr) * self.prototypes[index]) + (lr * encoding)
        self.prototypes[index] = _normalize(updated)
        self.confidence[index] = max(
            0.15,
            min(
                1.0,
                (0.85 * float(self.confidence[index].item())) + (0.15 * best_score),
            ),
        )
        self.usage[index] = self.usage[index] + 1.0
        current = torch.zeros(self.n_detectors, dtype=torch.float32, device=self.device)
        current[index] = 1.0
        self._update_associations(current, context)

    def should_boundary(self, buffer: Sequence[int], next_code: int) -> bool:
        if len(buffer) >= self.max_chunk_len:
            return True
        if len(buffer) < self.min_chunk_len:
            return False

        current_idx, current_score = self._best_detector(buffer)
        if current_idx is None or current_score < self.similarity_floor:
            return False

        extended_idx, extended_score = self._best_detector([*buffer, int(next_code)])
        changed_detector = extended_idx is None or extended_idx != current_idx
        score_drop = current_score - extended_score
        return bool(changed_detector and score_drop >= self.boundary_threshold)


class RTFEncoder:
    """Rate-Temporal Fusion encoder.

    Produces:
    - order-sensitive routing vector [128] derived from latency coding
    - spike-time tensor [128, n_bursts_max] for temporal simulation
    """

    def __init__(
        self,
        t_max: float = 20.0,
        n_bursts_max: int = 5,
        window_size: int = 10,
        representation: RepresentationMode = "order_weighted_ascii",
        hashed_ngram_dim: int = 2048,
        hashed_ngram_min_n: int = 2,
        hashed_ngram_max_n: int = 3,
        enable_learned_chunking: bool = False,
        learned_chunk_detector_count: int = 128,
        learned_chunk_min_len: int = 2,
        learned_chunk_max_len: int = 12,
        learned_chunk_feature_mode: Literal["blend", "concat"] = "blend",
        learned_chunk_concat_dim: int = 128,
        learned_chunk_blend: float = 0.5,
        learned_chunk_similarity_floor: float = 0.30,
        learned_chunk_boundary_threshold: float = 0.08,
        learned_chunk_update_lr: float = 0.25,
        learned_chunk_association_blend: float = 0.35,
        learned_chunk_association_lr: float = 0.15,
        learned_chunk_association_decay: float = 0.995,
        device: torch.device | str | None = None,
    ) -> None:
        self.device = torch.device("cpu" if device is None else device)
        self.t_max = float(t_max)
        self.n_bursts_max = int(n_bursts_max)
        self.window_size = int(window_size)
        self.t_spacing = self.t_max / max(1, self.window_size + 1)
        self.representation = representation
        self.hashed_ngram_dim = int(hashed_ngram_dim)
        self.hashed_ngram_min_n = int(hashed_ngram_min_n)
        self.hashed_ngram_max_n = int(hashed_ngram_max_n)
        self.learned_chunk_feature_mode = str(learned_chunk_feature_mode)
        self.learned_chunk_concat_dim = int(learned_chunk_concat_dim)
        self.learned_chunk_blend = float(learned_chunk_blend)
        self.learned_chunking = (
            LearnedChunkingLayer(
                n_detectors=learned_chunk_detector_count,
                min_chunk_len=learned_chunk_min_len,
                max_chunk_len=learned_chunk_max_len,
                similarity_floor=learned_chunk_similarity_floor,
                boundary_threshold=learned_chunk_boundary_threshold,
                update_lr=learned_chunk_update_lr,
                association_blend=learned_chunk_association_blend,
                association_lr=learned_chunk_association_lr,
                association_decay=learned_chunk_association_decay,
                device=self.device,
            )
            if enable_learned_chunking
            else None
        )
        self._last_feature_vector_device: str | None = None
        self._last_feature_vector_shape: tuple[int, ...] | None = None
        self._last_spike_trace_device: str | None = None
        self._last_spike_trace_shape: tuple[int, ...] | None = None

    def _remember_feature_vector(self, vector: torch.Tensor) -> torch.Tensor:
        self._last_feature_vector_device = str(vector.device)
        self._last_feature_vector_shape = tuple(int(item) for item in vector.shape)
        return vector

    def _remember_spike_trace(self, trace: torch.Tensor) -> torch.Tensor:
        self._last_spike_trace_device = str(trace.device)
        self._last_spike_trace_shape = tuple(int(item) for item in trace.shape)
        return trace

    @classmethod
    def from_config(cls, config: "MarulhoConfig", device: torch.device | str | None = None) -> "RTFEncoder":
        return cls(
            window_size=config.window_size,
            representation=config.input_representation,
            hashed_ngram_dim=config.hashed_ngram_dim,
            hashed_ngram_min_n=config.hashed_ngram_min_n,
            hashed_ngram_max_n=config.hashed_ngram_max_n,
            enable_learned_chunking=config.enable_learned_chunking,
            learned_chunk_detector_count=config.learned_chunk_detector_count,
            learned_chunk_min_len=config.learned_chunk_min_len,
            learned_chunk_max_len=config.learned_chunk_max_len,
            learned_chunk_feature_mode=config.learned_chunk_feature_mode,
            learned_chunk_concat_dim=config.learned_chunk_concat_dim,
            learned_chunk_blend=config.learned_chunk_blend,
            learned_chunk_similarity_floor=config.learned_chunk_similarity_floor,
            learned_chunk_boundary_threshold=config.learned_chunk_boundary_threshold,
            learned_chunk_update_lr=config.learned_chunk_update_lr,
            learned_chunk_association_blend=config.learned_chunk_association_blend,
            learned_chunk_association_lr=config.learned_chunk_association_lr,
            learned_chunk_association_decay=config.learned_chunk_association_decay,
            device=config.resolve_device() if device is None else device,
        )

    def device_report(self) -> dict[str, object]:
        return {
            "encoder": "rtf",
            "device": str(self.device),
            "last_feature_vector_device": self._last_feature_vector_device,
            "last_feature_vector_shape": self._last_feature_vector_shape,
            "last_spike_trace_device": self._last_spike_trace_device,
            "last_spike_trace_shape": self._last_spike_trace_shape,
            "learned_chunking": None if self.learned_chunking is None else self.learned_chunking.device_report(),
        }

    @property
    def base_output_dim(self) -> int:
        return self.hashed_ngram_dim if self.representation == "hashed_ngram" else 128

    @property
    def uses_learned_chunking(self) -> bool:
        return self.learned_chunking is not None

    @property
    def uses_concat_chunk_channel(self) -> bool:
        return self.learned_chunking is not None and self.learned_chunk_feature_mode == "concat"

    @property
    def chunk_output_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.learned_chunk_concat_dim
        return self.base_output_dim

    @property
    def chunk_projection_work_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.output_dim
        return self.base_output_dim

    @property
    def output_dim(self) -> int:
        if self.uses_concat_chunk_channel:
            return self.base_output_dim + self.learned_chunk_concat_dim
        return self.base_output_dim

    def state_dict(self) -> dict[str, Any]:
        return {
            "learned_chunking": None if self.learned_chunking is None else self.learned_chunking.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self.learned_chunking is None:
            return
        learned_chunking = state.get("learned_chunking")
        if isinstance(learned_chunking, dict):
            self.learned_chunking.load_state_dict(learned_chunking)

    def character_window_to_pattern(self, chars: Iterable[int]) -> torch.Tensor:
        window: List[int] = list(chars)[-self.window_size :]
        pattern = torch.zeros(128, dtype=torch.float32, device=self.device)
        if not window:
            return pattern

        for c in window:
            if 0 <= c < 128:
                pattern[c] += 1.0

        return pattern / float(len(window))

    def _hash_ngram(self, ngram: Iterable[int]) -> int:
        hash_value = 2166136261
        for code in ngram:
            hash_value ^= int(code) + 1
            hash_value = (hash_value * 16777619) & 0xFFFFFFFF
        return int(hash_value % max(1, self.hashed_ngram_dim))

    def hashed_ngram_vector(self, chars: Iterable[int]) -> torch.Tensor:
        window: List[int] = [c for c in list(chars)[-self.window_size :] if 0 <= c < 128]
        vector = torch.zeros(self.hashed_ngram_dim, dtype=torch.float32, device=self.device)
        if not window:
            return vector

        min_n = self.hashed_ngram_min_n if len(window) >= self.hashed_ngram_min_n else 1
        max_n = min(self.hashed_ngram_max_n, len(window))
        for n in range(min_n, max_n + 1):
            for start in range(0, len(window) - n + 1):
                bucket = self._hash_ngram(window[start : start + n])
                vector[bucket] += 1.0

        return vector / (torch.norm(vector, p=2) + 1e-8)

    def routing_vector(self, chars: Iterable[int]) -> torch.Tensor:
        """Canonical routing representation used by Stage-0.

        This keeps dimensionality fixed at 128 while injecting order via
        latency-derived position weighting, so anagrams no longer collide.
        """
        window: List[int] = list(chars)[-self.window_size :]
        while len(window) < self.window_size:
            window.insert(0, 0)

        route = torch.zeros(128, dtype=torch.float32, device=self.device)
        for pos, c in enumerate(window):
            if 0 <= c < 128:
                latency = pos * self.t_spacing
                weight = max(0.0, (self.t_max - latency) / max(1e-8, self.t_max))
                route[c] += float(weight)

        return route / (torch.norm(route, p=2) + 1e-8)

    def _base_feature_vector(self, chars: Iterable[int]) -> torch.Tensor:
        if self.representation == "order_weighted_ascii":
            return self.routing_vector(chars)
        if self.representation == "unigram_ascii":
            return self.character_window_to_pattern(chars)
        if self.representation == "hashed_ngram":
            return self.hashed_ngram_vector(chars)
        raise ValueError(f"Unsupported representation: {self.representation}")

    def _project_chunk_vector(self, detector_vector: torch.Tensor) -> torch.Tensor:
        detector_vector = detector_vector.to(self.device)
        projected = torch.zeros(self.chunk_projection_work_dim, dtype=torch.float32, device=self.device)
        if detector_vector.numel() == 0 or float(detector_vector.sum().item()) <= 0.0:
            return projected
        if int(detector_vector.numel()) == int(projected.numel()):
            return _normalize(detector_vector.float())
        for index, value in enumerate(detector_vector.tolist()):
            if float(value) <= 0.0:
                continue
            projected[index % int(projected.numel())] += float(value)
        return _normalize(projected)

    def _chunk_signature_vector(self, chunk_codes: Sequence[int]) -> torch.Tensor:
        signature = torch.zeros(self.chunk_projection_work_dim, dtype=torch.float32, device=self.device)
        if not chunk_codes:
            return signature

        rolling = 2166136261
        for code in chunk_codes:
            rolling ^= int(code) + 1
            rolling = (rolling * 16777619) & 0xFFFFFFFF
            signature[int(rolling % max(1, self.chunk_projection_work_dim))] += 1.0

        for left, right in zip(chunk_codes, chunk_codes[1:]):
            pair_hash = ((int(left) + 17) * 1315423911) ^ ((int(right) + 31) * 2654435761)
            signature[int(pair_hash % max(1, self.chunk_projection_work_dim))] += 0.75

        signature[int(len(chunk_codes) % max(1, self.chunk_projection_work_dim))] += 0.5
        return _normalize(signature)

    def _chunk_projection(
        self,
        *,
        chunk_state: torch.Tensor | None = None,
        chunk_codes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if self.learned_chunking is None or chunk_state is None:
            return torch.zeros(self.chunk_output_dim, dtype=torch.float32, device=self.device)
        projected = self._project_chunk_vector(chunk_state)
        signature = self._chunk_signature_vector(chunk_codes or [])
        chunk_projection = _normalize(projected + signature)
        if self.uses_concat_chunk_channel and int(chunk_projection.numel()) != self.chunk_output_dim:
            chunk_projection = _normalize(chunk_projection[: self.chunk_output_dim])
        return chunk_projection

    def _combine_features(
        self,
        base: torch.Tensor,
        *,
        chunk_state: torch.Tensor | None = None,
        chunk_codes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        base_features = _normalize(base.float())
        if self.learned_chunking is None:
            return base_features

        chunk_projection = self._chunk_projection(chunk_state=chunk_state, chunk_codes=chunk_codes)
        if self.uses_concat_chunk_channel:
            return _normalize(torch.cat([base_features, chunk_projection], dim=0))
        if chunk_state is None or self.learned_chunk_blend <= 0.0:
            return base_features
        if float(chunk_projection.sum().item()) <= 0.0:
            return base_features
        blended = ((1.0 - self.learned_chunk_blend) * base_features) + (self.learned_chunk_blend * chunk_projection)
        return _normalize(blended)

    def feature_vector(self, chars: Iterable[int]) -> torch.Tensor:
        return self._remember_feature_vector(self._combine_features(self._base_feature_vector(chars)))

    def blended_feature_vector(
        self,
        chars: Iterable[int],
        *,
        chunk_state: torch.Tensor | None = None,
        chunk_codes: Sequence[int] | None = None,
    ) -> torch.Tensor:
        return self._remember_feature_vector(
            self._combine_features(
                self._base_feature_vector(chars),
                chunk_state=chunk_state,
                chunk_codes=chunk_codes,
            )
        )

    def _update_chunk_context(self, context: torch.Tensor, chunk_codes: Sequence[int]) -> torch.Tensor:
        if self.learned_chunking is None:
            return context
        chunk_vector = self.learned_chunking.detector_activations(chunk_codes)
        if float(chunk_vector.sum().item()) <= 0.0:
            return context
        if float(context.sum().item()) <= 0.0:
            return chunk_vector
        return _normalize((0.7 * context) + (0.3 * chunk_vector))

    def _current_chunk_state(self, context: torch.Tensor, chunk_codes: Sequence[int]) -> torch.Tensor:
        if self.learned_chunking is None:
            return torch.zeros(0, dtype=torch.float32, device=self.device)
        current = self.learned_chunking.detector_activations(chunk_codes)
        if float(current.sum().item()) <= 0.0:
            return context
        if float(context.sum().item()) <= 0.0:
            return current
        return _normalize(context + current)

    @staticmethod
    def _lexical_segments(text: str) -> list[str]:
        segments: list[str] = []
        chunk: list[str] = []
        for ch in str(text):
            is_separator = ch.isspace() or ch in ",.;:!?()[]{}<>\""
            if is_separator:
                if chunk:
                    if ch in ".!?":
                        chunk.append(ch)
                    segment = "".join(chunk).strip()
                    if segment:
                        segments.append(segment)
                    chunk = []
                continue
            chunk.append(ch)
        if chunk:
            segment = "".join(chunk).strip()
            if segment:
                segments.append(segment)
        return segments

    def iter_char_patterns(
        self,
        chars: Iterable[str],
        window_size: int,
        *,
        learn: bool = False,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        maxlen = max(1, int(window_size))
        window_codes: list[int] = []
        window_chars: list[str] = []
        chunk_codes: list[int] = []
        chunk_context = (
            torch.zeros(self.learned_chunking.n_detectors, dtype=torch.float32, device=self.device)
            if self.learned_chunking is not None
            else torch.zeros(0, dtype=torch.float32, device=self.device)
        )

        for ch in chars:
            code = _ascii_code(ch)
            display = ch if ord(ch) < 128 else "?"
            window_codes.append(code)
            window_chars.append(display)
            if len(window_codes) > maxlen:
                window_codes.pop(0)
                window_chars.pop(0)

            if self.learned_chunking is not None:
                if self.learned_chunking.is_separator(code):
                    if chunk_codes:
                        if learn:
                            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                        chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                        chunk_codes = []
                    if self.learned_chunking.is_hard_boundary(code):
                        chunk_context = torch.zeros_like(chunk_context)
                    chunk_state = chunk_context
                else:
                    if chunk_codes and self.learned_chunking.should_boundary(chunk_codes, code):
                        if learn:
                            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                        chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                        chunk_codes = [code]
                    else:
                        chunk_codes.append(code)
                    chunk_state = self._current_chunk_state(chunk_context, chunk_codes)
            else:
                chunk_state = None

            yield "".join(window_chars), self.blended_feature_vector(
                window_codes,
                chunk_state=chunk_state,
                chunk_codes=chunk_codes,
            )

        if learn and self.learned_chunking is not None and chunk_codes:
            self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)

    def segment_text(self, text: str, *, learn: bool = False) -> list[str]:
        if not text:
            return []
        if self.learned_chunking is None:
            return self._lexical_segments(str(text))

        segments: list[str] = []
        chunk_codes: list[int] = []
        chunk_chars: list[str] = []
        chunk_context = torch.zeros(self.learned_chunking.n_detectors, dtype=torch.float32, device=self.device)
        sentence_punct = {".", "!", "?"}
        for ch in text:
            code = _ascii_code(ch)
            if self.learned_chunking.is_separator(code):
                if chunk_chars:
                    if ch in sentence_punct:
                        chunk_chars.append(ch)
                    segment = "".join(chunk_chars).strip()
                    if segment:
                        segments.append(segment)
                    if learn:
                        self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                    chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                    chunk_codes = []
                    chunk_chars = []
                if self.learned_chunking.is_hard_boundary(code):
                    chunk_context = torch.zeros_like(chunk_context)
                continue

            if chunk_codes and self.learned_chunking.should_boundary(chunk_codes, code):
                segment = "".join(chunk_chars).strip()
                if segment:
                    segments.append(segment)
                if learn:
                    self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
                chunk_context = self._update_chunk_context(chunk_context, chunk_codes)
                chunk_codes = [code]
                chunk_chars = [ch]
            else:
                chunk_codes.append(code)
                chunk_chars.append(ch)

        if chunk_chars:
            segment = "".join(chunk_chars).strip()
            if segment:
                segments.append(segment)
            if learn:
                self.learned_chunking.learn_chunk(chunk_codes, context=chunk_context)
        return segments

    def iter_segment_patterns(
        self,
        text: str,
        window_size: int,
        *,
        learn: bool = False,
        context_segments: int = 5,
        use_learned_boundaries: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        maxlen = max(1, int(window_size))
        segment_window: list[str] = []
        max_segments = max(1, int(context_segments))
        segments = (
            self.segment_text(str(text), learn=learn)
            if use_learned_boundaries
            else self._lexical_segments(str(text))
        )
        for segment in segments:
            segment_window.append(segment)
            if len(segment_window) > max_segments:
                segment_window.pop(0)
            raw_window = " ".join(segment_window)
            codes = [_ascii_code(ch) for ch in raw_window][-maxlen:]
            if not codes:
                continue
            chunk_state = (
                self.learned_chunking.detector_activations(codes)
                if self.learned_chunking is not None and use_learned_boundaries
                else None
            )
            yield raw_window, self.blended_feature_vector(
                codes,
                chunk_state=chunk_state,
                chunk_codes=codes,
            )

    def encode(self, chars: Iterable[int], context_confidence: float) -> torch.Tensor:
        window: List[int] = list(chars)[-self.window_size :]
        while len(window) < self.window_size:
            window.insert(0, 0)

        spike_times = torch.full((128, self.n_bursts_max), -1.0, dtype=torch.float32, device=self.device)
        n_spikes = max(1, int(self.n_bursts_max * max(0.0, min(1.0, context_confidence))))
        burst_offsets = torch.arange(n_spikes, dtype=torch.float32, device=self.device) * 3.0

        for pos, c in enumerate(window):
            if 0 <= c < 128:
                spike_times[c, :n_spikes] = pos * self.t_spacing + burst_offsets

        return spike_times

    def spike_trace(
        self,
        chars: Iterable[int],
        context_confidence: float,
        *,
        tau: float | None = None,
        burst_decay: float = 0.85,
    ) -> torch.Tensor:
        trace_tau = float(self.t_spacing if tau is None else tau)
        if trace_tau <= 0.0:
            raise ValueError("tau must be positive")
        if not 0.0 < float(burst_decay) <= 1.0:
            raise ValueError("burst_decay must be in (0, 1]")

        return self._remember_spike_trace(
            self._spike_trace_fused(chars, context_confidence, trace_tau, burst_decay)
        )

    def _spike_trace_fused(
        self,
        chars: Iterable[int],
        context_confidence: float,
        tau: float,
        burst_decay: float,
    ) -> torch.Tensor:
        """Fused spike trace: skips the [128, n_bursts] intermediate tensor.

        Computes collapsed trace directly. Only the last occurrence of each
        character in the window is used (matching encode() overwrite semantics).
        """
        window: List[int] = list(chars)[-self.window_size :]
        while len(window) < self.window_size:
            window.insert(0, 0)

        n_spikes = max(1, int(self.n_bursts_max * max(0.0, min(1.0, context_confidence))))

        # Precompute burst kernel K (cached)
        cache_key = (n_spikes, tau, burst_decay)
        if getattr(self, '_fused_cache_key', None) != cache_key:
            j = torch.arange(n_spikes, dtype=torch.float32, device=self.device)
            self._fused_K = float(torch.sum(torch.exp(-j * 3.0 / tau) * (burst_decay ** j)).item())
            self._fused_cache_key = cache_key

        K = self._fused_K

        # Keep only the LAST position of each char (matching encode() overwrite)
        last_pos: dict[int, int] = {}
        for pos, c in enumerate(window):
            if 0 <= c < 128:
                last_pos[c] = pos

        if not last_pos:
            return torch.zeros(128, dtype=torch.float32, device=self.device)

        codes = torch.tensor(list(last_pos.keys()), dtype=torch.long, device=self.device)
        positions = torch.tensor(list(last_pos.values()), dtype=torch.float32, device=self.device)
        pos_weights = torch.exp(-positions * self.t_spacing / tau) * K

        collapsed = torch.zeros(128, dtype=torch.float32, device=self.device)
        collapsed[codes] = pos_weights

        total = collapsed.sum()
        if total <= 0.0:
            return collapsed
        return collapsed / (total + 1e-8)
