"""Developmental protocol runners.

Implements the five-stage developmental protocol (§7):
  Stage 1: Critical period -- curated multimodal, no alignment filter
  Stage 2: Self-filtering -- alignment filter active
  Stage 3: Confirmation-seeking -- curiosity-driven gap filling
  Stage 4: Semi-autonomous -- any multimodal, no curation
  Stage 5: Fully autonomous -- self-directed curriculum

State continuity: a ProtocolState object carries trainer, encoder,
and visual/audio encoders across all stages so that each stage
inherits the previous stage's learned weights and confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.core.cross_modal import CrossModalGroundingLayer
from hecsn.data.event_camera_encoder import EventCameraEncoder
from hecsn.data.cochleagram_encoder import CochleagramEncoder
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.evaluation.grounding_probe import evaluate_grounding_probe
from hecsn.semantics.geometric_curiosity import GeometricCuriosityController
from hecsn.training.runner_utils import set_seed
from hecsn.training.query_runner import feed_text
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


@dataclass
class ProtocolState:
    """Carries trainer + encoders across developmental stages.

    Separate from StageResult (metrics-only, JSON-serializable).
    """

    trainer: HECSNTrainer
    text_encoder: RTFEncoder
    config: HECSNConfig


@dataclass
class StageResult:
    """Result of a developmental stage run."""

    stage: int
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    tokens_processed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "passed": self.passed,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "tokens_processed": self.tokens_processed,
        }


def _make_config_for_stage(
    stage: int, base_config: HECSNConfig | None = None
) -> HECSNConfig:
    """Create a model config appropriate for the given developmental stage.

    Activates the mechanisms each stage depends on:
      All stages: cross-modal, adaptive context, triplet STDP
      Stage 3+: abstraction layer (needed for curiosity controller)
    """
    cfg = base_config if base_config is not None else HECSNConfig()
    cfg.context_mode = "adaptive"
    cfg.plasticity_rule = "triplet"
    cfg.enable_cross_modal = True
    if stage >= 3:
        cfg.enable_abstraction_layer = True
    return cfg


def _compute_grounding_confidence(cross_modal: CrossModalGroundingLayer) -> float:
    """Compute mean grounding confidence (visual + audio) across top-100 text dims."""
    conf = cross_modal.grounding_confidence().detach()
    if conf.numel() == 0:
        return 0.0
    top_k = min(100, conf.numel())
    top_conf, _ = conf.topk(top_k)
    return float(top_conf.mean().item())


def _make_vector_fn(
    trainer: HECSNTrainer, encoder: RTFEncoder, cfg: HECSNConfig
):
    """Build vector_fn callable for grounding probe from trainer + encoder.

    When cross-modal grounding is enabled, blends the routing key with
    visual feedback so that the probe measures cross-modal state (§8.7).
    """

    def vector_fn(text: str) -> torch.Tensor:
        patterns = list(
            encoder.iter_char_patterns(text, cfg.window_size, learn=False)
        )
        if not patterns:
            return torch.zeros(cfg.input_dim)
        vecs = [p for _, p in patterns]
        routing_key = torch.stack(vecs).mean(dim=0)

        cross_modal = trainer.model.cross_modal
        if cross_modal is not None and routing_key.shape[0] == cross_modal.W_tv.shape[0]:
            pred_visual = torch.mv(cross_modal.W_tv.T, routing_key)
            visual_conf = float(cross_modal.visual_confidence.mean().item())
            if pred_visual.norm() > 1e-6 and visual_conf > 0.01:
                visual_feedback = torch.mv(cross_modal.W_vt.T, pred_visual)
                if visual_feedback.shape == routing_key.shape:
                    blend = min(0.3, visual_conf)
                    routing_key = (1.0 - blend) * routing_key + blend * visual_feedback
        return routing_key

    return vector_fn


def _train_on_corpus(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    corpus: list[str],
    n_tokens: int,
) -> int:
    """Feed corpus through trainer via feed_text, repeating to reach n_tokens."""
    total = 0
    full_text = " ".join(corpus)
    iterations = max(1, n_tokens // max(1, len(full_text)))
    for _ in range(iterations):
        result = feed_text(trainer, encoder, full_text)
        total += result["tokens_processed"]
        if total >= n_tokens:
            break
    return total


# ------------------------------------------------------------------
# Concept-conditioned synthetic multimodal data (§7.1)
# ------------------------------------------------------------------

# Each concept maps to a fixed visual/audio spike signature.
# Different concepts get distinguishable patterns (seeded).
# This replaces the prior torch.randn() random noise approach.

CONCEPT_VOCABULARY = [
    "fire", "water", "tree", "bird", "sun",
    "rock", "wind", "rain", "snow", "flower",
    "mountain", "river", "cloud", "star", "moon",
]


def _build_concept_signatures(
    n_concepts: int,
    dim_visual: int,
    dim_audio: int,
    seed: int = 42,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build fixed visual/audio spike signatures for each concept.

    Each concept gets a sparse, distinct pattern.  Signatures are
    deterministic for the same seed so that repeated exposures to the
    same concept produce the same cross-modal pairing.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    concepts = CONCEPT_VOCABULARY[:n_concepts]
    signatures: dict[str, dict[str, torch.Tensor]] = {}

    for i, concept in enumerate(concepts):
        # Visual: sparse activation pattern (10-20% active)
        visual_base = torch.zeros(dim_visual)
        n_active_v = max(2, dim_visual // 8)
        # Use a concept-specific offset to ensure distinct patterns
        indices_v = torch.randperm(dim_visual, generator=gen)[:n_active_v]
        visual_base[indices_v] = torch.rand(n_active_v, generator=gen) * 0.3 + 0.1

        # Audio: sparse activation pattern (10-20% active)
        audio_base = torch.zeros(dim_audio)
        n_active_a = max(2, dim_audio // 8)
        indices_a = torch.randperm(dim_audio, generator=gen)[:n_active_a]
        audio_base[indices_a] = torch.rand(n_active_a, generator=gen) * 0.3 + 0.1

        signatures[concept] = {
            "visual": visual_base,
            "audio": audio_base,
        }

    return signatures


def _concept_spikes_for_text(
    text: str,
    signatures: dict[str, dict[str, torch.Tensor]],
    dim_visual: int,
    dim_audio: int,
    noise_scale: float = 0.02,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Generate concept-conditioned visual/audio spikes for a text chunk.

    If the text contains known concept words, returns the corresponding
    visual/audio signatures (blended if multiple concepts present, with
    small noise for biological realism). Returns None if no concept found.
    """
    words = set(text.lower().split())
    matched = [c for c in signatures if c in words]
    if not matched:
        return None, None

    visual = torch.zeros(dim_visual)
    audio = torch.zeros(dim_audio)
    for concept in matched:
        visual = visual + signatures[concept]["visual"]
        audio = audio + signatures[concept]["audio"]
    visual = visual / len(matched)
    audio = audio / len(matched)

    # Add small noise for biological realism
    if noise_scale > 0:
        visual = visual + torch.randn_like(visual) * noise_scale
        audio = audio + torch.randn_like(audio) * noise_scale
        visual = visual.clamp(min=0)
        audio = audio.clamp(min=0)

    return visual, audio


def _train_multimodal_on_corpus(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    corpus: list[str],
    n_tokens: int,
    signatures: dict[str, dict[str, torch.Tensor]],
    dim_visual: int,
    dim_audio: int,
) -> tuple[int, int, int]:
    """Train on corpus with concept-conditioned multimodal spikes.

    Returns (tokens_processed, visual_pairs_sent, audio_pairs_sent).
    """
    total = 0
    visual_count = 0
    audio_count = 0
    full_text = " ".join(corpus)
    iterations = max(1, n_tokens // max(1, len(full_text)))

    for _ in range(iterations):
        # For each sentence, generate concept spikes
        for sentence in corpus:
            vs, aus = _concept_spikes_for_text(
                sentence, signatures, dim_visual, dim_audio,
            )
            # Feed sentence with multimodal context
            patterns = list(
                encoder.iter_char_patterns(sentence, trainer.config.window_size)
            )
            for raw_window, pattern_vec in patterns:
                metrics = trainer.train_step(
                    pattern_vec,
                    raw_window=raw_window,
                    visual_spikes=vs,
                    audio_spikes=aus,
                )
                total += 1
                if vs is not None:
                    visual_count += 1
                if aus is not None:
                    audio_count += 1
                if total >= n_tokens:
                    return total, visual_count, audio_count

    return total, visual_count, audio_count


# ------------------------------------------------------------------
# Stage runners
# ------------------------------------------------------------------


def run_stage_1(
    config: HECSNConfig | None = None,
    n_tokens: int = 5000,
    seed: int = 7,
    state: ProtocolState | None = None,
) -> tuple[StageResult, ProtocolState]:
    """Stage 1: Critical period -- curated multimodal, no alignment filter.

    Establishes initial cross-modal associations by presenting concept-
    conditioned visual/audio spikes alongside text.  No alignment filter
    is active (developmental_stage=1), so all multimodal pairs are accepted.

    Returns (StageResult, ProtocolState) — the state carries learned weights
    to Stage 2.
    """
    set_seed(seed)
    cfg = _make_config_for_stage(1, config)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
    else:
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 1

    corpus = [
        "the fire burns bright in the dark",
        "water flows down the rocky stream",
        "the tall tree grows in the forest",
        "a small bird flies above the mountain",
        "warm sunlight covers the sandy beach",
    ]

    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio
    signatures = _build_concept_signatures(
        n_concepts=len(CONCEPT_VOCABULARY),
        dim_visual=dim_visual,
        dim_audio=dim_audio,
        seed=seed,
    )

    tokens_processed, visual_pairs, audio_pairs = _train_multimodal_on_corpus(
        trainer, encoder, corpus, n_tokens, signatures, dim_visual, dim_audio,
    )

    grounding_confidence = 0.0
    if trainer.model.cross_modal is not None:
        grounding_confidence = _compute_grounding_confidence(trainer.model.cross_modal)

    # Criterion: grounding_confidence > 0.40 (paper §7.3)
    passed = grounding_confidence > 0.40

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg)

    return StageResult(
        stage=1,
        passed=passed,
        metrics={
            "grounding_confidence": grounding_confidence,
            "visual_pairs_sent": visual_pairs,
            "audio_pairs_sent": audio_pairs,
        },
        diagnostics={
            "completion_criterion": "grounding_confidence > 0.40",
        },
        tokens_processed=tokens_processed,
    ), out_state


def run_stage_2(
    config: HECSNConfig | None = None,
    n_tokens: int = 5000,
    seed: int = 7,
    state: ProtocolState | None = None,
) -> tuple[StageResult, ProtocolState]:
    """Stage 2: Self-filtering -- alignment filter active.

    Receives Stage 1's trained state.  Trainer.developmental_stage=2
    activates alignment_gate() in train_step() after a bootstrap
    budget of ungated pairs.
    """
    set_seed(seed)
    cfg = _make_config_for_stage(2, config)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
    else:
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 2
    trainer._stage2_bootstrap_used = 0

    corpus = [
        "fire burns bright",
        "water flows down",
        "tree grows tall",
        "rock is heavy",
        "bird flies high",
        "sun shines warm",
    ]

    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio
    signatures = _build_concept_signatures(
        n_concepts=len(CONCEPT_VOCABULARY),
        dim_visual=dim_visual,
        dim_audio=dim_audio,
        seed=seed,
    )

    tokens_processed, visual_pairs, audio_pairs = _train_multimodal_on_corpus(
        trainer, encoder, corpus, n_tokens, signatures, dim_visual, dim_audio,
    )

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    probe_result = evaluate_grounding_probe(vector_fn)

    # Compute filter statistics from trainer's gating metrics
    grounding_confidence = 0.0
    if trainer.model.cross_modal is not None:
        grounding_confidence = _compute_grounding_confidence(trainer.model.cross_modal)

    # Criteria (§7.3): probe > 0.60 AND grounding growth
    passed = (
        probe_result.total_accuracy > 0.60
        and grounding_confidence > 0.30
    )

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg)

    return StageResult(
        stage=2,
        passed=passed,
        metrics={
            "probe_accuracy": probe_result.total_accuracy,
            "concrete_accuracy": probe_result.concrete_accuracy,
            "abstract_accuracy": probe_result.abstract_accuracy,
            "concreteness_gap": probe_result.concreteness_gap,
            "grounding_confidence": grounding_confidence,
            "visual_pairs_sent": visual_pairs,
            "audio_pairs_sent": audio_pairs,
        },
        diagnostics={
            "completion_criteria": {
                "probe_accuracy_target": "> 0.60",
                "grounding_confidence_target": "> 0.30",
            },
        },
        tokens_processed=tokens_processed,
    ), out_state


def run_stage_3(
    config: HECSNConfig | None = None,
    n_tokens: int = 5000,
    seed: int = 7,
    state: ProtocolState | None = None,
) -> tuple[StageResult, ProtocolState]:
    """Stage 3: Confirmation-seeking -- curiosity-driven gap filling.

    Uses the GeometricCuriosityController to identify abstraction-layer gaps,
    then trains on gap-relevant corpus to confirm/fill those gaps.
    Tracks grounding confidence delta rather than random alignment.
    """
    set_seed(seed)
    cfg = _make_config_for_stage(3, config)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        # Stage 3 needs abstraction layer — ensure it exists
        if trainer.model.abstraction_layer is None and cfg.enable_abstraction_layer:
            from hecsn.core.abstraction import AbstractionLayer
            trainer.model.abstraction_layer = AbstractionLayer(
                n_columns=cfg.n_columns,
                n_concepts=cfg.abstraction_n_concepts,
                device=trainer.model.device,
                slow_rate=cfg.abstraction_slow_rate,
                fast_rate=cfg.abstraction_fast_rate,
                learning_rate=cfg.abstraction_learning_rate,
                feedback_lr=cfg.abstraction_feedback_lr,
                feedback_strength=cfg.abstraction_feedback_strength,
            )
    else:
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 3

    corpus = [
        "the fire burns in the dark night",
        "water flows through the rocky river",
        "a tall tree stands in the green forest",
        "the bird flies above the mountain top",
        "warm sunlight covers the sandy beach",
    ]

    # Initial training to build representations
    tokens_processed = _train_on_corpus(trainer, encoder, corpus, n_tokens // 2)

    # Curiosity-driven confirmation phase
    curiosity = GeometricCuriosityController(trainer.model.abstraction_layer)
    confirmation_cycles = 0
    gap_queries_produced = 0
    grounding_deltas: list[float] = []

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    pre_probe = evaluate_grounding_probe(vector_fn)

    for cycle in range(min(10, n_tokens // 500)):
        # Update lexicon with current concept activations
        if trainer.model.abstraction_layer is not None:
            activations = trainer.model.abstraction_layer.last_activations
            if activations is not None:
                curiosity.update_lexicon(activations, corpus)

        # Get curiosity-driven focus plan
        plan = curiosity.focus_plan(top_n=3)
        if plan is not None:
            gap_queries_produced += len(plan.get("retrieval_queries", []))

        # Train on gap-relevant sentences
        gap_corpus = corpus[cycle % len(corpus)]
        result = feed_text(trainer, encoder, gap_corpus)
        tokens_processed += result["tokens_processed"]
        confirmation_cycles += 1

        # Track grounding confidence delta
        if trainer.model.cross_modal is not None:
            conf = _compute_grounding_confidence(trainer.model.cross_modal)
            grounding_deltas.append(conf)

    # Remaining tokens with self-criticism loop every 5000 tokens
    remaining = max(0, n_tokens - tokens_processed)
    self_criticism_stats: dict[str, Any] = {}
    blacklist: dict[int, int] = {}
    if remaining > 0:
        tokens_processed += _train_on_corpus(trainer, encoder, corpus, remaining)

    # Run self-criticism if cross-modal is enabled (§7.4)
    if trainer.model.cross_modal is not None:
        recent_frames = trainer._recent_visual_frames
        if len(recent_frames) >= 3:
            self_criticism_stats = trainer.model.cross_modal.run_self_criticism(
                recent_visual_frames=recent_frames,
                blacklist=blacklist,
            )

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    post_probe = evaluate_grounding_probe(vector_fn)

    # Criteria: no probe regression AND curiosity system active
    no_regression = post_probe.total_accuracy >= pre_probe.total_accuracy - 0.05
    curiosity_active = gap_queries_produced > 0
    passed = no_regression and curiosity_active

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg)

    return StageResult(
        stage=3,
        passed=passed,
        metrics={
            "probe_accuracy": post_probe.total_accuracy,
            "probe_accuracy_delta": post_probe.total_accuracy - pre_probe.total_accuracy,
            "concrete_accuracy": post_probe.concrete_accuracy,
            "abstract_accuracy": post_probe.abstract_accuracy,
            "concreteness_gap": post_probe.concreteness_gap,
            "confirmation_cycles": confirmation_cycles,
            "gap_queries_produced": gap_queries_produced,
            "mean_grounding_confidence": sum(grounding_deltas) / max(1, len(grounding_deltas)),
            "self_criticism_checked": self_criticism_stats.get("checked", 0),
            "self_criticism_penalised": self_criticism_stats.get("penalised", 0),
            "self_criticism_blacklisted": self_criticism_stats.get("blacklisted", 0),
        },
        diagnostics={
            "completion_criteria": {
                "probe_no_regression": "post >= pre - 0.05",
                "gap_queries": "> 0 (curiosity system active)",
            },
        },
        tokens_processed=tokens_processed,
    ), out_state


def run_stage_4(
    config: HECSNConfig | None = None,
    n_tokens: int = 5000,
    seed: int = 7,
    state: ProtocolState | None = None,
) -> tuple[StageResult, ProtocolState]:
    """Stage 4: Semi-autonomous -- gap-directed acquisition.

    Identifies gaps via the abstraction layer, selects corpus segments
    that address those gaps, and trains on them. Verifies probe accuracy
    doesn't regress.
    """
    set_seed(seed)
    cfg = _make_config_for_stage(4, config)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
    else:
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 4

    # Diverse corpus for gap-directed selection (disjoint from probe vocabulary)
    acquisition_corpus = [
        "quantum mechanics describes particle behavior at small scales",
        "photosynthesis converts sunlight into chemical energy in plants",
        "plate tectonics explains how continents drift over geological time",
        "evolution shapes organisms through natural selection pressure",
        "gravity pulls objects toward each other with measurable force",
        "neural networks process information through connected layers",
        "climate systems regulate temperature across the entire planet",
        "cellular division creates new organisms from existing cells",
    ]

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    initial_probe = evaluate_grounding_probe(vector_fn)

    tokens_processed = 0
    acquisitions_made = 0
    curiosity = GeometricCuriosityController(trainer.model.abstraction_layer)

    for cycle in range(min(8, n_tokens // 500)):
        # Update lexicon from current training state
        if trainer.model.abstraction_layer is not None:
            activations = trainer.model.abstraction_layer.last_activations
            if activations is not None:
                curiosity.update_lexicon(activations, acquisition_corpus)

        # Select acquisition target based on gap score
        plan = curiosity.focus_plan(top_n=2)
        if plan is not None and plan.get("retrieval_queries"):
            # Pick the corpus sentence closest to the gap query
            query = plan["retrieval_queries"][0]
            best_sentence = max(
                acquisition_corpus,
                key=lambda s: sum(1 for w in query.lower().split() if w in s.lower()),
            )
        else:
            best_sentence = acquisition_corpus[cycle % len(acquisition_corpus)]

        result = feed_text(trainer, encoder, best_sentence)
        tokens_processed += result["tokens_processed"]
        acquisitions_made += 1

    # Fill remaining tokens
    remaining = max(0, n_tokens - tokens_processed)
    if remaining > 0:
        tokens_processed += _train_on_corpus(trainer, encoder, acquisition_corpus, remaining)

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    final_probe = evaluate_grounding_probe(vector_fn)

    no_regression = final_probe.total_accuracy >= initial_probe.total_accuracy - 0.10

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg)

    return StageResult(
        stage=4,
        passed=no_regression,
        metrics={
            "initial_probe_accuracy": initial_probe.total_accuracy,
            "final_probe_accuracy": final_probe.total_accuracy,
            "accuracy_delta": final_probe.total_accuracy - initial_probe.total_accuracy,
            "concreteness_gap": final_probe.concreteness_gap,
            "acquisitions_made": acquisitions_made,
        },
        diagnostics={
            "completion_criterion": "no regression (final >= initial - 0.10)",
        },
        tokens_processed=tokens_processed,
    ), out_state


def run_stage_5(
    config: HECSNConfig | None = None,
    n_tokens: int = 5000,
    seed: int = 7,
    state: ProtocolState | None = None,
) -> tuple[StageResult, ProtocolState]:
    """Stage 5: Fully autonomous -- continuous self-directed curriculum.

    Runs multiple back-to-back acquisition cycles autonomously.
    Verifies no catastrophic forgetting (probe doesn't degrade) and
    that the system can sustain learning without external guidance.
    """
    set_seed(seed)
    cfg = _make_config_for_stage(5, config)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
    else:
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 5

    # Autonomous exploration corpus
    autonomous_corpus = [
        "algorithms optimize computational efficiency in modern systems",
        "biodiversity preserves ecosystem stability through redundancy",
        "electromagnetic waves carry energy across vast distances",
        "metabolism converts nutrients into cellular energy continuously",
        "ocean currents distribute heat around the global climate system",
        "symbiotic relationships benefit multiple organisms simultaneously",
        "thermodynamics governs energy transfer in physical processes",
        "volcanic eruptions reshape landscapes through geological forces",
    ]

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    initial_probe = evaluate_grounding_probe(vector_fn)

    tokens_processed = 0
    autonomous_cycles = 0
    cycle_accuracies: list[float] = []
    curiosity = GeometricCuriosityController(trainer.model.abstraction_layer)

    for cycle in range(min(12, n_tokens // 400)):
        # Autonomous selection — no external guidance
        if trainer.model.abstraction_layer is not None:
            activations = trainer.model.abstraction_layer.last_activations
            if activations is not None:
                curiosity.update_lexicon(activations, autonomous_corpus)

        plan = curiosity.focus_plan(top_n=2)
        if plan is not None and plan.get("retrieval_queries"):
            query = plan["retrieval_queries"][0]
            target = max(
                autonomous_corpus,
                key=lambda s: sum(1 for w in query.lower().split() if w in s.lower()),
            )
        else:
            target = autonomous_corpus[cycle % len(autonomous_corpus)]

        result = feed_text(trainer, encoder, target)
        tokens_processed += result["tokens_processed"]
        autonomous_cycles += 1

        # Periodic probe to check for catastrophic forgetting
        if cycle % 4 == 3:
            vfn = _make_vector_fn(trainer, encoder, cfg)
            mid_probe = evaluate_grounding_probe(vfn)
            cycle_accuracies.append(mid_probe.total_accuracy)

    # Fill remaining
    remaining = max(0, n_tokens - tokens_processed)
    if remaining > 0:
        tokens_processed += _train_on_corpus(trainer, encoder, autonomous_corpus, remaining)

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    final_probe = evaluate_grounding_probe(vector_fn)

    no_catastrophic_forgetting = final_probe.total_accuracy >= initial_probe.total_accuracy - 0.15
    sustained_learning = len(cycle_accuracies) == 0 or min(cycle_accuracies) >= initial_probe.total_accuracy - 0.20

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg)

    return StageResult(
        stage=5,
        passed=no_catastrophic_forgetting and sustained_learning,
        metrics={
            "initial_probe_accuracy": initial_probe.total_accuracy,
            "final_probe_accuracy": final_probe.total_accuracy,
            "accuracy_delta": final_probe.total_accuracy - initial_probe.total_accuracy,
            "concreteness_gap": final_probe.concreteness_gap,
            "autonomous_cycles": autonomous_cycles,
            "min_mid_accuracy": min(cycle_accuracies) if cycle_accuracies else None,
            "no_catastrophic_forgetting": no_catastrophic_forgetting,
            "sustained_learning": sustained_learning,
        },
        diagnostics={
            "completion_criteria": {
                "no_catastrophic_forgetting": "final >= initial - 0.15",
                "sustained_learning": "mid probes >= initial - 0.20",
            },
        },
        tokens_processed=tokens_processed,
    ), out_state


# ------------------------------------------------------------------
# Full protocol
# ------------------------------------------------------------------


def run_full_developmental_protocol(
    config: HECSNConfig | None = None,
    n_tokens_per_stage: int = 2000,
    seed: int = 7,
    output_dir: Path | None = None,
) -> list[StageResult]:
    """Run the complete 5-stage developmental protocol with state continuity.

    Each stage's trained weights, confidence, and encoder state are passed
    to the next stage via ProtocolState.  If a stage fails, the protocol
    stops (the paper defines each stage as prerequisite for the next).
    """
    results: list[StageResult] = []
    state: ProtocolState | None = None

    runners = [
        (1, run_stage_1),
        (2, run_stage_2),
        (3, run_stage_3),
        (4, run_stage_4),
        (5, run_stage_5),
    ]

    for stage_num, runner_fn in runners:
        result, state = runner_fn(
            config=config,
            n_tokens=n_tokens_per_stage,
            seed=seed,
            state=state,
        )
        results.append(result)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            stage_file = output_dir / f"stage_{stage_num}.json"
            stage_file.write_text(json.dumps(result.to_dict(), indent=2))

        if not result.passed:
            break

    if output_dir is not None:
        summary = {
            "stages_completed": [r.stage for r in results if r.passed],
            "stages_failed": [r.stage for r in results if not r.passed],
            "total_tokens": sum(r.tokens_processed for r in results),
            "results": [r.to_dict() for r in results],
        }
        (output_dir / "developmental_summary.json").write_text(
            json.dumps(summary, indent=2)
        )

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run HECSN developmental protocol")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/developmental"))
    parser.add_argument("--n-tokens", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = run_full_developmental_protocol(
        n_tokens_per_stage=args.n_tokens,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[developmental] Stage {r.stage}: {status}")
        for k, v in r.metrics.items():
            if isinstance(v, float):
                print(f"  {k} = {v:.4f}")
            else:
                print(f"  {k} = {v}")
