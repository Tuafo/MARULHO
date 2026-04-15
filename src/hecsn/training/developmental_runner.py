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
import torch.nn.functional as F

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
    concept_signatures: dict[str, dict[str, torch.Tensor]] | None = None


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

    Activates layers progressively following the developmental protocol:
      All stages: context layer (adaptive), cross-modal, full triplet STDP
      Stage 2+: binding layer (requires grounded associations from Stage 1)
      Stage 3+: abstraction layer (needed for curiosity controller)

    Never mutates the caller's config — always returns a fresh copy.
    """
    import copy
    cfg = copy.deepcopy(base_config) if base_config is not None else HECSNConfig()
    cfg.context_mode = "adaptive"
    cfg.enable_context_layer = True
    cfg.plasticity_rule = "triplet"
    cfg.plasticity_mode = "local_stdp"
    cfg.enable_cross_modal = True
    if stage >= 2:
        cfg.enable_binding_layer = True
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

    Produces a grounded representation by:
    1. Encoding text → routing_key via competitive column assembly
    2. Predicting visual/audio from raw text pattern via cross-modal weights
    3. Weighting predictions by per-word grounding confidence
    4. Concatenating [routing_key_norm, visual * word_conf, audio * word_conf]

    Per-word confidence is key: only words that were actually trained with
    cross-modal pairs develop high confidence.  Abstract words that were
    never paired with sensory data retain near-zero confidence, so their
    cross-modal predictions are suppressed → positive concreteness gap.
    """

    def vector_fn(text: str) -> torch.Tensor:
        patterns = list(
            encoder.iter_char_patterns(text, cfg.window_size, learn=False)
        )
        if not patterns:
            return torch.zeros(cfg.column_latent_dim)
        vecs = [p for _, p in patterns]
        raw_pattern = torch.stack(vecs).mean(dim=0)

        routing_key = trainer.model.routing_key_from_pattern(raw_pattern)

        cross_modal = trainer.model.cross_modal
        if cross_modal is None:
            return routing_key.cpu()

        # Probe runs on CPU regardless of model device
        routing_key = routing_key.cpu()

        # Per-word grounding confidence
        word = text.lower().strip()
        word_conf = trainer.word_grounding_confidence.get(word, 0.0)

        # Use per-word accumulated sensory signatures (cell assembly
        # encoding) instead of W_tv predictions.  Each grounded word's
        # signature reflects the actual visual/audio patterns it was
        # paired with during training — a direct, discriminative
        # representation that doesn't suffer from text-pattern overlap.
        vis_sig = trainer.word_visual_signature.get(word)
        aud_sig = trainer.word_audio_signature.get(word)

        if vis_sig is not None:
            vis_centered = vis_sig.cpu() - vis_sig.cpu().mean()
            vis_n = vis_centered.norm()
            vis_part = (vis_centered / vis_n) * word_conf if vis_n > 1e-8 else vis_centered
        else:
            vis_part = torch.zeros(cfg.cross_modal_dim_visual)

        if aud_sig is not None:
            aud_centered = aud_sig.cpu() - aud_sig.cpu().mean()
            aud_n = aud_centered.norm()
            aud_part = (aud_centered / aud_n) * word_conf if aud_n > 1e-8 else aud_centered
        else:
            aud_part = torch.zeros(cfg.cross_modal_dim_audio)

        # Grounding replaces text encoding: as cross-modal confidence
        # grows, the routing_key fades and sensory signatures dominate.
        rk_weight = 1.0 - word_conf
        rk_norm = (
            F.normalize(routing_key.unsqueeze(0), dim=1).squeeze(0) * rk_weight
        )

        grounded = torch.cat([rk_norm, vis_part, aud_part])
        return grounded

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
    # Fire family — share hot/orange/crackling attributes
    "fire", "flame", "heat", "burn", "lava", "ash",
    # Water family — share blue/flowing/rushing attributes
    "water", "ocean", "wave", "flow", "rain", "wet", "swim", "crash",
    # Cold family — share white/still/crisp attributes
    "ice", "snow", "cold", "frost", "freeze", "white", "crystal",
    # Earth family — share brown/rough/heavy attributes
    "rock", "stone", "mountain", "earth", "sand", "hard",
    "earthquake", "shake", "desert", "dry", "stable",
    # Air family — share light/moving/whooshing attributes
    "wind", "breeze", "cloud", "sky", "flight", "fly",
    # Plant family — share green/organic/rustling attributes
    "tree", "leaf", "flower", "rose", "petal", "thorn",
    # Animal family — distinct but all animate/moving
    "dog", "bird", "fish", "feather",
    # Light family — share bright/yellow/flash attributes
    "sun", "light", "lightning", "flash",
    # Dark family — absence of light
    "darkness", "shadow", "black",
    # Sound family — share intense/vibrating audio
    "thunder", "loud", "bark",
    # Quiet family — absence of sound
    "silence", "quiet", "calm", "still", "stillness",
    # Tool family — hard/sharp/metallic attributes
    "hammer", "knife", "metal", "sharp",
    # Texture family — soft/yielding attributes
    "soft", "cotton",
    # Misc concrete — individual signatures
    "star", "moon", "river", "smoke", "volcano", "scale",
]

# Concept families: members share visual/audio base patterns
_CONCEPT_FAMILIES: dict[str, list[str]] = {
    "fire": ["fire", "flame", "heat", "burn", "lava", "ash"],
    "water": ["water", "ocean", "wave", "flow", "rain", "wet", "swim", "crash"],
    "cold": ["ice", "snow", "cold", "frost", "freeze", "white", "crystal"],
    "earth": [
        "rock", "stone", "mountain", "earth", "sand", "hard",
        "earthquake", "shake", "desert", "dry", "stable",
    ],
    "air": ["wind", "breeze", "cloud", "sky", "flight", "fly"],
    "plant": ["tree", "leaf", "flower", "rose", "petal", "thorn"],
    "animal": ["dog", "bird", "fish", "feather"],
    "light": ["sun", "light", "lightning", "flash"],
    "dark": ["darkness", "shadow", "black"],
    "sound": ["thunder", "loud", "bark"],
    "quiet": ["silence", "quiet", "calm", "still", "stillness"],
    "tool": ["hammer", "knife", "metal", "sharp"],
    "texture": ["soft", "cotton"],
}


def _build_concept_signatures(
    n_concepts: int,
    dim_visual: int,
    dim_audio: int,
    seed: int = 42,
) -> dict[str, dict[str, torch.Tensor]]:
    """Build fixed visual/audio spike signatures for each concept.

    Concepts within the same family share a base pattern (with small
    per-member variation), so semantically related words produce
    correlated multimodal representations — just as in biological
    grounding.  Signatures are deterministic for the same seed.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    concepts = CONCEPT_VOCABULARY[:n_concepts]
    signatures: dict[str, dict[str, torch.Tensor]] = {}

    # Build one base pattern per family
    concept_to_family: dict[str, str] = {}
    for fam_name, members in _CONCEPT_FAMILIES.items():
        for m in members:
            concept_to_family[m] = fam_name

    family_bases: dict[str, dict[str, torch.Tensor]] = {}
    for fam_name in _CONCEPT_FAMILIES:
        n_active_v = max(4, dim_visual // 6)
        indices_v = torch.randperm(dim_visual, generator=gen)[:n_active_v]
        vbase = torch.zeros(dim_visual)
        vbase[indices_v] = torch.rand(n_active_v, generator=gen) * 0.3 + 0.1

        n_active_a = max(3, dim_audio // 6)
        indices_a = torch.randperm(dim_audio, generator=gen)[:n_active_a]
        abase = torch.zeros(dim_audio)
        abase[indices_a] = torch.rand(n_active_a, generator=gen) * 0.3 + 0.1

        family_bases[fam_name] = {"visual": vbase, "audio": abase}

    for concept in concepts:
        fam = concept_to_family.get(concept)
        if fam and fam in family_bases:
            # Family member: shared base + small per-member variation
            visual = family_bases[fam]["visual"].clone()
            audio = family_bases[fam]["audio"].clone()
            visual += torch.randn(dim_visual, generator=gen) * 0.03
            audio += torch.randn(dim_audio, generator=gen) * 0.03
            visual = visual.clamp(min=0)
            audio = audio.clamp(min=0)
        else:
            # Standalone concept: independent random pattern
            visual = torch.zeros(dim_visual)
            n_active_v = max(2, dim_visual // 8)
            indices_v = torch.randperm(dim_visual, generator=gen)[:n_active_v]
            visual[indices_v] = torch.rand(n_active_v, generator=gen) * 0.3 + 0.1

            audio = torch.zeros(dim_audio)
            n_active_a = max(2, dim_audio // 8)
            indices_a = torch.randperm(dim_audio, generator=gen)[:n_active_a]
            audio[indices_a] = torch.rand(n_active_a, generator=gen) * 0.3 + 0.1

        signatures[concept] = {"visual": visual, "audio": audio}

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


def _resolve_signatures(
    state: ProtocolState | None,
    cfg: HECSNConfig,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return concept signatures from state if available, else build fresh."""
    if state is not None and state.concept_signatures is not None:
        return state.concept_signatures
    return _build_concept_signatures(
        n_concepts=len(CONCEPT_VOCABULARY),
        dim_visual=cfg.cross_modal_dim_visual,
        dim_audio=cfg.cross_modal_dim_audio,
        seed=seed,
    )


def _train_multimodal_on_corpus(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    corpus: list[str],
    n_tokens: int,
    signatures: dict[str, dict[str, torch.Tensor]],
    dim_visual: int,
    dim_audio: int,
) -> tuple[int, int, int]:
    """Train on corpus with window-local concept-conditioned multimodal spikes.

    Only char windows containing a concept word receive the paired
    visual/audio spikes — function-word windows get ``None``.

    Returns (tokens_processed, visual_pairs_sent, audio_pairs_sent).
    """
    total = 0
    visual_count = 0
    audio_count = 0
    full_text = " ".join(corpus)
    iterations = max(1, n_tokens // max(1, len(full_text)))

    for _ in range(iterations):
        for sentence in corpus:
            patterns = list(
                encoder.iter_char_patterns(sentence, trainer.config.window_size)
            )
            for raw_window, pattern_vec in patterns:
                # Window-local: only ground windows that contain a concept
                vs, aus = _concept_spikes_for_text(
                    raw_window, signatures, dim_visual, dim_audio,
                )
                metrics = trainer.train_step(
                    pattern_vec,
                    raw_window=raw_window,
                    visual_spikes=vs,
                    audio_spikes=aus,
                )
                # Per-word grounding confidence update: after each multimodal
                # pairing, measure how well the cross-modal layer predicted the
                # actual sensory input for each concept word in this window.
                if vs is not None or aus is not None:
                    words_in_window = set(raw_window.lower().split())
                    matched = [c for c in signatures if c in words_in_window]
                    if matched:
                        text_spike = F.normalize(
                            pattern_vec.detach().unsqueeze(0), dim=1,
                        ).squeeze(0)
                        for concept in matched:
                            trainer.update_word_grounding(
                                concept, text_spike,
                                actual_visual=vs,
                                actual_audio=aus,
                            )
                total += 1
                if vs is not None:
                    visual_count += 1
                if aus is not None:
                    audio_count += 1
                if total >= n_tokens:
                    return total, visual_count, audio_count

    return total, visual_count, audio_count


def _train_on_real_digits(
    trainer: HECSNTrainer,
    encoder: RTFEncoder,
    episodes: list,
    visual_encoder: EventCameraEncoder | None,
    audio_encoder: CochleagramEncoder | None,
    n_episodes: int,
) -> tuple[int, int, int]:
    """Train on real multimodal digit episodes (§5.4).

    Unlike ``_train_multimodal_on_corpus`` which uses synthetic concept
    signatures on text-window streams, this function trains from actual
    visual events and spoken audio paired by digit class via
    ``PairedDigitDataset``.

    Design decisions (informed by §5.4 + review):
    - **Full-word encoding:** the digit name (e.g. "seven") is encoded via
      ``iter_char_patterns`` and the *last* yielded window is used as the
      word-level RTF representation (contains all characters).
    - **Per-step training:** each episode time step calls ``train_step()``
      with the encoded visual frame + audio chunk.
    - **Per-episode grounding:** ``update_word_grounding()`` is called once
      per episode using the *mean* of accepted sensory evidence, not once
      per step.  This prevents over-weighting long episodes.
    - **Accepted-only grounding:** only modalities that ``train_step()``
      reports as accepted contribute to the grounding update.

    Args:
        trainer: HECSN trainer instance.
        encoder: RTF text encoder.
        episodes: List of DigitEpisode objects.
        visual_encoder: EventCameraEncoder (or None).
        audio_encoder: CochleagramEncoder (or None).
        n_episodes: Maximum number of episodes to process.

    Returns:
        (steps_processed, visual_steps, audio_steps).
    """
    from hecsn.data.dataset_adapters import iter_episode_steps, MultimodalStep

    total_steps = 0
    visual_count = 0
    audio_count = 0
    episodes_done = 0

    for episode in episodes:
        if episodes_done >= n_episodes:
            break

        # Reset stateful encoders at episode boundary
        if visual_encoder is not None:
            visual_encoder.reset()
        if audio_encoder is not None:
            audio_encoder.reset()

        # Pre-compute full-word RTF pattern (use last window = full word)
        word = episode.text
        patterns = list(
            encoder.iter_char_patterns(word, trainer.config.window_size)
        )
        if not patterns:
            episodes_done += 1
            continue
        raw_window, pattern_vec = patterns[-1]  # last window has full word

        # Accumulators for per-episode grounding update
        accepted_visual_sum = None
        accepted_audio_sum = None
        n_visual_accepted = 0
        n_audio_accepted = 0

        n_steps = len(episode.visual_frames)
        for step_i in range(n_steps):
            # Encode this step's modalities
            vs = None
            aus = None
            if visual_encoder is not None and step_i < len(episode.visual_frames):
                vs = visual_encoder.encode(episode.visual_frames[step_i])
            if audio_encoder is not None and step_i < len(episode.audio_chunks):
                aus = audio_encoder.encode(episode.audio_chunks[step_i])

            metrics = trainer.train_step(
                pattern_vec,
                raw_window=raw_window,
                visual_spikes=vs,
                audio_spikes=aus,
            )

            total_steps += 1
            if vs is not None:
                visual_count += 1
            if aus is not None:
                audio_count += 1

            # Accumulate accepted sensory evidence for episode-level grounding
            if metrics.get("cross_modal_visual_accepted") and vs is not None:
                if accepted_visual_sum is None:
                    accepted_visual_sum = vs.detach().clone()
                else:
                    accepted_visual_sum = accepted_visual_sum + vs.detach()
                n_visual_accepted += 1

            if metrics.get("cross_modal_audio_accepted") and aus is not None:
                if accepted_audio_sum is None:
                    accepted_audio_sum = aus.detach().clone()
                else:
                    accepted_audio_sum = accepted_audio_sum + aus.detach()
                n_audio_accepted += 1

        # Per-episode grounding update (aggregated, not per-step)
        has_visual = accepted_visual_sum is not None and n_visual_accepted > 0
        has_audio = accepted_audio_sum is not None and n_audio_accepted > 0
        if has_visual or has_audio:
            text_spike = F.normalize(
                pattern_vec.detach().unsqueeze(0), dim=1,
            ).squeeze(0)
            avg_visual = (
                accepted_visual_sum / n_visual_accepted if has_visual else None
            )
            avg_audio = (
                accepted_audio_sum / n_audio_accepted if has_audio else None
            )
            trainer.update_word_grounding(
                word, text_spike,
                actual_visual=avg_visual,
                actual_audio=avg_audio,
            )

        episodes_done += 1

    return total_steps, visual_count, audio_count


def _build_concept_corpus() -> list[str]:
    """Build corpus containing every CONCEPT_VOCABULARY word at least once.

    Each sentence uses the exact word form so ``_concept_spikes_for_text``
    can match it and pair with the correct visual/audio signature.
    Sentences are grouped by concept family for natural co-occurrence.
    """
    return [
        # Fire family
        "the fire is bright and warm",
        "a flame dances in the night",
        "heat rises from the ground",
        "burn the old dry wood",
        "hot lava flows down the slope",
        "ash falls like grey snow",
        # Water family
        "water is cold and clear",
        "the deep ocean stretches far",
        "a wave crashes on the shore",
        "flow of the river is strong",
        "rain falls from dark clouds",
        "the ground is wet after the storm",
        "fish swim in the clear stream",
        "the wave hit with a crash",
        # Cold family
        "ice covers the still pond",
        "snow blankets the quiet land",
        "cold wind cuts through the air",
        "frost forms on the glass",
        "freeze the water into solid ice",
        "snow is white and bright",
        "the crystal is clear like ice",
        # Earth family
        "the rock is heavy and rough",
        "a stone sits by the path",
        "mountain peaks touch the sky",
        "earth and soil are rich",
        "sand shifts under the hot sun",
        "the wall is hard and thick",
        "the earthquake shook the ground",
        "shake the dust from your coat",
        "the dry desert stretches for miles",
        "the old stone bridge is stable and strong",
        # Air family
        "wind blows through the trees",
        "a gentle breeze cools the skin",
        "cloud drifts across the blue sky",
        "the bird took flight over the lake",
        "birds fly high in the wind",
        # Plant family
        "the tall tree grows strong",
        "a green leaf falls to the ground",
        "flower blooms in the spring",
        "the red rose smells sweet",
        "a soft petal floats down",
        "the thorn is sharp on the stem",
        # Animal family
        "the dog runs across the field",
        "a bird sits on the branch",
        "the feather is light and soft",
        # Light family
        "the sun is bright and warm",
        "light fills the open room",
        "lightning strikes the tall tree",
        "a flash of light cuts the dark",
        # Dark family
        "darkness covers the night sky",
        "a shadow moves across the wall",
        "the black cat sits in the dark",
        # Sound family
        "thunder rolls across the sky",
        "that sound is very loud",
        "the dog began to bark",
        # Quiet family
        "silence fills the empty room",
        "the night is quiet and still",
        "a calm sea reflects the moon",
        "stillness hangs in the cold air",
        # Tool family
        "the hammer strikes the nail",
        "a knife cuts through the rope",
        "the metal is cold and hard",
        "the blade is sharp and clean",
        # Texture family
        "the pillow is soft and warm",
        "cotton grows in the hot field",
        # Misc concrete
        "a star shines in the night sky",
        "the moon is full and bright",
        "the river flows to the sea",
        "smoke rises from the fire",
        "the volcano erupts with force",
        "the fish has a shiny scale",
    ]


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

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        cfg = _make_config_for_stage(1, state.config)
        trainer.config = cfg
        trainer.model.config = cfg
    else:
        cfg = _make_config_for_stage(1, config)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 1

    # Corpus must include every CONCEPT_VOCABULARY word in exact form so
    # _concept_spikes_for_text can match them and pair with visual/audio.
    corpus = _build_concept_corpus()

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

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg,
                              concept_signatures=signatures)

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

    Completion criteria (§7.3):
    1. Grounding probe accuracy (50-triple) > 0.60
    2. Self-criticism find-rate < 10% over last 5 cycles
    3. Grounding confidence growth on active dimensions
    """
    set_seed(seed)

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        cfg = _make_config_for_stage(2, state.config)
        trainer.config = cfg
        trainer.model.config = cfg
        # Stage 2 needs binding layer — ensure it exists
        if trainer.model.binding_layer is None and cfg.enable_binding_layer:
            from hecsn.core.context import BindingLayer
            trainer.model.binding_layer = BindingLayer(
                n_columns=cfg.n_columns,
                device=trainer.model.device,
                threshold=cfg.binding_threshold,
                association_lr=cfg.binding_association_lr,
                association_decay=cfg.binding_association_decay,
                gain_strength=cfg.binding_gain_strength,
                n_bindings=cfg.binding_n_bindings,
                fan_in=cfg.binding_fan_in,
                tau_binding=cfg.binding_tau,
                stp_u_inc=cfg.binding_stp_u_inc,
                stp_tau_f=cfg.binding_stp_tau_f,
                stp_tau_d=cfg.binding_stp_tau_d,
                pv_threshold=cfg.binding_pv_threshold,
                pv_gain=cfg.binding_pv_gain,
            )
    else:
        cfg = _make_config_for_stage(2, config)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 2
    # Preserve bootstrap counter if continuing from prior state;
    # reset only when cold-starting Stage 2.
    if state is None:
        trainer._stage2_bootstrap_used_visual = 0
        trainer._stage2_bootstrap_used_audio = 0
        trainer._stage2_bootstrap_used = 0

    corpus = _build_concept_corpus()

    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio
    signatures = _build_concept_signatures(
        n_concepts=len(CONCEPT_VOCABULARY),
        dim_visual=dim_visual,
        dim_audio=dim_audio,
        seed=seed,
    )

    # Snapshot confidence before training for growth rate computation
    initial_confidence = 0.0
    if trainer.model.cross_modal is not None:
        initial_confidence = _compute_grounding_confidence(trainer.model.cross_modal)

    tokens_processed, visual_pairs, audio_pairs = _train_multimodal_on_corpus(
        trainer, encoder, corpus, n_tokens, signatures, dim_visual, dim_audio,
    )

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    probe_result = evaluate_grounding_probe(vector_fn)

    # Compute grounding confidence (mean of top-100 dimensions)
    grounding_confidence = 0.0
    if trainer.model.cross_modal is not None:
        grounding_confidence = _compute_grounding_confidence(trainer.model.cross_modal)

    # Criterion 2: self-criticism find-rate < 10% over last 5 cycles
    find_rate = trainer.self_criticism_find_rate(last_n=5)
    # If no cycles ran yet, treat as passing (system hasn't had
    # enough data to self-criticize, which is fine early on)
    sc_history_len = len(trainer._self_criticism_history)
    find_rate_ok = (sc_history_len == 0) or (find_rate < 0.10)

    # Criterion 3: grounding confidence grew during Stage 2
    confidence_growth = grounding_confidence - initial_confidence
    # At 5K tokens the growth rate per 1K is confidence_growth / (n_tokens / 1000)
    tokens_k = max(1.0, tokens_processed / 1000.0)
    growth_rate_per_k = confidence_growth / tokens_k
    growth_ok = growth_rate_per_k > 0.001

    # Full criteria (§7.3): all three must pass
    passed = (
        probe_result.total_accuracy > 0.60
        and find_rate_ok
        and growth_ok
    )

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg,
                              concept_signatures=signatures)

    return StageResult(
        stage=2,
        passed=passed,
        metrics={
            "probe_accuracy": probe_result.total_accuracy,
            "concrete_accuracy": probe_result.concrete_accuracy,
            "abstract_accuracy": probe_result.abstract_accuracy,
            "concreteness_gap": probe_result.concreteness_gap,
            "grounding_confidence": grounding_confidence,
            "grounding_confidence_initial": initial_confidence,
            "confidence_growth_rate_per_k": growth_rate_per_k,
            "self_criticism_find_rate": find_rate,
            "self_criticism_cycles": sc_history_len,
            "visual_pairs_sent": visual_pairs,
            "audio_pairs_sent": audio_pairs,
        },
        diagnostics={
            "completion_criteria": {
                "criterion_1_probe": f"accuracy={probe_result.total_accuracy:.2f} > 0.60: {'PASS' if probe_result.total_accuracy > 0.60 else 'FAIL'}",
                "criterion_2_find_rate": f"find_rate={find_rate:.3f} < 0.10: {'PASS' if find_rate_ok else 'FAIL'} ({sc_history_len} cycles)",
                "criterion_3_growth": f"growth_rate={growth_rate_per_k:.4f}/1K > 0.001: {'PASS' if growth_ok else 'FAIL'}",
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

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        cfg = _make_config_for_stage(3, state.config)
        trainer.config = cfg
        trainer.model.config = cfg
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
        cfg = _make_config_for_stage(3, config)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 3

    corpus = _build_concept_corpus()
    signatures = _resolve_signatures(state, cfg, seed)
    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio

    # Initial multimodal training to build representations
    initial_tokens, _, _ = _train_multimodal_on_corpus(
        trainer, encoder, corpus, n_tokens // 2,
        signatures, dim_visual, dim_audio,
    )
    tokens_processed = initial_tokens

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

        # Train on gap-relevant sentences with multimodal spikes
        gap_sentence = corpus[cycle % len(corpus)]
        gap_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, [gap_sentence], 50,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += gap_tokens
        confirmation_cycles += 1

        # Track grounding confidence delta
        if trainer.model.cross_modal is not None:
            conf = _compute_grounding_confidence(trainer.model.cross_modal)
            grounding_deltas.append(conf)

    # Remaining tokens with multimodal training
    remaining = max(0, n_tokens - tokens_processed)
    self_criticism_stats: dict[str, Any] = {}
    if remaining > 0:
        rem_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, corpus, remaining,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += rem_tokens

    # Self-criticism runs automatically via trainer; also do explicit pass
    if trainer.model.cross_modal is not None:
        if len(trainer._recent_visual_frames) >= 3:
            self_criticism_stats = trainer.model.cross_modal.run_self_criticism(
                recent_visual_frames=trainer._recent_visual_frames,
                blacklist=trainer._self_criticism_blacklist,
            )
        if len(trainer._recent_audio_frames) >= 3:
            audio_stats = trainer.model.cross_modal.run_self_criticism_audio(
                recent_audio_frames=trainer._recent_audio_frames,
                blacklist=trainer._self_criticism_audio_blacklist,
            )
            self_criticism_stats["audio_checked"] = audio_stats.get("checked", 0)
            self_criticism_stats["audio_penalised"] = audio_stats.get("penalised", 0)

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    post_probe = evaluate_grounding_probe(vector_fn)

    # Criteria: no probe regression AND curiosity system active AND genuine grounding
    no_regression = post_probe.total_accuracy >= pre_probe.total_accuracy - 0.05
    curiosity_active = gap_queries_produced > 0
    # Absolute thresholds: untrained models score ~0.44 with negative gap
    genuine_grounding = (
        post_probe.total_accuracy >= 0.52
        and post_probe.concreteness_gap > 0.0
    )
    passed = no_regression and curiosity_active and genuine_grounding

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg,
                              concept_signatures=signatures)

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
                "genuine_grounding": "probe >= 0.52 AND concreteness_gap > 0.0",
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

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        cfg = _make_config_for_stage(4, state.config)
        trainer.config = cfg
        trainer.model.config = cfg
    else:
        cfg = _make_config_for_stage(4, config)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 4

    # Use same concept corpus + new acquisition sentences for diversity
    concept_corpus = _build_concept_corpus()
    signatures = _resolve_signatures(state, cfg, seed)
    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio

    # Acquisition corpus extends beyond core concepts
    acquisition_corpus = concept_corpus + [
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
            query = plan["retrieval_queries"][0]
            best_sentence = max(
                acquisition_corpus,
                key=lambda s: sum(1 for w in query.lower().split() if w in s.lower()),
            )
        else:
            best_sentence = acquisition_corpus[cycle % len(acquisition_corpus)]

        # Multimodal training on selected sentence
        acq_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, [best_sentence], 50,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += acq_tokens
        acquisitions_made += 1

    # Fill remaining tokens with multimodal training
    remaining = max(0, n_tokens - tokens_processed)
    if remaining > 0:
        rem_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, acquisition_corpus, remaining,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += rem_tokens

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    final_probe = evaluate_grounding_probe(vector_fn)

    no_regression = final_probe.total_accuracy >= initial_probe.total_accuracy - 0.10
    # Absolute thresholds: untrained models score ~0.44 with negative gap
    genuine_grounding = (
        final_probe.total_accuracy >= 0.52
        and final_probe.concreteness_gap > 0.0
    )

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg,
                              concept_signatures=signatures)

    return StageResult(
        stage=4,
        passed=no_regression and genuine_grounding,
        metrics={
            "initial_probe_accuracy": initial_probe.total_accuracy,
            "final_probe_accuracy": final_probe.total_accuracy,
            "accuracy_delta": final_probe.total_accuracy - initial_probe.total_accuracy,
            "concreteness_gap": final_probe.concreteness_gap,
            "acquisitions_made": acquisitions_made,
        },
        diagnostics={
            "completion_criterion": "no regression (final >= initial - 0.10) AND probe >= 0.52 AND gap > 0.0",
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

    if state is not None:
        trainer = state.trainer
        encoder = state.text_encoder
        cfg = _make_config_for_stage(5, state.config)
        trainer.config = cfg
        trainer.model.config = cfg
    else:
        cfg = _make_config_for_stage(5, config)
        model = HECSNModelLite(cfg)
        trainer = HECSNTrainer(model, cfg)
        encoder = RTFEncoder.from_config(cfg)

    trainer.developmental_stage = 5

    # Concept corpus for multimodal grounding + new autonomous sentences
    concept_corpus = _build_concept_corpus()
    signatures = _resolve_signatures(state, cfg, seed)
    dim_visual = cfg.cross_modal_dim_visual
    dim_audio = cfg.cross_modal_dim_audio

    autonomous_corpus = concept_corpus + [
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

        # Multimodal training on selected sentence
        cyc_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, [target], 50,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += cyc_tokens
        autonomous_cycles += 1

        # Periodic probe to check for catastrophic forgetting
        if cycle % 4 == 3:
            vfn = _make_vector_fn(trainer, encoder, cfg)
            mid_probe = evaluate_grounding_probe(vfn)
            cycle_accuracies.append(mid_probe.total_accuracy)

    # Fill remaining with multimodal training
    remaining = max(0, n_tokens - tokens_processed)
    if remaining > 0:
        rem_tokens, _, _ = _train_multimodal_on_corpus(
            trainer, encoder, autonomous_corpus, remaining,
            signatures, dim_visual, dim_audio,
        )
        tokens_processed += rem_tokens

    vector_fn = _make_vector_fn(trainer, encoder, cfg)
    final_probe = evaluate_grounding_probe(vector_fn)

    no_catastrophic_forgetting = final_probe.total_accuracy >= initial_probe.total_accuracy - 0.15
    sustained_learning = len(cycle_accuracies) == 0 or min(cycle_accuracies) >= initial_probe.total_accuracy - 0.20
    # Absolute thresholds: untrained models score ~0.44 with negative gap
    genuine_grounding = (
        final_probe.total_accuracy >= 0.52
        and final_probe.concreteness_gap > 0.0
    )

    out_state = ProtocolState(trainer=trainer, text_encoder=encoder, config=cfg,
                              concept_signatures=signatures)

    return StageResult(
        stage=5,
        passed=no_catastrophic_forgetting and sustained_learning and genuine_grounding,
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
                "genuine_grounding": "probe >= 0.52 AND concreteness_gap > 0.0",
            },
        },
        tokens_processed=tokens_processed,
    ), out_state


# ------------------------------------------------------------------
# Shared developmental corpus (used for both training and calibration)
# ------------------------------------------------------------------

DEVELOPMENTAL_CORPUS = (
    # Stage 1/2 concept-conditioned sentences (concrete, sensory)
    "the fire burns bright in the dark. "
    "water flows down the rocky stream. "
    "the tall tree grows in the forest. "
    "a small bird flies above the mountain. "
    "warm sunlight covers the sandy beach. "
    "fire burns bright. water flows down. tree grows tall. "
    "rock is heavy. bird flies high. sun shines warm. "
    # Stage 3 elaborated concrete
    "the fire burns in the dark night. "
    "water flows through the rocky river. "
    "a tall tree stands in the green forest. "
    "the bird flies above the mountain top. "
    "warm sunlight covers the sandy beach. "
    # Stage 4 acquisition domain (scientific)
    "quantum mechanics describes particle behavior at small scales. "
    "photosynthesis converts sunlight into chemical energy in plants. "
    "plate tectonics explains how continents drift over geological time. "
    "evolution shapes organisms through natural selection pressure. "
    "gravity pulls objects toward each other with measurable force. "
    "neural networks process information through connected layers. "
    "climate systems regulate temperature across the entire planet. "
    "cellular division creates new organisms from existing cells. "
    # Stage 5 autonomous domain
    "algorithms optimize computational efficiency in modern systems. "
    "biodiversity preserves ecosystem stability through redundancy. "
    "electromagnetic waves carry energy across vast distances. "
    "metabolism converts nutrients into cellular energy continuously. "
    "ocean currents distribute heat around the global climate system. "
    "symbiotic relationships benefit multiple organisms simultaneously. "
    "thermodynamics governs energy transfer in physical processes. "
    "volcanic eruptions reshape landscapes through geological forces. "
)


def run_baseline_calibration(
    corpus: str | None = None,
    input_dim: int = 128,
    n_prototypes: int = 64,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run baseline calibration (§8.1) on the developmental corpus.

    Trains SOM, 4-gram, and fastText baselines on the same text corpus
    used by the developmental protocol, evaluates them on the 50-triple
    grounding probe, and returns calibrated thresholds.
    """
    from hecsn.evaluation.baselines import run_all_baselines

    if corpus is None:
        corpus = DEVELOPMENTAL_CORPUS

    results = run_all_baselines(
        corpus=corpus,
        input_dim=input_dim,
        n_prototypes=n_prototypes,
        seed=seed,
    )

    summary = results.summary()

    # Compute calibrated thresholds per §4.10 / §10.4
    ft_score = summary["fasttext"]["grounding_probe_accuracy"]
    som_score = summary["online_som"]["grounding_probe_accuracy"]

    calibrated = {
        "baselines": summary,
        "calibrated_thresholds": {
            "stage2_criterion": max(0.60, ft_score + 0.03),
            "publication_threshold": max(0.65, ft_score + 0.05),
            "fasttext_score": ft_score,
            "som_score": som_score,
        },
        "notes": (
            f"fastText scored {ft_score:.3f} on 50-triple probe. "
            f"Online SOM scored {som_score:.3f}. "
            f"Stage 2 criterion set to max(0.60, fastText + 0.03) = "
            f"{max(0.60, ft_score + 0.03):.3f}. "
            f"Publication threshold set to max(0.65, fastText + 0.05) = "
            f"{max(0.65, ft_score + 0.05):.3f}."
        ),
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "baseline_calibration.json").write_text(
            json.dumps(calibrated, indent=2)
        )

    return calibrated


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
