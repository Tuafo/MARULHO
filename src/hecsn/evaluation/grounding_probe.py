"""50-triple grounding probe for HECSN v4 evaluation protocol.

Implements §8.7 of the v4 paper: 25 concrete + 25 abstract structural triples
scored by cosine similarity, with a concreteness gap metric that measures
whether multimodal grounding genuinely improves concrete-concept representations
beyond what pure text co-occurrence can achieve.

Each triple is (anchor, positive, negative) where a correct system should place
cos(anchor, positive) > cos(anchor, negative).

The concreteness gap = mean(concrete_accuracy) - mean(abstract_accuracy).
Target: >0.10 — evidence of perceptual grounding that text-only systems cannot
replicate (§8.7 of v4 paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 25 concrete triples — physical objects, actions, sensory properties
# A human should consistently agree: anchor is closer to positive than negative.
# Tagged with modality: "visual" (primarily visual) or "audio" (primarily auditory).
# ---------------------------------------------------------------------------
CONCRETE_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("ocean", "water", "desert"),
    ("fire", "heat", "cold"),
    ("dog", "bark", "silence"),
    ("hammer", "metal", "feather"),
    ("ice", "cold", "heat"),
    ("sun", "light", "darkness"),
    ("rain", "wet", "dry"),
    ("knife", "sharp", "soft"),
    ("thunder", "loud", "quiet"),
    ("snow", "white", "black"),
    ("mountain", "rock", "sand"),
    ("river", "flow", "still"),
    ("bird", "flight", "swim"),
    ("tree", "leaf", "scale"),
    ("volcano", "lava", "frost"),
    ("stone", "hard", "cotton"),
    ("wind", "breeze", "calm"),
    ("lightning", "flash", "shadow"),
    ("rose", "petal", "thorn"),
    ("fish", "swim", "fly"),
    ("earthquake", "shake", "stable"),
    ("smoke", "ash", "crystal"),
    ("wave", "crash", "stillness"),
    ("cloud", "sky", "earth"),
    ("flame", "burn", "freeze"),
)

# Modality tags for concrete triples (§5.2): indices of audio-primary triples.
# Audio-primary: dog/bark, thunder/loud, wind/breeze — perceptual anchored in sound.
# All others are visual-primary.
CONCRETE_AUDIO_INDICES: frozenset[int] = frozenset({2, 8, 16})

# ---------------------------------------------------------------------------
# 10 held-out concrete triples — physical objects NOT in CONCEPT_VOCABULARY.
# These test TRANSFER: can grounding learned on training concepts generalize
# to unseen concrete words?  If held-out concrete accuracy > abstract accuracy,
# that is evidence of genuine grounding structure, not lexical memorization.
# IMPORTANT: No word in these triples may appear in CONCEPT_VOCABULARY.
# ---------------------------------------------------------------------------
HELD_OUT_CONCRETE_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("tiger", "stripe", "plain"),
    ("drum", "beat", "melody"),
    ("glass", "transparent", "opaque"),
    ("bell", "ring", "mute"),
    ("honey", "sweet", "bitter"),
    ("rope", "knot", "straight"),
    ("brick", "heavy", "hollow"),
    ("candle", "wax", "wire"),
    ("wheel", "spin", "lock"),
    ("apple", "fruit", "root"),
)

# ---------------------------------------------------------------------------
# 25 abstract triples — social/relational concepts (harder — perceptual
# grounding is indirect)
# ---------------------------------------------------------------------------
ABSTRACT_TRIPLES: tuple[tuple[str, str, str], ...] = (
    ("justice", "equality", "tyranny"),
    ("theory", "hypothesis", "evidence"),
    ("freedom", "liberty", "captivity"),
    ("courage", "bravery", "cowardice"),
    ("wisdom", "knowledge", "ignorance"),
    ("peace", "harmony", "conflict"),
    ("trust", "loyalty", "betrayal"),
    ("hope", "optimism", "despair"),
    ("grief", "sorrow", "joy"),
    ("progress", "advance", "decline"),
    ("truth", "honesty", "deception"),
    ("power", "authority", "weakness"),
    ("mercy", "compassion", "cruelty"),
    ("pride", "dignity", "shame"),
    ("patience", "endurance", "impatience"),
    ("ambition", "drive", "apathy"),
    ("honor", "respect", "disgrace"),
    ("curiosity", "wonder", "indifference"),
    ("gratitude", "thankful", "resentment"),
    ("innovation", "creativity", "stagnation"),
    # Purely linguistic/function-word triples — no visual correlate at all
    ("therefore", "hence", "however"),
    ("whereas", "unless", "although"),
    ("perhaps", "possibly", "certainly"),
    ("moreover", "furthermore", "nevertheless"),
    ("indeed", "truly", "hardly"),
)

GROUNDING_PROBE_TRIPLES_50: tuple[tuple[str, str, str], ...] = (
    *CONCRETE_TRIPLES,
    *ABSTRACT_TRIPLES,
)

# Extended 60-triple probe including held-out concrete concepts
GROUNDING_PROBE_TRIPLES_60: tuple[tuple[str, str, str], ...] = (
    *CONCRETE_TRIPLES,
    *HELD_OUT_CONCRETE_TRIPLES,
    *ABSTRACT_TRIPLES,
)


@dataclass
class GroundingProbeResult:
    """Full grounding probe output including concreteness gap and modality splits."""

    total_accuracy: float
    concrete_accuracy: float
    abstract_accuracy: float
    concreteness_gap: float
    total_count: int
    concrete_count: int
    abstract_count: int
    mean_margin: float
    concrete_mean_margin: float
    abstract_mean_margin: float
    # Visual-text / audio-text split metrics (§5.2)
    visual_text_accuracy: float = 0.0
    audio_text_accuracy: float = 0.0
    visual_text_count: int = 0
    audio_text_count: int = 0
    # Held-out concrete metrics (words NOT in training vocabulary)
    held_out_concrete_accuracy: float = 0.0
    held_out_concrete_count: int = 0
    held_out_concrete_mean_margin: float = 0.0
    held_out_concreteness_gap: float = 0.0  # held_out_concrete - abstract
    per_triple: list[dict[str, Any]] = field(default_factory=list)

    @property
    def concreteness_gap_pass(self) -> bool:
        return self.concreteness_gap > 0.10

    @property
    def probe_pass(self) -> bool:
        """Paper threshold: total accuracy > 0.65."""
        return self.total_accuracy > 0.65

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_accuracy": self.total_accuracy,
            "concrete_accuracy": self.concrete_accuracy,
            "abstract_accuracy": self.abstract_accuracy,
            "concreteness_gap": self.concreteness_gap,
            "concreteness_gap_pass": self.concreteness_gap_pass,
            "probe_pass": self.probe_pass,
            "total_count": self.total_count,
            "concrete_count": self.concrete_count,
            "abstract_count": self.abstract_count,
            "mean_margin": self.mean_margin,
            "concrete_mean_margin": self.concrete_mean_margin,
            "abstract_mean_margin": self.abstract_mean_margin,
            "visual_text_accuracy": self.visual_text_accuracy,
            "audio_text_accuracy": self.audio_text_accuracy,
            "visual_text_count": self.visual_text_count,
            "audio_text_count": self.audio_text_count,
            "held_out_concrete_accuracy": self.held_out_concrete_accuracy,
            "held_out_concrete_count": self.held_out_concrete_count,
            "held_out_concrete_mean_margin": self.held_out_concrete_mean_margin,
            "held_out_concreteness_gap": self.held_out_concreteness_gap,
            "per_triple": self.per_triple,
        }


def _cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Safe cosine similarity between two 1-D tensors."""
    a_norm = a.norm()
    b_norm = b.norm()
    if a_norm < 1e-12 or b_norm < 1e-12:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def evaluate_grounding_probe(
    vector_fn: Any,
    *,
    triples: Sequence[tuple[str, str, str]] | None = None,
    concrete_count: int = 25,
    held_out_count: int = 0,
) -> GroundingProbeResult:
    """Evaluate the grounding probe.

    Args:
        vector_fn: Callable(str) -> torch.Tensor that returns the
            routing-key representation for a given text token/phrase.
        triples: If None, uses the default 50 triples
            (first ``concrete_count`` are concrete, rest abstract).
        concrete_count: How many of the first triples are concrete (in-vocab).
        held_out_count: How many triples immediately after concrete_count are
            held-out concrete (not in training vocabulary).

    Returns:
        GroundingProbeResult with per-triple details and the concreteness gap.
    """
    if triples is None:
        triples = GROUNDING_PROBE_TRIPLES_50
        concrete_count = len(CONCRETE_TRIPLES)
        held_out_count = 0

    concrete_correct = 0
    abstract_correct = 0
    visual_correct = 0
    audio_correct = 0
    visual_total = 0
    audio_total = 0
    held_out_correct = 0
    concrete_margins: list[float] = []
    abstract_margins: list[float] = []
    held_out_margins: list[float] = []
    all_margins: list[float] = []
    per_triple: list[dict[str, Any]] = []
    total_correct = 0

    for i, (anchor_text, pos_text, neg_text) in enumerate(triples):
        is_concrete = i < concrete_count
        is_held_out = concrete_count <= i < concrete_count + held_out_count
        is_abstract = i >= concrete_count + held_out_count
        is_audio = is_concrete and i in CONCRETE_AUDIO_INDICES
        anchor_vec = vector_fn(anchor_text)
        pos_vec = vector_fn(pos_text)
        neg_vec = vector_fn(neg_text)

        pos_sim = _cosine_sim(anchor_vec, pos_vec)
        neg_sim = _cosine_sim(anchor_vec, neg_vec)
        margin = pos_sim - neg_sim
        correct = margin > 0.0

        if is_held_out:
            category = "held_out_concrete"
            modality = "visual"
        elif is_concrete:
            category = "concrete"
            modality = "audio" if is_audio else "visual"
        else:
            category = "abstract"
            modality = "abstract"

        entry = {
            "index": i,
            "category": category,
            "modality": modality,
            "anchor": anchor_text,
            "positive": pos_text,
            "negative": neg_text,
            "positive_similarity": pos_sim,
            "negative_similarity": neg_sim,
            "margin": margin,
            "correct": correct,
        }
        per_triple.append(entry)
        all_margins.append(margin)

        if correct:
            total_correct += 1
        if is_concrete:
            concrete_margins.append(margin)
            if correct:
                concrete_correct += 1
            if is_audio:
                audio_total += 1
                if correct:
                    audio_correct += 1
            else:
                visual_total += 1
                if correct:
                    visual_correct += 1
        elif is_held_out:
            held_out_margins.append(margin)
            if correct:
                held_out_correct += 1
        else:
            abstract_margins.append(margin)
            if correct:
                abstract_correct += 1

    n_concrete = len(concrete_margins)
    n_abstract = len(abstract_margins)
    n_held_out = len(held_out_margins)
    n_total = n_concrete + n_abstract + n_held_out

    concrete_acc = concrete_correct / max(n_concrete, 1)
    abstract_acc = abstract_correct / max(n_abstract, 1)
    held_out_acc = held_out_correct / max(n_held_out, 1)
    total_acc = total_correct / max(n_total, 1)

    return GroundingProbeResult(
        total_accuracy=total_acc,
        concrete_accuracy=concrete_acc,
        abstract_accuracy=abstract_acc,
        concreteness_gap=concrete_acc - abstract_acc,
        total_count=n_total,
        concrete_count=n_concrete,
        abstract_count=n_abstract,
        mean_margin=sum(all_margins) / max(len(all_margins), 1),
        concrete_mean_margin=sum(concrete_margins) / max(len(concrete_margins), 1),
        abstract_mean_margin=sum(abstract_margins) / max(len(abstract_margins), 1),
        visual_text_accuracy=visual_correct / max(visual_total, 1),
        audio_text_accuracy=audio_correct / max(audio_total, 1),
        visual_text_count=visual_total,
        audio_text_count=audio_total,
        held_out_concrete_accuracy=held_out_acc,
        held_out_concrete_count=n_held_out,
        held_out_concrete_mean_margin=(
            sum(held_out_margins) / max(len(held_out_margins), 1)
        ),
        held_out_concreteness_gap=held_out_acc - abstract_acc,
        per_triple=per_triple,
    )


def evaluate_grounding_probe_extended(
    vector_fn: Any,
) -> GroundingProbeResult:
    """Evaluate the extended 60-triple probe with held-out concrete words.

    Uses 25 in-vocab concrete + 10 held-out concrete + 25 abstract triples.
    The held-out concrete words are NOT in CONCEPT_VOCABULARY, so their
    accuracy tests whether grounding transfers beyond trained words.
    """
    return evaluate_grounding_probe(
        vector_fn,
        triples=GROUNDING_PROBE_TRIPLES_60,
        concrete_count=len(CONCRETE_TRIPLES),
        held_out_count=len(HELD_OUT_CONCRETE_TRIPLES),
    )
