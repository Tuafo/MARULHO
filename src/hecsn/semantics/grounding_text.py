from __future__ import annotations

import re
from typing import Any, Sequence


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "kind",
    "me",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "place",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "thing",
    "this",
    "to",
    "type",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
    "yours",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_RE.findall(normalize_text(text).lower()):
        if match in _STOPWORDS:
            continue
        if len(match) == 1 and not match.isdigit():
            continue
        tokens.append(match)
    return tokens


def salient_query_terms(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def token_forms(token: str) -> tuple[str, ...]:
    normalized = normalize_text(token).lower()
    if not normalized:
        return ()

    variants: list[str] = []

    def add(value: str) -> None:
        item = normalize_text(value).lower()
        if not item or item in variants:
            return
        variants.append(item)

    add(normalized)
    if len(normalized) > 4 and normalized.endswith("ies"):
        add(normalized[:-3] + "y")
    if len(normalized) > 4 and normalized.endswith("oes"):
        add(normalized[:-2])
    if len(normalized) > 4 and normalized.endswith(("ches", "shes", "sses", "xes", "zes")):
        add(normalized[:-2])
    elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        add(normalized[:-1])
    if len(normalized) > 5 and normalized.endswith("ing"):
        add(normalized[:-3])
    if len(normalized) > 4 and normalized.endswith("ed"):
        add(normalized[:-2])
    return tuple(variants)


def _expanded_terms(tokens: Sequence[str]) -> set[str]:
    expanded: set[str] = set()
    for token in tokens:
        expanded.update(token_forms(token))
    return expanded


def match_terms(query_terms: Sequence[str], text: str) -> list[str]:
    evidence_terms = tokenize(text)
    if not query_terms or not evidence_terms:
        return []

    evidence_forms = _expanded_terms(evidence_terms)
    matches: list[str] = []
    for term in query_terms:
        if evidence_forms & set(token_forms(term)):
            if term not in matches:
                matches.append(term)
    return matches


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    segments = [segment.strip() for segment in _SENTENCE_SPLIT_RE.split(normalized) if segment.strip()]
    return segments or [normalized]


def query_focused_clauses(text: str, query_terms: Sequence[str]) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    if not query_terms:
        return [normalized]

    clauses = split_sentences(normalized)
    if len(clauses) <= 1:
        return clauses

    focused = [clause for clause in clauses if match_terms(query_terms, clause)]
    return focused or clauses


def query_focused_text(text: str, query_terms: Sequence[str]) -> str:
    return " ".join(query_focused_clauses(text, query_terms)).strip()
