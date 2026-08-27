"""Inspectable hybrid candidate retrieval for recent papers."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from openai import OpenAI, OpenAIError
from rank_bm25 import BM25Okapi

from arxiv_rec.models import (
    Paper,
    RecalledPaper,
    RecallReport,
    RecallSignals,
    ResearchProfile,
)
from arxiv_rec.openai_errors import openai_error_message

TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")
NEGATIVE_QUALIFIER_PATTERN: Final = re.compile(
    r"\b(?:without|unrelated\s+to|not\s+involving|outside)\b", re.IGNORECASE
)
STOP_WORDS: Final = frozenset(
    {
        "a",
        "an",
        "and",
        "for",
        "in",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "without",
        "with",
    }
)


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be produced or validated."""


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    input_tokens: int | None = None


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = match.group(0).casefold()
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


def _concept_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOP_WORDS}


def _concept_coverage(concept: str, document_tokens: set[str]) -> float:
    terms = _concept_tokens(concept)
    return len(terms & document_tokens) / len(terms) if terms else 0.0


def _normalized_phrase_score(
    profile: ResearchProfile, paper: Paper
) -> tuple[float, tuple[str, ...]]:
    title_tokens = set(tokenize(paper.title))
    document_tokens = title_tokens | set(tokenize(paper.abstract))
    weighted_concepts: list[tuple[str, float]] = []
    weighted_concepts.extend((topic, 1.0) for topic in profile.core_topics)
    weighted_concepts.extend(
        (concept, 1.1) for group in profile.required_intersections for concept in group
    )
    weighted_concepts.extend((facet, 0.9) for facet in profile.primary_facets)
    weighted_concepts.extend((method, 0.7) for method in profile.methods)
    weighted_concepts.extend((platform, 0.7) for platform in profile.platforms)
    weighted_concepts.extend((topic, 0.45) for topic in profile.adjacent_topics)
    weighted_concepts.extend((topic, 0.15) for topic in profile.weakly_related_topics)

    total_weight = sum(weight for _, weight in weighted_concepts)
    if total_weight <= 0:
        return 0.0, ()
    score = 0.0
    matched: list[str] = []
    for concept, weight in weighted_concepts:
        coverage = _concept_coverage(concept, document_tokens)
        title_coverage = _concept_coverage(concept, title_tokens)
        effective = min(1.0, coverage + 0.25 * title_coverage)
        score += weight * effective
        if coverage >= 0.5:
            matched.append(concept)
    return min(1.0, score / total_weight), tuple(dict.fromkeys(matched))


def _intersection_coverage(profile: ResearchProfile, paper: Paper) -> float:
    if not profile.required_intersections:
        return 0.0
    document_tokens = set(tokenize(f"{paper.title} {paper.abstract}"))
    group_scores = [
        min(_concept_coverage(concept, document_tokens) for concept in group)
        for group in profile.required_intersections
    ]
    # Each group is an AND; multiple groups are alternative sufficient intersections (OR).
    return max(group_scores, default=0.0)


def _negative_penalty(profile: ResearchProfile, paper: Paper) -> tuple[float, tuple[str, ...]]:
    document_tokens = set(tokenize(f"{paper.title} {paper.abstract}"))
    coverages: list[tuple[str, float]] = []
    for topic in profile.negative_topics:
        # In "X without Y", X is the negative anchor and Y describes the missing
        # positive context. Treating Y as negative evidence would invert the user's intent.
        anchor = NEGATIVE_QUALIFIER_PATTERN.split(topic, maxsplit=1)[0].strip()
        coverage = _concept_coverage(anchor or topic, document_tokens)
        coverages.append((topic, coverage))
    matched = tuple(topic for topic, coverage in coverages if coverage >= 0.75)
    penalty = max((coverage for _, coverage in coverages if coverage >= 0.75), default=0.0)
    return penalty, matched


def profile_embedding_text(profile: ResearchProfile) -> str:
    sections = [
        "Core research topics: " + "; ".join(profile.core_topics),
        "Required concept intersections: "
        + "; ".join(" AND ".join(group) for group in profile.required_intersections),
        "Primary facets: " + "; ".join(profile.primary_facets),
        "Methods: " + "; ".join(profile.methods),
        "Platforms: " + "; ".join(profile.platforms),
        "Adjacent topics: " + "; ".join(profile.adjacent_topics),
    ]
    return "\n".join(section for section in sections if not section.endswith(": "))


def paper_embedding_text(paper: Paper) -> str:
    return (
        f"Title: {paper.title}\n"
        f"Abstract: {paper.abstract}\n"
        f"Categories: {', '.join(paper.categories)}"
    )


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or not first:
        raise EmbeddingError("embedding vectors must have the same non-zero dimension")
    first_norm = math.sqrt(math.fsum(value * value for value in first))
    second_norm = math.sqrt(math.fsum(value * value for value in second))
    denominator = first_norm * second_norm
    if denominator == 0:
        raise EmbeddingError("embedding vectors cannot be zero vectors")
    dot_product = math.fsum(left * right for left, right in zip(first, second, strict=True))
    return max(-1.0, min(1.0, dot_product / denominator))


def _normalize_nonnegative(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(0.0, value) for value in values]
    maximum = max(clipped)
    if maximum <= 0:
        return [0.0] * len(clipped)
    return [value / maximum for value in clipped]


class OpenAITextEmbedder:
    """Batch text embeddings using the official OpenAI client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if client is None and not api_key:
            raise ValueError("an OpenAI API key is required for live embeddings")
        self.model = model
        self._client = client or OpenAI(api_key=api_key)

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts or any(not text.strip() for text in texts):
            raise ValueError("embedding input must contain non-empty text")
        try:
            response = self._client.embeddings.create(model=self.model, input=list(texts))
        except OpenAIError as exc:
            raise EmbeddingError(openai_error_message("embedding request", exc)) from exc
        ordered = sorted(response.data, key=lambda item: item.index)
        vectors = tuple(tuple(float(value) for value in item.embedding) for item in ordered)
        if len(vectors) != len(texts):
            raise EmbeddingError("OpenAI returned an unexpected number of embeddings")
        usage = getattr(response, "usage", None)
        tokens = getattr(usage, "total_tokens", None)
        return EmbeddingBatch(
            vectors=vectors,
            provider="openai",
            model=self.model,
            input_tokens=tokens if isinstance(tokens, int) and tokens >= 0 else None,
        )


class StaticTextEmbedder:
    """Precomputed vectors for deterministic fixture and regression tests."""

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        fixture_name: str = "static-fixture",
    ) -> None:
        self.vectors = tuple(tuple(float(value) for value in vector) for vector in vectors)
        self.fixture_name = fixture_name

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if len(texts) != len(self.vectors):
            raise EmbeddingError(
                f"fixture has {len(self.vectors)} vectors for {len(texts)} input texts"
            )
        return EmbeddingBatch(
            vectors=self.vectors,
            provider="fixture",
            model=self.fixture_name,
        )


class HybridRetriever:
    """Combine lexical, category, intersection, and optional embedding signals."""

    def __init__(self, embedder: Any | None = None) -> None:
        self.embedder = embedder

    def retrieve(
        self,
        profile: ResearchProfile,
        papers: Sequence[Paper],
        *,
        candidate_threshold: float = 0.15,
        max_candidates: int = 100,
    ) -> RecallReport:
        if not 0 <= candidate_threshold <= 1:
            raise ValueError("candidate_threshold must be between 0 and 1")
        if max_candidates < 1:
            raise ValueError("max_candidates must be positive")
        if not papers:
            return RecallReport(
                items=(),
                candidate_threshold=candidate_threshold,
                max_candidates=max_candidates,
            )

        corpus_tokens = [tokenize(f"{paper.title} {paper.abstract}") for paper in papers]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = tokenize(profile_embedding_text(profile))
        bm25_scores = _normalize_nonnegative(
            [float(value) for value in bm25.get_scores(query_tokens)]
        )

        embedding_scores: list[float | None] = [None] * len(papers)
        embedding_provider: str | None = None
        embedding_model: str | None = None
        embedding_tokens: int | None = None
        if self.embedder is not None:
            batch = self.embedder.embed(
                [
                    profile_embedding_text(profile),
                    *(paper_embedding_text(paper) for paper in papers),
                ]
            )
            if len(batch.vectors) != len(papers) + 1:
                raise EmbeddingError("embedder returned an unexpected number of vectors")
            embedding_scores = [
                cosine_similarity(batch.vectors[0], vector) for vector in batch.vectors[1:]
            ]
            embedding_provider = batch.provider
            embedding_model = batch.model
            embedding_tokens = batch.input_tokens

        items: list[RecalledPaper] = []
        profile_categories = set(profile.arxiv_categories)
        for index, paper in enumerate(papers):
            keyword_score, matched_terms = _normalized_phrase_score(profile, paper)
            intersection = _intersection_coverage(profile, paper)
            negative, excluded_terms = _negative_penalty(profile, paper)
            category_match = bool(profile_categories & set(paper.categories))
            embedding = embedding_scores[index]
            embedding_positive = max(0.0, embedding) if embedding is not None else 0.0
            embedding_weight = 0.50 if embedding is not None else 0.0
            base_weight = 0.50 + embedding_weight
            positive_score = (
                0.20 * keyword_score
                + 0.15 * bm25_scores[index]
                + 0.10 * intersection
                + 0.05 * float(category_match)
                + embedding_weight * embedding_positive
            ) / base_weight
            hybrid_score = max(0.0, min(1.0, positive_score - 0.25 * negative))
            sources = ["category"] if category_match else []
            if keyword_score > 0:
                sources.append("keyword")
            if bm25_scores[index] > 0:
                sources.append("bm25")
            if intersection > 0:
                sources.append("intersection")
            if embedding is not None:
                sources.append("embedding")
            signals = RecallSignals(
                keyword_score=keyword_score,
                bm25_score=bm25_scores[index],
                embedding_score=embedding,
                category_match=category_match,
                intersection_coverage=intersection,
                negative_penalty=negative,
                hybrid_score=hybrid_score,
                matched_terms=matched_terms,
                excluded_terms=excluded_terms,
                sources=tuple(sources) or ("none",),
            )
            items.append(RecalledPaper(paper=paper, signals=signals, included=False))

        ordered = sorted(items, key=lambda item: item.signals.hybrid_score, reverse=True)
        eligible = [
            item.paper.arxiv_id
            for item in ordered
            if item.signals.hybrid_score >= candidate_threshold
        ][:max_candidates]
        included_ids = set(eligible)
        finalized = tuple(
            item.model_copy(update={"included": item.paper.arxiv_id in included_ids})
            for item in ordered
        )
        return RecallReport(
            items=finalized,
            candidate_threshold=candidate_threshold,
            max_candidates=max_candidates,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            embedding_tokens=embedding_tokens,
        )
