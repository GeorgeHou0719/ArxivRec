"""Provider-neutral interfaces for the recommendation pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol

from arxiv_rec.models import (
    Paper,
    ProfileAnalysisResult,
    RankingReport,
    RecalledPaper,
    RecallReport,
    ResearchProfile,
)


class PaperSource(Protocol):
    def fetch(
        self,
        *,
        categories: Sequence[str],
        published_after: datetime,
        max_results: int,
    ) -> list[Paper]: ...


class ProfileAnalyzer(Protocol):
    def analyze(self, source_text: str) -> ProfileAnalysisResult: ...


class TextEmbedder(Protocol):
    def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class CandidateRetriever(Protocol):
    def retrieve(
        self,
        profile: ResearchProfile,
        papers: Sequence[Paper],
        *,
        candidate_threshold: float,
        max_candidates: int,
    ) -> RecallReport: ...


class RelevanceRanker(Protocol):
    def assess(
        self,
        profile: ResearchProfile,
        candidates: Sequence[RecalledPaper],
        *,
        relevance_threshold: int,
        review_margin: int,
    ) -> RankingReport: ...


class EmbeddingResult(Protocol):
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    input_tokens: int | None
