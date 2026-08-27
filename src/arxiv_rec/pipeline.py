"""Shared orchestration for fixture and live recommendation runs."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import Field

from arxiv_rec.arxiv_client import ArxivClient
from arxiv_rec.models import (
    ArxivFetchReport,
    Paper,
    RankingReport,
    RecallReport,
    ResearchProfile,
    RunMode,
    StrictModel,
)
from arxiv_rec.ranking import FixtureRelevanceRanker, assessments_from_fixture
from arxiv_rec.retrieval import HybridRetriever, StaticTextEmbedder


class RecommendationRun(StrictModel):
    """Complete inspectable output from one recommendation run."""

    mode: RunMode
    profile: ResearchProfile
    fetched_count: int = Field(ge=0)
    fetch_report: ArxivFetchReport | None = None
    recall: RecallReport
    ranking: RankingReport
    started_at: datetime
    completed_at: datetime


def run_recommendation(
    *,
    mode: RunMode,
    profile: ResearchProfile,
    papers: Sequence[Paper],
    retriever: Any,
    ranker: Any,
    candidate_threshold: float = 0.15,
    max_candidates: int = 100,
    relevance_threshold: int = 75,
    review_margin: int = 5,
    fetch_report: ArxivFetchReport | None = None,
    now: Any = lambda: datetime.now(UTC),
) -> RecommendationRun:
    """Retrieve and rank papers while preserving every intermediate signal."""
    started_at = now()
    recall = retriever.retrieve(
        profile,
        papers,
        candidate_threshold=candidate_threshold,
        max_candidates=max_candidates,
    )
    ranking = ranker.assess(
        profile,
        recall.candidates,
        relevance_threshold=relevance_threshold,
        review_margin=review_margin,
    )
    return RecommendationRun(
        mode=mode,
        profile=profile,
        fetched_count=len(papers),
        fetch_report=fetch_report,
        recall=recall,
        ranking=ranking,
        started_at=started_at,
        completed_at=now(),
    )


def load_fixture(path: Path) -> tuple[ResearchProfile, tuple[Paper, ...], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = ResearchProfile.model_validate(payload["profile"])
    papers = tuple(Paper.model_validate(item) for item in payload["papers"])
    return profile, papers, payload


def run_fixture(
    *,
    path: Path,
    profile: ResearchProfile | None = None,
    candidate_threshold: float = 0.0,
    max_candidates: int = 100,
    relevance_threshold: int = 75,
    review_margin: int = 5,
) -> RecommendationRun:
    fixture_profile, papers, payload = load_fixture(path)
    active_profile = profile or fixture_profile
    retriever = HybridRetriever(
        StaticTextEmbedder(payload["embedding_vectors"], fixture_name=path.stem)
    )
    ranker = FixtureRelevanceRanker(
        assessments_from_fixture(payload["llm_assessments"]), fixture_name=path.stem
    )
    return run_recommendation(
        mode=RunMode.FIXTURE,
        profile=active_profile,
        papers=papers,
        retriever=retriever,
        ranker=ranker,
        candidate_threshold=candidate_threshold,
        max_candidates=max_candidates,
        relevance_threshold=relevance_threshold,
        review_margin=review_margin,
    )


def fetch_live_papers(
    *,
    profile: ResearchProfile,
    client: ArxivClient,
    lookback_days: int,
    max_results: int,
    now: datetime | None = None,
) -> list[Paper]:
    if not 1 <= lookback_days <= 30:
        raise ValueError("lookback_days must be between 1 and 30")
    reference = now or datetime.now(UTC)
    return client.fetch(
        categories=profile.arxiv_categories,
        published_after=reference - timedelta(days=lookback_days),
        max_results=max_results,
    )


def fetch_live_report(
    *,
    profile: ResearchProfile,
    client: ArxivClient,
    lookback_days: int,
    max_results: int,
    now: datetime | None = None,
) -> ArxivFetchReport:
    if not 1 <= lookback_days <= 30:
        raise ValueError("lookback_days must be between 1 and 30")
    reference = now or datetime.now(UTC)
    return client.fetch_report(
        categories=profile.arxiv_categories,
        published_after=reference - timedelta(days=lookback_days),
        max_results=max_results,
    )
