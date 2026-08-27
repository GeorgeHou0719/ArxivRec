"""Local single-user relevance feedback storage."""

from __future__ import annotations

import hashlib
import math
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from arxiv_rec.models import (
    Paper,
    Recommendation,
    ResearchProfile,
    Score,
    StrictModel,
    label_for_score,
)


class FeedbackRecord(StrictModel):
    profile_hash: str = Field(min_length=64, max_length=64)
    arxiv_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    predicted_score: Score
    user_score: Score
    notes: str = ""
    updated_at: datetime


class ScoreDiagnostics(StrictModel):
    prediction_count: int = Field(ge=0)
    unique_score_count: int = Field(ge=0)
    score_counts: dict[int, int]
    score_regions: dict[str, int]
    boundary_count: int = Field(ge=0)
    boundary_rate: float = Field(ge=0.0, le=1.0)
    feedback_count: int = Field(ge=0)
    mean_absolute_error: float | None = Field(default=None, ge=0.0, le=100.0)
    mean_bias: float | None = Field(default=None, ge=-100.0, le=100.0)
    band_agreement: float | None = Field(default=None, ge=0.0, le=1.0)


def analyze_score_diagnostics(
    recommendations: Sequence[Recommendation],
    feedback: Sequence[FeedbackRecord],
) -> ScoreDiagnostics:
    scores = [item.assessment.relevance_score for item in recommendations]
    score_counts = {score: scores.count(score) for score in sorted(set(scores))}
    regions = {
        "0-19": sum(0 <= score <= 19 for score in scores),
        "20-34": sum(20 <= score <= 34 for score in scores),
        "35-44": sum(35 <= score <= 44 for score in scores),
        "45-54": sum(45 <= score <= 54 for score in scores),
        "55-64": sum(55 <= score <= 64 for score in scores),
        "65-74": sum(65 <= score <= 74 for score in scores),
        "75-89": sum(75 <= score <= 89 for score in scores),
        "90-100": sum(90 <= score <= 100 for score in scores),
    }
    boundary_count = sum(score in {35, 55, 75, 90} for score in scores)
    feedback_by_id = {record.arxiv_id: record for record in feedback}
    paired = [
        (item.assessment.relevance_score, feedback_by_id[item.paper.arxiv_id].user_score)
        for item in recommendations
        if item.paper.arxiv_id in feedback_by_id
    ]
    feedback_count = len(paired)
    mae = (
        math.fsum(abs(predicted - user) for predicted, user in paired) / feedback_count
        if paired
        else None
    )
    bias = (
        math.fsum(predicted - user for predicted, user in paired) / feedback_count
        if paired
        else None
    )
    band_agreement = (
        sum(label_for_score(predicted) is label_for_score(user) for predicted, user in paired)
        / feedback_count
        if paired
        else None
    )
    return ScoreDiagnostics(
        prediction_count=len(scores),
        unique_score_count=len(score_counts),
        score_counts=score_counts,
        score_regions=regions,
        boundary_count=boundary_count,
        boundary_rate=boundary_count / len(scores) if scores else 0.0,
        feedback_count=feedback_count,
        mean_absolute_error=mae,
        mean_bias=bias,
        band_agreement=band_agreement,
    )


def profile_hash(profile: ResearchProfile) -> str:
    return hashlib.sha256(profile.model_dump_json().encode("utf-8")).hexdigest()


class FeedbackStore:
    """SQLite-backed feedback with one current label per profile and paper."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS relevance_feedback (
                    profile_hash TEXT NOT NULL,
                    arxiv_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    predicted_score INTEGER NOT NULL CHECK(predicted_score BETWEEN 0 AND 100),
                    user_score INTEGER NOT NULL CHECK(user_score BETWEEN 0 AND 100),
                    notes TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (profile_hash, arxiv_id)
                )
                """
            )

    def save(
        self,
        *,
        profile: ResearchProfile,
        paper: Paper,
        predicted_score: int,
        user_score: int,
        notes: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            profile_hash=profile_hash(profile),
            arxiv_id=paper.arxiv_id,
            title=paper.title,
            predicted_score=predicted_score,
            user_score=user_score,
            notes=notes.strip(),
            updated_at=datetime.now(UTC),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO relevance_feedback (
                    profile_hash, arxiv_id, title, predicted_score,
                    user_score, notes, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_hash, arxiv_id) DO UPDATE SET
                    title=excluded.title,
                    predicted_score=excluded.predicted_score,
                    user_score=excluded.user_score,
                    notes=excluded.notes,
                    updated_at=excluded.updated_at
                """,
                (
                    record.profile_hash,
                    record.arxiv_id,
                    record.title,
                    record.predicted_score,
                    record.user_score,
                    record.notes,
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def list_for_profile(self, profile: ResearchProfile) -> tuple[FeedbackRecord, ...]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT profile_hash, arxiv_id, title, predicted_score,
                       user_score, notes, updated_at
                FROM relevance_feedback
                WHERE profile_hash = ?
                ORDER BY updated_at DESC
                """,
                (profile_hash(profile),),
            ).fetchall()
        return tuple(FeedbackRecord.model_validate(dict(row)) for row in rows)
