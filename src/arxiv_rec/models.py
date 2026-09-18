"""Validated domain models shared by the UI, CLI, and recommendation pipeline."""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Score = Annotated[int, Field(ge=0, le=100)]


class RelevanceLabel(StrEnum):
    ESSENTIAL = "essential"
    HIGHLY_RELEVANT = "highly_relevant"
    RELATED = "related"
    PERIPHERAL = "peripheral"
    IRRELEVANT = "irrelevant"


class ProfileMatchTier(StrEnum):
    CORE_INTERSECTION = "core_intersection"
    PRIMARY_OR_HIGH_VALUE_ADJACENT = "primary_or_high_value_adjacent"
    MEANINGFUL_ADJACENT = "meaningful_adjacent"
    WEAKLY_RELATED = "weakly_related"
    BROAD_OVERLAP = "broad_overlap"
    NONE_OR_NEGATIVE = "none_or_negative"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RunMode(StrEnum):
    FIXTURE = "fixture"
    LIVE = "live"


class PaperUpdateKind(StrEnum):
    NEW_SUBMISSION = "new_submission"
    REVISED_VERSION = "revised_version"


def label_for_score(score: int) -> RelevanceLabel:
    """Map a provisional relevance score to its user-facing label."""
    if not 0 <= score <= 100:
        raise ValueError("score must be between 0 and 100")
    if score >= 90:
        return RelevanceLabel.ESSENTIAL
    if score >= 75:
        return RelevanceLabel.HIGHLY_RELEVANT
    if score >= 55:
        return RelevanceLabel.RELATED
    if score >= 35:
        return RelevanceLabel.PERIPHERAL
    return RelevanceLabel.IRRELEVANT


def _clean_unique(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(value.split())
        key = cleaned.casefold()
        if cleaned and key not in seen:
            result.append(cleaned)
            seen.add(key)
    return tuple(result)


def canonicalize_arxiv_id(value: str) -> tuple[str, int | None]:
    """Return an unversioned arXiv identifier and an optional version number."""
    candidate = value.strip()
    if "/abs/" in candidate:
        candidate = candidate.split("/abs/", maxsplit=1)[1]
    elif "/pdf/" in candidate:
        candidate = candidate.split("/pdf/", maxsplit=1)[1]
    else:
        parsed = urlsplit(candidate)
        if parsed.scheme and parsed.netloc:
            candidate = parsed.path.strip("/")
    candidate = candidate.removesuffix(".pdf").split("?", maxsplit=1)[0].strip("/")
    version_match = re.search(r"v(?P<version>\d+)$", candidate, flags=re.IGNORECASE)
    version = int(version_match.group("version")) if version_match else None
    if version_match:
        candidate = candidate[: version_match.start()]
    if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Za-z-]+)?/\d{7})", candidate):
        raise ValueError(f"invalid arXiv identifier: {value!r}")
    return candidate, version


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paper(StrictModel):
    arxiv_id: str
    version: int | None = Field(default=None, ge=1)
    title: str = Field(min_length=1)
    abstract: str = Field(min_length=1)
    authors: tuple[str, ...] = Field(min_length=1)
    categories: tuple[str, ...] = Field(min_length=1)
    primary_category: str = Field(min_length=1)
    published_at: datetime
    updated_at: datetime
    abs_url: str | None = None
    pdf_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def normalize_arxiv_id(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "arxiv_id" not in value:
            return value
        normalized = dict(value)
        identifier, identifier_version = canonicalize_arxiv_id(str(normalized["arxiv_id"]))
        normalized["arxiv_id"] = identifier
        if normalized.get("version") is None:
            normalized["version"] = identifier_version
        return normalized

    @field_validator("title", "abstract", mode="before")
    @classmethod
    def normalize_text(cls, value: object) -> str:
        return " ".join(str(value).split())

    @field_validator("authors", "categories", mode="before")
    @classmethod
    def normalize_string_collections(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))

    @field_validator("published_at", "updated_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("paper timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def validate_and_complete_metadata(self) -> Self:
        if self.primary_category not in self.categories:
            raise ValueError("primary_category must be present in categories")
        if self.updated_at < self.published_at:
            raise ValueError("updated_at cannot be earlier than published_at")
        object.__setattr__(self, "abs_url", self.abs_url or f"https://arxiv.org/abs/{self.arxiv_id}")
        object.__setattr__(self, "pdf_url", self.pdf_url or f"https://arxiv.org/pdf/{self.arxiv_id}")
        return self


class FetchedPaper(StrictModel):
    paper: Paper
    update_kind: PaperUpdateKind


class ArxivFetchReport(StrictModel):
    items: tuple[FetchedPaper, ...]
    categories: tuple[str, ...] = Field(min_length=1)
    cutoff: datetime
    api_total_results: int = Field(ge=0)
    scanned_count: int = Field(ge=0)
    page_size: int = Field(ge=1)
    safety_cap: int = Field(ge=1)
    complete_through_cutoff: bool
    truncated: bool
    catchup_batch: bool = False

    @field_validator("categories", mode="before")
    @classmethod
    def normalize_categories(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))

    @field_validator("cutoff")
    @classmethod
    def cutoff_requires_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetch cutoff must include a timezone")
        return value

    @model_validator(mode="after")
    def coverage_flags_are_consistent(self) -> Self:
        if self.complete_through_cutoff and self.truncated:
            raise ValueError("a complete fetch cannot also be truncated")
        return self

    @property
    def papers(self) -> tuple[Paper, ...]:
        return tuple(item.paper for item in self.items)

    @property
    def new_submission_count(self) -> int:
        return sum(item.update_kind is PaperUpdateKind.NEW_SUBMISSION for item in self.items)

    @property
    def revised_version_count(self) -> int:
        return sum(item.update_kind is PaperUpdateKind.REVISED_VERSION for item in self.items)


class ResearchProfile(StrictModel):
    source_text: str = Field(min_length=20)
    core_topics: tuple[str, ...] = Field(min_length=1)
    required_intersections: tuple[tuple[str, ...], ...] = ()
    primary_facets: tuple[str, ...] = ()
    methods: tuple[str, ...] = ()
    platforms: tuple[str, ...] = ()
    adjacent_topics: tuple[str, ...] = ()
    weakly_related_topics: tuple[str, ...] = ()
    negative_topics: tuple[str, ...] = ()
    arxiv_categories: tuple[str, ...] = Field(min_length=1)
    positive_examples: tuple[str, ...] = ()
    negative_examples: tuple[str, ...] = ()

    @field_validator(
        "core_topics",
        "primary_facets",
        "methods",
        "platforms",
        "adjacent_topics",
        "weakly_related_topics",
        "negative_topics",
        "arxiv_categories",
        "positive_examples",
        "negative_examples",
        mode="before",
    )
    @classmethod
    def normalize_lists(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))

    @field_validator("required_intersections", mode="before")
    @classmethod
    def normalize_intersections(cls, value: object) -> tuple[tuple[str, ...], ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of intersections")
        intersections: list[tuple[str, ...]] = []
        for item in value:
            if not isinstance(item, (list, tuple)):
                raise TypeError("each intersection must be a list or tuple")
            cleaned = _clean_unique(tuple(str(term) for term in item))
            if len(cleaned) < 2:
                raise ValueError("each required intersection needs at least two concepts")
            intersections.append(cleaned)
        return tuple(intersections)


class RecallSignals(StrictModel):
    keyword_score: float = Field(ge=0.0, le=1.0)
    bm25_score: float = Field(ge=0.0, le=1.0)
    embedding_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    category_match: bool
    intersection_coverage: float = Field(ge=0.0, le=1.0)
    negative_penalty: float = Field(ge=0.0, le=1.0)
    hybrid_score: float = Field(ge=0.0, le=1.0)
    matched_terms: tuple[str, ...] = ()
    excluded_terms: tuple[str, ...] = ()
    sources: tuple[str, ...] = Field(min_length=1)

    @field_validator("matched_terms", "excluded_terms", "sources", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))


class RelevanceAssessment(StrictModel):
    relevance_score: Score
    raw_relevance_score: Score | None = None
    label: RelevanceLabel
    profile_tier: ProfileMatchTier | None = None
    confidence: Confidence
    matched_facets: tuple[str, ...] = ()
    missing_facets: tuple[str, ...] = ()
    reason: str = Field(min_length=10, max_length=800)

    @field_validator("matched_facets", "missing_facets", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))

    @model_validator(mode="after")
    def label_must_match_score(self) -> Self:
        expected = label_for_score(self.relevance_score)
        if self.label is not expected:
            raise ValueError(
                f"label {self.label.value!r} does not match score; expected {expected.value!r}"
            )
        return self


class Recommendation(StrictModel):
    paper: Paper
    recall: RecallSignals
    assessment: RelevanceAssessment
    selected: bool
    needs_review: bool = False


class RunRequest(StrictModel):
    profile: ResearchProfile
    mode: RunMode = RunMode.FIXTURE
    lookback_days: int = Field(default=1, ge=1, le=30)
    relevance_threshold: Score = 75
    fixture_path: str | None = None

    @model_validator(mode="after")
    def fixture_mode_requires_path(self) -> Self:
        if self.mode is RunMode.FIXTURE and not self.fixture_path:
            raise ValueError("fixture_path is required in fixture mode")
        return self


class ProfileAnalysisResult(StrictModel):
    profile: ResearchProfile
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class RecalledPaper(StrictModel):
    paper: Paper
    signals: RecallSignals
    included: bool


class RecallReport(StrictModel):
    items: tuple[RecalledPaper, ...]
    candidate_threshold: float = Field(ge=0.0, le=1.0)
    max_candidates: int = Field(ge=1)
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_tokens: int | None = Field(default=None, ge=0)

    @property
    def candidates(self) -> tuple[RecalledPaper, ...]:
        return tuple(item for item in self.items if item.included)


class ModelRelevanceJudgment(StrictModel):
    relevance_score: Score
    profile_tier: ProfileMatchTier
    confidence: Confidence
    matched_facets: tuple[str, ...] = ()
    missing_facets: tuple[str, ...] = ()
    reason: str = Field(min_length=10, max_length=800)

    @field_validator("matched_facets", "missing_facets", mode="before")
    @classmethod
    def normalize_lists(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            raise TypeError("expected a list or tuple of strings")
        return _clean_unique(tuple(str(item) for item in value))

    def to_assessment(self) -> RelevanceAssessment:
        calibrated_score = calibrate_score_for_tier(
            self.relevance_score, self.profile_tier
        )
        return RelevanceAssessment(
            relevance_score=calibrated_score,
            raw_relevance_score=self.relevance_score,
            label=label_for_score(calibrated_score),
            profile_tier=self.profile_tier,
            confidence=self.confidence,
            matched_facets=self.matched_facets,
            missing_facets=self.missing_facets,
            reason=self.reason,
        )


_TIER_SCORE_RULES: dict[ProfileMatchTier, tuple[int, int, int]] = {
    ProfileMatchTier.CORE_INTERSECTION: (90, 100, 96),
    ProfileMatchTier.PRIMARY_OR_HIGH_VALUE_ADJACENT: (75, 89, 82),
    ProfileMatchTier.MEANINGFUL_ADJACENT: (55, 74, 64),
    ProfileMatchTier.WEAKLY_RELATED: (35, 54, 47),
    ProfileMatchTier.BROAD_OVERLAP: (20, 34, 28),
    ProfileMatchTier.NONE_OR_NEGATIVE: (0, 19, 10),
}


def calibrate_score_for_tier(score: int, tier: ProfileMatchTier) -> int:
    """Keep in-band LLM scores and replace contradictory scores with a stable anchor."""
    minimum, maximum, anchor = _TIER_SCORE_RULES[tier]
    return score if minimum <= score <= maximum else anchor


class RankingReport(StrictModel):
    items: tuple[Recommendation, ...]
    relevance_threshold: Score
    review_margin: int = Field(ge=0, le=25)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cached_count: int = Field(default=0, ge=0)

    @property
    def selected(self) -> tuple[Recommendation, ...]:
        return tuple(item for item in self.items if item.selected)
