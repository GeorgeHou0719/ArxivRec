"""Offline metrics for relevance scores, threshold selection, and ranking quality."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from pydantic import Field, model_validator

from arxiv_rec.models import Score, StrictModel, label_for_score


class GoldJudgment(StrictModel):
    arxiv_id: str = Field(min_length=1)
    relevance_score: Score
    notes: str = ""


class GoldSet(StrictModel):
    name: str = Field(min_length=1)
    description: str = ""
    judgments: tuple[GoldJudgment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_are_unique(self) -> GoldSet:
        identifiers = [judgment.arxiv_id for judgment in self.judgments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("gold set contains duplicate arXiv identifiers")
        return self


class ScorePrediction(StrictModel):
    arxiv_id: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=100.0)


class EvaluationMetrics(StrictModel):
    system: str = Field(min_length=1)
    threshold: Score
    evaluated_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)
    false_positive_rate: float = Field(ge=0.0, le=1.0)
    mean_absolute_error: float = Field(ge=0.0, le=100.0)
    ndcg: float = Field(ge=0.0, le=1.0)


class EvaluationReport(StrictModel):
    gold_set: str
    threshold: Score
    metrics: tuple[EvaluationMetrics, ...]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _grade(score: float) -> int:
    grades: dict[str, int] = {
        "essential": 4,
        "highly_relevant": 3,
        "related": 2,
        "peripheral": 1,
        "irrelevant": 0,
    }
    return grades[label_for_score(round(score)).value]


def _dcg(grades: Sequence[int]) -> float:
    return math.fsum(
        (2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades)
    )


def evaluate_system(
    *,
    system: str,
    gold_set: GoldSet,
    predictions: Sequence[ScorePrediction],
    threshold: int,
) -> EvaluationMetrics:
    if not 0 <= threshold <= 100:
        raise ValueError("threshold must be between 0 and 100")
    prediction_map = {prediction.arxiv_id: prediction.score for prediction in predictions}
    if len(prediction_map) != len(predictions):
        raise ValueError("predictions contain duplicate arXiv identifiers")

    paired = [
        (judgment, prediction_map[judgment.arxiv_id])
        for judgment in gold_set.judgments
        if judgment.arxiv_id in prediction_map
    ]
    missing_count = len(gold_set.judgments) - len(paired)
    true_positive = sum(
        gold.relevance_score >= threshold and predicted >= threshold for gold, predicted in paired
    )
    false_positive = sum(
        gold.relevance_score < threshold and predicted >= threshold for gold, predicted in paired
    )
    false_negative = sum(
        gold.relevance_score >= threshold and predicted < threshold for gold, predicted in paired
    )
    true_negative = sum(
        gold.relevance_score < threshold and predicted < threshold for gold, predicted in paired
    )
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall) if precision + recall else 0.0
    mean_absolute_error = (
        sum(abs(gold.relevance_score - predicted) for gold, predicted in paired) / len(paired)
        if paired
        else 0.0
    )

    ranked = sorted(paired, key=lambda pair: pair[1], reverse=True)
    actual_grades = [_grade(gold.relevance_score) for gold, _ in ranked]
    ideal_grades = sorted((_grade(gold.relevance_score) for gold, _ in paired), reverse=True)
    ideal_dcg = _dcg(ideal_grades)
    ndcg = _dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

    return EvaluationMetrics(
        system=system,
        threshold=threshold,
        evaluated_count=len(paired),
        missing_count=missing_count,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=_safe_ratio(false_positive, false_positive + true_negative),
        mean_absolute_error=mean_absolute_error,
        ndcg=ndcg,
    )


def evaluate_all(
    *,
    gold_set: GoldSet,
    systems: Mapping[str, Sequence[ScorePrediction]],
    threshold: int,
) -> EvaluationReport:
    return EvaluationReport(
        gold_set=gold_set.name,
        threshold=threshold,
        metrics=tuple(
            evaluate_system(
                system=name,
                gold_set=gold_set,
                predictions=predictions,
                threshold=threshold,
            )
            for name, predictions in systems.items()
        ),
    )
