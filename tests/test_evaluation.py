from pathlib import Path

import pytest

from arxiv_rec.cli import evaluate_fixture
from arxiv_rec.evaluation import GoldJudgment, GoldSet, ScorePrediction, evaluate_system


def test_threshold_metrics_detect_false_positive_and_false_negative() -> None:
    gold = GoldSet(
        name="small",
        judgments=[
            GoldJudgment(arxiv_id="a", relevance_score=90),
            GoldJudgment(arxiv_id="b", relevance_score=80),
            GoldJudgment(arxiv_id="c", relevance_score=20),
            GoldJudgment(arxiv_id="d", relevance_score=10),
        ],
    )
    predictions = [
        ScorePrediction(arxiv_id="a", score=95),
        ScorePrediction(arxiv_id="b", score=50),
        ScorePrediction(arxiv_id="c", score=90),
        ScorePrediction(arxiv_id="d", score=5),
    ]

    metrics = evaluate_system(
        system="test",
        gold_set=gold,
        predictions=predictions,
        threshold=75,
    )

    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.f1 == pytest.approx(0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)


def test_fixture_comparison_reports_all_baselines() -> None:
    report = evaluate_fixture(
        Path("fixtures/cavity_qed_papers.json"),
        Path("evals/gold/cavity_qed_seed.json"),
        threshold=75,
    )

    metrics = {metric.system: metric for metric in report.metrics}
    assert set(metrics) == {"keyword", "bm25", "embedding", "hybrid", "llm"}
    assert metrics["keyword"].recall == pytest.approx(0.5)
    assert metrics["hybrid"].f1 == pytest.approx(1.0)
    assert metrics["llm"].mean_absolute_error == pytest.approx(0.0)

