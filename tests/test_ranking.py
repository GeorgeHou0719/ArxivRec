import json
from pathlib import Path
from types import SimpleNamespace

from arxiv_rec.cli import inspect_ranking_fixture
from arxiv_rec.models import (
    Confidence,
    ModelRelevanceJudgment,
    Paper,
    ProfileMatchTier,
    RecalledPaper,
    RecallSignals,
    ResearchProfile,
)
from arxiv_rec.ranking import (
    RANKING_INSTRUCTIONS,
    OpenAIRelevanceRanker,
)


def load_one_candidate() -> tuple[ResearchProfile, RecalledPaper]:
    payload = json.loads(Path("fixtures/cavity_qed_papers.json").read_text(encoding="utf-8"))
    profile = ResearchProfile.model_validate(payload["profile"])
    paper = Paper.model_validate(payload["papers"][0])
    signals = RecallSignals(
        keyword_score=0.8,
        bm25_score=1.0,
        embedding_score=0.99,
        category_match=True,
        intersection_coverage=1.0,
        negative_penalty=0.0,
        hybrid_score=0.95,
        matched_terms=["cavity QED", "quantum network"],
        sources=["category", "embedding"],
    )
    return profile, RecalledPaper(paper=paper, signals=signals, included=True)


class FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, **kwargs):
        self.calls += 1
        assert kwargs["text_format"] is ModelRelevanceJudgment
        assert kwargs["store"] is False
        return SimpleNamespace(
            output_parsed=ModelRelevanceJudgment(
                relevance_score=94,
                profile_tier=ProfileMatchTier.CORE_INTERSECTION,
                confidence=Confidence.HIGH,
                matched_facets=["optical cavity", "quantum network"],
                missing_facets=[],
                reason="The paper directly matches the required cavity-network intersection.",
            ),
            usage=SimpleNamespace(input_tokens=300, output_tokens=60),
        )


def test_openai_ranker_scores_selects_and_caches(tmp_path: Path) -> None:
    profile, candidate = load_one_candidate()
    responses = FakeResponses()
    client = SimpleNamespace(responses=responses)
    ranker = OpenAIRelevanceRanker(
        model="ranking-test",
        cache_dir=tmp_path / "ranking-cache",
        client=client,
    )

    first = ranker.assess(profile, [candidate], relevance_threshold=75, review_margin=5)
    second = ranker.assess(profile, [candidate], relevance_threshold=75, review_margin=5)

    assert first.items[0].selected is True
    assert first.items[0].assessment.label.value == "essential"
    assert first.input_tokens == 300
    assert second.cached_count == 1
    assert responses.calls == 1


def test_fixture_ranking_is_threshold_based_and_explainable() -> None:
    report = inspect_ranking_fixture(
        Path("fixtures/cavity_qed_papers.json"),
        threshold=75,
        review_margin=5,
    )

    assert [item.paper.arxiv_id for item in report.selected] == ["2608.10001", "2608.10002"]
    assert [item.assessment.relevance_score for item in report.items] == [96, 82, 47, 12]
    assert all(item.assessment.reason for item in report.items)


def test_low_confidence_is_flagged_for_review_even_far_from_threshold(tmp_path: Path) -> None:
    profile, candidate = load_one_candidate()
    responses = FakeResponses()
    original_parse = responses.parse

    def low_confidence_parse(**kwargs):
        response = original_parse(**kwargs)
        response.output_parsed = response.output_parsed.model_copy(
            update={
                "relevance_score": 40,
                "profile_tier": ProfileMatchTier.NONE_OR_NEGATIVE,
                "confidence": Confidence.LOW,
            }
        )
        return response

    responses.parse = low_confidence_parse
    ranker = OpenAIRelevanceRanker(
        model="ranking-test-low",
        cache_dir=tmp_path / "ranking-cache",
        client=SimpleNamespace(responses=responses),
    )

    report = ranker.assess(profile, [candidate], relevance_threshold=75, review_margin=5)

    assert report.items[0].selected is False
    assert report.items[0].needs_review is True


def test_ranking_prompt_is_semantic_and_compositional() -> None:
    assert "Do not rely only on word overlap" in RANKING_INSTRUCTIONS
    assert "all concepts inside an intersection" in RANKING_INSTRUCTIONS
    assert "satisfy every listed intersection" in RANKING_INSTRUCTIONS


def test_out_of_band_llm_score_is_calibrated_from_profile_tier() -> None:
    judgment = ModelRelevanceJudgment(
        relevance_score=10,
        profile_tier=ProfileMatchTier.WEAKLY_RELATED,
        confidence=Confidence.HIGH,
        matched_facets=["optical cavity"],
        reason="This directly matches an explicitly weakly related cavity-optics topic.",
    )

    assessment = judgment.to_assessment()

    assert assessment.raw_relevance_score == 10
    assert assessment.relevance_score == 47
    assert assessment.label.value == "peripheral"
