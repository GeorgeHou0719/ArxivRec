import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arxiv_rec.models import Paper, ResearchProfile
from arxiv_rec.retrieval import (
    EmbeddingBatch,
    EmbeddingError,
    HybridRetriever,
    OpenAITextEmbedder,
    cosine_similarity,
    profile_embedding_text,
)


def load_fixture() -> tuple[ResearchProfile, list[Paper]]:
    payload = json.loads(Path("fixtures/cavity_qed_papers.json").read_text(encoding="utf-8"))
    return (
        ResearchProfile.model_validate(payload["profile"]),
        [Paper.model_validate(item) for item in payload["papers"]],
    )


class FixtureEmbedder:
    def embed(self, texts):
        assert len(texts) == 5
        return EmbeddingBatch(
            vectors=(
                (1.0, 0.0),
                (0.99, 0.05),
                (0.85, 0.25),
                (0.55, 0.65),
                (0.05, 0.99),
            ),
            provider="fixture",
            model="ordered-cavity-vectors",
            input_tokens=200,
        )


def test_hybrid_retrieval_orders_cavity_fixture_and_exposes_diagnostics() -> None:
    profile, papers = load_fixture()
    report = HybridRetriever(FixtureEmbedder()).retrieve(
        profile,
        papers,
        candidate_threshold=0.0,
        max_candidates=10,
    )

    assert [item.paper.arxiv_id for item in report.items] == [
        "2608.10001",
        "2608.10002",
        "2608.10003",
        "2608.10004",
    ]
    assert report.items[0].signals.intersection_coverage == pytest.approx(1.0)
    assert report.items[0].signals.negative_penalty == 0
    assert "embedding" in report.items[0].signals.sources
    assert report.items[-1].signals.negative_penalty == pytest.approx(1.0)
    assert report.embedding_tokens == 200


def test_hybrid_retrieval_applies_threshold_without_fixed_top_k() -> None:
    profile, papers = load_fixture()
    report = HybridRetriever(FixtureEmbedder()).retrieve(
        profile,
        papers,
        candidate_threshold=0.65,
        max_candidates=10,
    )

    included = [item.paper.arxiv_id for item in report.candidates]
    assert included == ["2608.10001", "2608.10002"]


def test_lexical_only_mode_does_not_require_api() -> None:
    profile, papers = load_fixture()
    report = HybridRetriever().retrieve(
        profile,
        papers,
        candidate_threshold=0.0,
        max_candidates=10,
    )

    assert report.embedding_provider is None
    assert all(item.signals.embedding_score is None for item in report.items)
    assert report.items[0].paper.arxiv_id == "2608.10001"


def test_profile_embedding_text_excludes_negative_topics_from_positive_query() -> None:
    profile, _ = load_fixture()
    text = profile_embedding_text(profile)

    assert "cavity-mediated quantum interconnects" in text
    assert "integrated photonics without" not in text


def test_cosine_similarity_validates_dimensions_and_zero_vectors() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    with pytest.raises(EmbeddingError, match="same non-zero dimension"):
        cosine_similarity([1], [1, 2])
    with pytest.raises(EmbeddingError, match="zero vectors"):
        cosine_similarity([0, 0], [1, 0])


class FakeEmbeddings:
    def create(self, **kwargs):
        assert kwargs["model"] == "embedding-test"
        assert kwargs["input"] == ["first", "second"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ],
            usage=SimpleNamespace(total_tokens=7),
        )


def test_openai_embedder_restores_response_order_and_usage() -> None:
    fake_client = SimpleNamespace(embeddings=FakeEmbeddings())
    embedder = OpenAITextEmbedder(model="embedding-test", client=fake_client)

    batch = embedder.embed(["first", "second"])

    assert batch.vectors == ((1.0, 0.0), (0.0, 1.0))
    assert batch.input_tokens == 7
    assert batch.provider == "openai"
