from types import SimpleNamespace

import pytest

from arxiv_rec.models import ResearchProfile
from arxiv_rec.profile_analyzer import (
    PROFILE_INSTRUCTIONS,
    FixtureProfileAnalyzer,
    OpenAIProfileAnalyzer,
    ProfileAnalysisError,
)

SOURCE = (
    "I work on cavity QED and quantum interconnects, especially optical cavities "
    "that connect remote quantum nodes."
)


def parsed_profile() -> ResearchProfile:
    return ResearchProfile(
        source_text=SOURCE,
        core_topics=["cavity-mediated quantum interconnects"],
        required_intersections=[["optical cavity", "quantum network"]],
        primary_facets=["cavity QED", "remote quantum nodes"],
        adjacent_topics=["quantum networks without cavities"],
        weakly_related_topics=["cavity optics without quantum information"],
        negative_topics=["general integrated photonics"],
        arxiv_categories=["quant-ph", "physics.optics"],
    )


class FakeResponses:
    def __init__(self, output: ResearchProfile | None) -> None:
        self.output = output
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            output_parsed=self.output,
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
        )


class FakeClient:
    def __init__(self, output: ResearchProfile | None) -> None:
        self.responses = FakeResponses(output)


def test_openai_analyzer_uses_structured_output_and_preserves_source() -> None:
    client = FakeClient(parsed_profile())
    analyzer = OpenAIProfileAnalyzer(model="test-model", client=client)

    result = analyzer.analyze(f"  {SOURCE}  ")

    assert result.profile.source_text == SOURCE
    assert result.profile.required_intersections == (("optical cavity", "quantum network"),)
    assert result.profile.positive_examples[-1] == (
        "A paper directly studying optical cavity and quantum network."
    )
    assert result.provider == "openai"
    assert result.input_tokens == 123
    assert client.responses.kwargs["text_format"] is ResearchProfile
    assert client.responses.kwargs["store"] is False
    assert SOURCE in client.responses.kwargs["input"]


def test_openai_analyzer_rejects_missing_parsed_output() -> None:
    analyzer = OpenAIProfileAnalyzer(model="test-model", client=FakeClient(None))

    with pytest.raises(ProfileAnalysisError, match="no parsed"):
        analyzer.analyze(SOURCE)


def test_live_analyzer_requires_key_or_injected_client() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenAIProfileAnalyzer(model="test-model")


def test_fixture_analyzer_is_deterministic() -> None:
    analyzer = FixtureProfileAnalyzer(parsed_profile(), fixture_name="cavity-qed")

    first = analyzer.analyze(SOURCE)
    second = analyzer.analyze(SOURCE)

    assert first == second
    assert first.provider == "fixture"
    assert first.model == "cavity-qed"


def test_prompt_requires_compositional_intersections() -> None:
    assert "required_intersections" in PROFILE_INSTRUCTIONS
    assert "matching only one concept" in PROFILE_INSTRUCTIONS
    assert '"generally interested"' in PROFILE_INSTRUCTIONS
    assert "every independent required_intersection" in PROFILE_INSTRUCTIONS
