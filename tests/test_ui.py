from datetime import date, timedelta
from pathlib import Path

from arxiv_rec.pipeline import load_fixture
from arxiv_rec.ui import (
    FIXTURE_PATH,
    _local_date_range_utc,
    _parse_intersections,
    _profile_field_values,
)


def test_profile_editor_round_trip_helpers() -> None:
    profile, _, _ = load_fixture(FIXTURE_PATH)
    values = _profile_field_values(profile)

    assert _parse_intersections(values["required_intersections"]) == (
        ("optical cavity", "quantum network"),
    )


def test_live_only_ui_defaults_and_date_selector() -> None:
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert len(app.radio) == 1
    assert app.radio[0].label == "Search period"
    assert app.radio[0].value == "Lookback days"
    assert {slider.label: slider.value for slider in app.slider}[
        "Relevance threshold"
    ] == 40

    assert {button.label for button in app.button} >= {
        "1. Analyze research interests",
        "2. Find relevant papers",
    }

    app.radio[0].set_value("Specific date range").run()

    assert not app.exception
    assert len(app.date_input) == 1
    assert app.date_input[0].label == "Start and end dates"
    assert isinstance(app.date_input[0].value, tuple)
    assert len(app.date_input[0].value) == 2


def test_specific_date_range_includes_both_endpoint_dates() -> None:
    start, end = _local_date_range_utc(date(2026, 8, 7), date(2026, 8, 9))

    assert end - start == timedelta(days=3)
