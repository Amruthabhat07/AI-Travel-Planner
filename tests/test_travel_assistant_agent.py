"""Offline tests for the weather and practical Travel Assistant Agent."""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.agents import travel_assistant_agent as assistant_module
from backend.agents.travel_assistant_agent import travel_assistant_agent
from backend.models.schemas import TravelRequest, WeatherReport
from backend.tools import weather_tool
from backend.tools.weather_tool import OpenMeteoWeatherProvider, WeatherProviderError


class MockWeatherProvider:
    """Deterministic weather-provider test double."""

    def __init__(self, value: object = None, error: Exception | None = None) -> None:
        self.value = value
        self.error = error
        self.calls: list[tuple[str, date, date]] = []

    def fetch_forecast(self, destination: str, start_date: date, end_date: date) -> object:
        self.calls.append((destination, start_date, end_date))
        if self.error:
            raise self.error
        return self.value


def _weather_report(*, status: str = "forecast", missing: list[date] | None = None) -> WeatherReport:
    daily = [
        {
            "date": date(2027, 5, 1),
            "condition": "Light rain",
            "temperature_min_c": 12.0,
            "temperature_max_c": 21.0,
            "precipitation_mm": 4.5,
            "source": {
                "title": "Open-Meteo forecast",
                "url": "https://open-meteo.com/en/docs",
                "provider": "Open-Meteo",
                "retrieved_at": "2026-11-12T10:00:00+00:00",
                "information_type": "forecast",
            },
        }
    ]
    if status == "forecast":
        daily.extend(
            [
                {
                    "date": date(2027, 5, 2),
                    "condition": "Partly cloudy",
                    "temperature_min_c": 14.0,
                    "temperature_max_c": 24.0,
                    "precipitation_mm": 0.0,
                    "source": {
                        "title": "Open-Meteo forecast",
                        "url": "https://open-meteo.com/en/docs",
                        "provider": "Open-Meteo",
                        "retrieved_at": "2026-11-12T10:00:00+00:00",
                        "information_type": "forecast",
                    },
                }
            ]
        )
    return WeatherReport(
        status=status,
        location="Sample City",
        retrieved_at="2026-11-12T10:00:00+00:00",
        daily_forecasts=daily,
        unavailable_dates=missing or [],
        unavailable_reason="Some requested dates outside forecast coverage." if missing else None,
    )


def _draft_payload(*, source_url: str = "https://example.test/transit") -> dict[str, object]:
    source = {
        "title": "Official transit guide",
        "url": source_url,
        "provider": "Tavily",
        "information_type": "web_research",
    }
    verified = {"text": "Check the official transit guide for current service details.", "basis": "verified_source", "sources": [source]}
    general = {"text": "Carry a reusable water bottle if suitable for your itinerary.", "basis": "general_guidance", "sources": []}
    return {
        "transportation": {
            "status": "verified",
            "local_options": [verified],
            "intercity_options": [verified],
            "logistics_tips": [verified],
        },
        "packing": {
            "clothing": [general],
            "footwear": [{"text": "Choose comfortable walking shoes.", "basis": "itinerary", "sources": []}],
            "personal_essentials": [],
            "travel_documents": [],
            "electronics_accessories": [],
            "weather_specific": [{"text": "Pack a rain layer based on the returned rain forecast.", "basis": "forecast", "sources": [{"title": "Open-Meteo forecast", "url": "https://open-meteo.com/en/docs", "provider": "Open-Meteo", "information_type": "forecast"}]}],
            "activity_specific": [],
            "forecast_based": True,
        },
        "practical_tips": [verified, general],
    }


def _base_state(**overrides: object) -> dict[str, object]:
    request = TravelRequest(
        destination="Sample City",
        start_date="2027-05-01",
        end_date="2027-05-02",
        duration=2,
        travelers=2,
        preferences=["walking", "food"],
    )
    state: dict[str, object] = {
        **request.model_dump(mode="python", exclude_none=True),
        "user_request": "Plan a walking trip.",
        "destination_research": {
            "destination": "Sample City",
            "local_culture": "Research notes about local customs.",
            "research_sources": ["https://example.test/culture"],
        },
        "itinerary": [{"day_number": 1, "date": date(2027, 5, 1), "attraction_order": ["Museum"]}],
        "budget_breakdown": {"currency": "INR", "total_estimated_cost": "100.00"},
        "agent_status": {"itinerary_planning": "completed"},
        "errors": [],
    }
    state.update(overrides)
    return state


def _mock_assistant_llm(monkeypatch: pytest.MonkeyPatch, payload: object) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=payload)
    monkeypatch.setattr(assistant_module, "get_llm", lambda: llm)
    return llm


def _mock_search(monkeypatch: pytest.MonkeyPatch, results: list[dict[str, object]]) -> MagicMock:
    search = MagicMock(return_value=results)
    monkeypatch.setattr(assistant_module, "search_tavily", search)
    return search


def test_successful_weather_retrieval_and_structured_result(monkeypatch: pytest.MonkeyPatch) -> None:
    weather = _weather_report()
    provider = MockWeatherProvider(weather)
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Official transport guide."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    result = update["travel_information"]
    assert result["weather"]["status"] == "forecast"
    assert result["weather"]["daily_forecasts"][0]["temperature_max_c"] == 21.0
    assert provider.calls == [("Sample City", date(2027, 5, 1), date(2027, 5, 2))]
    assert update["agent_status"]["travel_assistant"] == "completed"


def test_weather_api_failure_is_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, "{invalid")
    provider = MockWeatherProvider(error=WeatherProviderError("network timeout"))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    info = update["travel_information"]
    assert info["weather"]["status"] == "unavailable"
    assert info["weather"]["daily_forecasts"] == []
    assert info.get("temperature") is None
    assert any("network timeout" in error for error in update["errors"])
    assert update["agent_status"]["travel_assistant"] == "completed_with_warnings"


def test_partial_weather_forecast_marks_unavailable_dates(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = [date(2027, 5, 2)]
    provider = MockWeatherProvider(_weather_report(status="partial_forecast", missing=missing))
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    assert update["travel_information"]["weather"]["status"] == "partial_forecast"
    assert update["travel_information"]["weather"]["unavailable_dates"] == missing


def test_transportation_information_uses_and_preserves_research_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [{"title": "Transit guide", "url": "https://example.test/transit", "content": "Public transport information."}]
    search = _mock_search(monkeypatch, sources)
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))
    provider = MockWeatherProvider(_weather_report())

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    transport = update["travel_information"]["transportation"]
    assert transport["status"] == "verified"
    assert transport["local_options"][0]["sources"][0]["url"] == sources[0]["url"]
    assert search.call_count == 3


def test_missing_transport_research_is_labeled_general_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, "not valid JSON")
    provider = MockWeatherProvider(_weather_report())

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    transport = update["travel_information"]["transportation"]
    assert transport["status"] == "general_guidance"
    assert transport["local_options"][0]["basis"] == "general_guidance"
    assert "availability" in transport["local_options"][0]["text"].lower()


def test_packing_recommendations_have_all_requested_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Transit facts."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))
    provider = MockWeatherProvider(_weather_report())

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    packing = update["travel_information"]["packing"]
    for category in (
        "clothing", "footwear", "personal_essentials", "travel_documents",
        "electronics_accessories", "weather_specific", "activity_specific",
    ):
        assert category in packing


def test_forecast_based_packing_is_marked_and_weather_cited(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Transit facts."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    packing = update["travel_information"]["packing"]
    assert packing["forecast_based"] is True
    assert packing["weather_specific"][0]["basis"] == "forecast"


def test_general_packing_when_weather_is_unavailable_does_not_claim_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, "invalid output")
    provider = MockWeatherProvider(error=WeatherProviderError("offline"))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    packing = update["travel_information"]["packing"]
    assert packing["forecast_based"] is False
    assert packing["weather_specific"] == []
    assert all(
        item["basis"] != "forecast"
        for group in packing.values()
        if isinstance(group, list)
        for item in group
    )


def test_practical_travel_tips_are_structured(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Customs guide", "url": "https://example.test/customs", "content": "Etiquette reference."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    tips = update["travel_information"]["practical_tips"]
    assert tips
    assert all("text" in tip and "basis" in tip for tip in tips)


def test_sources_are_preserved_without_model_fabrication(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Evidence."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    info = update["travel_information"]
    urls = {source["url"] for source in info["sources"] if source.get("url")}
    assert "https://example.test/transit" in urls
    assert "https://open-meteo.com/en/docs" in urls
    assert "https://example.test/culture" in urls


def test_unknown_gemini_source_url_is_rejected_and_fallback_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Evidence."}])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload(source_url="https://fabricated.test")))

    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    assert update["agent_status"]["travel_assistant"] == "completed_with_warnings"
    assert any("Gemini advice organization unavailable" in item for item in update["errors"])
    assert update["travel_information"]["weather"]["daily_forecasts"]


def test_missing_travel_request_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    update = travel_assistant_agent({}, weather_provider=MockWeatherProvider())  # type: ignore[arg-type]
    assert update["agent_status"]["travel_assistant"] == "failed"
    assert "Travel request is missing" in update["errors"][0]


def test_missing_destination_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    update = travel_assistant_agent({"start_date": "2027-05-01", "end_date": "2027-05-02"}, weather_provider=MockWeatherProvider())  # type: ignore[arg-type]
    assert update["agent_status"]["travel_assistant"] == "failed"
    assert "destination" in update["errors"][0].lower()


def test_missing_travel_dates_still_returns_non_weather_assistance(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _base_state()
    state.pop("start_date")
    state.pop("end_date")
    state.pop("duration")
    _mock_search(monkeypatch, [])
    update = travel_assistant_agent(state, weather_provider=MockWeatherProvider())  # type: ignore[arg-type]

    assert update["travel_information"]["weather"]["status"] == "unavailable"
    assert update["agent_status"]["travel_assistant"] == "completed_with_warnings"
    assert any("Travel dates are missing" in item for item in update["errors"])


def test_missing_optional_itinerary_and_budget_are_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _base_state()
    state.pop("itinerary")
    state.pop("budget_breakdown")
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))
    update = travel_assistant_agent(state, weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    assert update["travel_information"]["packing"] is not None
    assert update["agent_status"]["travel_assistant"] == "completed_with_warnings"


def test_correct_travel_state_update_only_targets_travel_information(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))
    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    assert "travel_information" in update
    assert set(update).issubset({"travel_information", "agent_status", "errors"})


def test_unrelated_state_fields_are_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _base_state(budget_status="pending", itinerary=[{"day_number": 1, "title": "keep"}], destination_research={"destination": "Sample City"})
    before = dict(state)
    _mock_search(monkeypatch, [])
    _mock_assistant_llm(monkeypatch, json.dumps(_draft_payload()))

    update = travel_assistant_agent(state, weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]
    result = {**state, **update}

    assert result["budget_status"] == before["budget_status"]
    assert result["itinerary"] == before["itinerary"]
    assert result["destination_research"] == before["destination_research"]


def test_weather_failure_never_fabricates_weather_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    provider = MockWeatherProvider(error=RuntimeError("connection lost"))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    report = update["travel_information"]["weather"]
    assert report["status"] == "unavailable"
    assert report["daily_forecasts"] == []
    assert report["unavailable_dates"] == [date(2027, 5, 1), date(2027, 5, 2)]
    assert update["travel_information"].get("temperature") is None


def test_invalid_weather_provider_payload_is_marked_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    provider = MockWeatherProvider({"status": "forecast", "daily_forecasts": [{"date": "bad"}]})

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    assert update["travel_information"]["weather"]["status"] == "unavailable"
    assert update["travel_information"]["weather"]["daily_forecasts"] == []
    assert any("Weather forecast unavailable" in item for item in update["errors"])


def test_invalid_gemini_output_falls_back_to_deterministic_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [{"title": "Transit", "url": "https://example.test/transit", "content": "Evidence."}])
    llm = _mock_assistant_llm(monkeypatch, "this is not JSON")

    update = travel_assistant_agent(_base_state(), weather_provider=MockWeatherProvider(_weather_report()))  # type: ignore[arg-type]

    assert llm.invoke.call_count == 1
    assert update["travel_information"]["transportation"]["status"] == "general_guidance"
    assert update["agent_status"]["travel_assistant"] == "completed_with_warnings"
    assert any("Gemini advice organization unavailable" in item for item in update["errors"])


def test_gemini_is_not_called_when_no_external_evidence_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_search(monkeypatch, [])
    llm = MagicMock()
    monkeypatch.setattr(assistant_module, "get_llm", lambda: llm)
    provider = MockWeatherProvider(error=WeatherProviderError("unavailable"))

    update = travel_assistant_agent(_base_state(), weather_provider=provider)  # type: ignore[arg-type]

    llm.invoke.assert_not_called()
    assert update["travel_information"]["packing"]["forecast_based"] is False


def test_open_meteo_provider_parses_mocked_geocoding_and_daily_forecast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    today = date.today()
    first_day = today.replace(day=today.day)  # Keep a deterministic date object for request assertions.
    geocoding_response = MagicMock()
    geocoding_response.json.return_value = {
        "results": [{"name": "Sample City", "latitude": 12.5, "longitude": 77.5}]
    }
    forecast_response = MagicMock()
    forecast_response.json.return_value = {
        "daily": {
            "time": [first_day.isoformat()],
            "weather_code": [61],
            "temperature_2m_max": [28.2],
            "temperature_2m_min": [19.1],
            "precipitation_sum": [3.4],
        }
    }
    monkeypatch.setattr(weather_tool.httpx, "get", MagicMock(side_effect=[geocoding_response, forecast_response]))

    result = OpenMeteoWeatherProvider().fetch_forecast("Sample City", first_day, first_day)

    assert result.status == "forecast"
    assert result.daily_forecasts[0].condition == "Slight rain"
    assert result.daily_forecasts[0].temperature_max_c == 28.2
    assert result.daily_forecasts[0].precipitation_mm == 3.4
    assert result.daily_forecasts[0].source.provider == "Open-Meteo"


def test_open_meteo_provider_marks_out_of_horizon_dates_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geocoding_response = MagicMock()
    geocoding_response.json.return_value = {
        "results": [{"name": "Sample City", "latitude": 12.5, "longitude": 77.5}]
    }
    get = MagicMock(return_value=geocoding_response)
    monkeypatch.setattr(weather_tool.httpx, "get", get)
    future = date.today().replace(year=date.today().year + 2)

    result = OpenMeteoWeatherProvider().fetch_forecast("Sample City", future, future)

    assert result.status == "unavailable"
    assert result.daily_forecasts == []
    assert result.unavailable_dates == [future]
    assert "forecast horizon" in result.unavailable_reason
    get.assert_called_once()  # Geocoding only; never requests unsupported forecast dates.
