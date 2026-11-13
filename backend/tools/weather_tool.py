"""Open-Meteo geocoding and forecast adapter with no API key requirement."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

import httpx

from backend.models.schemas import WeatherReport

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_PROVIDER_DOCS = "https://open-meteo.com/en/docs"


class WeatherProviderError(RuntimeError):
    """Raised when location lookup or a forecast response is unusable."""


class WeatherProvider(Protocol):
    """Interface for providers that return forecast coverage for a date window."""

    def fetch_forecast(self, destination: str, start_date: date, end_date: date) -> WeatherReport:
        """Return verified daily forecast data and explicit unavailable dates."""


class OpenMeteoWeatherProvider:
    """Open-Meteo forecast provider; its forecast API does not require an API key."""

    def __init__(self, *, timeout_seconds: float = 10.0, max_forecast_days: int = 16) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if max_forecast_days < 1:
            raise ValueError("max_forecast_days must be at least one.")
        self.timeout_seconds = timeout_seconds
        self.max_forecast_days = max_forecast_days

    def fetch_forecast(self, destination: str, start_date: date, end_date: date) -> WeatherReport:
        """Fetch only provider-supported actual forecast dates for the trip."""
        if not destination.strip():
            raise ValueError("Destination must not be empty.")
        if end_date < start_date:
            raise ValueError("Forecast end date must not precede start date.")

        requested_dates = _date_range(start_date, end_date)
        retrieved_at = datetime.now(timezone.utc).isoformat()
        location = self._geocode(destination)
        if location is None:
            return WeatherReport(
                status="unavailable",
                provider="Open-Meteo",
                retrieved_at=retrieved_at,
                unavailable_dates=requested_dates,
                unavailable_reason="Destination could not be resolved by the geocoding provider.",
            )

        today = date.today()
        forecast_start = max(start_date, today)
        forecast_end = min(end_date, today + timedelta(days=self.max_forecast_days - 1))
        if forecast_end < forecast_start:
            return WeatherReport(
                status="unavailable",
                location=location["name"],
                provider="Open-Meteo",
                retrieved_at=retrieved_at,
                unavailable_dates=requested_dates,
                unavailable_reason=(
                    "The requested dates are outside Open-Meteo's forecast horizon; "
                    "no climate averages are substituted."
                ),
            )

        try:
            response = httpx.get(
                _FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "temperature_unit": "celsius",
                    "precipitation_unit": "mm",
                    "timezone": "auto",
                    "start_date": forecast_start.isoformat(),
                    "end_date": forecast_end.isoformat(),
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise WeatherProviderError("Open-Meteo forecast request failed or returned invalid JSON.") from exc

        try:
            daily = payload["daily"]
            dates = daily["time"]
            codes = daily.get("weather_code", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precipitation = daily.get("precipitation_sum", [])
            if not isinstance(dates, list):
                raise TypeError("daily.time must be a list")
        except (KeyError, TypeError) as exc:
            raise WeatherProviderError("Open-Meteo forecast response is missing daily fields.") from exc

        forecast_by_date: dict[date, dict[str, Any]] = {}
        for index, raw_date in enumerate(dates):
            try:
                forecast_date = date.fromisoformat(raw_date)
            except (TypeError, ValueError) as exc:
                raise WeatherProviderError("Open-Meteo returned an invalid forecast date.") from exc
            forecast_by_date[forecast_date] = {
                "date": forecast_date,
                "condition": _weather_condition(_list_item(codes, index)),
                "temperature_min_c": _number_or_none(_list_item(min_temps, index)),
                "temperature_max_c": _number_or_none(_list_item(max_temps, index)),
                "precipitation_mm": _number_or_none(_list_item(precipitation, index)),
                "source": {
                    "title": "Open-Meteo forecast",
                    "url": _PROVIDER_DOCS,
                    "provider": "Open-Meteo",
                    "retrieved_at": retrieved_at,
                    "information_type": "forecast",
                },
            }

        requested_forecast_dates = {
            current for current in requested_dates if forecast_start <= current <= forecast_end
        }
        available_dates = requested_forecast_dates.intersection(forecast_by_date)
        forecast_items = [forecast_by_date[current] for current in sorted(available_dates)]
        unavailable = [current for current in requested_dates if current not in available_dates]
        if not forecast_items:
            status = "unavailable"
            reason = "No forecast observations were returned for the requested trip dates."
        elif unavailable:
            status = "partial_forecast"
            reason = "Some requested dates fall outside the provider forecast coverage or were not returned."
        else:
            status = "forecast"
            reason = None

        return WeatherReport(
            status=status,
            location=location["name"],
            provider="Open-Meteo",
            retrieved_at=retrieved_at,
            daily_forecasts=forecast_items,
            unavailable_dates=unavailable,
            unavailable_reason=reason,
        )

    def _geocode(self, destination: str) -> dict[str, Any] | None:
        """Resolve a destination using Open-Meteo's public geocoding endpoint."""
        try:
            response = httpx.get(
                _GEOCODING_URL,
                params={"name": destination.strip(), "count": 1, "language": "en", "format": "json"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            raise WeatherProviderError("Open-Meteo destination lookup failed.") from exc
        results = payload.get("results", []) if isinstance(payload, dict) else []
        if not results:
            return None
        result = results[0]
        try:
            return {
                "name": str(result.get("name") or destination),
                "latitude": float(result["latitude"]),
                "longitude": float(result["longitude"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise WeatherProviderError("Open-Meteo geocoding result lacked valid coordinates.") from exc


def fetch_weather_forecast(
    destination: str, start_date: date, end_date: date
) -> WeatherReport:
    """Convenience function using the default Open-Meteo provider."""
    return OpenMeteoWeatherProvider().fetch_forecast(destination, start_date, end_date)


def _date_range(start_date: date, end_date: date) -> list[date]:
    """Return inclusive trip dates."""
    return [start_date + timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]


def _list_item(values: Any, index: int) -> Any:
    """Get a parallel-array value if supplied by the provider."""
    return values[index] if isinstance(values, list) and index < len(values) else None


def _number_or_none(value: Any) -> float | None:
    """Convert finite numeric provider readings, leaving invalid/missing values absent."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return float(value)


def _weather_condition(code: Any) -> str | None:
    """Map documented WMO weather codes into short factual condition labels."""
    try:
        numeric_code = int(code)
    except (TypeError, ValueError):
        return None
    return {
        0: "Clear sky",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing rime fog",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        56: "Light freezing drizzle",
        57: "Dense freezing drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        66: "Light freezing rain",
        67: "Heavy freezing rain",
        71: "Slight snow fall",
        73: "Moderate snow fall",
        75: "Heavy snow fall",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }.get(numeric_code)
