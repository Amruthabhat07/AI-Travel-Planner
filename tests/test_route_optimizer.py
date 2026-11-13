"""Offline tests for geographic distance, DBSCAN, and route optimization."""

from __future__ import annotations

import pytest

from backend.tools.route_optimizer import (
    assign_attractions_to_days,
    cluster_attractions,
    haversine_distance_km,
    optimize_route,
)


def _place(name: str, latitude: float | None, longitude: float | None) -> dict[str, object]:
    return {"name": name, "latitude": latitude, "longitude": longitude}


def test_haversine_known_city_pair_is_approximate_straight_line_distance() -> None:
    # Central London to central Paris: great-circle distance is approximately 344 km.
    distance = haversine_distance_km((51.5074, -0.1278), (48.8566, 2.3522))

    assert distance == pytest.approx(343.6, abs=1.0)


def test_haversine_identical_coordinates_is_zero() -> None:
    assert haversine_distance_km((35.0, 139.0), (35.0, 139.0)) == 0


@pytest.mark.parametrize(
    "point_a,point_b",
    [
        ((91, 0), (0, 0)),
        ((0, 181), (0, 0)),
        ((float("nan"), 0), (0, 0)),
    ],
)
def test_invalid_coordinates_are_rejected(
    point_a: tuple[float, float], point_b: tuple[float, float]
) -> None:
    with pytest.raises(ValueError):
        haversine_distance_km(point_a, point_b)


def test_dbscan_groups_nearby_attractions() -> None:
    attractions = [
        _place("A", 35.0000, 139.0000),
        _place("B", 35.0010, 139.0010),
        _place("C", 35.0020, 139.0010),
    ]

    result = cluster_attractions(attractions, epsilon_km=1.0, min_samples=2)

    assert len(result.clusters) == 1
    assert {item["name"] for item in result.clusters[0]} == {"A", "B", "C"}
    assert result.noise == []


def test_dbscan_keeps_distant_attractions_separate_as_noise() -> None:
    attractions = [_place("Tokyo", 35.6762, 139.6503), _place("Kyoto", 35.0116, 135.7681)]

    result = cluster_attractions(attractions, epsilon_km=5.0, min_samples=2)

    assert result.clusters == []
    assert {item["name"] for item in result.noise} == {"Tokyo", "Kyoto"}


def test_missing_coordinates_are_kept_separate() -> None:
    result = cluster_attractions(
        [_place("Located", 35, 139), _place("Unlocated", None, None)]
    )

    assert len(result.without_coordinates) == 1
    assert result.without_coordinates[0]["name"] == "Unlocated"
    assert result.noise[0]["name"] == "Located"


def test_single_attraction_is_preserved_as_noise() -> None:
    attraction = _place("One", 35, 139)

    result = cluster_attractions([attraction])

    assert result.clusters == []
    assert result.noise == [attraction]


def test_nearest_neighbor_route_is_deterministic_and_preserves_input() -> None:
    attractions = [
        _place("A", 0, 0),
        _place("B", 0, 1),
        _place("C", 1, 1),
        _place("D", 1, 0),
    ]

    route = optimize_route(attractions)

    assert route.ordered_attractions[0] == attractions[0]
    assert {item["name"] for item in route.ordered_attractions} == {"A", "B", "C", "D"}
    assert route.distance_km is not None
    assert optimize_route(attractions).ordered_attractions == route.ordered_attractions


def test_two_opt_never_worsens_seed_order_distance() -> None:
    attractions = [
        _place("A", 0, 0),
        _place("B", 0, 2),
        _place("C", 1, 0),
        _place("D", 1, 2),
        _place("E", 0.5, 1),
    ]
    seed_length = sum(
        haversine_distance_km(
            (attractions[index]["latitude"], attractions[index]["longitude"]),
            (attractions[index + 1]["latitude"], attractions[index + 1]["longitude"]),
        )
        for index in range(len(attractions) - 1)
    )

    route = optimize_route(attractions)

    assert route.distance_km is not None
    assert route.distance_km <= seed_length + 1e-8
    assert len(route.ordered_attractions) == len(attractions)


def test_duplicate_attractions_are_removed_without_losing_distinct_places() -> None:
    first = _place("Museum", 35, 139)
    duplicate = _place(" museum ", 35, 139)
    other = _place("Museum", 36, 140)

    route = optimize_route([first, duplicate, other])

    assert len(route.ordered_attractions) == 2
    assert {item["latitude"] for item in route.ordered_attractions} == {35, 36}


def test_optional_start_and_end_coordinates_are_included_in_distance() -> None:
    attractions = [_place("A", 0, 1), _place("B", 0, 2)]

    route = optimize_route(attractions, start=(0, 0), end=(0, 3))

    assert route.ordered_attractions[0]["name"] == "A"
    assert route.distance_km == pytest.approx(
        haversine_distance_km((0, 0), (0, 1))
        + haversine_distance_km((0, 1), (0, 2))
        + haversine_distance_km((0, 2), (0, 3))
    )


def test_route_preserves_every_selected_attraction() -> None:
    attractions = [_place(f"Spot {index}", 35 + index * 0.01, 139) for index in range(7)]

    route = optimize_route(attractions)

    assert {item["name"] for item in route.ordered_attractions} == {
        item["name"] for item in attractions
    }


def test_assignments_use_clusters_and_preserve_missing_coordinate_candidates() -> None:
    attractions = [
        _place("A", 35.000, 139.000),
        _place("B", 35.001, 139.001),
        _place("C", 35.002, 139.001),
        _place("Far", 36.0, 140.0),
        _place("No coordinates", None, None),
    ]

    plan = assign_attractions_to_days(attractions, duration=2, epsilon_km=1, min_samples=2)

    assert len(plan.days) == 2
    assert {item["name"] for day in plan.days for item in day} == {
        "A", "B", "C", "Far", "No coordinates"
    }
    # The DBSCAN cluster remains a cohesive group because it fits the daily cap.
    assert any({item["name"] for item in day} >= {"A", "B", "C"} for day in plan.days)


def test_large_cluster_is_split_to_respect_daily_soft_capacity() -> None:
    attractions = [
        _place(f"Nearby {index}", 35 + index * 0.0001, 139 + index * 0.0001)
        for index in range(8)
    ]

    plan = assign_attractions_to_days(
        attractions,
        duration=2,
        epsilon_km=1.0,
        min_samples=2,
        max_attractions_per_day=4,
    )

    assert [len(day) for day in plan.days] == [4, 4]
    assert {item["name"] for day in plan.days for item in day} == {
        item["name"] for item in attractions
    }


def test_route_rejects_missing_or_inconsistent_coordinates() -> None:
    with pytest.raises(ValueError, match="verified coordinates"):
        optimize_route([_place("No coordinates", None, None)])
    with pytest.raises(ValueError, match="provided together"):
        cluster_attractions([_place("Partial", 35, None)])
