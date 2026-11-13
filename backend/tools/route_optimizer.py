"""Deterministic straight-line geographic clustering and route ordering.

Distances computed here are Haversine great-circle estimates, not road distances
or travel-time estimates. The module uses only the Python standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

_EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class ClusteringResult:
    """DBSCAN groups plus coordinate-bearing noise and unlocated attractions."""

    clusters: list[list[Any]]
    noise: list[Any]
    without_coordinates: list[Any]


@dataclass(frozen=True)
class RouteOptimization:
    """A deterministic route order and approximate straight-line path length."""

    ordered_attractions: list[Any]
    distance_km: float | None


@dataclass(frozen=True)
class DailyAttractionPlan:
    """Attraction allocation for each travel day and source clustering result."""

    days: list[list[Any]]
    clustering: ClusteringResult


def _validate_coordinate(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate and normalize decimal-degree latitude and longitude."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError) as exc:
        raise ValueError("Latitude and longitude must be numeric.") from exc
    if not math.isfinite(lat) or not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")
    if not math.isfinite(lon) or not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees.")
    return lat, lon


def haversine_distance_km(
    point_a: tuple[float, float], point_b: tuple[float, float]
) -> float:
    """Calculate great-circle straight-line distance between two lat/lon pairs.

    Args:
        point_a: ``(latitude, longitude)`` in decimal degrees.
        point_b: ``(latitude, longitude)`` in decimal degrees.

    Returns:
        Approximate surface distance in kilometers. This is not a road distance.
    """
    lat1, lon1 = _validate_coordinate(*point_a)
    lat2, lon2 = _validate_coordinate(*point_b)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    hav = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    # Clamp against tiny floating-point drift outside [0, 1].
    hav = min(1.0, max(0.0, hav))
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(hav))


def _get_field(item: Any, key: str) -> Any:
    """Read a field from a mapping or an attribute-bearing object."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _coordinates(item: Any) -> tuple[float, float] | None:
    """Read and validate paired coordinates; return None if both are missing."""
    latitude = _get_field(item, "latitude")
    longitude = _get_field(item, "longitude")
    if latitude is None and longitude is None:
        return None
    if latitude is None or longitude is None:
        raise ValueError("Attraction latitude and longitude must be provided together.")
    return _validate_coordinate(latitude, longitude)


def _attraction_key(item: Any) -> tuple[Any, ...]:
    """Build a stable key for removing repeated attraction selections."""
    identifier = _get_field(item, "id")
    if identifier is not None:
        return ("id", str(identifier))
    name = str(_get_field(item, "name") or "").strip().casefold()
    coordinates = _coordinates(item)
    if coordinates is None:
        return ("name", name)
    return ("name", name, coordinates)


def _deduplicate(attractions: Sequence[Any]) -> list[Any]:
    """Remove repeated IDs or identical name/coordinate combinations."""
    unique: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for attraction in attractions:
        key = _attraction_key(attraction)
        if key not in seen:
            seen.add(key)
            unique.append(attraction)
    return unique


def cluster_attractions(
    attractions: Sequence[Any], *, epsilon_km: float = 5.0, min_samples: int = 2
) -> ClusteringResult:
    """Cluster located attractions using deterministic DBSCAN and Haversine km.

    ``epsilon_km`` is the neighborhood radius. Points that do not belong to a
    density-connected cluster are returned as individual ``noise`` items; they
    are not silently joined to a distant cluster. Attractions with no coordinate
    pair are returned separately. DBSCAN uses radians internally as required by
    the Haversine metric convention.
    """
    if not math.isfinite(epsilon_km) or epsilon_km <= 0:
        raise ValueError("epsilon_km must be a positive finite number.")
    if min_samples < 1:
        raise ValueError("min_samples must be at least one.")

    located: list[Any] = []
    radians: list[tuple[float, float]] = []
    without_coordinates: list[Any] = []
    for attraction in _deduplicate(attractions):
        coordinates = _coordinates(attraction)
        if coordinates is None:
            without_coordinates.append(attraction)
        else:
            located.append(attraction)
            radians.append((math.radians(coordinates[0]), math.radians(coordinates[1])))

    if not located:
        return ClusteringResult([], [], without_coordinates)

    epsilon_radians = epsilon_km / _EARTH_RADIUS_KM

    def neighbor_indices(index: int) -> list[int]:
        lat1, lon1 = radians[index]
        neighbors: list[int] = []
        for candidate, (lat2, lon2) in enumerate(radians):
            delta_lat = lat2 - lat1
            delta_lon = lon2 - lon1
            hav = (
                math.sin(delta_lat / 2) ** 2
                + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
            )
            distance_radians = 2 * math.asin(math.sqrt(min(1.0, max(0.0, hav))))
            if distance_radians <= epsilon_radians:
                neighbors.append(candidate)
        return neighbors

    labels: list[int | None] = [None] * len(located)
    cluster_id = 0
    for point_index in range(len(located)):
        if labels[point_index] is not None:
            continue
        neighbors = neighbor_indices(point_index)
        if len(neighbors) < min_samples:
            labels[point_index] = -1
            continue

        labels[point_index] = cluster_id
        pending = list(neighbors)
        pending_seen = set(pending)
        cursor = 0
        while cursor < len(pending):
            neighbor = pending[cursor]
            cursor += 1
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
            if labels[neighbor] is not None:
                continue
            labels[neighbor] = cluster_id
            expanded = neighbor_indices(neighbor)
            if len(expanded) >= min_samples:
                for expanded_index in expanded:
                    if expanded_index not in pending_seen:
                        pending_seen.add(expanded_index)
                        pending.append(expanded_index)
        cluster_id += 1

    clusters: list[list[Any]] = [[] for _ in range(cluster_id)]
    noise: list[Any] = []
    for attraction, label in zip(located, labels, strict=True):
        if label == -1 or label is None:
            noise.append(attraction)
        else:
            clusters[label].append(attraction)
    return ClusteringResult(clusters, noise, without_coordinates)


def _path_distance(
    ordered: Sequence[Any],
    start: tuple[float, float] | None,
    end: tuple[float, float] | None,
) -> float:
    """Measure route legs by Haversine distance, including optional endpoints."""
    coordinates = [_coordinates(item) for item in ordered]
    if any(point is None for point in coordinates):
        raise ValueError("Route optimization requires coordinates for every attraction.")
    path = [start] if start is not None else []
    path.extend(point for point in coordinates if point is not None)
    if end is not None:
        path.append(end)
    return sum(
        haversine_distance_km(path[index], path[index + 1])
        for index in range(len(path) - 1)
    )


def optimize_route(
    attractions: Sequence[Any],
    *,
    start: tuple[float, float] | None = None,
    end: tuple[float, float] | None = None,
) -> RouteOptimization:
    """Order located attractions with nearest neighbor followed by 2-opt.

    Without a supplied start, the first unique attraction is the deterministic
    nearest-neighbor seed; this does not imply a hotel or geographic start point.
    Optional start/end are explicit coordinate pairs. The returned distance is
    straight-line Haversine distance, never an actual road route estimate.
    """
    unique = _deduplicate(attractions)
    if not unique:
        return RouteOptimization([], 0.0 if start is not None or end is not None else None)
    located = [item for item in unique if _coordinates(item) is not None]
    if len(located) != len(unique):
        raise ValueError("Pass only attractions with verified coordinates to optimize_route.")
    start_point = _validate_coordinate(*start) if start is not None else None
    end_point = _validate_coordinate(*end) if end is not None else None

    if len(located) == 1:
        return RouteOptimization(located, _path_distance(located, start_point, end_point))

    remaining = list(located)
    if start_point is not None:
        first = min(
            remaining,
            key=lambda item: haversine_distance_km(start_point, _coordinates(item)),  # type: ignore[arg-type]
        )
    else:
        first = remaining[0]
    order = [first]
    remaining.remove(first)
    while remaining:
        current = _coordinates(order[-1])
        next_item = min(
            remaining,
            key=lambda item: haversine_distance_km(current, _coordinates(item)),  # type: ignore[arg-type]
        )
        order.append(next_item)
        remaining.remove(next_item)

    # 2-opt reversals preserve optional endpoints and retain every attraction.
    improved = True
    while improved:
        improved = False
        current_distance = _path_distance(order, start_point, end_point)
        for left in range(0, len(order) - 1):
            for right in range(left + 1, len(order)):
                candidate = [*order[:left], *reversed(order[left : right + 1]), *order[right + 1 :]]
                candidate_distance = _path_distance(candidate, start_point, end_point)
                if candidate_distance + 1e-9 < current_distance:
                    order = candidate
                    improved = True
                    break
            if improved:
                break

    return RouteOptimization(order, _path_distance(order, start_point, end_point))


def assign_attractions_to_days(
    attractions: Sequence[Any],
    duration: int,
    *,
    epsilon_km: float = 5.0,
    min_samples: int = 2,
    max_attractions_per_day: int = 4,
) -> DailyAttractionPlan:
    """Allocate clusters and singletons across days without merging distant clusters.

    Density clusters are treated as geographic groups and split into chunks if
    they exceed the soft daily visit cap. DBSCAN noise points and unlocated
    attractions remain individual candidates, then are distributed by current
    day load. The cap is soft: more days than attractions yields empty day lists,
    while excess attractions are still preserved rather than discarded.
    """
    if duration < 1:
        raise ValueError("duration must be at least one day.")
    if max_attractions_per_day < 1:
        raise ValueError("max_attractions_per_day must be at least one.")

    unique = _deduplicate(attractions)
    clustering = cluster_attractions(
        unique, epsilon_km=epsilon_km, min_samples=min_samples
    )
    groups: list[list[Any]] = []
    for cluster in clustering.clusters:
        route = optimize_route(cluster)
        ordered = route.ordered_attractions
        groups.extend(
            ordered[index : index + max_attractions_per_day]
            for index in range(0, len(ordered), max_attractions_per_day)
        )
    # Keep DBSCAN noise and missing-coordinate locations independent as groups.
    groups.extend([[item] for item in clustering.noise])
    groups.extend([[item] for item in clustering.without_coordinates])
    groups.sort(key=lambda group: (-len(group), min(_stable_name(item) for item in group)))

    days: list[list[Any]] = [[] for _ in range(duration)]
    for group in groups:
        remaining = list(group)
        while remaining:
            available_days = [
                index
                for index in range(duration)
                if len(days[index]) < max_attractions_per_day
            ]
            if available_days:
                target_day = min(available_days, key=lambda index: (len(days[index]), index))
                capacity = max_attractions_per_day - len(days[target_day])
                days[target_day].extend(remaining[:capacity])
                del remaining[:capacity]
            else:
                # The total candidate count exceeds the soft cap across all days.
                # Preserve all candidates and spread unavoidable overflow evenly.
                target_day = min(range(duration), key=lambda index: (len(days[index]), index))
                days[target_day].append(remaining.pop(0))
    return DailyAttractionPlan(days=days, clustering=clustering)


def _stable_name(item: Any) -> str:
    """Return a stable display name for deterministic group sorting."""
    return str(_get_field(item, "name") or "").casefold()
