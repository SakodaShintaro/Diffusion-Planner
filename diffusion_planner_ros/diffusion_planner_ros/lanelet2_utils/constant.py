from __future__ import annotations

from .map import MapType

MAP_TYPE_MAPPING: dict[str, MapType] = {
    "road": MapType.ROADWAY,
    "highway": MapType.ROADWAY,
    "road_shoulder": MapType.ROADWAY,
    "bicycle_lane": MapType.BIKE_LANE,
    "dashed": MapType.DASHED,
    "solid": MapType.SOLID,
    "dashed_dashed": MapType.DOUBLE_DASH,
    "virtual": MapType.UNKNOWN,
    "road_border": MapType.SOLID,
    "crosswalk": MapType.CROSSWALK,
    "unknown": MapType.UNKNOWN,
}

T4_LANE: tuple[str, ...] = ("road", "highway", "road_shoulder", "bicycle_lane")
T4_ROADLINE: tuple[str, ...] = ("dashed", "solid", "dashed_dashed", "virtual")
T4_ROADEDGE: tuple[str, ...] = ("road_border",)
