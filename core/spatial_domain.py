"""VALLIS-3C spatial conditioning with a soft firm-domain constraint.

The semantic regions are deliberately named ``firm_domain`` and
``spatially_variable_domain``.  "Lomas" is only a UI label; the implementation
does not claim that every point outside the reference polygon is the same
geological unit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import MultiPoint, Point, Polygon


WGS84 = "EPSG:4326"
CDMX_UTM = "EPSG:32614"


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-value)))


def _coords(values) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in values]


@dataclass
class SpatialDomain:
    outer_lonlat: list[list[float]]
    inner_lonlat: dict[str, list[list[float]]]
    origin_lonlat: tuple[float, float]
    origin_utm_m: tuple[float, float]
    boundary_transition_km: float = 2.0
    firm_reference_Ts_s: float = 0.5
    Ts_transition_width_s: float = 0.075
    coordinate_system: str = CDMX_UTM
    boundary_version: str = ""
    support_hull_lonlat: list[list[float]] | None = None

    def __post_init__(self):
        self._forward = Transformer.from_crs(WGS84, self.coordinate_system, always_xy=True)
        self._outer_xy = Polygon([self._project_raw(lon, lat) for lon, lat in self.outer_lonlat])
        self._inners_xy = {
            name: Polygon([self._project_raw(lon, lat) for lon, lat in coords])
            for name, coords in self.inner_lonlat.items()
        }
        self._support_xy = None
        if self.support_hull_lonlat and len(self.support_hull_lonlat) >= 3:
            self._support_xy = Polygon(
                [self._project_raw(lon, lat) for lon, lat in self.support_hull_lonlat]
            )

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        boundary_transition_km: float = 2.0,
        firm_reference_Ts_s: float = 0.5,
        Ts_transition_width_s: float = 0.075,
        support_points_lonlat: Iterable[tuple[float, float]] | None = None,
    ) -> "SpatialDomain":
        source = Path(path)
        frame = pd.read_csv(source)
        needed = {"X", "Y", "Group"}
        if not needed.issubset(frame.columns):
            raise KeyError(f"Boundary CSV must contain {sorted(needed)}")
        groups = {
            str(name): _coords(part[["X", "Y"]].to_numpy(dtype=float))
            for name, part in frame.groupby("Group", sort=False)
        }
        if "outer" not in groups:
            raise KeyError("Boundary CSV does not contain Group='outer'")
        inners = {name: coords for name, coords in groups.items() if name != "outer"}
        if not inners:
            raise ValueError("Boundary CSV does not contain inner firm islands")
        transformer = Transformer.from_crs(WGS84, CDMX_UTM, always_xy=True)
        outer_polygon_lonlat = Polygon(groups["outer"])
        centroid = outer_polygon_lonlat.centroid
        origin_lonlat = (float(centroid.x), float(centroid.y))
        origin_utm = transformer.transform(*origin_lonlat)
        support_hull = None
        if support_points_lonlat:
            points = [Point(float(lon), float(lat)) for lon, lat in support_points_lonlat]
            if len(points) >= 3:
                hull = MultiPoint(points).convex_hull
                if isinstance(hull, Polygon):
                    support_hull = _coords(hull.exterior.coords)
        version = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
        return cls(
            outer_lonlat=groups["outer"],
            inner_lonlat=inners,
            origin_lonlat=origin_lonlat,
            origin_utm_m=(float(origin_utm[0]), float(origin_utm[1])),
            boundary_transition_km=float(boundary_transition_km),
            firm_reference_Ts_s=float(firm_reference_Ts_s),
            Ts_transition_width_s=float(Ts_transition_width_s),
            boundary_version=version,
            support_hull_lonlat=support_hull,
        )

    @classmethod
    def from_metadata(cls, metadata: dict) -> "SpatialDomain":
        return cls(
            outer_lonlat=metadata["outer_polygon_lonlat"],
            inner_lonlat=metadata["inner_polygons_lonlat"],
            origin_lonlat=tuple(metadata["origin_lonlat"]),
            origin_utm_m=tuple(metadata["origin_utm_m"]),
            boundary_transition_km=float(metadata["boundary_transition_km"]),
            firm_reference_Ts_s=float(metadata["firm_reference_Ts_s"]),
            Ts_transition_width_s=float(metadata["Ts_transition_width_s"]),
            coordinate_system=str(metadata.get("coordinate_system", CDMX_UTM)),
            boundary_version=str(metadata.get("firm_boundary_version", "")),
            support_hull_lonlat=metadata.get("training_support_hull_lonlat"),
        )

    def _project_raw(self, lon: float, lat: float) -> tuple[float, float]:
        x_m, y_m = self._forward.transform(float(lon), float(lat))
        return float(x_m), float(y_m)

    def local_xy_km(self, lon: float, lat: float) -> tuple[float, float]:
        x_m, y_m = self._project_raw(lon, lat)
        return (
            (x_m - float(self.origin_utm_m[0])) / 1000.0,
            (y_m - float(self.origin_utm_m[1])) / 1000.0,
        )

    def classify_site(self, lon: float, lat: float) -> dict:
        point = Point(*self._project_raw(lon, lat))
        inside_outer = bool(self._outer_xy.covers(point))
        inner_id = next(
            (name for name, polygon in self._inners_xy.items() if polygon.covers(point)),
            None,
        )
        inside_inner = inner_id is not None
        spatial_zone = bool(inside_outer and not inside_inner)
        firm_zone = not spatial_zone
        boundaries = [self._outer_xy.boundary] + [p.boundary for p in self._inners_xy.values()]
        distance_km = min(float(point.distance(boundary)) for boundary in boundaries) / 1000.0
        signed_distance = distance_km if spatial_zone else -distance_km
        return {
            "inside_outer": inside_outer,
            "inside_inner": inside_inner,
            "firm_zone": firm_zone,
            "spatial_zone": spatial_zone,
            "inner_id": inner_id,
            "distance_to_firm_boundary_km": float(signed_distance),
        }

    def conditioned_features(self, lon: float, lat: float, Ts_s: float) -> dict:
        classification = self.classify_site(lon, lat)
        x_km, y_km = self.local_xy_km(lon, lat)
        geometry_gate = _sigmoid(
            classification["distance_to_firm_boundary_km"]
            / max(float(self.boundary_transition_km), 1e-9)
        )
        ts_gate = _sigmoid(
            (float(Ts_s) - float(self.firm_reference_Ts_s))
            / max(float(self.Ts_transition_width_s), 1e-9)
        )
        final_gate = float(geometry_gate * ts_gate)
        return {
            "longitude": float(lon),
            "latitude": float(lat),
            "x_km": float(x_km),
            "y_km": float(y_km),
            **classification,
            "spatial_gate_geometry": float(geometry_gate),
            "spatial_gate_Ts": float(ts_gate),
            "spatial_gate_final": final_gate,
            "x_conditioned_km": float(final_gate * x_km),
            "y_conditioned_km": float(final_gate * y_km),
        }

    def spatial_block_id(self, lon: float, lat: float, block_km: float) -> str:
        x_km, y_km = self.local_xy_km(lon, lat)
        size = max(float(block_km), 1e-6)
        return f"{int(np.floor(x_km / size))}:{int(np.floor(y_km / size))}"

    def distance_outside_training_support_km(self, lon: float, lat: float) -> float:
        if self._support_xy is None:
            return 0.0
        point = Point(*self._project_raw(lon, lat))
        return float(point.distance(self._support_xy)) / 1000.0

    def to_metadata(self) -> dict:
        return {
            "enabled": True,
            "coordinate_system": self.coordinate_system,
            "origin_lonlat": list(self.origin_lonlat),
            "origin_utm_m": list(self.origin_utm_m),
            "firm_boundary_version": self.boundary_version,
            "outer_polygon_lonlat": self.outer_lonlat,
            "inner_polygons_lonlat": self.inner_lonlat,
            "firm_reference_Ts_s": float(self.firm_reference_Ts_s),
            "Ts_transition_width_s": float(self.Ts_transition_width_s),
            "boundary_transition_km": float(self.boundary_transition_km),
            "training_support_hull_lonlat": self.support_hull_lonlat,
            "feature_definition": [
                "Mw", "Rrup_km", "depth", "Ts_s",
                "spatial_gate_final*x_km", "spatial_gate_final*y_km",
            ],
        }
