"""Calibration-domain checks for VALLIS-3C scenarios."""
from __future__ import annotations

from pathlib import Path
import json
import math


ROOT = Path(__file__).resolve().parents[1]
DOMAIN_FILE = ROOT / "config" / "calibration_domain.json"
SPATIAL_FILE = ROOT / "config" / "spatial_conditioning.json"


def _finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _point_in_polygon(x: float, y: float, polygon) -> bool:
    points = [(float(p[0]), float(p[1])) for p in polygon]
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (xi, yi) in enumerate(points):
        xj, yj = points[j]
        on_segment = abs((y - yi) * (xj - xi) - (x - xi) * (yj - yi)) <= 1e-10
        if on_segment and min(xi, xj) - 1e-10 <= x <= max(xi, xj) + 1e-10 and min(yi, yj) - 1e-10 <= y <= max(yi, yj) + 1e-10:
            return True
        crosses = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-30) + xi)
        if crosses:
            inside = not inside
        j = i
    return inside


def calibration_definition() -> dict:
    domain = json.loads(DOMAIN_FILE.read_text(encoding="utf-8-sig"))
    spatial = json.loads(SPATIAL_FILE.read_text(encoding="utf-8-sig"))
    domain["training_support_hull_lonlat"] = spatial.get("training_support_hull_lonlat", [])
    domain["source"] = {
        "ranges_from": "models/release/runtime/descriptor_metadata.json",
        "spatial_hull_from": "config/spatial_conditioning.json",
    }
    return domain


def evaluate_calibration_domain(values: dict) -> dict:
    definition = calibration_definition()
    violations = []
    checks = {}
    mapping = {
        "Mw": "Mw",
        "Rrup_km": "Rrup_km",
        "depth_used_km": "depth_used_km",
        "Ts_s": "Ts_s",
    }
    for key, value_key in mapping.items():
        value = _finite(values.get(value_key))
        lo, hi = map(float, definition["ranges"][key])
        ok = value is not None and lo <= value <= hi
        checks[key] = {"value": value, "range": [lo, hi], "in_support": ok}
        if not ok:
            shown = "missing" if value is None else f"{value:g}"
            violations.append(f"{key}={shown} is outside [{lo:g}, {hi:g}]")

    source_type = str(values.get("source_type", "")).upper()
    allowed_sources = list(definition.get("source_types", []))
    source_ok = source_type in allowed_sources
    checks["source_type"] = {"value": source_type, "allowed": allowed_sources, "in_support": source_ok}
    if not source_ok:
        violations.append(f"source_type={source_type or 'missing'} is not in {allowed_sources}")

    theta = _finite(values.get("theta_manual")) if str(values.get("theta_mode", "")) == "manual" else None
    if theta is not None:
        lo, hi = map(float, definition["ranges"]["source_path_theta_deg"])
        ok = lo <= theta <= hi
        checks["source_path_theta_deg"] = {"value": theta, "range": [lo, hi], "in_support": ok}
        if not ok:
            violations.append(f"theta={theta:g} deg is outside [{lo:g}, {hi:g}]")

    lon = _finite(values.get("site_longitude"))
    lat = _finite(values.get("site_latitude"))
    hull = definition.get("training_support_hull_lonlat", [])
    spatial_ok = lon is not None and lat is not None and _point_in_polygon(lon, lat, hull)
    checks["site_location"] = {"longitude": lon, "latitude": lat, "in_training_hull": spatial_ok}
    if not spatial_ok:
        violations.append("site longitude/latitude is outside the training support hull")

    if str(values.get("theta_mode", "")) == "automatic":
        for key, value_key in (("event_longitude", "event_longitude"), ("event_latitude", "event_latitude")):
            value = _finite(values.get(value_key))
            lo, hi = map(float, definition["ranges"][key])
            ok = value is not None and lo <= value <= hi
            checks[key] = {"value": value, "range": [lo, hi], "in_support": ok}
            if not ok:
                shown = "missing" if value is None else f"{value:g}"
                violations.append(f"{key}={shown} is outside [{lo:g}, {hi:g}]")

    return {
        "status": "IN SUPPORT" if not violations else "EXTRAPOLATION",
        "in_support": not violations,
        "violations": violations,
        "checks": checks,
        "definition_version": str(definition.get("version", "1.6.0")),
    }
