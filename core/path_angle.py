"""Resolve the Arroyo path angle before evaluating the VALLIS-3C FAS model."""
from __future__ import annotations

from pathlib import Path
from functools import lru_cache
import math
import numpy as np

CU_LAT_DEG = 19.33
CU_LON_DEG = -99.181
CONVENTION = "Arroyo 2024: 0 deg west from CU; positive toward south"
SERIALIZED_FAS_ARCHITECTURE_ID = "vallis3c.fas.release_1_6_0"
CANONICAL_FAS_ARCHITECTURE_ID = "gm_parametric_3c.fas.vallis3c"


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def angle_from_event_coordinates(latitude, longitude):
    """Great-circle initial bearing CU -> source, expressed in Arroyo degrees."""
    lat, lon = _number(latitude), _number(longitude)
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Event latitude/longitude is invalid for the CU path-angle calculation.")
    p1, p2 = math.radians(CU_LAT_DEG), math.radians(lat)
    dl = math.radians(lon - CU_LON_DEG)
    bearing = math.degrees(math.atan2(
        math.sin(dl) * math.cos(p2),
        math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl),
    )) % 360.0
    return (270.0 - bearing) % 360.0


@lru_cache(maxsize=3)
def _spectral_model(variant):
    names = {
        "full": "full",
        "oos2017": "oos_2017",
        "oos2012_2017": "oos_2012_2017",
    }
    if variant not in names:
        raise ValueError(f"Unknown VALLIS-3C variant for path-angle resolution: {variant}")
    artifact = Path(__file__).resolve().parents[1] / "models" / "release" / "variants" / names[variant] / "spectral_model.joblib"
    if not artifact.is_file():
        raise FileNotFoundError(f"VALLIS-3C spectral model is missing: {artifact}")
    from gm_parametric_3c.fas_spectral import FASSpectralModel
    model = FASSpectralModel.load(artifact)
    serialized_id = str(getattr(model, "architecture_id", ""))
    report_id = str((getattr(model, "training_report", {}) or {}).get("architecture_id", ""))
    if serialized_id != SERIALIZED_FAS_ARCHITECTURE_ID or report_id != CANONICAL_FAS_ARCHITECTURE_ID:
        raise RuntimeError(
            "The loaded artifact does not match the VALLIS-3C 1.6.0 spectral architecture."
        )
    return model


def _model_angular_choice(scenario, choice, variant="full"):
    """Query the fitted periodic path term; never use source-type constants."""
    model = _spectral_model(variant)
    source = str(scenario.get("source_type", "")).upper()
    if source == "INTRASLAB":
        source = "INSLAB"
    component = model.cu_models[source]
    cfg = component["kernel"]
    rr = _number(scenario.get("Rrup_km"))
    if rr is None or rr <= 0:
        raise ValueError("Rrup must be positive to evaluate the VALLIS-3C path term.")
    grid = np.linspace(0.0, 150.0, 301)
    angle = np.deg2rad((270.0 - grid) % 360.0)
    trained = np.asarray(component["train_theta_rad"], float)
    aa = 1.0 / float(cfg["l_theta_rad"]) ** 2
    kernel = (np.exp(-aa * (1.0 - np.cos(angle[:, None] - trained[None, :])))
              - float(cfg["periodic_center"])) / float(cfg["periodic_denom"])
    kernel *= float(cfg["var_theta"]) * (rr * np.asarray(component["train_Rrup_km"], float)[None, :] / 300.0**2)
    effect = kernel @ np.asarray(component["amp_alpha"], float)
    index = int(np.argmax(effect)) if choice == "critical" else int(np.argmin(np.abs(effect - np.median(effect))))
    return float(grid[index])


def resolve_source_path_theta(scenario, *, mode="automatic", manual=None, variant="full"):
    """Return θ and provenance; explicit > event coordinates > model-derived choice."""
    explicit = _number(manual)
    if explicit is None:
        explicit = _number(scenario.get("source_path_theta_deg"))
    if explicit is not None:
        theta, origin = explicit, "manual"
    else:
        lat = _number(scenario.get("event_latitude", scenario.get("ev_lat")))
        lon = _number(scenario.get("event_longitude", scenario.get("ev_lon")))
        if lat is not None and lon is not None:
            theta, origin = angle_from_event_coordinates(lat, lon), "automatic"
        elif mode in ("critical", "median"):
            theta, origin = _model_angular_choice(scenario, mode, variant=variant), mode
        else:
            raise ValueError("The event has neither coordinates nor an explicit path angle. Select critical, median, or manual mode before simulation.")
    if not 0.0 <= theta <= 150.0:
        raise ValueError(f"The path angle {theta:.2f}° is outside the supported 0–150° domain.")
    return theta, origin


def apply_source_path_theta(scenario, *, mode="automatic", manual=None, variant="full"):
    result = dict(scenario)
    theta, origin = resolve_source_path_theta(result, mode=mode, manual=manual, variant=variant)
    if manual is None and "source_path_theta_deg" in result:
        prior_origin = str(result.get("source_path_theta_origin", ""))
        if prior_origin in {"automatic", "critical", "median", "manual"}:
            origin = prior_origin
    result.update(source_path_theta_deg=theta, source_path_theta_origin=origin,
                  source_path_theta_convention=CONVENTION,
                  source_azimuth_deg=(270.0 - theta) % 360.0)
    return result
