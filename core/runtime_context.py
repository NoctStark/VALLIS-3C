"""Runtime context for VALLIS-3C 1.6.0."""
from __future__ import annotations
from pathlib import Path
import json

from .spatial_domain import SpatialDomain

ROOT=Path(__file__).resolve().parents[1]
MODELS_ROOT=ROOT/"models"
RELEASE_ROOT=MODELS_ROOT/"release"
DEFAULT_MODEL_DIR=RELEASE_ROOT/"runtime"
TEXTURES_DIR=RELEASE_ROOT/"textures"

def resolve_model_dir(path=None):
    selected=Path(path).resolve() if path else DEFAULT_MODEL_DIR.resolve()
    if selected != DEFAULT_MODEL_DIR.resolve():
        raise ValueError("VALLIS-3C 1.6.0 uses only the packaged release runtime root.")
    if not (selected/"runtime_manifest.json").is_file():
        raise FileNotFoundError(f"VALLIS-3C runtime manifest missing: {selected}")
    return selected

def _spatial_conditioning():
    p=ROOT/"config"/"spatial_conditioning.json"
    if not p.is_file():
        raise FileNotFoundError(f"Spatial conditioning definition missing: {p}")
    d=dict(json.loads(p.read_text(encoding="utf-8-sig")) or {})
    if not d.get("origin_lonlat") or not d.get("coordinate_system"):
        raise ValueError("Invalid spatial conditioning definition")
    d["metadata_source"]=str(p)
    return d

def inspect_parametric_3c_model(path=None,**_):
    model_dir=resolve_model_dir(path)
    manifest_path=model_dir/"runtime_manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    return {
        "model_dir":str(model_dir),
        "product_version":"1.6.0",
        "engine_generation":"1.6.0",
        "engine_module":"VALLIS-3C runtime",
        "bundle_variant":"full",
        "production_bundle":True,
        "architecture":"VALLIS-3C 1.6.0 · 3C complex transport · unified H/V duration model",
        "active_fas_architecture":"vallis3c_release",
        "components":["major","intermediate","vertical"],
        "condition_mode":"xy",
        "predictor_names":["Mw","Rrup_km","depth_used_km","Ts_s","source_type","x_conditioned_km","y_conditioned_km"],
        "spatial_conditioning":_spatial_conditioning(),
        "target_dt_s":0.01,
        "runtime_manifest":manifest,
    }

def spatial_features(info:dict,longitude:float,latitude:float,ts_s:float)->dict:
    metadata=dict(info.get("spatial_conditioning") or _spatial_conditioning())
    domain=SpatialDomain.from_metadata(metadata)
    diagnostics=domain.conditioned_features(float(longitude),float(latitude),float(ts_s))
    diagnostics["distance_outside_training_support_km"]=domain.distance_outside_training_support_km(float(longitude),float(latitude))
    return diagnostics
