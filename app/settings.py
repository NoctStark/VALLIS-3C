"""Persistent UI settings for VALLIS-3C.


Author: Joel D. Cruz-Arguelles
Instituto de Ingeniería, Universidad Nacional Autónoma de México
2026
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class AppSettings:
    output_components: int = 3
    fas_fmin_hz: float = 0.10
    fas_fmax_hz: float = 20.0
    fas_smoothing: bool = True
    fas_smoothing_bandwidth: float = 40.0
    save_accelerograms: bool = True
    save_response_spectra: bool = True
    save_fas: bool = True
    acceleration_format: str = "accel"
    response_format: str = "csv"
    fas_format: str = "csv"
    output_dt_s: float = 0.01
    rs_damping: float = 0.05
    rs_tmin_s: float = 0.01
    rs_tmax_s: float = 10.0
    rs_points: int = 100
    rs_spacing: str = "log"
    rs_plot_linear: bool = False
    spectral_display_mode: str = "percentiles"
    spectral_show_p50: bool = True
    spectral_show_p16_p84: bool = True
    spectral_show_p05_p95: bool = True
    spectral_overlay_selected: bool = True
    rs_percentile_basis: str = "components"
    pad_start_s: float = 20.0
    pad_end_s: float = 20.0
    apply_taper: bool = True
    apply_filter: bool = True
    filter_low_hz: float = 0.10
    filter_high_hz: float = 25.0
    filter_order: int = 4
    baseline_correction_mode: int = 0
    displacement_stabilization: bool = True
    rotation_mode: str = "none"
    rotation_angle_deg: float = 0.0
    fas_event_mode: str = "independent_realization"
    output_directory: str = "outputs"


    @classmethod
    def load(cls, path: Path, defaults_path: Path | None = None) -> "AppSettings":
        """Load packaged defaults, then overlay user-specific settings if present."""
        raw = {}
        if defaults_path is not None and Path(defaults_path).is_file():
            try:
                defaults = json.loads(Path(defaults_path).read_text(encoding="utf-8-sig"))
                if isinstance(defaults, dict):
                    raw.update(defaults)
            except (OSError, json.JSONDecodeError):
                pass
        if Path(path).is_file():
            try:
                user = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                if isinstance(user, dict):
                    raw.update(user)
            except (OSError, json.JSONDecodeError):
                pass
        try:
            allowed = set(cls.__dataclass_fields__)
            obj = cls(**{k: v for k, v in raw.items() if k in allowed})


            obj.output_components = 2 if int(getattr(obj, "output_components", 3)) == 2 else 3
            obj.acceleration_format = obj.acceleration_format if obj.acceleration_format in {"accel", "accel_separate", "npz"} else "accel"
            obj.fas_format = obj.fas_format if obj.fas_format in {"csv", "npz"} else "csv"
            obj.response_format = obj.response_format if obj.response_format in {"csv", "npz"} else "csv"
            obj.fas_event_mode = obj.fas_event_mode if obj.fas_event_mode in {"record_only", "shared_batch", "independent_realization"} else "independent_realization"
            obj.spectral_display_mode = obj.spectral_display_mode if obj.spectral_display_mode in {"percentiles", "individual"} else "percentiles"
            obj.rs_percentile_basis = obj.rs_percentile_basis if obj.rs_percentile_basis in {"components", "rms"} else "components"


            # Hidden production defaults.
            obj.apply_taper = True
            obj.baseline_correction_mode = 0
            obj.displacement_stabilization = True
            if not (0.001 <= float(obj.output_dt_s) < 0.025):
                obj.output_dt_s = 0.01
            return obj
        except Exception:
            return cls()


    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
