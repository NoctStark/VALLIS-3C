"""Export utilities for VALLIS-3C.


Author: Joel D. Cruz-Arguelles
Instituto de Ingeniería, Universidad Nacional Autónoma de México
2026
"""
from __future__ import annotations


from pathlib import Path
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import sys
import numpy as np

from core.spectra import FAS_SMOOTHING_OUTPUT_POINTS


AUTHOR = {
    "name": "Joel D. Cruz-Arguelles",
    "affiliation": "Instituto de Ingeniería, Universidad Nacional Autónoma de México, Coyoacán, 04510 Ciudad de México, México",
    "emails": ["JCruzAr@iingen.unam.mx", "joeldan.cruz@gmail.com"],
    "orcid": "0009-0002-6878-0173",
    "year": 2026,
}
ROOT = Path(__file__).resolve().parents[1]
SOFTWARE_VERSION = "1.6.0"
BUILD_ID = SOFTWARE_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _artifact_hashes(variant: str) -> dict:
    directory = {
        "full": "full",
        "oos2017": "oos_2017",
        "oos2012_2017": "oos_2012_2017",
    }.get(variant, "full")
    base = ROOT / "models" / "release" / "variants" / directory
    artifacts = {
        "spectral_model": base / "spectral_model.joblib",
        "duration_clock_final_hv": base / "temporal" / "duration_clock_final_hv_portable.npz",
        "conditional_selector": base / "temporal" / "selector_portable.npz",
        "vertical_clock_bank": base / "temporal" / "hf_vertical_clock_exactbank_5_15.csv",
    }
    return {name: _sha256(path) for name, path in artifacts.items() if path.is_file()}


def _release_metadata() -> dict:
    texture_manifest = ROOT / "models" / "release" / "textures" / "texture_manifest.json"
    manifest = ROOT / "SOFTWARE_MANIFEST.json"
    snapshot = json.loads(texture_manifest.read_text(encoding="utf-8-sig")) if texture_manifest.is_file() else {}
    return {
        "software_version": SOFTWARE_VERSION,
        "release": f"VALLIS-3C {SOFTWARE_VERSION}",
        "build": BUILD_ID,
        "license": "GPL-3.0-only",
        "copyright": "Copyright (C) 2026 Joel D. Cruz-Arguelles",
        "data_provenance": "DATA_PROVENANCE.md",
        "scientific_disclaimer": "DISCLAIMER.md",
        "release_hash": _sha256(manifest) if manifest.is_file() else None,
        "dataset_snapshot": {
            "created_at": snapshot.get("created_at_utc"),
            "representation": "vallis3c.phase_diffusion.texture_carrier.v1",
        },
        "dataset_hash": _sha256(texture_manifest) if texture_manifest.is_file() else None,
}


def _jsonable(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def _write_long_table(path: Path, header: str, rows, delimiter=","):
    with path.open("w", encoding="utf-8") as h:
        h.write(header.rstrip() + "\n")
        for row in rows:
            h.write(delimiter.join(str(x) for x in row) + "\n")


def _scenario_metadata(scenario: dict):
    src = dict(scenario or {})
    mw = float(src.get("Mw", 0.0))
    rrup = float(src.get("Rrup_km", 0.0))
    source_type = str(src.get("source_type", "")).upper()
    event_id = str(src.get("event_id", "")).strip()
    if not event_id or event_id == "VALLIS3C_SCENARIO":
        event_id = f"Mw{mw:.2f} - Rrup{rrup:.1f} - {source_type}"
    out = {
        "event_id": event_id,
        "Mw": mw,
        "Rrup_km": rrup,
        "depth_used_km": float(src.get("depth_used_km", 0.0)),
        "Ts_s": float(src.get("Ts_s", 0.0)),
        "source_type": source_type,
        "model_variant": str(src.get("model_variant", "full")),
        "x_conditioned_km": float(src.get("x_conditioned_km", 0.0)),
        "y_conditioned_km": float(src.get("y_conditioned_km", 0.0)),
        "source_path_theta_deg": float(src.get("source_path_theta_deg", 0.0)),
        "source_path_theta_origin": str(src.get("source_path_theta_origin", "")),
        "source_path_theta_convention": str(src.get("source_path_theta_convention", "0 deg west from CU; positive toward south")),
        "source_azimuth_deg": float(src.get("source_azimuth_deg", 0.0)),
        "site_id": str(src.get("site_id", "Site_001")),
        "site_longitude": float(src.get("site_longitude", 0.0)),
        "site_latitude": float(src.get("site_latitude", 0.0)),
        "site_zone": str(src.get("site_zone", "unresolved")),
    }
    station_id = str(src.get("station_id", "")).strip()
    if station_id:
        out["station_id"] = station_id
    return out


def export_family(base_dir: Path, payload: dict, fas, rs, settings, scenario: dict, labels, rotation_angles, *, site_id="Site_001", rotated=False):
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    acc = np.asarray(payload["accelerations"], float)
    time_s = np.asarray(payload.get("time_s"), float)
    dt = float(payload["dt_s"])
    valid_npts = np.asarray(payload.get("valid_npts", np.full(len(acc), acc.shape[-1])), dtype=int)


    identity = _release_metadata()
    run_context = dict(payload.get("run_context") or {})
    output_conditioning = dict(payload.get("output_conditioning") or {})
    executed_taper = bool(output_conditioning.get("join_taper", True))
    executed_filter = bool(output_conditioning.get("zero_phase_butterworth", settings.apply_filter))
    executed_baseline_mode = int(output_conditioning.get("baseline_correction_mode", 0))
    executed_stabilization = bool(output_conditioning.get("terminal_displacement_stabilization", True))
    executed_padding = output_conditioning.get("zero_padding_s", [settings.pad_start_s, settings.pad_end_s])
    if not isinstance(executed_padding, (list, tuple)) or len(executed_padding) != 2:
        executed_padding = [settings.pad_start_s, settings.pad_end_s]
    scenario_meta = _scenario_metadata(scenario)
    variant = str(scenario_meta.get("model_variant", "full"))
    environment = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "scikit_learn_version": _package_version("scikit-learn"),
        "pandas_version": _package_version("pandas"),
        "matplotlib_version": _package_version("matplotlib"),
        "PyQt6_version": _package_version("PyQt6"),
        "joblib_version": _package_version("joblib"),
        "h5py_version": _package_version("h5py"),
        "pyproj_version": _package_version("pyproj"),
        "shapely_version": _package_version("shapely"),
    }
    metadata = {
        "software_name": "VALLIS-3C",
        **identity,
        "developed_by": AUTHOR["name"],
        "affiliation": AUTHOR["affiliation"],
        "emails": list(AUTHOR["emails"]),
        "year": int(AUTHOR["year"]),
        "orcid": AUTHOR["orcid"],
        "acceleration_units": "cm/s2",
        "response_spectrum_units": "cm/s2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dt_s": dt,
        "site_id": str(site_id or "Site_001"),
        "scenario": scenario_meta,
        "model_variant": variant,
        "model_artifact_hashes": _artifact_hashes(variant),
        "n_realizations": int(run_context.get("n_realizations", len(acc))),
        "seed": run_context.get("seed"),
        "shared_event_seed": run_context.get("shared_event_seed"),
        "event_residual_mode": run_context.get("event_residual_mode"),
        "carrier": "phase_diffusion",
        "padding": {"start_s": float(executed_padding[0]), "end_s": float(executed_padding[1])},
        "taper": {"enabled": executed_taper, "type": "cosine_edges" if executed_taper else "none"},
        "filter": {
            "enabled": executed_filter,
            "type": "zero_phase_butterworth_bandpass" if executed_filter else "none",
            "order": output_conditioning.get("filter_order") if executed_filter else None,
            "frequencies_hz": output_conditioning.get("bandpass_hz") if executed_filter else None,
        },
        "baseline_correction": {"mode": executed_baseline_mode, "name": "none" if executed_baseline_mode == 0 else "polynomial"},
        "displacement_stabilization": {
            "enabled": executed_stabilization,
            "method": "compact_terminal_velocity_and_displacement_closure",
            "tail_s": output_conditioning.get("terminal_displacement_stabilization_tail_s"),
        },
        "response_spectrum": {
            "algorithm": "Generalized Single-Step Single-Solve (GSSSS)",
            "reference_doi": "10.1002/nme.873",
            "damping_ratio": float(settings.rs_damping),
            "Tmin_s": float(settings.rs_tmin_s),
            "Tmax_s": float(settings.rs_tmax_s),
            "N": int(settings.rs_points),
            "spacing": str(settings.rs_spacing),
        },
        "FAS_range_hz": [float(settings.fas_fmin_hz), float(settings.fas_fmax_hz)],
        "FAS_smoothing": {
            "enabled": bool(settings.fas_smoothing),
            "method": "Konno-Ohmachi" if settings.fas_smoothing else "none",
            "bandwidth": float(settings.fas_smoothing_bandwidth) if settings.fas_smoothing else None,
            "output_points": FAS_SMOOTHING_OUTPUT_POINTS if settings.fas_smoothing else None,
            "output_frequency_spacing": "logarithmic" if settings.fas_smoothing else "native_fft",
            "source_ordinates": "all_native_FFT_ordinates_in_configured_band",
            "reference_doi": "10.1785/bssa0880010228" if settings.fas_smoothing else None,
        },
        "rotation_mode": str(settings.rotation_mode) if rotated else "native_axes",
        "rotation_angle_deg": float(settings.rotation_angle_deg) if rotated and str(settings.rotation_mode) == "manual" else None,
        "rotation_angles_deg": _jsonable(rotation_angles),
        "calibration_domain": _jsonable(payload.get("calibration_domain") or {}),
        "environment": environment,
        "component_labels": list(labels),
    }
    (base_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


    if settings.save_accelerograms:
        ncomp = int(acc.shape[1])
        component_count_tag = f"{ncomp}C"
        fmt = str(settings.acceleration_format)
        for i, rec_full in enumerate(acc, start=1):
            n_valid = int(np.clip(valid_npts[i-1] if i-1 < len(valid_npts) else rec_full.shape[-1], 2, rec_full.shape[-1]))
            rec = rec_full[:, :n_valid]
            stem = f"simulation_{i:04d}"
            if fmt == "accel":
                np.savetxt(
                    base_dir / f"{stem}_{component_count_tag}.accel", rec.T, fmt="%.9e",
                    header=f"acceleration_cm_s2; components={','.join(labels)}; dt_s = {dt:.9g}",
                )
            elif fmt == "accel_separate":
                for j, lbl in enumerate(labels):
                    np.savetxt(
                        base_dir / f"{stem}_{lbl}.accel", rec[j], fmt="%.9e",
                        header=f"acceleration_cm_s2; component={lbl}; dt_s = {dt:.9g}",
                    )
        if fmt == "npz":
            np.savez_compressed(
                base_dir / f"accelerograms_{component_count_tag}.npz",
                accelerations=acc,
                dt_s=dt,
                valid_npts=valid_npts,
                rotation_angles_deg=np.asarray(rotation_angles, float),
                component_labels=np.asarray(labels, dtype="U32"),
                metadata_json=json.dumps(metadata, ensure_ascii=False),
            )


    if settings.save_fas:
        for i, rec in enumerate(fas, start=1):
            stem = base_dir / f"simulation_{i:04d}_FAS"
            rows = []
            for j, lbl in enumerate(labels):
                f, amp = rec[j]
                rows.extend((lbl, f"{x:.9g}", f"{y:.9e}") for x, y in zip(f, amp))
            if settings.fas_format == "csv":
                _write_long_table(stem.with_suffix(".csv"), "component,frequency_hz,fas_cm_per_s", rows)
            if settings.fas_format == "npz":
                np.savez_compressed(
                    stem.with_suffix(".npz"),
                    **{f"f_{labels[j]}": np.asarray(rec[j][0], float) for j in range(len(labels))},
                    **{f"FAS_{labels[j]}": np.asarray(rec[j][1], float) for j in range(len(labels))},
                )


    if settings.save_response_spectra:
        for i, rec in enumerate(rs, start=1):
            stem = base_dir / f"simulation_{i:04d}_response_spectrum"
            rows = []
            for j, lbl in enumerate(labels):
                t, sa = rec[j]
                rows.extend((lbl, f"{x:.9g}", f"{y:.9e}") for x, y in zip(t, sa))
            if settings.response_format == "csv":
                _write_long_table(stem.with_suffix(".csv"), "component,period_s,Sa_cm_per_s2", rows)
            if settings.response_format == "npz":
                np.savez_compressed(
                    stem.with_suffix(".npz"),
                    **{f"T_{labels[j]}": np.asarray(rec[j][0], float) for j in range(len(labels))},
                    **{f"Sa_{labels[j]}": np.asarray(rec[j][1], float) for j in range(len(labels))},
                )
