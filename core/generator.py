from __future__ import annotations
from pathlib import Path
from .runtime_context import DEFAULT_MODEL_DIR, inspect_parametric_3c_model, spatial_features
from runtime.vallis3c_runtime.adapter import simulate_vallis3c_sync
from .path_angle import apply_source_path_theta
from runtime.vallis3c_runtime.config import VallisConfig, fas_preset
from sim.signal_ops import apply_baseline_correction, bandpass_butter, stabilize_terminal_displacement
import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def build_scenario(values: dict):
    # The packaged VALLIS-3C runtime supplies the phase-diffusion carrier,
    # spatial conditioning, and fitted FAS/selector.
    info=inspect_parametric_3c_model(DEFAULT_MODEL_DIR)
    spatial=spatial_features(info, values["site_longitude"], values["site_latitude"], values["Ts_s"])
    mw = float(values["Mw"])
    rrup = float(values["Rrup_km"])
    source_type = str(values["source_type"]).upper()
    scenario={
        "event_id":f"Mw{mw:.2f} - Rrup{rrup:.1f} - {source_type}",
        "Mw":mw, "Rrup_km":rrup,
        "depth_used_km":float(values["depth_used_km"]), "Ts_s":float(values["Ts_s"]),
        "source_type":source_type,
        "model_variant":"full",
        "site_id":str(values.get("site_id") or "Site_001"),
        "site_longitude":float(values["site_longitude"]),
        "site_latitude":float(values["site_latitude"]),
        "site_zone":str(values.get("site_zone") or "unresolved"),
        "x_conditioned_km":float(spatial["x_conditioned_km"]),
        "y_conditioned_km":float(spatial["y_conditioned_km"]),
    }
    station_id = str(values.get("station_id", "")).strip()
    if station_id:
        scenario["station_id"] = station_id
    if values.get("event_latitude") is not None: scenario["event_latitude"]=float(values["event_latitude"])
    if values.get("event_longitude") is not None: scenario["event_longitude"]=float(values["event_longitude"])
    manual=values.get("theta_manual")
    scenario=apply_source_path_theta(scenario, mode=values.get("theta_mode","automatic"), manual=manual)
    return scenario, spatial, info


def _condition_payload(payload: dict, values: dict) -> dict:
    """Apply the shared Simulation-tab output conditioning without Arias trimming.


    Native support comes from each realization's ``unpadded_npts`` metadata.
    Realizations are conditioned individually and only rectangularized after the
    user-requested leading/trailing padding has been applied.
    """
    out = dict(payload)
    acc = np.asarray(payload["accelerations"], dtype=float)
    dt = float(payload["dt_s"])
    meta = [dict(m) if isinstance(m, dict) else {} for m in (payload.get("metadata") or [])]
    pad_start_s = max(0.0, float(values.get("pad_start_s", 20.0)))
    pad_end_s = max(0.0, float(values.get("pad_end_s", 20.0)))
    pad_start_n = int(round(pad_start_s / dt))
    pad_end_n = int(round(pad_end_s / dt))
    use_taper = True
    use_filter = bool(values.get("apply_filter", True))
    baseline_mode = 0
    fmin = float(values.get("filter_low_hz", 0.10))
    fmax = float(values.get("filter_high_hz", 25.0))
    order = int(values.get("filter_order", 4))
    # Terminal velocity/displacement stabilization is an internal, always-on
    # production operation. It is intentionally not exposed as a UI control.
    stabilize = True
    stabilization_tail_s = max(1.0, float(pad_end_s or 20.0))
    Ts = max(0.05, float(values.get("Ts_s", 1.0)))


    runs = []
    valid_npts = []
    all_diag = []
    for r in range(acc.shape[0]):
        md = meta[r] if r < len(meta) else {}
        native_n = int(md.get("unpadded_npts", acc.shape[-1]))
        native_n = int(np.clip(native_n, 16, acc.shape[-1]))
        active = np.asarray(acc[r, :, :native_n], dtype=float).copy()
        run = np.pad(active, ((0, 0), (pad_start_n, pad_end_n)), mode="constant")


        if use_taper and native_n >= 8:
            max_edge = max(2, native_n // 8)
            n_start = min(int(round(float(np.clip(3.0 * Ts, 3.0, 12.0)) / dt)), max_edge, native_n // 4)
            n_end = min(int(round(float(np.clip(4.0 * Ts, 4.0, 16.0)) / dt)), max_edge, native_n // 4)
            n_start = max(2, n_start); n_end = max(2, n_end)
            sr = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_start)))
            er = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_end)))
            run[:, pad_start_n:pad_start_n+n_start] *= sr[None, :]
            active_end = pad_start_n + native_n
            run[:, active_end-n_end:active_end] *= er[::-1][None, :]


        active_end = pad_start_n + native_n
        comp_diag = []
        for j in range(3):
            y = apply_baseline_correction(run[j], dt, baseline_mode)
            if use_filter:
                y = bandpass_butter(y, fs=1.0/dt, fmin=fmin, fmax=fmax, order=order, zero_phase=True)
            if stabilize:
                y, diag = stabilize_terminal_displacement(
                    y, dt, tail_duration_s=stabilization_tail_s, return_diagnostics=True,
                    target_tail_displacement_cm=0.0, active_start_index=pad_start_n,
                    active_end_index_exclusive=active_end,
                )
                comp_diag.append(diag)
            run[j] = np.asarray(y, dtype=float)
        runs.append(run.astype(np.float32))
        valid_npts.append(int(run.shape[-1]))
        all_diag.append(comp_diag)
        md.update({
            "conditioned_npts": int(run.shape[-1]),
            "native_npts": int(native_n),
            "zero_padding_start_s": float(pad_start_s),
            "zero_padding_end_s": float(pad_end_s),
            "output_conditioning_no_arias_trim": True,
        })
        if r < len(meta): meta[r] = md
        else: meta.append(md)


    nmax = max(valid_npts) if valid_npts else acc.shape[-1]
    rect = np.zeros((len(runs), 3, nmax), dtype=np.float32)
    for r, run in enumerate(runs):
        rect[r, :, :run.shape[-1]] = run
    out["accelerations"] = rect
    out["time_s"] = np.arange(nmax, dtype=float) * dt
    out["valid_npts"] = np.asarray(valid_npts, dtype=int)
    out["metadata"] = meta
    out["units"] = "cm/s2"
    out["output_conditioning"] = {
        "support_source": "VALLIS-3C metadata unpadded_npts",
        "arias_trim_used": False,
        "zero_padding_s": [float(pad_start_s), float(pad_end_s)],
        "join_taper": bool(use_taper),
        "baseline_correction_mode": int(baseline_mode),
        "zero_phase_butterworth": bool(use_filter),
        "bandpass_hz": [float(fmin), float(fmax)] if use_filter else None,
        "filter_order": int(order) if use_filter else None,
        "terminal_displacement_stabilization": bool(stabilize),
        "terminal_displacement_stabilization_tail_s": float(stabilization_tail_s) if stabilize else None,
        "terminal_state_diagnostics": all_diag,
    }
    return out


def _select_output_components(payload: dict, n_components: int) -> dict:
    """Select the requested 2C/3C output after three-component synthesis.


    The model remains a three-component synthesis internally; the user-facing output
    may retain only the two horizontal components.
    """
    n_components = 2 if int(n_components) == 2 else 3
    out = dict(payload)
    acc = np.asarray(payload["accelerations"])
    out["accelerations"] = acc[:, :n_components, :]
    out["component_names"] = ("Major", "Intermediate") if n_components == 2 else ("Major", "Intermediate", "Vertical")
    out["output_components"] = n_components
    for key in ("target_fas", "requested_target_fas"):
        if key in out:
            arr = np.asarray(out[key])
            if arr.ndim >= 2 and arr.shape[-2] >= n_components:
                sl = [slice(None)] * arr.ndim
                sl[-2] = slice(0, n_components)
                out[key] = arr[tuple(sl)]
    return out


def generate(values: dict, progress=None, cancel=None):
    scenario, spatial, info = build_scenario(values)
    output_dt = float(values.get("output_dt_s", 0.01))
    cfg=VallisConfig.from_dict({"model_variant":"full", "architecture":"final_hv", "exclude_same_event":False, "strict_oos":False, "dt_s":output_dt}).validate()
    fas_cfg=fas_preset()
    persisted_fas=dict(values.get("fas_options") or {})
    # Production exposes only event semantics here. Calibrated FAS/site constants
    # are reapplied by the adapter/provider and cannot be overridden by settings.
    if "fas_event_mode" in persisted_fas:
        fas_cfg["fas_event_mode"]=persisted_fas["fas_event_mode"]
    if persisted_fas.get("fas_event_seed") is not None:
        fas_cfg["fas_event_seed"]=int(persisted_fas["fas_event_seed"])
    request={
        "scenario":scenario, "n_realizations":int(values["n_realizations"]), "seed":int(values["seed"]),
        "method":"vallis3c", "generator_family":"vallis3c_1_6_0",
        "vallis_config":cfg.to_dict(), "fas_options":fas_cfg,
        "fas_input_mode":"model",
        "dt_s":float(cfg.dt_s), "spatial_diagnostics":spatial,
        "defer_output_postprocess": True,
    }
    payload=simulate_vallis3c_sync(DEFAULT_MODEL_DIR, request, progress=progress, cancel=cancel)
    payload=_condition_payload(payload, values)
    payload=_select_output_components(payload, int(values.get("output_components", 3)))
    payload["calibration_domain"] = dict(values.get("calibration_domain") or {})
    payload["run_context"] = {
        "n_realizations": int(values["n_realizations"]),
        "seed": int(values["seed"]),
        "shared_event_seed": values.get("fas_options", {}).get("fas_event_seed"),
        "event_residual_mode": values.get("fas_options", {}).get("fas_event_mode", "independent_realization"),
        "carrier": "phase_diffusion",
    }
    return payload, scenario, spatial
