from __future__ import annotations
import numpy as np
from sim.response_spectrum import response_spectrum
try:
    from sim.spectra_ops import konno_ohmachi_smooth
except Exception:
    konno_ohmachi_smooth = None

FAS_SMOOTHING_OUTPUT_POINTS = 320


def fas_record(acc_components_cm_s2, dt_s, fmin=0.1, fmax=20.0,
               smooth=True, bandwidth=40.0):
    """Return component FAS, sharing one FFT grid and smoothing operation."""
    a = np.asarray(acc_components_cm_s2, dtype=float)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("Acceleration must contain one or more component rows.")
    dt = float(dt_s)
    f = np.fft.rfftfreq(a.shape[1], d=dt)
    amp = np.abs(np.fft.rfft(a, axis=1)) * dt
    mask = (f >= float(fmin)) & (f <= float(fmax)) & (f > 0)
    f = np.asarray(f[mask], dtype=float)
    amp = np.asarray(amp[:, mask], dtype=float)
    if smooth and konno_ohmachi_smooth is not None and f.size > 4:
        try:
            f, amp = konno_ohmachi_smooth(
                f,
                amp,
                bandwidth=float(bandwidth),
                fmin=float(fmin),
                fmax=float(fmax),
                output_points=FAS_SMOOTHING_OUTPUT_POINTS,
            )
        except Exception:
            pass
    return [(np.asarray(f, float), np.asarray(row, float)) for row in amp]


def fas_one(acc_cm_s2, dt_s, fmin=0.1, fmax=20.0, smooth=True, bandwidth=40.0):
    return fas_record(
        np.asarray(acc_cm_s2, dtype=float).ravel(), dt_s, fmin, fmax, smooth, bandwidth
    )[0]


def response_one(acc_cm_s2, dt_s, damping=.05, tmin=.01, tmax=10., points=100, spacing="log"):
    # The SDOF solver is homogeneous in acceleration units. Supplying cm/s²
    # therefore returns pseudo-spectral acceleration directly in cm/s².
    return response_spectrum(np.asarray(acc_cm_s2, float), float(dt_s), xi=float(damping),
                             np_points=int(points), t_min=float(tmin), t_max=float(tmax),
                             t_scale=1 if str(spacing).lower().startswith("log") else 0)


def family_spectra(accelerations, dt_s, settings, valid_npts=None, progress=None):
    fas = []; rs = []
    a = np.asarray(accelerations, float)
    valid = np.asarray(valid_npts if valid_npts is not None else [a.shape[-1]] * len(a), int)
    for i, rec in enumerate(a):
        n = int(np.clip(valid[i] if i < len(valid) else a.shape[-1], 2, a.shape[-1]))
        ff = fas_record(rec[:, :n], dt_s, settings.fas_fmin_hz, settings.fas_fmax_hz,
                        settings.fas_smoothing, settings.fas_smoothing_bandwidth)
        rr=[]
        for comp in rec[:, :n]:
            rr.append(response_one(comp, dt_s, settings.rs_damping, settings.rs_tmin_s,
                                   settings.rs_tmax_s, settings.rs_points, settings.rs_spacing))
        fas.append(ff); rs.append(rr)
        if progress is not None:
            progress(i + 1, len(a))
    return fas, rs
