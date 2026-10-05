"""
Signal operations - taper, detrend, filters, padding.

This module contains standard signal processing utilities that are
not part of the core simulation method but are used in preprocessing.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional


def apply_taper_start(signal_array, dt, taper_duration):
    """Apply cosine taper at the start of the signal."""
    x = np.asarray(signal_array, dtype=float).ravel()
    n = len(x)
    if n < 4:
        return x
    taper_s = float(taper_duration)
    if taper_s <= 0:
        return x
    n_taper = int(round(taper_s / max(dt, 1e-9)))
    n_taper = max(2, min(n_taper, n // 4))
    taper = np.ones(n, dtype=float)
    t = np.arange(n_taper, dtype=float) / float(n_taper - 1) if n_taper > 1 else np.array([0.0])
    taper[:n_taper] = 0.5 * (1.0 - np.cos(2.0 * np.pi * t * 0.5))
    return x * taper


def apply_taper_final(signal_array, dt, taper_duration):
    """Apply cosine taper at the end of the signal."""
    x = np.asarray(signal_array, dtype=float).ravel()
    n = len(x)
    if n < 4:
        return x
    taper_s = float(taper_duration)
    if taper_s <= 0:
        return x
    n_taper = int(round(taper_s / max(dt, 1e-9)))
    n_taper = max(2, min(n_taper, n // 4))
    taper = np.ones(n, dtype=float)
    t = np.arange(n_taper, dtype=float) / float(n_taper - 1) if n_taper > 1 else np.array([0.0])
    taper[-n_taper:] = 0.5 * (1.0 - np.cos(2.0 * np.pi * t * 0.5))
    return x * taper


def _poly_detrend_signal(y: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
    yy = np.asarray(y, dtype=float).ravel().copy()
    tt = np.asarray(t, dtype=float).ravel()
    if len(yy) != len(tt) or len(yy) < 3:
        return yy
    ord_v = int(max(0, min(int(order), 5)))
    if len(yy) <= ord_v + 1:
        ord_v = max(0, len(yy) - 2)
    if ord_v <= 0:
        return yy - float(np.mean(yy))
    try:
        coef = np.polyfit(tt, yy, ord_v)
        return np.asarray(yy - np.polyval(coef, tt), dtype=float)
    except Exception:
        return yy - float(np.mean(yy))


def apply_baseline_correction(signal_array, dt: float, mode: int) -> np.ndarray:
    """Apply the baseline-correction mode shared by the simulation methods."""
    yy = np.nan_to_num(
        np.asarray(signal_array, dtype=float).ravel(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if len(yy) < 4 or int(mode) <= 0:
        return yy

    mode_v = int(mode)
    dt_v = float(max(float(dt), 1e-9))
    tt = np.arange(len(yy), dtype=float) * dt_v
    yy = yy - float(np.mean(yy))
    if mode_v in (1, 6):
        order = 1
    elif mode_v == 2:
        order = 2
    else:
        order = 3
    yy = _poly_detrend_signal(yy, tt, order)

    if mode_v in (4, 5) and len(yy) >= 8:
        try:
            velocity = np.cumsum(yy) * dt_v
            velocity_trend = np.polyval(np.polyfit(tt, velocity, 1), tt)
            yy = np.asarray(yy - np.gradient(velocity_trend, dt_v), dtype=float)
            yy = _poly_detrend_signal(yy, tt, 1)
        except Exception:
            pass
    return np.nan_to_num(yy, nan=0.0, posinf=0.0, neginf=0.0)


def stabilize_terminal_displacement(
    signal_array,
    dt: float,
    tail_duration_s: float = 10.0,
    max_correction_rms_ratio: float = 0.10,
    return_diagnostics: bool = False,
    target_tail_displacement_cm: float | None = None,
    active_start_index: int | None = None,
    active_end_index_exclusive: int | None = None,
):
    """Remove post-filter integration drift with a smooth acceleration correction.

    ``None`` preserves the inactive final plateau. With an explicit target,
    two compact corrections at the active-window edges close terminal velocity
    and displacement without creating a broad arch upon double integration.
    """
    yy = np.nan_to_num(
        np.asarray(signal_array, dtype=float).ravel(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    dt_v = float(dt)
    empty_diag = {
        "applied": False,
        "reason": "insufficient_data",
        "velocity_end_before_cm_s": 0.0,
        "velocity_end_after_cm_s": 0.0,
        "tail_plateau_error_before_cm": 0.0,
        "tail_plateau_error_after_cm": 0.0,
        "correction_rms_ratio": 0.0,
    }
    if len(yy) < 16 or (not np.isfinite(dt_v)) or dt_v <= 0:
        return (yy, empty_diag) if return_diagnostics else yy

    def _integrate_twice(acceleration):
        aa = np.asarray(acceleration, dtype=float)
        velocity = np.zeros_like(aa)
        displacement = np.zeros_like(aa)
        velocity[1:] = np.cumsum(0.5 * (aa[1:] + aa[:-1])) * dt_v
        displacement[1:] = np.cumsum(0.5 * (velocity[1:] + velocity[:-1])) * dt_v
        return velocity, displacement

    n = len(yy)
    duration_s = float((n - 1) * dt_v)
    n_tail = int(round(max(float(tail_duration_s), 1.0) / dt_v))
    n_tail = int(np.clip(n_tail, 8, max(8, n // 3)))
    tail_slice = slice(n - n_tail, n)

    def _state(acceleration):
        velocity, displacement = _integrate_twice(acceleration)
        return np.asarray([
            velocity[-1],
            displacement[-1] - float(np.mean(displacement[tail_slice])),
            float(np.mean(displacement[tail_slice])),
        ], dtype=float)

    before = _state(yy)
    tail_target = before[2] if target_tail_displacement_cm is None else float(target_tail_displacement_cm)
    if not np.isfinite(tail_target):
        raise ValueError("target_tail_displacement_cm must be finite")
    if target_tail_displacement_cm is None:
        # Plateau-preserving path.
        alpha = float(np.clip(20.0 / max(duration_s, dt_v), 0.05, 1.0))
        window = sp_signal.windows.tukey(n, alpha=alpha)
        centered_time = np.linspace(-1.0, 1.0, n, dtype=float)
        bases = (
            window,
            window * centered_time,
            window * (1.5 * centered_time * centered_time - 0.5),
        )
        residual = before - np.asarray([0.0, 0.0, tail_target], dtype=float)
        matrix = np.column_stack([_state(base) for base in bases])
        correction_mode = "broad_plateau_preserving"
    else:
        # Correct spurious initial velocity at the start and terminal state at
        # the end. Keep the padded zero margins and central strong motion intact.
        start = int(np.clip(0 if active_start_index is None else active_start_index, 0, n - 2))
        end = int(np.clip(n if active_end_index_exclusive is None else active_end_index_exclusive, start + 2, n))
        width = min(max(8, int(round(5.0 / dt_v))), max(4, (end - start) // 3))
        if width < 4 or end - start < 2 * width + 2:
            diag = dict(empty_diag)
            diag["reason"] = "insufficient_active_window"
            return (yy, diag) if return_diagnostics else yy
        pulse = np.sin(np.linspace(0.0, np.pi, width, dtype=float)) ** 2
        early = np.zeros(n, dtype=float)
        late = np.zeros(n, dtype=float)
        early[start:start + width] = pulse
        late[end - width:end] = pulse
        bases = (early, late)
        residual = np.asarray([before[0], before[1] + before[2] - tail_target], dtype=float)
        basis_states = [_state(base) for base in bases]
        matrix = np.column_stack([
            np.asarray([state[0], state[1] + state[2]], dtype=float)
            for state in basis_states
        ])
        correction_mode = "localized_edge_pulses"
    try:
        condition = float(np.linalg.cond(matrix))
        if (not np.isfinite(condition)) or condition > 1.0e12:
            raise np.linalg.LinAlgError("ill-conditioned terminal correction")
        coefficients = np.linalg.solve(matrix, residual)
        correction = sum(coef * base for coef, base in zip(coefficients, bases))
    except Exception:
        diag = dict(empty_diag)
        diag["reason"] = "ill_conditioned"
        return (yy, diag) if return_diagnostics else yy

    signal_rms = float(np.sqrt(np.mean(yy * yy)))
    correction_rms = float(np.sqrt(np.mean(correction * correction)))
    correction_ratio = correction_rms / max(signal_rms, np.finfo(float).eps)
    ratio_limit = float(max(max_correction_rms_ratio, 0.0))
    if ratio_limit > 0.0 and correction_ratio > ratio_limit:
        correction *= ratio_limit / correction_ratio
        correction_ratio = ratio_limit
        reason = "limited"
    else:
        reason = "ok"

    corrected = np.nan_to_num(yy - correction, nan=0.0, posinf=0.0, neginf=0.0)
    after = _state(corrected)
    diagnostics = {
        "applied": True,
        "reason": reason,
        "velocity_end_before_cm_s": float(before[0]),
        "velocity_end_after_cm_s": float(after[0]),
        "tail_plateau_error_before_cm": float(before[1]),
        "tail_plateau_error_after_cm": float(after[1]),
        "tail_plateau_level_before_cm": float(before[2]),
        "tail_plateau_level_after_cm": float(after[2]),
        "tail_plateau_target_cm": float(tail_target),
        "correction_mode": correction_mode,
        "terminal_displacement_before_cm": float(before[1] + before[2]),
        "terminal_displacement_after_cm": float(after[1] + after[2]),
        "correction_rms_ratio": float(correction_ratio),
        "tail_duration_s": float(n_tail * dt_v),
    }
    return (corrected, diagnostics) if return_diagnostics else corrected


def stabilize_3c_output_displacement(
    accelerations,
    dt: float,
    metadata=None,
    tail_padding_s: float = 10.0,
):
    """Apply the same terminal-displacement control to a generated 3C batch.

    The generator's samples remain the active interval; a zero tail gives the
    double integral a genuine final plateau to close against. All three
    components of each realization receive identical support and padding.
    """
    traces = np.asarray(accelerations, dtype=float)
    dt_v = float(dt)
    if traces.ndim != 3 or traces.shape[1] != 3 or traces.shape[2] < 16:
        raise ValueError("Expected generated accelerations with shape (N, 3, samples)")
    if not np.isfinite(dt_v) or dt_v <= 0.0 or not np.isfinite(traces).all():
        raise ValueError("Invalid generated accelerations or dt")
    pad_n = max(0, int(round(float(tail_padding_s) / dt_v)))
    output = np.pad(traces, ((0, 0), (0, 0), (0, pad_n)), mode="constant")
    records = list(metadata or [])
    diagnostics = []
    for run in range(output.shape[0]):
        record = records[run] if run < len(records) and isinstance(records[run], dict) else {}
        active_end = int(record.get("unpadded_npts", traces.shape[2]))
        active_end = int(np.clip(active_end, 16, traces.shape[2]))
        for component in range(3):
            corrected, info = stabilize_terminal_displacement(
                output[run, component],
                dt_v,
                tail_duration_s=tail_padding_s,
                return_diagnostics=True,
                target_tail_displacement_cm=0.0,
                active_start_index=0,
                active_end_index_exclusive=active_end,
            )
            output[run, component] = corrected
            diagnostics.append({"run_index": run, "component": ("major", "intermediate", "vertical")[component], **info})
    return output.astype(np.float32), diagnostics


def detrend_linear(x: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Remove linear trend from signal.

    If t is provided, uses it for fitting. Otherwise uses sample index.
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 3:
        return x - np.mean(x)

    if t is None:
        t = np.arange(len(x), dtype=float)
    else:
        t = np.asarray(t, dtype=float).ravel()
        if len(t) != len(x):
            t = np.arange(len(x), dtype=float)

    # Fit linear trend
    coeffs = np.polyfit(t, x, 1)
    trend = np.polyval(coeffs, t)

    return x - trend


def bandpass_butter(x: np.ndarray, fs: float, fmin: float, fmax: float,
                    order: int = 4, zero_phase: bool = True) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to signal.

    Parameters
    ----------
    x : array
        Input signal
    fs : float
        Sampling frequency (Hz)
    fmin : float
        Low cutoff frequency (Hz)
    fmax : float
        High cutoff frequency (Hz)
    order : int
        Filter order
    zero_phase : bool
        If True, apply filtfilt for zero-phase filtering

    Returns
    -------
    x_filt : array
        Filtered signal
    """
    x = np.asarray(x, dtype=float).ravel()
    if len(x) < 8:
        return x

    # Handle edge cases
    nyq = fs / 2
    if fmin <= 0:
        fmin = 0.01
    if fmax >= nyq:
        fmax = nyq * 0.95
    if fmin >= fmax:
        fmin = fmax * 0.5

    # Design in second-order sections.  This matches the UI's SOS mode and is
    # numerically safer than a high-order direct-form (b, a) realization.
    try:
        sos = sp_signal.butter(
            order, [fmin / nyq, fmax / nyq], btype='band', output='sos'
        )
    except ValueError:
        # Fallback to lower order
        sos = sp_signal.butter(
            2, [fmin / nyq, fmax / nyq], btype='band', output='sos'
        )

    if zero_phase:
        x_filt = sp_signal.sosfiltfilt(sos, x)
    else:
        x_filt = sp_signal.sosfilt(sos, x)

    return x_filt


def _bandpass_iir(x: np.ndarray, fs: float, f1: float, f2: float, order: int = 4) -> np.ndarray:
    """Apply the configured zero-phase IIR band-pass filter."""
    return bandpass_butter(x, fs, f1, f2, order=order, zero_phase=True)


def pad_with_zeros(x: np.ndarray, n_pad: int, position: str = 'end') -> np.ndarray:
    """
    Pad signal with zeros.

    Parameters
    ----------
    x : array
        Input signal
    n_pad : int
        Number of samples to pad
    position : str
        'end' - pad at end, 'start' - pad at start, 'both' - pad both ends

    Returns
    -------
    x_padded : array
        Padded signal
    """
    x = np.asarray(x, dtype=float).ravel()
    n_pad = int(max(0, n_pad))

    if n_pad == 0:
        return x

    if position == 'end':
        return np.pad(x, (0, n_pad), mode='constant', constant_values=0)
    elif position == 'start':
        return np.pad(x, (n_pad, 0), mode='constant', constant_values=0)
    elif position == 'both':
        return np.pad(x, (n_pad // 2, n_pad - n_pad // 2), mode='constant', constant_values=0)
    else:
        return np.pad(x, (0, n_pad), mode='constant', constant_values=0)


def compute_husid(acc: np.ndarray, dt: float, normalize: bool = True) -> np.ndarray:
    """
    Compute Husid (integral of a^2) normalized to [0, 1].

    Parameters
    ----------
    acc : array
        Acceleration time series
    dt : float
        Time step
    normalize : bool
        If True, normalize to [0, 1]

    Returns
    -------
    husid : array
        Husid vector
    """
    acc = np.asarray(acc, dtype=float).ravel()
    if len(acc) < 2 or dt <= 0:
        return np.array([0.0])

    husid = np.cumsum(acc ** 2) * dt

    if normalize and husid[-1] > 0:
        husid = husid / husid[-1]

    return husid


def compute_arias_intensity(acc: np.ndarray, dt: float, g: float = 1.0) -> float:
    """
    Compute Arias intensity.

    IA = (π/2g) * ∫ a(t)² dt

    Parameters
    ----------
    acc : array
        Acceleration time series (in consistent units)
    dt : float
        Time step
    g : float
        Gravitational acceleration (default: 1.0 for cm/s² input)

    Returns
    -------
    ia : float
        Arias intensity
    """
    acc = np.asarray(acc, dtype=float).ravel()
    if len(acc) < 2 or dt <= 0:
        return 0.0

    ia = np.pi / 2.0 / g * np.sum(acc ** 2) * dt

    return ia


def apply_tf_cutoff(power_env: np.ndarray, t_axis: np.ndarray, tf: float, dt: float, taper_s: float = 8.0) -> np.ndarray:
    """
    Apply smooth tail damping only AFTER tf (no pre-tf truncation).

    Parameters
    ----------
    power_env : array
        Power envelope
    t_axis : array
        Time axis
    tf : float
        Final time (Hz)
    dt : float
        Time step
    taper_s : float
        Taper duration in seconds

    Returns
    -------
    p_out : array
        Damped power envelope
    """
    p = np.maximum(np.asarray(power_env, dtype=float), 0.0).copy()
    t = np.asarray(t_axis, dtype=float)
    if p.ndim != 1 or t.ndim != 1 or len(p) != len(t) or len(p) == 0:
        return p
    tf_v = float(tf)
    if not np.isfinite(tf_v):
        return p
    if tf_v >= float(t[-1]):
        return p

    tau = float(max(taper_s, 2.0 * dt, 1e-6))
    mask = t >= tf_v
    if np.any(mask):
        p[mask] *= np.exp(-(t[mask] - tf_v) / tau)
    return p


def apply_envelope_log_ou(
    power_env: np.ndarray,
    t_axis: np.ndarray,
    dt: float,
    t1: float,
    t2: float,
    tf: float,
    sigma_pre: float = 0.0,
    sigma_strong: float = 0.0,
    sigma_coda: float = 0.0,
    tau_coda: float = 5.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Multiplicative log-OU perturbation over envelope power:
      p_out(t) = p(t) * exp(sigma(t) * z(t) - 0.5 * sigma(t)^2)
    where z(t) is a correlated unit-variance OU process.
    """
    p = np.maximum(np.asarray(power_env, dtype=float).ravel(), 0.0)
    t = np.asarray(t_axis, dtype=float).ravel()
    if len(p) != len(t) or len(p) < 8:
        return p

    s_pre = float(max(0.0, sigma_pre))
    s_str = float(max(0.0, sigma_strong))
    s_cod = float(max(0.0, sigma_coda))
    if s_pre <= 0.0 and s_str <= 0.0 and s_cod <= 0.0:
        return p

    dt_v = float(max(float(dt), 1e-6))
    tau_v = float(tau_coda) if np.isfinite(tau_coda) else 5.0
    tau_v = float(max(tau_v, 2.0 * dt_v, 1e-3))

    rng = np.random.default_rng(None if seed is None else int(seed))
    z = np.zeros(len(p), dtype=float)
    phi = float(np.exp(-dt_v / tau_v))
    eta = float(np.sqrt(max(1.0 - phi * phi, 1e-12)))
    eps = rng.standard_normal(len(p))
    for i in range(1, len(p)):
        z[i] = phi * z[i - 1] + eta * eps[i]

    t1_v = float(t1) if np.isfinite(t1) else float(t[0])
    t2_v = float(t2) if np.isfinite(t2) else float(t1_v + dt_v)
    tf_v = float(tf) if np.isfinite(tf) else float(t[-1])
    if t2_v <= t1_v:
        t2_v = t1_v + dt_v

    sigma_t = np.zeros(len(p), dtype=float)
    m_pre = t < t1_v
    m_strong = (t >= t1_v) & (t < t2_v)
    m_coda = t >= t2_v
    sigma_t[m_pre] = s_pre
    sigma_t[m_strong] = s_str
    sigma_t[m_coda] = s_cod

    blend_w = float(max(0.5, 5.0 * dt_v))
    m_blend_12 = (t >= (t1_v - blend_w)) & (t <= (t1_v + blend_w))
    if int(np.sum(m_blend_12)) >= 3:
        u12 = (t[m_blend_12] - t1_v) / max(blend_w, 1e-9)
        w12 = 0.5 * (1.0 + np.tanh(u12))
        sigma_t[m_blend_12] = (1.0 - w12) * s_pre + w12 * s_str

    m_blend_23 = (t >= (t2_v - blend_w)) & (t <= (t2_v + blend_w))
    if int(np.sum(m_blend_23)) >= 3:
        u23 = (t[m_blend_23] - t2_v) / max(blend_w, 1e-9)
        w23 = 0.5 * (1.0 + np.tanh(u23))
        sigma_t[m_blend_23] = (1.0 - w23) * s_str + w23 * s_cod

    factor = np.exp(sigma_t * z - 0.5 * np.square(sigma_t))
    p_out = np.maximum(p * factor, 0.0)

    a = float(min(float(t[0]), tf_v))
    b = float(max(float(t[0]), tf_v))
    m_ref = (t >= a) & (t <= b)
    if int(np.sum(m_ref)) < 8:
        m_ref = np.ones_like(t, dtype=bool)
    if int(np.sum(m_ref)) >= 8:
        ref = float(np.mean(p[m_ref]))
        cur = float(np.mean(p_out[m_ref]))
        if np.isfinite(ref) and np.isfinite(cur) and cur > 1e-18:
            p_out *= float(ref / cur)

    return np.nan_to_num(np.maximum(p_out, 0.0), nan=0.0, posinf=0.0, neginf=0.0)


def build_tail_gate(t_axis: np.ndarray, tf: float, dt: float, taper_s: float = 8.0) -> np.ndarray:
    """
    Smooth post-tf damping for acceleration signal:
    - 1 for t < tf
    - exp(-(t-tf)/tau) for t >= tf

    Parameters
    ----------
    t_axis : array
        Time array
    tf : float
        Final time
    dt : float
        Time step
    taper_s : float
        Taper duration

    Returns
    -------
    g : array
        Gate array [0, 1]
    """
    t = np.asarray(t_axis, dtype=float)
    if t.ndim != 1 or len(t) == 0:
        return np.array([], dtype=float)
    tf_v = float(tf)
    if not np.isfinite(tf_v):
        return np.ones_like(t)

    g = np.ones_like(t)
    if tf_v >= float(t[-1]):
        return g
    tau = float(max(taper_s, 2.0 * max(dt, 1e-9), 1e-6))
    mask = t >= tf_v
    if np.any(mask):
        g[mask] = np.exp(-(t[mask] - tf_v) / tau)
    return g
