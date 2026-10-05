"""Spectral calculations used by the VALLIS-3C application."""

import numpy as np
from typing import Tuple


def _amplitude_spectrum(series: np.ndarray, dt: float, nfft: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute single-sided amplitude spectrum for a 1D signal."""
    x = np.asarray(series, dtype=float).ravel()
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Signal must be 1D with at least 2 samples.")
    dt_v = float(dt)
    if not np.isfinite(dt_v) or dt_v <= 0:
        raise ValueError("dt must be positive.")
    nfft_v = int(max(8, nfft))
    x = np.nan_to_num(x - float(np.mean(x)), nan=0.0, posinf=0.0, neginf=0.0)
    X = np.fft.rfft(x, n=nfft_v)
    f = np.fft.rfftfreq(nfft_v, d=dt_v)
    amp = np.abs(X) * dt_v
    return np.asarray(f, dtype=float), np.asarray(amp, dtype=float)


def compute_fas_smoothed(freq: np.ndarray, fas: np.ndarray, n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothed FAS using log-bin averaging.

    Parameters
    ----------
    freq : array
        Frequency array
    fas : array
        FAS amplitude array
    n_points : int
        Number of points per bin

    Returns
    -------
    f_smooth, fas_smooth : arrays
        Smoothed frequency and FAS
    """
    log_f = np.log10(np.maximum(freq, 1e-10))
    log_fas = np.log10(np.maximum(fas, 1e-20))

    n = len(freq)
    n_bins = max(1, n // n_points)

    f_bins = []
    fas_bins = []

    for i in range(n_bins):
        start = i * n_points
        end = min((i + 1) * n_points, n)
        if end - start < 2:
            continue
        f_bins.append(np.mean(freq[start:end]))
        fas_bins.append(np.mean(fas[start:end]))

    return np.array(f_bins), np.array(fas_bins)


def konno_ohmachi_smooth(
    freq: np.ndarray,
    fas: np.ndarray,
    bandwidth: float = 40.0,
    fmin: float = 0.1,
    fmax: float = 20.0,
    eps: float = 1e-20,
    output_points: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth FAS using Konno-Ohmachi logarithmic weighting.

    Reference: Konno, K. and Ohmachi, T. (1998). Ground-motion
    characteristics estimated from spectral ratio between horizontal and
    vertical components of microtremor. Bulletin of the Seismological Society
    of America, 88(1), 228-241. https://doi.org/10.1785/bssa0880010228

    The weighting kernel for center frequency fc is:
      W(f, fc) = [sin(b*log10(f/fc)) / (b*log10(f/fc))]^4

    Parameters
    ----------
    freq : array
        Frequency array (Hz)
    fas : array
        FAS amplitude array. The last dimension must match ``freq``; leading
        dimensions are evaluated together with the same smoothing weights.
    bandwidth : float
        Konno-Ohmachi bandwidth coefficient b (typical: 20-50)
    fmin, fmax : float
        Optional frequency band to return
    eps : float
        Small floor to avoid log/zero issues
    output_points : int or None
        When provided, evaluate the exact Konno-Ohmachi weighted averages at
        this many logarithmically spaced center frequencies. All native FFT
        ordinates in the requested band remain in the weighted averages.
        ``None`` retains the native frequency grid.

    Returns
    -------
    f_out, fas_out : arrays
        Smoothed spectrum in requested band.
    """
    f = np.asarray(freq, dtype=float).ravel()
    a = np.asarray(fas, dtype=float)
    scalar_input = a.ndim == 1
    if scalar_input:
        a = a[np.newaxis, :]
    if a.ndim != 2 or a.shape[-1] != len(f) or len(f) < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    b = float(max(float(bandwidth), 1e-6))
    m = np.isfinite(f) & (f > 0.0) & (f >= float(fmin)) & (f <= float(fmax))
    if output_points is None and scalar_input:
        # Preserve the native-grid behavior used by earlier calculations.
        m &= np.isfinite(a[0]) & (a[0] > 0.0)
    else:
        m &= np.all(np.isfinite(a), axis=0)
    if int(np.sum(m)) < 4:
        return np.array([], dtype=float), np.array([], dtype=float)

    f_sel = np.asarray(f[m], dtype=float)
    a_sel = np.maximum(np.asarray(a[:, m], dtype=float), float(eps))
    logf = np.log10(np.maximum(f_sel, float(eps)))

    if output_points is None:
        centers = f_sel
    else:
        n_out = max(4, int(output_points))
        centers = np.geomspace(float(f_sel[0]), float(f_sel[-1]), n_out)

    # Evaluate in modest row blocks so memory is proportional to
    # block_size × native FFT ordinates rather than N².
    log_centers = np.log10(np.maximum(centers, float(eps)))
    out = np.empty((a_sel.shape[0], centers.size), dtype=float)
    block_size = 64
    for start in range(0, centers.size, block_size):
        stop = min(start + block_size, centers.size)
        x = b * (log_centers[start:stop, np.newaxis] - logf[np.newaxis, :])
        w = np.ones_like(x, dtype=float)
        nz = np.abs(x) > 1e-12
        w[nz] = np.sin(x[nz]) / x[nz]
        np.power(w, 4.0, out=w)
        sw = np.sum(w, axis=1)
        sw = np.where(sw > 1e-18, sw, 1.0)
        out[:, start:stop] = (a_sel @ w.T) / sw[np.newaxis, :]

    out = np.maximum(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    return centers, out[0] if scalar_input else out


def compute_spectral_error(sim_fas: np.ndarray, target_fas: np.ndarray,
                           freq: np.ndarray, fmin: float = 0.1, fmax: float = 25.0) -> float:
    """
    Compute RMS error in log space between simulated and target FAS.

    Returns
    -------
    error : float
        RMS error in log10 space
    """
    mask = (freq >= fmin) & (freq <= fmax)
    if not np.any(mask):
        return np.nan

    log_sim = np.log10(np.maximum(sim_fas[mask], 1e-20))
    log_target = np.log10(np.maximum(target_fas[mask], 1e-20))

    return np.sqrt(np.mean((log_sim - log_target) ** 2))


def _q_of_f(
    f_hz: float,
    q_model: str,
    q0: float,
    eta: float,
    fref: float,
) -> float:
    fi = float(max(f_hz, 1e-6))
    q0_v = float(max(q0, 1e-6))
    fref_v = float(max(fref, 1e-6))
    model_v = str(q_model or "power_law").strip().lower()
    if model_v == "constant":
        return q0_v
    eta_v = float(eta) if np.isfinite(eta) else 0.0
    return float(max(q0_v * (max(fi, fref_v) / fref_v) ** eta_v, 1e-6))
