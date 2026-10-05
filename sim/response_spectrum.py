"""Elastic response spectra using a GSSSS time-integration algorithm.

Reference
---------
Zhou, X. and Tamma, K. K. (2004). Design, analysis, and synthesis of
generalized single step single solve and optimal algorithms for structural
dynamics. International Journal for Numerical Methods in Engineering, 59(5),
597-668. https://doi.org/10.1002/nme.873
"""

import numpy as np
from scipy import signal
from typing import Tuple


def response_spectrum(
    acel: np.ndarray,
    dt: float,
    xi: float = 0.05,
    np_points: int = 100,
    t_min: float = 0.01,
    t_max: float = 10.0,
    t_scale: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute elastic pseudo-spectral acceleration with the Generalized
    Single-Step Single-Solve (GSSSS) algorithm.

    Parameters
    ----------
    acel : array-like
        Acceleration time series in any consistent acceleration unit.
    dt : float
        Time step (s)
    xi : float
        Damping ratio (default 0.05 = 5%)
    np_points : int
        Number of period points
    t_min : float
        Minimum period (s)
    t_max : float
        Maximum period (s)
    t_scale : int
        1 = logarithmic, 0 = linear

    Returns
    -------
    T : ndarray
        Period vector (s)
    Sa : ndarray
        Spectral acceleration in the same unit as ``acel``.
    """
    acel = np.asarray(acel, dtype=float).ravel()
    lac = len(acel)
    if lac < 10 or dt <= 0:
        return np.array([]), np.array([])

    # Accept both accepted flags for logarithmic spacing (1 and 2).
    if int(t_scale) != 0:
        T = np.logspace(np.log10(t_min), np.log10(t_max), np_points)
    else:
        T = np.linspace(t_min, t_max, np_points)

    n = len(T)
    Sa = np.zeros(n)
    w0 = 2.0 * np.pi / T
    vmma = 1.0

    w1 = -15.0 * (1.0 - 2.0 * vmma) / (1.0 - 4.0 * vmma)
    w2 = 15.0 * (3.0 - 4.0 * vmma) / (1.0 - 4.0 * vmma)
    w3 = -35.0 * (1.0 - vmma) / (1.0 - 4.0 * vmma)

    W1 = (0.5 + w1 / 3.0 + w2 / 4.0 + w3 / 5.0) / (1.0 + w1 / 2.0 + w2 / 3.0 + w3 / 4.0)
    W1L1 = 1.0 / (1.0 + vmma)
    W2L2 = 0.5 / (1.0 + vmma)
    W3L3 = 0.5 / (1.0 + vmma) ** 2
    W1L4 = 1.0 / (1.0 + vmma)
    W2L5 = 1.0 / (1.0 + vmma) ** 2
    W1L6 = (3.0 - vmma) / 2.0 / (1.0 + vmma)

    lam1 = 1.0
    lam2 = 0.5
    lam3 = 0.5 / (1.0 + vmma)
    lam4 = lam1
    lam5 = 1.0 / (1.0 + vmma)

    x = -acel

    for i in range(n):
        w = w0[i]
        W = w * dt

        D = W1L6 + 2.0 * W2L5 * xi * W + W3L3 * W ** 2

        A31 = -W ** 2 / D
        A32 = -(2.0 * xi * W + W1L1 * W ** 2) / D
        A33 = 1.0 - (1.0 + 2.0 * W1L4 * xi * W + W2L2 * W ** 2) / D

        A11 = 1.0 + lam3 * A31
        A12 = lam1 + lam3 * A32
        A13 = lam2 - lam3 * (1.0 - A33)

        A21 = lam5 * A31
        A22 = 1.0 + lam5 * A32
        A23 = lam4 - lam5 * (1.0 - A33)

        B1 = dt ** 2 * lam3 * W1 / D
        B2 = dt ** 2 / D * (lam3 * (1.0 - W1) - (A22 + A33) * lam3 * W1 + A12 * lam5 * W1 + A13 * W1)
        B3 = dt ** 2 / D * (
            -(A22 + A33) * lam3 * (1.0 - W1)
            + A12 * lam5 * (1.0 - W1)
            + A13 * (1.0 - W1)
            + (A22 * A33 - A23 * A32) * lam3 * W1
            - (A12 * A33 - A13 * A32) * lam5 * W1
            + (A12 * A23 - A13 * A22) * W1
        )
        B4 = dt ** 2 / D * (
            (A22 * A33 - A23 * A32) * lam3 * (1.0 - W1)
            - (A12 * A33 - A13 * A32) * lam5 * (1.0 - W1)
            + (A12 * A23 - A13 * A22) * (1.0 - W1)
        )

        b = np.array([B1, B2, B3, B4])
        a = np.array([1.0, -(A11 + A22 + A33), (A11*A22 - A12*A21 + A11*A33 - A13*A31 + A22*A33 - A23*A32), -(A11*A22*A33 - A11*A23*A32 - A12*A21*A33 + A12*A23*A31 + A13*A21*A32 - A13*A22*A31)])

        des = signal.lfilter(b, a, x)
        Sa[i] = float(np.max(np.abs(des))) * (w * w)

    return T, Sa
