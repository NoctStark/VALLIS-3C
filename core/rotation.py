from __future__ import annotations
import numpy as np


def rotate_horizontals(accelerations, mode="none", angle_deg=0.0, seed=0):
    """Post-process horizontal components; an optional vertical component is unchanged.

    ``accelerations`` may have shape ``(n, 2, nt)`` or ``(n, 3, nt)``.
    """
    a = np.asarray(accelerations, dtype=float).copy()
    if a.ndim != 3 or a.shape[1] not in (2, 3):
        raise ValueError("accelerations must have shape (n, 2, nt) or (n, 3, nt)")
    n = a.shape[0]
    has_vertical = a.shape[1] == 3
    mode = str(mode).lower().strip()
    if mode == "none":
        labels = ("Major", "Intermediate", "Vertical") if has_vertical else ("Major", "Intermediate")
        return a, np.zeros(n), labels
    if mode == "random":
        rng = np.random.default_rng(int(seed) + 914521)
        angles = rng.uniform(0.0, 360.0, size=n)
    elif mode == "manual":
        angles = np.full(n, float(angle_deg))
    else:
        raise ValueError(f"Unknown horizontal-rotation mode: {mode}")
    for i, ang in enumerate(angles):
        th = np.deg2rad(float(ang)); c, s = np.cos(th), np.sin(th)
        m, q = a[i, 0].copy(), a[i, 1].copy()
        a[i, 0] = c*m + s*q
        a[i, 1] = -s*m + c*q
    labels = ("H1", "H2", "Vertical") if has_vertical else ("H1", "H2")
    return a, angles, labels
