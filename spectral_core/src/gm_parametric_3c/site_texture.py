from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .site_texture_base import SiteTextureBase, _site_gate

ARCHITECTURE_ID = "gm_parametric_3c.resonance_texture.core"
SCHEMA_VERSION = "1.6.0"


def _weighted_profile_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Pointwise weighted quantile for an empirical bank [sample, coordinate]."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    if v.ndim != 2 or v.shape[0] != len(w):
        raise ValueError("values/weights shape mismatch")
    if len(w) == 0:
        return np.zeros(v.shape[1], dtype=float)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    sw = float(w.sum())
    w = (w / sw) if sw > 0.0 else np.full(len(w), 1.0 / len(w), dtype=float)
    q = float(np.clip(q, 0.0, 1.0))
    out = np.empty(v.shape[1], dtype=float)
    for j in range(v.shape[1]):
        order = np.argsort(v[:, j], kind="mergesort")
        vv = v[order, j]
        ww = w[order]
        k = int(np.searchsorted(np.cumsum(ww), q, side="left"))
        out[j] = float(vv[min(k, len(vv) - 1)])
    return out


@dataclass
class SiteTextureModel(SiteTextureBase):
    """Localized resonance/texture architecture used by VALLIS-3C.

    The persistent XY–Ts field is the deterministic local site texture. The
    second layer is an empirical conditional record residual.  The residual is
    centered on the model frequency grid by the weighted conditional median
    of the same KNN pool used for sampling.  Thus stochastic texture changes
    spread/shape but cannot systematically move the persistent local spectral departure.
    """

    architecture_id: str = ARCHITECTURE_ID
    schema_version: str = SCHEMA_VERSION
    provenance: dict[str, Any] = field(default_factory=dict)
    model_version: str = "1.6.0"
    u_window: tuple[float, float, float, float] = (0.35, 0.50, 3.0, 4.0)

    @staticmethod
    def _finite_weighted_profile(bank, weights):
        """Blend station profiles pointwise using only observed/finite support."""
        bank = np.asarray(bank, float)
        weights = np.asarray(weights, float).reshape(-1)
        finite = np.isfinite(bank)
        ww = weights[:, None] * finite
        den = ww.sum(axis=0)
        num = np.sum(np.where(finite, bank, 0.0) * weights[:, None], axis=0)
        out = np.divide(num, den, out=np.zeros(bank.shape[1], float), where=den > 0.0)
        return out, den

    def _predict_site_profile_one(self, row):
        """Persistent local texture with pointwise support-aware interpolation.

        Long-period stations may lack only the lowest normalized-frequency nodes
        because the observed FAS begins at 0.1 Hz. Those nodes are never
        extrapolated: a known station contracts to zero local texture there, and
        KNN interpolation renormalizes weights among neighbors that actually have
        support at that u.
        """
        sid = str(row.get("station_id", "") or "")
        if sid:
            m = np.where(self.station_ids.astype(str) == sid)[0]
            if len(m):
                raw = np.asarray(self.station_profiles[int(m[0])], float)
                finite = np.isfinite(raw)
                prof = np.where(finite, raw, 0.0)
                return prof, {
                    "site_texture_source": "known_station",
                    "station_id": sid,
                    "neighbor_count": 1,
                    "profile_support_fraction": float(np.mean(finite)),
                    "partial_support_identity_nodes": int(np.sum(~finite)),
                }
        x = float(row.get("x_conditioned_km", row.get("x_km", np.nan)))
        y = float(row.get("y_conditioned_km", row.get("y_km", np.nan)))
        ts = float(row["Ts_s"])
        if not np.isfinite(x) or not np.isfinite(y):
            return np.zeros_like(self.u_grid), {"site_texture_source": "zero_no_xy", "neighbor_count": 0}
        q = (self._station_feature(x, y, ts) - self.station_feature_center) / self.station_feature_scale
        X = np.column_stack([self.station_x_km, self.station_y_km, np.log(np.clip(self.station_Ts_s, 1e-6, None))])
        X = (X - self.station_feature_center) / self.station_feature_scale
        d = np.sqrt(np.sum((X - q[None, :]) ** 2, axis=1))
        k = min(int(self.station_k), len(d)); idx = np.argpartition(d, k - 1)[:k]
        ww = 1.0 / np.maximum(d[idx], 0.20) ** 2; ww /= ww.sum()
        prof, den = self._finite_weighted_profile(self.station_profiles[idx], ww)
        d0 = float(np.min(d[idx]))
        shrink = float(np.exp(-0.10 * max(0.0, d0 - 2.0) ** 2))
        prof *= shrink
        return prof, {
            "site_texture_source": "knn",
            "neighbor_count": int(k),
            "nearest_std_distance": d0,
            "support_shrinkage": shrink,
            "neighbor_station_ids": [str(x) for x in self.station_ids[idx]],
            "profile_support_fraction": float(np.mean(den > 0.0)),
            "partial_support_identity_nodes": int(np.sum(den <= 0.0)),
        }

    def _window(self, u):
        u = np.asarray(u, float)
        u0, u1, u2, u3 = [float(v) for v in self.u_window]
        if not (u0 < u1 <= u2 < u3):
            raise ValueError(f"Invalid texture u_window={self.u_window}")
        w = np.zeros_like(u)
        def smoothstep01(x):
            x = np.clip(np.asarray(x, float), 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)
        m = (u > u0) & (u < u1)
        w[m] = smoothstep01((u[m] - u0) / (u1 - u0))
        w[(u >= u1) & (u <= u2)] = 1.0
        m = (u > u2) & (u < u3)
        w[m] = 1.0 - smoothstep01((u[m] - u2) / (u3 - u2))
        return w

    def _conditional_residual_pool(self, row, k=None, exclude_same_event=False):
        source = str(row.get("source_type", "")).upper()
        ev = str(row.get("event_id", "") or "")
        x = float(row.get("x_conditioned_km", row.get("x_km", np.nan)))
        y = float(row.get("y_conditioned_km", row.get("y_km", np.nan)))
        ts = float(row["Ts_s"])
        mw = float(row.get("Mw", np.nan))
        rr = float(row.get("Rrup_km", np.nan))
        qv = np.array([mw, np.log(max(rr, 1e-6)), np.log(max(ts, 1e-6)), x, y], float)

        pool = np.ones(len(self.residual_profiles), bool)
        if source:
            pool &= (np.char.upper(self.residual_source_type.astype(str)) == source)
        if exclude_same_event and ev:
            pool &= (self.residual_event_id.astype(str) != ev)
        ids = np.where(pool)[0]
        if len(ids) == 0:
            return ids, np.zeros(0, float), np.zeros(0, float), "zero_no_pool"

        if not np.isfinite(qv).all():
            w = np.full(len(ids), 1.0 / len(ids), dtype=float)
            return ids, w, np.full(len(ids), np.nan), "random_source_pool_centered"

        qq = (qv - self.residual_feature_center) / self.residual_feature_scale
        XX = (self.residual_features[ids] - self.residual_feature_center) / self.residual_feature_scale
        d = np.sqrt(np.sum((XX - qq[None, :]) ** 2, axis=1))
        kk = min(int(self.residual_k if k is None else k), len(ids))
        loc = np.argpartition(d, kk - 1)[:kk]
        cand = ids[loc]
        dc = d[loc]
        temp = max(float(np.median(dc) + 1e-6), 0.35)
        w = np.exp(-0.5 * (dc / temp) ** 2)
        w /= w.sum()
        return cand, w, dc, "empirical_knn_centered"

    def _sample_residual_one(self, row, rng, k=None, exclude_same_event=False):
        cand, w, dc, source_name = self._conditional_residual_pool(
            row, k=k, exclude_same_event=exclude_same_event
        )
        if len(cand) == 0:
            return np.zeros_like(self.u_grid), {
                "residual_texture_source": "zero_no_pool",
                "residual_centering": "conditional_weighted_median_final_grid",
            }
        center = _weighted_profile_quantile(self.residual_profiles[cand], w, 0.5)
        jj = int(rng.choice(len(cand), p=w))
        j = int(cand[jj])
        return self.residual_profiles[j].copy() - center, {
            "residual_texture_source": source_name,
            "donor_event_id": str(self.residual_event_id[j]),
            "donor_station_id": str(self.residual_station_id[j]),
            "donor_std_distance": float(dc[jj]) if len(dc) and np.isfinite(dc[jj]) else np.nan,
            "residual_centering": "conditional_weighted_median_u_grid",
            "conditional_pool_size": int(len(cand)),
        }


    def conditional_quantile_factors(
        self, frequencies_hz, scenarios, site_strength=1.0, residual_scale=1.0,
        quantiles=(0.16, 0.50, 0.84), ts_gate_low=0.55, ts_gate_high=0.80,
        log_clip=1.10, exclude_same_event=False,
    ):
        """Exact conditional empirical quantiles of the texture multiplier.

        Unlike a Monte Carlo display, this uses the same KNN donor probabilities
        directly, so the P50 curve is not affected by finite-sample jitter.
        """
        import pandas as pd

        freq = np.asarray(frequencies_hz, float)
        s = scenarios if isinstance(scenarios, pd.DataFrame) else pd.DataFrame(scenarios)
        s = s.reset_index(drop=True)
        site, site_diag = self.predict_site_profiles(s)
        lp = np.log(self.u_grid)
        qvals = tuple(float(q) for q in quantiles)
        out = np.ones((len(qvals), len(s), len(freq)), dtype=float)
        diags = []

        for i, row in s.iterrows():
            ts = float(row["Ts_s"])
            gate = float(_site_gate(ts, ts_gate_low, ts_gate_high))
            u = freq * ts
            win = self._window(u)
            lu = np.log(np.clip(u, self.u_grid[0], self.u_grid[-1]))
            sp = np.interp(lu, lp, site[i], left=0.0, right=0.0)
            cand, w, dc, source_name = self._conditional_residual_pool(
                row, exclude_same_event=exclude_same_event
            )
            if len(cand):
                bank = np.vstack([
                    np.interp(lu, lp, self.residual_profiles[int(j)], left=0.0, right=0.0)
                    for j in cand
                ])
                center_f = _weighted_profile_quantile(bank, w, 0.5)
                centered = bank - center_f[None, :]
                corr_bank = gate * win[None, :] * (
                    float(site_strength) * sp[None, :] + float(residual_scale) * centered
                )
                corr_bank = np.clip(corr_bank, -float(log_clip), float(log_clip))
                factor_bank = np.exp(corr_bank)
                for iq, q in enumerate(qvals):
                    out[iq, i] = _weighted_profile_quantile(factor_bank, w, q)
            else:
                corr = gate * win * float(site_strength) * sp
                corr = np.clip(corr, -float(log_clip), float(log_clip))
                out[:, i, :] = np.exp(corr)[None, :]
            diags.append({
                **site_diag[i],
                "residual_texture_source": source_name,
                "residual_centering": "conditional_weighted_median_final_frequency_grid",
                "conditional_pool_size": int(len(cand)),
                "quantiles": list(qvals),
                "u_window": [float(v) for v in self.u_window],
            })
        return {"factors": out, "quantiles": qvals, "diagnostics": diags}

    def apply(self, frequencies_hz, major, intermediate, scenarios, n_realizations=None, seed=None,
              site_strength=1.0, residual_enabled=True, residual_scale=1.0,
              ts_gate_low=0.55, ts_gate_high=0.80, log_clip=1.10):
        """Apply persistent + stochastic texture without shifting local median."""
        import pandas as pd

        freq = np.asarray(frequencies_hz, float)
        M = np.asarray(major, float).copy()
        I = np.asarray(intermediate, float).copy()
        if M.ndim == 2:
            M = M[None, :, :]
        if I.ndim == 2:
            I = I[None, :, :]
        nr = M.shape[0] if n_realizations is None else int(n_realizations)
        if nr != M.shape[0] or I.shape != M.shape:
            raise ValueError("major/intermediate texture arrays must have shape [realization, scenario, frequency]")
        s = scenarios if isinstance(scenarios, pd.DataFrame) else pd.DataFrame(scenarios)
        s = s.reset_index(drop=True)
        site, site_diag = self.predict_site_profiles(s)
        rng = np.random.default_rng(seed)
        lp = np.log(self.u_grid)
        all_diag = [[None for _ in range(len(s))] for _ in range(nr)]

        for i, row in s.iterrows():
            ts = float(row["Ts_s"])
            gate = float(_site_gate(ts, ts_gate_low, ts_gate_high))
            u = freq * ts
            win = self._window(u)
            lu = np.log(np.clip(u, self.u_grid[0], self.u_grid[-1]))
            sp = np.interp(lu, lp, site[i], left=0.0, right=0.0)

            if residual_enabled:
                cand, w, dc, source_name = self._conditional_residual_pool(
                    row, exclude_same_event=False
                )
                if len(cand):
                    # Interpolate every candidate first, then center on the exact
                    # requested frequency grid.  This is what guarantees a zero
                    # conditional median after interpolation.
                    bank = np.vstack([
                        np.interp(lu, lp, self.residual_profiles[int(j)], left=0.0, right=0.0)
                        for j in cand
                    ])
                    center_f = _weighted_profile_quantile(bank, w, 0.5)
                    picks = rng.choice(len(cand), size=nr, p=w)
                else:
                    bank = np.zeros((1, len(freq)), float)
                    center_f = np.zeros(len(freq), float)
                    picks = np.zeros(nr, dtype=int)
                    source_name = "zero_no_pool"
            else:
                cand = np.zeros(0, dtype=int)
                dc = np.zeros(0, float)
                bank = np.zeros((1, len(freq)), float)
                center_f = np.zeros(len(freq), float)
                picks = np.zeros(nr, dtype=int)
                source_name = "disabled"

            for r in range(nr):
                if residual_enabled and len(cand):
                    jj = int(picks[r])
                    rp = bank[jj] - center_f
                    j = int(cand[jj])
                    resid_diag = {
                        "residual_texture_source": source_name,
                        "donor_event_id": str(self.residual_event_id[j]),
                        "donor_station_id": str(self.residual_station_id[j]),
                        "donor_std_distance": float(dc[jj]) if len(dc) and np.isfinite(dc[jj]) else np.nan,
                        "residual_centering": "conditional_weighted_median_final_frequency_grid",
                        "conditional_pool_size": int(len(cand)),
                        "u_window": [float(v) for v in self.u_window],
                    }
                else:
                    rp = np.zeros(len(freq), float)
                    resid_diag = {
                        "residual_texture_source": source_name,
                        "residual_centering": "conditional_weighted_median_final_frequency_grid",
                        "u_window": [float(v) for v in self.u_window],
                    }
                corr = gate * win * (float(site_strength) * sp + float(residual_scale) * rp)
                corr = np.clip(corr, -float(log_clip), float(log_clip))
                M[r, i] *= np.exp(corr)
                I[r, i] *= np.exp(corr)
                all_diag[r][i] = {
                    "site_gate": gate,
                    "site_strength": float(site_strength),
                    "residual_scale": float(residual_scale),
                    "max_abs_log_correction": float(np.max(np.abs(corr))),
                    **site_diag[i],
                    **resid_diag,
                }

        return {
            "major": M,
            "intermediate": I,
            "diagnostics": all_diag,
            "model_version": self.model_version,
        }
