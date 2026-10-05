from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer


DEFAULT_NEAR_TRENCH = ("1995-10-09", "1996-02-25", "1997-07-19")


def _restore_bspline_public_state(spline: SplineTransformer | None) -> None:
    """Restore the public BSpline state required by the packaged runtime.

    The release artifact stores BSpline knots/coefficients as ``_t``/``_c``.
    The packaged SplineTransformer reads the equivalent ``t``/``c`` attributes.
    Aliasing the same arrays is lossless and avoids rebuilding a fitted model.
    """
    for bspline in getattr(spline, "bsplines_", ()):
        if not hasattr(bspline, "t") and hasattr(bspline, "_t"):
            bspline.t = bspline._t
        if not hasattr(bspline, "c") and hasattr(bspline, "_c"):
            bspline.c = bspline._c


def _restore_model_splines(obj) -> None:
    for part_name in ("eas_mean", "delta_mean", "vh_mean"):
        part = getattr(obj, part_name, None)
        if isinstance(part, dict):
            _restore_bspline_public_state(part.get("site_spline"))
    for name in ("record_spline", "delta_residual_spline", "vh_residual_spline"):
        _restore_bspline_public_state(getattr(obj, name, None))


def _phys_x(df: pd.DataFrame) -> np.ndarray:
    """Low-dimensional source/path design. Depth acts only for intraslab events."""
    mw = pd.to_numeric(df["Mw"], errors="coerce").to_numpy(float)
    rr = pd.to_numeric(df["Rrup_km"], errors="coerce").to_numpy(float)
    dep = pd.to_numeric(df["depth_used_km"], errors="coerce").to_numpy(float)
    ins = df["source_type"].astype(str).str.upper().str.contains("INSLAB").to_numpy(float)
    m = mw - 7.0
    lr = np.log(np.maximum(rr, 1.0) / 300.0)
    r = (rr - 300.0) / 100.0
    d = ins * (dep - 40.0) / 30.0
    return np.column_stack(
        [np.ones(len(df)), m, m * m, lr, r, ins, d, m * lr, ins * m, ins * lr]
    )


def _event_weights(df: pd.DataFrame) -> np.ndarray:
    c = df.groupby("event_id")["event_id"].transform("count").to_numpy(float)
    w = 1.0 / c
    return w / np.mean(w)


def _site_design(
    df: pd.DataFrame,
    freq: np.ndarray,
    spline: SplineTransformer,
    ts_order: int,
    tref: float,
    dtype=np.float64,
) -> np.ndarray:
    """Reference-site constrained site design in u=f*Ts.

    The correction is identically zero at Ts=tref. This makes the source/path
    backbone interpretable as the conditional spectrum at the reference site.
    """
    ts = pd.to_numeric(df["Ts_s"], errors="coerce").to_numpy(float)
    n, nf = len(df), len(freq)
    logu = np.log(np.maximum(ts[:, None] * freq[None, :], 1e-12)).reshape(-1, 1)
    bu = spline.transform(logu).astype(dtype, copy=False)
    bref1 = spline.transform(np.log(tref * freq)[:, None]).astype(dtype, copy=False)
    bref = np.tile(bref1, (n, 1))
    z = np.log(ts / tref).astype(dtype)
    zrep = np.repeat(z, nf)
    blocks = [bu - bref]
    if ts_order >= 1:
        blocks.append(bu * zrep[:, None])
    if ts_order >= 2:
        blocks.append(bu * (zrep * zrep)[:, None])
    return np.concatenate(blocks, axis=1)


def _predict_part(part: dict[str, Any], df: pd.DataFrame, freq: np.ndarray) -> np.ndarray:
    x = _phys_x(df)
    src = x @ np.asarray(part["beta"], float)
    z = _site_design(
        df,
        freq,
        part["site_spline"],
        int(part["ts_order"]),
        float(part["tref"]),
        dtype=np.float64,
    )
    site = (z @ np.asarray(part["site_coef"], float)).reshape(len(df), len(freq))
    return src + site


def _reconstruct_logs(loge: np.ndarray, delta: np.ndarray, logvh: np.ndarray):
    # delta = ln(Major/Intermediate), EAS^2=(M^2+I^2)/2.
    log_i = loge + 0.5 * (np.log(2.0) - np.logaddexp(2.0 * delta, 0.0))
    log_m = log_i + delta
    log_v = loge + logvh
    return log_m, log_i, log_v


@dataclass
class FAS3CFunctionalHierarchicalModel:
    """Functional hierarchical three-component FAS model for the VALLIS-3C release.

    Horizontal EAS is decomposed into a smooth source/path backbone in absolute
    frequency plus a reference-site-constrained correction in u=f*Ts. Stochastic
    variability is represented by low-rank *functional* random effects, never by
    an Amax scalar and never by per-frequency independent noise.
    """

    frequencies_hz: np.ndarray
    eas_mean: dict[str, Any]
    delta_mean: dict[str, Any]
    vh_mean: dict[str, Any]
    eas_bias_curve: np.ndarray
    delta_bias_curve: np.ndarray
    vh_bias_curve: np.ndarray
    event_modes: np.ndarray
    event_std: np.ndarray
    record_spline: SplineTransformer
    record_modes_coef: np.ndarray
    record_std: np.ndarray
    delta_residual_spline: SplineTransformer
    delta_residual_modes_coef: np.ndarray
    delta_residual_std: np.ndarray
    vh_residual_spline: SplineTransformer
    vh_residual_modes_coef: np.ndarray
    vh_residual_std: np.ndarray
    training_report: dict[str, Any]
    model_version: str = "1.6.0"

    @property
    def config(self):
        return self.training_report.get("config", {})

    def save(self, path: str | Path):
        joblib.dump(self, path, compress=3)

    @classmethod
    def load(cls, path: str | Path):
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        _restore_model_splines(obj)
        return obj

    def write_report(self, path: str | Path):
        import json
        Path(path).write_text(json.dumps(self.training_report, indent=2, ensure_ascii=False), encoding="utf-8")

    def predict_latent(self, scenarios: pd.DataFrame):
        df = scenarios.copy().reset_index(drop=True)
        loge = _predict_part(self.eas_mean, df, self.frequencies_hz) + self.eas_bias_curve[None, :]
        delta = _predict_part(self.delta_mean, df, self.frequencies_hz) + self.delta_bias_curve[None, :]
        logvh = _predict_part(self.vh_mean, df, self.frequencies_hz) + self.vh_bias_curve[None, :]
        return loge, delta, logvh

    def predict_log(self, scenarios: pd.DataFrame, return_diagnostics: bool = False):
        loge, delta, logvh = self.predict_latent(scenarios)
        lm, li, lv = _reconstruct_logs(loge, delta, logvh)
        out = np.concatenate([lm, li, lv], axis=1)
        if not return_diagnostics:
            return out
        diag = {
            "architecture": "functional_hierarchical_release",
            "horizontal_definition": "EAS=sqrt((Major^2+Intermediate^2)/2)",
            "frequency_coordinates": {"source_path_event": "f", "site_record": "u=f*Ts"},
            "site_reference_Ts_s": float(self.eas_mean["tref"]),
            "event_modes": int(self.event_modes.shape[0]),
            "record_modes": int(self.record_modes_coef.shape[0]),
            "site_correction_mode": ["functional_u=fTs"] * len(scenarios),
            "no_amax_normalization": True,
            "no_frequencywise_clipping": True,
        }
        return out, diag

    def predict(self, scenarios: pd.DataFrame):
        l = self.predict_log(scenarios)
        n = len(self.frequencies_hz)
        return np.exp(l[:, :n]), np.exp(l[:, n : 2*n]), np.exp(l[:, 2*n :])

    def _groups_for_rows(self, df: pd.DataFrame):
        if "event_id" in df.columns:
            ids = df["event_id"].astype(str).fillna("").to_numpy()
            if np.any(np.char.str_len(ids.astype(str)) > 0):
                return ids
        # A multi-station call without event_id is normally one scenario event.
        return np.array(["__scenario_event__"] * len(df), dtype=object)

    @staticmethod
    def _eval_residual_curves(
        df: pd.DataFrame,
        freq: np.ndarray,
        spline: SplineTransformer,
        mode_coef: np.ndarray,
        mode_std: np.ndarray,
        rng: np.random.Generator,
        n_realizations: int,
    ) -> np.ndarray:
        n = len(df)
        k = len(mode_std)
        q = rng.normal(size=(n_realizations, n, k)) * mode_std[None, None, :]
        coeff = np.einsum("rnk,kb->rnb", q, mode_coef)
        out = np.zeros((n_realizations, n, len(freq)), float)
        ts = pd.to_numeric(df["Ts_s"], errors="coerce").to_numpy(float)
        for i in range(n):
            b = spline.transform(np.log(ts[i] * freq)[:, None])
            out[:, i, :] = coeff[:, i, :] @ b.T
        return out

    def sample_fas(
        self,
        mean_model,
        scenarios: pd.DataFrame,
        n_realizations: int,
        seed: int | None = None,
        residual_scale: float = 1.0,
        shared_event_effect: bool = True,
        enforce_support: bool = True,
        event_sampling_mode: str = "independent_realization",
    ):
        del mean_model, shared_event_effect, enforce_support  # unused runtime parameters
        df = scenarios.copy().reset_index(drop=True)
        rng = np.random.default_rng(seed)
        loge0, delta0, vh0 = self.predict_latent(df)
        nr, n, nf = int(n_realizations), len(df), len(self.frequencies_hz)
        loge = np.repeat(loge0[None, :, :], nr, axis=0)
        delta = np.repeat(delta0[None, :, :], nr, axis=0)
        logvh = np.repeat(vh0[None, :, :], nr, axis=0)
        scale = float(residual_scale)

        mode = str(event_sampling_mode or "independent_realization").strip().lower()
        if mode not in {"independent_realization", "shared_batch", "record_only"}:
            raise ValueError("event_sampling_mode must be independent_realization, shared_batch, or record_only")

        # Smooth event random field in absolute frequency f.
        if mode != "record_only" and self.event_modes.size:
            row_groups = self._groups_for_rows(df)
            unique_groups = list(dict.fromkeys(row_groups.tolist()))
            if mode == "shared_batch":
                z0 = rng.normal(size=(len(unique_groups), len(self.event_std))) * self.event_std[None, :]
                for gi, g in enumerate(unique_groups):
                    curve = (z0[gi] @ self.event_modes) * scale
                    mask = row_groups == g
                    loge[:, mask, :] += curve[None, None, :]
            else:
                for r in range(nr):
                    z = rng.normal(size=(len(unique_groups), len(self.event_std))) * self.event_std[None, :]
                    for gi, g in enumerate(unique_groups):
                        curve = (z[gi] @ self.event_modes) * scale
                        mask = row_groups == g
                        loge[r, mask, :] += curve[None, :]

        # Smooth intraevent/site-record field in u=f*Ts.
        if self.record_modes_coef.size:
            loge += scale * self._eval_residual_curves(
                df, self.frequencies_hz, self.record_spline,
                self.record_modes_coef, self.record_std, rng, nr
            )

        # Orientation and V/EAS uncertainty are also smooth functional fields.
        if self.delta_residual_modes_coef.size:
            delta += scale * self._eval_residual_curves(
                df, self.frequencies_hz, self.delta_residual_spline,
                self.delta_residual_modes_coef, self.delta_residual_std, rng, nr
            )
        if self.vh_residual_modes_coef.size:
            logvh += scale * self._eval_residual_curves(
                df, self.frequencies_hz, self.vh_residual_spline,
                self.vh_residual_modes_coef, self.vh_residual_std, rng, nr
            )

        li = loge + 0.5 * (np.log(2.0) - np.logaddexp(2.0 * delta, 0.0))
        lm = li + delta
        lv = loge + logvh
        return {
            "frequencies_hz": self.frequencies_hz.copy(),
            "major": np.exp(lm),
            "intermediate": np.exp(li),
            "vertical": np.exp(lv),
            "event_sampling_mode": mode,
            "architecture": "functional_hierarchical_release",
            "no_amax_normalization": True,
            "no_frequencywise_clipping": True,
        }
