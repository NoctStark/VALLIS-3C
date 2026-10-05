"""3C_release source-conditioned FAS model.

Author: Joel D. Cruz-Arguelles
Instituto de Ingeniería, Universidad Nacional Autónoma de México,
Coyoacán, 04510 Ciudad de México, México
JCruzAr@iingen.unam.mx, joeldan.cruz@gmail.com
2026
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd

from .fas_carrier import FASCarrierModel, _source_norm
from .fas_functional_hierarchical import _reconstruct_logs

ARCHITECTURE_ID = "gm_parametric_3c.fas.vallis3c"
SCHEMA_VERSION = "1.6.0"
CU_LAT_DEG = 19.33
CU_LON_DEG = -99.181


def _matern52(dist: np.ndarray) -> np.ndarray:
    q = np.sqrt(5.0) * np.asarray(dist, float)
    return (1.0 + q + q * q / 3.0) * np.exp(-q)


def _garcia04_omega2_prior(mw: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Regional INSLAB source normalization fixed from Garcia et al. (2004).

    Relative to Mw=7, so the absolute source constant cancels. M0 is dyne-cm.
    """
    mw = np.asarray(mw, float)
    f = np.asarray(freq, float)
    m0 = 10.0 ** (1.5 * mw + 16.1)
    m07 = 10.0 ** (1.5 * 7.0 + 16.1)
    fc = 1.956e7 * m0 ** (-0.297)
    fc7 = 1.956e7 * m07 ** (-0.297)
    return (
        np.log(m0 / m07)[:, None]
        - np.log1p((f[None, :] / fc[:, None]) ** 2)
        + np.log1p((f / fc7) ** 2)
    )


def _bearing_to_cu(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.deg2rad(np.asarray(lat_deg, float))
    lon = np.deg2rad(np.asarray(lon_deg, float))
    clat = np.deg2rad(CU_LAT_DEG)
    clon = np.deg2rad(CU_LON_DEG)
    dl = lon - clon
    return np.arctan2(
        np.sin(dl) * np.cos(lat),
        np.cos(clat) * np.sin(lat) - np.sin(clat) * np.cos(lat) * np.cos(dl),
    )


def _theta_for_scenarios(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n = len(df)
    theta = np.zeros(n, float)
    known = np.zeros(n, bool)
    if "event_latitude" in df and "event_longitude" in df:
        la = pd.to_numeric(df["event_latitude"], errors="coerce").to_numpy(float)
        lo = pd.to_numeric(df["event_longitude"], errors="coerce").to_numpy(float)
        m = np.isfinite(la) & np.isfinite(lo)
        theta[m] = _bearing_to_cu(la[m], lo[m]); known[m] = True
    # An explicitly resolved path angle must override event coordinates.
    if "source_azimuth_deg" in df:
        v = pd.to_numeric(df["source_azimuth_deg"], errors="coerce").to_numpy(float)
        m = np.isfinite(v)
        theta[m] = np.deg2rad(v[m]); known[m] = True
    return theta, known


def _prior(name: str, mw: np.ndarray, freq: np.ndarray) -> np.ndarray:
    if str(name).lower() in {"none", "interplate_none"}:
        return np.zeros((len(mw), len(freq)), float)
    if str(name).lower() in {"garcia04_omega2", "garcia2004_omega2"}:
        return _garcia04_omega2_prior(mw, freq)
    raise ValueError(f"Unknown VALLIS-3C source prior: {name}")


def _base_features(mw: np.ndarray, rr: np.ndarray) -> np.ndarray:
    mw = np.asarray(mw, float); rr = np.asarray(rr, float)
    return np.column_stack([mw - 7.0, np.log(np.maximum(rr, 1e-6) / 300.0), (rr - 300.0) / 100.0])


def _cross_kernels(model: dict[str, Any], mw: np.ndarray, rr: np.ndarray, zz: np.ndarray,
                   theta: np.ndarray, theta_known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mt = np.asarray(model["train_Mw"], float)
    rt = np.asarray(model["train_Rrup_km"], float)
    zt = np.asarray(model["train_Ztor_km"], float)
    tt = np.asarray(model["train_theta_rad"], float)
    cfg = model["kernel"]
    bnew = _base_features(mw, rr)
    btr = _base_features(mt, rt)
    ridge = float(cfg["linear_ridge_scale"])
    k = (bnew @ btr.T) / ridge
    dist = np.sqrt(
        ((np.asarray(mw)[:, None] - mt[None, :]) / float(cfg["l_Mw"])) ** 2
        + (np.log(np.asarray(rr)[:, None] / rt[None, :]) / float(cfg["l_logR"])) ** 2
        + ((np.asarray(zz)[:, None] - zt[None, :]) / float(cfg["l_Ztor_km"])) ** 2
    )
    k += float(cfg["var_local"]) * _matern52(dist)
    kp = np.zeros_like(k)
    if np.any(theta_known):
        aa = 1.0 / float(cfg["l_theta_rad"]) ** 2
        # Centered periodic kernel used by the VALLIS-3C architecture.
        # The Bessel-centering constant is stored to keep runtime scipy-free.
        center = float(cfg["periodic_center"])
        denom = float(cfg["periodic_denom"])
        dt = np.asarray(theta)[:, None] - tt[None, :]
        circ = (np.exp(-aa * (1.0 - np.cos(dt))) - center) / denom
        kp = float(cfg["var_theta"]) * (np.asarray(rr)[:, None] * rt[None, :] / 300.0**2) * circ
        kp[~np.asarray(theta_known, bool), :] = 0.0
    return k, kp


def predict_cu_eas(model: dict[str, Any], df: pd.DataFrame, freq: np.ndarray) -> np.ndarray:
    mw = pd.to_numeric(df["Mw"], errors="coerce").to_numpy(float)
    rr = pd.to_numeric(df["Rrup_km"], errors="coerce").to_numpy(float)
    zcol = "Ztor_km" if "Ztor_km" in df else "depth_used_km"
    zz = pd.to_numeric(df[zcol], errors="coerce").to_numpy(float)
    theta, known = _theta_for_scenarios(df)
    mu = _prior(model["prior"], mw, freq)
    k, kp = _cross_kernels(model, mw, rr, zz, theta, known)
    perp = np.asarray(model["perp_intercept"], float)[None, :] + k @ np.asarray(model["perp_alpha"], float)
    amp = float(model["amp_intercept"]) + (k + kp) @ np.asarray(model["amp_alpha"], float)
    phi = np.asarray(model["phi"], float)
    return mu + perp + amp[:, None] * phi[None, :]


@dataclass
class FASSpectralModel(FASCarrierModel):
    """Production 3C_release artifact.

    The 3C stochastic layers are self-contained, and the CU EAS conditional
    center uses the source-conditioned ML architecture.
    """
    cu_models: dict[str, Any] = field(default_factory=dict)
    # Packaged artifacts embed the site texture so mean-site and local-site layers remain aligned.
    site_texture_model: Any = None
    artifact_metadata: dict[str, Any] = field(default_factory=dict)
    architecture_id: str = ARCHITECTURE_ID
    schema_version: str = SCHEMA_VERSION
    model_version: str = "1.6.0"

    def _source_backbone(self, df: pd.DataFrame) -> np.ndarray:
        src = _source_norm(df["source_type"])
        out = np.zeros((len(df), len(self.frequencies_hz)), float)
        for name in ("INTERPLATE", "INSLAB"):
            ix = np.where(src == name)[0]
            if len(ix):
                out[ix] = predict_cu_eas(self.cu_models[name], df.iloc[ix].reset_index(drop=True), self.frequencies_hz)
        return out

    def _source_event_bias(self, df: pd.DataFrame) -> np.ndarray:
        # The source-specific GP predictor already contains its fitted intercept.
        return np.zeros((len(df), len(self.frequencies_hz)), float)

    def predict_log(self, scenarios: pd.DataFrame, return_diagnostics: bool = False):
        df = scenarios.copy().reset_index(drop=True)
        loge, delta, logvh = self.predict_latent(df)
        lm, li, lv = _reconstruct_logs(loge, delta, logvh)
        out = np.concatenate([lm, li, lv], axis=1)
        if not return_diagnostics:
            return out
        theta, known = _theta_for_scenarios(df)
        return out, {
            "architecture": ARCHITECTURE_ID,
            "schema_version": SCHEMA_VERSION,
            "model_version": self.model_version,
            "horizontal_definition": "EAS=sqrt((Major^2+Intermediate^2)/2)",
            "eas_center": "source-conditioned Matern-5/2 residual + periodic path term",
            "phi": "sqrt(f)/sqrt(mean(f[0.1,10]))",
            "angular_input_available": known.tolist(),
            "angular_missing_policy": "conditional mean of centered periodic field = 0",
            "source_priors": {k: str(v.get("prior")) for k, v in self.cu_models.items()},
            "artifact_metadata": dict(self.artifact_metadata),
            "metadata_policy": dict(self.metadata_policy),
        }
