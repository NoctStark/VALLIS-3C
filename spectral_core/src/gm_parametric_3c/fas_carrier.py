from __future__ import annotations


from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import SplineTransformer


from .fas_functional_hierarchical import (
    _predict_part,
    _reconstruct_logs,
    _restore_bspline_public_state,
    _restore_model_splines,
    _site_design,
)


ARCHITECTURE_ID = "gm_parametric_3c.fas.core"
SCHEMA_VERSION = "1.6.0"


# Source classification for the 68 CU OOF event residual curves used by 3C_release.
# Derived from the horizontal FAS training snapshot recorded by hash.
HISTORICAL_EVENT_SOURCE = {
    "1964-07-06": "INSLAB",
    "1965-08-23": "INTERPLATE",
    "1968-02-03": "INTERPLATE",
    "1968-08-02": "INTERPLATE",
    "1976-02-01": "INTERPLATE",
    "1976-06-07": "INTERPLATE",
    "1978-11-29": "INTERPLATE",
    "1979-03-14": "INTERPLATE",
    "1982-06-07": "INTERPLATE",
    "1985-09-19": "INTERPLATE",
    "1985-09-21": "INTERPLATE",
    "1986-04-30": "INTERPLATE",
    "1988-02-08": "INTERPLATE",
    "1989-04-25": "INTERPLATE",
    "1990-05-11": "INTERPLATE",
    "1990-05-31": "INTERPLATE",
    "1993-05-15": "INTERPLATE",
    "1993-08-05": "INSLAB",
    "1994-02-23": "INSLAB",
    "1994-05-06": "INSLAB",
    "1994-05-23": "INSLAB",
    "1994-12-10": "INSLAB",
    "1995-09-14": "INTERPLATE",
    "1996-07-15": "INTERPLATE",
    "1997-04-03": "INSLAB",
    "1997-05-22": "INSLAB",
    "1997-12-16": "INTERPLATE",
    "1998-02-03": "INTERPLATE",
    "1998-04-20": "INSLAB",
    "1998-07-11": "INTERPLATE",
    "1998-07-12": "INTERPLATE",
    "1999-06-15": "INSLAB",
    "1999-06-21": "INSLAB",
    "1999-09-30": "INSLAB",
    "2000-07-21": "INSLAB",
    "2000-08-09": "INTERPLATE",
    "2003-01-22": "INTERPLATE",
    "2003-11-19": "INSLAB",
    "2004-01-01": "INTERPLATE",
    "2006-02-20": "INSLAB",
    "2006-08-11": "INSLAB",
    "2007-04-13": "INSLAB",
    "2008-04-28": "INSLAB",
    "2009-01-31": "INTERPLATE",
    "2009-05-22": "INSLAB",
    "2010-06-30": "INTERPLATE",
    "2011-12-11": "INSLAB",
    "2012-03-20": "INTERPLATE",
    "2012-04-11": "INTERPLATE",
    "2012-11-15": "INSLAB",
    "2013-06-16": "INSLAB",
    "2013-08-21": "INTERPLATE",
    "2014-04-18": "INTERPLATE",
    "2014-05-08": "INTERPLATE",
    "2014-05-10": "INTERPLATE",
    "2014-05-24": "INTERPLATE",
    "2016-05-08": "INTERPLATE",
    "2016-06-27": "INTERPLATE",
    "2017-09-08": "INSLAB",
    "2017-09-19": "INSLAB",
    "2018-02-16": "INTERPLATE",
    "2018-02-17": "INTERPLATE",
    "2018-02-19": "INTERPLATE",
    "2020-06-23": "INTERPLATE",
    "2021-09-08": "INTERPLATE",
    "2022-09-19": "INTERPLATE",
    "2022-09-22": "INTERPLATE",
    "2022-12-11": "INTERPLATE"
}


def _source_norm(series: pd.Series) -> np.ndarray:
    s = series.astype(str).str.strip().str.upper()
    return np.where(s.str.contains("INSLAB") | s.str.contains("INTRA"), "INSLAB", "INTERPLATE")


def _source_design(df: pd.DataFrame, form: str) -> np.ndarray:
    mw = pd.to_numeric(df["Mw"], errors="coerce").to_numpy(float)
    rr = pd.to_numeric(df["Rrup_km"], errors="coerce").to_numpy(float)
    m = mw - 7.0
    lr = np.log(np.maximum(rr, 1.0) / 300.0)
    r = (rr - 300.0) / 100.0
    if form == "lin_lr":
        return np.column_stack([np.ones(len(df)), m, lr])
    if form == "lin_lr_r":
        return np.column_stack([np.ones(len(df)), m, lr, r])
    if form == "quad_lr":
        return np.column_stack([np.ones(len(df)), m, m * m, lr])
    if form == "quad_lr_r":
        return np.column_stack([np.ones(len(df)), m, m * m, lr, r])
    if form == "hinge7_lr_r":
        return np.column_stack([np.ones(len(df)), m, np.maximum(m, 0.0), lr, r])
    raise ValueError(f"Unknown source backbone form: {form}")


def _unit_t(rng: np.random.Generator, df: float, shape):
    if (not np.isfinite(df)) or df > 200:
        return rng.normal(size=shape)
    if df <= 2.05:
        df = 2.05
    z = rng.standard_t(df, size=shape)
    return z * np.sqrt((df - 2.0) / df)


@dataclass
class FASCarrierModel:
    """Unified self-contained three-component FAS architecture for VALLIS-3C 1.6.0.


Every learned FAS parameter lives in this object. Training constructs the core
    directly from the declared calibration data and configuration.


Horizontal mean:
      ln EAS = source-separated CU backbone + dynamic-Ts population site term
               + source-specific event bias


Persistent station/local resonance is intentionally NOT part of FAS core.
    It belongs to the independent resonance-texture component, which avoids
    double counting and matches the production ``ts_only`` FAS convention.


Component reconstruction:
      delta = ln(Major / Intermediate)
      logvh = ln(Vertical / EAS)


All source/path, site, event, record, orientation-ratio and V/H parameters
    are stored directly in this dataclass.
    """


# Frequency support.
    frequencies_hz: np.ndarray


# Source-separated CU reference backbone.
    backbone_beta_interplate: np.ndarray
    backbone_beta_intraslab: np.ndarray
    backbone_form_interplate: str
    backbone_form_intraslab: str


# Dynamic-Ts population site mean.
    site_spline: SplineTransformer
    site_coef: np.ndarray
    site_order: int
    site_tref: float


# Source-specific mean event-bias curves and stochastic event field.
    event_bias_interplate: np.ndarray
    event_bias_intraslab: np.ndarray
    event_modes: np.ndarray
    event_std: np.ndarray
    event_df: np.ndarray
    historical_event_curves: dict[str, np.ndarray]


# Intra-event record field in u=f*Ts.
    record_spline: SplineTransformer
    record_modes_coef: np.ndarray
    record_std: np.ndarray
    record_df: np.ndarray


# Major/intermediate orientation ratio, trained directly in core.
    delta_mean: dict[str, Any]
    delta_bias_curve: np.ndarray
    delta_residual_spline: SplineTransformer
    delta_residual_modes_coef: np.ndarray
    delta_residual_std: np.ndarray


# Vertical/EAS ratio, trained directly in core.
    vh_mean: dict[str, Any]
    vh_bias_curve: np.ndarray
    vh_residual_spline: SplineTransformer
    vh_residual_modes_coef: np.ndarray
    vh_residual_std: np.ndarray


    # Provenance / audit.
    metadata_policy: dict[str, Any] = field(default_factory=dict)
    training_report: dict[str, Any] = field(default_factory=dict)
    architecture_id: str = ARCHITECTURE_ID
    schema_version: str = SCHEMA_VERSION
    model_version: str = "1.6.0"


    @property
    def config(self):
        return self.training_report.get("config", {})


    def save(self, path: str | Path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p, compress=3)
        return p


    @classmethod
    def load(cls, path: str | Path):
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        _restore_bspline_public_state(obj.site_spline)
        _restore_model_splines(obj)
        return obj


    def write_report(self, path: str | Path):
        p = Path(path)
        p.write_text(json.dumps(self.training_report, indent=2, ensure_ascii=False), encoding="utf-8")
        return p


    def _site_mean(self, df: pd.DataFrame) -> np.ndarray:
        z = _site_design(
            df,
            self.frequencies_hz,
            self.site_spline,
            int(self.site_order),
            float(self.site_tref),
            dtype=np.float64,
        )
        return (z @ np.asarray(self.site_coef, float)).reshape(len(df), len(self.frequencies_hz))


    def _source_backbone(self, df: pd.DataFrame) -> np.ndarray:
        src = _source_norm(df["source_type"])
        out = np.zeros((len(df), len(self.frequencies_hz)), float)
        for name, beta, form in (
            ("INTERPLATE", self.backbone_beta_interplate, self.backbone_form_interplate),
            ("INSLAB", self.backbone_beta_intraslab, self.backbone_form_intraslab),
        ):
            ix = np.where(src == name)[0]
            if len(ix):
                out[ix] = _source_design(df.iloc[ix], form) @ np.asarray(beta, float)
        return out


    def _source_event_bias(self, df: pd.DataFrame) -> np.ndarray:
        src = _source_norm(df["source_type"])
        out = np.zeros((len(df), len(self.frequencies_hz)), float)
        out[src == "INTERPLATE"] = np.asarray(self.event_bias_interplate, float)
        out[src == "INSLAB"] = np.asarray(self.event_bias_intraslab, float)
        return out


    def predict_latent(self, scenarios: pd.DataFrame):
        df = scenarios.copy().reset_index(drop=True)
        loge = self._source_backbone(df)
        loge = loge + self._site_mean(df) + self._source_event_bias(df)
        delta = _predict_part(self.delta_mean, df, self.frequencies_hz) + self.delta_bias_curve[None, :]
        logvh = _predict_part(self.vh_mean, df, self.frequencies_hz) + self.vh_bias_curve[None, :]
        # core: source-specific HF shape correction learned from event-balanced
        # training residuals and anchored to exactly zero at the existing 5-Hz
        # architecture boundary. This is a data-derived mean-shape correction,
        # not a hand-tuned spectral tilt; <=5 Hz remains bitwise unchanged.
        corr=getattr(self,'hf_shape_corrections',None)
        if corr:
            src=_source_norm(df['source_type'])
            for name in ('INTERPLATE','INSLAB'):
                ix=np.where(src==name)[0]
                if len(ix) and name in corr:
                    cc=corr[name]
                    loge[ix]+=np.asarray(cc.get('loge',0.0),float)[None,:]
                    delta[ix]+=np.asarray(cc.get('delta',0.0),float)[None,:]
                    logvh[ix]+=np.asarray(cc.get('logvh',0.0),float)[None,:]
        return loge, delta, logvh


    def predict_log(self, scenarios: pd.DataFrame, return_diagnostics: bool = False):
        df = scenarios.copy().reset_index(drop=True)
        loge, delta, logvh = self.predict_latent(df)
        lm, li, lv = _reconstruct_logs(loge, delta, logvh)
        out = np.concatenate([lm, li, lv], axis=1)
        if not return_diagnostics:
            return out
        return out, {
            "architecture": ARCHITECTURE_ID,
            "schema_version": SCHEMA_VERSION,
            "horizontal_definition": "EAS=sqrt((Major^2+Intermediate^2)/2)",
            "backbone_reference": "CU",
            "source_backbones": {
                "INTERPLATE": self.backbone_form_interplate,
                "INSLAB": self.backbone_form_intraslab,
            },
            "metadata_policy": dict(self.metadata_policy),
            "site_identification": "within-event station/CU ratio after source-separated CU path correction",
            "site_coordinate": "u=f*Ts",
            "site_reference_Ts_s": float(self.site_tref),
            "event_coordinate": "absolute frequency f",
            "component_ratio_layers": "trained directly in core; no inherited FAS artifact",
            "self_contained_release_artifacts": True,
            "site_correction_mode": "dynamic_Ts_population_mean; persistent spatial resonance handled by texture_core",
        }


    def predict(self, scenarios: pd.DataFrame):
        l = self.predict_log(scenarios)
        n = len(self.frequencies_hz)
        return np.exp(l[:, :n]), np.exp(l[:, n : 2 * n]), np.exp(l[:, 2 * n :])


    @staticmethod
    def _groups(df: pd.DataFrame):
        if "event_id" in df.columns:
            ids = df["event_id"].astype(str).fillna("").to_numpy()
            if np.any(np.char.str_len(ids.astype(str)) > 0):
                return ids
        return np.array(["__scenario_event__"] * len(df), dtype=object)


    @staticmethod
    def _eval_record(df, freq, spline, modes, std, dfs, rng, nr):
        n = len(df)
        k = len(std)
        if k == 0:
            return np.zeros((nr, n, len(freq)), float)
        q = np.empty((nr, n, k), float)
        for j in range(k):
            q[..., j] = _unit_t(rng, float(dfs[j]), (nr, n)) * float(std[j])
        coeff = np.einsum("rnk,kb->rnb", q, modes)
        out = np.zeros((nr, n, len(freq)), float)
        ts = pd.to_numeric(df["Ts_s"], errors="coerce").to_numpy(float)
        for i in range(n):
            b = spline.transform(np.log(ts[i] * freq)[:, None])
            out[:, i, :] = coeff[:, i, :] @ b.T
        return out


    def _event_level_shape_library(self, source_type=None):
        """Empirical OOF event library decomposed into global level + shape.


PC1 of the fitted event field is sign-coherent across the complete
        frequency support, so it is used as the diagnostic broadband-level
        axis. Predictive sampling keeps each OOF level/shape pair together:
        this preserves the observed covariance and dependence while avoiding
        arbitrary combinations of higher PCA modes never observed as events.
        """
        items=list((self.historical_event_curves or {}).items())
        if source_type is not None:
            src=str(source_type).strip().upper()
            src_map=dict(getattr(self,"historical_event_sources",{}) or {})
            if not src_map:
                src_map=HISTORICAL_EVENT_SOURCE
            items=[(k,v) for k,v in items if str(src_map.get(str(k),"")).strip().upper()==src]
        if not items:
            return None
        ids=np.asarray([str(k) for k,_ in items],dtype=object)
        curves=np.stack([np.asarray(v,float) for _,v in items],axis=0)
        if self.event_modes.size:
            axis=np.asarray(self.event_modes[0],float).copy()
            norm=float(np.linalg.norm(axis))
            axis=axis/norm if norm>0 else np.ones(curves.shape[1],float)/np.sqrt(curves.shape[1])
        else:
            axis=np.ones(curves.shape[1],float)/np.sqrt(curves.shape[1])
        if float(np.mean(axis))<0:
            axis=-axis
        levels=curves@axis
        shapes=curves-levels[:,None]*axis[None,:]
        return ids,curves,axis,levels,shapes


    def _draw_empirical_event_curve(self,rng,source_type=None):
        lib=self._event_level_shape_library(source_type=source_type)
        if lib is None:
            return None,None,None,None
        ids,curves,axis,levels,shapes=lib
        j=int(rng.integers(0,len(ids)))
        curve=curves[j].copy()
        level=float(levels[j])
        shape=shapes[j]
        shape_rms=float(np.sqrt(np.mean(shape*shape)))
        return curve,str(ids[j]),level,shape_rms


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
        event_seed: int | None = None,
    ):
        del mean_model, shared_event_effect, enforce_support
        df = scenarios.copy().reset_index(drop=True)
        rng = np.random.default_rng(seed)
        # Optional independent RNG namespace for the between-event FAS field.
        # Without event_seed, use the inactive main RNG. With event_seed, predictive
        # realization r can be synchronized across sites while record/local streams
        # remain controlled by the ordinary station-specific seed.
        event_rng = rng if event_seed is None else np.random.default_rng(int(event_seed))
        loge0, delta0, vh0 = self.predict_latent(df)
        nr, n = int(n_realizations), len(df)
        loge = np.repeat(loge0[None, :, :], nr, axis=0)
        delta = np.repeat(delta0[None, :, :], nr, axis=0)
        logvh = np.repeat(vh0[None, :, :], nr, axis=0)
        scale = float(residual_scale)
        mode = str(event_sampling_mode or "independent_realization").strip().lower()
        allowed = {"independent_realization", "shared_batch", "record_only", "historical_conditioned"}
        if mode not in allowed:
            raise ValueError(f"event_sampling_mode must be one of {sorted(allowed)}")


        groups = self._groups(df)
        unique = list(dict.fromkeys(groups.tolist()))
        src_all=_source_norm(df["source_type"])
        group_sources=[]
        for g in unique:
            vals=np.unique(src_all[groups==g])
            if len(vals)!=1:
                raise ValueError(f"Each event group must have one source_type; event_id={g!r} has {vals.tolist()}")
            group_sources.append(str(vals[0]))
        event_donors=[[None for _ in unique] for _ in range(nr)]
        event_levels=np.zeros((nr,len(unique)),float)
        event_shape_rms=np.zeros((nr,len(unique)),float)
        event_sampler="record_only"
        if mode == "historical_conditioned":
            event_sampler="historical_oof_exact"
            lib=self._event_level_shape_library()
            axis=(lib[2] if lib is not None else None)
            for gi,g in enumerate(unique):
                if g not in self.historical_event_curves:
                    raise ValueError(f"No OOF historical event effect stored for event_id={g!r}")
                curve=np.asarray(self.historical_event_curves[g],float)
                applied=curve*scale
                loge[:,groups==g,:]+=applied[None,None,:]
                if axis is not None:
                    level=float(curve@axis)
                    shape=curve-level*axis
                    srms=float(np.sqrt(np.mean(shape*shape)))
                    for rr in range(nr):
                        event_donors[rr][gi]=str(g)
                        event_levels[rr,gi]=level*scale
                        event_shape_rms[rr,gi]=srms*abs(scale)
        elif mode != "record_only":
            if self.historical_event_curves:
                event_sampler="empirical_oof_level_shape_bootstrap_release_source_conditioned"
                if mode == "shared_batch":
                    for gi,g in enumerate(unique):
                        curve,donor,level,srms=self._draw_empirical_event_curve(event_rng,source_type=group_sources[gi])
                        applied=np.asarray(curve,float)*scale
                        loge[:,groups==g,:]+=applied[None,None,:]
                        for rr in range(nr):
                            event_donors[rr][gi]=donor
                            event_levels[rr,gi]=float(level)*scale
                            event_shape_rms[rr,gi]=float(srms)*abs(scale)
                else:
                    for rr in range(nr):
                        for gi,g in enumerate(unique):
                            curve,donor,level,srms=self._draw_empirical_event_curve(event_rng,source_type=group_sources[gi])
                            applied=np.asarray(curve,float)*scale
                            loge[rr,groups==g,:]+=applied[None,:]
                            event_donors[rr][gi]=donor
                            event_levels[rr,gi]=float(level)*scale
                            event_shape_rms[rr,gi]=float(srms)*abs(scale)
            elif self.event_modes.size:
                event_sampler="parametric_pca_fallback"
                if mode == "shared_batch":
                    z=np.empty((len(unique),len(self.event_std)),float)
                    for j in range(len(self.event_std)):
                        z[:,j]=_unit_t(event_rng,float(self.event_df[j]),len(unique))*float(self.event_std[j])
                    for gi,g in enumerate(unique):
                        curve=(z[gi]@self.event_modes)*scale
                        loge[:,groups==g,:]+=curve[None,None,:]
                else:
                    for rr in range(nr):
                        z=np.empty((len(unique),len(self.event_std)),float)
                        for j in range(len(self.event_std)):
                            z[:,j]=_unit_t(event_rng,float(self.event_df[j]),len(unique))*float(self.event_std[j])
                        for gi,g in enumerate(unique):
                            curve=(z[gi]@self.event_modes)*scale
                            loge[rr,groups==g,:]+=curve[None,:]


        loge += scale * self._eval_record(
            df,
            self.frequencies_hz,
            self.record_spline,
            self.record_modes_coef,
            self.record_std,
            self.record_df,
            rng,
            nr,
        )


        if self.delta_residual_modes_coef.size:
            k = len(self.delta_residual_std)
            q = rng.normal(size=(nr, n, k)) * self.delta_residual_std[None, None, :]
            coeff = np.einsum("rnk,kb->rnb", q, self.delta_residual_modes_coef)
            ts = pd.to_numeric(df["Ts_s"], errors="coerce").to_numpy(float)
            for i in range(n):
                b = self.delta_residual_spline.transform(np.log(ts[i] * self.frequencies_hz)[:, None])
                delta[:, i, :] += scale * (coeff[:, i, :] @ b.T)
        if self.vh_residual_modes_coef.size:
            k = len(self.vh_residual_std)
            q = rng.normal(size=(nr, n, k)) * self.vh_residual_std[None, None, :]
            coeff = np.einsum("rnk,kb->rnb", q, self.vh_residual_modes_coef)
            ts = pd.to_numeric(df["Ts_s"], errors="coerce").to_numpy(float)
            for i in range(n):
                b = self.vh_residual_spline.transform(np.log(ts[i] * self.frequencies_hz)[:, None])
                logvh[:, i, :] += scale * (coeff[:, i, :] @ b.T)


        li = loge + 0.5 * (np.log(2.0) - np.logaddexp(2.0 * delta, 0.0))
        lm = li + delta
        lv = loge + logvh
        return {
            "frequencies_hz": self.frequencies_hz.copy(),
            "major": np.exp(lm),
            "intermediate": np.exp(li),
            "vertical": np.exp(lv),
            "event_sampling_mode": mode,
            "event_seed": (int(event_seed) if (event_seed is not None and mode in {"independent_realization", "shared_batch"}) else None),
            "event_seed_namespace": ("separate_between_event_rng" if event_seed is not None else "main_rng"),
            "predictive_event_coupling": ("realization_index_shared_when_event_seed_reused_across_sites" if mode == "independent_realization" and event_seed is not None else None),
            "event_sampling_architecture": event_sampler,
            "event_group_ids": [str(x) for x in unique],
            "event_group_source_types": group_sources,
            "event_donor_event_ids": event_donors,
            "event_level_scores": event_levels,
            "event_shape_rms": event_shape_rms,
            "architecture": ARCHITECTURE_ID,
            "schema_version": SCHEMA_VERSION,
            "self_contained_release_artifacts": True,
        }
