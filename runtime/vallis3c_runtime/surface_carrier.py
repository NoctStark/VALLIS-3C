"""Spectral surface carrier used by VALLIS-3C 1.6.0.

This module contains the FAS realization logic used by the VALLIS-3C 1.6.0
architecture.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

COMPONENTS=("major","intermediate","vertical")

class SurfaceCarrierEngine:
    """FAS engine used by the phase-diffusion synthesis runtime."""

    _MODULE_CODES={"fas":101,"texture":102}

    def __init__(self, spectral_model, empirical_fas_asset: str | Path, *, variant: str="full"):
        self.fas_mean=spectral_model
        self.fas_residual=spectral_model
        self.resonance_texture_model=getattr(spectral_model,"site_texture_model",None)
        self.empirical_fas_asset=Path(empirical_fas_asset)
        self.variant=str(variant)
        self.active_fas_architecture="VALLIS3C"
        self._empirical=None

    def _seed_for(self,seed,module,realization=0,scenario=0,component=0,retry=0):
        if seed is None:return None
        code=int(self._MODULE_CODES.get(str(module),999))
        sequence=np.random.SeedSequence([
            int(seed),2100,code,int(realization),int(scenario),int(component),int(retry)
        ])
        return int(sequence.generate_state(1,dtype=np.uint32)[0])

    @staticmethod
    def _as_frame(scenarios):
        if isinstance(scenarios,pd.DataFrame):return scenarios.copy().reset_index(drop=True)
        if isinstance(scenarios,pd.Series):return pd.DataFrame([scenarios]).reset_index(drop=True)
        if isinstance(scenarios,dict):return pd.DataFrame([scenarios])
        return pd.DataFrame(scenarios).reset_index(drop=True)

    @staticmethod
    def _check_scenarios(frame,require_xy=True):
        required=("Mw","Rrup_km","depth_used_km","Ts_s","source_type")
        if require_xy:required+=("x_conditioned_km","y_conditioned_km")
        missing=[name for name in required if name not in frame]
        if missing:raise ValueError(f"Missing scenario columns: {missing}")
        numeric=("Mw","Rrup_km","depth_used_km","Ts_s")
        if require_xy:numeric+=("x_conditioned_km","y_conditioned_km")
        for name in numeric:
            values=pd.to_numeric(frame[name],errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():raise ValueError(f"Non-finite scenario feature {name}")
        if (pd.to_numeric(frame.Rrup_km)<=0).any() or (pd.to_numeric(frame.Ts_s)<=0).any():
            raise ValueError("Rrup_km and Ts_s must be positive")

    def _select_fas_pair(self,mode="default"):
        mode=str(mode or "default").strip().lower()
        if mode not in {"default","vallis3c","release"}:
            raise ValueError("VALLIS-3C 1.6.0 exposes only the installed release FAS model")
        return self.fas_mean,self.fas_residual,"vallis3c_release"

    def predict_mean_fas(self,scenarios,fas_mean_mode="default",fas_site_mode="ts_only"):
        frame=self._as_frame(scenarios);self._check_scenarios(frame,require_xy=False)
        frame["_fas_site_mode"]=str(fas_site_mode)
        mean,_,resolved=self._select_fas_pair(fas_mean_mode)
        log_fas,diagnostics=mean.predict_log(frame,return_diagnostics=True)
        nfreq=len(mean.frequencies_hz)
        major=np.exp(log_fas[:,:nfreq]);intermediate=np.exp(log_fas[:,nfreq:2*nfreq]);vertical=np.exp(log_fas[:,2*nfreq:])
        frequency=mean.frequencies_hz.copy()
        if hasattr(mean,"extend_for_synthesis"):
            frequency,major,intermediate,vertical=mean.extend_for_synthesis(major,intermediate,vertical)
        diagnostics=dict(diagnostics or {})
        diagnostics["fas_mean_mode"]=resolved
        diagnostics["synthesis_frequency_range_hz"]=[float(frequency[0]),float(frequency[-1])]
        return {
            "frequencies_hz":frequency,"major":major,"intermediate":intermediate,"vertical":vertical,
            "diagnostics":diagnostics,
        }

    @staticmethod
    def _prepare_exact_fas(exact_fas,n_scenarios,n_realizations):
        if not isinstance(exact_fas,dict):
            raise ValueError("exact_fas must contain frequencies_hz, major, intermediate and vertical")
        frequency=np.asarray(exact_fas.get("frequencies_hz",exact_fas.get("freq",[])),float).ravel()
        if len(frequency)<8 or not np.isfinite(frequency).all():raise ValueError("Invalid exact FAS frequency grid")
        keep=frequency>0;frequency=frequency[keep]
        if len(frequency)<8:raise ValueError("Invalid exact FAS frequency grid")
        order=np.argsort(frequency);frequency=frequency[order]
        result={"frequencies_hz":frequency}
        for name in COMPONENTS:
            values=np.asarray(exact_fas[name],float)
            if values.ndim==1:values=values[None,:]
            if values.ndim!=2 or values.shape[1]!=len(keep):
                raise ValueError(f"Invalid exact FAS shape for {name}")
            values=values[:,keep][:,order]
            if values.shape[0]==1 and int(n_scenarios)>1:
                values=np.repeat(values,int(n_scenarios),axis=0)
            if values.shape[0]!=int(n_scenarios):raise ValueError(f"Exact FAS scenario count mismatch for {name}")
            values=np.maximum(np.nan_to_num(values,nan=1e-18,posinf=1e-18,neginf=1e-18),1e-18)
            result[name]=np.repeat(values[None,:,:],int(n_realizations),axis=0)
        result.update(event_sampling_mode="exact_event",fas_mean_mode="exact_event",fas_input_mode="exact_event",resonance_texture_enabled=False)
        return result

    def sample_parameters(self,scenarios,n_realizations=1,seed=None,fas_residual=True,
                          fas_resonance_texture=True,fas_resonance_texture_residual=True,
                          fas_resonance_texture_site_strength=1.0,
                          fas_resonance_texture_residual_scale=1.0,
                          fas_residual_scale=1.0,enforce_support=True,
                          fas_event_mode="independent_realization",fas_mean_mode="default",
                          fas_site_mode="ts_only",fas_input_mode="model",exact_fas=None,
                          fas_event_seed=None,**_):
        frame=self._as_frame(scenarios);self._check_scenarios(frame)
        frame["_fas_site_mode"]=str(fas_site_mode)
        input_mode=str(fas_input_mode or "model").strip().lower()
        if input_mode not in {"model","exact_event"}:raise ValueError("fas_input_mode must be model or exact_event")
        if input_mode=="exact_event":
            fas=self._prepare_exact_fas(exact_fas,len(frame),n_realizations)
            resolved="exact_event"
        else:
            mean,residual,resolved=self._select_fas_pair(fas_mean_mode)
            if fas_residual:
                fas=residual.sample_fas(
                    mean,frame,int(n_realizations),seed=self._seed_for(seed,"fas"),
                    residual_scale=float(fas_residual_scale),shared_event_effect=True,
                    enforce_support=bool(enforce_support),event_sampling_mode=str(fas_event_mode),
                    event_seed=(int(fas_event_seed) if fas_event_seed is not None else None),
                )
            else:
                major,intermediate,vertical=mean.predict(frame)
                fas={
                    "frequencies_hz":mean.frequencies_hz.copy(),
                    "major":np.repeat(major[None,:,:],int(n_realizations),axis=0),
                    "intermediate":np.repeat(intermediate[None,:,:],int(n_realizations),axis=0),
                    "vertical":np.repeat(vertical[None,:,:],int(n_realizations),axis=0),
                    "event_sampling_mode":str(fas_event_mode),
                }
            if hasattr(mean,"extend_samples_for_synthesis"):
                frequency,major,intermediate,vertical=mean.extend_samples_for_synthesis(
                    frame,fas["major"],fas["intermediate"],fas["vertical"]
                )
                fas.update(frequencies_hz=frequency,major=major,intermediate=intermediate,vertical=vertical,synthesis_band_extended=True)
            elif hasattr(mean,"extend_for_synthesis"):
                frequency,major,intermediate,vertical=mean.extend_for_synthesis(
                    fas["major"],fas["intermediate"],fas["vertical"]
                )
                fas.update(frequencies_hz=frequency,major=major,intermediate=intermediate,vertical=vertical,synthesis_band_extended=True)
            fas["resonance_texture_enabled"]=False
            if fas_resonance_texture:
                texture=self.resonance_texture_model
                if texture is None:raise RuntimeError("The embedded release site texture is missing")
                applied=texture.apply(
                    fas["frequencies_hz"],fas["major"],fas["intermediate"],frame,
                    n_realizations=int(n_realizations),seed=self._seed_for(seed,"texture"),
                    site_strength=float(fas_resonance_texture_site_strength),
                    residual_enabled=bool(fas_resonance_texture_residual),
                    residual_scale=float(fas_resonance_texture_residual_scale),
                )
                fas["major"]=applied["major"];fas["intermediate"]=applied["intermediate"]
                fas["resonance_texture_enabled"]=True
                fas["resonance_texture_diagnostics"]=applied["diagnostics"]
                fas["resonance_texture_model_version"]=applied["model_version"]
        fas["fas_mean_mode"]=resolved;fas["fas_input_mode"]=input_mode
        return {"scenarios":frame,"fas":fas}

    @property
    def empirical(self):
        if self._empirical is None:
            from .empirical_fas import EmpiricalFASLibrary
            self._empirical=EmpiricalFASLibrary(self.empirical_fas_asset)
        return self._empirical

    def capabilities(self):
        metadata=dict(getattr(self.fas_mean,"artifact_metadata",{}) or {})
        report=dict(getattr(self.fas_mean,"training_report",{}) or {})
        exclusions=report.get("excluded_event_ids",metadata.get("excluded_event_ids",metadata.get("excluded_events",[]))) or []
        return {
            "package_version":"1.6.0",
            "model_family":"vallis3c.surface_carrier",
            "active_fas_architecture":"VALLIS3C",
            "bundle_variant":self.variant,
            "runtime_role":"FAS surface carrier only",
            "training_exclusions":sorted(map(str,exclusions)),
            "components":["Major","Intermediate","Vertical"],
        }

    @staticmethod
    def _resolve_variability_source(kwargs:dict,input_mode:str)->str:
        source=str(kwargs.get("fas_variability_source","surface_hybrid") or "surface_hybrid").strip().lower()
        allowed={"surface_hybrid","stochastic_surface","none"}
        if source not in allowed:
            raise ValueError(f"fas_variability_source must be one of {sorted(allowed)}")
        if input_mode=="exact_event" and not bool(kwargs.get("city_exact_rock",False)):
            source="none"
        return source

    def _fas_targets(self,s,n_realizations,seed,kwargs,chosen_method=None):
        input_mode=str(kwargs.get("fas_input_mode","model") or "model").strip().lower()
        source=self._resolve_variability_source(kwargs,input_mode)
        if input_mode not in {"model","exact_event"}:
            raise ValueError("fas_input_mode must be model or exact_event")

        exact=kwargs.get("exact_fas")
        if input_mode=="exact_event" and isinstance(exact,dict) and any(
            np.asarray(exact.get(c,[])).ndim==3 for c in COMPONENTS
        ):
            freq=np.asarray(exact.get("frequencies_hz",exact.get("freq",[])),float).ravel()
            if len(freq)<8 or not np.isfinite(freq).all() or np.any(freq<=0):
                raise ValueError("exact_fas frequencies_hz is invalid")
            fs={"frequencies_hz":freq}
            for c in COMPONENTS:
                arr=np.asarray(exact[c],float)
                if arr.shape != (int(n_realizations),len(s),len(freq)):
                    raise ValueError(
                        f"exact_fas {c} must have shape {(int(n_realizations),len(s),len(freq))}, got {arr.shape}"
                    )
                fs[c]=np.maximum(np.nan_to_num(arr,nan=1e-18,posinf=1e-18,neginf=1e-18),1e-18)
            return fs,source,{
                "backend":"exact_surface_targets","source":source,
                "fas_input_mode":input_mode,"source_path_residual":False,
                "site_ft_realization_specific":True,
            }

        stochastic=(source in {"surface_hybrid","stochastic_surface"} and input_mode=="model")
        pars=self.sample_parameters(
            s,n_realizations=int(n_realizations),seed=seed,
            fas_residual=bool(kwargs.get("fas_residual",True)) if stochastic else False,
            fas_resonance_texture=bool(kwargs.get("fas_resonance_texture",True)) if stochastic else False,
            fas_resonance_texture_residual=bool(kwargs.get("fas_resonance_texture_residual",True)) if stochastic else False,
            fas_resonance_texture_site_strength=float(kwargs.get("fas_resonance_texture_site_strength",1.0)),
            fas_resonance_texture_residual_scale=float(kwargs.get("fas_resonance_texture_residual_scale",1.0)),
            temporal_residual=False,
            c2_constraint=False,
            fas_residual_scale=float(kwargs.get("fas_residual_scale",1.0)),
            enforce_support=True,
            fas_event_mode=str(kwargs.get("fas_event_mode","independent_realization")),
            fas_event_seed=kwargs.get("fas_event_seed"),
            fas_mean_mode=str(kwargs.get("fas_mean_mode","default")),
            fas_site_mode=str(kwargs.get("fas_site_mode","ts_only")),
            fas_input_mode=input_mode,
            exact_fas=kwargs.get("exact_fas"),
        )
        fs=pars["fas"]

        fs_without_local_ft=None
        if source=="surface_hybrid" and input_mode=="model" and bool(kwargs.get("fas_resonance_texture",True)):
            pars_without_local_ft=self.sample_parameters(
                s,n_realizations=int(n_realizations),seed=seed,
                fas_residual=bool(kwargs.get("fas_residual",True)),
                fas_resonance_texture=False,
                fas_resonance_texture_residual=False,
                fas_resonance_texture_site_strength=float(kwargs.get("fas_resonance_texture_site_strength",1.0)),
                fas_resonance_texture_residual_scale=float(kwargs.get("fas_resonance_texture_residual_scale",1.0)),
                temporal_residual=False,
                c2_constraint=False,
                fas_residual_scale=float(kwargs.get("fas_residual_scale",1.0)),
                enforce_support=True,
                fas_event_mode=str(kwargs.get("fas_event_mode","independent_realization")),
                fas_event_seed=kwargs.get("fas_event_seed"),
                fas_mean_mode=str(kwargs.get("fas_mean_mode","default")),
                fas_site_mode=str(kwargs.get("fas_site_mode","ts_only")),
                fas_input_mode=input_mode,
                exact_fas=kwargs.get("exact_fas"),
            )
            fs_without_local_ft=pars_without_local_ft["fas"]

        broad_audit=None
        if source=="surface_hybrid" and input_mode=="model":
            from scipy.ndimage import gaussian_filter1d
            mean=self.predict_mean_fas(
                s,fas_mean_mode=str(kwargs.get("fas_mean_mode","default")),
                fas_site_mode=str(kwargs.get("fas_site_mode","ts_only"))
            )
            f=np.asarray(fs["frequencies_hz"],float)
            dlog=float(np.median(np.diff(np.log10(np.maximum(f,1e-8)))))
            bw=float(kwargs.get("fas_split_decades",0.18))
            alpha=float(kwargs.get("fas_broad_strength",1.0))
            sig=max(bw/max(dlog,1e-6),0.5)
            for c in COMPONENTS:
                full=np.asarray(fs[c],float)
                stochastic_arr=np.asarray(
                    fs_without_local_ft[c] if fs_without_local_ft is not None else full,float
                )
                base=np.asarray(mean[c],float)[None,:,:]
                broad_log=gaussian_filter1d(
                    np.log(np.maximum(stochastic_arr,1e-20)/np.maximum(base,1e-20)),
                    sig,axis=-1,mode="nearest",
                )
                local_ft_log=np.log(
                    np.maximum(full,1e-20)/np.maximum(stochastic_arr,1e-20)
                )
                fs[c]=base*np.exp(alpha*broad_log+local_ft_log)
            broad_audit={
                "split_decades":bw,"broad_strength":alpha,
                "localized_ft_preserved":bool(fs_without_local_ft is not None),
                "definition":"surface mean + smooth broadband residual + exact local FT texture",
            }

        audit={
            "backend":"vallis_surface_sampler",
            "source":source,
            "fas_input_mode":input_mode,
            "fas_mean_mode":str(fs.get("fas_mean_mode",kwargs.get("fas_mean_mode","default"))),
            "fas_site_mode":str(kwargs.get("fas_site_mode","ts_only")),
            "fas_event_mode":str(fs.get("event_sampling_mode",kwargs.get("fas_event_mode","independent_realization"))),
            "fas_residual":bool(kwargs.get("fas_residual",True)) if stochastic else False,
            "fas_residual_scale":float(kwargs.get("fas_residual_scale",1.0)),
            "fas_resonance_texture":bool(fs.get("resonance_texture_enabled",False)) if stochastic else False,
            "fas_resonance_texture_residual":bool(kwargs.get("fas_resonance_texture_residual",True)) if stochastic else False,
            "fas_resonance_texture_site_strength":float(kwargs.get("fas_resonance_texture_site_strength",1.0)),
            "fas_resonance_texture_residual_scale":float(kwargs.get("fas_resonance_texture_residual_scale",1.0)),
            "broad_carrier":broad_audit,
        }
        return fs,source,audit
