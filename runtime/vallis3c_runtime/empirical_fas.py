from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence
import math
import re
import numpy as np
import pandas as pd

COMPONENTS=("major","intermediate","vertical")
_META_COLS=("event_id","station_id","Mw","Rrup_km","depth_used_km","Ts_s","source_type","event_longitude","event_latitude","x_conditioned_km","y_conditioned_km","texture_usable_3c","record_observed","texture_hdf5")


def _safe_float(x, default=np.nan):
    try:
        x=float(x)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _scenario_value(s: Mapping[str,Any], *names, default=np.nan):
    for n in names:
        if n in s:
            v=_safe_float(s[n],np.nan)
            if np.isfinite(v): return v
    return default


class EmpiricalFASLibrary:
    """Real-data residual library for the VALLIS-3C 1.6.0 FAS carrier.

    FAS residuals are observed/smooth log ratios from real records. For each
    candidate donor set they are mean-corrected so the candidate ensemble mean
    multiplier is one at every frequency.
    """

    def __init__(self, runtime_data: str|Path):
        supplied=Path(runtime_data).resolve()
        self.asset_path=(supplied/"empirical_fas_library.npz") if supplied.is_dir() else supplied
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"VALLIS-3C empirical FAS asset not found: {self.asset_path}")
        with np.load(self.asset_path,allow_pickle=False) as archive:
            format_name=str(archive["format_name"].item())
            if format_name!="vallis3c.empirical_fas_library":
                raise ValueError(f"Unsupported empirical FAS asset format: {format_name}")
            self.index=pd.DataFrame({c:archive[f"index__{c}"] for c in _META_COLS})
            self.timing=pd.DataFrame({
                "event_id":archive["timing__event_id"],
                "station_id":archive["timing__station_id"],
                "t1_H_local_s":archive["timing__t1_H_local_s"],
            })
            rho_event=archive["rho__event_id"]
            rho_station=archive["rho__station_id"]
            rho_value=archive["rho__rho_envelope"]
        usable=(self.index["texture_usable_3c"]==True)&(self.index["record_observed"]==True)
        self.index=self.index.loc[usable].reset_index(drop=True)
        self.timing["event_id"]=self.timing["event_id"].astype(str); self.timing["station_id"]=self.timing["station_id"].astype(str)
        # Phase-texture records must have an actual finite timing origin. A
        # missing t1 must never be silently converted to 0 s:
        # that can create an artificial early onset, saturate the global time-scale
        # lower bound, and pile several transferred tpeak values near the onset.
        _t1=pd.to_numeric(self.timing["t1_H_local_s"],errors="coerce").to_numpy(float)
        self._timing_valid_pairs={
            (str(e),str(st)) for e,st,ok in zip(self.timing.event_id,self.timing.station_id,np.isfinite(_t1)) if ok
        }
        self._timing_valid_mask=np.array([
            (str(e),str(st)) in self._timing_valid_pairs
            for e,st in zip(self.index.event_id,self.index.station_id)
        ],dtype=bool)
        self._rho_lookup={
            (str(e),str(st)):float(r)
            for e,st,r in zip(rho_event,rho_station,rho_value)
            if np.isfinite(_safe_float(r))
        }
        self._prepare_distance_features()
        self._fas_cache=None
        self._donor_pool_cache={}
        self._fas_delta_cache={}

    def _prepare_distance_features(self):
        vals=[]
        for name in ("Mw","Rrup_km","depth_used_km","Ts_s","x_conditioned_km","y_conditioned_km"):
            v=pd.to_numeric(self.index[name],errors="coerce").to_numpy(float)
            if name=="Rrup_km": v=np.log1p(np.clip(v,0,None))
            if name=="Ts_s": v=np.log(np.clip(v,.05,None))
            vals.append(v)
        X=np.column_stack(vals)
        loc=np.nanmedian(X,axis=0)
        mad=np.nanmedian(np.abs(X-loc[None,:]),axis=0)*1.4826
        std=np.nanstd(X,axis=0)
        scale=np.where(np.isfinite(mad)&(mad>1e-8),mad,np.where(std>1e-8,std,1.0))
        self._loc=loc; self._scale=scale
        X=np.where(np.isfinite(X),X,loc[None,:])
        self._X=(X-loc[None,:])/scale[None,:]

    def _scenario_vector(self,s):
        raw=np.array([
            _scenario_value(s,"Mw","mw"),
            math.log1p(max(_scenario_value(s,"Rrup_km","rrup_km","R_km",default=0.0),0.0)),
            _scenario_value(s,"depth_used_km","depth_km","Hhyp_km","depth"),
            math.log(max(_scenario_value(s,"Ts_s","ts_s","Ts",default=.5),.05)),
            _scenario_value(s,"x_conditioned_km","x_km"),
            _scenario_value(s,"y_conditioned_km","y_km"),
        ],float)
        raw=np.where(np.isfinite(raw),raw,self._loc)
        return (raw-self._loc)/self._scale

    def select_donor(self, scenario, rng, *, k=30, source_match=True, exclude_same_event=False, require_fas=False, exclude_events=()):
        """Build the donor pool by event rounds, then apply the existing kernel weights.

        Whole three-component donor fields remain joint. Source/exclusion/availability
        constraints are applied BEFORE ranking; no output-duration target is used.
        """
        if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k must be a positive integer')
        q = self._scenario_vector(scenario)
        src = str(scenario.get('source_type', '')).upper()
        target = str(scenario.get('event_id', '')).strip()
        global_exclusions=tuple(sorted({str(e).strip() for e in (exclude_events or ()) if str(e).strip()}))
        key = ('event_round_robin_release', tuple(np.round(q, 8)), src, target, int(k), bool(source_match), bool(exclude_same_event), bool(require_fas), global_exclusions)
        cached = self._donor_pool_cache.get(key)
        if cached is None:
            weights = np.array([3.0, 1.5, 1.0, 1.5, 0.35, 0.35])
            dist = np.linalg.norm((self._X - q[None, :]) * weights, axis=1)
            allowed = np.isfinite(dist) & self._timing_valid_mask
            if source_match and src:
                allowed &= self.index.source_type.astype(str).str.upper().to_numpy() == src
            evs = self.index.event_id.astype(str).to_numpy()
            if exclude_same_event and target:
                allowed &= evs != target
            if global_exclusions:
                allowed &= ~np.isin(evs, np.asarray(global_exclusions,dtype=str))
            if require_fas:
                keys = self._fas_keys()
                allowed &= np.array([(str(e), str(s)) in keys for e, s in zip(self.index.event_id, self.index.station_id)])
            order = np.argsort(np.where(allowed, dist, np.inf))
            groups = {}
            for j in order:
                if np.isfinite(dist[j]) and allowed[j]:
                    groups.setdefault(evs[j], []).append(int(j))
            pool = []
            rank = 0
            while len(pool) < max(2 * k, k):
                group_round = sorted([g[rank] for g in groups.values() if len(g) > rank], key=lambda j: dist[j])
                if not group_round:
                    break
                pool.extend(group_round[:max(2 * k, k) - len(pool)])
                rank += 1
            if not pool:
                raise RuntimeError('No valid same-source phase-texture records')
            dp = dist[pool]
            temp = max(float(np.median(dp)), 0.35)
            events = evs[pool]
            counts = {e: int(np.sum(events == e)) for e in set(events)}
            w = np.exp(-0.5 * (dp / temp) ** 2) / np.array([counts[e] for e in events])
            w /= w.sum()
            cached = (np.array(pool, int), w, dist)
            if len(self._donor_pool_cache) >= 128:
                self._donor_pool_cache.pop(next(iter(self._donor_pool_cache)))
            self._donor_pool_cache[key] = cached
        pool, w, dist = cached
        pool_events=set(self.index.iloc[pool].event_id.astype(str))
        bad=pool_events.intersection(global_exclusions)
        if bad:
            raise RuntimeError(f'Held-out events reached empirical FAS donor pool: {sorted(bad)}')
        chosen = int(rng.choice(pool, p=w))
        cand = list(pool[:max(k, 10)])
        if chosen not in cand:
            cand = [chosen] + cand[:-1]
        row = self.index.iloc[chosen]
        pair = (str(row.event_id), str(row.station_id))
        return (chosen, cand, {'event_id': pair[0], 'station_id': pair[1], 'source_type': str(row.source_type), 'Mw': float(row.Mw), 'Rrup_km': float(row.Rrup_km), 'Ts_s': float(row.Ts_s), 'distance': float(dist[chosen]), 'candidate_count': len(cand), 'exclude_same_event': bool(exclude_same_event), 'excluded_events': list(global_exclusions), 'source_match': bool(source_match), 'observed_rho_envelope_MI': self._rho_lookup.get(pair, np.nan), 'donor_pool_cache': True, 'phase_timing_validated': True, 'pool_policy': 'event_round_robin_release', 'duration_model_id': 'final_hv', 'pool_events': len(pool_events), 'pool_size': len(pool)})

    def donor_scenario(self, donor_index:int) -> dict[str,Any]:
        r=self.index.iloc[int(donor_index)]
        return {k:(r[k] if k in r.index else np.nan) for k in _META_COLS if k not in {"texture_hdf5","texture_usable_3c","record_observed"}}

    def _fas_keys(self):
        if hasattr(self,"_fas_key_cache"): return self._fas_key_cache
        tables,_=self._load_fas()
        sets=[set((str(e),str(s)) for e,s in table.index) for table in tables.values()]
        self._fas_key_cache=set.intersection(*sets)
        return self._fas_key_cache

    @staticmethod
    def _parse_fas_columns(columns,prefix):
        cols=[]; freqs=[]
        rgx=re.compile(r"_f\d+_([0-9]+)p([0-9]+)hz$")
        for c in columns:
            if c.startswith(prefix):
                m=rgx.search(c)
                if m:
                    cols.append(c);freqs.append(float(m.group(1)+"."+m.group(2)))
        o=np.argsort(freqs)
        return [cols[i] for i in o],np.asarray(freqs,float)[o]

    def _load_fas(self):
        if self._fas_cache is not None: return self._fas_cache
        tables={}
        with np.load(self.asset_path,allow_pickle=False) as archive:
            for key in ("h_obs","h_smooth","v_obs","v_smooth"):
                columns=archive[f"{key}__columns"].astype(str).tolist()
                table=pd.DataFrame(archive[f"{key}__values"],columns=columns)
                table.insert(0,"station_id",archive[f"{key}__station_id"].astype(str))
                table.insert(0,"event_id",archive[f"{key}__event_id"].astype(str))
                tables[key]=table.set_index(["event_id","station_id"],drop=False)
        specs={}
        for c in COMPONENTS:
            pre={"major":"fas_major_cm_per_s_","intermediate":"fas_intermediate_cm_per_s_","vertical":"fas_vertical_cm_per_s_"}[c]
            to=tables["v_obs" if c=="vertical" else "h_obs"]
            cols,f=self._parse_fas_columns(to.columns,pre)
            specs[c]=(cols,f)
        self._fas_cache=(tables,specs)
        return self._fas_cache

    def empirical_fas(self, donor_index:int, candidate_indices:Sequence[int], frequencies_hz:Sequence[float], base_fas:Mapping[str,Sequence[float]], *, strength:float=1.0):
        tables,specs=self._load_fas(); ft=np.asarray(frequencies_hz,float)
        selected=self.index.iloc[int(donor_index)]
        selected_key=(str(selected.event_id),str(selected.station_id))
        ctuple=tuple(int(x) for x in candidate_indices)
        fkey=tuple(np.round(ft,8)); cache_key=(ctuple,fkey,round(float(strength),6))
        cached=self._fas_delta_cache.get(cache_key)
        if cached is None:
            deltas={}; base_diag={}
            for c in COMPONENTS:
                kind_o="v_obs" if c=="vertical" else "h_obs"; kind_s="v_smooth" if c=="vertical" else "h_smooth"
                cols,fs=specs[c]; R=[]; keys=[]
                for ii in candidate_indices:
                    rr=self.index.iloc[int(ii)]; key=(str(rr.event_id),str(rr.station_id))
                    if key not in tables[kind_o].index or key not in tables[kind_s].index: continue
                    ro=tables[kind_o].loc[key]; rs=tables[kind_s].loc[key]
                    yo=np.asarray(ro[cols],float); ys=np.asarray(rs[cols],float)
                    good=np.isfinite(yo)&np.isfinite(ys)&(yo>0)&(ys>0)&(fs>0)
                    if np.sum(good)<10: continue
                    resid=np.log(yo[good]/ys[good])
                    ri=np.interp(np.log(np.clip(ft,fs[good].min(),fs[good].max())),np.log(fs[good]),resid)
                    R.append(ri);keys.append(key)
                R=np.asarray(R,float)
                meanlog=np.mean(R,axis=0,keepdims=True); Rc=R-meanlog
                scaled=float(strength)*Rc; bias=np.log(np.mean(np.exp(scaled),axis=0))
                deltas[c]={key:(scaled[j]-bias) for j,key in enumerate(keys)}
                base_diag[c]={"candidate_residuals":len(keys),"mean_multiplier_check_median":float(np.median(np.mean(np.exp(scaled-bias[None,:]),axis=0)))}
            cached=(deltas,base_diag)
            if len(self._fas_delta_cache)>=64: self._fas_delta_cache.pop(next(iter(self._fas_delta_cache)))
            self._fas_delta_cache[cache_key]=cached
        deltas,base_diag=cached
        result={}; diag={}
        for c in COMPONENTS:
            if selected_key not in deltas[c]:
                raise RuntimeError(f"Selected empirical donor {selected_key} has no FAS residual for {c}")
            delta=deltas[c][selected_key]
            result[c]=np.asarray(base_fas[c],float)*np.exp(delta)
            diag[c]={**base_diag[c],"log_multiplier_std":float(np.std(delta)),"cache_hit":cache_key in self._fas_delta_cache}
        return result,diag

# Bind the fine-detail implementation to the empirical FAS library.
# The empirical API applies the same convention across runtime modes.
def _empirical_fas_detail_multiscale(self, donor_index:int, candidate_indices, frequencies_hz, base_fas, *, strength:float=0.60, split_decades:float=0.18):
    """Apply only the fine-scale part of real ln(FASobs/FASsmooth) residuals.

    The broad component is removed in log-frequency. The carrier applies this
    fine multiplier directly to the full empirical realization FAS,
    so real peaks/notches oscillate around that carrier rather than replacing its
    resonant/amplitude backbone. The candidate ensemble is corrected in linear
    space so E[exp(delta_detail)]=1 frequency-by-frequency.
    """
    tables,specs=self._load_fas(); ft=np.asarray(frequencies_hz,float)
    selected=self.index.iloc[int(donor_index)]
    selected_key=(str(selected.event_id),str(selected.station_id))
    ctuple=tuple(int(x) for x in candidate_indices)
    fkey=tuple(np.round(ft,8))
    cache_key=("detail_multiscale",ctuple,fkey,round(float(strength),6),round(float(split_decades),6))
    cached=self._fas_delta_cache.get(cache_key)
    if cached is None:
        from scipy.ndimage import gaussian_filter1d
        dlog=float(np.median(np.diff(np.log10(np.maximum(ft,1e-8)))))
        sig=max(float(split_decades)/max(dlog,1e-6),0.5)
        deltas={}; base_diag={}
        for c in COMPONENTS:
            kind_o="v_obs" if c=="vertical" else "h_obs"
            kind_s="v_smooth" if c=="vertical" else "h_smooth"
            cols,fs=specs[c]; R=[]; keys=[]
            for ii in candidate_indices:
                rr=self.index.iloc[int(ii)]; key=(str(rr.event_id),str(rr.station_id))
                if key not in tables[kind_o].index or key not in tables[kind_s].index: continue
                ro=tables[kind_o].loc[key]; rs=tables[kind_s].loc[key]
                yo=np.asarray(ro[cols],float); ys=np.asarray(rs[cols],float)
                good=np.isfinite(yo)&np.isfinite(ys)&(yo>0)&(ys>0)&(fs>0)
                if np.sum(good)<10: continue
                resid=np.log(yo[good]/ys[good])
                ri=np.interp(np.log(np.clip(ft,fs[good].min(),fs[good].max())),np.log(fs[good]),resid)
                R.append(ri); keys.append(key)
            R=np.asarray(R,float)
            if R.ndim!=2 or len(R)<2:
                raise RuntimeError(f"Insufficient empirical FAS donors for fine-scale detail: {c}")
            Rc=R-np.mean(R,axis=0,keepdims=True)
            H=Rc-gaussian_filter1d(Rc,sig,axis=1,mode="nearest")
            scaled=float(strength)*H
            bias=np.log(np.mean(np.exp(scaled),axis=0))
            Dc=scaled-bias[None,:]
            deltas[c]={key:Dc[j] for j,key in enumerate(keys)}
            base_diag[c]={
                "candidate_residuals":len(keys),
                "split_decades":float(split_decades),
                "detail_strength":float(strength),
                "mean_multiplier_check_median":float(np.median(np.mean(np.exp(Dc),axis=0))),
                "detail_log_std_mean":float(np.mean(np.std(Dc,axis=0,ddof=1))),
            }
        cached=(deltas,base_diag)
        if len(self._fas_delta_cache)>=64: self._fas_delta_cache.pop(next(iter(self._fas_delta_cache)))
        self._fas_delta_cache[cache_key]=cached
    deltas,base_diag=cached
    result={}; diag={}
    for c in COMPONENTS:
        if selected_key not in deltas[c]:
            raise RuntimeError(f"Selected empirical donor {selected_key} has no fine-scale FAS detail residual for {c}")
        delta=deltas[c][selected_key]
        result[c]=np.asarray(base_fas[c],float)*np.exp(delta)
        diag[c]={**base_diag[c],"selected_log_detail_std":float(np.std(delta)),"definition":"highpass ln(FASobs/FASsmooth)"}
    diag["hybrid_mode"]="carrier-preserving empirical fine detail"
    return result,diag

EmpiricalFASLibrary.empirical_fas_detail = _empirical_fas_detail_multiscale
