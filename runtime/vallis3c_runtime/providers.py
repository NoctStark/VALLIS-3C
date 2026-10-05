"""FAS providers for VALLIS-3C 1.6.0."""
from __future__ import annotations
from dataclasses import dataclass
import copy,hashlib
import numpy as np
import pandas as pd
from .numerics import COMPONENTS
from .config import fas_preset,FAS_RUNTIME_INVARIANTS,ConfigurationError

def array_hash(a):return hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest()

def validated_targets(f,targets,n,ns):
    f=np.asarray(f,float);t=np.asarray(targets,float)
    if f.ndim!=1 or len(f)<8 or not np.isfinite(f).all() or np.any(f<=0) or np.any(np.diff(f)<=0):
        raise ConfigurationError('Invalid FAS frequency grid')
    if t.shape!=(n,ns,3,len(f)) or not np.isfinite(t).all() or np.any(t<=0):
        raise ConfigurationError(f'Invalid FAS target array: expected {(n,ns,3,len(f))}, got {t.shape}')
    if f[0]>.1+1e-10 or f[-1]<20.-1e-10:
        raise ConfigurationError('This release requires a surface FAS covering 0.1–20 Hz; no hidden extrapolation')
    return f,t

@dataclass
class FASTargets:
    frequencies_hz:np.ndarray
    amplitudes:np.ndarray
    metadata:dict

class SuppliedFASProvider:
    def __init__(self,exact_fas,mode='supplied_surface'):
        self.exact=copy.deepcopy(exact_fas);self.mode=mode
    def resolve(self,scenarios,n,seed=0,options=None):
        s=pd.DataFrame(scenarios);f=np.asarray(self.exact.get('frequencies_hz',self.exact.get('freq',[])),float)
        keep=f>0
        if f.ndim!=1 or not np.isfinite(f).all() or not np.any(keep):
            raise ConfigurationError('Invalid FAS frequency grid')
        nf=int(np.count_nonzero(keep));vals=[]
        for c in COMPONENTS:
            a=np.asarray(self.exact[c],float)
            if a.shape[-1]!=len(f):raise ConfigurationError(f'Invalid exact {c} FAS length')
            a=a[...,keep]
            if a.ndim==1:a=np.broadcast_to(a,(n,len(s),nf))
            elif a.shape==(len(s),nf):a=np.broadcast_to(a,(n,len(s),nf))
            elif a.shape==(n,nf) and len(s)==1:a=a[:,None,:]
            if a.shape!=(n,len(s),nf):raise ConfigurationError(f'Invalid exact {c} FAS shape')
            vals.append(np.maximum(a,1e-12))
        f=f[keep];f,t=validated_targets(f,np.stack(vals,axis=2),n,len(s))
        return FASTargets(f,t.copy(),{'provider':self.mode,'sha256':array_hash(t),'FT_reapplied':False,'stochastic_FAS_added':False})

class ModelFASProvider:
    def __init__(self,surface_engine):
        if not callable(getattr(surface_engine,'_fas_targets',None)):
            raise ConfigurationError('The VALLIS-3C surface carrier with _fas_targets is required')
        self.engine=surface_engine
    def resolve(self,scenarios,n,seed=0,options=None):
        s=pd.DataFrame(scenarios).reset_index(drop=True)
        kw=fas_preset();kw.update(copy.deepcopy(options or {}));kw.update(FAS_RUNTIME_INVARIANTS)
        if kw.get('fas_variability_source')!='surface_hybrid':
            raise ConfigurationError('VALLIS-3C 1.6.0 requires the packaged conditional FAS carrier')
        fas,source,base_audit=self.engine._fas_targets(s,int(n),int(seed),kw,None)
        if base_audit.get('backend')=='mean_fallback':
            raise RuntimeError('Conditional carrier returned mean_fallback instead of the packaged FAS architecture')
        f=np.asarray(fas['frequencies_hz'],float)
        t=np.stack([np.asarray(fas[c],float) for c in COMPONENTS],axis=2).copy()
        per=[];exact=kw.get('fas_input_mode','model')=='exact_event'
        add_detail=source=='surface_hybrid' and (not exact or bool(kw.get('fas_empirical_on_exact_input',False)))
        excluded_events=tuple(str(e).strip() for e in (kw.get('excluded_donor_event_ids',()) or ()) if str(e).strip())
        if add_detail:
            for i in range(len(s)):
                for r in range(n):
                    rs=int(np.random.SeedSequence([int(seed),i,r,4100]).generate_state(1)[0]);rng=np.random.default_rng(rs)
                    j,candidates,da=self.engine.empirical.select_donor(
                        s.iloc[i].to_dict(),rng,
                        k=int(kw.get('fas_donor_k',30)),
                        source_match=bool(kw.get('fas_donor_source_match',True)),
                        exclude_same_event=False,require_fas=True,
                        exclude_events=excluded_events
                    )
                    candidate_events=set(self.engine.empirical.index.iloc[candidates].event_id.astype(str))
                    bad=candidate_events.intersection(excluded_events)
                    if bad:
                        raise RuntimeError(f'Held-out events reached empirical FAS donor pool: {sorted(bad)}')
                    original={c:t[r,i,ci].copy() for ci,c in enumerate(COMPONENTS)}
                    result,audit=self.engine.empirical.empirical_fas_detail(
                        j,candidates,f,original,
                        strength=float(kw['fas_detail_strength']),
                        split_decades=float(kw['fas_split_decades'])
                    )
                    for ci,c in enumerate(COMPONENTS):t[r,i,ci]=result[c]
                    per.append({
                        'scenario_index':i,'realization_index':r,'seed':rs,'donor':da,
                        'empirical_fas_candidate_events':sorted(candidate_events),
                        'excluded_donor_event_ids':list(excluded_events),
                        'empirical_detail':audit
                    })

        f,t=validated_targets(f,t,n,len(s))
        return FASTargets(f,t,{
            'provider':'vallis3c_surface_carrier',
            'base_audit':base_audit,
            'realization_audit':per,
            'options':{k:v for k,v in kw.items() if k!='exact_fas'},
            'site_state_mode':'Ts_conditioned_production',
            'sha256':array_hash(t),
            'FT_reapplied':False,
            'temporal_carrier_executed':False,
        })
