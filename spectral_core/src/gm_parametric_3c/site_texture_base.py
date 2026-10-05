from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


def _smoothstep01(x):
    x=np.clip(np.asarray(x,float),0.0,1.0)
    return x*x*(3.0-2.0*x)


def _site_gate(ts, t0=0.55, t1=0.80):
    ts=np.asarray(ts,float)
    return _smoothstep01((ts-float(t0))/max(float(t1-t0),1e-9))


def _u_window(u, u0=0.35, u1=0.50, u2=2.0, u3=2.8):
    u=np.asarray(u,float)
    w=np.zeros_like(u)
    m=(u>u0)&(u<u1); w[m]=_smoothstep01((u[m]-u0)/(u1-u0))
    w[(u>=u1)&(u<=u2)]=1.0
    m=(u>u2)&(u<u3); w[m]=1.0-_smoothstep01((u[m]-u2)/(u3-u2))
    return w


def _robust_scale(x, floor=1e-6):
    x=np.asarray(x,float)
    med=np.nanmedian(x,axis=0)
    q25=np.nanpercentile(x,25,axis=0); q75=np.nanpercentile(x,75,axis=0)
    s=(q75-q25)/1.349
    sd=np.nanstd(x,axis=0)
    s=np.where(np.isfinite(s)&(s>floor),s,sd)
    s=np.where(np.isfinite(s)&(s>floor),s,1.0)
    return med,s


@dataclass
class SiteTextureBase:
    """Localized site-resonance FAS texture model for release.

    The base FAS remains the broad conditional trend.  This object adds only
    a high-pass texture in normalized frequency u=f*Ts around the dominant
    site-resonance band.  The persistent field is inferred from neighboring
    stations; a modest empirical OOF-like record texture can be sampled as a
    second layer.  The correction is smoothly suppressed for Ts close to the
    CU/reference value so u=1 is not misinterpreted as a physical deposit
    resonance there.
    """
    u_grid: np.ndarray
    station_ids: np.ndarray
    station_x_km: np.ndarray
    station_y_km: np.ndarray
    station_Ts_s: np.ndarray
    station_profiles: np.ndarray
    station_counts: np.ndarray
    station_feature_center: np.ndarray
    station_feature_scale: np.ndarray
    station_k: int
    residual_profiles: np.ndarray
    residual_event_id: np.ndarray
    residual_station_id: np.ndarray
    residual_source_type: np.ndarray
    residual_features: np.ndarray
    residual_feature_center: np.ndarray
    residual_feature_scale: np.ndarray
    residual_k: int
    model_version: str='1.6.0'

    def save(self,path):
        joblib.dump(self,Path(path)); return Path(path)

    @classmethod
    def load(cls,path): return joblib.load(Path(path))

    def _station_feature(self, x, y, ts):
        return np.array([float(x),float(y),np.log(max(float(ts),1e-6))],float)

    def _predict_site_profile_one(self,row):
        sid=str(row.get('station_id','') or '')
        if sid:
            m=np.where(self.station_ids.astype(str)==sid)[0]
            if len(m): return self.station_profiles[int(m[0])].copy(), {'site_texture_source':'known_station','station_id':sid,'neighbor_count':1}
        x=float(row.get('x_conditioned_km',row.get('x_km',np.nan)))
        y=float(row.get('y_conditioned_km',row.get('y_km',np.nan)))
        ts=float(row['Ts_s'])
        if not np.isfinite(x) or not np.isfinite(y):
            # zero is safer than inventing a texture without location support
            return np.zeros_like(self.u_grid), {'site_texture_source':'zero_no_xy','neighbor_count':0}
        q=(self._station_feature(x,y,ts)-self.station_feature_center)/self.station_feature_scale
        X=np.column_stack([self.station_x_km,self.station_y_km,np.log(np.clip(self.station_Ts_s,1e-6,None))])
        X=(X-self.station_feature_center)/self.station_feature_scale
        d=np.sqrt(np.sum((X-q[None,:])**2,axis=1))
        k=min(int(self.station_k),len(d)); idx=np.argpartition(d,k-1)[:k]
        # inverse-distance weights with a small floor; shrink toward zero if remote
        ww=1.0/np.maximum(d[idx],0.20)**2; ww/=ww.sum()
        prof=np.sum(self.station_profiles[idx]*ww[:,None],axis=0)
        d0=float(np.min(d[idx]))
        shrink=float(np.exp(-0.10*max(0.0,d0-2.0)**2))
        prof*=shrink
        return prof, {'site_texture_source':'knn','neighbor_count':int(k),'nearest_std_distance':d0,'support_shrinkage':shrink,'neighbor_station_ids':[str(x) for x in self.station_ids[idx]]}

    def predict_site_profiles(self,scenarios):
        s=scenarios if isinstance(scenarios,pd.DataFrame) else pd.DataFrame(scenarios)
        out=[];diag=[]
        for _,r in s.iterrows():
            p,d=self._predict_site_profile_one(r); out.append(p);diag.append(d)
        return np.asarray(out,float),diag

    def _sample_residual_one(self,row,rng,k=None,exclude_same_event=False):
        source=str(row.get('source_type','')).upper()
        ev=str(row.get('event_id','') or '')
        x=float(row.get('x_conditioned_km',row.get('x_km',np.nan)))
        y=float(row.get('y_conditioned_km',row.get('y_km',np.nan)))
        ts=float(row['Ts_s']); mw=float(row.get('Mw',np.nan)); rr=float(row.get('Rrup_km',np.nan))
        q=np.array([mw,np.log(max(rr,1e-6)),np.log(max(ts,1e-6)),x,y],float)
        valid=np.isfinite(q).all()
        pool=np.ones(len(self.residual_profiles),bool)
        if source: pool &= (np.char.upper(self.residual_source_type.astype(str))==source)
        if exclude_same_event and ev: pool &= (self.residual_event_id.astype(str)!=ev)
        ids=np.where(pool)[0]
        if len(ids)==0: return np.zeros_like(self.u_grid), {'residual_texture_source':'zero_no_pool'}
        if not valid:
            j=int(rng.choice(ids)); return self.residual_profiles[j].copy(), {'residual_texture_source':'random_source_pool','donor_event_id':str(self.residual_event_id[j]),'donor_station_id':str(self.residual_station_id[j])}
        qq=(q-self.residual_feature_center)/self.residual_feature_scale
        XX=(self.residual_features[ids]-self.residual_feature_center)/self.residual_feature_scale
        d=np.sqrt(np.sum((XX-qq[None,:])**2,axis=1))
        kk=min(int(self.residual_k if k is None else k),len(ids)); loc=np.argpartition(d,kk-1)[:kk]
        cand=ids[loc]; dc=d[loc]
        # soft nearest-neighbor draw; not deterministic matching
        temp=max(float(np.median(dc)+1e-6),0.35)
        w=np.exp(-0.5*(dc/temp)**2); w/=w.sum()
        j=int(rng.choice(cand,p=w))
        return self.residual_profiles[j].copy(), {'residual_texture_source':'empirical_knn','donor_event_id':str(self.residual_event_id[j]),'donor_station_id':str(self.residual_station_id[j]),'donor_std_distance':float(d[np.where(ids==j)[0][0]]) if j in ids else np.nan}

    def sample_residual_profiles(self,scenarios,n_realizations=1,seed=None,exclude_same_event=False):
        s=scenarios if isinstance(scenarios,pd.DataFrame) else pd.DataFrame(scenarios)
        rng=np.random.default_rng(seed); out=np.zeros((int(n_realizations),len(s),len(self.u_grid)),float);diag=[]
        for r in range(int(n_realizations)):
            dr=[]
            for i,(_,row) in enumerate(s.iterrows()):
                p,d=self._sample_residual_one(row,rng,exclude_same_event=exclude_same_event);out[r,i]=p;dr.append(d)
            diag.append(dr)
        return out,diag

    def apply(self, frequencies_hz, major, intermediate, scenarios, n_realizations=None, seed=None,
              site_strength=0.90, residual_enabled=True, residual_scale=0.75,
              ts_gate_low=0.55, ts_gate_high=0.80, log_clip=1.10):
        freq=np.asarray(frequencies_hz,float)
        M=np.asarray(major,float).copy(); I=np.asarray(intermediate,float).copy()
        if M.ndim==2: M=M[None,:,:]
        if I.ndim==2: I=I[None,:,:]
        nr=M.shape[0] if n_realizations is None else int(n_realizations)
        if nr!=M.shape[0] or I.shape!=M.shape: raise ValueError('major/intermediate texture arrays must have shape [realization, scenario, frequency]')
        s=scenarios if isinstance(scenarios,pd.DataFrame) else pd.DataFrame(scenarios)
        site,site_diag=self.predict_site_profiles(s)
        if residual_enabled:
            resid,resid_diag=self.sample_residual_profiles(s,nr,seed=seed,exclude_same_event=False)
        else:
            resid=np.zeros((nr,len(s),len(self.u_grid)),float);resid_diag=[[{'residual_texture_source':'disabled'} for _ in range(len(s))] for _ in range(nr)]
        all_diag=[]
        for r in range(nr):
            rd=[]
            for i,row in s.iterrows():
                ts=float(row['Ts_s']); gate=float(_site_gate(ts,ts_gate_low,ts_gate_high)); u=freq*ts; win=_u_window(u)
                # interpolate in log-u; texture is already locally centered around zero
                lp=np.log(self.u_grid); lu=np.log(np.clip(u,self.u_grid[0],self.u_grid[-1]))
                sp=np.interp(lu,lp,site[i],left=0.0,right=0.0)
                rp=np.interp(lu,lp,resid[r,i],left=0.0,right=0.0)
                corr=gate*win*(float(site_strength)*sp + float(residual_scale)*rp)
                corr=np.clip(corr,-float(log_clip),float(log_clip))
                M[r,i]*=np.exp(corr); I[r,i]*=np.exp(corr)
                rd.append({'site_gate':gate,'site_strength':float(site_strength),'residual_scale':float(residual_scale),'max_abs_log_correction':float(np.max(np.abs(corr))),**site_diag[i],**resid_diag[r][i]})
            all_diag.append(rd)
        return {'major':M,'intermediate':I,'diagnostics':all_diag,'model_version':self.model_version}
