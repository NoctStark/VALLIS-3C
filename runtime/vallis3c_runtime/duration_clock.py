"""Unified horizontal and vertical duration model for VALLIS-3C 1.6.0."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FINAL_HV_FEATURES=('Mw','ln_Rrup_km','depth_used_km','ln_Ts_s')
VERTICAL_BANDS=tuple(range(31,39))

def weighted_quantile(values,q,weights):
    v=np.asarray(values,float);w=np.asarray(weights,float)
    if v.ndim!=1 or w.shape!=v.shape or len(v)==0 or not np.isfinite(v).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0:raise ValueError('Invalid weighted quantile inputs')
    q=float(q)
    if not 0<=q<=1:raise ValueError('Quantile must be in [0,1]')
    ix=np.argsort(v,kind='mergesort');v=v[ix];w=w[ix];c=np.cumsum(w)-.5*w;c/=w.sum()
    return float(np.interp(q,c,v,left=v[0],right=v[-1]))

def rank_probabilities(values):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)==0 or not np.isfinite(x).all():raise ValueError('Invalid rank inputs')
    order=np.argsort(x,kind='mergesort');q=np.empty(len(x),float);q[order]=(np.arange(len(x),dtype=float)+.5)/len(x)
    return q

def desired_duration(condition,q):
    r=weighted_quantile(condition['residual_ln'],q,condition['weights'])
    return float(np.exp(np.log(condition['center_D5H_s'])+r))

class PortableHVMultiForest:
    """NumPy-only inference for the unified H/V multioutput RF."""
    def __init__(self,path):
        self.arr=np.load(Path(path),allow_pickle=False)
        if self.arr['feature_names'].tolist()!=list(FINAL_HV_FEATURES):
            raise ValueError('Incompatible unified duration-model features')
        self.target_names=self.arr['target_names'].tolist()
        self.band_indices=np.asarray(self.arr['band_indices'],int)
        if self.band_indices.tolist()!=list(VERTICAL_BANDS):raise ValueError('Incompatible V/H duration-ratio bands')
        self.block=np.asarray(self.arr['block_weights'],float)
        self.grid_ts=np.asarray(self.arr['grid_ts_s'],float);self.grid_ln=np.log(self.grid_ts)
        self.dm=float(np.asarray(self.arr['dm'],float).ravel()[0]);self.dlnr=float(np.asarray(self.arr['dlnr'],float).ravel()[0])
        self.mw_cap_ip=float(np.asarray(self.arr['mw_cap_interplate'],float).ravel()[0])

    def _forest_transformed(self,src,X):
        X=np.atleast_2d(np.asarray(X,float));p=src+'_'
        roots=self.arr[p+'roots'];feat=self.arr[p+'features'];thr=self.arr[p+'thresholds'];left=self.arr[p+'children_left'];right=self.arr[p+'children_right'];val=self.arr[p+'values']
        nodes=np.tile(roots,(len(X),1));rows=np.arange(len(X))[:,None]
        for _ in range(10000):
            active=feat[nodes]>=0
            if not active.any():return val[nodes].mean(axis=1)
            f=np.maximum(feat[nodes],0);xx=X[rows,f];nxt=np.where(xx<=thr[nodes],left[nodes],right[nodes]);nodes=np.where(active,nxt,nodes)
        raise RuntimeError('Invalid unified duration-model tree cycle')

    def _model_X(self,src,X):
        X=np.atleast_2d(np.asarray(X,float)).copy()
        if X.shape[1]!=4 or not np.isfinite(X).all():raise ValueError('Invalid unified duration-model inputs')
        if src=='INTERPLATE':X[:,0]=np.minimum(X[:,0],self.mw_cap_ip)
        return X

    def raw_predict_source(self,source,X):
        src=str(source).upper()
        if src not in ('INSLAB','INTERPLATE'):raise ValueError('source_type must be INSLAB or INTERPLATE')
        Xm=self._model_X(src,X);z=self._forest_transformed(src,Xm);mu=np.asarray(self.arr[src+'_y_mu'],float);sd=np.asarray(self.arr[src+'_y_sd'],float)
        return z/self.block[None,:]*sd[None,:]+mu[None,:]

    def _support_distance(self,src,X):
        center=np.asarray(self.arr[src+'_support_center'],float);scale=np.asarray(self.arr[src+'_support_scale'],float);proto=np.asarray(self.arr[src+'_support_event_prototypes'],float);d0=float(np.asarray(self.arr[src+'_support_d0'],float).ravel()[0])
        Xm=self._model_X(src,X);z=(Xm[:,:3]-center[None,:])/scale[None,:];zp=(proto-center[None,:])/scale[None,:]
        dist=np.sqrt(((z[:,None,:]-zp[None,:,:])**2).sum(axis=2)).min(axis=1)
        return dist,d0

    def predict_source(self,source,X,original_mw=None,return_details=False):
        src=str(source).upper();X=np.atleast_2d(np.asarray(X,float));out=self.raw_predict_source(src,X);base_h=out[:,0].copy()
        mw=np.asarray(original_mw if original_mw is not None else X[:,0],float).reshape(-1);thr=float(np.asarray(self.arr[src+'_event_mw_q90'],float).ravel()[0]);upper=mw>thr
        dist,d0=self._support_distance(src,X);active=upper if src=='INTERPLATE' else (upper&(dist>d0))
        slopes=np.zeros(len(X),float);anchor_level=base_h.copy()
        if np.any(active):
            curves=[]
            for ts in self.grid_ts:
                z=X.copy();z[:,3]=np.log(ts);curves.append(self.raw_predict_source(src,z)[:,0])
            curves=np.stack(curves,axis=1);xc=self.grid_ln-self.grid_ln.mean();slopes=np.maximum((curves@xc)/(xc@xc),0.0)
            lnTa=float(np.asarray(self.arr[src+'_lnTs_anchor'],float).ravel()[0]);anch=np.array([np.interp(lnTa,self.grid_ln,curves[i]) for i in range(len(X))])
            if src=='INSLAB':
                beta=np.asarray(self.arr[src+'_linear_beta'],float);inter=float(np.asarray(self.arr[src+'_linear_intercept'],float).ravel()[0]);Xa=X.copy();Xa[:,3]=lnTa;Xlo=Xa.copy();Xlo[:,0]-=self.dm;Xlo[:,1]-=self.dlnr
                plo=self.raw_predict_source(src,Xlo)[:,0];expected=max(float(beta[0]),0.0)*self.dm+max(float(beta[1]),0.0)*self.dlnr;lin_anchor=inter+Xa@beta;anch=np.maximum(anch,np.maximum(plo+expected,lin_anchor))
            out[active,0]=anch[active]+slopes[active]*(X[active,3]-lnTa);anchor_level=anch
        details={'functionalized':active,'event_mw_q90':thr,'support_distance':dist,'support_d0':d0,'beta_T_RF':slopes,'raw_lnD5H':base_h,'anchor_lnD5H':anchor_level}
        return (out,details) if return_details else out


class FinalHVDurationMapper:
    """Single production duration architecture for horizontal and vertical clocks."""
    def __init__(self,model_dir):
        root=Path(model_dir);self.model=PortableHVMultiForest(root/'duration_clock_final_hv_portable.npz')
        tab=pd.read_csv(root/'duration_support.csv');tab=tab[np.isfinite(tab.D5H_s)&(tab.D5H_s>0)].drop_duplicates('record_id');self.duration_lookup=dict(zip(tab.record_id.astype(str),tab.D5H_s.astype(float)))

    def donor_duration(self,record_id):
        d=float(self.duration_lookup.get(str(record_id),np.nan))
        if not np.isfinite(d) or d<=0:raise ValueError(f'Missing donor D5-95 duration for {record_id}')
        return d

    @staticmethod
    def _scenario_vector(scenario):
        s=dict(scenario)
        def f(key):
            try:return float(s.get(key,np.nan))
            except (TypeError,ValueError):return np.nan
        rr=f('Rrup_km');ts=f('Ts_s');x=np.array([f('Mw'),np.log(rr) if rr>0 else np.nan,f('depth_used_km'),np.log(ts) if ts>0 else np.nan],float)
        if not np.isfinite(x).all():raise ValueError('Unified duration model requires finite Mw, positive Rrup_km, depth_used_km and positive Ts_s')
        return x

    def predict_hv(self,scenario,return_details=False):
        src=str(scenario.get('source_type','')).upper();x=self._scenario_vector(scenario);mw=float(scenario.get('Mw',np.nan));return self.model.predict_source(src,x,original_mw=[mw],return_details=return_details)

    def center_duration(self,scenario):
        pred=self.predict_hv(scenario);d=float(np.exp(pred[0,0]))
        if not np.isfinite(d) or d<=0:raise ValueError('Invalid unified duration-model prediction')
        return d

    def vertical_target_ratios(self,scenario):
        pred=self.predict_hv(scenario)[0,1:];out={}
        for b,lnr in zip(self.model.band_indices,pred):
            r=float(np.exp(lnr));out[int(b)]=r if np.isfinite(r) and r>0 else np.nan
        return out

    def condition(self,scenario,freq=None,fas=None,exclude_same_event=False):
        src=str(scenario.get('source_type','')).upper();pred,detail=self.predict_hv(scenario,return_details=True);center=float(np.exp(pred[0,0]));residual=np.asarray(self.model.arr[src+'_residual_ln'],float);weights=np.asarray(self.model.arr[src+'_residual_weights'],float);weights=weights/weights.sum();mu=float(np.sum(weights*residual));sig=float(np.sqrt(np.sum(weights*(residual-mu)**2)))
        return {'architecture':'final_hv','center_D5H_s':center,'support_sigma_ln':sig,'sigma_oos_ln':sig,'alpha':1.0,'residual_ln':residual,'weights':weights,'support_count':int(len(residual)),'feature_names':list(FINAL_HV_FEATURES),'vertical_ratio_bands':self.model.band_indices.tolist(),'functionalized_upper_tail':bool(detail['functionalized'][0]),'event_mw_q90':float(detail['event_mw_q90']),'support_distance':float(detail['support_distance'][0]),'support_d0':float(detail['support_d0']),'beta_T_RF':float(detail['beta_T_RF'][0])}

    @staticmethod
    def desired_duration(condition,q):return desired_duration(condition,q)
