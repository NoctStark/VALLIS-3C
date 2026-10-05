"""Portable ExtraTrees conditional selector. No sklearn dependency at inference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .numerics import SPECTRAL_NODES,COMPONENTS,interp_fas
from .config import ConfigurationError

RAW_FEATURES=('Mw','Rrup_km','depth_used_km','Ts_s','x_conditioned_km','y_conditioned_km','event_longitude','event_latitude')
FEATURE_NAMES=['Mw','ln_Rrup_km','depth_used_km','ln_Ts_s','x_conditioned_km','y_conditioned_km','event_longitude','event_latitude']+[
 f'ln_FAS_{c}_{f:.9g}Hz_cm_per_s' for c in COMPONENTS for f in SPECTRAL_NODES]

def feature_vector(scenario,freq,fas):
    s=dict(scenario);missing=[];v=[]
    source=str(s.get('source_type','')).upper()
    if source not in ('INSLAB','INTERPLATE'):raise ConfigurationError('source_type must be INSLAB or INTERPLATE')
    for key in RAW_FEATURES:
        try:x=float(s.get(key,np.nan))
        except (TypeError,ValueError):x=np.nan
        if not np.isfinite(x):
            if key not in RAW_FEATURES[-2:]:raise ConfigurationError(f'Required feature missing: {key}')
            missing.append(key)
        v.append(x)
    if v[1]<=0 or v[3]<=0 or v[2]<0:raise ConfigurationError('Rrup/Ts must be positive; depth nonnegative')
    v[1]=np.log(v[1]);v[3]=np.log(v[3])
    if min(freq)>.2 or max(freq)<8:raise ConfigurationError('FAS must cover 0.2–8 Hz for conditioning')
    a=interp_fas(freq,fas,SPECTRAL_NODES)
    if a.shape!=(3,12):raise ConfigurationError('Conditioning needs all three FAS components')
    return np.r_[v,np.log(a).ravel()],missing

class PortableSelector:
    def __init__(self,model_dir):
        root=Path(model_dir)
        manifest_path=root/'training_manifest.json'
        if manifest_path.exists():
            self.manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            self._training_manifest_available=True
        else:
            # FULL production runs do not require training metadata at inference.
            # Keep initialization resilient to transient sync/missing-manifest issues;
            # strict OOS validation will fail explicitly in assert_validation_scope.
            self.manifest={'excluded_events': []}
            self._training_manifest_available=False
        self.catalog=pd.read_csv(root/'catalog.csv').reset_index(drop=True)
        self.artifact_name='selector_portable.npz'
        self.arr=np.load(root/self.artifact_name,allow_pickle=False)
        self.feature_names=self.arr['feature_names'].tolist()
        if self.feature_names!=FEATURE_NAMES:raise ValueError('Incompatible feature definition')
        self.lookup={k:i for i,k in enumerate(self.catalog.record_id.astype(str))}
        self.selector_id='temporal_texture_selector'
        self.training_records=int(sum(len(self.arr[s+'_catalog_indices']) for s in ('INSLAB','INTERPLATE')))
        self.min_samples_leaf=int(np.asarray(self.arr['min_samples_leaf']).ravel()[0]) if 'min_samples_leaf' in self.arr.files else int(self.manifest.get('training_configuration',{}).get('min_samples_leaf',3))
    def encode(self,scenario,freq,fas):
        x,missing=feature_vector(scenario,freq,fas)
        source=str(scenario['source_type']).upper();med=self.arr[source+'_feature_median']
        return np.where(np.isfinite(x),x,med).astype(np.float32),missing
    def apply(self,source,X):
        X=np.atleast_2d(X).astype(np.float32);a=self.arr;p=source+'_';roots=a[p+'roots'];feat=a[p+'features'];thr=a[p+'thresholds'];left=a[p+'children_left'];right=a[p+'children_right']
        nodes=np.tile(roots,(len(X),1));rows=np.arange(len(X))[:,None]
        for _ in range(10000):
            active=feat[nodes]>=0
            if not active.any():return nodes
            values=X[rows,np.maximum(feat[nodes],0)];nxt=np.where(values<=thr[nodes],left[nodes],right[nodes]);nodes=np.where(active,nxt,nodes)
        raise RuntimeError('Invalid tree cycle')
    def predict(self,source,X):
        leaves=self.apply(source,X);return self.arr[source+'_values'][leaves].mean(axis=1)
    def _event_weights(self,rows):
        events=self.catalog.event_id.iloc[rows].astype(str).to_numpy();_,inverse,count=np.unique(events,return_inverse=True,return_counts=True)
        w=1/count[inverse];return w/w.sum()
    def select(self,scenario,freq,fas,rng,exclude_same_event=False,complete_donor_only=False,donor_eligibility_mode='energy'):
        source=str(scenario['source_type']).upper();x,missing=self.encode(scenario,freq,fas);query=self.apply(source,x)[0]
        inds=self.arr[source+'_catalog_indices'];leaves=self.arr[source+'_training_leaves'];mask=np.ones(len(inds),bool)
        if exclude_same_event:
            eid=str(scenario.get('event_id',''));mask &= self.catalog.event_id.iloc[inds].astype(str).to_numpy()!=eid
        eligible=[]
        for t,node in enumerate(query):
            rows=inds[(leaves[:,t]==node)&mask]
            if len(rows):eligible.append((t,rows))
        if not eligible:raise ValueError('No conditional leaf support after exclusions; no global fallback')
        tree,rows=eligible[int(rng.integers(len(eligible)))];weights=self._event_weights(rows)
        original_donor=int(rng.choice(rows,p=weights));donor=original_donor;pool=rows;pool_weights=weights;reselected=False
        if complete_donor_only:
            mode=str(donor_eligibility_mode).lower()
            if mode=='energy' and 'temporal_donor_eligible' in self.catalog.columns:
                okcol=self.catalog.temporal_donor_eligible.astype(bool).to_numpy()
            else:
                okcol=~self.catalog.boundary_warning.astype(bool).to_numpy()
                mode='energy'
            clean=rows[okcol[rows]]
            if len(clean)==0:
                clean_eligible=[]
                for t,node in enumerate(query):
                    rr=inds[(leaves[:,t]==node)&mask]
                    cc=rr[okcol[rr]]
                    if len(cc):clean_eligible.append((t,rr,cc))
                if not clean_eligible:raise ValueError('No eligible waveform donor support after exclusions')
                tree,rows,clean=clean_eligible[int(rng.integers(len(clean_eligible)))];weights=self._event_weights(rows);reselected=True
            pool=clean;pool_weights=self._event_weights(pool);donor=int(rng.choice(pool,p=pool_weights))
        pred=self.predict(source,x)[0]
        return {'source_type':source,'tree':int(tree),'node':int(query[tree]),'indices':rows,'weights':weights,
                'donor_index':donor,'original_donor_index':original_donor,'predicted_targets':pred,'features':x,
                'imputed_features':missing,'admissible_trees':len(eligible),'donor_pool_indices':np.asarray(pool,int),
                'donor_pool_weights':np.asarray(pool_weights,float),'complete_donor_only':bool(complete_donor_only),
                'tree_reselected_for_complete_donor':bool(reselected),'donor_eligibility_mode':str(donor_eligibility_mode)}
    def fitted_log_duration(self,index):return self.arr['fitted_log_durations'][int(index)]
    def assert_validation_scope(self,scenario,strict):
        if not strict:return
        if not getattr(self,'_training_manifest_available',False):
            raise ConfigurationError('Strict validation requires training_manifest.json for the selected OOS bank')
        if not str(scenario.get('event_id','')):raise ConfigurationError('Strict validation requires event_id')
        excluded=set(self.manifest.get('excluded_events',[]))
        if str(scenario['event_id']) not in excluded:raise ConfigurationError('Strict event-held-out request cannot use this full-trained selector')
