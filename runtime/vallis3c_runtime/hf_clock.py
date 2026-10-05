"""Observed donor vertical-duration ratio bank for VALLIS-3C 1.6.0."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .numerics import BAND_CENTERS

class VerticalRatioBank:
    def __init__(self,model_dir,selector):
        df=pd.read_csv(Path(model_dir)/'hf_vertical_clock_exactbank_5_15.csv')
        band_cols=[c for c in df.columns if c.startswith('b') and c[1:].isdigit()]
        self.band_indices=np.array([int(c[1:]) for c in band_cols],dtype=int)
        self.band_centers_hz=BAND_CENTERS[self.band_indices].astype(float)
        self.record_ids=df['record_id'].astype(str).to_numpy();self.ratios=df[band_cols].to_numpy(float);self.lookup={rid:i for i,rid in enumerate(self.record_ids)}
        if selector.catalog.record_id.astype(str).tolist()!=self.record_ids.tolist():raise ValueError('Vertical ratio bank/catalog order mismatch')
    def donor_ratios(self,record_id):
        j=self.lookup.get(str(record_id));out={}
        if j is None:return out
        for k,b in enumerate(self.band_indices):
            x=float(self.ratios[j,k])
            if np.isfinite(x) and x>0:out[int(b)]=x
        return out
    def one_sided_relative_scales(self,target_ratios,record_id):
        donor=self.donor_ratios(record_id);out={}
        for b in self.band_indices:
            b=int(b);rt=float(target_ratios.get(b,np.nan));rd=float(donor.get(b,np.nan))
            out[b]=float(min(1.0,rt/rd)) if np.isfinite(rt) and np.isfinite(rd) and rt>0 and rd>0 else 1.0
        return out
