"""VALLIS-3C 1.6.0 synthesis core: complex coefficient transport only."""
from __future__ import annotations
import numpy as np
from scipy.fft import rfft,irfft
from scipy.signal import hilbert
from .numerics import BAND_CENTERS,filters,even_fast_len,energy_times

def _length(n,scale,dt,ceiling):
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid temporal scale')
    out=even_fast_len(int(np.ceil(n*scale)))
    if out*dt>ceiling:raise ValueError(f'Needed support {out*dt:.1f}s exceeds safety ceiling; not truncated')
    return out

def transport_complex(a,dt,scale,rng,rotate=True,ceiling=1200,vertical_hf_relative_scales=None,n_components=3):
    n_components=int(n_components)
    if n_components not in (2,3):raise ValueError('Transport output must contain 2 or 3 components')
    n0=a.shape[-1];rel=dict(vertical_hf_relative_scales or {}) if n_components==3 else {}
    for b,v in rel.items():
        if int(b)<0 or int(b)>=len(BAND_CENTERS) or not np.isfinite(v) or not (0<v<=1):
            raise ValueError('Vertical HF relative scales must be finite in (0,1]')
    maxscale=max([scale]+[scale*float(v) for v in rel.values()]);n=_length(n0,maxscale,dt,ceiling)
    H0=filters(n0,dt);H=filters(n,dt);X=rfft(a[:n_components],axis=-1);t0=np.arange(n0)*dt;t=np.arange(n)*dt
    # Keep the same RNG namespace for M/I as the 3C route.
    phase=rng.uniform(-np.pi,np.pi,3) if rotate else np.zeros(3);out=np.zeros((n_components,n//2+1),complex)
    for b,f in enumerate(BAND_CENTERS):
        z=hilbert(irfft(X*H0[b][None,:],n=n0,axis=-1),axis=-1);base=z*np.exp(-2j*np.pi*f*t0)
        t50v=float(energy_times(np.abs(z[2])**2,dt,(.5,))[0]) if b in rel else None
        for c in range(n_components):
            sb=scale
            if c==2 and b in rel:
                sb=scale*float(rel[b]);anchor_out=scale*t50v;tq=t50v+(t-anchor_out)/sb
            else:tq=t/scale
            u=np.interp(tq,t0,base[c].real,left=0,right=0)+1j*np.interp(tq,t0,base[c].imag,left=0,right=0)
            moved=u*np.exp(2j*np.pi*f*t+1j*phase[c])/np.sqrt(sb);out[c]+=rfft(moved.real)*H[b]
    return irfft(out,n=n,axis=-1)
