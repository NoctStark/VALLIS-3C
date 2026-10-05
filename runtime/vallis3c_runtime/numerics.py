"""Numerical definitions used by training and synthesis. No temporal envelope fitting."""
from __future__ import annotations
from functools import lru_cache
from fractions import Fraction
import numpy as np
from scipy.fft import rfft,irfft,rfftfreq,next_fast_len
from scipy.signal import hilbert,resample_poly
from scipy.integrate import trapezoid

COMPONENTS=('major','intermediate','vertical')
BAND_CENTERS=np.geomspace(.1,20.,42)
SPECTRAL_NODES=np.geomspace(.2,8.,12)
PERSISTENCE_NODES=np.array([.2,.5,1.,2.,4.,8.])
DT_NATIVE=.02
GUARD_S=11.25
SMOOTH_NODES=np.geomspace(.1,20.,256)

def even_fast_len(n):
    n=next_fast_len(int(n),real=True)
    return n+n%2

def bandpass(f):
    f=np.asarray(f,float)
    p=np.zeros_like(f)
    p[(f>=.15)&(f<=18.)]=1
    m=(f>=.1)&(f<.15);p[m]=np.sin(np.pi/2*(f[m]-.1)/.05)**2
    m=(f>18.)&(f<=20.);p[m]=np.cos(np.pi/2*(f[m]-18.)/2.)**2
    return p

def prepare_signal(a,dt,output_dt=DT_NATIVE):
    a=np.asarray(a,dtype=float)
    if a.ndim!=2 or a.shape[0] not in (2,3) or a.shape[1]<8 or not np.isfinite(a).all():
        raise ValueError('Invalid component array')
    if not np.isfinite(dt) or dt<=0:raise ValueError('Invalid dt')
    if 0.5/output_dt<=20:raise ValueError('Nyquist must exceed 20 Hz')
    rat=Fraction(float(dt)/float(output_dt)).limit_denominator(100000)
    if not np.isclose(float(rat),dt/output_dt,rtol=1e-9):raise ValueError('Inexact rational resampling ratio')
    if rat!=1:a=resample_poly(a,rat.numerator,rat.denominator,axis=-1)
    margin=int(round(GUARD_S/output_dt));n=even_fast_len(a.shape[1]+2*margin)
    y=np.zeros((len(a),n));y[:,margin:margin+a.shape[1]]=a
    Y=rfft(y,axis=-1)*bandpass(rfftfreq(n,output_dt))
    Y[:,0]=0;Y[:,-1]=0
    return irfft(Y,n=n,axis=-1)

def energy_times(power,dt,qs=(.025,.05,.25,.5,.75,.95,.975)):
    p=np.asarray(power,float)
    if p.ndim!=1 or len(p)<2 or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('Power must be a finite nonnegative vector')
    area=.5*(p[:-1]+p[1:])*dt
    acc=np.r_[0.,np.cumsum(area)]
    if acc[-1]<=0:raise ValueError('Zero energy')
    # Piecewise linear cumulative trapezoidal energy, with repeated CDF levels removed.
    unique,ix=np.unique(acc/acc[-1],return_index=True)
    return np.interp(qs,unique,ix*dt)

def duration(a,dt):
    q=energy_times(np.asarray(a)**2,dt)
    return float(q[-1]-q[0])

@lru_cache(maxsize=12)
def filters(n,dt):
    f=rfftfreq(n,dt)
    s=np.median(np.diff(np.log(BAND_CENTERS)))
    h=np.exp(-.5*((np.log(np.maximum(f,1e-30))[None,:]-np.log(BAND_CENTERS)[:,None])/s)**2)
    h/=np.sqrt(np.maximum((h*h).sum(axis=0),1e-300))[None,:]
    h[:,0]=0;h[:,-1]=0
    return h

@lru_cache(maxsize=12)
def ko_weights(n,dt):
    f=rfftfreq(n,dt)[1:]
    u=40*np.log10(f[None,:]/SMOOTH_NODES[:,None])
    w=np.sinc(u/np.pi)**4
    w/=w.sum(axis=1,keepdims=True)
    return w

def smooth_spectrum(a,dt):
    a=np.atleast_2d(a)
    amps=dt*abs(rfft(a,axis=-1))[:,1:]
    return amps @ ko_weights(a.shape[-1],float(dt)).T

def interp_fas(freq,amp,query):
    f=np.asarray(freq,float);v=np.asarray(amp,float);q=np.asarray(query,float)
    if f.ndim!=1 or len(f)<8 or np.any(f<=0) or np.any(np.diff(f)<=0) or not np.isfinite(f).all():
        raise ValueError('FAS frequency grid must be finite, positive and strictly increasing')
    if v.shape[-1]!=len(f) or np.any(v<=0) or not np.isfinite(v).all():
        raise ValueError('FAS amplitudes must be finite, strictly positive, and match the grid')
    return np.array([np.exp(np.interp(np.log(np.maximum(q,1e-30)),np.log(f),np.log(row))) for row in np.atleast_2d(v)])

def match_smooth(a,dt,freq,fas):
    """One frequency-smooth correction, NOT a bin-by-bin phase-only projection."""
    n=a.shape[-1];ff=rfftfreq(n,dt);Y=rfft(a,axis=-1)
    sm=smooth_spectrum(a,dt)
    # Both numerator and denominator are smoothed in the same analysis grid.
    target_fft=interp_fas(freq,fas,ff)*bandpass(ff)[None,:]
    tar=target_fft[:,1:] @ ko_weights(n,float(dt)).T
    ratio=tar/np.maximum(sm,1e-30)
    gains=interp_fas(SMOOTH_NODES,np.maximum(ratio,1e-30),ff)
    Y*=gains*bandpass(ff)[None,:]
    Y[:,0]=0;Y[:,-1]=0
    return irfft(Y,n=n,axis=-1)

def analytic_rotate(a,rng):
    theta=rng.uniform(-np.pi,np.pi,size=len(a))
    return np.real(hilbert(a,axis=-1)*np.exp(1j*theta[:,None]))

def decorrelate(a):
    a=np.asarray(a,float).copy();a-=a.mean(axis=-1,keepdims=True)
    if a.ndim!=2 or a.shape[0] not in (2,3):raise ValueError('Covariance correction requires 2 or 3 components')
    h=a[0];z=hilbert(a[1]);p=np.dot(h,z.real);q=np.dot(h,z.imag)
    theta=np.arctan2(p,q)
    a[1]=np.real(z*np.exp(1j*theta))
    a-=a.mean(axis=-1,keepdims=True)
    if a.shape[0]==2:
        corr=np.corrcoef(a);err=float(np.max(abs(corr-np.eye(2))))
        if not np.isfinite(err) or err>1e-9:raise ValueError('Horizontal covariance constraint failed; no seed replacement')
        return a,{'max_abs_correlation':err,'horizontal_analytic_rotation_rad':float(theta),'vertical_relative_change':None}
    before=a[2].copy();H=a[:2]
    w=abs(hilbert(before))**2
    if w.max()<=0:raise ValueError('Zero vertical signal')
    w/=w.max();B=H*w[None,:];B-=B.mean(axis=-1,keepdims=True)
    G=H@B.T;rhs=H@before
    if np.linalg.cond(G)>1e12:raise ValueError('Singular localized covariance projection')
    a[2]-=np.linalg.solve(G,rhs)@B
    a-=a.mean(axis=-1,keepdims=True)
    corr=np.corrcoef(a)
    err=float(np.max(abs(corr-np.eye(3))))
    if not np.isfinite(err) or err>1e-9:raise ValueError('Covariance constraint failed; no seed replacement')
    return a,{'max_abs_correlation':err,'horizontal_analytic_rotation_rad':float(theta),
              'vertical_relative_change':float(np.linalg.norm(a[2]-before)/np.linalg.norm(before))}

def metrics(a,dt,freq=None,target=None):
    a=np.asarray(a,float)
    corr=np.corrcoef(a);env=abs(hilbert(a,axis=-1))
    ev={}
    for label,qs in [('active',(.005,.995)),('wide',(.0001,.9999))]:
        t=energy_times((a[:2]**2).sum(axis=0),dt,qs)
        lo=int(np.floor(t[0]/dt));hi=min(len(a[0]),int(np.ceil(t[1]/dt))+1)
        ev[label]=float(np.corrcoef(env[:2,lo:hi])[0,1])
    ev['full']=float(np.corrcoef(env[:2])[0,1])
    d=[]
    for x in a:
        q=energy_times(x*x,dt)
        d.append({'D2p5_97p5_s':float(q[6]-q[0]),'D5_95_s':float(q[5]-q[1]),
                  'D5_75_s':float(q[4]-q[1]),'PGA_cm_s2':float(abs(x).max()),
                  'arias_m_per_s':float(np.pi/(2*9.80665)*trapezoid((x*.01)**2,dx=dt))})
    result={'components':dict(zip(COMPONENTS[:len(d)],d)), 'acceleration_correlation_matrix':corr.tolist(),
            'max_abs_correlation':float(np.max(abs(corr-np.eye(len(d))))),
            'horizontal_envelope_correlation':ev,'support_seconds':float((a.shape[-1]-1)*dt)}
    if target is not None:
        sm=smooth_spectrum(a,dt);ta=interp_fas(freq,target,SMOOTH_NODES)
        mask=(SMOOTH_NODES>=.2)&(SMOOTH_NODES<=8.)
        err=np.sqrt(np.mean(np.log(np.maximum(sm[:,mask],1e-30)/ta[:,mask])**2,axis=-1))
        result['FAS_RMSE_ln_0p2_8Hz']=dict(zip(COMPONENTS[:len(err)],err.tolist()))
    return result

def training_targets(a,dt=DT_NATIVE):
    """54 standardized regression targets; see docs/ML_FEATURES.md."""
    hq=energy_times((a[:2]**2).sum(axis=0),dt)
    DH=hq[6]-hq[0];h50=hq[3]
    q=np.array([energy_times(x*x,dt) for x in a])
    logD=np.log(q[:,6]-q[:,0])
    rel=((q[:,[1,2,3,4,5]]-h50)/DH).ravel()
    n=a.shape[-1];ff=rfftfreq(n,dt);A=rfft(a,axis=-1)
    result=[]
    for c in range(3):
        for f in PERSISTENCE_NODES:
            w=np.exp(-.5*(np.log(np.maximum(ff,1e-30)/f)/.25)**2);w[0]=0
            z=hilbert(irfft(A[c]*w,n=n))
            lag=max(1,int(round(1/(f*dt))))
            x=z[:-lag];y=z[lag:]
            v=np.vdot(x,y)/max(np.sqrt(np.vdot(x,x).real*np.vdot(y,y).real),1e-30)
            result.extend([float(v.real),float(v.imag)])
    return np.r_[logD,rel,result]
