"""Shared ECG preprocessing for CinC2017 (300 Hz) and CPSC2018 (500 Hz).

Both dataset profiles apply training augmentation after filtering and
normalization. Dataset-specific augmentation strengths are retained.
All filtering is offline and noncausal.
"""
import numpy as np
from scipy import signal


def preprocess_ecg(x, duration, window_length, preprocessing='corrected', training=False, seed=42, fs=500.,
                   augmentation_profile='cpsc2018'):
    """Deterministic numpy/scipy preprocessing; zero-phase filtering is noncausal."""
    if augmentation_profile not in ('cpsc2018', 'cinc2017'):
        raise ValueError('Unknown augmentation profile')
    if preprocessing not in ('legacy', 'corrected') or duration not in ('fixed', 'variable'):
        raise ValueError('Invalid preprocessing/duration')
    if window_length < 1 or not np.isfinite(fs) or fs <= 100:
        raise ValueError('Positive window length and sampling rate >100 Hz required for 50 Hz notch')
    x=np.asarray(x)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Invalid or too-short ECG')
    sos_hp=signal.butter(2,.5,btype='highpass',fs=fs,output='sos')
    sos_bp=signal.butter(4,[.5,40.],btype='band',fs=fs,output='sos')
    b,a=signal.iirnotch(50.,Q=30.,fs=fs);sos_notch=signal.tf2sos(b,a)
    x=np.asarray(x,dtype=np.float64).copy()
    rng=np.random.default_rng(seed)
    
    if preprocessing=='legacy':
        x=x.astype(np.float32)  # Same cast location as baseline data_common.py.
        for sos in (sos_hp,sos_notch,sos_bp):x=signal.sosfiltfilt(sos,x,axis=0)
    else:
        # Fixed inputs must not use discarded samples for filtering or normalization.
        if duration=='fixed':x=x[:window_length]
        # Bandpass already includes high-pass: avoid applying the 0.5 Hz cutoff twice.
        for sos in (sos_notch,sos_bp):
            x=signal.sosfiltfilt(sos,x,axis=0,padlen=min(3*(2*len(sos)+1),len(x)-1))
    x=(x-x.mean(axis=0,keepdims=True))/(x.std(axis=0,keepdims=True)+1e-7)
    if preprocessing=='legacy' and training and augmentation_profile=='cpsc2018':
        # Apply after normalization so scaling/offset survive preprocessing.
        # Preserve the original CPSC probabilities and amplitudes.
        if rng.random()<.5:x=x*rng.normal(1.,.1,size=(1,x.shape[1]))
        if rng.random()<.5:x=x+rng.integers(-20,20,size=(1,x.shape[1]))/1000.
    if preprocessing=='corrected' and training and augmentation_profile=='cpsc2018':
        # Mild amplitude/noise augmentation survives normalization. Preserve RR timing.
        if rng.random()<.5:x*=np.clip(rng.normal(1.,.05,size=(1,x.shape[1])),.85,1.15)
        if rng.random()<.3:x+=rng.normal(0.,.01,size=x.shape)
    if training and augmentation_profile=='cinc2017':
        x=augment_cinc2017(x, rng)
    if not np.isfinite(x).all():raise ValueError('Invalid preprocessed signal')
    return x.astype(np.float32)



def augment_cinc2017(x, rng):
    """Apply the old CinC amplitude/offset/masking AFTER normalization.

    No time warp or resampling. Keep record length, sample order and float32.
    Validation/test never call this function. No renormalization follows it.
    """
    x=np.asarray(x,dtype=np.float32).copy()
    if rng.random()<.5:
        factors=rng.normal(1.,.2,size=(1,x.shape[1])).astype(np.float32)
        x*=factors
    if rng.random()<.5:
        offsets=(rng.integers(-30,30,size=(1,x.shape[1]))/1000.).astype(np.float32)
        x+=offsets
    if rng.random()<.3:
        count=int(.01*x.shape[0])
        if count:
            indices=rng.choice(x.shape[0],size=count,replace=False)
            x[indices,:]=0.
    return x
