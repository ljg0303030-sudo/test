# Frozen boundary definitions from speech_rate_review_20260925/engine.py
import numpy as np

SETTINGS=dict(frame_ms=10,hop_ms=5,anchor_run_ms=20,join_gap_ms=80,
    high_noise_margin_db=10,high_signal_drop_db=30,low_noise_margin_db=3,
    low_signal_drop_db=45,guard_ms=10)

def validate(y,sr):
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or len(y)==0 or not np.isfinite(y).all() or sr<=0:raise ValueError('Invalid mono signal or sample rate')
    return y

def bounds(y,sr):
    """Canonicalize digital padding, estimate noise and expand weak edge activity.
    Energy is not semantic speech detection. Internal pauses never removed.
    """
    y=validate(y,sr);nz=np.flatnonzero(y!=0)
    if not len(nz):raise ValueError('All-zero audio: no speech boundary')
    origin=int(nz[0]);x=y[origin:int(nz[-1])+1]
    win=round(sr*.010);hop=round(sr*.005)
    if len(x)<win:raise ValueError('Too short')
    starts=np.arange(0,len(x)-win+1,hop)
    frames=np.lib.stride_tricks.sliding_window_view(x,win)[::hop]
    rms=np.sqrt(np.mean(frames*frames,axis=1))
    db=20*np.log10(np.maximum(rms,1e-12))
    floor=float(np.percentile(db,10));signal=float(np.percentile(db,95))
    high=max(floor+10,signal-30);low=max(floor+3,signal-45)
    strong=db>=high;minimum=max(1,round(.020/(hop/sr)))
    runs=[];i=0
    while i<len(strong):
        if not strong[i]:i+=1;continue
        j=i+1
        while j<len(strong) and strong[j]:j+=1
        if j-i>=minimum:runs.append((i,j-1))
        i=j
    if not runs:raise ValueError('No persistent activity anchor: manual review needed')
    left,right=runs[0][0],runs[-1][1]
    gap=max(1,round(.080/(hop/sr)))
    # Expand through nearby low-energy activity, permitting short gaps at edges.
    while left>0:
        weak=np.flatnonzero(db[max(0,left-gap):left]>=low)
        if not len(weak):break
        left=max(0,left-gap)+int(weak[0])
    while right<len(db)-1:
        weak=np.flatnonzero(db[right+1:min(len(db),right+gap+1)]>=low)
        if not len(weak):break
        right=right+1+int(weak[-1])
    guard=round(sr*.010)
    a=max(0,origin+int(starts[left])-guard)
    b=min(len(y),origin+int(starts[right])+win+guard)
    warnings=['AUTO_BOUNDARY_UNCONFIRMED','ENERGY_CANNOT_DISTINGUISH_WEAK_SPEECH_FROM_NOISE']
    if signal-floor<20:warnings.append('LOW_ENERGY_CONTRAST_NOT_SNR')
    if a==0 or b==len(y):warnings.append('ACTIVITY_TOUCHES_RECORDING_EDGE')
    return dict(start_sample=a,end_sample=b,start_sec=a/sr,end_sec=b/sr,duration_sec=(b-a)/sr,
        noise_proxy_dbfs=floor,signal_proxy_dbfs=signal,high_dbfs=high,low_dbfs=low,
        anchor_start_sec=(origin+int(starts[runs[0][0]]))/sr,
        anchor_end_sec=(origin+int(starts[runs[-1][1]])+win)/sr,
        uncertainty_start_sec=[max(0,a/sr-.080),min(len(y)/sr,a/sr+.080)],
        uncertainty_end_sec=[max(0,b/sr-.080),min(len(y)/sr,b/sr+.080)],warnings=warnings,
        method='energy_hysteresis_v1',settings=SETTINGS)
