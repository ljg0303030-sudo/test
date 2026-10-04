"""Runtime measurement guard for the original console. No threshold changes."""
import numpy as np
import parselmouth
from parselmouth.praat import call
from original_feature_extractor import extract_features, FeatureExtractionError
from threshold_compare import AcousticFeatures
from measurement_boundary import bounds

CONTEXT_PADDING=(0,.025,.05,.1,.25,.5)

def measure_variants(audio,sr):
    y=np.asarray(audio,dtype=float)
    if y.ndim!=1 or not len(y) or not np.isfinite(y).all():raise ValueError('유효한 모노 입력이 필요합니다.')
    if np.any(np.abs(y)>=.999):raise ValueError('입력 포화 의심: 측정 보류')
    region=bounds(y,sr);a,b=region['start_sample'],region['end_sample']
    features=extract_features(y[a:b],sr)
    whole=parselmouth.Sound(y,sampling_frequency=sr)
    variants=[];unique=set()
    for pad in CONTEXT_PADDING:
        lo=max(0,a-round(pad*sr));hi=min(len(y),b+round(pad*sr));unique.add((lo,hi))
        sub=parselmouth.Sound(y[lo:hi],sampling_frequency=sr)
        pp=call(sub,'To PointProcess (periodic, cc)',75,600)
        mapped=call('Create empty PointProcess','runtime',0,len(y)/sr)
        n=0
        for i in range(1,int(call(pp,'Get number of points'))+1):
            t=call(pp,'Get time from index',i)+lo/sr;call(mapped,'Add point',t)
            n+=int(a/sr<=t<=b/sr)
        if n<30:raise ValueError('평가 구간의 주기 지점 부족')
        j=float(call(mapped,'Get jitter (local)',a/sr,b/sr,.0001,.02,1.3))
        s=float(call([whole,mapped],'Get shimmer (local)',a/sr,b/sr,.0001,.02,1.3,1.6))
        if not np.isfinite([j,s,features['speech_rate']]).all():raise ValueError('유효하지 않은 특징값')
        variants.append(dict(context_start_sec=lo/sr,context_end_sec=hi/sr,padding_sec=pad,
            features=AcousticFeatures(features['speech_rate'],j,s,features['f0'],features['silence_duration'])))
    # Include exact production extraction in addition to the mapped-context queries.
    original=AcousticFeatures(**{k:features[k] for k in ['speech_rate','jitter','shimmer','f0','silence_duration']})
    return dict(features=features,original=original,region=region,variants=variants,unique_contexts=len(unique))

def judge_and_update(measured,baseline):
    if len(measured['variants'])!=len(CONTEXT_PADDING) or measured['unique_contexts']<2:
        return dict(status='withheld',is_sensitive=None,reason='서로 다른 주변 구간 확보 부족',baseline_updated=False)
    variants=[measured['original']]+[r['features'] for r in measured['variants']]
    if any(not np.isfinite([v.speech_rate,v.jitter,v.shimmer]).all() for v in variants):
        return dict(status='withheld',is_sensitive=None,reason='유효하지 않은 측정값',baseline_updated=False)
    # All judgments use the SAME pre-update personal baseline.
    decisions=[baseline.judge(v) for v in variants]
    unstable=[k for k in ['speech_rate','jitter','shimmer'] if len({bool(d.details[k]) for d in decisions})>1]
    if unstable:
        return dict(status='withheld',is_sensitive=None,reason='측정 문맥에 따라 표가 바뀜: '+', '.join(unstable),baseline_updated=False)
    chosen=decisions[0];before=baseline.n_samples
    baseline.update(measured['original'],chosen)
    return dict(status='provisional',is_sensitive=bool(chosen.is_sensitive),reason='시험 문맥의 표 일치. 분류 정확성 미검증',
        mode=chosen.mode,votes=int(chosen.votes),details={k:bool(v) for k,v in chosen.details.items()},
        baseline_updated=baseline.n_samples!=before,classification_validated=False)

def process(audio,sr,baseline):
    try:
        measured=measure_variants(audio,sr)
        verdict=judge_and_update(measured,baseline)
        return dict(**verdict,features=measured['features'],region=measured['region'],unique_contexts=measured['unique_contexts'])
    except (ValueError,RuntimeError,FeatureExtractionError,parselmouth.PraatError) as e:
        return dict(status='withheld',is_sensitive=None,reason=str(e),baseline_updated=False)
