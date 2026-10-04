"""Reject context-dependent votes before any baseline mutation or model display."""
from comparison_model import ComparisonModel
import threshold_compare

def check_measurement(features,baseline):
    m=features.get('measurement')
    if m is None:
        raise ValueError('측정 안정성 검사 결과 누락')
    if m['unique_contexts']<2 or len(m['variants'])!=6:
        raise ValueError('주변 구간 부족: 측정 안정성 확인 불가')
    variants=[m['original']]+[v['features'] for v in m['variants']]
    judge=threshold_compare.judge if getattr(baseline,'collecting',False) else baseline.judge
    decisions=[judge(v) for v in variants]
    if any(len({d.details[k] for d in decisions})>1 for k in ('speech_rate','jitter','shimmer')):
        raise ValueError('주변 분석 구간에 따라 특징별 판단이 달라짐')
    if any(d.is_sensitive is None for d in decisions):
        raise ValueError('개인 기준선의 변동 정보 부족')
    if not getattr(baseline,'collecting',False):
        model=ComparisonModel()
        predictions=[model.predict(vars(v)) for v in variants]
        if any(len({v[k] for v in predictions})>1 for k in ('model_default','model_relaxed')):
            raise ValueError('주변 분석 구간에 따라 비교 모델 판단이 달라짐')
