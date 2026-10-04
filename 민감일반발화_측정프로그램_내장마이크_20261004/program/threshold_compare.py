# -*- coding: utf-8 -*-
"""
threshold_compare.py
민감 발화 탐지 SW - 1.3 임계값 판정 로직

[임계값 재도출 이력 — 중요]
원래 SMA2026 논문 임계값(발화속도 2.5, Jitter 0.019, Shimmer 0.096)은 오프라인
분석 스크립트(analyze_v5.praat + 한글 글자수 세기)로 잰 값 기준이었다. 실시간
SW는 STT가 없어 발화속도를 음향적 방식(de Jong & Wempe 피크 카운팅)으로 다르게
측정하므로, 같은 임계값을 그대로 쓰면 척도가 안 맞을 수 있다는 문제가 있었다.

이를 해결하기 위해:
  1. batch_extract_features.py로 원본 KtelSpeech 라벨링 데이터 전체(D60_0/D61_0/
     D62_0 세 데이터셋, sensitive 2,855개 + normal 4,363개 중 2,855개 무작위 매칭
     — 원 연구와 동일한 1:1 매칭 규모)를 지금 이 실시간 파이프라인
     (feature_extractor.py)으로 통째로 다시 측정
     [1차 시도에서는 D61_0/D62_0 폴더가 스캔 경로에서 누락되어 D60_0 분량(602개)
     만으로 재도출했었으나, 이후 세 데이터셋 전부 포함하도록 수정해 재실행함]
  2. recompute_thresholds.py로 그 결과를 K-means/GMM(k=2~5) 재클러스터링
  3. k=2가 5개 특성 대부분에서 k=3과 같거나 더 높은 실루엣 점수를 보여 k=2를
     최종으로 채택 (SMA2026 원 논문도 k=2를 최종으로 썼던 것과 동일한 근거)
  4. K-means/GMM k=2 임계값의 평균을 새 임계값으로 사용

결과(전체 데이터, sensitive 2,855 + normal 2,855): F0(185.78Hz), 침묵시간(2.83초),
Jitter(0.0194), Shimmer(0.0965)는 원본 논문 값(184Hz, 2.7초, 0.019, 0.096)과
거의 완벽하게 일치 — Praat 파라미터를 원본과 맞춘 노력이 검증됐고, 표본을
602개→2,855개로 늘려도 임계값 자체는 크게 안 변해 안정적임이 확인됐다.
반면 발화속도(3.978음절/s)는 원본(2.5)과 여전히 크게 달랐는데, 이는 예상대로
측정 방식 차이(글자수 세기 vs 음향 피크 카운팅) 때문이다.

[분류 성능 실측 결과 — 반드시 논문 한계점에 명시]
evaluate_classifier.py로 전체 데이터(sensitive 2,855 + normal 4,363, 총 7,218개)에
지금 판정 로직(3특성 중 2개 이상 투표)을 그대로 돌려 검증한 결과:
  정확도 59.5%, 정밀도 49.3%, 재현율 81.6%, F1 61.5%
  (무작위 추측의 정확도가 약 50%이므로, 이 분류기는 그보다 다소 나은 수준)
집단 평균 차이는 통계적으로 유의미(원 논문 p<0.001)하지만, 개별 발화 단위로는
민감/일반 분포가 상당히 겹쳐서 단일 발화 판별력에는 뚜렷한 한계가 있다.
로지스틱회귀/랜덤포레스트 등 결합모델을 시도했을 때 F1이 소폭 개선되는 것도
확인했으나(52%대), 특허 청구항이 "군집분석 기반 임계값 도출" 방식으로 이미
확정되어 있어 실제 SW 구현은 이 방식(독립적 임계값 비교 + 다수결)을 유지하기로
결정했다 — 성능 한계는 논문/특허 모두에 정직하게 기재할 것.

판정 규칙: 3개 특성 중 2개 이상이 "민감 발화 방향"으로 임계값을 넘으면 민감 발화로 판정.
(단일 특성만으로 판정하면 잡음/개인차로 오탐이 잦아지므로 다수결 방식 채택)

방향 확인(재클러스터링 결과에서도 동일하게 재확인됨):
- 민감 발화는 발화속도가 느리고, Jitter가 낮고, Shimmer가 낮은 경향 (일반 발화 대비)
  → 즉 "임계값보다 작을 때" 민감 발화 방향
"""

from dataclasses import dataclass


# [재도출된 임계값] — 지금 이 실시간 파이프라인(feature_extractor.py)으로 전체
# 데이터셋(sensitive 2,855 + normal 2,855)을 다시 측정한 K-means/GMM k=2 평균값.
# 위 docstring의 "임계값 재도출 이력" 참고.
# 원본 SMA2026 값(오프라인 스크립트 기준, 참고용으로 남겨둠): 발화속도 2.5, Jitter 0.019, Shimmer 0.096
SPEECH_RATE_THRESHOLD = 3.978  # 음절/초 (미만이면 민감 발화 방향) — 원본 대비 측정방식 차이로 크게 변경됨
JITTER_THRESHOLD = 0.0194      # 미만이면 민감 발화 방향 — 원본(0.019)과 거의 일치
SHIMMER_THRESHOLD = 0.0965     # 미만이면 민감 발화 방향 — 원본(0.096)과 거의 일치

MIN_VOTES_REQUIRED = 2        # 3개 중 몇 개 이상 넘어야 민감 발화로 판정할지


@dataclass
class AcousticFeatures:
    """1.2 특성추출 모듈의 출력. F0/침묵시간도 참고용으로 들고 있지만 판정에는 쓰지 않는다."""
    speech_rate: float   # 음절/초
    jitter: float
    shimmer: float
    f0: float = None         # 참고용 (판정 미사용)
    silence_duration: float = None  # 참고용 (판정 미사용)


@dataclass
class JudgmentResult:
    is_sensitive: bool
    votes: int                 # 몇 개 특성이 임계값을 넘었는지
    details: dict               # 특성별 판정 결과 (True=민감 방향)


def judge(features: AcousticFeatures) -> JudgmentResult:
    """3개 특성(발화속도, Jitter, Shimmer)을 SMA2026 임계값과 비교해 민감 발화 여부를 판정한다."""
    details = {
        "speech_rate": features.speech_rate < SPEECH_RATE_THRESHOLD,
        "jitter": features.jitter < JITTER_THRESHOLD,
        "shimmer": features.shimmer < SHIMMER_THRESHOLD,
    }
    votes = sum(details.values())
    is_sensitive = votes >= MIN_VOTES_REQUIRED

    return JudgmentResult(is_sensitive=is_sensitive, votes=votes, details=details)


if __name__ == "__main__":
    examples = [
        AcousticFeatures(speech_rate=2.0, jitter=0.015, shimmer=0.090),  # 3개 다 민감 방향
        AcousticFeatures(speech_rate=3.0, jitter=0.015, shimmer=0.090),  # 2개만 민감 방향
        AcousticFeatures(speech_rate=3.0, jitter=0.025, shimmer=0.120),  # 0개 민감 방향
    ]
    for i, f in enumerate(examples, 1):
        r = judge(f)
        print(f"예시{i}: is_sensitive={r.is_sensitive}, votes={r.votes}/3, details={r.details}")
