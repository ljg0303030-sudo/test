# -*- coding: utf-8 -*-
"""
personal_baseline.py
민감 발화 탐지 SW - 개인별 목소리 기준선 적응 판정

[배경] 문헌상 Jitter/Shimmer는 화자 개개인의 특성이 뚜렷해서(화자인식에도
쓰일 정도) "민감 발화 vs 일반 발화"의 차이보다 "사람 A vs 사람 B"의 차이가
더 클 수 있다. SMA2026의 모집단 임계값(수많은 화자를 평균낸 값)은 집단
수준에서는 유의미했지만, 특정 개인 한 명에게 그대로 적용하면 그 사람의
타고난 목소리 특성 때문에 항상 한쪽으로만 판정될 수 있다(실측으로 확인됨).

[해결 방식] audio_input.py의 NoiseFloorEstimator와 같은 구조:
  - 처음엔 비교할 개인 기준이 없으니 SMA2026 모집단 임계값(threshold_compare.py)을
    그대로 쓴다.
  - "일반 발화로 판정된" 발화만 개인 기준선에 누적한다 (민감 발화까지 섞으면
    기준선이 계속 낮은 쪽으로 끌려가는 피드백 루프가 생기므로 제외).
  - 표본이 min_samples 이상 쌓이면, 그때부터는 모집단 임계값 대신 "이 사람의
    개인 기준선 대비 얼마나 벗어났는지"로 판정한다.
  - 기준점은 평균/표준편차가 아니라 중앙값(median)/MAD(중앙값절대편차)를 쓴다.
    표본이 적을 때(수십 개) 이상치 하나(헛기침, 잡음 섞인 발화 등)에 평균이
    쉽게 왜곡되는 걸 막기 위함. 다만 중앙값/MAD도 "표본이 원래 적다"는
    근본 문제를 없애주진 않는다 — 그래서 min_samples를 넉넉히(기본 15) 잡았다.
  - 슬라이딩 윈도우(max_samples, 기본 200)로 오래된 표본은 자연히 빠지게 해서,
    시간이 지나며 목소리 상태(피로, 환경 변화 등)가 바뀌어도 어느 정도 추적한다.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque

import threshold_compare as population


@dataclass
class JudgmentResult:
    is_sensitive: bool | None
    votes: int
    details: dict
    scores: dict = field(default_factory=dict)
    # mode identifies the reference baseline.
    mode: str = "personal"  # "population"(모집단 임계값 사용) 또는 "personal"(개인 기준선 사용)


class PersonalBaseline:
    def __init__(self, min_samples=15, max_samples=200, z_threshold=1.0,
                 bootstrap_unconditional=10, freeze_after_ready=False):
        """
        freeze_after_ready: True이면 최소 표본 수 도달 후 갱신을 중단한다.
            GUI 수집량 비교 실험에서 사용하며 기존 콘솔 기본값은 False.
        min_samples: 개인 기준선으로 전환하기 전 최소 표본 수.
        max_samples: 슬라이딩 윈도우 크기 (이보다 오래된 표본은 버림).
        z_threshold: modified z-score가 이 값보다 낮으면(더 음정면) "민감 방향".
        bootstrap_unconditional: 처음 이 개수만큼은 판정 결과와 무관하게
            무조건 기준선에 반영한다.

            [왜 필요한가 — 실제로 시뮬레이션하다 발견한 교착상태]
            "일반로 판정된 발화만 기준선에 넣는다"는 규칙만 있으면, 만약 모집단
            임계값이 이 사람에게 처음부터 안 맞아서 거의 매번 "민감"으로 잘못
            판정한다면(예: 원래 목소리가 모집단 평균보다 Jitter가 낮은 사람),
            "일반 판정"이 나올 기회 자체가 거의 없어서 표본이 min_samples까지
            영원히 못 쌓이고 계속 모집단 모드에 갇히는 문제가 생긴다. 이를
            막기 위해 초반 몇 개는 무조건 기준선에 반영해 이 교착상태를 벗어난다.
            (일상 대화 대부분은 민감하지 않다는 전제 하의 절충 — 만약 정말
            하필 이 초반 구간에 민감한 내용을 말하면 기준선이 약간 오염될
            위험은 있으나, 표본이 계속 쌓이며 희석된다)
        """
        if min_samples < 1 or max_samples < min_samples:
            raise ValueError("max_samples must be >= min_samples >= 1")
        self.freeze_after_ready = freeze_after_ready
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.z_threshold = z_threshold
        # bootstrap_unconditional이 min_samples보다 작으면, 부트스트랩이 끝난 뒤에도
        # "일반 판정"이 나올 때만 표본이 늘어야 하는데, 모집단 임계값이 이 사람에게
        # 안 맞을 경우 그 판정 자체가 거의 안 나와서 다시 교착상태에 빠질 수 있다.
        # 따라서 bootstrap_unconditional은 최소 min_samples만큼은 확보되도록 강제한다.
        self.bootstrap_unconditional = max(bootstrap_unconditional, min_samples)
        self._total_seen = 0
        self._speech_rate = deque(maxlen=max_samples)
        self._jitter = deque(maxlen=max_samples)
        self._shimmer = deque(maxlen=max_samples)

    @property
    def n_samples(self) -> int:
        return len(self._jitter)

    @property
    def is_ready(self) -> bool:
        return self.n_samples >= self.min_samples

    @staticmethod
    def _modified_zscore(value, samples):
        arr = np.array(samples)
        median = np.median(arr)
        mad = np.median(np.abs(arr - median))
        if not np.isfinite(mad) or mad <= np.finfo(float).eps * max(1., abs(median)):
            # No defensible dispersion estimate: abstain for this feature.
            return float('nan')
        return 0.6745 * (value - median) / mad

    def judge(self, features) -> JudgmentResult:
        """features: threshold_compare.AcousticFeatures와 동일한 필드(speech_rate, jitter, shimmer)."""
        if not self.is_ready:
            pop_result = population.judge(features)
            return JudgmentResult(
                is_sensitive=pop_result.is_sensitive,
                votes=pop_result.votes,
                details=pop_result.details,
                mode=f"population (개인표본 {self.n_samples}/{self.min_samples})",
            )

        z_speech_rate = self._modified_zscore(features.speech_rate, self._speech_rate)
        z_jitter = self._modified_zscore(features.jitter, self._jitter)
        z_shimmer = self._modified_zscore(features.shimmer, self._shimmer)

        details = {
            "speech_rate": z_speech_rate < -self.z_threshold,
            "jitter": z_jitter < -self.z_threshold,
            "shimmer": z_shimmer < -self.z_threshold,
        }
        votes = sum(details.values())
        scores = dict(speech_rate=float(z_speech_rate), jitter=float(z_jitter), shimmer=float(z_shimmer))
        unavailable = sum(not np.isfinite(v) for v in scores.values())
        if votes >= population.MIN_VOTES_REQUIRED:
            is_sensitive = True
        elif votes + unavailable < population.MIN_VOTES_REQUIRED:
            is_sensitive = False
        else:
            is_sensitive = None  # Missing dispersion could change the decision.
        return JudgmentResult(is_sensitive=is_sensitive, votes=int(votes), details=details,
                              mode="personal", scores=scores)

    def update(self, features, judgment: JudgmentResult):
        """일반 발화로 판정된 경우, 또는 아직 부트스트랩 구간(bootstrap_unconditional)
        안이면 판정 결과와 무관하게 개인 기준선에 반영한다."""
        # GUI experiments freeze the initial N valid observations.
        if self.freeze_after_ready and self.is_ready:
            return
        self._total_seen += 1
        in_bootstrap = self._total_seen <= self.bootstrap_unconditional
        if judgment.is_sensitive and not in_bootstrap:
            return
        self._speech_rate.append(features.speech_rate)
        self._jitter.append(features.jitter)
        self._shimmer.append(features.shimmer)


if __name__ == "__main__":
    # 검증: "타고나길 Jitter가 낮은 화자"를 흉내낸 합성 데이터로,
    # 모집단 임계값으로는 계속 민감판정만 나오다가, 개인기준선이 쌓이면
    # 그 사람 내에서도 민감/일반이 갈리기 시작하는지, 그리고 실제 정답 대비
    # 정확도가 나아지는지 확인
    from threshold_compare import AcousticFeatures
    import threshold_compare as population

    rng = np.random.default_rng(0)
    baseline = PersonalBaseline(min_samples=15, max_samples=200, z_threshold=1.0)

    print("이 화자는 원래 Jitter가 낮은 편(개인 평균 0.014)이라고 가정 (모집단 임계값 0.0194보다 항상 낮음)")
    print(f"{'번호':<5}{'jitter':<10}{'실제정답':<8}{'모집단판정':<10}{'개인기준판정':<12}{'모드'}")

    pop_correct, personal_correct, personal_n = 0, 0, 0
    n_sensitive_true = 0
    for i in range(60):
        is_truly_sensitive = rng.random() < 0.3
        if is_truly_sensitive:
            n_sensitive_true += 1
            jitter = rng.normal(0.010, 0.002)
            shimmer = rng.normal(0.070, 0.01)
        else:
            jitter = rng.normal(0.014, 0.002)  # 이 화자의 평소 수준(모집단 임계값보다 항상 낮음)
            shimmer = rng.normal(0.090, 0.01)
        speech_rate = rng.normal(3.5, 0.3)

        feats = AcousticFeatures(speech_rate=speech_rate, jitter=jitter, shimmer=shimmer)
        pop_result = population.judge(feats)
        personal_result = baseline.judge(feats)

        if pop_result.is_sensitive == is_truly_sensitive:
            pop_correct += 1
        if personal_result.mode == "personal":
            personal_n += 1
            if personal_result.is_sensitive == is_truly_sensitive:
                personal_correct += 1

        print(f"{i+1:<5}{jitter:<10.4f}{'민감' if is_truly_sensitive else '일반':<8}"
              f"{'민감' if pop_result.is_sensitive else '일반':<10}"
              f"{'민감' if personal_result.is_sensitive else '일반':<12}{personal_result.mode}")

        baseline.update(feats, personal_result)

    print(f"\n(실제 민감 비율: {n_sensitive_true}/60)")
    print(f"모집단 임계값 정확도(전체 60개): {pop_correct}/60 = {pop_correct/60:.1%}")
    if personal_n > 0:
        print(f"개인기준선 정확도(전환 이후 {personal_n}개만): {personal_correct}/{personal_n} = {personal_correct/personal_n:.1%}")


class ProgressiveBaseline(PersonalBaseline):
    """Collect exactly 45 valid samples; automatically freeze before sample 46."""
    def __init__(self):
        super().__init__(min_samples=45)
        self._speech_rate = deque()
        self._jitter = deque()
        self._shimmer = deque()
        self.collecting = True
        self.checkpoints = {}

    def summary(self):
        return {name: {'median': float(np.median(values)),
                       'mad': float(np.median(np.abs(np.asarray(values)-np.median(values))))}
                for name, values in [('speech_rate', self._speech_rate),
                                     ('jitter', self._jitter), ('shimmer', self._shimmer)]
                if values}

    def collect(self, features):
        if not self.collecting:
            raise RuntimeError('수집이 종료되어 기준선이 고정되었습니다.')
        values = (features.speech_rate, features.jitter, features.shimmer)
        if not all(np.isfinite(v) and v >= 0 for v in values):
            raise ValueError('유효하지 않은 음향 특성입니다.')
        for samples, value in zip((self._speech_rate, self._jitter, self._shimmer), values):
            samples.append(value)
        if self.n_samples in (15, 30, 45):
            self.checkpoints[self.n_samples] = self.summary()
        if self.n_samples == 45:
            self.freeze()

    def freeze(self):
        if not self.is_ready:
            raise ValueError('유효 발화 45개를 수집해야 합니다.')
        self.collecting = False

    def judge(self, features):
        if self.collecting:
            raise RuntimeError('수집 중에는 민감 여부를 판정하지 않습니다.')
        return super().judge(features)

    def update(self, features, judgment):
        # Evaluation data must never change the collected baseline.
        return
