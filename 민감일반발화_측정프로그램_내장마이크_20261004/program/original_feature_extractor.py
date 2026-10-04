# -*- coding: utf-8 -*-
"""Acoustic feature extraction without STT (auto45-fix1).
Speech rate is an acoustic proxy, not a verified Korean syllable count.
The intensity-peak heuristic uses valley separation and voiced-peak filtering.
It is inspired by de Jong & Wempe, not an exact validated reimplementation.
Jitter/Shimmer settings are unchanged. Real speech accuracy is unverified.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call


class FeatureExtractionError(Exception):
    """발화가 너무 짧거나 무성음이라 특성을 안정적으로 뽑을 수 없을 때 발생."""
    pass


def _estimate_syllable_count(sound: parselmouth.Sound, filtered=False) -> int:
    """Count separated, voiced intensity peaks; experimental rate proxy."""
    minimum_pitch_for_intensity = 50  # 논문 Step 1
    intensity = sound.to_intensity(minimum_pitch=minimum_pitch_for_intensity)
    intensity_values = intensity.values[0]

    median_intensity = np.median(intensity_values)
    db_threshold = 2.0 if filtered else 0.0
    db_min_dip = 4.0 if filtered else 2.0
    silence_threshold = median_intensity + db_threshold

    # Independent peaks must have a sufficient intervening valley. Do not
    # reuse an earlier peak's height for a later, smaller local fluctuation.
    candidates = []
    for i in range(1, len(intensity_values) - 1):
        if (intensity_values[i] > silence_threshold
                and intensity_values[i] > intensity_values[i-1]
                and intensity_values[i] >= intensity_values[i+1]):
            candidates.append(i)
    selected = []
    for i in candidates:
        if not selected:
            selected.append(i)
            continue
        previous = selected[-1]
        valley = np.min(intensity_values[previous:i+1])
        if min(intensity_values[previous], intensity_values[i]) - valley >= db_min_dip:
            selected.append(i)
        elif intensity_values[i] > intensity_values[previous]:
            selected[-1] = i

    # Intensity peaks in unvoiced regions must not count as syllable nuclei.
    pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
    return sum(bool(np.isfinite(hz) and hz > 0)
               for hz in (pitch.get_value_at_time(intensity.x1 + i * intensity.dx)
                          for i in selected))


def extract_features(audio: np.ndarray, samplerate: int,
                      apply_undercounting_correction: bool = False):
    """오디오 배열(1단계에서 넘어온 발화 구간)에서 음향 특성을 추출한다.

    Returns: dict {speech_rate, jitter, shimmer, f0, silence_duration}
    Raises: FeatureExtractionError

    apply_undercounting_correction: True면 de Jong & Wempe(2009)가 보고한
        과소추정 보정계수(약 1.28)를 음절수에 곱한다. 이 값은 네덜란드어
        코퍼스로 도출된 것이라 한국어에 그대로 맞는지 검증되지 않았으므로
        기본값은 False — 실측 후 판단해서 켤지 결정할 것.

    [duration_sec < 0.1 체크의 위치 — 근거 보완]
    1단계(audio_input.py)의 UtteranceSegmenter가 이미 min_utterance_sec(0.5초)
    미만의 발화를 걸러서 넘기므로, 정상 파이프라인에서는 이 체크에 걸릴 일이 없다.
    이 체크는 이 모듈을 단독으로(테스트 등에서) 호출할 때를 위한 방어적 안전장치일
    뿐이다. 0.1초라는 구체적 숫자 자체는 여전히 임의값이지만, 실제로 걸러내는
    역할은 1단계가 이미 하고 있다는 점을 명확히 해둔다.
    """
    duration_sec = len(audio) / samplerate
    if duration_sec < 0.1:
        raise FeatureExtractionError(f"발화가 너무 짧음: {duration_sec:.3f}초")

    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=samplerate)

    # --- F0 (기본주파수) ---
    # [원본 analyze_v5.praat과 대조] "To Pitch: 0, 75, 600" — 명시적으로 맞춰서 씀
    # (parselmouth 기본값과 같을 것으로 추정되지만, 암묵적 의존 대신 명시)
    F0_MIN, F0_MAX = 75, 600
    pitch = sound.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0_values = pitch.selected_array['frequency']
    f0_voiced = f0_values[f0_values > 0]
    if len(f0_voiced) == 0:
        raise FeatureExtractionError("유성음 구간을 찾지 못해 F0를 계산할 수 없음")
    f0_mean = float(np.mean(f0_voiced))

    # --- PointProcess (Jitter/Shimmer 계산의 기반이 되는 성문 펄스 지점들) ---
    # [수정 이력] 원본 analyze_v5.praat과 대조한 결과, 원본은
    # "To PointProcess (periodic, cc): 75, 600"으로 F0 상한을 600Hz까지 잡았는데
    # 여기서는 500Hz로 잘못 썼었다. Jitter/Shimmer local 파라미터 자체는 원본과
    # 정확히 일치했지만(0,0,0.0001,0.02,1.3 / +1.6), 성문 펄스를 뽑는 이 범위가
    # 다르면 그 위에서 계산되는 Jitter/Shimmer 값도 달라질 수 있어 600으로 맞췄다.
    point_process = call(sound, "To PointProcess (periodic, cc)", F0_MIN, F0_MAX)

    n_points = call(point_process, "Get number of points")
    # [n_points 최소 기준 근거 — 업데이트]
    # 임상 음성분석 커뮤니티의 NCVS(National Center for Voice and Speech) 가이드라인은
    # 신뢰할 만한 Jitter/Shimmer 측정에 약 100주기(cycle)가 필요하다고 명시한다
    # (지속모음 3~5초 발성 기준). 원래 썼던 n_points<3은 이 기준의 1/30도 안 되는
    # 터무니없이 느슨한 값이었다.
    # 다만 NCVS 100주기 기준은 "지속모음을 늘여서 발성"하는 임상 상황 기준이고,
    # 문헌들은 자연스러운 연결발화(connected speech)에서는 무성음·묵음이 섞여
    # 있어 지속모음 대비 신뢰도가 떨어진다는 것을 명시적 한계로 인정한다. 우리
    # SW는 지속모음이 아닌 자연발화 전체를 다루므로 100주기를 요구하면 거의 모든
    # 발화가 걸러진다. 따라서 NCVS 기준(100)과 원래값(3) 사이에서 30으로 절충했다.
    # 이 절충 자체는 여전히 완전히 검증된 값은 아니며, 실제 마이크 데이터로
    # Jitter/Shimmer 값의 변동성을 확인하며 재조정이 필요하다.
    MIN_PULSE_POINTS = 30
    if n_points < MIN_PULSE_POINTS:
        raise FeatureExtractionError(
            f"성문 펄스 포인트가 너무 적어({n_points}개, 최소 {MIN_PULSE_POINTS}개 필요) "
            f"Jitter/Shimmer를 신뢰성 있게 계산할 수 없음"
        )

    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer_local = call(
        [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    if jitter_local is None or (isinstance(jitter_local, float) and np.isnan(jitter_local)):
        raise FeatureExtractionError("Jitter 계산 결과가 유효하지 않음(NaN)")
    if shimmer_local is None or (isinstance(shimmer_local, float) and np.isnan(shimmer_local)):
        raise FeatureExtractionError("Shimmer 계산 결과가 유효하지 않음(NaN)")

    syllable_count = _estimate_syllable_count(sound)
    if apply_undercounting_correction:
        syllable_count = syllable_count * 1.28  # de Jong & Wempe(2009) 보정계수 (네덜란드어 코퍼스 기준)
    speech_rate = syllable_count / duration_sec if duration_sec > 0 else 0.0

    # [침묵시간 계산 — 원본과 정확히 일치시킴]
    # 원본 analyze_v5.praat을 확보한 뒤, 자체 구현한 프레임카운팅 대신 원본이 쓴
    # Praat 내장 명령 "To TextGrid (silences)"를 그대로 쓰도록 교체했다.
    # 원본 파라미터: Intensity(minimum_pitch=100, time_step=0[자동], subtract_mean=yes),
    # silence_threshold=-25dB, min_silent_interval=0.1초, min_sounding_interval=0.1초.
    # 이 값은 여전히 3단계(threshold_compare.py) 판정에는 쓰이지 않는 참고용이지만,
    # 실제 데이터와 구간 분리 방식에 따른 측정 검증은 별도로 필요하다.
    intensity = call(sound, "To Intensity", 100, 0, "yes")
    textgrid = call(intensity, "To TextGrid (silences)", -25, 0.1, 0.1, "silent", "sounding")
    n_intervals = call(textgrid, "Get number of intervals", 1)
    silence_duration = 0.0
    for j in range(1, n_intervals + 1):
        label = call(textgrid, "Get label of interval", 1, j)
        if label == "silent":
            t_start = call(textgrid, "Get start time of interval", 1, j)
            t_end = call(textgrid, "Get end time of interval", 1, j)
            silence_duration += (t_end - t_start)

    return {
        "speech_rate": speech_rate,
        "jitter": float(jitter_local),
        "shimmer": float(shimmer_local),
        "f0": f0_mean,
        "silence_duration": silence_duration,
        "_syllable_count": syllable_count,
        "_duration_sec": duration_sec,
    }


if __name__ == "__main__":
    sr = 16000
    t = np.arange(int(sr * 1.0)) / sr
    f0 = 150 + 10 * np.sin(2 * np.pi * 3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    audio = (0.3 * np.sin(phase)).astype(np.float32)

    try:
        result = extract_features(audio, sr)
        print("추출 성공:", {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in result.items()})
    except FeatureExtractionError as e:
        print("추출 실패:", e)
