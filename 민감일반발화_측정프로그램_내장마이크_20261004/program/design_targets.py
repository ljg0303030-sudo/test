# -*- coding: utf-8 -*-
"""
design_targets.py
민감 발화 탐지 SW - 전체 파이프라인 설계 목표치 및 설계 원칙

WSCoach(Zhang et al., 2025, ACM IMWUT - "Wearable Real-time Auditory Feedback
for Reducing Unwanted Words in Daily Communication")에서 가져온 벤치마크와
설계 원칙을 정리한다. 5단계(전체 파이프라인 통합 및 실측 테스트)에서 이 값들과
비교 검증한다.
"""

TARGET_END_TO_END_LATENCY_SEC = 1.1
MAX_ACCEPTABLE_FEEDBACK_DELAY_SEC = 2.0
REFERENCE_FEEDBACK_DURATION_SEC = 1.0

DISCARD_RAW_AUDIO_AFTER_PROCESSING = True  # 항상 True 유지 (설계 불변조건)

KNOWN_LIMITATION_NO_SPEAKER_DIARIZATION = (
    "단일 마이크 기반으로 화자분리 기능이 없어, 주변 사람의 발화도 사용자 발화로 "
    "오인식될 수 있음. WSCoach(Zhang et al., 2025)도 동일한 한계를 보고함."
)
