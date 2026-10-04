# -*- coding: utf-8 -*-
"""
audio_input.py
민감 발화 탐지 SW - 1.1 마이크 입력 모듈

실시간 마이크 스트리밍 캡처 + 에너지 기반 발화구간(VAD) 분리를 담당한다.
완성된 발화 구간(numpy 배열)을 콜백으로 다음 단계(1.2 특성추출 모듈)에 넘긴다.

[임계값 방식 변경 이력]
초기 버전은 energy_threshold를 고정 상수(0.02)로 뒀는데, 이는 마이크 게인/거리/
주변 소음에 따라 완전히 달라지는 값이라 근거로 삼기 어려웠다. Rabiner & Sambur(1975)
계열의 고전적 에너지 기반 VAD와, 이후 문헌들이 권장하는 적응형(adaptive) 임계값
방식으로 교체했다: 무음 구간의 에너지 통계(노이즈 플로어)를 실시간으로 추정하고,
그 위에 배수 마진을 둔 값을 임계값으로 쓴다. 이렇게 하면 환경이 바뀌어도(방을
옮기거나, 에어컨이 켜지는 등) 어느 정도 추적이 가능하다.
"""

import time
import numpy as np
# sounddevice(PortAudio)는 실제 마이크 하드웨어가 있는 환경에서만 필요하므로
# MicStream 클래스 내부에서 지연 import한다. 이렇게 하면 UtteranceSegmenter만
# 따로 임포트해서(마이크 없는 환경에서도) 단위 테스트할 수 있다.


class NoiseFloorEstimator:
    """무음 구간의 에너지 통계로부터 '노이즈 플로어'를 추정하고, 그 위에 마진을 둔
    적응형 음성 판정 임계값을 계산한다.

    - calibration_sec 동안은 무조건 무음으로 간주하고 노이즈 플로어의 평균/표준편차를
      추정한다 (Rabiner-Sambur 방식: 초반 무음 구간 통계로 초기 임계값을 잡는 것과 동일한 아이디어).
    - 이후에는 '무음으로 판정된 블록'만 가지고 지수이동평균(EMA)으로 노이즈 플로어를
      계속 갱신해서, 서서히 변하는 배경 소음(예: 에어컨 소음)을 추적한다.
    - 임계값 = noise_floor + margin_k * noise_std, 그리고 아주 조용한 환경에서
      noise_floor가 0에 가까워 임계값이 지나치게 낮아지는 것을 막기 위한 절대 하한(floor_min)도 둔다.
    """

    def __init__(self, calibration_sec=1.5, margin_k=3.0, ema_alpha=0.05, floor_min=0.0008):
        # [margin_k 근거 — 두 번째 교차검증]
        # 통신 VAD 특허(US6175634)의 24dB급 마진은 통신 코덱이 잡음으로 오작동하는 걸
        # 극도로 피하려는 보수적 설계다. 반면 OBS Studio 노이즈 게이트는 open/close
        # 임계값 간격을 5~8dB로 권장하고, 별개의 통신기기 특허(US7386109)도 노이즈
        # 플로어 대비 발화시작 +9dB / 발화종료 +6dB를 실제 기준값으로 쓴다. 서로 다른
        # 두 출처가 5~9dB대로 수렴하므로, margin_k=3.0(진폭 기준 약 9.5dB 상당)은
        # 이 수렴 범위 안에 있어 근거가 한층 보강됐다.
        #
        # [ema_alpha 근거]
        # 노이즈 억제 특허(US6175634)는 500ms 슬라이딩 윈도우를 20ms마다 갱신한다.
        # 여기서 쓰는 EMA(alpha=0.05)를 64ms 블록 주기로 환산하면 시간상수가 약 1.3초로
        # (=block_dur/alpha), 그 특허의 500ms보다는 느리지만 자릿수(초 단위)는 비슷하다.
        #
        # [calibration_sec 근거 — 신규]
        # Google의 개인화 키워드 인식 논문(arXiv:2104.13970)은 발화 직전 3초짜리
        # 비발화 구간으로 적응형 필터 계수를 추정한다("Speech Cleaner" 모듈). 다만
        # 그쪽은 노이즈 캔슬링 필터 계수(더 복잡한 통계량)를 추정하는 것이고, 우리는
        # 에너지의 평균/표준편차만 구하면 되므로 반드시 3초씩 필요하진 않다고 보고,
        # 원래 1.0초에서 1.5초로 다소 늘려 절충했다. 여전히 정확히 검증된 값은 아니다.
        #
        # [floor_min 근거 — 신규]
        # Dolby의 계층적 음성감지 특허(US9064503)는 정확히 같은 개념(MinAbsThresh,
        # 최소 절대 임계값)을 쓰며, 실사용값 0.000001(-60dB)에 적합 범위를
        # 0.00000001~0.001로 명시한다. 기존 floor_min=0.003은 이 범위 상한보다도
        # 높아 조용한 발화를 놓칠 위험이 있었다. 범위 상한 근처인 0.0008로 낮췄다.
        self.calibration_sec = calibration_sec
        self.margin_k = margin_k
        self.ema_alpha = ema_alpha
        self.floor_min = floor_min

        self._calibration_energies = []
        self._calibrated = False
        self.noise_mean = 0.0
        self.noise_std = 0.0
        self._calibration_elapsed = 0.0

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def current_threshold(self) -> float:
        return max(self.noise_mean + self.margin_k * self.noise_std, self.floor_min)

    def feed_calibration_block(self, energy: float, block_dur: float):
        """캘리브레이션 구간(초반 무음 가정) 동안 호출. calibration_sec에 도달하면 자동 확정."""
        self._calibration_energies.append(energy)
        self._calibration_elapsed += block_dur
        if self._calibration_elapsed >= self.calibration_sec:
            arr = np.array(self._calibration_energies, dtype=np.float64)
            self.noise_mean = float(arr.mean())
            self.noise_std = float(arr.std())
            self._calibrated = True

    def update_from_silence_block(self, energy: float):
        """캘리브레이션 이후, 무음으로 판정된 블록으로 노이즈 플로어를 서서히 갱신(EMA)."""
        self.noise_mean = (1 - self.ema_alpha) * self.noise_mean + self.ema_alpha * energy
        # 표준편차도 같은 방식으로 부드럽게 추적 (편차의 EMA)
        dev = abs(energy - self.noise_mean)
        self.noise_std = (1 - self.ema_alpha) * self.noise_std + self.ema_alpha * dev


class UtteranceSegmenter:
    """오디오 블록을 순차적으로 받아 에너지 기반으로 발화 구간을 분리하는 상태 머신.

    sounddevice 콜백과 분리된 순수 클래스로 만들어서,
    실제 마이크 없이도(테스트/시뮬레이션) 동작을 검증할 수 있게 했다.

    주의: 여기서 쓰는 '무음 판정 간격(silence_gap_sec)'은 "발화가 끝났다"고 판단하기
    위한 값이다. 실제 배포된 Silero VAD(faster-whisper 등 다수의 실시간 STT 서비스에
    탑재된 오픈소스 VAD)의 min_silence_duration_ms 기본값은 용도에 따라 100ms(실시간
    스트리밍)~2000ms(자막용 배치처리)로 편차가 크고, OpenAI Realtime API의 서버 VAD
    기본값은 500ms다. 우리가 쓰는 0.6초(600ms)는 이 범위 안에 들어가며, 실시간이지만
    순간 응답이 필요한 turn-taking은 아닌 우리 용도에 맞는 중간값으로 판단해 채택했다.
    SMA2026 논문에서 도출한 '침묵시간 임계값(~2.7초)'은 완성된 발화 구간 자체의 음향
    특성(민감발화 판정용)이라 목적이 다르므로 값을 섞어 쓰지 않는다.

    min_utterance_sec(0.5초)은 Silero VAD의 min_speech_duration_ms 기본값(250ms)보다
    2배 엄격하다. 이는 단순 "말이 있었는지" 판정이 아니라 이후 단계에서 Jitter/Shimmer를
    안정적으로 추출하려면 어느 정도 발화 길이가 필요하다는 우리 도메인 특성을 반영한
    의도적 선택이다. max_utterance_sec(15초)은 Silero 등 대부분 구현체의 기본값이
    사실상 무제한인 것과 달리, 명시적 안전장치로 넣은 값이다.

    시작 직후 calibration_sec(기본 1.5초)는 조용한 상태여야 노이즈 플로어가 정확히
    잡힌다. 이 구간 동안은 발화 감지를 하지 않는다.
    """

    def __init__(self, samplerate=16000,
                 silence_gap_sec=0.6, min_utterance_sec=0.5, max_utterance_sec=15.0,
                 calibration_sec=1.5, margin_k=3.0, ema_alpha=0.05, floor_min=0.0008):
        self.samplerate = samplerate
        self.silence_gap_sec = silence_gap_sec
        self.min_utterance_sec = min_utterance_sec
        self.max_utterance_sec = max_utterance_sec

        self.noise_floor = NoiseFloorEstimator(
            calibration_sec=calibration_sec, margin_k=margin_k,
            ema_alpha=ema_alpha, floor_min=floor_min,
        )

        self._buffer = []            # 현재 진행 중인 발화의 오디오 블록들
        self._in_speech = False
        self._silence_accum = 0.0    # 무음 누적 시간(초)
        self._elapsed_accum = 0.0
        self._trailing_samples = 0
        self._speech_accum = 0.0     # 발화 누적 시간(초)

    @staticmethod
    def _block_energy(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))

    @property
    def energy_threshold(self) -> float:
        """현재 적용 중인 적응형 임계값 (참고/로깅용)."""
        return self.noise_floor.current_threshold()

    def process_block(self, block: np.ndarray):
        """block: 1차원 float32 numpy 배열(모노).
        발화가 완결되면 numpy 배열을 반환하고, 아니면 None을 반환한다."""
        block_dur = len(block) / self.samplerate
        energy = self._block_energy(block)

        # 캘리브레이션 중에는 무조건 무음으로 간주하고 발화 감지를 하지 않는다.
        if not self.noise_floor.is_calibrated:
            self.noise_floor.feed_calibration_block(energy, block_dur)
            return None

        is_voiced = energy >= self.noise_floor.current_threshold()

        if self._in_speech or is_voiced:
            self._elapsed_accum += block_dur

        if is_voiced:
            self._trailing_samples = 0
            self._buffer.append(block)
            self._speech_accum += block_dur
            self._silence_accum = 0.0
            self._in_speech = True
        else:
            # 무음으로 판정된 블록으로 노이즈 플로어를 계속 추적(서서히 변하는 배경소음 대응)
            self.noise_floor.update_from_silence_block(energy)
            if self._in_speech:
                # 발화 중 짧은 무음은 자연스러운 파형을 위해 버퍼에 계속 포함
                self._buffer.append(block)
                self._trailing_samples += len(block)
                self._silence_accum += block_dur
                if self._silence_accum >= self.silence_gap_sec:
                    return self._finalize()

        # 안전장치: 발화가 비정상적으로 길어지면 강제 종료
        if self._in_speech and self._elapsed_accum >= self.max_utterance_sec:
            return self._finalize()

        return None

    def _finalize(self):
        utterance = np.concatenate(self._buffer) if self._buffer else np.array([], dtype=np.float32)
        # Remove endpoint-detection waiting silence, preserving internal pauses.
        if self._trailing_samples:
            utterance = utterance[:-self._trailing_samples]
        # 판정은 "실제 발화(음성) 누적 시간" 기준으로 한다 — 총 버퍼 길이로 판정하면
        # 발화종료 대기용으로 뒤에 붙은 무음 패딩(silence_gap_sec)이 길이를 부풀려서
        # 짧은 잡음도 발화로 잘못 통과될 수 있다.
        speech_duration = self._speech_accum
        self._buffer = []
        self._in_speech = False
        self._silence_accum = 0.0
        self._elapsed_accum = 0.0
        self._trailing_samples = 0
        self._speech_accum = 0.0

        if speech_duration < self.min_utterance_sec:
            return None  # 너무 짧으면(숨소리, 잡음 등) 버림
        return utterance


class MicStream:
    """sounddevice로 실제 마이크에서 블록 단위 오디오를 계속 받아
    UtteranceSegmenter에 흘려보내고, 완성된 발화 구간을 콜백으로 전달한다.

    [프라이버시 설계 원칙] (design_targets.py의 DISCARD_RAW_AUDIO_AFTER_PROCESSING 참고)
    완성된 발화(numpy 배열)는 on_utterance 콜백으로 메모리상에서만 전달되고,
    이 클래스는 어떤 발화 오디오도 디스크에 저장하지 않는다. 콜백을 처리한 뒤
    audio 배열에 대한 참조가 남지 않으면 가비지 컬렉션된다. WSCoach(Zhang et al.,
    2025, ACM IMWUT)의 "원본 미저장 + 최소 메타데이터만 보관" 원칙을 따른 것이다.

    [알려진 한계 — 화자분리 부재]
    단일 마이크만 쓰고 화자분리(speaker diarization)나 빔포밍이 없어서, 사용자
    본인이 아닌 주변 사람의 발화도 여기서 감지되면 그대로 다음 단계로 넘어간다.
    WSCoach도 동일한 구조적 한계를 보고한다 — 이 분야에서 흔한 미해결 과제다.
    """

    def __init__(self, on_utterance, samplerate=16000, blocksize=1024, device=None, **segmenter_kwargs):
        import sounddevice as sd  # 실제 마이크 하드웨어 필요 시점에만 import
        self._sd = sd
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.device = device
        self.on_utterance = on_utterance
        self.segmenter = UtteranceSegmenter(samplerate=samplerate, **segmenter_kwargs)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicStream] 상태 경고: {status}")
        block = indata[:, 0].copy()  # 모노 채널만 사용
        utterance = self.segmenter.process_block(block)
        if utterance is not None:
            self.on_utterance(utterance, self.samplerate)

    def start(self):
        self._stream = self._sd.InputStream(
            samplerate=self.samplerate,
            channels=1,
            blocksize=self.blocksize,
            dtype="float32",
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()
        print(f"[MicStream] 마이크 스트리밍 시작 (samplerate={self.samplerate}, blocksize={self.blocksize})")
        print(f"[MicStream] 노이즈 캘리브레이션 중입니다 ({self.segmenter.noise_floor.calibration_sec}초) — 조용히 기다려주세요.")

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            print("[MicStream] 마이크 스트리밍 종료")


if __name__ == "__main__":
    def handle_utterance(audio: np.ndarray, sr: int):
        dur = len(audio) / sr
        print(f"[발화 감지] 길이={dur:.2f}초, 샘플수={len(audio)}")
        # TODO: 1.2 단계에서 feature_extractor.py로 전달 예정

    mic = MicStream(on_utterance=handle_utterance)
    mic.start()
    print("마이크에 대고 말씀해보세요. 종료하려면 Ctrl+C를 누르세요.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        mic.stop()
