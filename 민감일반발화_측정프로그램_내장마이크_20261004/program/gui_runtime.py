"""GUI-independent streaming worker. Audio is held only in bounded memory queues."""
import queue
import threading
import time
import math
from audio_input import UtteranceSegmenter
from personal_baseline import PersonalBaseline
from threshold_compare import AcousticFeatures


class LiveSession:
    def __init__(self, baseline=None, device=None, samplerate=16000,
                 extractor=None, stream_factory=None):
        self.baseline = baseline if baseline is not None else PersonalBaseline()
        self.device, self.samplerate = device, samplerate
        self.extractor, self.stream_factory = extractor, stream_factory
        self.events = queue.Queue(maxsize=8)
        self.blocks = queue.Queue(maxsize=32)
        self.stopping = threading.Event()
        self.accepting = threading.Event()
        self.ack = threading.Event()
        self.overflow = threading.Event()
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.accepting.clear()
        self.stopping.set()
        self.ack.set()

    def resume(self):
        self.ack.set()

    def _callback(self, indata, frames, time_info, status):
        if not self.accepting.is_set() or self.stopping.is_set():
            return
        if status:
            self.overflow.set()
        try:
            self.blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self.overflow.set()

    def _clear_blocks(self):
        while True:
            try:
                self.blocks.get_nowait()
            except queue.Empty:
                return

    def _emit(self, kind, value=None):
        # Only one result is pending at a time; stopping always remains possible.
        while not self.stopping.is_set() or kind == 'stopped':
            try:
                self.events.put((kind, value), timeout=.1)
                return
            except queue.Full:
                if self.stopping.is_set():
                    return

    def _run(self):
        try:
            extractor = self.extractor
            if extractor is None:
                from feature_extractor import extract_features
                extractor = extract_features
            factory = self.stream_factory
            if factory is None:
                import sounddevice as sd
                factory = sd.InputStream
            segmenter = UtteranceSegmenter(samplerate=self.samplerate)
            self.accepting.set()
            with factory(samplerate=self.samplerate, channels=1, blocksize=1024,
                         dtype='float32', device=self.device, callback=self._callback):
                self._emit('status', '주변 소음 측정 중 · 1.5초 동안 조용히 기다려 주세요.')
                calibrated = False
                while not self.stopping.is_set():
                    if self.overflow.is_set():
                        self.overflow.clear()
                        self._clear_blocks()
                        segmenter = UtteranceSegmenter(samplerate=self.samplerate)
                        calibrated = False
                        self._emit('status', '입력 누락 감지 · 소음을 다시 측정합니다. 잠시 조용히 기다려 주세요.')
                    try:
                        block = self.blocks.get(timeout=.1)
                    except queue.Empty:
                        continue
                    audio = segmenter.process_block(block)
                    del block
                    if segmenter.noise_floor.is_calibrated and not calibrated:
                        calibrated = True
                        self._emit('status', '듣는 중 · 문장을 말한 뒤 잠시 쉬어 주세요.')
                    if audio is None:
                        continue
                    self.accepting.clear()
                    self.ack.clear()
                    self._clear_blocks()
                    started = time.perf_counter()
                    try:
                        feats = extractor(audio, self.samplerate)
                        if not all(math.isfinite(float(feats[k])) and feats[k] >= 0
                                   for k in ('speech_rate', 'jitter', 'shimmer')):
                            raise ValueError('판정 특성값이 유효하지 않습니다.')
                        from measurement_guard import check_measurement
                        check_measurement(feats, self.baseline)
                        acoustic = AcousticFeatures(**{k: feats[k] for k in
                            ('speech_rate', 'jitter', 'shimmer', 'f0', 'silence_duration')})
                        collecting = getattr(self.baseline, 'collecting', False)
                        if collecting:
                            self.baseline.collect(acoustic)
                            result = None
                        else:
                            result = self.baseline.judge(acoustic)
                            self.baseline.update(acoustic, result)
                        payload = dict(features=feats, result=result,
                            samples=self.baseline.n_samples,
                            elapsed_ms=(time.perf_counter()-started)*1000)
                        self._emit('collection' if collecting else 'result', payload)
                    except Exception as exc:
                        self._emit('quality', str(exc))
                    finally:
                        del audio
                    while not self.stopping.is_set() and not self.ack.wait(.1):
                        pass
                    self._clear_blocks()
                    if not self.stopping.is_set():
                        self.accepting.set()
        except Exception as exc:
            self._emit('error', str(exc))
        finally:
            self.accepting.clear()
            self._clear_blocks()
            self._emit('stopped')
