"""Headless check of the modified GUI flow with a FAKE sounddevice (no microphone is opened).

Usage: xvfb-run python3.12 verify_internalmic.py OUT.json WAV [WAV ...]
A real WAV is streamed through the real SpeechApp / LiveSession / feature pipeline in
1024-sample blocks. Device names, host APIs and the 16 kHz refusal are simulated, so this
does NOT verify Windows drivers or live capture on the user's laptop.
"""
import json, shutil, sys, tempfile, threading, time, types, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROGRAM = HERE.parent / 'program'
OUT, WAVS = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
SR = 16000
checks = []


def check(name, ok, detail=''):
    checks.append(dict(name=name, passed=bool(ok), detail=str(detail)))
    print(('PASS ' if ok else 'FAIL ') + name + (f' · {detail}' if detail else ''))


def read_any(path):
    import parselmouth
    s = parselmouth.Sound(str(path))
    return s.values[0].astype(np.float32), int(s.sampling_frequency)


# ---- fake sounddevice -------------------------------------------------------
FEED = {'audio': None}
fake = types.ModuleType('sounddevice')
DEVICES = [
    dict(name='Microsoft 사운드 매퍼 - Input', hostapi=0, max_input_channels=2, default_samplerate=44100.0),
    dict(name='마이크 배열(Realtek(R) Audio)', hostapi=0, max_input_channels=2, default_samplerate=44100.0),
    dict(name='스피커', hostapi=0, max_input_channels=0, default_samplerate=44100.0),
    dict(name='마이크 배열(Realtek(R) Audio)', hostapi=1, max_input_channels=2, default_samplerate=48000.0),
]
fake.query_devices = lambda: DEVICES
fake.query_hostapis = lambda: [dict(name='MME'), dict(name='Windows WASAPI')]
fake.default = types.SimpleNamespace(device=(0, 2))


def check_input_settings(device=None, channels=None, dtype=None, samplerate=None):
    if DEVICES[device]['hostapi'] == 1 and samplerate != DEVICES[device]['default_samplerate']:
        raise Exception('Invalid sample rate [PaErrorCode -9997]')
fake.check_input_settings = check_input_settings


class FakeStream:
    def __init__(self, samplerate, channels, blocksize, dtype, device, callback):
        self.samplerate, self.device, self.latency = float(samplerate), device, 0.05
        self.blocksize, self.callback, self.stop = blocksize, callback, threading.Event()

    def __enter__(self):
        audio = FEED['audio']
        rng = np.random.default_rng(0)
        quiet = lambda s: (rng.normal(0, 3e-4, int(s * SR))).astype(np.float32)
        signal = np.concatenate([quiet(2.0), audio, quiet(1.5)])
        def run():
            for i in range(0, len(signal) - self.blocksize, self.blocksize):
                if self.stop.is_set():
                    return
                self.callback(signal[i:i + self.blocksize, None], self.blocksize, None, None)
                time.sleep(0.002)
            while not self.stop.is_set():  # keep feeding silence like an open mic
                self.callback(quiet(self.blocksize / SR)[:, None], self.blocksize, None, None)
                time.sleep(0.002)
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(2)
fake.InputStream = FakeStream
sys.modules['sounddevice'] = fake

# ---- run on a temporary copy so packaged records stay untouched -------------
work = Path(tempfile.mkdtemp()) / 'program'
shutil.copytree(PROGRAM, work, ignore=shutil.ignore_patterns('__pycache__'))
for f in (work / 'records').glob('speech_compare_*'):
    f.unlink()
sys.path.insert(0, str(work))

import tkinter as tk
from tkinter import messagebox
shown = []
for kind in ('showinfo', 'showerror', 'showwarning'):
    setattr(messagebox, kind, lambda title, msg, kind=kind, **kw: shown.append((kind, title, msg)))
messagebox.askyesno = lambda *a, **k: True

from device_info import list_input_devices
items = list_input_devices(fake)
check('장치 목록: 입력 장치만, 드라이버 방식 표시', [d['label'] for d in items] == [
    '0: Microsoft 사운드 매퍼 - Input [MME] · Windows 기본 입력',
    '1: 마이크 배열(Realtek(R) Audio) [MME]',
    '3: 마이크 배열(Realtek(R) Audio) [Windows WASAPI]'], [d['label'] for d in items])

import main_gui
root = tk.Tk()
app = main_gui.SpeechApp(root)
check('기준선 45개 복원·판정 단계', app.baseline.n_samples == 45 and not app.baseline.collecting, app.baseline.n_samples)
app.start()
check('장치 미선택 시 시작 거부', app.session is None and shown and '직접 선택' in shown[-1][2])
app.device_box.current(3)  # WASAPI entry, 48 kHz only in the fake
app.start()
check('16 kHz로 못 여는 장치는 시작 전 안내', app.session is None and shown[-1][1] == '입력 장치 열기 불가', shown[-1][2].splitlines()[0])


def pump(until, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if until():
            return True
        time.sleep(0.01)
    return False


def run_one(audio, label):
    FEED['audio'] = audio
    before = len(app.log.rows)
    app.device_box.current(2)  # MME built-in array
    app.start()
    got = pump(lambda: len(app.log.rows) > before and app.awaiting_label)
    row = app.log.rows[-1] if got else {}
    if got:
        app.set_label(label)
    app.stop()
    pump(lambda: app.session is None, 10)
    return got, row


results = []
for path in WAVS:
    audio, sr = read_any(path)
    assert sr == SR, (path, sr)
    got, row = run_one(audio, 'normal')
    results.append(dict(file=path.name, processed=got, phase=row.get('phase'), prediction=row.get('prediction'),
                        reason=row.get('withheld_reason', ''), speech_rate=row.get('speech_rate'),
                        jitter=row.get('jitter'), shimmer=row.get('shimmer'),
                        analysis_duration_sec=row.get('analysis_duration_sec'),
                        device=row.get('device'), stream_samplerate=row.get('stream_samplerate')))
    check(f'{path.name}: GUI 처리·정답 대기', got, f"{row.get('phase')} / {row.get('prediction')} {row.get('withheld_reason', '')}")
    check(f'{path.name}: 실제 열린 장치·표본화율 기록', row.get('device_index') == 1 and row.get('device_hostapi') == 'MME'
          and row.get('stream_samplerate') == 16000.0, (row.get('device'), row.get('stream_samplerate')))

# Force a withheld result on the first WAV to check the withheld path end-to-end.
import measurement_guard
original_check = measurement_guard.check_measurement
def forced(features, baseline):
    raise ValueError('검증용 강제 보류')
measurement_guard.check_measurement = forced
got, row = run_one(read_any(WAVS[0])[0], 'sensitive')
measurement_guard.check_measurement = original_check
check('보류 발화: 특징값·사유 기록', got and row.get('phase') == 'withheld' and row.get('withheld_reason') == '검증용 강제 보류'
      and isinstance(row.get('jitter'), float) and row.get('context_shimmer_max') != '', row.get('withheld_reason'))
check('보류 발화: 정답 지정 가능', row.get('label') == 'sensitive', row.get('label'))
check('보류 발화: 기준선 불변', app.baseline.n_samples == 45)

app.show_metrics()
metrics = shown[-1][2]
check('성적: 보류율 표시', '민감 정답 1개 중 측정 보류 1개' in metrics, metrics.replace('\n', ' | '))

csv_rows = list(csv.DictReader(open(app.autosave_path, encoding='utf-8-sig')))
check('자동 저장 CSV에 보류 행·장치 열 존재', any(r['phase'] == 'withheld' for r in csv_rows)
      and {'device_hostapi', 'stream_samplerate', 'withheld_reason'} <= set(csv_rows[0]), len(csv_rows))
root.destroy()

# ---- CONTROLLED_20 baseline reuse must never wipe the restored baseline ------
import controlled_gui, persistence
for has_reference in (False, True):
    (work / 'records' / 'baseline.json').write_text((PROGRAM / 'records' / 'baseline.json').read_text(encoding='utf-8'), encoding='utf-8')
    ref = work / 'reference_baseline_original.json'
    backup = ref.read_bytes()
    if not has_reference:
        ref.unlink()
    root = tk.Tk()
    capp = controlled_gui.ControlledApp(root)
    capp.reuse()
    root.update()
    n = capp.baseline.n_samples
    saved = persistence.load_baseline(work / 'records' / 'baseline.json').n_samples
    root.destroy()
    ref.write_bytes(backup)
    check(f'CONTROLLED_20 기준선 불러오기 (참조 파일 {"있음" if has_reference else "없음"})', n == 45 and saved == 45, (n, saved))

OUT.write_text(json.dumps(dict(note='Fake sounddevice; real WAV streamed through real GUI pipeline. Not a live microphone test.',
    checks=checks, recordings=results, passed=sum(c['passed'] for c in checks), total=len(checks)),
    ensure_ascii=False, indent=1), encoding='utf-8')
print(f"{sum(c['passed'] for c in checks)}/{len(checks)} passed")
sys.exit(0 if all(c['passed'] for c in checks) else 1)
