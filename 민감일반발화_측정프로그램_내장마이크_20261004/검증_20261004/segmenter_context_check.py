"""Does the live segmenter change what the boundary/feature pipeline sees?

(a) validated path : measure_variants() on the whole 8 s WAV (boundary module chooses the region)
(b) live path      : WAV streamed in 1024-sample blocks through UtteranceSegmenter as shipped
(c) live + context : same, but the segmenter keeps up to CONTEXT_SEC of audio before the first
                     voiced block and after the last one (boundary module still chooses the region)

Usage: python verify/segmenter_context_check.py PROGRAM_DIR OUT.json WAV...
Thresholds, peak counter and Jitter/Shimmer settings are untouched.
"""
import json, sys
from pathlib import Path
import numpy as np
import parselmouth

PROGRAM, OUT, WAVS = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
sys.path.insert(0, str(PROGRAM))
from context_measurement import measure_variants
from audio_input import UtteranceSegmenter
import measurement_guard, threshold_compare

SR, BLOCK = 16000, 1024


def load(path):
    s = parselmouth.Sound(str(path))
    assert int(s.sampling_frequency) == SR
    return s.values[0].astype(np.float32)


def stream(y, segmenter_kwargs=None):
    """Prepend 1.6 s made of the recording's own leading noise for calibration, then stream."""
    region = measure_variants(y, SR)['region']
    noise = y[:max(BLOCK, region['start_sample'] - int(.05 * SR))]
    calib = np.resize(noise, int(1.6 * SR))
    signal = np.concatenate([calib, y, np.resize(noise, int(1.0 * SR))])
    seg = UtteranceSegmenter(samplerate=SR, **(segmenter_kwargs or {}))
    out = []
    for i in range(0, len(signal) - BLOCK + 1, BLOCK):
        u = seg.process_block(signal[i:i + BLOCK].copy())
        if u is not None:
            out.append(u)
    return out


def summary(audio):
    try:
        m = measure_variants(audio, SR)
        f = m['features']
        try:
            measurement_guard.check_measurement(dict(f, measurement=m), type('B', (), {'collecting': True})())
            guard = 'pass'
        except ValueError as exc:
            guard = 'withheld: ' + str(exc)
        return dict(status='ok', input_sec=len(audio) / SR, duration_sec=m['region']['duration_sec'],
                    peaks=round(f['speech_rate'] * f['_duration_sec']), speech_rate=f['speech_rate'],
                    jitter_pct=f['jitter'] * 100, shimmer_pct=f['shimmer'] * 100,
                    noise_proxy_dbfs=m['region']['noise_proxy_dbfs'], guard=guard)
    except Exception as exc:
        return dict(status='withheld', input_sec=len(audio) / SR, reason=str(exc))


rows = []
for path in WAVS:
    y = load(path)
    row = dict(file=path.name, a=summary(y))
    for key, kwargs in [('b', None), ('c', dict(context_sec=0.3))]:
        try:
            utts = stream(y, kwargs)
        except TypeError:
            utts = None  # shipped segmenter has no context option
        row[key] = None if utts is None else dict(n_utterances=len(utts), results=[summary(u) for u in utts])
    rows.append(row)
    def brief(r):
        if r is None:
            return 'n/a'
        return ' | '.join(f"{x.get('status')} {x.get('duration_sec', 0):.3f}s peaks={x.get('peaks')} J={x.get('jitter_pct', 0):.3f}% S={x.get('shimmer_pct', 0):.3f}% {x.get('guard', x.get('reason', ''))}"
                          for x in (r['results'] if 'results' in r else [r]))
    print(path.name)
    for key in 'abc':
        print(f'  ({key})', brief(row[key]))
OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
