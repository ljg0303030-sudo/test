"""Atomic numeric records and personal baseline persistence."""
import json, os
from pathlib import Path
from personal_baseline import ProgressiveBaseline
from threshold_compare import AcousticFeatures

def save_baseline(baseline, path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    rows=list(zip(baseline._speech_rate,baseline._jitter,baseline._shimmer))
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'pipeline':'auto45-fix1','samples':rows},allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def load_baseline(path):
    path=Path(path)
    baseline=ProgressiveBaseline()
    if not path.exists(): return baseline
    data=json.loads(path.read_text(encoding='utf-8'))
    if data['pipeline']!='auto45-fix1' or len(data['samples'])>45:
        raise ValueError('저장된 기준선 형식이 다릅니다.')
    for row in data['samples']:
        baseline.collect(AcousticFeatures(speech_rate=row[0],jitter=row[1],shimmer=row[2]))
    return baseline

def save_log(log,path):
    if not log.rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');log.export(tmp);os.replace(tmp,path)
