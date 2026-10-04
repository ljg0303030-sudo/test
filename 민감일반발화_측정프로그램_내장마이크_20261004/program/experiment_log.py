"""In-memory numeric experiment records; explicit local CSV export only."""
import csv
import math
from datetime import datetime, timezone
from uuid import uuid4

class ExperimentLog:
    def __init__(self):
        self.session_id = uuid4().hex
        self.baseline_id = 1
        self.rows = []

    def new_baseline(self):
        self.baseline_id += 1

    def add(self, payload, phase, baseline):
        result = payload['result']
        row = dict(record_id=len(self.rows)+1, session_id=self.session_id,
                   baseline_id=self.baseline_id, timestamp=datetime.now(timezone.utc).isoformat(),
                   pipeline_version='gui-boundary-context-20261002', phase=phase, samples=payload['samples'],
                   label='', prediction=('collection' if result is None else
                       'uncertain' if result.is_sensitive is None else
                       'sensitive' if result.is_sensitive else 'normal'),
                   votes='' if result is None else result.votes,
                   elapsed_ms=payload['elapsed_ms'])
        for key, value in payload['features'].items():
            if key!='measurement': row[key] = value
        m=payload['features'].get('measurement',{})
        region=m.get('region',{})
        for key in ['start_sec','end_sec','duration_sec']:row['analysis_'+key]=region.get(key,'')
        row['baseline_provenance']='restored_reference_or_user_reset; cross_device_validity_unverified'
        for key, stats in baseline.summary().items():
            row[key+'_median'] = stats['median']
            row[key+'_mad'] = stats['mad']
            score = getattr(result, 'scores', {}).get(key)
            row[key+'_z'] = score if score is not None and math.isfinite(score) else ''
        self.rows.append(row)
        return row['record_id']

    def label(self, record_id, label):
        if label not in ('normal', 'sensitive', 'unknown'):
            raise ValueError('Invalid label')
        self.rows[int(record_id)-1]['label'] = label

    def export(self, path):
        if not self.rows:
            raise ValueError('저장할 기록이 없습니다.')
        fields = list(dict.fromkeys(k for row in self.rows for k in row))
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
