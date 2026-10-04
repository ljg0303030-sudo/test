"""Portable inference for exported Random Forest; no pickle loading."""
import json, math
from pathlib import Path

class ComparisonModel:
    def __init__(self, path=None):
        self.data = json.loads(Path(path or Path(__file__).with_name('model.json')).read_text(encoding='utf-8'))
    def predict(self, features):
        x = [float(features[k]) for k in self.data['runtime_features']]
        if not all(math.isfinite(v) for v in x):
            raise ValueError('모델 입력값이 유효하지 않습니다.')
        # sklearn trees consume float32 features.
        import numpy as np
        x = np.asarray(x, dtype=np.float32)
        probabilities = []
        for tree in self.data['trees']:
            node = 0
            while tree['left'][node] != -1:
                node = tree['left'][node] if float(x[tree['feature'][node]]) <= tree['threshold'][node] else tree['right'][node]
            probabilities.append(tree['probability'][node])
        score = sum(probabilities)/len(probabilities)
        return dict(model_score=score, model_default=int(score >= .5),
                    model_relaxed=int(score >= self.data['relaxed_threshold']),
                    model_relaxed_threshold=self.data['relaxed_threshold'],
                    model_version=self.data['version'])
