"""Boundary correction and context diagnostics for the existing GUI."""
from original_feature_extractor import FeatureExtractionError, _estimate_syllable_count
from context_measurement import measure_variants

def extract_features(audio,samplerate,apply_undercounting_correction=False):
    if apply_undercounting_correction:
        raise FeatureExtractionError('미검증 음절 수 보정은 사용하지 않습니다.')
    measured=measure_variants(audio,samplerate)
    return dict(measured['features'],measurement=measured)
