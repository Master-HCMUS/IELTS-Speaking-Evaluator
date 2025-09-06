"""Tests for scoring engine.

Tasks: SCORE-6 covers SCORE-1..5 logic boundaries.
"""
from models.core import FeatureSet, FeatureFluency, FeatureLexical, FeatureGrammar, FeaturePronunciation
from scoring.engine import score_features, ScoringConfig
import json
from pathlib import Path


def make_features(wpm=90, run=6, pause_rate=40, ttr=0.55, rare=0.18, err=40, complexity=4.6, rhythm_var=60000, wer=0.1):
    return FeatureSet(
        fluency=FeatureFluency(wpm=wpm, mean_length_of_run=run, pause_rate_per_min=pause_rate),
        lexical=FeatureLexical(ttr=ttr, rare_word_share=rare),
        grammar=FeatureGrammar(error_density_per_100=err, complexity_index=complexity),
        pronunciation=FeaturePronunciation(rhythm_variance=rhythm_var, wer_proxy=wer),
    )


def test_score_midrange(tmp_path):
    fs = make_features()
    scores = score_features(fs)
    assert 4 <= scores.overall_band <= 8
    assert scores.fluency_coherence.band >= 5
    assert scores.lexical_resource.rationale


def test_score_high_values(tmp_path):
    fs = make_features(wpm=130, run=11, pause_rate=20, ttr=0.7, rare=0.3, err=20, complexity=5.8, rhythm_var=20000, wer=0.04)
    scores = score_features(fs)
    assert scores.overall_band >= 7.5


def test_score_low_values(tmp_path):
    fs = make_features(wpm=40, run=2, pause_rate=80, ttr=0.3, rare=0.05, err=70, complexity=3.0, rhythm_var=110000, wer=0.3)
    scores = score_features(fs)
    assert scores.overall_band <= 5