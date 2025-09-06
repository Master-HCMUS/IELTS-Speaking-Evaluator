"""Unit tests for feature extraction engine.

Tasks: FEAT-9 validates FEAT-1..7 outputs.
"""
from models.core import Transcript, WordTiming
from features.extractor import extract_features


def build_transcript():
    # Words spaced with pauses >=250ms at positions to create runs
    words = [
        WordTiming(w="Hello", start_ms=0, end_ms=300),
        WordTiming(w="world", start_ms=310, end_ms=600),  # short gap -> same run
        WordTiming(w="this", start_ms=900, end_ms=1000),  # pause 300ms -> new run
        WordTiming(w="is", start_ms=1010, end_ms=1100),
        WordTiming(w="a", start_ms=1110, end_ms=1150),
        WordTiming(w="test", start_ms=1160, end_ms=1300),
        WordTiming(w="test", start_ms=1310, end_ms=1400),  # repetition for lexical uniqueness impact
    ]
    text = "Hello world this is a test test"
    return Transcript(words=words, text=text)


def test_extract_features_basic():
    tr = build_transcript()
    feats = extract_features(tr)
    assert feats.fluency.wpm > 0
    assert feats.fluency.mean_length_of_run >= 1
    # Expect 2 runs: [Hello,world] and [this,is,a,test,test]; mean length >2
    assert feats.fluency.mean_length_of_run > 2
    assert feats.fluency.pause_rate_per_min >= 0
    # TTR less than 1 due to repetition of 'test'
    assert feats.lexical.ttr < 1
    # Rare word share between 0 and 1
    assert 0 <= feats.lexical.rare_word_share <= 1
    assert feats.grammar.complexity_index > 0
    # Rhythm variance numeric
    assert feats.pronunciation.rhythm_variance is not None