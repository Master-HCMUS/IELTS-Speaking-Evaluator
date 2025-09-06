"""Local feature extraction engine.

Tasks: FEAT-1..FEAT-9 (P06)
Spec Ref: Section 6 Feature Extraction.

Design Principles:
  - Pure functions (no I/O) for deterministic, testable metrics.
  - No heavy external NLP dependencies (lightweight heuristics only).
  - Graceful handling of edge cases (avoid ZeroDivisionError -> return 0.0 / None).

Metrics Implemented (MVP heuristics):
  Fluency:
    - Words Per Minute (WPM)
    - Mean Length of Run (continuous words separated by pauses >= threshold)
    - Pause Rate Per Minute (count of pauses / total minutes)
    - Repetitions (immediate exact token repeats) (used internally for future scoring, not in model)
  Lexical:
    - Type Token Ratio (TTR)
    - Rare Word Share (token not in COMMON_WORDS list)
  Grammar (heuristic placeholders):
    - Error Density (simple: count of tokens containing '??' or malformed apostrophes / per 100 words)
    - Complexity Index (average characters per word)
  Pronunciation Proxies:
    - Rhythm Variance (variance of inter-word onset gaps in ms)
    - WER Proxy (placeholder: 0.0; real alignment requires ref transcript)

Future Improvements (not implemented now): syllable-based rhythm, discourse markers density, idiom detection.
"""
from __future__ import annotations
from typing import List, Tuple
import statistics
import re

from models.core import (
    Transcript,
    FeatureSet,
    FeatureFluency,
    FeatureLexical,
    FeatureGrammar,
    FeaturePronunciation,
)

PAUSE_THRESHOLD_MS = 250  # Spec hint (>250ms)
COMMON_WORDS = {
    # Tiny frequent list (can be expanded); used to approximate rare word share.
    "the","a","an","i","you","he","she","it","we","they","and","or","but","to","of","in","on","with","for","that","this","is","are","was","were","be","have","has","had","at","from","as","so","if","not","there","about","my","your","his","her","their","our","me","him","them","what","which","when","who","how","why","because","just","like"
}
FILLERS = {"um", "uh", "erm", "ah", "uhm"}


def _tokenize(transcript: Transcript) -> List[str]:  # FEAT-1
    # Basic whitespace / punctuation tokenization preserving apostrophes
    text = transcript.text
    # Replace punctuation (except apostrophes) with space
    cleaned = re.sub(r"[\.,!?;:\-]", " ", text)
    tokens = [t for t in cleaned.split() if t.strip()]
    return tokens


def _compute_pauses(transcript: Transcript) -> Tuple[int, List[int]]:  # FEAT-2
    pauses = 0
    gaps: List[int] = []
    words = transcript.words
    for i in range(1, len(words)):
        gap = words[i].start_ms - words[i-1].end_ms
        if gap >= 0:
            gaps.append(gap)
        if gap >= PAUSE_THRESHOLD_MS:
            pauses += 1
    return pauses, gaps


def _fluency_metrics(transcript: Transcript, tokens: List[str]):  # FEAT-3
    if not transcript.words or not tokens:
        return 0.0, 0.0, 0.0, 0  # wpm, mean_run, pause_rate, repetitions
    duration_ms = max(1, transcript.words[-1].end_ms - transcript.words[0].start_ms)
    minutes = duration_ms / 60000.0
    wpm = len(tokens) / minutes if minutes > 0 else 0.0
    # Runs
    run_lengths: List[int] = []
    current_run = 1
    words = transcript.words
    for i in range(1, len(words)):
        gap = words[i].start_ms - words[i-1].end_ms
        if gap >= PAUSE_THRESHOLD_MS:
            run_lengths.append(current_run)
            current_run = 1
        else:
            current_run += 1
    run_lengths.append(current_run)
    mean_run = sum(run_lengths)/len(run_lengths) if run_lengths else 0.0
    pauses, _ = _compute_pauses(transcript)
    pause_rate = pauses / minutes if minutes > 0 else 0.0
    # Repetitions (immediate)
    reps = 0
    for i in range(1, len(tokens)):
        if tokens[i].lower() == tokens[i-1].lower():
            reps += 1
    return wpm, mean_run, pause_rate, reps


def _lexical_metrics(tokens: List[str]):  # FEAT-5
    if not tokens:
        return 0.0, 0.0
    lowered = [t.lower() for t in tokens]
    unique = len(set(lowered))
    ttr = unique / len(lowered)
    rare = sum(1 for t in lowered if t not in COMMON_WORDS)
    rare_share = rare / len(lowered)
    return ttr, rare_share


def _grammar_metrics(tokens: List[str]):  # FEAT-6
    if not tokens:
        return 0.0, 0.0
    lowered = [t.lower() for t in tokens]
    # Error heuristic: tokens with suspicious punctuation or double question marks etc.
    errors = sum(1 for t in lowered if "??" in t or t.endswith("???") or t.count("'") > 2)
    error_density = (errors / len(tokens)) * 100.0
    avg_chars = sum(len(t) for t in tokens) / len(tokens)
    return error_density, avg_chars


def _pronunciation_metrics(transcript: Transcript):  # FEAT-7
    words = transcript.words
    if len(words) < 2:
        return 0.0, None
    # Inter-word onset gaps
    gaps = []
    for i in range(1, len(words)):
        gap = words[i].start_ms - words[i-1].start_ms
        if gap >= 0:
            gaps.append(gap)
    rhythm_var = statistics.pvariance(gaps) if len(gaps) > 1 else 0.0
    wer_proxy = 0.0  # placeholder (needs reference or alt hypothesis)
    return rhythm_var, wer_proxy


def extract_features(transcript: Transcript) -> FeatureSet:  # FEAT-8/9
    tokens = _tokenize(transcript)
    wpm, mean_run, pause_rate, _reps = _fluency_metrics(transcript, tokens)
    ttr, rare_share = _lexical_metrics(tokens)
    err_density, complexity = _grammar_metrics(tokens)
    rhythm_var, wer_proxy = _pronunciation_metrics(transcript)
    return FeatureSet(
        fluency=FeatureFluency(wpm=wpm, mean_length_of_run=mean_run, pause_rate_per_min=pause_rate),
        lexical=FeatureLexical(ttr=ttr, rare_word_share=rare_share),
        grammar=FeatureGrammar(error_density_per_100=err_density, complexity_index=complexity),
        pronunciation=FeaturePronunciation(wer_proxy=wer_proxy, rhythm_variance=rhythm_var),
    )


__all__ = [
    "extract_features",
]
