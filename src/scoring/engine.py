"""Scoring engine – rule-based mapping of features to IELTS bands.

Tasks: SCORE-1..SCORE-6 (P07)
Spec Ref: Section 8 Scoring.

Design:
  - thresholds.json config describes per-metric band ladders.
  - For metrics with increasing-good (e.g., wpm, ttr) we use 'min' thresholds ascending.
  - For metrics where lower is better (pause_rate, error_density, rhythm_variance, wer_proxy) we use 'max' thresholds ascending.
  - Per criterion band = median of metric bands (robust vs mean for small set).
  - Overall band = weighted mean of criteria, rounded to nearest 0.5.
  - Rationale summarises key metrics with threshold references.
"""
from __future__ import annotations
from pathlib import Path
import json
from statistics import median
from typing import Dict, Any, List

from models.core import FeatureSet, ScoreSet, CriterionScore

CONFIG_PATH = Path(__file__).parent / "thresholds.json"


class ScoringConfig:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "ScoringConfig":
        return cls(json.loads(path.read_text()))


def _metric_band(value: float, ladder: List[dict], higher_is_better: bool) -> int:
    if higher_is_better:
        # ladder sorted ascending by min
        best = ladder[0]["band"]
        for step in ladder:
            if value >= step.get("min", float('-inf')):
                best = step["band"]
        return best
    # lower better path: pick first with value <= max
    for step in ladder:
        if value <= step.get("max", float('inf')):
            return step["band"]
    return ladder[-1]["band"]


def _criterion_band(name: str, features: FeatureSet, cfg: ScoringConfig) -> tuple[int, List[str]]:
    rules = cfg.data[name]
    metric_bands: List[int] = []
    rationale_parts: List[str] = []
    def add(metric_name: str, value: float, higher_is_better: bool):
        ladder = rules.get(metric_name, [])
        if not ladder:
            return
        band = _metric_band(value, ladder, higher_is_better)
        metric_bands.append(band)
        rationale_parts.append(f"{metric_name}={value:.2f}->B{band}")
    if name == "fluency_coherence":
        add("wpm", features.fluency.wpm, True)
        add("mean_length_of_run", features.fluency.mean_length_of_run, True)
        add("pause_rate_per_min", features.fluency.pause_rate_per_min or 0.0, False)
    elif name == "lexical_resource":
        add("ttr", features.lexical.ttr, True)
        add("rare_word_share", features.lexical.rare_word_share or 0.0, True)
    elif name == "grammatical_range_accuracy":
        add("error_density_per_100", features.grammar.error_density_per_100 or 0.0, False)
        add("complexity_index", features.grammar.complexity_index or 0.0, True)
    elif name == "pronunciation":
        add("rhythm_variance", features.pronunciation.rhythm_variance or 0.0, False)
        add("wer_proxy", features.pronunciation.wer_proxy or 0.0, False)
    band = int(round(median(metric_bands))) if metric_bands else 4
    return band, rationale_parts


def _round_overall(value: float) -> float:
    return round(value * 2) / 2.0


def score_features(features: FeatureSet, cfg: ScoringConfig | None = None) -> ScoreSet:
    cfg = cfg or ScoringConfig.load()
    weights = cfg.data.get("weights", {})
    crit_results = {}
    rationales = {}
    for crit in ["fluency_coherence", "lexical_resource", "grammatical_range_accuracy", "pronunciation"]:
        band, parts = _criterion_band(crit, features, cfg)
        crit_results[crit] = band
        rationales[crit] = "; ".join(parts)
    # Weighted mean
    total = 0.0
    weight_sum = 0.0
    for crit, band in crit_results.items():
        w = float(weights.get(crit, 0.25))
        weight_sum += w
        total += band * w
    overall = _round_overall(total / weight_sum if weight_sum else 0.0)
    return ScoreSet(
        fluency_coherence=CriterionScore(band=crit_results["fluency_coherence"], rationale=rationales["fluency_coherence"]),
        lexical_resource=CriterionScore(band=crit_results["lexical_resource"], rationale=rationales["lexical_resource"]),
        grammatical_range_accuracy=CriterionScore(band=crit_results["grammatical_range_accuracy"], rationale=rationales["grammatical_range_accuracy"]),
        pronunciation=CriterionScore(band=crit_results["pronunciation"], rationale=rationales["pronunciation"]),
        overall_band=overall,
    )


__all__ = ["score_features", "ScoringConfig"]
