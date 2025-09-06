"""Core domain models.

Task: DM-2 (P02) – Schema models for session artifacts.
Spec Ref: Sections 7 Data Model, Storage Strategy Updated.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

SCHEMA_VERSION = 1

class WordTiming(BaseModel):
    w: str
    start_ms: int
    end_ms: int
    confidence: float | None = None

class Transcript(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    words: List[WordTiming]
    text: str

class FeatureFluency(BaseModel):
    wpm: float
    mean_length_of_run: float
    pause_rate_per_min: float | None = None

class FeatureLexical(BaseModel):
    ttr: float
    rare_word_share: float | None = None

class FeatureGrammar(BaseModel):
    error_density_per_100: float | None = None
    complexity_index: float | None = None

class FeaturePronunciation(BaseModel):
    wer_proxy: float | None = None
    rhythm_variance: float | None = None

class FeatureSet(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    fluency: FeatureFluency
    lexical: FeatureLexical
    grammar: FeatureGrammar
    pronunciation: FeaturePronunciation

class CriterionScore(BaseModel):
    band: float
    rationale: str

class ScoreSet(BaseModel):
    schema_version: int = Field(default=SCHEMA_VERSION)
    fluency_coherence: CriterionScore
    lexical_resource: CriterionScore
    grammatical_range_accuracy: CriterionScore
    pronunciation: CriterionScore
    overall_band: float

@dataclass
class SessionMeta:
    session_id: str
    created_at: datetime
    cue_card_id: Optional[str]
    model_version: Optional[str]
    overall_band: Optional[float] = None
