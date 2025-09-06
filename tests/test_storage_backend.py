"""Tests for LocalFileStorageBackend.

Task: DM-9 (P02) – lifecycle tests.
"""
from datetime import datetime
from uuid import uuid4
from storage.backend import LocalFileStorageBackend, StorageError, DATA_ROOT
from models.core import SessionMeta, Transcript, WordTiming, FeatureSet, FeatureFluency, FeatureLexical, FeatureGrammar, FeaturePronunciation, ScoreSet, CriterionScore


def make_transcript() -> Transcript:
    return Transcript(words=[WordTiming(w="hello", start_ms=0, end_ms=400)], text="hello")


def make_features() -> FeatureSet:
    return FeatureSet(
        fluency=FeatureFluency(wpm=120.0, mean_length_of_run=6.0, pause_rate_per_min=10.0),
        lexical=FeatureLexical(ttr=0.5, rare_word_share=0.1),
        grammar=FeatureGrammar(error_density_per_100=5.0, complexity_index=1.2),
        pronunciation=FeaturePronunciation(wer_proxy=0.1, rhythm_variance=0.2),
    )


def make_scores() -> ScoreSet:
    return ScoreSet(
        fluency_coherence=CriterionScore(band=6.0, rationale="Sample"),
        lexical_resource=CriterionScore(band=6.5, rationale="Sample"),
        grammatical_range_accuracy=CriterionScore(band=6.0, rationale="Sample"),
        pronunciation=CriterionScore(band=6.5, rationale="Sample"),
        overall_band=6.25,
    )


def test_lifecycle(tmp_path, monkeypatch):
    # Redirect data root
    monkeypatch.setattr("storage.backend.DATA_ROOT", tmp_path)
    monkeypatch.setattr("storage.backend.INDEX_FILE", tmp_path / "index.json")
    backend = LocalFileStorageBackend()
    backend.init()
    sid = uuid4().hex
    meta = SessionMeta(session_id=sid, created_at=datetime.utcnow(), cue_card_id="card1", model_version="gpt-4o-mini-audio-preview")
    backend.create_session(meta)
    backend.save_transcript(sid, make_transcript())
    backend.save_features(sid, make_features())
    backend.save_scores(sid, make_scores())

    # Export snapshot
    snapshot = backend.export_snapshot(sid)
    assert snapshot["session"]["session_id"] == sid

    # Integrity scan
    scan = backend.integrity_scan()
    assert scan["checked"] == 1
    assert not scan["problems"]

    sessions = backend.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].overall_band == 6.25

    scores = backend.load_scores(sid)
    assert scores is not None and scores.overall_band == 6.25

    # Delete session
    backend.delete_session(sid)
    assert backend.list_sessions() == []

    # Integrity after delete
    post_scan = backend.integrity_scan()
    assert post_scan["checked"] == 0
