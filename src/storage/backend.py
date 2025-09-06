"""Storage backend abstractions.

Task: DM-3 (P02) – Define StorageBackend protocol + errors.
Spec Ref: Storage Strategy Updated section.
"""
from __future__ import annotations
import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional
from datetime import datetime

from models.core import SessionMeta, Transcript, FeatureSet, ScoreSet, SCHEMA_VERSION

DATA_ROOT = Path("data")
INDEX_FILE = DATA_ROOT / "index.json"

class StorageError(Exception):
    """Base storage error."""

class StorageBackend(ABC):
    @abstractmethod
    def init(self) -> None: ...

    @abstractmethod
    def create_session(self, meta: SessionMeta) -> None: ...

    @abstractmethod
    def save_transcript(self, session_id: str, transcript: Transcript) -> None: ...

    @abstractmethod
    def save_features(self, session_id: str, features: FeatureSet) -> None: ...

    @abstractmethod
    def save_scores(self, session_id: str, scores: ScoreSet) -> None: ...

    @abstractmethod
    def list_sessions(self) -> List[SessionMeta]: ...

    @abstractmethod
    def load_scores(self, session_id: str) -> ScoreSet | None: ...

class LocalFileStorageBackend(StorageBackend):
    def init(self) -> None:
        DATA_ROOT.mkdir(exist_ok=True)
        if not INDEX_FILE.exists():
            INDEX_FILE.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "sessions": {}}, indent=2))

    def _read_index(self) -> dict:
        try:
            return json.loads(INDEX_FILE.read_text())
        except FileNotFoundError:
            raise StorageError("Index file missing; run init().")

    def _write_index(self, index: dict) -> None:
        tmp = INDEX_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(index, indent=2))
        tmp.replace(INDEX_FILE)

    def create_session(self, meta: SessionMeta) -> None:
        index = self._read_index()
        sessions = index.setdefault("sessions", {})
        if meta.session_id in sessions:
            raise StorageError("Session already exists")
        sessions[meta.session_id] = {
            "created_at": meta.created_at.isoformat(),
            "cue_card_id": meta.cue_card_id,
            "model_version": meta.model_version,
            "overall_band": meta.overall_band,
        }
        sess_dir = DATA_ROOT / "sessions" / meta.session_id
        sess_dir.mkdir(parents=True, exist_ok=False)
        # Write session.json
        (sess_dir / "session.json").write_text(json.dumps({
            "schema_version": SCHEMA_VERSION,
            "session_id": meta.session_id,
            "created_at": meta.created_at.isoformat(),
            "cue_card_id": meta.cue_card_id,
            "model_version": meta.model_version,
        }, indent=2))
        self._write_index(index)

    def save_transcript(self, session_id: str, transcript: Transcript) -> None:
        sess_dir = DATA_ROOT / "sessions" / session_id
        if not sess_dir.exists():
            raise StorageError("Session not found")
        (sess_dir / "transcript.json").write_text(transcript.model_dump_json(indent=2))

    def save_features(self, session_id: str, features: FeatureSet) -> None:
        sess_dir = DATA_ROOT / "sessions" / session_id
        if not sess_dir.exists():
            raise StorageError("Session not found")
        (sess_dir / "features.json").write_text(features.model_dump_json(indent=2))

    def save_scores(self, session_id: str, scores: ScoreSet) -> None:
        sess_dir = DATA_ROOT / "sessions" / session_id
        if not sess_dir.exists():
            raise StorageError("Session not found")
        (sess_dir / "scores.json").write_text(scores.model_dump_json(indent=2))
        # update index overall band
        index = self._read_index()
        if session_id in index.get("sessions", {}):
            index["sessions"][session_id]["overall_band"] = scores.overall_band
            self._write_index(index)

    def list_sessions(self) -> List[SessionMeta]:
        index = self._read_index()
        out: List[SessionMeta] = []
        for sid, meta in index.get("sessions", {}).items():
            out.append(SessionMeta(
                session_id=sid,
                created_at=datetime.fromisoformat(meta["created_at"]),
                cue_card_id=meta.get("cue_card_id"),
                model_version=meta.get("model_version"),
                overall_band=meta.get("overall_band")
            ))
        return sorted(out, key=lambda m: m.created_at, reverse=True)

    def load_scores(self, session_id: str) -> ScoreSet | None:
        fp = DATA_ROOT / "sessions" / session_id / "scores.json"
        if not fp.exists():
            return None
        return ScoreSet.model_validate_json(fp.read_text())
