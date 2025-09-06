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
    """Abstract storage contract (Task: DM-3, P02)."""

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

    @abstractmethod
    def delete_session(self, session_id: str) -> None: ...

    @abstractmethod
    def export_snapshot(self, session_id: str, include_audio: bool = False) -> dict: ...

    @abstractmethod
    def integrity_scan(self) -> dict: ...

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

    def delete_session(self, session_id: str) -> None:
        index = self._read_index()
        if session_id not in index.get("sessions", {}):
            raise StorageError("Session not found")
        sess_dir = DATA_ROOT / "sessions" / session_id
        if sess_dir.exists():
            shutil.rmtree(sess_dir)
        del index["sessions"][session_id]
        self._write_index(index)

    def export_snapshot(self, session_id: str, include_audio: bool = False) -> dict:
        """Compose consolidated snapshot (Task: DM-6)."""
        sess_dir = DATA_ROOT / "sessions" / session_id
        if not sess_dir.exists():
            raise StorageError("Session not found")
        snapshot: dict = {"schema_version": SCHEMA_VERSION, "session_id": session_id}
        for name in ["session.json", "transcript.json", "features.json", "scores.json"]:
            fp = sess_dir / name
            if fp.exists():
                snapshot[name.replace('.json','')] = json.loads(fp.read_text())
        if include_audio:
            audio_fp = sess_dir / "audio.wav"
            if audio_fp.exists():
                snapshot["audio_base64"] = audio_fp.read_bytes().hex()  # hex to avoid b64 lib dependency yet
        # Write export/snapshot.json
        export_dir = sess_dir / "export"
        export_dir.mkdir(exist_ok=True)
        (export_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2))
        return snapshot

    def integrity_scan(self) -> dict:
        """Validate index references & schema versions (Task: DM-7)."""
        index = self._read_index()
        sessions = index.get("sessions", {})
        problems: list[str] = []
        checked = 0
        for sid in sessions:
            sess_dir = DATA_ROOT / "sessions" / sid
            if not sess_dir.exists():
                problems.append(f"missing_dir:{sid}")
                continue
            for fname in ["session.json", "transcript.json", "features.json", "scores.json"]:
                fp = sess_dir / fname
                if fp.exists():
                    try:
                        data = json.loads(fp.read_text())
                        if data.get("schema_version") != SCHEMA_VERSION:
                            problems.append(f"schema_mismatch:{sid}:{fname}")
                    except Exception:  # noqa: BLE001
                        problems.append(f"invalid_json:{sid}:{fname}")
            checked += 1
        return {"checked": checked, "problems": problems}


class AzureBlobStorageBackend(StorageBackend):  # pragma: no cover (stub for future)
    """Placeholder for future Azure implementation (Task: DM-8)."""

    def init(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def create_session(self, meta: SessionMeta) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_transcript(self, session_id: str, transcript: Transcript) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_features(self, session_id: str, features: FeatureSet) -> None:  # pragma: no cover
        raise NotImplementedError

    def save_scores(self, session_id: str, scores: ScoreSet) -> None:  # pragma: no cover
        raise NotImplementedError

    def list_sessions(self) -> List[SessionMeta]:  # pragma: no cover
        raise NotImplementedError

    def load_scores(self, session_id: str) -> ScoreSet | None:  # pragma: no cover
        raise NotImplementedError

    def delete_session(self, session_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    def export_snapshot(self, session_id: str, include_audio: bool = False) -> dict:  # pragma: no cover
        raise NotImplementedError

    def integrity_scan(self) -> dict:  # pragma: no cover
        raise NotImplementedError
