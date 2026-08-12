from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import safe_job_id, validate_sha256

_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SPEAKER_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_NATIVE_SPEAKER_LINE_RE = re.compile(r"(?im)^\s*Speaker\s+\d+\s*:")
_MAX_DIALOGUE_CHARS = 40_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LifecycleState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    FAILED = "failed"


def _exact_commit(value: str, field_name: str) -> str:
    normalized = value.strip().lower()
    if not _COMMIT_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be an exact 40-character commit SHA")
    return normalized


def _speaker_id(value: str) -> str:
    if not _SPEAKER_ID_RE.fullmatch(value):
        raise ValueError(
            "speaker_id must start with a letter and contain only letters, digits, '_' or '-'"
        )
    return value


class SpeakerBinding(StrictModel):
    speaker_id: str = Field(min_length=1, max_length=64)
    reference_path: str = Field(min_length=1, max_length=4096)
    reference_sha256: str

    @field_validator("speaker_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _speaker_id(value)

    @field_validator("reference_path")
    @classmethod
    def _path(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("reference_path cannot contain NUL characters")
        if not value.strip():
            raise ValueError("reference_path cannot be blank")
        return value

    @field_validator("reference_sha256")
    @classmethod
    def _hash(cls, value: str) -> str:
        try:
            return validate_sha256(value)
        except Exception as exc:
            raise ValueError(str(exc)) from exc


class DialogueTurn(StrictModel):
    speaker_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("speaker_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return _speaker_id(value)

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank")
        if "\x00" in value:
            raise ValueError("text cannot contain NUL characters")
        # Native VibeVoice uses `Speaker N:` as control syntax. A turn may contain
        # newlines, but it may not smuggle another native speaker declaration.
        if _NATIVE_SPEAKER_LINE_RE.search(value):
            raise ValueError("text cannot contain native 'Speaker N:' control lines")
        return value


class GenerationSettings(StrictModel):
    cfg_scale: float | None = Field(default=None, ge=0.1, le=5.0)
    inference_steps: int | None = Field(default=None, ge=1, le=100)
    disable_prefill: bool = False
    max_length_times: float = Field(default=2.0, ge=1.0, le=4.0)


class LoadRequest(StrictModel):
    model_revision: str | None = None
    source_revision: str | None = None

    @field_validator("model_revision")
    @classmethod
    def _model_revision(cls, value: str | None) -> str | None:
        return None if value is None else _exact_commit(value, "model_revision")

    @field_validator("source_revision")
    @classmethod
    def _source_revision(cls, value: str | None) -> str | None:
        return None if value is None else _exact_commit(value, "source_revision")


class DialogueRequest(StrictModel):
    job_id: str
    model_revision: str
    source_revision: str | None = None
    language: Literal["en"] = "en"
    seed: int = Field(default=18427, ge=0, le=2**63 - 1)
    speakers: list[SpeakerBinding] = Field(min_length=1, max_length=4)
    turns: list[DialogueTurn] = Field(min_length=1, max_length=128)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)

    @field_validator("job_id")
    @classmethod
    def _job_id(cls, value: str) -> str:
        try:
            return safe_job_id(value)
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("model_revision")
    @classmethod
    def _model_revision(cls, value: str) -> str:
        return _exact_commit(value, "model_revision")

    @field_validator("source_revision")
    @classmethod
    def _source_revision(cls, value: str | None) -> str | None:
        return None if value is None else _exact_commit(value, "source_revision")

    @model_validator(mode="after")
    def _validate_speakers(self) -> "DialogueRequest":
        speaker_ids = [speaker.speaker_id for speaker in self.speakers]
        if len(speaker_ids) != len(set(speaker_ids)):
            raise ValueError("speaker_id values must be unique")
        known = set(speaker_ids)
        unknown = sorted(
            {turn.speaker_id for turn in self.turns if turn.speaker_id not in known}
        )
        if unknown:
            raise ValueError(f"turns reference unknown speakers: {unknown}")
        total_chars = sum(len(turn.text) for turn in self.turns)
        if total_chars > _MAX_DIALOGUE_CHARS:
            raise ValueError(
                f"dialogue text exceeds {_MAX_DIALOGUE_CHARS} characters for one bounded synthesis request"
            )
        return self


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: str = "stickman-vibevoice"


class ReadyResponse(StrictModel):
    state: LifecycleState
    ready: bool
    active_job_id: str | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    device_mode: str | None = None


class CapabilitiesResponse(StrictModel):
    source_repository: str
    source_revision: str
    model_repository: str
    model_revision: str
    multi_speaker: bool = True
    max_speakers: int = 4
    voice_prompting: bool = True
    streaming: bool = False
    languages: list[str] = Field(default_factory=lambda: ["en"])
    sample_rate_hz: int = 24_000
    max_concurrent_jobs: int = 1
    device_mode: str | None = None


class ActionResponse(StrictModel):
    state: LifecycleState
    message: str


class CancelResponse(StrictModel):
    job_id: str
    cancellation_requested: bool


class TimingInfo(StrictModel):
    load_seconds: float = 0.0
    generation_seconds: float = 0.0


class DialogueResult(StrictModel):
    job_id: str
    status: Literal["completed"] = "completed"
    output_path: str
    duration_seconds: float
    sample_rate_hz: int
    source_repository: str
    source_revision: str
    model_repository: str
    model_revision: str
    device_mode: str
    seed: int
    generation: dict[str, Any]
    speaker_reference_hashes: dict[str, str]
    audio_sha256: str
    timings: TimingInfo


class ErrorBody(StrictModel):
    error: str
    message: str
    details: Any = None
