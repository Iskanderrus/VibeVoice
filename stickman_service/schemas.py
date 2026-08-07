from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .hashing import safe_job_id, validate_sha256


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LifecycleState(str, Enum):
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    UNLOADING = "unloading"
    FAILED = "failed"


class SpeakerBinding(StrictModel):
    speaker_id: str = Field(min_length=1, max_length=64)
    reference_path: str = Field(min_length=1)
    reference_sha256: str

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

    @field_validator("text")
    @classmethod
    def _text(cls, value: str) -> str:
        # Preserve content exactly except for rejecting whitespace-only strings.
        if not value.strip():
            raise ValueError("text cannot be blank")
        return value


class GenerationSettings(StrictModel):
    cfg_scale: float | None = Field(default=None, ge=0.1, le=5.0)
    inference_steps: int | None = Field(default=None, ge=1, le=100)
    disable_prefill: bool = False
    max_length_times: float = Field(default=2.0, ge=1.0, le=4.0)


class LoadRequest(StrictModel):
    model_revision: str | None = None
    source_revision: str | None = None


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
