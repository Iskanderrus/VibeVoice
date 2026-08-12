from __future__ import annotations

import wave
from pathlib import Path

from .errors import InvalidOutputError, ReferencePathError
from .schemas import DialogueRequest


def compile_dialogue(request: DialogueRequest) -> tuple[str, list[str]]:
    """Compile approved ordered turns to the native VibeVoice multi-speaker format."""
    number_by_id = {
        speaker.speaker_id: index
        for index, speaker in enumerate(request.speakers, start=1)
    }
    lines = [
        f"Speaker {number_by_id[turn.speaker_id]}: {turn.text}"
        for turn in request.turns
    ]
    ordered_ids = [speaker.speaker_id for speaker in request.speakers]
    return "\n".join(lines), ordered_ids


def _wav_metadata(path: Path, *, reference: bool) -> tuple[float, int, int, int]:
    error_cls = ReferencePathError if reference else InvalidOutputError
    if not path.is_file() or path.stat().st_size <= 44:
        raise error_cls(
            "reference WAV is missing or empty"
            if reference
            else "generated WAV is missing or empty"
        )
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
    except (wave.Error, EOFError, OSError) as exc:
        kind = "reference" if reference else "generated"
        raise error_cls(f"{kind} file is not a valid WAV: {exc}") from exc

    if frames <= 0 or rate <= 0 or channels <= 0 or width <= 0:
        kind = "reference" if reference else "generated"
        raise error_cls(f"{kind} WAV has invalid audio metadata")
    return frames / float(rate), rate, channels, width


def validate_reference_wav(path: Path, *, max_seconds: float) -> tuple[float, int]:
    duration, rate, channels, width = _wav_metadata(path, reference=True)
    if duration < 0.1:
        raise ReferencePathError("speaker reference must be at least 0.1 seconds")
    if duration > max_seconds:
        raise ReferencePathError(
            f"speaker reference exceeds maximum duration of {max_seconds:.1f} seconds"
        )
    if channels not in {1, 2}:
        raise ReferencePathError("speaker reference must be mono or stereo")
    if width not in {1, 2, 3, 4}:
        raise ReferencePathError("speaker reference uses an unsupported sample width")
    if not 8_000 <= rate <= 192_000:
        raise ReferencePathError("speaker reference uses an unsupported sample rate")
    return duration, rate


def validate_wav(path: Path) -> tuple[float, int]:
    duration, rate, channels, width = _wav_metadata(path, reference=False)
    if channels != 1:
        raise InvalidOutputError(f"generated WAV must be mono; got {channels} channels")
    if width not in {2, 4}:
        raise InvalidOutputError(
            f"generated WAV has unexpected sample width: {width} bytes"
        )
    return duration, rate
