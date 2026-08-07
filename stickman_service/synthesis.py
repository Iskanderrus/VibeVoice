from __future__ import annotations

import wave
from pathlib import Path

from .errors import InvalidOutputError
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


def validate_wav(path: Path) -> tuple[float, int]:
    if not path.is_file() or path.stat().st_size <= 44:
        raise InvalidOutputError("generated WAV is missing or empty")
    try:
        with wave.open(str(path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
    except (wave.Error, EOFError) as exc:
        raise InvalidOutputError(f"generated file is not a valid WAV: {exc}") from exc

    if frames <= 0 or rate <= 0 or channels <= 0 or width <= 0:
        raise InvalidOutputError("generated WAV has invalid audio metadata")
    return frames / float(rate), rate
