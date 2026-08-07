from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .errors import InvalidRequestError, ReferencePathError

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise InvalidRequestError("expected a lowercase or uppercase SHA-256 hex digest")
    return normalized


def safe_job_id(value: str) -> str:
    value = value.strip()
    if not _JOB_ID_RE.fullmatch(value):
        raise InvalidRequestError(
            "job_id must be 1-128 characters using letters, digits, '.', '_' or '-'"
        )
    return value


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_under_root(path_value: str, root: Path, *, require_wav: bool = False) -> Path:
    root_resolved = root.resolve()
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = root_resolved / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ReferencePathError("path escapes configured reference root") from exc
    if not candidate.is_file():
        raise ReferencePathError(f"reference file not found: {candidate}")
    if require_wav and candidate.suffix.lower() != ".wav":
        raise ReferencePathError("speaker reference must be a .wav file")
    return candidate
