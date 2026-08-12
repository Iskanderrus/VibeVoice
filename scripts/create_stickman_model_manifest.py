#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

APPROVED_REPOSITORY = "vibevoice/VibeVoice-1.5B"
APPROVED_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_IGNORED_DIRS = {".cache", "__pycache__"}
_TOKENIZER_CACHE_ROOT = Path(".hf-cache")
_TOKENIZER_CACHE_RELATIVE = Path(".hf-cache/hub/models--Qwen--Qwen2.5-1.5B")
_REQUIRED_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
)


def _sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(model_dir: Path, path: Path) -> dict[str, object]:
    root = model_dir.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"model path escapes model directory: {path}") from exc
    if not resolved.is_file():
        raise ValueError(f"required model artifact is missing: {path}")
    return {
        "path": path.relative_to(model_dir).as_posix(),
        "size": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _artifact_inventory(model_dir: Path) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    for path in sorted(model_dir.rglob("*")):
        relative_parts = path.relative_to(model_dir).parts
        if not relative_parts:
            continue
        # Hugging Face cache internals are intentionally excluded here. A stable,
        # curated tokenizer subset is added separately below so ephemeral locks,
        # tree metadata, duplicate blobs, and other cache bookkeeping never become
        # part of the governed model identity.
        if relative_parts[0] == _TOKENIZER_CACHE_ROOT.name:
            continue
        if any(part in _IGNORED_DIRS for part in relative_parts):
            continue
        if path.name == ".stickman-model.json":
            continue
        if path.is_symlink() or path.is_file():
            artifacts.append(_artifact_record(model_dir, path))
    return artifacts


def _tokenizer_artifacts(model_dir: Path, revision: str) -> list[dict[str, object]]:
    cache_repo = model_dir / _TOKENIZER_CACHE_RELATIVE
    ref_file = cache_repo / "refs/main"
    snapshot = cache_repo / "snapshots" / revision
    artifacts = [_artifact_record(model_dir, ref_file)]
    artifacts.extend(
        _artifact_record(model_dir, snapshot / filename)
        for filename in _REQUIRED_TOKENIZER_FILES
    )
    return artifacts


def _validate_tokenizer_cache(model_dir: Path, revision: str) -> None:
    root = model_dir.resolve()
    cache_repo = model_dir / _TOKENIZER_CACHE_RELATIVE
    ref_file = cache_repo / "refs/main"
    if not ref_file.is_file():
        raise ValueError(f"pinned Qwen tokenizer cache ref is missing: {ref_file}")
    cached_revision = ref_file.read_text(encoding="utf-8").strip().lower()
    if cached_revision != revision:
        raise ValueError(
            "Qwen tokenizer cache refs/main does not match --tokenizer-revision"
        )
    snapshot = cache_repo / "snapshots" / revision
    if not snapshot.is_dir():
        raise ValueError(f"pinned Qwen tokenizer snapshot is missing: {snapshot}")
    for filename in _REQUIRED_TOKENIZER_FILES:
        artifact = snapshot / filename
        try:
            resolved = artifact.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"Qwen tokenizer artifact escapes model bundle: {artifact}"
            ) from exc
        if not resolved.is_file():
            raise ValueError(
                f"required Qwen tokenizer artifact is missing: {artifact}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Record and hash the exact local VibeVoice 1.5B model plus its pinned "
            "Qwen tokenizer cache for Stickman."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--repository",
        default=APPROVED_REPOSITORY,
        choices=[APPROVED_REPOSITORY],
    )
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--tokenizer-repository",
        default=APPROVED_TOKENIZER_REPOSITORY,
        choices=[APPROVED_TOKENIZER_REPOSITORY],
    )
    parser.add_argument("--tokenizer-revision", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    revision = args.revision.strip().lower()
    tokenizer_revision = args.tokenizer_revision.strip().lower()
    if not _COMMIT_RE.fullmatch(revision):
        parser.error("--revision must be an exact 40-character commit SHA")
    if not _COMMIT_RE.fullmatch(tokenizer_revision):
        parser.error("--tokenizer-revision must be an exact 40-character commit SHA")

    try:
        _validate_tokenizer_cache(model_dir, tokenizer_revision)
        artifacts = _artifact_inventory(model_dir)
        artifacts.extend(_tokenizer_artifacts(model_dir, tokenizer_revision))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if not artifacts:
        parser.error("model directory contains no model artifacts")
    names = {str(item["path"]) for item in artifacts}
    if len(names) != len(artifacts):
        parser.error("model artifact inventory contains duplicate paths")
    if "config.json" not in names:
        parser.error("model directory must contain config.json")
    if not any(Path(name).suffix.lower() in {".safetensors", ".bin"} for name in names):
        parser.error("model directory must contain at least one weight file")

    manifest = {
        "schema_version": 1,
        "repository": args.repository,
        "revision": revision,
        "tokenizer": {
            "repository": args.tokenizer_repository,
            "revision": tokenizer_revision,
            "cache_ref": (_TOKENIZER_CACHE_RELATIVE / "refs/main").as_posix(),
        },
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    path = model_dir / ".stickman-model.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
