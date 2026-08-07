#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

APPROVED_REPOSITORY = "vibevoice/VibeVoice-1.5B"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record the exact local VibeVoice 1.5B model identity for Stickman."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--repository",
        default=APPROVED_REPOSITORY,
        choices=[APPROVED_REPOSITORY],
    )
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        parser.error(f"model directory does not exist: {model_dir}")
    revision = args.revision.strip()
    if not revision:
        parser.error("--revision cannot be empty")

    manifest = {
        "schema_version": 1,
        "repository": args.repository,
        "revision": revision,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
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
