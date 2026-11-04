"""Trim voice references to a fixed length and record hashes of source and trimmed files.

Long references enlarge the reference-encoding activations on the GPU (measured: OOM at
27 s references, success at 8 s on a 4 GiB card). Trimming is therefore a parameter here.

usage: prepare_references.py --src DIR --dst DIR --seconds 8 --names NAME [NAME ...]
writes DST/references.json with source and output SHA-256 for each reference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import soundfile as sf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--names", nargs="+", required=True, help="file names without .wav")
    args = ap.parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)
    record = {}
    for name in args.names:
        src = args.src / f"{name}.wav"
        audio, sr = sf.read(src)
        out = args.dst / f"{name}.wav"
        sf.write(out, audio[: int(args.seconds * sr)], sr, subtype="PCM_16")
        record[name] = {
            "source_sha256": sha256(src),
            "output_sha256": sha256(out),
            "source_seconds": round(len(audio) / sr, 2),
            "output_seconds": round(min(len(audio), int(args.seconds * sr)) / sr, 2),
            "sample_rate": sr,
        }
    (args.dst / "references.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
