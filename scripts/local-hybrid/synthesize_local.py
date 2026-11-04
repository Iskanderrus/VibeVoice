"""Run one approved dialogue through the local hybrid profile and write WAV + provenance.

usage: synthesize_local.py --dialogue dialogue.json --out-dir DIR
dialogue.json: {"job_id", "seed", "speakers": [{"speaker_id", "reference_path"}],
                "turns": [{"speaker_id", "text"}], "generation": {"cfg_scale", "inference_steps",
                "disable_prefill", "max_length_times"}}
Settings come from the environment (load the profile with run_local.sh first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORK = HERE.parents[1]
sys.path.insert(0, str(FORK))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_text(turns: list[dict]) -> str:
    return "\n".join(f"Speaker {i}: {t['text']}" for i, t in _numbered(turns))


def _numbered(turns):
    order: dict[str, int] = {}
    for t in turns:
        order.setdefault(t["speaker_id"], len(order) + 1)
    return [(order[t["speaker_id"]], t) for t in turns]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogue", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    from pathlib import Path as P

    from stickman_service.model_manager import VibeVoiceRuntime
    from stickman_service.settings import Settings

    spec = json.loads(args.dialogue.read_text())
    gen = spec.get("generation", {})
    settings = Settings.from_env()
    settings.validate_static_policy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = args.out_dir / f"{spec['job_id']}.wav"

    refs = [P(s["reference_path"]) for s in spec["speakers"]]
    runtime = VibeVoiceRuntime(settings)
    load_s = runtime.load()
    started = time.monotonic()
    try:
        result = runtime.synthesize(
            script=script_text(spec["turns"]),
        voice_paths=refs,
        output_path=out_wav,
        seed=int(spec.get("seed", 18427)),
        cfg_scale=float(gen.get("cfg_scale", 1.3)),
        inference_steps=int(gen.get("inference_steps", 10)),
        disable_prefill=bool(gen.get("disable_prefill", False)),
        max_length_times=float(gen.get("max_length_times", 2.0)),
        stop_check_fn=lambda: time.monotonic() - started > settings.generation_timeout_seconds,
    )
    finally:
        device_mode = runtime.device_mode  # capture before unload
        runtime.unload()
    provenance = {
        "job_id": spec["job_id"],
        "wav": str(out_wav),
        "wav_sha256": sha256(out_wav),
        "profile": {k: v for k, v in __import__("os").environ.items() if k.startswith("VIBEVOICE_")},
        "device_mode": device_mode,
        "load_seconds": round(load_s, 2),
        "generation_seconds": round(time.monotonic() - started, 2),
        "references": {str(p): sha256(p) for p in refs},
        "seed": spec.get("seed", 18427),
        "generation": gen,
        "fork_commit": _git(FORK),
        "python": platform.python_version(),
        "result": repr(result),
    }
    (args.out_dir / f"{spec['job_id']}.provenance.json").write_text(json.dumps(provenance, indent=2))
    print(json.dumps({k: provenance[k] for k in ("wav", "wav_sha256", "device_mode", "generation_seconds")}, indent=2))


def _git(repo: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
