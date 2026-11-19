#!/usr/bin/env python3
"""One-command, local-only VibeVoice 1.5B runner. No hand-edited .env or Docker."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path.home() / ".local" / "share" / "stickman-vibevoice"
MODEL = DATA / "models" / "VibeVoice-1.5B"
MIN_GPU_MIB = 8192  # 4 GiB GTX 1650 Ti OOMs during generation.
EXAMPLE = (
    "RED: Why does remembering something feel like proof it happened?\n"
    "BLUE: Because the brain reconstructs memories rather than replaying them."
)
LABEL = re.compile(r"^(?:Speaker\s+)?(RED|BLUE|[12])\s*:\s*(.+)$", re.I)


def source_revision() -> str:
    sha = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise RuntimeError("Cannot determine exact VibeVoice source commit")
    return sha


def load_manifest(model: Path) -> dict:
    path = model / ".stickman-model.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("repository") != "vibevoice/VibeVoice-1.5B":
        raise ValueError("Unexpected model repository in manifest")
    if not data.get("revision"):
        raise ValueError("Missing model revision in manifest")
    return data


def find_references(data: Path = DATA, repo: Path = REPO) -> tuple[Path, dict]:
    candidates = [
        (data / "references", {"RED": "red.wav", "BLUE": "blue.wav"}),
        (repo / "demo" / "voices", {
            "RED": "en-Carter_man.wav", "BLUE": "en-Alice_woman.wav"
        }),
    ]
    for directory, names in candidates:
        if all((directory / file).is_file() for file in names.values()):
            return directory.resolve(), {
                speaker: (directory / file).resolve()
                for speaker, file in names.items()
            }
    raise FileNotFoundError(
        "No two-speaker local test WAV pair found in "
        f"{data / 'references'} or {repo / 'demo' / 'voices'}"
    )


def dialogue(text: str) -> list[tuple[str, str]]:
    turns: list[tuple[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = LABEL.fullmatch(line.strip())
        if match is None:
            raise ValueError(f"Expected RED:/BLUE: or Speaker 1:/2: line: {line[:80]}")
        name = match.group(1).upper()
        turns.append(("RED" if name in {"RED", "1"} else "BLUE", match.group(2)))
    if not turns:
        raise ValueError("Dialogue cannot be empty")
    return turns


def gpu_vram_mib() -> int:
    try:
        p = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True
        )
        return int(p.stdout.splitlines()[0].strip())
    except (FileNotFoundError, ValueError, IndexError,
            subprocess.SubprocessError):
        return 0


def select_device(requested: str, vram_mib: int) -> str:
    if requested != "auto":
        return requested
    if 3500 <= vram_mib < MIN_GPU_MIB:
        return "hybrid"
    return "cuda" if vram_mib >= MIN_GPU_MIB else "cpu"


def write_json(path: Path, record: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


DEFAULT_SEED = 18427
DEFAULT_CFG_SCALE = 1.3


def parse_voices(items: list[str] | None) -> dict[str, Path]:
    """`--voice RED=/path/a.wav --voice BLUE=/path/b.wav`: an explicit, complete speaker pair."""
    voices: dict[str, Path] = {}
    for item in items or []:
        speaker, separator, raw = item.partition("=")
        speaker = speaker.strip().upper()
        if not separator or speaker not in {"RED", "BLUE"} or not raw.strip():
            raise ValueError("--voice must be RED=<wav path> or BLUE=<wav path>")
        if speaker in voices:
            raise ValueError(f"--voice given twice for {speaker}")
        path = Path(raw.strip()).expanduser().resolve()
        if not path.is_file() or path.suffix.lower() != ".wav":
            raise FileNotFoundError(f"Voice reference for {speaker} is not a WAV file: {path}")
        voices[speaker] = path
    if voices and set(voices) != {"RED", "BLUE"}:
        raise ValueError("--voice must set both RED and BLUE, never one of them")
    return voices


def job_config(args) -> dict:
    model = (args.model_dir or MODEL).expanduser().resolve()
    manifest = load_manifest(model)
    voices = parse_voices(getattr(args, "voice", None))
    if voices:
        references_root = Path(os.path.commonpath([str(p.parent) for p in voices.values()]))
        references = voices
    else:
        references_root, references = find_references()
    seed = getattr(args, "seed", None)
    seed = DEFAULT_SEED if seed is None else seed
    cfg_scale = getattr(args, "cfg_scale", None)
    cfg_scale = DEFAULT_CFG_SCALE if cfg_scale is None else cfg_scale
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31:
        raise ValueError("--seed must be an integer in 0..2147483647")
    if not 1.0 <= float(cfg_scale) <= 3.0:
        raise ValueError("--cfg-scale must be between 1.0 and 3.0")
    turns = dialogue(
        args.script.read_text(encoding="utf-8") if args.script else EXAMPLE
    )
    vram = gpu_vram_mib()
    device = select_device(args.device, vram)
    return {
        "model_path": model, "manifest": manifest, "references_root": references_root,
        "references": references, "turns": turns, "device": device, "vram_mib": vram,
        "source_revision": source_revision(), "seed": seed, "cfg_scale": float(cfg_scale),
    }



def prepare_hybrid_references(refs: dict, directory: Path) -> dict:
    """Trim temporary test voice samples automatically; never touch originals."""
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    prepared = {}
    for speaker, source in refs.items():
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        target = directory / f"{speaker.lower()}.wav"
        stamp = directory / f"{speaker.lower()}.source-sha256"
        if not (target.is_file() and stamp.is_file()
                and stamp.read_text().strip() == source_hash):
            audio, sample_rate = sf.read(str(source))
            if sample_rate <= 0:
                raise ValueError("Reference WAV has invalid sample rate")
            frames = audio[: int(sample_rate * 8)]
            if len(frames) == 0:
                raise ValueError(f"Empty voice sample: {source}")
            temp = directory / f".{speaker.lower()}.tmp.wav"
            sf.write(str(temp), frames, sample_rate, subtype="PCM_16")
            os.replace(temp, target)
            stamp.write_text(source_hash + "\n", encoding="utf-8")
        prepared[speaker] = target.resolve()
    return prepared


def run_hybrid(job_id: str, config: dict, output: Path) -> dict:
    """Use the on-device 4 GiB hybrid executor and its external thermal guard."""
    prepared = prepare_hybrid_references(
        config["references"], DATA / "references-8s"
    )
    runs = DATA / "runs" / job_id
    runs.mkdir(parents=True, exist_ok=True)
    spec_path = runs / "dialogue.json"
    if str(REPO) not in sys.path:  # direct script runs put scripts/, not REPO, on sys.path
        sys.path.insert(0, str(REPO))
    from stickman_service.hashing import sha256_file

    spec = {
        "job_id": job_id,
        "seed": config.get("seed", DEFAULT_SEED),
        "speakers": [
            {"speaker_id": key, "reference_path": str(path),
             "reference_sha256": sha256_file(path)}
            for key, path in prepared.items()
        ],
        "turns": [
            {"speaker_id": key, "text": phrase}
            for key, phrase in config["turns"]
        ],
        "generation": {
            "cfg_scale": config.get("cfg_scale", DEFAULT_CFG_SCALE),
            "inference_steps": 10,
            "disable_prefill": False,
            "max_length_times": 2.0,
        },
    }
    write_json(spec_path, spec)
    command = [
        "bash", str(REPO / "scripts/local-hybrid/run_local.sh"),
        str(spec_path), str(output),
        str(REPO / "scripts/local-hybrid/profile-gtx1650ti-hybrid.env"),
        str(runs),
    ]
    result = subprocess.run(command, cwd=REPO, check=False)
    if result.returncode != 0:
        thermal_log = runs / "thermal.log"
        tail = (thermal_log.read_text(encoding="utf-8")[-1000:]
                if thermal_log.is_file() else "no thermal log")
        raise RuntimeError(
            f"Hybrid VibeVoice exit={result.returncode}; thermal tail: {tail}"
        )
    provenance = output / f"{job_id}.provenance.json"
    if not provenance.is_file():
        raise RuntimeError("Hybrid runner exited without provenance JSON")
    data = json.loads(provenance.read_text(encoding="utf-8"))
    wav = output / f"{job_id}.wav"
    if (not wav.is_file() or not data.get("wav_sha256")
            or data["wav_sha256"] != sha256_file(wav)):
        raise RuntimeError("Hybrid WAV missing or provenance digest mismatch")
    return {
        "job_id": job_id, "status": "completed",
        "output_path": str(wav),
        "audio_sha256": data["wav_sha256"],
        "device_mode": data["device_mode"],
        "source_revision": config["source_revision"],
        "model_revision": config["manifest"]["revision"],
        "speaker_reference_hashes": data.get("references", {}),
        "timings": {
            "load_seconds": data.get("load_seconds"),
            "generation_seconds": data.get("generation_seconds"),
        },
        "provenance_path": str(provenance),
        "thermal_log": str(runs / "thermal.log"),
    }


QUALIFIED_DEVICES = frozenset({"hybrid"})


def require_qualified_device(device: str) -> None:
    """Only the hybrid route traverses the hardened admission, watchdog and cleanup path.

    Explicit or automatic CPU/CUDA execution is deliberately not offered: it would bypass those
    safeguards, and it is never a fallback for a blocked or failed hybrid run.
    """
    if device not in QUALIFIED_DEVICES:
        raise RuntimeError(
            f"BLOCKED_AUDIO_RUNTIME: device '{device}' is not qualified; only "
            f"{sorted(QUALIFIED_DEVICES)} runs under the thermal gate, watchdog and scope cleanup"
        )


def execute(job_id: str, config: dict, output: Path) -> dict:
    require_qualified_device(config["device"])
    return run_hybrid(job_id, config, output)


def run(args) -> int:
    config = job_config(args)
    public = {
        "job_id": args.job_id,
        "device": config["device"],
        "gpu_vram_mib": config["vram_mib"],
        "source_revision": config["source_revision"],
        "model_revision": config["manifest"]["revision"],
        "reference_hashes": {
            key: hashlib.sha256(path.read_bytes()).hexdigest()
            for key, path in config["references"].items()
        },
        "output_dir": str(DATA / "output"),
        "qualified_route": config["device"] in QUALIFIED_DEVICES,
        "seed": config["seed"],
        "cfg_scale": config["cfg_scale"],
        "voice_sources": {key: str(path) for key, path in config["references"].items()},
    }
    if args.dry_run:
        print(json.dumps(public, indent=2))
        return 0

    require_qualified_device(config["device"])
    DATA.mkdir(parents=True, exist_ok=True)
    runs = DATA / "runs"
    runs.mkdir(exist_ok=True)
    with (runs / "execution.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another VibeVoice local job is running")
        state = runs / f"{args.job_id}.json"
        write_json(state, {**public, "status": "RUNNING", "pid": os.getpid()})
        try:
            output = DATA / "output"
            output.mkdir(exist_ok=True)
            result = execute(args.job_id, config, output)
            write_json(state, {**public, "status": "SUCCEEDED", "result": result})
            print(json.dumps(result, indent=2), flush=True)
            return 0
        except Exception as exc:
            write_json(state, {**public, "status": "FAILED", "error": str(exc)})
            raise


def main() -> int:
    # Use the existing project venv automatically if this entrypoint was invoked
    # with system Python; no pip install, Docker pull or env-file editing.
    interpreters = [
        DATA / "venv-rebuild" / "bin" / "python",
        REPO / ".venv" / "bin" / "python",
    ]
    if sys.prefix == sys.base_prefix:
        for interpreter in interpreters:
            if interpreter.is_file():
                os.execv(str(interpreter), [str(interpreter), str(Path(__file__)), *sys.argv[1:]])

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, help="RED:/BLUE: lines; default short sample")
    parser.add_argument("--device", choices=("auto", "hybrid", "cpu", "cuda"), default="auto")
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--voice", action="append", metavar="SPEAKER=WAV",
                        help="reference voice per speaker, both required: RED=a.wav --voice BLUE=b.wav")
    parser.add_argument("--seed", type=int, default=None, help=f"default {DEFAULT_SEED}")
    parser.add_argument("--cfg-scale", type=float, default=None,
                        help=f"voice guidance 1.0-3.0, default {DEFAULT_CFG_SCALE}")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--foreground", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    runs = DATA / "runs"
    if args.status:
        states = sorted(runs.glob("vv-*.json"), key=lambda p: p.stat().st_mtime)
        if not states:
            print("No local VibeVoice runs yet")
            return 0
        print(states[-1].read_text(encoding="utf-8"))
        return 0
    args.job_id = args.job_id or f"vv-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", args.job_id):
        parser.error("--job-id must be a safe, short filename component")
    try:
        if args.foreground or args.dry_run:
            return run(args)
        # A detached worker survives tool/terminal timeouts. The worker owns the
        # file lock, thermal guard, model lifecycle and status JSON.
        require_qualified_device(job_config(args)["device"])  # Fail fast, before detaching.
        runs.mkdir(parents=True, exist_ok=True)
        log = runs / f"{args.job_id}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()), "--foreground",
            "--job-id", args.job_id, "--device", args.device
        ]
        if args.script:
            command += ["--script", str(args.script.resolve())]
        if args.model_dir:
            command += ["--model-dir", str(args.model_dir.resolve())]
        for voice in args.voice or []:
            command += ["--voice", voice]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        if args.cfg_scale is not None:
            command += ["--cfg-scale", str(args.cfg_scale)]
        status_path = runs / f"{args.job_id}.json"
        write_json(status_path, {
            "job_id": args.job_id, "status": "STARTING", "log": str(log)
        })
        with log.open("w") as stream:
            child = subprocess.Popen(
                command, cwd=REPO, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True
            )
        print(json.dumps({
            "job_id": args.job_id, "pid": child.pid, "log": str(log),
            "status": str(runs / f"{args.job_id}.json"),
            "output": str(DATA / "output" / f"{args.job_id}.wav")
        }, indent=2))
        return 0
    except Exception as exc:
        print(f"VibeVoice local run failed: {exc}", file=sys.stderr)
        if args.foreground:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
