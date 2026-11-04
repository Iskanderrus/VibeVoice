"""Bounded, model-free tests of automatic VibeVoice workstation configuration."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stickman_local.py"
spec = importlib.util.spec_from_file_location("run_stickman_local", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_auto_device_avoids_known_4gib_cuda_oom():
    assert runner.select_device("auto", 4096) == "hybrid"
    assert runner.select_device("auto", 8192) == "cuda"
    assert runner.select_device("auto", 6000) == "hybrid"
    assert runner.select_device("auto", 0) == "cpu"
    assert runner.select_device("cuda", 4096) == "cuda"


def test_reference_discovery_and_hashes_are_automatic(tmp_path: Path):
    directory = tmp_path / "references"
    directory.mkdir()
    for name in ("red.wav", "blue.wav"):
        (directory / name).write_bytes(b"local test WAV fixture")
    found, refs = runner.find_references(tmp_path, tmp_path / "missing_repo")
    assert found == directory
    assert refs["RED"] == directory / "red.wav"
    assert refs["BLUE"] == directory / "blue.wav"


def test_demo_references_are_fallback_only(tmp_path: Path):
    repo = tmp_path / "repository"
    voices = repo / "demo" / "voices"
    voices.mkdir(parents=True)
    (voices / "en-Carter_man.wav").write_bytes(b"red")
    (voices / "en-Alice_woman.wav").write_bytes(b"blue")
    found, refs = runner.find_references(tmp_path, repo)
    assert found == voices
    assert refs["BLUE"].name == "en-Alice_woman.wav"


def test_manifest_is_read_without_manual_revision_configuration(tmp_path: Path):
    model = tmp_path / "VibeVoice-1.5B"
    model.mkdir()
    (model / ".stickman-model.json").write_text(json.dumps({
        "repository": "vibevoice/VibeVoice-1.5B",
        "revision": "d374386b2a51d8e05277a64d85b296c89ec52376",
    }))
    assert runner.load_manifest(model)["revision"].startswith("d374")


def test_invalid_or_missing_model_manifest_rejected(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        runner.load_manifest(tmp_path)
    (tmp_path / ".stickman-model.json").write_text(json.dumps({
        "repository": "other-model", "revision": "some-revision"
    }))
    with pytest.raises(ValueError, match="Unexpected model"):
        runner.load_manifest(tmp_path)


def test_dialogue_accepts_red_blue_and_numbered_speaker_lines():
    assert runner.dialogue("RED: Hello.\nSpeaker 2: World.") == [
        ("RED", "Hello."), ("BLUE", "World.")
    ]
    with pytest.raises(ValueError):
        runner.dialogue("unsupported source")


def test_run_status_is_written_atomically(tmp_path: Path):
    path = tmp_path / "run.json"
    runner.write_json(path, {"status": "RUNNING"})
    assert json.loads(path.read_text()) == {"status": "RUNNING"}
    assert not path.with_name("run.json.tmp").exists()


def test_hybrid_reference_trimming_keeps_originals_unchanged(tmp_path: Path, monkeypatch):
    sf = pytest.importorskip("soundfile")
    monkeypatch.setattr(runner, "DATA", tmp_path)
    import numpy as np
    sources = {}
    for name in ("RED", "BLUE"):
        source = tmp_path / f"{name}.wav"
        sf.write(str(source), np.zeros(24000 * 10), 24000, subtype="PCM_16")
        sources[name] = source
    prepared = runner.prepare_hybrid_references(sources, tmp_path / "short")
    assert set(prepared) == {"RED", "BLUE"}
    assert all(sf.info(str(p)).duration == 8.0 for p in prepared.values())
    assert all(sf.info(str(p)).duration == 10.0 for p in sources.values())
    assert runner.prepare_hybrid_references(sources, tmp_path / "short") == prepared


HYBRID_IMPORT_CHILD = """
import importlib.util, sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_stickman_local", sys.argv[1])
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
data = Path(sys.argv[2])
runner.DATA = data
runner.prepare_hybrid_references = lambda refs, directory: refs

def stop_before_synthesis(*args, **kwargs):
    raise SystemExit(42)

runner.subprocess.run = stop_before_synthesis
config = {
    "references": {"RED": data / "RED.wav", "BLUE": data / "BLUE.wav"},
    "turns": [("RED", "Hello.")],
    "source_revision": "0" * 40,
    "manifest": {"revision": "test-revision"},
}
runner.run_hybrid("import-path", config, data / "out")
"""


def test_hybrid_runner_imports_stickman_service_outside_repo_without_pythonpath(
    tmp_path: Path,
):
    for name in ("RED", "BLUE"):
        (tmp_path / f"{name}.wav").write_bytes(b"fixture")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", HYBRID_IMPORT_CHILD, str(SCRIPT), str(tmp_path)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    # Exit 42 means run_hybrid got past the stickman_service import to the launch point.
    assert result.returncode == 42, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
