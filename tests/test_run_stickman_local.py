"""Bounded, model-free tests of automatic VibeVoice workstation configuration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_stickman_local.py"
spec = importlib.util.spec_from_file_location("run_stickman_local", SCRIPT)
assert spec is not None and spec.loader is not None
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_auto_device_avoids_known_4gib_cuda_oom():
    assert runner.select_device("auto", 4096) == "cpu"
    assert runner.select_device("auto", 8192) == "cuda"
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
