from __future__ import annotations

import json
import threading
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stickman_service.api import create_app
from stickman_service.errors import (
    BusyError,
    ModelLoadError,
    ReferenceHashError,
    ReferencePathError,
    RevisionMismatchError,
    SourcePolicyError,
)
from stickman_service.hashing import sha256_file
from stickman_service.model_manager import (
    ModelManager,
    RuntimeSynthesisResult,
    VibeVoiceRuntime,
)
from stickman_service.schemas import DialogueRequest, LifecycleState
from stickman_service.settings import (
    ALLOWED_MODEL_REPOSITORY,
    OWNED_SOURCE_REPOSITORY,
    Settings,
)

SOURCE_SHA = "a" * 40
MODEL_REV = "model-revision-123"


def write_wav(path: Path, seconds: float = 0.05, rate: int = 24000) -> None:
    frames = max(1, int(seconds * rate))
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"\x00\x00" * frames)


class FakeRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device_mode = "fake_cpu"
        self.loaded = False
        self.block = False
        self.started = threading.Event()
        self.release = threading.Event()

    def load(self) -> float:
        self.loaded = True
        return 0.01

    def synthesize(
        self,
        *,
        script,
        voice_paths,
        output_path,
        seed,
        cfg_scale,
        inference_steps,
        disable_prefill,
        max_length_times,
        stop_check_fn,
    ):
        self.started.set()
        if self.block:
            while not self.release.wait(0.01):
                if stop_check_fn():
                    from stickman_service.errors import GenerationCancelledError
                    raise GenerationCancelledError("cancelled")
        assert "Speaker 1:" in script
        write_wav(output_path)
        return RuntimeSynthesisResult(
            generation_seconds=0.02,
            input_tokens=10,
            generated_tokens=20,
        )

    def unload(self) -> None:
        self.loaded = False


@pytest.fixture
def service_env(tmp_path: Path):
    refs = tmp_path / "refs"
    output = tmp_path / "output"
    model = tmp_path / "model"
    offload = tmp_path / "offload"
    refs.mkdir()
    output.mkdir()
    model.mkdir()
    offload.mkdir()

    red = refs / "red.wav"
    blue = refs / "blue.wav"
    write_wav(red)
    write_wav(blue)

    settings = Settings(
        model_path=model,
        model_repository=ALLOWED_MODEL_REPOSITORY,
        model_revision=MODEL_REV,
        source_repository=OWNED_SOURCE_REPOSITORY,
        source_revision=SOURCE_SHA,
        source_revision_file=tmp_path / "no-source-file",
        model_manifest_path=model / ".stickman-model.json",
        require_model_manifest=False,
        device="cpu",
        cpu_offload=False,
        idle_unload_seconds=0,
        generation_timeout_seconds=2.0,
        cleanup_timeout_seconds=0.5,
        output_dir=output,
        reference_dir=refs,
        local_files_only=True,
    )
    return settings, red, blue


def make_request(settings: Settings, red: Path, blue: Path, job_id: str = "job-1"):
    return DialogueRequest(
        job_id=job_id,
        model_revision=settings.model_revision,
        source_revision=settings.source_revision,
        language="en",
        seed=123,
        speakers=[
            {
                "speaker_id": "RED",
                "reference_path": str(red),
                "reference_sha256": sha256_file(red),
            },
            {
                "speaker_id": "BLUE",
                "reference_path": str(blue),
                "reference_sha256": sha256_file(blue),
            },
        ],
        turns=[
            {"speaker_id": "RED", "text": "This is the first line."},
            {"speaker_id": "BLUE", "text": "And this is the reply."},
        ],
        generation={"cfg_scale": 1.3, "inference_steps": 10},
    )


def test_owned_source_policy_is_mandatory(tmp_path: Path):
    settings = Settings(
        source_repository="vibevoice-community/VibeVoice",
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing",
        idle_unload_seconds=0,
    )
    with pytest.raises(SourcePolicyError):
        settings.validate_static_policy()


def test_only_approved_1_5b_model_is_allowed(tmp_path: Path):
    settings = Settings(
        model_repository="vibevoice/VibeVoice-7B",
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing",
        idle_unload_seconds=0,
    )
    with pytest.raises(SourcePolicyError):
        settings.validate_static_policy()


def test_baked_source_revision_must_match(tmp_path: Path):
    baked = tmp_path / "source-revision"
    baked.write_text("b" * 40)
    settings = Settings(
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=baked,
        idle_unload_seconds=0,
    )
    with pytest.raises(RevisionMismatchError):
        settings.validate_static_policy()


def test_manager_load_generate_unload(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load(model_revision=MODEL_REV, source_revision=SOURCE_SHA)
    assert manager.state == LifecycleState.READY
    assert manager.device_mode == "fake_cpu"

    result = manager.synthesize(make_request(settings, red, blue))
    assert result.status == "completed"
    assert result.sample_rate_hz == 24000
    assert Path(result.output_path).is_file()
    assert result.source_repository == OWNED_SOURCE_REPOSITORY
    assert result.model_repository == ALLOWED_MODEL_REPOSITORY
    assert result.audio_sha256 == sha256_file(Path(result.output_path))
    assert manager.state == LifecycleState.READY

    manager.unload()
    assert manager.state == LifecycleState.UNLOADED


def test_reference_hash_mismatch_does_not_poison_model(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load()
    request = make_request(settings, red, blue)
    request.speakers[0].reference_sha256 = "0" * 64

    with pytest.raises(ReferenceHashError):
        manager.synthesize(request)
    assert manager.state == LifecycleState.READY


def test_revision_mismatch_rejected_without_generation(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load()
    request = make_request(settings, red, blue)
    request.model_revision = "different"

    with pytest.raises(RevisionMismatchError):
        manager.synthesize(request)
    assert manager.state == LifecycleState.READY


def test_reference_path_escape_rejected(service_env, tmp_path: Path):
    settings, red, blue = service_env
    outside = tmp_path / "outside.wav"
    write_wav(outside)
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load()
    request = make_request(settings, red, blue)
    request.speakers[0].reference_path = str(outside)
    request.speakers[0].reference_sha256 = sha256_file(outside)

    with pytest.raises(ReferencePathError):
        manager.synthesize(request)
    assert manager.state == LifecycleState.READY


def test_missing_required_model_manifest_fails_before_heavy_imports(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    settings = Settings(
        model_path=model,
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing-source",
        model_manifest_path=model / ".stickman-model.json",
        require_model_manifest=True,
        idle_unload_seconds=0,
    )
    runtime = VibeVoiceRuntime(settings)
    with pytest.raises(ModelLoadError):
        runtime._verify_model_manifest()


def test_model_manifest_revision_is_enforced(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = model / ".stickman-model.json"
    manifest.write_text(
        json.dumps(
            {
                "repository": ALLOWED_MODEL_REPOSITORY,
                "revision": "wrong-revision",
            }
        )
    )
    settings = Settings(
        model_path=model,
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing-source",
        model_manifest_path=manifest,
        require_model_manifest=True,
        idle_unload_seconds=0,
    )
    runtime = VibeVoiceRuntime(settings)
    with pytest.raises(RevisionMismatchError):
        runtime._verify_model_manifest()


def test_concurrent_operation_is_rejected(service_env):
    settings, red, blue = service_env
    runtime_holder = {}

    def factory(settings):
        runtime = FakeRuntime(settings)
        runtime.block = True
        runtime_holder["runtime"] = runtime
        return runtime

    manager = ModelManager(settings, runtime_factory=factory)
    manager.load()
    request = make_request(settings, red, blue, job_id="blocking-job")

    errors = []
    def run_generation():
        try:
            manager.synthesize(request)
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(
        target=run_generation,
        daemon=True,
    )
    thread.start()
    assert runtime_holder["runtime"].started.wait(1.0)

    with pytest.raises(BusyError):
        manager.unload()

    assert manager.cancel("blocking-job") is True
    runtime_holder["runtime"].release.set()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors


def test_api_contract_and_stable_validation_error(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    app = create_app(settings, manager)
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    caps = client.get("/capabilities")
    assert caps.status_code == 200
    assert caps.json()["source_repository"] == OWNED_SOURCE_REPOSITORY
    assert caps.json()["model_repository"] == ALLOWED_MODEL_REPOSITORY
    assert caps.json()["max_speakers"] == 4

    loaded = client.post(
        "/load",
        json={"model_revision": MODEL_REV, "source_revision": SOURCE_SHA},
    )
    assert loaded.status_code == 200
    assert loaded.json()["state"] == "ready"

    request = make_request(settings, red, blue, job_id="api-job")
    response = client.post(
        "/synthesize-dialogue",
        json=request.model_dump(mode="json"),
    )
    assert response.status_code == 200
    assert response.json()["status"] == "completed"

    bad = request.model_dump(mode="json")
    bad["job_id"] = "bad job id!"
    bad["language"] = "de"
    invalid = client.post("/synthesize-dialogue", json=bad)
    assert invalid.status_code == 422
    payload = invalid.json()
    assert payload["error"] == "INVALID_REQUEST"
    assert payload["message"] == "request validation failed"
    assert isinstance(payload["details"], list)

    unloaded = client.post("/unload")
    assert unloaded.status_code == 200
    assert unloaded.json()["state"] == "unloaded"
