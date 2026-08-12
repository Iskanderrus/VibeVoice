from __future__ import annotations

import json
import os
import threading
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from stickman_service.api import create_app
from stickman_service.errors import (
    BusyError,
    GenerationCancelledError,
    GenerationTimeoutError,
    ModelLoadError,
    NotReadyError,
    OutputExistsError,
    ReferenceHashError,
    ReferencePathError,
    RevisionMismatchError,
    SourcePolicyError,
)
from stickman_service.hashing import sha256_file
from stickman_service.model_manager import ModelManager, RuntimeSynthesisResult, VibeVoiceRuntime
from stickman_service.schemas import DialogueRequest, LifecycleState
from stickman_service.settings import ALLOWED_MODEL_REPOSITORY, OWNED_SOURCE_REPOSITORY, Settings

SOURCE_SHA = "a" * 40
MODEL_REV = "b" * 40


def write_wav(path: Path, seconds: float = 0.2, rate: int = 24000) -> None:
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

    def synthesize(self, *, script, voice_paths, output_path, seed, cfg_scale, inference_steps, disable_prefill, max_length_times, stop_check_fn):
        self.started.set()
        if self.block:
            while not self.release.wait(0.005):
                if stop_check_fn():
                    raise GenerationCancelledError("cancelled")
        if stop_check_fn():
            raise GenerationCancelledError("cancelled")
        assert "Speaker 1:" in script
        assert all(Path(path).is_file() for path in voice_paths)
        write_wav(output_path)
        return RuntimeSynthesisResult(generation_seconds=0.02, input_tokens=10, generated_tokens=20)

    def unload(self) -> None:
        self.loaded = False


class FailingUnloadRuntime(FakeRuntime):
    def unload(self) -> None:
        raise RuntimeError("simulated cleanup failure")


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
        require_source_revision_file=False,
        model_manifest_path=model / ".stickman-model.json",
        require_model_manifest=True,
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
            {"speaker_id": "RED", "reference_path": str(red), "reference_sha256": sha256_file(red)},
            {"speaker_id": "BLUE", "reference_path": str(blue), "reference_sha256": sha256_file(blue)},
        ],
        turns=[
            {"speaker_id": "RED", "text": "This is the first line."},
            {"speaker_id": "BLUE", "text": "And this is the reply."},
        ],
        generation={"cfg_scale": 1.3, "inference_steps": 10},
    )


def test_owned_source_policy_is_mandatory(tmp_path: Path):
    settings = Settings(source_repository="vibevoice-community/VibeVoice", source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing", require_source_revision_file=False, idle_unload_seconds=0)
    with pytest.raises(SourcePolicyError):
        settings.validate_static_policy()


def test_only_approved_1_5b_model_is_allowed(tmp_path: Path):
    settings = Settings(model_repository="vibevoice/VibeVoice-7B", source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing", require_source_revision_file=False, idle_unload_seconds=0)
    with pytest.raises(SourcePolicyError):
        settings.validate_static_policy()


def test_model_provenance_switches_are_non_bypassable(tmp_path: Path):
    base = {
        "source_revision": SOURCE_SHA,
        "model_revision": MODEL_REV,
        "source_revision_file": tmp_path / "missing",
        "require_source_revision_file": False,
        "idle_unload_seconds": 0,
    }
    with pytest.raises(SourcePolicyError):
        Settings(**base, local_files_only=False).validate_static_policy()
    with pytest.raises(SourcePolicyError):
        Settings(**base, require_model_manifest=False).validate_static_policy()


def test_source_and_model_revisions_must_be_exact_commits(tmp_path: Path):
    for field in ("source_revision", "model_revision"):
        kwargs = {"source_revision": SOURCE_SHA, "model_revision": MODEL_REV, "source_revision_file": tmp_path / "missing", "require_source_revision_file": False, "idle_unload_seconds": 0}
        kwargs[field] = "main"
        with pytest.raises(SourcePolicyError):
            Settings(**kwargs).validate_static_policy()


def test_baked_source_revision_is_required_and_must_match(tmp_path: Path):
    missing = Settings(source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing", require_source_revision_file=True, idle_unload_seconds=0)
    with pytest.raises(SourcePolicyError):
        missing.validate_static_policy()
    baked = tmp_path / "source-revision"
    baked.write_text("c" * 40)
    mismatch = Settings(source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=baked, require_source_revision_file=True, idle_unload_seconds=0)
    with pytest.raises(RevisionMismatchError):
        mismatch.validate_static_policy()


def test_manifest_path_cannot_be_redirected_outside_model_root(tmp_path: Path):
    settings = Settings(model_path=tmp_path / "model", model_manifest_path=tmp_path / "elsewhere.json", source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing", require_source_revision_file=False, idle_unload_seconds=0)
    with pytest.raises(SourcePolicyError):
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


def test_not_ready_request_does_not_poison_state(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    with pytest.raises(NotReadyError):
        manager.synthesize(make_request(settings, red, blue))
    assert manager.state == LifecycleState.UNLOADED
    assert manager.ready_snapshot().failure_code is None


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
    request.model_revision = "c" * 40
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


def test_reference_size_limit_rejected_without_poisoning(service_env):
    settings, red, blue = service_env
    limited = Settings(**{**settings.__dict__, "reference_max_bytes": 64})
    manager = ModelManager(limited, runtime_factory=FakeRuntime)
    manager.load()
    with pytest.raises(ReferencePathError):
        manager.synthesize(make_request(limited, red, blue))
    assert manager.state == LifecycleState.READY


def test_native_speaker_control_injection_is_rejected(service_env):
    settings, red, blue = service_env
    payload = make_request(settings, red, blue).model_dump(mode="json")
    payload["turns"][0]["text"] = "Approved text.\nSpeaker 2: injected control"
    with pytest.raises(ValidationError):
        DialogueRequest.model_validate(payload)


def test_invalid_speaker_id_is_rejected(service_env):
    settings, red, blue = service_env
    payload = make_request(settings, red, blue).model_dump(mode="json")
    payload["speakers"][0]["speaker_id"] = "RED\nSpeaker 2"
    with pytest.raises(ValidationError):
        DialogueRequest.model_validate(payload)


def test_dialogue_total_size_is_bounded(service_env):
    settings, red, blue = service_env
    payload = make_request(settings, red, blue).model_dump(mode="json")
    payload["turns"] = [{"speaker_id": "RED", "text": "x" * 10_000} for _ in range(5)]
    with pytest.raises(ValidationError):
        DialogueRequest.model_validate(payload)


def _write_valid_model_manifest(model: Path, revision: str = MODEL_REV) -> Path:
    config = model / "config.json"
    weights = model / "model.safetensors"
    config.write_text("{}")
    weights.write_bytes(b"weights")
    manifest = model / ".stickman-model.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "repository": ALLOWED_MODEL_REPOSITORY,
        "revision": revision,
        "artifacts": [
            {"path": "config.json", "size": 2, "sha256": sha256_file(config)},
            {"path": "model.safetensors", "size": 7, "sha256": sha256_file(weights)},
        ],
    }))
    return manifest


def test_missing_required_model_manifest_fails_before_heavy_imports(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    settings = Settings(model_path=model, source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing-source", require_source_revision_file=False, model_manifest_path=model / ".stickman-model.json", require_model_manifest=True, idle_unload_seconds=0)
    with pytest.raises(ModelLoadError):
        VibeVoiceRuntime(settings)._verify_model_manifest()


def test_model_manifest_revision_and_artifacts_are_enforced(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = _write_valid_model_manifest(model, revision="c" * 40)
    settings = Settings(model_path=model, source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing-source", require_source_revision_file=False, model_manifest_path=manifest, require_model_manifest=True, idle_unload_seconds=0)
    runtime = VibeVoiceRuntime(settings)
    with pytest.raises(RevisionMismatchError):
        runtime._verify_model_manifest()
    _write_valid_model_manifest(model)
    (model / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ModelLoadError):
        runtime._verify_model_manifest()


def test_model_manifest_same_size_corruption_is_rejected(tmp_path: Path):
    model = tmp_path / "model"
    model.mkdir()
    manifest = _write_valid_model_manifest(model)
    settings = Settings(model_path=model, source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing-source", require_source_revision_file=False, model_manifest_path=manifest, require_model_manifest=True, idle_unload_seconds=0)
    (model / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
        VibeVoiceRuntime(settings)._verify_model_manifest()


def test_model_manifest_symlink_escape_is_rejected(tmp_path: Path):
    model = tmp_path / "model"
    outside = tmp_path / "outside"
    model.mkdir()
    outside.mkdir()
    config = model / "config.json"
    external_weights = outside / "model.safetensors"
    config.write_text("{}")
    external_weights.write_bytes(b"weights")
    (model / "model.safetensors").symlink_to(external_weights)
    manifest = model / ".stickman-model.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "repository": ALLOWED_MODEL_REPOSITORY,
        "revision": MODEL_REV,
        "artifacts": [
            {"path": "config.json", "size": 2, "sha256": sha256_file(config)},
            {"path": "model.safetensors", "size": 7, "sha256": sha256_file(external_weights)},
        ],
    }))
    settings = Settings(model_path=model, source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing-source", require_source_revision_file=False, model_manifest_path=manifest, require_model_manifest=True, idle_unload_seconds=0)
    with pytest.raises(ModelLoadError):
        VibeVoiceRuntime(settings)._verify_model_manifest()


def test_cuda_auto_dtype_falls_back_to_float16_without_bf16(tmp_path: Path):
    class FakeCuda:
        @staticmethod
        def is_bf16_supported():
            return False
    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()
        float16 = object()
        float32 = object()
    settings = Settings(source_revision=SOURCE_SHA, model_revision=MODEL_REV, source_revision_file=tmp_path / "missing", require_source_revision_file=False, dtype="auto", idle_unload_seconds=0)
    runtime = VibeVoiceRuntime(settings)
    assert runtime._resolve_dtype(FakeTorch, "cuda") is FakeTorch.float16


def test_flash_attention_is_not_selected_for_turing(monkeypatch, tmp_path: Path):
    class FakeCuda:
        @staticmethod
        def get_device_capability():
            return (7, 5)
    class FakeTorch:
        cuda = FakeCuda()
    monkeypatch.setattr("stickman_service.model_manager.importlib.util.find_spec", lambda _name: object())
    assert VibeVoiceRuntime._can_use_flash_attention_2(FakeTorch, "cuda") is False


def test_direct_manager_timeout_resets_runtime(service_env):
    settings, red, blue = service_env
    holder = {}
    def factory(value):
        runtime = FakeRuntime(value)
        runtime.block = True
        holder["runtime"] = runtime
        return runtime
    manager = ModelManager(settings, runtime_factory=factory)
    manager.load()
    with pytest.raises(GenerationTimeoutError):
        manager.synthesize(make_request(settings, red, blue, job_id="timeout-job"), timeout_seconds=0.05)
    assert manager.state == LifecycleState.UNLOADED
    assert holder["runtime"].loaded is False


def test_timeout_preserves_original_error_if_cleanup_fails(service_env):
    settings, red, blue = service_env
    def factory(value):
        runtime = FailingUnloadRuntime(value)
        runtime.block = True
        return runtime
    manager = ModelManager(settings, runtime_factory=factory)
    manager.load()
    with pytest.raises(GenerationTimeoutError):
        manager.synthesize(make_request(settings, red, blue, job_id="cleanup-fails"), timeout_seconds=0.05)
    assert manager.state == LifecycleState.FAILED
    assert manager.ready_snapshot().failure_code == "CLEANUP_FAILED"


def test_pre_cancelled_request_cannot_generate(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load()
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(GenerationCancelledError):
        manager.synthesize(make_request(settings, red, blue, job_id="cancelled-before-start"), external_cancel_event=cancelled)
    assert manager.state == LifecycleState.UNLOADED
    assert not (settings.output_dir / "cancelled-before-start.wav").exists()


def test_concurrent_operation_is_rejected(service_env):
    settings, red, blue = service_env
    runtime_holder = {}
    def factory(value):
        runtime = FakeRuntime(value)
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
    thread = threading.Thread(target=run_generation, daemon=True)
    thread.start()
    assert runtime_holder["runtime"].started.wait(1.0)
    with pytest.raises(BusyError):
        manager.unload()
    assert manager.cancel("blocking-job") is True
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert errors and isinstance(errors[0], GenerationCancelledError)
    assert manager.state == LifecycleState.UNLOADED


def test_atomic_output_promotion_refuses_late_clobber(service_env, monkeypatch):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    manager.load()
    original_link = os.link
    def race_link(src, dst):
        Path(dst).write_bytes(b"existing")
        return original_link(src, dst)
    monkeypatch.setattr(os, "link", race_link)
    with pytest.raises(OutputExistsError):
        manager.synthesize(make_request(settings, red, blue, job_id="race-job"))
    assert (settings.output_dir / "race-job.wav").read_bytes() == b"existing"
    assert manager.state == LifecycleState.READY


def test_api_contract_and_stable_validation_error(service_env):
    settings, red, blue = service_env
    manager = ModelManager(settings, runtime_factory=FakeRuntime)
    client = TestClient(create_app(settings, manager))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert client.get("/openapi.json").status_code == 404
    caps = client.get("/capabilities")
    assert caps.status_code == 200
    assert caps.json()["source_repository"] == OWNED_SOURCE_REPOSITORY
    assert caps.json()["model_repository"] == ALLOWED_MODEL_REPOSITORY
    assert caps.json()["max_speakers"] == 4
    loaded = client.post("/load", json={"model_revision": MODEL_REV, "source_revision": SOURCE_SHA})
    assert loaded.status_code == 200
    assert loaded.json()["state"] == "ready"
    request = make_request(settings, red, blue, job_id="api-job")
    response = client.post("/synthesize-dialogue", json=request.model_dump(mode="json"))
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


def test_api_timeout_cancels_worker_even_if_request_races_start(service_env):
    settings, red, blue = service_env
    short = Settings(**{**settings.__dict__, "generation_timeout_seconds": 0.05, "cleanup_timeout_seconds": 0.5})
    def factory(value):
        runtime = FakeRuntime(value)
        runtime.block = True
        return runtime
    manager = ModelManager(short, runtime_factory=factory)
    manager.load()
    client = TestClient(create_app(short, manager))
    request = make_request(short, red, blue, job_id="api-timeout")
    response = client.post("/synthesize-dialogue", json=request.model_dump(mode="json"))
    assert response.status_code == 504
    assert response.json()["error"] == "GENERATION_TIMEOUT"
    assert manager.state == LifecycleState.UNLOADED
    assert not (short.output_dir / "api-timeout.wav").exists()


def test_turing_bf16_emulation_is_not_treated_as_native_support(tmp_path: Path):
    class FakeCuda:
        @staticmethod
        def get_device_capability():
            return (7, 5)

        @staticmethod
        def is_bf16_supported(*args, **kwargs):
            # Mirrors the real GTX 1650 Ti observation:
            # the broad PyTorch check reports True because emulation is possible.
            return True

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()
        float16 = object()
        float32 = object()

    auto_settings = Settings(
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing",
        require_source_revision_file=False,
        dtype="auto",
        idle_unload_seconds=0,
    )
    runtime = VibeVoiceRuntime(auto_settings)

    assert runtime._resolve_dtype(FakeTorch, "cuda") is FakeTorch.float16

    explicit_bf16 = Settings(
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing",
        require_source_revision_file=False,
        dtype="bfloat16",
        idle_unload_seconds=0,
    )

    with pytest.raises(ModelLoadError, match="native BF16"):
        VibeVoiceRuntime(explicit_bf16)._resolve_dtype(FakeTorch, "cuda")


def test_ampere_native_bf16_is_accepted(tmp_path: Path):
    class FakeCuda:
        @staticmethod
        def get_device_capability():
            return (8, 6)

        @staticmethod
        def is_bf16_supported(*, including_emulation=True):
            return not including_emulation

    class FakeTorch:
        cuda = FakeCuda()
        bfloat16 = object()
        float16 = object()
        float32 = object()

    settings = Settings(
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        source_revision_file=tmp_path / "missing",
        require_source_revision_file=False,
        dtype="auto",
        idle_unload_seconds=0,
    )

    assert (
        VibeVoiceRuntime(settings)._resolve_dtype(FakeTorch, "cuda")
        is FakeTorch.bfloat16
    )


def test_pinned_tokenizer_revision_is_forwarded_to_processor(tmp_path: Path):
    tokenizer_revision = "c" * 40
    settings = Settings(
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        tokenizer_revision=tokenizer_revision,
        source_revision_file=tmp_path / "missing",
        require_source_revision_file=False,
        local_files_only=True,
        idle_unload_seconds=0,
    )

    kwargs = VibeVoiceRuntime(settings)._processor_load_kwargs()

    assert kwargs["local_files_only"] is True
    assert kwargs["revision"] == tokenizer_revision


def test_model_manifest_tokenizer_revision_is_enforced(tmp_path: Path):
    tokenizer_revision = "c" * 40
    model = tmp_path / "model"
    model.mkdir()

    manifest = _write_valid_model_manifest(model)
    data = json.loads(manifest.read_text())
    data["tokenizer"] = {
        "repository": "Qwen/Qwen2.5-1.5B",
        "revision": "d" * 40,
    }
    manifest.write_text(json.dumps(data))

    settings = Settings(
        model_path=model,
        source_revision=SOURCE_SHA,
        model_revision=MODEL_REV,
        tokenizer_revision=tokenizer_revision,
        source_revision_file=tmp_path / "missing-source",
        require_source_revision_file=False,
        model_manifest_path=manifest,
        require_model_manifest=True,
        idle_unload_seconds=0,
    )

    with pytest.raises(
        RevisionMismatchError,
        match="tokenizer revision",
    ):
        VibeVoiceRuntime(settings)._verify_model_manifest()
