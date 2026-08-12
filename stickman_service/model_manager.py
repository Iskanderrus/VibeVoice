from __future__ import annotations

import gc
import importlib.util
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .errors import (
    BusyError,
    GenerationCancelledError,
    GenerationError,
    GenerationTimeoutError,
    InvalidOutputError,
    ModelLoadError,
    NotReadyError,
    OutputExistsError,
    ReferenceHashError,
    ReferencePathError,
    RevisionMismatchError,
    ServiceError,
    classify_runtime_exception,
)
from .hashing import resolve_under_root, sha256_file
from .schemas import DialogueRequest, DialogueResult, LifecycleState, ReadyResponse, TimingInfo
from .settings import Settings
from .synthesis import compile_dialogue, validate_reference_wav, validate_wav


@dataclass
class RuntimeSynthesisResult:
    generation_seconds: float
    input_tokens: int | None = None
    generated_tokens: int | None = None


class VibeVoiceRuntime:
    """Thin, lazy wrapper around the fork's native VibeVoice inference classes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None
        self.device_mode: str | None = None
        self.input_device: str = "cpu"

    def _verify_model_manifest(self) -> None:
        model_path = self.settings.model_path
        if self.settings.local_files_only and not model_path.is_dir():
            raise ModelLoadError(f"local model path does not exist: {model_path}")

        manifest_path = self.settings.model_manifest_path
        if not manifest_path.is_file():
            if self.settings.require_model_manifest:
                raise ModelLoadError(f"required model manifest is missing: {manifest_path}")
            return

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"cannot read model manifest: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ModelLoadError("model manifest schema_version must be 1")
        if data.get("repository") != self.settings.model_repository:
            raise RevisionMismatchError("model manifest repository does not match configured repository")
        if str(data.get("revision", "")).lower() != self.settings.model_revision.lower():
            raise RevisionMismatchError("model manifest revision does not match configured revision")

        if self.settings.tokenizer_revision:
            tokenizer = data.get("tokenizer")
            if not isinstance(tokenizer, dict):
                raise ModelLoadError("model manifest must contain tokenizer identity")
            if tokenizer.get("repository") != self.settings.tokenizer_repository:
                raise RevisionMismatchError(
                    "model manifest tokenizer repository does not match configured repository"
                )
            if (
                str(tokenizer.get("revision", "")).lower()
                != self.settings.tokenizer_revision.lower()
            ):
                raise RevisionMismatchError(
                    "model manifest tokenizer revision does not match configured revision"
                )

        artifacts = data.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ModelLoadError("model manifest must contain a non-empty artifacts list")

        root = model_path.resolve()
        saw_config = False
        saw_weight = False
        seen_paths: set[str] = set()
        for item in artifacts:
            if not isinstance(item, dict):
                raise ModelLoadError("model manifest artifact entries must be objects")
            relative = item.get("path")
            size = item.get("size")
            expected_sha256 = item.get("sha256")
            if not isinstance(relative, str) or not relative:
                raise ModelLoadError("model manifest artifact path is invalid")
            if relative in seen_paths:
                raise ModelLoadError(f"model manifest contains duplicate artifact: {relative}")
            seen_paths.add(relative)
            rel_path = Path(relative)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise ModelLoadError("model manifest artifact path escapes model root")
            if not isinstance(size, int) or size < 0:
                raise ModelLoadError("model manifest artifact size is invalid")
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in expected_sha256
            ):
                raise ModelLoadError(f"model artifact SHA-256 is invalid: {relative}")
            artifact = model_path / rel_path
            try:
                resolved = artifact.resolve()
                resolved.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ModelLoadError(f"model artifact escapes model root: {relative}") from exc
            if not resolved.is_file():
                raise ModelLoadError(f"model artifact is missing: {relative}")
            if resolved.stat().st_size != size:
                raise ModelLoadError(f"model artifact size mismatch: {relative}")
            try:
                actual_sha256 = sha256_file(resolved)
            except OSError as exc:
                raise ModelLoadError(f"cannot hash model artifact {relative}: {exc}") from exc
            if actual_sha256 != expected_sha256.lower():
                raise ModelLoadError(f"model artifact SHA-256 mismatch: {relative}")
            if relative == "config.json":
                saw_config = True
            if resolved.suffix.lower() in {".safetensors", ".bin"}:
                saw_weight = True
        if not saw_config:
            raise ModelLoadError("model manifest does not contain config.json")
        if not saw_weight:
            raise ModelLoadError("model manifest does not contain model weight files")

    def _resolve_device(self, torch: Any) -> str:
        configured = self.settings.device
        if configured != "auto":
            if configured == "cuda" and not torch.cuda.is_available():
                raise ModelLoadError("CUDA requested but not available")
            if configured == "mps" and not torch.backends.mps.is_available():
                raise ModelLoadError("MPS requested but not available")
            return configured
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _cuda_has_native_bf16(torch: Any) -> bool:
        """Return true only when the CUDA device has native BF16 support.

        PyTorch's no-argument is_bf16_supported() may report True when BF16 is
        emulated on pre-Ampere GPUs. Stickman's constrained-GPU path must not
        select that emulated route automatically.
        """
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            return False
        if major < 8:
            return False

        check = getattr(torch.cuda, "is_bf16_supported", None)
        if check is None:
            return False
        try:
            return bool(check(including_emulation=False))
        except TypeError:
            # Compatibility with PyTorch versions predating the
            # including_emulation argument. Capability >= 8 remains mandatory.
            try:
                return bool(check())
            except Exception:
                return False
        except Exception:
            return False

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        configured = self.settings.dtype
        if configured == "auto":
            if device == "cuda":
                return (
                    torch.bfloat16
                    if self._cuda_has_native_bf16(torch)
                    else torch.float16
                )
            return torch.float32
        if device in {"cpu", "mps"} and configured != "float32":
            raise ModelLoadError(
                f"{configured} is not an approved dtype for {device}; use float32 or auto"
            )
        if (
            device == "cuda"
            and configured == "bfloat16"
            and not self._cuda_has_native_bf16(torch)
        ):
            raise ModelLoadError(
                "bfloat16 requested but CUDA device does not have native BF16 support"
            )
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[configured]

    @staticmethod
    def _can_use_flash_attention_2(torch: Any, device: str) -> bool:
        if device != "cuda" or importlib.util.find_spec("flash_attn") is None:
            return False
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            return False
        return major >= 8

    def _resolve_input_device(self, fallback: str) -> str:
        candidates: list[Any] = []
        try:
            candidates.append(self.model.get_input_embeddings().weight.device)
        except Exception:
            pass
        try:
            candidates.append(self.model.device)
        except Exception:
            pass
        for candidate in candidates:
            value = str(candidate)
            if value and value != "meta":
                return value
        return fallback

    def _processor_load_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.settings.local_files_only:
            kwargs["local_files_only"] = True
        if self.settings.tokenizer_revision:
            # VibeVoiceProcessor forwards this revision to the Qwen tokenizer
            # loader. Without it, Hugging Face resolves the moving `main` ref,
            # which is intentionally absent from the governed offline cache.
            kwargs["revision"] = self.settings.tokenizer_revision
        return kwargs

    def load(self) -> float:
        self._verify_model_manifest()
        started = time.monotonic()
        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            self.torch = torch
            device = self._resolve_device(torch)
            dtype = self._resolve_dtype(torch, device)
            source = str(self.settings.model_path) if self.settings.model_path.exists() else self.settings.model_repository
            processor_kwargs = self._processor_load_kwargs()
            model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            using_local_path = self.settings.model_path.exists()
            if self.settings.local_files_only:
                model_kwargs["local_files_only"] = True
            elif not using_local_path:
                model_kwargs["revision"] = self.settings.model_revision

            attn_primary = "flash_attention_2" if self._can_use_flash_attention_2(torch, device) else "sdpa"
            model_kwargs["attn_implementation"] = attn_primary
            if device == "cuda" and self.settings.cpu_offload:
                self.settings.offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs.update({
                    "device_map": "auto",
                    "max_memory": {0: f"{self.settings.cuda_max_memory_mb}MiB", "cpu": f"{self.settings.cpu_max_memory_gb}GiB"},
                    "offload_folder": str(self.settings.offload_dir),
                    "offload_state_dict": True,
                })
                self.device_mode = "cuda_cpu_offload"
            elif device == "cuda":
                model_kwargs["device_map"] = "cuda"
                self.device_mode = "cuda"
            elif device == "mps":
                model_kwargs["device_map"] = None
                self.device_mode = "mps"
            else:
                model_kwargs["device_map"] = "cpu"
                self.device_mode = "cpu"

            self.processor = VibeVoiceProcessor.from_pretrained(source, **processor_kwargs)
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(source, **model_kwargs)
            except Exception:
                if attn_primary != "flash_attention_2":
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model_kwargs["attn_implementation"] = "sdpa"
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(source, **model_kwargs)
            if device == "mps":
                self.model.to("mps")
            self.model.eval()
            self.model.set_ddpm_inference_steps(num_steps=self.settings.default_inference_steps)
            self.input_device = self._resolve_input_device(device)
            return time.monotonic() - started
        except ServiceError:
            raise
        except Exception as exc:
            raise classify_runtime_exception(exc, phase="load") from exc

    def synthesize(self, *, script: str, voice_paths: list[Path], output_path: Path, seed: int, cfg_scale: float, inference_steps: int, disable_prefill: bool, max_length_times: float, stop_check_fn: Callable[[], bool]) -> RuntimeSynthesisResult:
        if self.model is None or self.processor is None or self.torch is None:
            raise NotReadyError("model runtime is not loaded")
        if stop_check_fn():
            raise GenerationCancelledError("generation cancelled before start")
        torch = self.torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model.set_ddpm_inference_steps(num_steps=inference_steps)
        try:
            inputs = self.processor(text=[script], voice_samples=[[str(path) for path in voice_paths]], padding=True, return_tensors="pt", return_attention_mask=True)
            if stop_check_fn():
                raise GenerationCancelledError("generation cancelled during preprocessing")
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.input_device)
            started = time.monotonic()
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={"do_sample": False},
                verbose=False,
                show_progress_bar=False,
                is_prefill=not disable_prefill,
                max_length_times=max_length_times,
                stop_check_fn=stop_check_fn,
            )
            generation_seconds = time.monotonic() - started
            if stop_check_fn():
                raise GenerationCancelledError("generation cancelled")
            if not outputs.speech_outputs or outputs.speech_outputs[0] is None:
                raise GenerationError("model returned no speech output")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.processor.save_audio(outputs.speech_outputs[0], output_path=str(output_path))
            input_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None
            generated_tokens = None
            if getattr(outputs, "sequences", None) is not None and input_tokens is not None:
                generated_tokens = max(0, int(outputs.sequences.shape[1]) - input_tokens)
            return RuntimeSynthesisResult(generation_seconds=generation_seconds, input_tokens=input_tokens, generated_tokens=generated_tokens)
        except GenerationCancelledError:
            raise
        except ServiceError:
            raise
        except Exception as exc:
            raise classify_runtime_exception(exc, phase="generation") from exc

    def unload(self) -> None:
        self.model = None
        self.processor = None
        gc.collect()
        torch = self.torch
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        self.torch = None
        self.device_mode = None
        self.input_device = "cpu"


class ModelManager:
    def __init__(self, settings: Settings, *, runtime_factory: Callable[[Settings], VibeVoiceRuntime] = VibeVoiceRuntime):
        settings.validate_static_policy()
        self.settings = settings
        self._runtime_factory = runtime_factory
        self._runtime: VibeVoiceRuntime | None = None
        self._state = LifecycleState.UNLOADED
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._active_job_id: str | None = None
        self._failure_code: str | None = None
        self._failure_message: str | None = None
        self._last_activity = time.monotonic()
        self._load_seconds = 0.0
        self._shutdown_event = threading.Event()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.settings.reference_dir.is_dir():
            raise ReferencePathError(f"configured reference directory does not exist: {self.settings.reference_dir}")
        self._idle_thread: threading.Thread | None = None
        if self.settings.idle_unload_seconds > 0:
            self._idle_thread = threading.Thread(target=self._idle_loop, name="vibevoice-idle-unload", daemon=True)
            self._idle_thread.start()

    @property
    def state(self) -> LifecycleState:
        with self._state_lock:
            return self._state

    @property
    def device_mode(self) -> str | None:
        with self._state_lock:
            runtime = self._runtime
            return runtime.device_mode if runtime else None

    def ready_snapshot(self) -> ReadyResponse:
        with self._state_lock:
            runtime = self._runtime
            return ReadyResponse(state=self._state, ready=self._state == LifecycleState.READY, active_job_id=self._active_job_id, failure_code=self._failure_code, failure_message=self._failure_message, device_mode=runtime.device_mode if runtime else None)

    def _set_state(self, state: LifecycleState, *, failure: ServiceError | None = None) -> None:
        with self._state_lock:
            self._state = state
            if failure is None:
                self._failure_code = None
                self._failure_message = None
            else:
                self._failure_code = failure.code
                self._failure_message = failure.message

    def _validate_expected_revisions(self, *, model_revision: str | None, source_revision: str | None) -> None:
        if model_revision and model_revision.lower() != self.settings.model_revision.lower():
            raise RevisionMismatchError("requested model revision does not match running service")
        if source_revision and source_revision.lower() != self.settings.source_revision.lower():
            raise RevisionMismatchError("requested source revision does not match running service")

    def load(self, *, model_revision: str | None = None, source_revision: str | None = None) -> None:
        self._validate_expected_revisions(model_revision=model_revision, source_revision=source_revision)
        if not self._operation_lock.acquire(blocking=False):
            raise BusyError("another model operation is in progress")
        try:
            if self.state == LifecycleState.READY:
                self._last_activity = time.monotonic()
                return
            if self.state not in {LifecycleState.UNLOADED, LifecycleState.FAILED}:
                raise BusyError(f"cannot load while state is {self.state.value}")
            self._set_state(LifecycleState.LOADING)
            runtime = self._runtime_factory(self.settings)
            try:
                self._load_seconds = runtime.load()
            except ServiceError as exc:
                try:
                    runtime.unload()
                except Exception:
                    pass
                self._runtime = None
                self._set_state(LifecycleState.FAILED, failure=exc)
                raise
            except Exception as exc:
                mapped = classify_runtime_exception(exc, phase="load")
                try:
                    runtime.unload()
                except Exception:
                    pass
                self._runtime = None
                self._set_state(LifecycleState.FAILED, failure=mapped)
                raise mapped from exc
            with self._state_lock:
                self._runtime = runtime
            self._last_activity = time.monotonic()
            self._set_state(LifecycleState.READY)
        finally:
            self._operation_lock.release()

    def _stage_references(self, request: DialogueRequest, staging_dir: Path, stop_check_fn: Callable[[], bool]) -> tuple[list[Path], dict[str, str]]:
        paths: list[Path] = []
        hashes: dict[str, str] = {}
        for index, speaker in enumerate(request.speakers, start=1):
            if stop_check_fn():
                raise GenerationCancelledError("generation cancelled during reference staging")
            source = resolve_under_root(speaker.reference_path, self.settings.reference_dir, require_wav=True)
            try:
                size = source.stat().st_size
            except OSError as exc:
                raise ReferencePathError(f"cannot stat reference for speaker {speaker.speaker_id}: {exc}") from exc
            if size > self.settings.reference_max_bytes:
                raise ReferencePathError(f"reference for speaker {speaker.speaker_id} exceeds maximum size")
            staged = staging_dir / f"speaker-{index}.wav"
            try:
                shutil.copyfile(source, staged)
            except OSError as exc:
                raise ReferencePathError(f"cannot stage reference for speaker {speaker.speaker_id}: {exc}") from exc
            actual_hash = sha256_file(staged)
            if actual_hash != speaker.reference_sha256:
                raise ReferenceHashError(f"reference hash mismatch for speaker {speaker.speaker_id}")
            validate_reference_wav(staged, max_seconds=self.settings.reference_max_seconds)
            paths.append(staged)
            hashes[speaker.speaker_id] = actual_hash
        return paths, hashes

    def _reset_interrupted_runtime(self) -> None:
        runtime = self._runtime
        cleanup_failure: ServiceError | None = None
        try:
            if runtime is not None:
                runtime.unload()
        except Exception as exc:
            cleanup_failure = classify_runtime_exception(exc, phase="cleanup")
        finally:
            with self._state_lock:
                self._runtime = None
            self._load_seconds = 0.0
            self._last_activity = time.monotonic()
        if cleanup_failure is None:
            self._set_state(LifecycleState.UNLOADED)
        else:
            self._set_state(LifecycleState.FAILED, failure=cleanup_failure)

    def synthesize(self, request: DialogueRequest, *, external_cancel_event: threading.Event | None = None, timeout_seconds: float | None = None) -> DialogueResult:
        self._validate_expected_revisions(model_revision=request.model_revision, source_revision=request.source_revision)
        if not self._operation_lock.acquire(blocking=False):
            raise BusyError("another model operation is in progress")
        temp_path = self.settings.output_dir / f".{request.job_id}.{uuid.uuid4().hex}.tmp.wav"
        timeout = self.settings.generation_timeout_seconds if timeout_seconds is None else timeout_seconds
        deadline = time.monotonic() + timeout
        self._cancel_event.clear()

        def stop_check() -> bool:
            return self._cancel_event.is_set() or self._shutdown_event.is_set() or (external_cancel_event is not None and external_cancel_event.is_set()) or time.monotonic() >= deadline

        def interrupted_error() -> ServiceError:
            if time.monotonic() >= deadline:
                return GenerationTimeoutError("generation exceeded service timeout")
            return GenerationCancelledError("generation cancelled")

        try:
            with self._state_lock:
                runtime = self._runtime
                if self._state != LifecycleState.READY or runtime is None:
                    raise NotReadyError("model is not ready")
                self._active_job_id = request.job_id
            self._set_state(LifecycleState.GENERATING)
            output_path = self.settings.output_dir / f"{request.job_id}.wav"
            if output_path.exists():
                raise OutputExistsError(f"output already exists: {output_path}")
            if stop_check():
                raise interrupted_error()

            cfg_scale = request.generation.cfg_scale if request.generation.cfg_scale is not None else self.settings.default_cfg_scale
            steps = request.generation.inference_steps if request.generation.inference_steps is not None else self.settings.default_inference_steps
            with tempfile.TemporaryDirectory(prefix=f".{request.job_id}.refs-", dir=self.settings.output_dir, ignore_cleanup_errors=True) as staging:
                voice_paths, reference_hashes = self._stage_references(request, Path(staging), stop_check)
                script, _speaker_order = compile_dialogue(request)
                if stop_check():
                    raise interrupted_error()
                try:
                    result = runtime.synthesize(
                        script=script,
                        voice_paths=voice_paths,
                        output_path=temp_path,
                        seed=request.seed,
                        cfg_scale=cfg_scale,
                        inference_steps=steps,
                        disable_prefill=request.generation.disable_prefill,
                        max_length_times=request.generation.max_length_times,
                        stop_check_fn=stop_check,
                    )
                except GenerationCancelledError as exc:
                    raise interrupted_error() from exc

            if stop_check():
                raise interrupted_error()
            duration_seconds, sample_rate = validate_wav(temp_path)
            if sample_rate != 24_000:
                raise InvalidOutputError(f"unexpected sample rate {sample_rate}; expected 24000")
            if stop_check():
                raise interrupted_error()
            audio_hash = sha256_file(temp_path)
            response = DialogueResult(
                job_id=request.job_id,
                output_path=str(output_path),
                duration_seconds=duration_seconds,
                sample_rate_hz=sample_rate,
                source_repository=self.settings.source_repository,
                source_revision=self.settings.source_revision,
                model_repository=self.settings.model_repository,
                model_revision=self.settings.model_revision,
                device_mode=runtime.device_mode or "unknown",
                seed=request.seed,
                generation={
                    "cfg_scale": cfg_scale,
                    "inference_steps": steps,
                    "disable_prefill": request.generation.disable_prefill,
                    "max_length_times": request.generation.max_length_times,
                    "input_tokens": result.input_tokens,
                    "generated_tokens": result.generated_tokens,
                },
                speaker_reference_hashes=reference_hashes,
                audio_sha256=audio_hash,
                timings=TimingInfo(load_seconds=self._load_seconds, generation_seconds=result.generation_seconds),
            )
            try:
                os.link(temp_path, output_path)
            except FileExistsError as exc:
                raise OutputExistsError(f"output already exists: {output_path}") from exc
            try:
                temp_path.unlink()
            except OSError:
                pass
            self._last_activity = time.monotonic()
            self._set_state(LifecycleState.READY)
            return response

        except NotReadyError:
            raise
        except (GenerationCancelledError, GenerationTimeoutError):
            self._reset_interrupted_runtime()
            raise
        except (OutputExistsError, ReferenceHashError, ReferencePathError) as exc:
            self._last_activity = time.monotonic()
            self._set_state(LifecycleState.READY)
            raise exc
        except ServiceError as exc:
            runtime = self._runtime
            try:
                if runtime is not None:
                    runtime.unload()
            except Exception:
                pass
            with self._state_lock:
                self._runtime = None
            self._set_state(LifecycleState.FAILED, failure=exc)
            raise
        except Exception as exc:
            mapped = classify_runtime_exception(exc, phase="generation")
            runtime = self._runtime
            try:
                if runtime is not None:
                    runtime.unload()
            except Exception:
                pass
            with self._state_lock:
                self._runtime = None
            self._set_state(LifecycleState.FAILED, failure=mapped)
            raise mapped from exc
        finally:
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                pass
            with self._state_lock:
                self._active_job_id = None
            self._cancel_event.clear()
            self._operation_lock.release()

    def cancel(self, job_id: str) -> bool:
        with self._state_lock:
            if self._state == LifecycleState.GENERATING and self._active_job_id == job_id:
                self._cancel_event.set()
                return True
        return False

    def unload(self) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise BusyError("another model operation is in progress")
        try:
            if self.state == LifecycleState.UNLOADED:
                return
            if self.state == LifecycleState.GENERATING:
                raise BusyError("cannot unload while generation is active")
            self._set_state(LifecycleState.UNLOADING)
            runtime = self._runtime
            try:
                if runtime is not None:
                    runtime.unload()
                with self._state_lock:
                    self._runtime = None
                self._load_seconds = 0.0
                self._last_activity = time.monotonic()
                self._set_state(LifecycleState.UNLOADED)
            except ServiceError as exc:
                self._set_state(LifecycleState.FAILED, failure=exc)
                raise
            except Exception as exc:
                mapped = classify_runtime_exception(exc, phase="cleanup")
                self._set_state(LifecycleState.FAILED, failure=mapped)
                raise mapped from exc
        finally:
            self._operation_lock.release()

    def shutdown(self) -> None:
        self._shutdown_event.set()
        self._cancel_event.set()
        deadline = time.monotonic() + self.settings.cleanup_timeout_seconds
        while self.state == LifecycleState.GENERATING and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.state != LifecycleState.GENERATING:
            try:
                self.unload()
            except (BusyError, ServiceError):
                pass
        if self._idle_thread is not None and self._idle_thread.is_alive():
            self._idle_thread.join(timeout=1.0)

    def _idle_loop(self) -> None:
        interval = min(5.0, max(0.5, self.settings.idle_unload_seconds / 4.0))
        while not self._shutdown_event.wait(interval):
            if self.state != LifecycleState.READY:
                continue
            if time.monotonic() - self._last_activity < self.settings.idle_unload_seconds:
                continue
            try:
                self.unload()
            except (BusyError, ServiceError):
                pass
