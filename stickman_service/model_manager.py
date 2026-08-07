from __future__ import annotations

import gc
import json
import os
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
    InvalidOutputError,
    ModelLoadError,
    NotReadyError,
    OutputExistsError,
    ReferenceHashError,
    RevisionMismatchError,
    ServiceError,
    classify_runtime_exception,
)
from .hashing import resolve_under_root, sha256_file
from .schemas import (
    DialogueRequest,
    DialogueResult,
    LifecycleState,
    ReadyResponse,
    TimingInfo,
)
from .settings import Settings
from .synthesis import compile_dialogue, validate_wav


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
        if self.settings.local_files_only and not self.settings.model_path.exists():
            raise ModelLoadError(
                f"local model path does not exist: {self.settings.model_path}"
            )

        manifest_path = self.settings.model_manifest_path
        if not manifest_path.is_file():
            if self.settings.require_model_manifest:
                raise ModelLoadError(
                    f"required model manifest is missing: {manifest_path}"
                )
            return

        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelLoadError(f"cannot read model manifest: {exc}") from exc

        if data.get("repository") != self.settings.model_repository:
            raise RevisionMismatchError(
                "model manifest repository does not match configured repository"
            )
        if data.get("revision") != self.settings.model_revision:
            raise RevisionMismatchError(
                "model manifest revision does not match configured revision"
            )

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

    def _resolve_dtype(self, torch: Any, device: str) -> Any:
        configured = self.settings.dtype
        if configured == "auto":
            return torch.bfloat16 if device == "cuda" else torch.float32
        return {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[configured]

    def load(self) -> float:
        self._verify_model_manifest()
        started = time.monotonic()
        try:
            import torch
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            self.torch = torch
            device = self._resolve_device(torch)
            dtype = self._resolve_dtype(torch, device)
            source: str = (
                str(self.settings.model_path)
                if self.settings.model_path.exists()
                else self.settings.model_repository
            )

            processor_kwargs: dict[str, Any] = {}
            model_kwargs: dict[str, Any] = {"torch_dtype": dtype}
            using_local_path = self.settings.model_path.exists()
            if self.settings.local_files_only:
                processor_kwargs["local_files_only"] = True
                model_kwargs["local_files_only"] = True
            elif not using_local_path:
                processor_kwargs["revision"] = self.settings.model_revision
                model_kwargs["revision"] = self.settings.model_revision

            attn_primary = "flash_attention_2" if device == "cuda" else "sdpa"
            model_kwargs["attn_implementation"] = attn_primary

            if device == "cuda" and self.settings.cpu_offload:
                self.settings.offload_dir.mkdir(parents=True, exist_ok=True)
                model_kwargs.update(
                    {
                        "device_map": "auto",
                        "max_memory": {
                            0: f"{self.settings.cuda_max_memory_mb}MiB",
                            "cpu": f"{self.settings.cpu_max_memory_gb}GiB",
                        },
                        "offload_folder": str(self.settings.offload_dir),
                        "offload_state_dict": True,
                    }
                )
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

            self.processor = VibeVoiceProcessor.from_pretrained(
                source, **processor_kwargs
            )
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    source, **model_kwargs
                )
            except Exception:
                if attn_primary != "flash_attention_2":
                    raise
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                model_kwargs["attn_implementation"] = "sdpa"
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    source, **model_kwargs
                )

            if device == "mps":
                self.model.to("mps")

            self.model.eval()
            self.model.set_ddpm_inference_steps(
                num_steps=self.settings.default_inference_steps
            )
            try:
                self.input_device = str(self.model.device)
            except Exception:
                self.input_device = device
            return time.monotonic() - started
        except ServiceError:
            raise
        except Exception as exc:
            raise classify_runtime_exception(exc, phase="load") from exc

    def synthesize(
        self,
        *,
        script: str,
        voice_paths: list[Path],
        output_path: Path,
        seed: int,
        cfg_scale: float,
        inference_steps: int,
        disable_prefill: bool,
        max_length_times: float,
        stop_check_fn: Callable[[], bool],
    ) -> RuntimeSynthesisResult:
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
            inputs = self.processor(
                text=[script],
                voice_samples=[[str(path) for path in voice_paths]],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            target_device = self.input_device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(target_device)

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
            self.processor.save_audio(
                outputs.speech_outputs[0],
                output_path=str(output_path),
            )
            input_tokens = None
            generated_tokens = None
            if "input_ids" in inputs:
                input_tokens = int(inputs["input_ids"].shape[1])
            if getattr(outputs, "sequences", None) is not None and input_tokens is not None:
                total = int(outputs.sequences.shape[1])
                generated_tokens = max(0, total - input_tokens)
            return RuntimeSynthesisResult(
                generation_seconds=generation_seconds,
                input_tokens=input_tokens,
                generated_tokens=generated_tokens,
            )
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
    def __init__(
        self,
        settings: Settings,
        *,
        runtime_factory: Callable[[Settings], VibeVoiceRuntime] = VibeVoiceRuntime,
    ):
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
        self.settings.reference_dir.mkdir(parents=True, exist_ok=True)

        self._idle_thread: threading.Thread | None = None
        if self.settings.idle_unload_seconds > 0:
            self._idle_thread = threading.Thread(
                target=self._idle_loop,
                name="vibevoice-idle-unload",
                daemon=True,
            )
            self._idle_thread.start()

    @property
    def state(self) -> LifecycleState:
        with self._state_lock:
            return self._state

    @property
    def device_mode(self) -> str | None:
        runtime = self._runtime
        return runtime.device_mode if runtime else None

    def ready_snapshot(self) -> ReadyResponse:
        with self._state_lock:
            return ReadyResponse(
                state=self._state,
                ready=self._state == LifecycleState.READY,
                active_job_id=self._active_job_id,
                failure_code=self._failure_code,
                failure_message=self._failure_message,
                device_mode=self.device_mode,
            )

    def _set_state(
        self,
        state: LifecycleState,
        *,
        failure: ServiceError | None = None,
    ) -> None:
        with self._state_lock:
            self._state = state
            if failure is None:
                self._failure_code = None
                self._failure_message = None
            else:
                self._failure_code = failure.code
                self._failure_message = failure.message

    def _validate_expected_revisions(
        self,
        *,
        model_revision: str | None,
        source_revision: str | None,
    ) -> None:
        if model_revision and model_revision != self.settings.model_revision:
            raise RevisionMismatchError(
                "requested model revision does not match running service"
            )
        if source_revision and source_revision != self.settings.source_revision:
            raise RevisionMismatchError(
                "requested source revision does not match running service"
            )

    def load(
        self,
        *,
        model_revision: str | None = None,
        source_revision: str | None = None,
    ) -> None:
        self._validate_expected_revisions(
            model_revision=model_revision,
            source_revision=source_revision,
        )
        if not self._operation_lock.acquire(blocking=False):
            raise BusyError("another model operation is in progress")
        try:
            if self.state == LifecycleState.READY:
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

            self._runtime = runtime
            self._last_activity = time.monotonic()
            self._set_state(LifecycleState.READY)
        finally:
            self._operation_lock.release()

    def _validate_references(
        self, request: DialogueRequest
    ) -> tuple[list[Path], dict[str, str]]:
        paths: list[Path] = []
        hashes: dict[str, str] = {}
        for speaker in request.speakers:
            path = resolve_under_root(
                speaker.reference_path,
                self.settings.reference_dir,
                require_wav=True,
            )
            actual_hash = sha256_file(path)
            if actual_hash != speaker.reference_sha256:
                raise ReferenceHashError(
                    f"reference hash mismatch for speaker {speaker.speaker_id}"
                )
            paths.append(path)
            hashes[speaker.speaker_id] = actual_hash
        return paths, hashes

    def synthesize(self, request: DialogueRequest) -> DialogueResult:
        self._validate_expected_revisions(
            model_revision=request.model_revision,
            source_revision=request.source_revision,
        )
        if self.state != LifecycleState.READY or self._runtime is None:
            raise NotReadyError("model is not ready")

        output_path = self.settings.output_dir / f"{request.job_id}.wav"
        if output_path.exists():
            raise OutputExistsError(f"output already exists: {output_path}")

        voice_paths, reference_hashes = self._validate_references(request)
        script, _speaker_order = compile_dialogue(request)

        if not self._operation_lock.acquire(blocking=False):
            raise BusyError("another model operation is in progress")

        temp_path = self.settings.output_dir / (
            f".{request.job_id}.{uuid.uuid4().hex}.tmp.wav"
        )
        runtime = self._runtime
        assert runtime is not None
        self._cancel_event.clear()
        with self._state_lock:
            self._active_job_id = request.job_id
        self._set_state(LifecycleState.GENERATING)

        try:
            cfg_scale = (
                request.generation.cfg_scale
                if request.generation.cfg_scale is not None
                else self.settings.default_cfg_scale
            )
            steps = (
                request.generation.inference_steps
                if request.generation.inference_steps is not None
                else self.settings.default_inference_steps
            )
            result = runtime.synthesize(
                script=script,
                voice_paths=voice_paths,
                output_path=temp_path,
                seed=request.seed,
                cfg_scale=cfg_scale,
                inference_steps=steps,
                disable_prefill=request.generation.disable_prefill,
                max_length_times=request.generation.max_length_times,
                stop_check_fn=self._cancel_event.is_set,
            )

            if self._cancel_event.is_set():
                raise GenerationCancelledError("generation cancelled")

            duration_seconds, sample_rate = validate_wav(temp_path)
            if sample_rate != 24_000:
                raise InvalidOutputError(
                    f"unexpected sample rate {sample_rate}; expected 24000"
                )
            os.replace(temp_path, output_path)

            generation_meta: dict[str, Any] = {
                "cfg_scale": cfg_scale,
                "inference_steps": steps,
                "disable_prefill": request.generation.disable_prefill,
                "max_length_times": request.generation.max_length_times,
                "input_tokens": result.input_tokens,
                "generated_tokens": result.generated_tokens,
            }
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
                generation=generation_meta,
                speaker_reference_hashes=reference_hashes,
                audio_sha256=sha256_file(output_path),
                timings=TimingInfo(
                    load_seconds=self._load_seconds,
                    generation_seconds=result.generation_seconds,
                ),
            )
            self._last_activity = time.monotonic()
            self._set_state(LifecycleState.READY)
            return response

        except GenerationCancelledError:
            self._set_state(LifecycleState.READY)
            raise
        except ServiceError as exc:
            try:
                runtime.unload()
            except Exception:
                pass
            self._runtime = None
            self._set_state(LifecycleState.FAILED, failure=exc)
            raise
        except Exception as exc:
            mapped = classify_runtime_exception(exc, phase="generation")
            try:
                runtime.unload()
            except Exception:
                pass
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
            if (
                self._state == LifecycleState.GENERATING
                and self._active_job_id == job_id
            ):
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
                self._runtime = None
                self._load_seconds = 0.0
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
            except BusyError:
                pass
            except ServiceError:
                pass

    def _idle_loop(self) -> None:
        interval = min(5.0, max(0.5, self.settings.idle_unload_seconds / 4.0))
        while not self._shutdown_event.wait(interval):
            if (
                self.state == LifecycleState.READY
                and time.monotonic() - self._last_activity
                >= self.settings.idle_unload_seconds
            ):
                try:
                    self.unload()
                except (BusyError, ServiceError):
                    pass
