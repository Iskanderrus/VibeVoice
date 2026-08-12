from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import RevisionMismatchError, SourcePolicyError

OWNED_SOURCE_REPOSITORY = "Iskanderrus/VibeVoice"
ALLOWED_MODEL_REPOSITORY = "vibevoice/VibeVoice-1.5B"
ALLOWED_TOKENIZER_REPOSITORY = "Qwen/Qwen2.5-1.5B"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None else float(raw)


def _is_exact_commit(value: str) -> bool:
    return bool(_COMMIT_RE.fullmatch(value.strip()))


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("/models/VibeVoice-1.5B")
    model_repository: str = ALLOWED_MODEL_REPOSITORY
    model_revision: str = ""
    tokenizer_repository: str = ALLOWED_TOKENIZER_REPOSITORY
    tokenizer_revision: str = ""
    source_repository: str = OWNED_SOURCE_REPOSITORY
    source_revision: str = ""
    source_revision_file: Path = Path("/opt/vibevoice/.stickman-source-revision")
    require_source_revision_file: bool = True
    model_manifest_path: Path = Path("/models/VibeVoice-1.5B/.stickman-model.json")
    require_model_manifest: bool = True

    device: str = "auto"
    dtype: str = "auto"
    cpu_offload: bool = True
    cuda_max_memory_mb: int = 3400
    cpu_max_memory_gb: int = 24
    offload_dir: Path = Path("/offload")

    idle_unload_seconds: int = 300
    generation_timeout_seconds: float = 1800.0
    cleanup_timeout_seconds: float = 30.0
    max_concurrent_jobs: int = 1

    output_dir: Path = Path("/shared/output")
    reference_dir: Path = Path("/shared/references")
    reference_max_bytes: int = 50 * 1024 * 1024
    reference_max_seconds: float = 120.0
    local_files_only: bool = True

    default_cfg_scale: float = 1.3
    default_inference_steps: int = 10

    host: str = "0.0.0.0"
    port: int = 8765

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(os.getenv("VIBEVOICE_MODEL_PATH", "/models/VibeVoice-1.5B"))
        return cls(
            model_path=model_path,
            model_repository=os.getenv(
                "VIBEVOICE_MODEL_REPOSITORY", ALLOWED_MODEL_REPOSITORY
            ),
            model_revision=os.getenv("VIBEVOICE_MODEL_REVISION", ""),
            tokenizer_repository=os.getenv(
                "VIBEVOICE_TOKENIZER_REPOSITORY", ALLOWED_TOKENIZER_REPOSITORY
            ),
            tokenizer_revision=os.getenv("VIBEVOICE_TOKENIZER_REVISION", ""),
            source_repository=os.getenv(
                "VIBEVOICE_SOURCE_REPOSITORY", OWNED_SOURCE_REPOSITORY
            ),
            source_revision=os.getenv("VIBEVOICE_SOURCE_REVISION", ""),
            # Production source identity is deliberately not environment-overridable.
            source_revision_file=Path("/opt/vibevoice/.stickman-source-revision"),
            require_source_revision_file=True,
            model_manifest_path=Path(
                os.getenv(
                    "VIBEVOICE_MODEL_MANIFEST",
                    str(model_path / ".stickman-model.json"),
                )
            ),
            require_model_manifest=_env_bool(
                "VIBEVOICE_REQUIRE_MODEL_MANIFEST", True
            ),
            device=os.getenv("VIBEVOICE_DEVICE", "auto").strip().lower(),
            dtype=os.getenv("VIBEVOICE_DTYPE", "auto").strip().lower(),
            cpu_offload=_env_bool("VIBEVOICE_CPU_OFFLOAD", True),
            cuda_max_memory_mb=_env_int("VIBEVOICE_CUDA_MAX_MEMORY_MB", 3400),
            cpu_max_memory_gb=_env_int("VIBEVOICE_CPU_MAX_MEMORY_GB", 24),
            offload_dir=Path(os.getenv("VIBEVOICE_OFFLOAD_DIR", "/offload")),
            idle_unload_seconds=_env_int("VIBEVOICE_IDLE_UNLOAD_SECONDS", 300),
            generation_timeout_seconds=_env_float(
                "VIBEVOICE_GENERATION_TIMEOUT_SECONDS", 1800.0
            ),
            cleanup_timeout_seconds=_env_float(
                "VIBEVOICE_CLEANUP_TIMEOUT_SECONDS", 30.0
            ),
            max_concurrent_jobs=_env_int("VIBEVOICE_MAX_CONCURRENT_JOBS", 1),
            output_dir=Path(os.getenv("VIBEVOICE_OUTPUT_DIR", "/shared/output")),
            reference_dir=Path(
                os.getenv("VIBEVOICE_REFERENCE_DIR", "/shared/references")
            ),
            reference_max_bytes=_env_int(
                "VIBEVOICE_REFERENCE_MAX_BYTES", 50 * 1024 * 1024
            ),
            reference_max_seconds=_env_float(
                "VIBEVOICE_REFERENCE_MAX_SECONDS", 120.0
            ),
            local_files_only=_env_bool("VIBEVOICE_LOCAL_FILES_ONLY", True),
            default_cfg_scale=_env_float("VIBEVOICE_DEFAULT_CFG_SCALE", 1.3),
            default_inference_steps=_env_int(
                "VIBEVOICE_DEFAULT_INFERENCE_STEPS", 10
            ),
            host=os.getenv("VIBEVOICE_HOST", "0.0.0.0"),
            port=_env_int("VIBEVOICE_PORT", 8765),
        )

    def validate_static_policy(self) -> None:
        if self.source_repository != OWNED_SOURCE_REPOSITORY:
            raise SourcePolicyError(
                f"runtime source must be {OWNED_SOURCE_REPOSITORY!r}"
            )
        if self.model_repository != ALLOWED_MODEL_REPOSITORY:
            raise SourcePolicyError(
                f"only model repository {ALLOWED_MODEL_REPOSITORY!r} is approved"
            )
        if self.tokenizer_repository != ALLOWED_TOKENIZER_REPOSITORY:
            raise SourcePolicyError(
                f"only tokenizer repository {ALLOWED_TOKENIZER_REPOSITORY!r} is approved"
            )
        if not self.local_files_only:
            raise SourcePolicyError(
                "VIBEVOICE_LOCAL_FILES_ONLY must remain true for the governed service"
            )
        if not self.require_model_manifest:
            raise SourcePolicyError(
                "VIBEVOICE_REQUIRE_MODEL_MANIFEST must remain true for the governed service"
            )
        if not _is_exact_commit(self.source_revision):
            raise SourcePolicyError(
                "VIBEVOICE_SOURCE_REVISION must be an exact 40-character commit SHA"
            )
        if not _is_exact_commit(self.model_revision):
            raise SourcePolicyError(
                "VIBEVOICE_MODEL_REVISION must be an exact 40-character commit SHA"
            )
        # Production startup additionally requires a non-empty tokenizer
        # revision through verify_stickman_runtime.sh. Validate its shape here
        # whenever supplied so direct/test Settings remain lightweight.
        if self.tokenizer_revision and not _is_exact_commit(self.tokenizer_revision):
            raise SourcePolicyError(
                "VIBEVOICE_TOKENIZER_REVISION must be an exact 40-character commit SHA"
            )
        if self.max_concurrent_jobs != 1:
            raise SourcePolicyError("VIBEVOICE_MAX_CONCURRENT_JOBS must be exactly 1")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise SourcePolicyError("VIBEVOICE_DEVICE must be auto, cuda, mps or cpu")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise SourcePolicyError(
                "VIBEVOICE_DTYPE must be auto, bfloat16, float16 or float32"
            )
        if self.cuda_max_memory_mb < 512:
            raise SourcePolicyError("VIBEVOICE_CUDA_MAX_MEMORY_MB must be at least 512")
        if self.cpu_max_memory_gb < 4:
            raise SourcePolicyError("VIBEVOICE_CPU_MAX_MEMORY_GB must be at least 4")
        if self.idle_unload_seconds < 0:
            raise SourcePolicyError("VIBEVOICE_IDLE_UNLOAD_SECONDS cannot be negative")
        if self.generation_timeout_seconds <= 0:
            raise SourcePolicyError(
                "VIBEVOICE_GENERATION_TIMEOUT_SECONDS must be positive"
            )
        if self.cleanup_timeout_seconds <= 0:
            raise SourcePolicyError("VIBEVOICE_CLEANUP_TIMEOUT_SECONDS must be positive")
        if self.reference_max_bytes <= 0:
            raise SourcePolicyError("VIBEVOICE_REFERENCE_MAX_BYTES must be positive")
        if self.reference_max_seconds <= 0:
            raise SourcePolicyError("VIBEVOICE_REFERENCE_MAX_SECONDS must be positive")
        if not 0.1 <= self.default_cfg_scale <= 5.0:
            raise SourcePolicyError(
                "VIBEVOICE_DEFAULT_CFG_SCALE must be between 0.1 and 5.0"
            )
        if not 1 <= self.default_inference_steps <= 100:
            raise SourcePolicyError(
                "VIBEVOICE_DEFAULT_INFERENCE_STEPS must be between 1 and 100"
            )
        if not 1 <= self.port <= 65535:
            raise SourcePolicyError("VIBEVOICE_PORT must be between 1 and 65535")

        expected_manifest = self.model_path / ".stickman-model.json"
        if self.model_manifest_path.resolve(strict=False) != expected_manifest.resolve(
            strict=False
        ):
            raise SourcePolicyError(
                "VIBEVOICE_MODEL_MANIFEST must be the .stickman-model.json inside VIBEVOICE_MODEL_PATH"
            )

        if self.require_source_revision_file:
            if not self.source_revision_file.is_file():
                raise SourcePolicyError(
                    f"baked source revision file is missing: {self.source_revision_file}"
                )
            try:
                baked = self.source_revision_file.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise SourcePolicyError(
                    f"cannot read baked source revision file: {exc}"
                ) from exc
            if not _is_exact_commit(baked):
                raise SourcePolicyError("baked source revision is not an exact commit SHA")
            if baked.lower() != self.source_revision.lower():
                raise RevisionMismatchError(
                    "configured source revision does not match image source revision"
                )
