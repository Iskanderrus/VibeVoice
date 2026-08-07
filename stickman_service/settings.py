from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from .errors import RevisionMismatchError, SourcePolicyError

OWNED_SOURCE_REPOSITORY = "Iskanderrus/VibeVoice"
ALLOWED_MODEL_REPOSITORY = "vibevoice/VibeVoice-1.5B"
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


@dataclass(frozen=True)
class Settings:
    model_path: Path = Path("/models/VibeVoice-1.5B")
    model_repository: str = ALLOWED_MODEL_REPOSITORY
    model_revision: str = ""
    source_repository: str = OWNED_SOURCE_REPOSITORY
    source_revision: str = ""
    source_revision_file: Path = Path("/opt/vibevoice/.stickman-source-revision")
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
    local_files_only: bool = True

    default_cfg_scale: float = 1.3
    default_inference_steps: int = 10

    host: str = "0.0.0.0"
    port: int = 8765

    allow_unpinned: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        model_path = Path(os.getenv("VIBEVOICE_MODEL_PATH", "/models/VibeVoice-1.5B"))
        return cls(
            model_path=model_path,
            model_repository=os.getenv(
                "VIBEVOICE_MODEL_REPOSITORY", ALLOWED_MODEL_REPOSITORY
            ),
            model_revision=os.getenv("VIBEVOICE_MODEL_REVISION", ""),
            source_repository=os.getenv(
                "VIBEVOICE_SOURCE_REPOSITORY", OWNED_SOURCE_REPOSITORY
            ),
            source_revision=os.getenv("VIBEVOICE_SOURCE_REVISION", ""),
            source_revision_file=Path(
                os.getenv(
                    "VIBEVOICE_SOURCE_REVISION_FILE",
                    "/opt/vibevoice/.stickman-source-revision",
                )
            ),
            model_manifest_path=Path(
                os.getenv(
                    "VIBEVOICE_MODEL_MANIFEST",
                    str(model_path / ".stickman-model.json"),
                )
            ),
            require_model_manifest=_env_bool(
                "VIBEVOICE_REQUIRE_MODEL_MANIFEST", True
            ),
            device=os.getenv("VIBEVOICE_DEVICE", "auto"),
            dtype=os.getenv("VIBEVOICE_DTYPE", "auto"),
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
            output_dir=Path(
                os.getenv("VIBEVOICE_OUTPUT_DIR", "/shared/output")
            ),
            reference_dir=Path(
                os.getenv("VIBEVOICE_REFERENCE_DIR", "/shared/references")
            ),
            local_files_only=_env_bool("VIBEVOICE_LOCAL_FILES_ONLY", True),
            default_cfg_scale=_env_float("VIBEVOICE_DEFAULT_CFG_SCALE", 1.3),
            default_inference_steps=_env_int(
                "VIBEVOICE_DEFAULT_INFERENCE_STEPS", 10
            ),
            host=os.getenv("VIBEVOICE_HOST", "0.0.0.0"),
            port=_env_int("VIBEVOICE_PORT", 8765),
            allow_unpinned=_env_bool("VIBEVOICE_ALLOW_UNPINNED", False),
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
        if self.max_concurrent_jobs != 1:
            raise SourcePolicyError("VIBEVOICE_MAX_CONCURRENT_JOBS must be exactly 1")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise SourcePolicyError("VIBEVOICE_DEVICE must be auto, cuda, mps or cpu")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise SourcePolicyError(
                "VIBEVOICE_DTYPE must be auto, bfloat16, float16 or float32"
            )

        if self.source_revision_file.is_file():
            baked = self.source_revision_file.read_text(encoding="utf-8").strip()
            if baked and self.source_revision and baked != self.source_revision:
                raise RevisionMismatchError(
                    "configured source revision does not match image source revision"
                )

        if not self.allow_unpinned:
            if not _COMMIT_RE.fullmatch(self.source_revision):
                raise SourcePolicyError(
                    "VIBEVOICE_SOURCE_REVISION must be an exact 40-character commit SHA"
                )
            if not self.model_revision.strip():
                raise SourcePolicyError(
                    "VIBEVOICE_MODEL_REVISION must be pinned explicitly"
                )
