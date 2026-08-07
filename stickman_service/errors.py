from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    code = "SERVICE_ERROR"
    status_code = 500

    def __init__(self, message: str, *, details: Any = None):
        super().__init__(message)
        self.message = message
        self.details = details


class InvalidRequestError(ServiceError):
    code = "INVALID_REQUEST"
    status_code = 422


class BusyError(ServiceError):
    code = "BUSY"
    status_code = 409


class NotReadyError(ServiceError):
    code = "NOT_READY"
    status_code = 409


class RevisionMismatchError(ServiceError):
    code = "REVISION_MISMATCH"
    status_code = 409


class SourcePolicyError(ServiceError):
    code = "SOURCE_POLICY_VIOLATION"
    status_code = 409


class ReferencePathError(ServiceError):
    code = "REFERENCE_PATH_INVALID"
    status_code = 422


class ReferenceHashError(ServiceError):
    code = "REFERENCE_HASH_MISMATCH"
    status_code = 422


class OutputExistsError(ServiceError):
    code = "OUTPUT_EXISTS"
    status_code = 409


class InvalidOutputError(ServiceError):
    code = "INVALID_OUTPUT"
    status_code = 500


class ModelLoadError(ServiceError):
    code = "MODEL_LOAD_FAILED"
    status_code = 503


class GenerationError(ServiceError):
    code = "GENERATION_FAILED"
    status_code = 500


class GenerationTimeoutError(ServiceError):
    code = "GENERATION_TIMEOUT"
    status_code = 504


class GenerationCancelledError(ServiceError):
    code = "GENERATION_CANCELLED"
    status_code = 499


class OutOfMemoryError(ServiceError):
    code = "OUT_OF_MEMORY"
    status_code = 507


class CleanupError(ServiceError):
    code = "CLEANUP_FAILED"
    status_code = 500


def classify_runtime_exception(exc: Exception, *, phase: str) -> ServiceError:
    message = str(exc)
    lowered = message.lower()
    if "out of memory" in lowered or "cuda error: out of memory" in lowered:
        return OutOfMemoryError(f"{phase} failed: out of memory")
    if phase == "load":
        return ModelLoadError(f"model load failed: {message}")
    if phase == "generation":
        return GenerationError(f"generation failed: {message}")
    if phase == "cleanup":
        return CleanupError(f"cleanup failed: {message}")
    return ServiceError(f"{phase} failed: {message}")
