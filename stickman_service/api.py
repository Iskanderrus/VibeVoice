from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .errors import (
    GenerationCancelledError,
    GenerationTimeoutError,
    InvalidRequestError,
    ServiceError,
)
from .model_manager import ModelManager
from .schemas import (
    ActionResponse,
    CancelResponse,
    CapabilitiesResponse,
    DialogueRequest,
    DialogueResult,
    ErrorBody,
    HealthResponse,
    LoadRequest,
    ReadyResponse,
)
from .settings import Settings


def _validation_details(exc: RequestValidationError) -> list[dict[str, Any]]:
    return [
        {
            "loc": list(item.get("loc", ())),
            "msg": str(item.get("msg", "validation error")),
            "type": str(item.get("type", "value_error")),
        }
        for item in exc.errors()
    ]


def create_app(
    settings: Settings | None = None,
    manager: ModelManager | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.validate_static_policy()
    owns_manager = manager is None
    manager = manager or ModelManager(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            if owns_manager:
                await asyncio.to_thread(manager.shutdown)

    app = FastAPI(
        title="Stickman VibeVoice Service",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def service_error_handler(
        _request: Request, exc: ServiceError
    ) -> JSONResponse:
        body = ErrorBody(
            error=exc.code,
            message=exc.message,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        wrapped = InvalidRequestError(
            "request validation failed",
            details=_validation_details(exc),
        )
        body = ErrorBody(
            error=wrapped.code,
            message=wrapped.message,
            details=wrapped.details,
        )
        return JSONResponse(
            status_code=wrapped.status_code,
            content=body.model_dump(mode="json"),
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    async def ready() -> ReadyResponse:
        return manager.ready_snapshot()

    @app.get("/capabilities", response_model=CapabilitiesResponse)
    async def capabilities() -> CapabilitiesResponse:
        return CapabilitiesResponse(
            source_repository=settings.source_repository,
            source_revision=settings.source_revision,
            model_repository=settings.model_repository,
            model_revision=settings.model_revision,
            max_concurrent_jobs=settings.max_concurrent_jobs,
            device_mode=manager.device_mode,
        )

    @app.post("/load", response_model=ActionResponse)
    async def load(body: LoadRequest) -> ActionResponse:
        await asyncio.to_thread(
            manager.load,
            model_revision=body.model_revision,
            source_revision=body.source_revision,
        )
        return ActionResponse(state=manager.state, message="model ready")

    @app.post("/synthesize-dialogue", response_model=DialogueResult)
    async def synthesize(
        body: DialogueRequest, request: Request
    ) -> DialogueResult:
        request_cancel = threading.Event()
        task = asyncio.create_task(
            asyncio.to_thread(
                manager.synthesize,
                body,
                external_cancel_event=request_cancel,
                timeout_seconds=settings.generation_timeout_seconds,
            )
        )
        deadline = asyncio.get_running_loop().time() + settings.generation_timeout_seconds
        try:
            while not task.done():
                if await request.is_disconnected():
                    request_cancel.set()
                    manager.cancel(body.job_id)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=settings.cleanup_timeout_seconds,
                        )
                    except (
                        asyncio.TimeoutError,
                        GenerationCancelledError,
                        GenerationTimeoutError,
                        ServiceError,
                    ):
                        pass
                    raise GenerationCancelledError("client disconnected")
                if asyncio.get_running_loop().time() >= deadline:
                    request_cancel.set()
                    manager.cancel(body.job_id)
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(task),
                            timeout=settings.cleanup_timeout_seconds,
                        )
                    except (
                        asyncio.TimeoutError,
                        GenerationCancelledError,
                        GenerationTimeoutError,
                        ServiceError,
                    ):
                        pass
                    raise GenerationTimeoutError("generation exceeded service timeout")
                await asyncio.sleep(0.1)
            return await task
        except asyncio.CancelledError:
            request_cancel.set()
            manager.cancel(body.job_id)
            raise

    @app.post("/cancel/{job_id}", response_model=CancelResponse)
    async def cancel(job_id: str) -> CancelResponse:
        return CancelResponse(
            job_id=job_id,
            cancellation_requested=manager.cancel(job_id),
        )

    @app.post("/unload", response_model=ActionResponse)
    async def unload() -> ActionResponse:
        await asyncio.to_thread(manager.unload)
        return ActionResponse(state=manager.state, message="model unloaded")

    return app


def app_factory() -> FastAPI:
    return create_app()
