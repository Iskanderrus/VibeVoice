from __future__ import annotations

import uvicorn

from .settings import Settings


def main() -> None:
    settings = Settings.from_env()
    settings.validate_static_policy()
    uvicorn.run(
        "stickman_service.api:app_factory",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    main()
