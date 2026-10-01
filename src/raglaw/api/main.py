import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from raglaw.api.schemas import (
    AskRequest,
    ComponentStatus,
    ErrorResponse,
    FieldError,
    HealthResponse,
    UnavailableResponse,
    ValidationErrorResponse,
)
from raglaw.config import Settings
from raglaw.logging_conf import (
    configure_server_logging,
    correlation_id_var,
    get_logger,
)
from raglaw.schema import LogEvent

configure_server_logging()
logger = get_logger(__name__)

CORRELATION_ID_HEADER = "X-Request-ID"
SERVICE_VERSION = version("egypt-law-rag")

# Responses raised outside FastAPI's view (the exception handlers, HTTPException,
# the middleware), declared so /docs shows what clients actually receive.
UNAVAILABLE_RESPONSE: dict[int | str, dict[str, Any]] = {
    503: {
        "model": UnavailableResponse,
        "description": "A component /ask needs is not ready; `reasons` names each one.",
    }
}
INVALID_REQUEST_RESPONSE: dict[int | str, dict[str, Any]] = {
    422: {
        "model": ValidationErrorResponse,
        "description": "The request was rejected; `errors` lists every problem.",
    }
}
NOT_IMPLEMENTED_RESPONSE: dict[int | str, dict[str, Any]] = {
    501: {"model": ErrorResponse, "description": "Answering is not built yet."}
}


class ServiceUnavailableError(Exception):
    """Raised when a request needs a component that is not ready."""

    def __init__(self, reasons: dict[str, str]) -> None:
        super().__init__(", ".join(f"{k}: {v}" for k, v in reasons.items()))
        self.reasons = reasons


def load_components(settings: Settings) -> dict[str, ComponentStatus]:
    """
    Check or load every dependency /ask needs, once, at startup.

    A component that fails is recorded with its reason instead of stopping the
    process, so /health can report it and /ask can refuse with that reason.
    Loading the retrieval index and the embedding model belongs here from
    Phase 2 on; until then the retriever is reported as not loaded.

    returns:
    - components (dict[str, ComponentStatus]): readiness and detail per
      component name
    """
    key = settings.llm_api_key
    if key is not None and key.get_secret_value().strip():
        llm = ComponentStatus(ready=True, detail="API key is configured.")
    else:
        llm = ComponentStatus(ready=False, detail="RAGLAW_LLM_API_KEY is not set.")
    retriever = ComponentStatus(
        ready=False, detail="The retrieval index is not loaded."
    )
    return {"llm": llm, "retriever": retriever}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the service's components when the process starts, and log shutdown.

    Loading here rather than per request keeps /ask latency down to retrieval
    and the LLM call. Components that aren't ready are logged at WARNING, so a
    missing key shows up at boot and not only on the first request.
    """
    logger.info("Starting service", extra={"event_type": LogEvent.STARTUP})
    app.state.components = load_components(app.state.settings)
    for name, component in app.state.components.items():
        if not component.ready:
            logger.warning(
                "Component not ready: %s",
                component.detail,
                extra={"event_type": LogEvent.STARTUP, "component": name},
            )
    try:
        yield
    finally:
        logger.info("Stopping service", extra={"event_type": LogEvent.SHUTDOWN})


async def correlation_id_middleware(request: Request, call_next):
    """
    Give every request a correlation ID, reusing the caller's if they sent one,
    and return it on the response.

    The ID is held in a context variable, so every log line emitted while
    handling the request carries it and one request can be traced end to end.

    returns:
    - response (Response): the handler's response, with the correlation ID
      added as a header

    exceptions:
    - none: an unhandled error is logged and converted to a 500 rather than
      propagating to the client
    """
    correlation_id = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
    token = correlation_id_var.set(correlation_id)
    start = time.perf_counter()
    request_fields = {"path": request.url.path, "method": request.method}
    try:
        logger.info(
            "Request received",
            extra={**request_fields, "event_type": LogEvent.REQUEST_START},
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "Unhandled server error",
                extra={**request_fields, "event_type": LogEvent.REQUEST_FAILED},
            )
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    detail="An unexpected internal server error occurred."
                ).model_dump(),
            )

        response.headers[CORRELATION_ID_HEADER] = correlation_id
        logger.info(
            "Request completed",
            extra={
                **request_fields,
                "status_code": response.status_code,
                "latency_ms": (time.perf_counter() - start) * 1000,
                "event_type": LogEvent.REQUEST_COMPLETED,
            },
        )
        return response
    finally:
        correlation_id_var.reset(token)


def _to_field_error(error: dict[str, Any]) -> FieldError:
    """
    Turn one pydantic validation error into a structured entry naming the field
    that caused it.

    returns:
    - error (FieldError): the offending field path and a readable message, with
      no field name when the body was not valid JSON at all
    """
    if error["type"] == "json_invalid":
        position = error["loc"][-1]
        reason = error.get("ctx", {}).get("error", error["msg"])
        return FieldError(
            field=None,
            message=f"Request body is not valid JSON (character {position}: {reason})",
        )
    location = list(error["loc"])
    if location[:1] == ["body"]:
        location = location[1:]
    return FieldError(
        field=".".join(str(part) for part in location) or None,
        message=error["msg"],
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Convert a rejected request body into a 422 that lists every problem at once.

    returns:
    - response (JSONResponse): status 422, with one entry per invalid field so
      the caller can fix them all in one attempt
    """
    body = ValidationErrorResponse(
        detail="Invalid request payload",
        errors=[_to_field_error(error) for error in exc.errors()],
    ).model_dump()
    logger.warning(
        "Request validation failed",
        extra={
            "event_type": LogEvent.VALIDATION_FAILED,
            "path": request.url.path,
            "method": request.method,
            "errors": body["errors"],
        },
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content=body)


async def service_unavailable_handler(
    request: Request, exc: ServiceUnavailableError
) -> JSONResponse:
    """
    Convert a request refused for a missing component into a 503 naming each
    component that isn't ready and why.

    returns:
    - response (JSONResponse): status 503, with the reason per component
    """
    logger.warning(
        "Request refused: service not ready",
        extra={
            "event_type": LogEvent.REQUEST_FAILED,
            "path": request.url.path,
            "method": request.method,
            "reasons": exc.reasons,
        },
    )
    body = UnavailableResponse(
        detail="The service cannot answer questions right now.",
        reasons=exc.reasons,
    ).model_dump()
    return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=body)


router = APIRouter()


@router.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """
    Send the bare root path to the interactive API docs, so a browser opening
    the service lands somewhere useful.

    returns:
    - response (RedirectResponse): a redirect to ``/docs``
    """
    return RedirectResponse(url="/docs")


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Report that the process is up, and whether each component /ask needs is
    ready.

    Always 200: the process answering is what a container healthcheck asks.
    Readiness is in the body, so a missing key reads as ``degraded`` with its
    reason instead of the container being restarted for it.

    returns:
    - health (HealthResponse): ``ok`` or ``degraded``, the service version, and
      each component's readiness
    """
    components: dict[str, ComponentStatus] = request.app.state.components
    all_ready = all(component.ready for component in components.values())
    return HealthResponse(
        status="ok" if all_ready else "degraded",
        version=SERVICE_VERSION,
        components=components,
    )


@router.post(
    "/ask",
    responses={
        **INVALID_REQUEST_RESPONSE,
        **UNAVAILABLE_RESPONSE,
        **NOT_IMPLEMENTED_RESPONSE,
    },
)
async def ask(request: Request, body: AskRequest) -> None:
    """
    Answer a question about the Civil Code from the retrieved articles.

    exceptions:
    - ServiceUnavailableError: 503 when a component isn't ready, naming each one
    - RequestValidationError: 422 when the question is empty or too long
    - HTTPException: 501 until retrieval and generation are built in Phase 2
    """
    components: dict[str, ComponentStatus] = request.app.state.components
    reasons = {name: c.detail for name, c in components.items() if not c.ready}
    if reasons:
        raise ServiceUnavailableError(reasons)
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Answering is not implemented yet.",
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Build the application with its middleware, error handlers and routes.

    A factory rather than a module-level app alone, so tests can pass their own
    settings instead of whatever ``.env`` holds.

    returns:
    - app (FastAPI): the configured application; components load when its
      lifespan starts
    """
    app = FastAPI(
        title="Egyptian Civil Code RAG",
        description="Answers questions about the Egyptian Civil Code, citing articles.",
        version=SERVICE_VERSION,
        lifespan=lifespan,
        responses={
            500: {
                "model": ErrorResponse,
                "description": "Unexpected server error; details are only in the server logs.",
            }
        },
    )
    app.state.settings = settings or Settings()
    app.middleware("http")(correlation_id_middleware)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ServiceUnavailableError, service_unavailable_handler)
    app.include_router(router)
    return app


app = create_app()
