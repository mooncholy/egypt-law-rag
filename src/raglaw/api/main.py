import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from fastapi import APIRouter, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from langchain_core.embeddings import Embeddings

from raglaw.api.schemas import (
    AskRequest,
    AskResponse,
    ComponentStatus,
    ErrorResponse,
    FieldError,
    HealthResponse,
    UnavailableResponse,
    ValidationErrorResponse,
)
from raglaw.config import Settings
from raglaw.llm import ChatModel, LLMError, OpenAIChat
from raglaw.logging_conf import (
    configure_server_logging,
    correlation_id_var,
    get_logger,
)
from raglaw.rag import AnswerPipeline, load_prompt
from raglaw.retrieval.remote import (
    ModelsServiceError,
    RemoteEmbeddings,
    RemoteReranker,
    models_health,
)
from raglaw.retrieval.rerank import RerankScorer
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


class ServiceUnavailableError(Exception):
    """Raised when a request needs a component that is not ready."""

    def __init__(self, reasons: dict[str, str]) -> None:
        super().__init__(", ".join(f"{k}: {v}" for k, v in reasons.items()))
        self.reasons = reasons


@dataclass
class Service:
    """What the API loaded at startup: each component's state, and the pipeline.

    ``pipeline`` is None unless every component is ready; ``close`` releases
    the index.
    """

    components: dict[str, ComponentStatus]
    documents_indexed: int = 0
    pipeline: AnswerPipeline | None = None
    close: Callable[[], None] = field(default=lambda: None)


def llm_component(settings: Settings) -> tuple[ComponentStatus, ChatModel | None]:
    """
    The LLM client, when its endpoint, model and key are all set (D20).

    Configuration only: no call is made, so startup costs nothing and a
    hosted API isn't billed for a probe.

    returns:
    - status (ComponentStatus): ready, or naming each missing setting
    - llm (ChatModel | None): the client when ready
    """
    key = settings.llm_api_key
    values = {
        "RAGLAW_LLM_API_KEY": key.get_secret_value().strip() if key else "",
        "RAGLAW_LLM_BASE_URL": settings.llm_base_url or "",
        "RAGLAW_LLM_MODEL": settings.llm_model or "",
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        verb = "is" if len(missing) == 1 else "are"
        return ComponentStatus(
            ready=False, detail=f"{', '.join(missing)} {verb} not set."
        ), None
    llm = OpenAIChat(
        settings.llm_base_url, values["RAGLAW_LLM_API_KEY"], settings.llm_model
    )
    return ComponentStatus(
        ready=True, detail=f"{settings.llm_model} at {settings.llm_base_url}"
    ), llm


def models_component(settings: Settings) -> ComponentStatus:
    """
    Whether the ``models`` service answers and serves the pinned models (D18).

    returns:
    - status (ComponentStatus): ready, or why not (down, or other models)
    """
    try:
        served = models_health(settings.models_url)
    except ModelsServiceError as exc:
        return ComponentStatus(ready=False, detail=str(exc))
    wanted = {
        "embedding": f"{settings.embedding.model}@{settings.embedding.revision}",
        "reranker": f"{settings.reranker.model}@{settings.reranker.revision}",
    }
    differ = [f"{k} {served.get(k)}" for k, v in wanted.items() if served.get(k) != v]
    if differ:
        return ComponentStatus(
            ready=False,
            detail=f"{settings.models_url} serves {', '.join(differ)}, not the "
            "pinned models in params.yaml",
        )
    return ComponentStatus(
        ready=True, detail=f"{settings.models_url} on {served.get('device')}"
    )


def indexed_chunks(settings: Settings) -> int:
    """
    The chunk count in the dense index's manifest.

    returns:
    - count (int): 0 when there is no readable index
    """
    try:
        from raglaw.retrieval.dense import DenseManifest
        from raglaw.retrieval.manifest import read_manifest

        manifest = read_manifest(settings.paths.index_dir / "dense", DenseManifest)
    except ImportError, OSError, ValueError:  # no deps, no index, or unreadable
        return 0
    return len(manifest.chunk_ids)


def retriever_component(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    reranker: RerankScorer | None = None,
) -> tuple[ComponentStatus, Any]:
    """
    ``champion``, loaded by alias from the registry (D16) and opened on this
    machine's index, querying the ``models`` service unless ``embeddings`` and
    ``reranker`` are given. Opening it embeds a probe text, so the ``models``
    service must be up.

    returns:
    - status (ComponentStatus): ready, or why it couldn't load
    - champion (Champion | None): the open retriever when ready
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
        from langchain_qdrant.qdrant import QdrantVectorStoreError
        from mlflow.exceptions import MlflowException

        from raglaw.retrieval.champion import ALIAS, MODEL_NAME, load_champion
    except ImportError as exc:
        return ComponentStatus(
            ready=False, detail=f"retrieval dependencies missing: {exc}"
        ), None
    try:
        champion = load_champion(
            settings,
            embeddings=embeddings or RemoteEmbeddings(settings.models_url),
            reranker=reranker or RemoteReranker(settings.models_url),
        )
    # No registry or alias; the artifacts unreachable in S3; no index on this
    # machine; an index other than the registered one (IndexMismatchError is a
    # ValueError); no models service, or one embedding to another size.
    except (
        MlflowException,
        BotoCoreError,
        ClientError,
        OSError,
        ValueError,
        ModelsServiceError,
        QdrantVectorStoreError,
    ) as exc:
        return ComponentStatus(
            ready=False, detail=f"champion not loaded: {type(exc).__name__}: {exc}"
        ), None
    return ComponentStatus(
        ready=True, detail=f"models:/{MODEL_NAME}@{ALIAS} on {settings.paths.index_dir}"
    ), champion


def load_service(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
    reranker: RerankScorer | None = None,
) -> Service:
    """
    Check or load every component /ask needs, once, at startup.

    A component that fails is recorded with its reason instead of stopping the
    process, so /health can report it and /ask can refuse with that reason.
    ``embeddings`` and ``reranker`` replace the ``models`` service's (tests).

    returns:
    - service (Service): every component's state, and the pipeline when all
      are ready

    exceptions:
    - FileNotFoundError: ``answer.prompt_version`` names no prompt file
    """
    load_prompt(settings.answer.prompt_version)  # a config error stops startup
    llm_status, llm = llm_component(settings)
    retriever_status, champion = retriever_component(
        settings, embeddings=embeddings, reranker=reranker
    )
    components = {
        "llm": llm_status,
        "models": models_component(settings),
        "retriever": retriever_status,
    }
    service = Service(
        components=components,
        documents_indexed=(
            champion.documents_indexed if champion else indexed_chunks(settings)
        ),
    )
    if champion is not None:
        service.close = champion.close
        if all(c.ready for c in components.values()):
            service.pipeline = AnswerPipeline(champion.retrieve, llm, settings.answer)
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the service's components when the process starts, and log shutdown.

    Loading here rather than per request keeps /ask latency down to retrieval
    and the LLM call. Components that aren't ready are logged at WARNING, so a
    missing key shows up at boot and not only on the first request.
    """
    logger.info("Starting service", extra={"event_type": LogEvent.STARTUP})
    service: Service = app.state.load(app.state.settings)
    app.state.service = service
    for name, component in service.components.items():
        if not component.ready:
            logger.warning(
                "Component not ready: %s",
                component.detail,
                extra={"event_type": LogEvent.STARTUP, "component": name},
            )
    try:
        yield
    finally:
        service.close()
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
    reason instead of the container being restarted for it. Components are
    checked at startup; a component lost later fails /ask with a 503.

    returns:
    - health (HealthResponse): ``healthy`` or ``degraded``, the service
      version, the index's chunk count, and each component's readiness
    """
    service: Service = request.app.state.service
    all_ready = all(c.ready for c in service.components.values())
    return HealthResponse(
        status="healthy" if all_ready else "degraded",
        version=SERVICE_VERSION,
        documents_indexed=service.documents_indexed,
        components=service.components,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={**INVALID_REQUEST_RESPONSE, **UNAVAILABLE_RESPONSE},
)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    """
    Answer a question about the Civil Code from the retrieved articles.

    returns:
    - answer (AskResponse): the answer, and the article citations it rests on

    exceptions:
    - ServiceUnavailableError: 503 when a component isn't ready at startup, or
      the ``models`` service or the LLM fails on this request, naming each one
    - RequestValidationError: 422 when the question is empty or too long
    """
    service: Service = request.app.state.service
    reasons = {name: c.detail for name, c in service.components.items() if not c.ready}
    if reasons or service.pipeline is None:
        raise ServiceUnavailableError(reasons or {"pipeline": "not built"})
    try:
        answer = await service.pipeline.answer(body.question)
    except ModelsServiceError as exc:
        raise ServiceUnavailableError({"models": str(exc)}) from exc
    except LLMError as exc:
        raise ServiceUnavailableError({"llm": str(exc)}) from exc
    return AskResponse(answer=answer.text, sources=answer.sources)


def create_app(
    settings: Settings | None = None,
    load: Callable[[Settings], Service] = load_service,
) -> FastAPI:
    """
    Build the application with its middleware, error handlers and routes.

    A factory rather than a module-level app alone, so tests can pass their own
    settings instead of whatever ``.env`` holds, and their own ``load``.

    returns:
    - app (FastAPI): the configured application; ``load`` runs when its
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
    app.state.load = load
    app.middleware("http")(correlation_id_middleware)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(ServiceUnavailableError, service_unavailable_handler)
    app.include_router(router)
    return app


app = create_app()
