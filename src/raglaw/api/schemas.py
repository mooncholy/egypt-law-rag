from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

QUESTION_MAX_CHARS = 2000


class ComponentStatus(BaseModel):
    """Whether one dependency /ask needs is usable, and why not when it isn't."""

    ready: bool
    detail: str


class HealthResponse(BaseModel):
    """Liveness plus readiness: the process answers, and each dependency's state.

    ``status`` is ``ok`` when /ask can answer and ``degraded`` when any component
    is not ready. /health itself answers 200 either way, so a container
    healthcheck stays green while a missing key is reported rather than hidden.
    """

    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, ComponentStatus]


class ErrorResponse(BaseModel):
    """Body of every error the service returns on its own."""

    detail: str


class UnavailableResponse(ErrorResponse):
    """Body of a 503: which components are not ready, and why."""

    reasons: dict[str, str]


class FieldError(BaseModel):
    """One problem with the request; ``field`` is None when the body isn't JSON."""

    field: str | None
    message: str


class ValidationErrorResponse(ErrorResponse):
    """Body of a 422, listing every problem so the caller can fix them in one go."""

    errors: list[FieldError]


class AskRequest(BaseModel):
    """A question about the Egyptian Civil Code, in Arabic or English."""

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=QUESTION_MAX_CHARS
        ),
    ]
