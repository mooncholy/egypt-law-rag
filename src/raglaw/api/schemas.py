from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

QUESTION_MAX_CHARS = 2000


class ComponentStatus(BaseModel):
    """Whether one dependency /ask needs is usable, and why not when it isn't."""

    ready: bool = Field(description="True when the component can serve /ask.")
    detail: str = Field(
        description="What the component is using, or why it isn't ready."
    )


class HealthResponse(BaseModel):
    """Liveness plus readiness: the process answers, and each dependency's state.

    ``status`` is ``ok`` when /ask can answer and ``degraded`` when any component
    is not ready. /health itself answers 200 either way, so a container
    healthcheck stays green while a missing key is reported rather than hidden.
    """

    status: Literal["ok", "degraded"] = Field(
        description="`ok` when every component is ready, otherwise `degraded`."
    )
    version: str = Field(description="The running service's package version.")
    components: dict[str, ComponentStatus] = Field(
        description="Readiness per component, keyed by name (`llm`, `retriever`)."
    )


class ErrorResponse(BaseModel):
    """Body of every error the service returns on its own."""

    detail: str = Field(description="What went wrong, for a person to read.")


class UnavailableResponse(ErrorResponse):
    """Body of a 503: which components are not ready, and why."""

    reasons: dict[str, str] = Field(
        min_length=1,
        description="Why each unready component isn't ready, keyed by component "
        "name. A 503 always names at least one.",
    )


class FieldError(BaseModel):
    """One problem with the request; ``field`` is None when the body isn't JSON."""

    field: str | None = Field(
        description="Dotted path of the offending field (e.g. `question`); null "
        "when the body isn't valid JSON."
    )
    message: str = Field(description="What is wrong with it.")


class ValidationErrorResponse(ErrorResponse):
    """Body of a 422, listing every problem so the caller can fix them in one go."""

    errors: list[FieldError] = Field(
        min_length=1, description="Every problem found, not just the first."
    )


class AskRequest(BaseModel):
    """A question about the Egyptian Civil Code, in Arabic or English."""

    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True, min_length=1, max_length=QUESTION_MAX_CHARS
        ),
        Field(
            description="The question, in Arabic or English. Surrounding "
            f"whitespace is stripped; 1 to {QUESTION_MAX_CHARS} characters remain."
        ),
    ]
