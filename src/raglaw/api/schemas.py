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

    ``status`` is ``healthy`` when /ask can answer and ``degraded`` when any
    component is not ready. /health itself answers 200 either way, so a
    container healthcheck stays green while a missing key is reported rather
    than hidden.
    """

    status: Literal["healthy", "degraded"] = Field(
        description="`healthy` when every component is ready, otherwise `degraded`."
    )
    version: str = Field(description="The running service's package version.")
    documents_indexed: int = Field(
        ge=0,
        description="Chunks in the search index (its manifest's count); 0 when "
        "no index is found.",
    )
    components: dict[str, ComponentStatus] = Field(
        description="Readiness per component, keyed by name (`llm`, `models`, "
        "`retriever`)."
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


class AskResponse(BaseModel):
    """An answer from the Code's articles, and the articles it rests on."""

    answer: str = Field(
        description="The answer, in the question's language, citing articles by "
        "number; or the reply that the Code's articles don't address the question."
    )
    sources: list[str] = Field(
        description="Each article the answer cites that retrieval returned, as "
        "`Egyptian Civil Code, Article N`; empty when no article addresses it."
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
