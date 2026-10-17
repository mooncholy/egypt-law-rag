"""The API's Prometheus metrics (Phase 14), served on ``/metrics``.

- ``raglaw_requests_total{path,status}`` and ``raglaw_request_seconds{path}``:
  traffic and latency (Grafana's p95 panel);
- ``llm_tokens_total{model,type}``: prompt and completion tokens, from the
  LLM's reported usage (Grafana's cost panel prices them);
- ``raglaw_answers_total{outcome}``: answered, refused by the no-answer gate,
  or answered from a repeal note;
- ``raglaw_pii_decisions_total{decision}`` and ``raglaw_pii_seconds``: the
  guardrail's decision on every answer, and what it adds to latency;
- ``raglaw_pii_findings_total{kind}``: what it redacted;
- ``raglaw_query_drift_ratio`` and ``raglaw_query_drift_threshold``: the drift
  window's score and its alert line.

The scheduled evaluation's faithfulness reaches Prometheus through the
Pushgateway (``scripts/score_traces.py``), not from here.
"""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "raglaw_requests_total",
    "Requests served, by path and status code.",
    ["path", "status"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "raglaw_request_seconds",
    "Time to serve a request, by path.",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 180),
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "llm_tokens",
    "Tokens the LLM read (prompt) and wrote (completion), by model.",
    ["model", "type"],
    registry=REGISTRY,
)
ANSWERS = Counter(
    "raglaw_answers",
    "Answers by outcome: answered, refused (no-answer gate) or repealed.",
    ["outcome"],
    registry=REGISTRY,
)
PII_DECISIONS = Counter(
    "raglaw_pii_decisions",
    "The PII guardrail's decision on each answer: clean or redacted.",
    ["decision"],
    registry=REGISTRY,
)
PII_FINDINGS = Counter(
    "raglaw_pii_findings",
    "Personal data redacted from answers, by kind.",
    ["kind"],
    registry=REGISTRY,
)
PII_SECONDS = Histogram(
    "raglaw_pii_seconds",
    "Time the PII guardrail adds to an answer.",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05),
    registry=REGISTRY,
)
DRIFT = Gauge(
    "raglaw_query_drift_ratio",
    "Mean cosine of the recent questions to the eval set's centroid, over the "
    "eval set's own mean (1.0: on topic).",
    registry=REGISTRY,
)
DRIFT_THRESHOLD = Gauge(
    "raglaw_query_drift_threshold",
    "The drift ratio below which the current window counts as drift.",
    registry=REGISTRY,
)


def record_tokens(model: str, prompt: int, completion: int) -> None:
    """Count one LLM call's tokens (``OpenAIChat``'s ``on_usage``)."""
    LLM_TOKENS.labels(model, "prompt").inc(prompt)
    LLM_TOKENS.labels(model, "completion").inc(completion)
