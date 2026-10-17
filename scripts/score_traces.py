"""The scheduled evaluation (Phase 14): score sampled /ask traces' faithfulness.

Reads the ``ask`` traces Langfuse received in the last ``--hours``, keeps
those the LLM answered (not refused by the gate, not a repeal note's reply),
samples ``--sample`` of them, and scores each with RAGAS faithfulness
against the article texts it was given (the trace's ``contexts``), using the
judge in ``.env`` (``RAGLAW_JUDGE_*``). Each score is attached to its trace
(``faithfulness``), and the mean goes to the Pushgateway as
``raglaw_faithfulness``, where Grafana's alert fires under 0.80.

Run on a schedule (cron, or the CI runner) against the monitoring stack:
    uv run python scripts/score_traces.py --hours 24 --sample 20
"""

import argparse
import logging
import random
from datetime import UTC, datetime, timedelta

from raglaw.config import Settings

logger = logging.getLogger("score_traces")

TRACE = "ask"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument(
        "--until",
        type=datetime.fromisoformat,
        help="only traces started before this UTC time (ISO), e.g. to score one window",
    )
    ap.add_argument("--sample", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="judge calls at once; a local judge serves one at a time",
    )
    args = ap.parse_args()
    settings = Settings()
    from langchain_openai import ChatOpenAI
    from langfuse import Langfuse
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    from ragas import RunConfig

    from raglaw.answer_eval import ragas_scores
    from raglaw.monitoring.tracing import every_page, faithfulness_sample

    public, secret = settings.langfuse_public_key, settings.langfuse_secret_key
    if not (settings.langfuse_host and public and secret):
        raise SystemExit("set RAGLAW_LANGFUSE_HOST and both keys in .env")
    langfuse = Langfuse(
        public_key=public.get_secret_value(),
        secret_key=secret.get_secret_value(),
        base_url=settings.langfuse_host,
    )
    until = args.until.replace(tzinfo=UTC) if args.until else datetime.now(UTC)
    since = until - timedelta(hours=args.hours)
    found = every_page(
        langfuse.api.observations.get_many,
        name=TRACE,
        is_root_observation=True,
        from_start_time=since,
        to_start_time=until,
        fields="core,basic,io,metadata",
        expand_metadata="contexts",
        parse_io_as_json=False,
    )
    samples = [s for s in map(faithfulness_sample, found) if s]
    picked = random.Random(args.seed).sample(samples, min(args.sample, len(samples)))
    logger.info(
        "%d ask traces since %s, %d answered by the LLM, %d sampled",
        len(found),
        since.isoformat(timespec="minutes"),
        len(samples),
        len(picked),
    )
    if not picked:
        raise SystemExit("no answered traces to score")
    judge = ChatOpenAI(
        base_url=str(settings.judge_base_url),
        model=settings.judge_model,
        api_key=settings.judge_api_key.get_secret_value(),
        temperature=0,
    )
    rows = [{k: v for k, v in s.items() if k != "trace_id"} for s in picked]
    scores = ragas_scores(
        rows,
        judge,
        None,
        metrics=["faithfulness"],
        run_config=RunConfig(max_workers=args.workers, timeout=900),
    )
    values = []
    for sample, score in zip(picked, scores, strict=True):
        if (value := score["faithfulness"]) is None:
            continue
        values.append(value)
        langfuse.create_score(
            trace_id=sample["trace_id"],
            name="faithfulness",
            value=value,
            comment=f"scheduled evaluation, judge {settings.judge_model}",
        )
    langfuse.flush()
    if not values:
        raise SystemExit(
            f"the judge scored none of the {len(picked)} traces (its errors are "
            "above; a 429 means its quota ran out); nothing was pushed"
        )
    mean = sum(values) / len(values)
    logger.info("Faithfulness %.3f over %d scored traces", mean, len(values))
    if settings.pushgateway_url:
        registry = CollectorRegistry()
        Gauge(
            "raglaw_faithfulness",
            "Mean RAGAS faithfulness of the sampled /ask traces.",
            registry=registry,
        ).set(mean)
        Gauge(
            "raglaw_faithfulness_traces",
            "Traces the last scheduled evaluation scored.",
            registry=registry,
        ).set(len(values))
        push_to_gateway(settings.pushgateway_url, job="score_traces", registry=registry)
        logger.info("Pushed to %s", settings.pushgateway_url)


if __name__ == "__main__":
    main()
