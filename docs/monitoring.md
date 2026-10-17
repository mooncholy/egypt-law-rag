# Monitoring

## Start it

Add the monitoring override to either `up` command (from a clone; see the README's quickstart):

```bash
docker compose -f compose.yaml -f docker/compose.gpu.yml -f docker/compose.monitoring.yml up
```

| Service | Where | What |
| --- | --- | --- |
| Grafana | localhost:3200 (admin / admin) | The `egypt-law-rag` dashboard and two alert rules, provisioned from `docker/monitoring/grafana` |
| Prometheus | localhost:9090 | Scrapes the API's `/metrics` and the Pushgateway every 15 s |
| Langfuse | localhost:3100 (admin@raglaw.local / raglaw-admin) | A trace per `/ask`: `retrieve` (with `rerank` inside it), then `generate` with token usage; the prompt version, retriever version and correlation ID on every trace |
| alert-sink | `docker compose logs alert-sink` | The webhook each alert notification is posted to |

The login, the Langfuse project and its API keys are created at first start from local-only defaults in `docker/compose.monitoring.yml`; change them before exposing any port.

## What `/metrics` exposes

`raglaw.monitoring.metrics`:

- `raglaw_requests_total{path,status}` and `raglaw_request_seconds{path}`: traffic and latency (the dashboard's p50 and p95 panel).
- `llm_tokens_total{model,type}`: prompt and completion tokens, as the LLM endpoint reports them.
- `raglaw_answers_total{outcome}`: answered, refused by the no-answer gate, or answered from a repeal note.
- `raglaw_pii_decisions_total{decision}`, `raglaw_pii_findings_total{kind}` and `raglaw_pii_seconds`: the PII guardrail on every answer.
- `raglaw_query_drift_ratio` and `raglaw_query_drift_threshold`: query drift over the last 20 questions.

## Alerts


- Faithfulness below **0.80**: `scripts/score_traces.py` is the scheduled evaluation. It samples the last day's answered `/ask` traces, scores each with RAGAS faithfulness against the article texts it was given, attaches the score to the trace in Langfuse, and pushes the mean to the Pushgateway (`raglaw_faithfulness`). Run it from cron or the CI runner: `uv run python scripts/score_traces.py --hours 24 --sample 20`, with `RAGLAW_PUSHGATEWAY_URL=http://localhost:9091` in `.env`.
- Query drift below its line for 10 minutes.

## The alert, tested on a real regression


1. The API was pointed at the 3B generator (faithfulness 0.693 in Phase 9) and asked 25 eval questions.
2. `score_traces.py` scored 20 of the answered traces at **0.761** and pushed the mean.
3. Grafana's "Faithfulness below 0.80" fired, and the webhook received the notification.

One wrong answer stood out: "the deadline to appeal a civil court judgment is fifteen days". That rule is in procedure law, not the Civil Code.

## The judge

The cloud judge (`gpt-oss:120b-cloud`) ran out of its free quota (HTTP 429), so RAGAS now uses a local one: `gemma3:12b` on Ollama. It's stronger than the 7B generator and from another family. It serves one call at a time, so RAGAS sends it 4 at once with 900 s each (`answer_evaluation.judge_workers` and `judge_timeout`). Scores from the two judges aren't directly comparable. Re-measured with the new judge, the q8 baseline scores all 57 answered questions:

| Judge | Faithfulness | Answer relevancy | Context precision | Context recall |
| --- | --- | --- | --- | --- |
| `gpt-oss:120b-cloud` | 0.868 | 0.703 | 0.834 | 0.780 |
| `gemma3:12b` | 0.905 | 0.636 | 0.886 | 0.927 |

Source: `docs/metrics/answers_local_judge.json`. The new judge is more lenient on faithfulness. The 0.75 gate and the 0.80 alert still separate the 8-bit model (0.905) from the 3B regression (0.761). Eight traces alone once scored 0.796, so the scheduled evaluation samples 20.

## Query drift

Each question's embedding is compared with the eval set's centroid. The ratio is the window's mean cosine over the eval set's own (0.6689), so on-topic questions score about 1.0. The line is 3 standard errors below 1 for a 20-question window: 0.9461. On the real embedder (`docs/reports/drift.md`): the eval set scores 1.000, its halves 1.010 and 0.990, and 12 family-law questions (personal status law, not the Civil Code) 0.914.

## PII guardrail

The guardrail (`raglaw.monitoring.pii`) redacts Egyptian national IDs (14 digits with a real birth date and governorate code), Egyptian mobile numbers, email addresses, and Arabic names after an honorific, in Western or Arabic-Indic digits. It runs on every `/ask` answer, and on `/ask/stream`, where it holds back the last 10 words (the longest number it recognizes, a phone written with a separator between every digit) until they're checked. On the Code's own text (1,149 articles, both languages, no personal data) it redacted nothing (0 false positives in 2,298 texts) and took 0.046 ms per text.

## Token cost

The self-hosted price of a token is the GPU's hourly price over the tokens it serves in an hour at the measured throughput. The cost panel multiplies the completion tokens per second by that price. Both inputs are dashboard constants:
- `throughput_tokens_per_s` is 44.2, the default generator's measured rate. Ollama decodes one answer at a time, so that is its capacity (`docs/metrics/generator.json`).
- `gpu_usd_per_hour` is 0.35, an **assumed** hourly price for a 16 GB GPU. Set your own.

With those two, a token costs about $2.2 per million.
