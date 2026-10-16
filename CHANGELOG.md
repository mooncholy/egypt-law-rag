# Changelog

One entry per pull request into `feat-civil-law-chatbot`, in merge order. The handbook's phases are in parentheses.

## Unreleased

- **#14 Monitoring and release (Phases 14–15).**
  - Prometheus metrics on `/metrics`: requests, latency, LLM tokens, answer outcomes, PII decisions, query drift.
  - Langfuse traces of every `/ask`: `retrieve` → `rerank` → `generate`, with token usage and the prompt and retriever versions.
  - The PII guardrail (U42) on every answer, streamed ones included: 0 false positives on the Code's 2,298 texts.
  - Query drift (U41): family-law questions score 0.914 against a 0.946 line.
  - Grafana provisioned with a p95 panel, a token-cost panel, and alerts on faithfulness below 0.80 and on drift.
  - A scheduled scorer that puts RAGAS faithfulness on sampled traces.
  - The judge moved to a local model after the cloud judge's quota ran out.
  - This changelog and the architecture diagram.
- **#13 Serving and optimization (Phases 12–13).**
  - `/ask/stream` (server-sent events).
  - Batch re-indexing.
  - vLLM serving `Qwen2.5-7B-Instruct-AWQ` as an opt-in override.
  - A BentoML form of the service with batchable embed and rerank APIs.
  - An nginx 95/5 canary.
  - Locust at 50 users: p95 180 s on Ollama, 101 s on vLLM.
  - A benchmark harness (U40).
  - AWQ and a distilled reranker, measured and both rejected on quality: faithfulness −0.052 for AWQ; for the reranker, 14 of 131 questions lost for a 6.1× speedup.

## Merged

- **#12 CI/CD (Phase 11).**
  - A self-hosted GPU job that rebuilds the index, scores retrieval and RAGAS on the 20-question CI set, gates on faithfulness ≥ 0.75 and on recall within one question of the base branch, and exports the production retriever.
  - It builds the images and pushes them on merges.
- **#11 Corpus validation (Phase 4).** The `validate` stage checks the corpus against C1–C16 and the gold sample: Arabic character error rate 0.03%, with Article 54's accepted differences checked span by span.
- **#10 Docker (Phase 10).** `docker compose up` answers `/ask` on any machine. The retriever is baked into the `api` image (D26), the `models` service embeds and reranks, Ollama generates, and the images are on Docker Hub.
- **#9 Evaluation (Phase 9).**
  - Faithfulness 0.868 with the 7B generator and prompt v2.
  - Two Arabic text rules (punctuation order, paragraph markers).
  - Questions on the repealed ranges answered from their repeal note.
- **#8 Answer evaluation (Phase 9).** RAGAS (faithfulness, answer relevancy, context precision, context recall) on 65 questions, with a judge stronger than the generator, logged to MLflow.
- **#7 Tracking (Phase 8).** An MLflow tracking server (Postgres, artifacts in S3), one run per stage, and the retriever registered with `candidate` and `production` aliases.
- **#6 API (Phase 7).** `/ask` answers from the champion retriever, citing only retrieved articles, with a no-answer gate on the reranker's rank-1 score.
- **#5 Retrieval (Phase 6).** bge-m3 in Qdrant and BM25, fused, scored before any LLM is involved.
- **#3 Corpus (Phases 1–3 and 5).** The Civil Code PDF to 1,149 articles and 1,150 chunks as DVC stages.
- **#1 Setup (Phase 0).** The `raglaw` package, FastAPI, DVC and the repository's rules.
