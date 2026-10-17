# Serving and operations

How the service runs beyond the quickstart: vLLM as the generator, the load test, adding articles without the PDF pipeline, the canary rollout and the BentoML form of the service. The measured trade-offs behind these choices are in [reports/optimization.md](reports/optimization.md).

## vLLM (optional)

On a GPU, vLLM can serve the generator in place of Ollama, with continuous batching. It serves **`Qwen/Qwen2.5-7B-Instruct-AWQ`** (revision `b250375`), the same model in 4 bits, on `vllm/vllm-openai:v0.30.0`:

```bash
docker compose -f compose.yaml -f docker/compose.gpu.yml -f docker/compose.vllm.yml up
```

It isn't the default because it answers less faithfully: 0.765 against 0.868 on the same 65 questions. Phase 13 allows a drop of at most 0.03 (`docs/reports/optimization.md`).

## Load test

`loadtest/locustfile.py`: 50 users ask the retrieval eval set's 142 questions (Arabic and English, out-of-scope ones included) at random, waiting 1 to 3 s between requests, for 5 minutes. `loadtest/sample_usage.sh` records GPU and container CPU every 5 s alongside. Measured on one NVIDIA GeForce RTX 5060 Ti (16 GB) with `models` on the same GPU, under WSL 2:

| Generator | `/ask` p50 | p95 | p99 | Failed | Throughput | Report |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5:7b-instruct-q8_0` on Ollama (default) | 66 s | **180 s** | 180 s | 32% | 0.42/s | `reports/locust_50_ollama.html` |
| `Qwen2.5-7B-Instruct-AWQ` on vLLM | 49 s | **101 s** | 162 s | 1.7% | 0.82/s | `reports/locust_50_vllm.html` |

The GPU sat at 99% in both runs.
- **On Ollama:** it generated one answer at a time, so requests queued behind each other.
- **On vLLM:** batching doubled throughput, and the bottleneck moved to retrieval. Retrieval runs one question at a time (Qdrant's local mode) and shares the GPU with generation.

With one user, the 8-bit model's answer takes 2.4 s at p50 and 5.8 s at p95 (`docs/metrics/generator.json`).

## Batch re-indexing

`scripts/reindex.py` adds new articles (an amendment, say) without re-running the PDF pipeline. It chunks them as `chunk` does, appends them to the corpus's chunks, rebuilds both index halves into a new directory, and reports `documents_indexed` before and after. The tracked index is left untouched.

```bash
uv run python scripts/reindex.py scripts/examples/test_amendment.json --out build/reindexed
# documents_indexed: 1150 -> 1151 (articles [1150] added) in build/reindexed
```

## Canary rollout

A new `api` image reaches users in stages, behind nginx (`docker/canary/nginx.conf`). Each response says which version served it: `X-Raglaw-Variant` (`stable` or `canary`, from nginx) and `X-Raglaw-Version` (the package version, from the API).

```bash
RAGLAW_STABLE_TAG=0.1.0 RAGLAW_CANARY_TAG=sha-<7> \
  docker compose -f compose.yaml -f docker/compose.canary.yml up
curl -si localhost:8080/health | grep -i x-raglaw
```

| Stage | Canary share | Moves on when | Rolls back when |
| --- | --- | --- | --- |
| 1. CI | 0% | The deliver job passes: RAGAS faithfulness ≥ 0.75 on the CI set, recall@5 within one question of the base branch | Either gate fails; the image is never pushed |
| 2. Canary | 5% (`split_clients` 95/5) | An hour of traffic with the canary's error rate and p95 no worse than stable's | Its 5xx rate or p95 is above stable's: set its share to 0% and reload nginx |
| 3. Widen | 25%, then 50% | Same checks at each step | Same |
| 4. Promote | 100% | Retag the canary image as stable; the old stable tag stays for a rollback | Redeploy the old stable tag |

The split draws each request on its own (`$request_id`); to keep one client on one version, hash on `$remote_addr` instead. Moving a stage is an edit to the two percentages and `nginx -s reload`; nothing restarts.

## BentoML

`raglaw.serving.bento_service` is the same service as two BentoML services: `RagLaw` (an async `/ask`, the same `AnswerPipeline` and baked retriever as the FastAPI app) over `Models`, whose `embed` and `rerank` are batchable APIs. BentoML's adaptive batching merges calls that arrive within 50 ms into one model batch (up to 64 texts or 128 pairs).

```bash
uv sync --group bento --no-group torch-cpu --group torch-gpu   # keeps the default groups, embed included
RAGLAW_CHAMPION_DIR=build/champion uv run bentoml serve raglaw.serving.bento_service:RagLaw
curl -s localhost:3000/ask -H 'Content-Type: application/json' -d '{"question": "What is the age of majority?"}'
```

Retrieval runs under a lock (Qdrant's local mode isn't safe across threads), so concurrent questions reach `Models` one at a time; batching then pays off within a question (30 rerank pairs in one batch) more than across questions. A Qdrant server would lift the lock.
