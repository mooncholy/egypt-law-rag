# egypt-law-rag

[![ci](https://github.com/mooncholy/egypt-law-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/mooncholy/egypt-law-rag/actions/workflows/ci.yml)

A retrieval-augmented generation (RAG) chatbot that answers questions about the Egyptian Civil Code.
The source is a bilingual PDF (170 pages, English and Arabic side by side).
A DVC pipeline turns it into a validated corpus of articles and chunks, and a FastAPI service answers questions over that corpus.

**Status:** the corpus, retrieval, `/ask`, its containers, CI/CD and serving are built; monitoring is next.
- `profile → extract → repair → assemble → chunk` turn the PDF into 1,149 articles and 1,150 chunks.
- `embed` and `bm25` index the chunks (bge-m3 in Qdrant, BM25 over folded Arabic), and `evaluate_retrieval` scores retrieval before any LLM is involved.
- The registered retriever (`models:/civil-code-retriever@production`, version 4) fuses dense and BM25 search, reranks the top 30 with a cross-encoder, and answers questions on the repealed ranges (Articles 54–80, 389–417) from the Code's repeal note. It finds the governing articles in the top 5 for 86% of held-out questions (recall@5 0.864; the target is 0.9).
- `/ask` answers from that retriever and `qwen2.5:7b-instruct-q8_0`, citing only articles it retrieved: faithfulness 0.868 on 65 questions (RAGAS, `docs/metrics/answers.json`). When the rank-1 reranker score is below a threshold, it replies that the Code doesn't address the question, without calling the LLM.
- `validate` checks the corpus against C1–C16 and the gold sample (Arabic character error rate 0.03%).
- Serving (Phase 12–13): `/ask/stream` streams the answer; a canary config; a BentoML form of the service; vLLM as an opt-in generator; batch re-indexing; a load test at 50 users; AWQ and a distilled reranker measured, and both rejected on quality (`docs/reports/optimization.md`).

## Run it with Docker

Three commands on any machine with Docker; no AWS account, data download or Python setup:

```bash
git clone https://github.com/mooncholy/egypt-law-rag.git && cd egypt-law-rag
cp .env.example .env
docker compose up
```

The first start downloads the two retrieval models (about 4.5 GB) and the generator (about 8 GB) into Docker volumes; later starts reuse them. `api` starts once both are ready. Then:

```bash
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "ما هي سن الرشد في القانون المدني؟"}'
```

A real response from `docker compose up` (version 4 of the retriever, `qwen2.5:7b-instruct-q8_0`):

```json
{
  "answer": "سن الرشد في القانون المدني هو واحد وعشرون سنة ميلادية كاملة، وفقًا لـ `المادة ٤٤`.",
  "sources": ["Egyptian Civil Code, Article 44"]
}
```

A question on a repealed range is answered from the Code's repeal note, without the LLM:

```json
{
  "answer": "كانت أحكام إثبات الالتزام في القانون المدني واردة في المواد من ٣٨٩ إلى ٤١٧، وقد ألغيت. ولا يتضمن القانون المدني الحالي من هذه المواد إلا ما يفيد إلغاءها.",
  "sources": ["Egyptian Civil Code, Article 389"]
}
```

An empty question gets a 422. With `models` on CPU and the generator on an RTX 5060 Ti, the first answer took 55 s (Ollama loading the model) and the repealed-range answer 21 s, most of it CPU reranking; latency is measured properly in Phase 12.

On CPU the generator needs about 9 GB of free memory and answers slowly. On a machine with less than about 16 GB of RAM (e.g., a 4-core GitHub Codespace), set `RAGLAW_LLM_MODEL=qwen2.5:7b-instruct-q4_K_M` in `.env` before `docker compose up`: the same model in 4 bits, 4.7 GB instead of 8.1 GB. Its answers aren't measured; the faithfulness above is the default 8-bit model's. With an NVIDIA GPU (CUDA 13 driver and the NVIDIA Container Toolkit; Docker Desktop's WSL backend includes it):

```bash
docker compose -f compose.yaml -f docker/compose.gpu.yml up
```

This runs the CUDA build of the `models` image (`mooncholy/raglaw-models:latest-gpu`) and gives the GPU to `models` and `ollama`. A `-gpu` tag that isn't published is built locally on first `up`; that needs no project data.

| Service | Port | What |
| --- | --- | --- |
| `api` | 8000 | `/ask` and `/health`. Bakes in the index and a copy of the production retriever (D26); runs as a non-root user, without torch |
| `models` | 8001 | The embedder and reranker (D18), on pinned revisions; weights in the `hf-cache` volume |
| `ollama` | – | The generator, pulled once into the `ollama` volume by `ollama-pull` |

The answer can also be streamed as server-sent events: a `data: {"token": ...}` event per piece as the LLM writes it, then `event: sources` with the cited articles.

```bash
curl -N localhost:8000/ask/stream -H 'Content-Type: application/json' \
  -d '{"question": "What is the age of majority?"}'
```

### vLLM (optional)

On a GPU, vLLM can serve the generator in place of Ollama, with continuous batching. It serves **`Qwen/Qwen2.5-7B-Instruct-AWQ`** (revision `b250375`), the same model in 4 bits, on `vllm/vllm-openai:v0.30.0`:

```bash
docker compose -f compose.yaml -f docker/compose.gpu.yml -f docker/compose.vllm.yml up
```

It isn't the default because it answers less faithfully: 0.765 against 0.868 on the same 65 questions. Phase 13 allows a drop of at most 0.03 (`docs/reports/optimization.md`).

### Load test

`loadtest/locustfile.py`: 50 users ask the retrieval eval set's 142 questions (Arabic and English, out-of-scope ones included) at random, waiting 1 to 3 s between requests, for 5 minutes. `loadtest/sample_usage.sh` records GPU and container CPU every 5 s alongside. Measured on one NVIDIA GeForce RTX 5060 Ti (16 GB) with `models` on the same GPU, under WSL 2:

| Generator | `/ask` p50 | p95 | p99 | Failed | Throughput | Report |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen2.5:7b-instruct-q8_0` on Ollama (default) | 66 s | **180 s** | 180 s | 32% | 0.42/s | `reports/locust_50_ollama.html` |
| `Qwen2.5-7B-Instruct-AWQ` on vLLM | 49 s | **101 s** | 162 s | 1.7% | 0.82/s | `reports/locust_50_vllm.html` |

The GPU sat at 99% in both runs.
- **On Ollama:** it generated one answer at a time, so requests queued behind each other.
- **On vLLM:** batching doubled throughput, and the bottleneck moved to retrieval. Retrieval runs one question at a time (Qdrant's local mode) and shares the GPU with generation.

With one user, the 8-bit model's answer takes 2.4 s at p50 and 5.8 s at p95 (`docs/metrics/generator.json`).

### Batch re-indexing

`scripts/reindex.py` adds new articles (an amendment, say) without re-running the PDF pipeline. It chunks them as `chunk` does, appends them to the corpus's chunks, rebuilds both index halves into a new directory, and reports `documents_indexed` before and after. The tracked index is left untouched.

```bash
uv run python scripts/reindex.py scripts/examples/test_amendment.json --out build/reindexed
# documents_indexed: 1150 -> 1151 (articles [1150] added) in build/reindexed
```

### Build the images (maintainers)

The `api` image needs the index (`dvc pull`) and a copy of the `production` retriever, exported by alias from the tracking server, so only a maintainer with the project's AWS profile can build it:

```bash
uv run dvc pull data/index
uv run python -m raglaw.retrieval.champion export         # production -> build/champion (gitignored)
docker compose build
docker compose push                                       # mooncholy/raglaw-api, mooncholy/raglaw-models (latest)
RAGLAW_IMAGE_TAG=0.1.0 docker compose build && RAGLAW_IMAGE_TAG=0.1.0 docker compose push
GPU="-f compose.yaml -f docker/compose.gpu.yml"           # the CUDA models image, tagged <tag>-gpu
docker compose $GPU build models && docker compose $GPU push models
RAGLAW_IMAGE_TAG=0.1.0 docker compose $GPU build models && RAGLAW_IMAGE_TAG=0.1.0 docker compose $GPU push models
```

Moving the `production` alias changes what a new `api` image serves; a running container keeps the copy it was built with.

### Canary rollout

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

### BentoML

`raglaw.serving.bento_service` is the same service as two BentoML services: `RagLaw` (an async `/ask`, the same `AnswerPipeline` and baked retriever as the FastAPI app) over `Models`, whose `embed` and `rerank` are batchable APIs. BentoML's adaptive batching merges calls that arrive within 50 ms into one model batch (up to 64 texts or 128 pairs).

```bash
uv sync --group bento --no-group torch-cpu --group torch-gpu   # keeps the default groups, embed included
RAGLAW_CHAMPION_DIR=build/champion uv run bentoml serve raglaw.serving.bento_service:RagLaw
curl -s localhost:3000/ask -H 'Content-Type: application/json' -d '{"question": "What is the age of majority?"}'
```

Retrieval runs under a lock (Qdrant's local mode isn't safe across threads), so concurrent questions reach `Models` one at a time; batching then pays off within a question (30 rerank pairs in one batch) more than across questions. A Qdrant server would lift the lock.

## Develop locally

### Prerequisites

- [uv](https://docs.astral.sh/uv/) **0.12.20**, which is pinned in `pyproject.toml`. uv installs Python 3.14 itself.
- For the data and the tracking store only: an AWS profile with access to the project bucket.
  - Unit tests don't need it. They run on synthetic inputs, including a small PDF generated at test time.
- Optional: an NVIDIA GPU (driver with CUDA 13 support) to embed and rerank faster. See [GPU (optional)](#gpu-optional).

### Setup

```bash
git clone <repo-url> && cd egypt-law-rag
uv sync                      # dev, ingest, embed and CPU-only torch, from uv.lock
cp .env.example .env         # then set RAGLAW_AWS_PROFILE, and the RAGLAW_LLM_* values to answer questions
uv run pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type post-checkout
uv run dvc pull              # fetches data/raw/civil_code.pdf from S3
```

### Run

```bash
uv run pytest -m unit --cov              # what CI runs; no data needed
uv run pytest -m "profile or corpus"     # needs `dvc pull` and `dvc repro`
uv run dvc repro                         # run the pipeline
uv run dvc metrics show                  # each stage's checks and counts
uv run python scripts/compare_retrieval.py   # retrieval runs vs the baseline, question by question
uv run dvc repro evaluate_answers        # RAGAS on /ask's answers; needs RAGLAW_LLM_* and RAGLAW_JUDGE_* in .env
uv run python scripts/compare_answers.py     # answer runs side by side, with refusals, from MLflow
uv run python -m raglaw.retrieval.champion register   # the `champion` run (and its held-out run) becomes the `candidate`
uv run python -m raglaw.retrieval.champion promote    # `production` moves to it, unless its held-out recall@5 is lower
# One MLflow run per stage execution. Log artifacts live in S3, so the UI needs
# the same profile as RAGLAW_AWS_PROFILE in .env (it doesn't read .env itself):
AWS_PROFILE=<your profile> uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Ask a question

`/ask` needs three things running: the `models` service (the embedder and reranker, so the API itself has no torch), the `production` version of the registered retriever with the index it was scored on (`dvc pull`), and an OpenAI-compatible LLM endpoint set in `.env` (`RAGLAW_LLM_BASE_URL`, `RAGLAW_LLM_MODEL`, `RAGLAW_LLM_API_KEY`).

```bash
uv run uvicorn raglaw.serving.models_app:app --port 8001    # models service; loads both models once
uv run uvicorn raglaw.api.main:app --port 8000              # API on http://127.0.0.1:8000/docs
curl -s localhost:8000/health
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"question": "ما هي سن الرشد في القانون المدني؟"}'
# stop the API first (Qdrant's local mode locks the index to one process), then:
uv run python scripts/check_answers.py   # C19: 20 real questions end to end, to docs/reports/answer_check.md
```

`GET /health` reports `degraded`, with the reason per component (`llm`, `models`, `retriever`), until all three are ready, and `documents_indexed` from the index's manifest. It still answers 200, so a container healthcheck stays green. Run the API with one worker: Qdrant's local mode locks the index directory to one process.

### Tracking server (optional)

By default runs and the registry live in the local `mlflow.db`. The shared server keeps them in Postgres and proxies artifacts to the project bucket, so a client needs only its URL (Docker required):

```bash
docker compose --env-file .env -f docker/compose.tracking.yml up -d --build
# in .env: RAGLAW_MLFLOW_TRACKING_URI=http://localhost:5000, RAGLAW_MLFLOW_ARTIFACT_ROOT blank
```

The UI is on http://localhost:5000. The API serves whichever version the `production` alias names, read at startup: moving the alias (`promote`, or `set_registered_model_alias` for a rollback) and restarting the API changes the served config, with no code change and no rebuild.

### GPU (optional)

The embedding and reranking models run on CPU by default, which CI and the Docker image rely on. On CPU, embedding the corpus takes about 15 minutes and reranking about 15 s per question. On a machine with an NVIDIA GPU:

```bash
uv sync --no-group torch-cpu --group torch-gpu   # swaps in torch built for CUDA 13
echo "RAGLAW_DEVICE=cuda" >> .env                # the device is per machine
alias uvg='uv run --no-group torch-cpu --group torch-gpu'
uvg dvc repro                                    # every command goes through uvg
```

A plain `uv run` resyncs the default groups and puts the CPU build back, so on that machine every command goes through `uvg`. A stage set to `cuda` without a GPU fails at startup instead of falling back to CPU. GPU arithmetic differs from CPU in the last digits, so retrieval runs are only compared on one device: `scripts/compare_retrieval.py` leaves out runs on another device than the baseline's.

### Repository layout

```text
egypt-law-rag/
├── compose.yaml            # api, models and ollama (Phase 10)
├── docker/                 # api/ and models/ Dockerfiles, the GPU override, the tracking server
├── dvc.yaml, dvc.lock      # the pipeline, PDF to index to evaluation
├── params.yaml             # tracked parameters, shared with dvc.yaml
├── .env.example            # per-machine and secret values
├── data/                   # DVC-tracked: raw PDF, interim rows, corpus, index, gold sets, repealed ranges
│   └── errata.yaml         # git-tracked: owner-approved fixes
├── src/raglaw/
│   ├── ingest/             # one module per pipeline stage
│   ├── retrieval/          # hybrid retriever, reranker, scoring, the registered champion
│   ├── text/               # Arabic normalization
│   ├── api/                # FastAPI: /ask and /health
│   ├── serving/            # the models service
│   ├── prompts/            # versioned answer prompts
│   └── rag.py, llm.py, answer_eval.py, config.py, schema.py, …
├── scripts/                # comparison reports, the answer check, the repealed-range builder
├── docs/                   # source analysis, normalization, stage metrics, reports
└── tests/                  # pytest; every fixture in conftest.py
```

### Where things live

| Path | What |
| --- | --- |
| `params.yaml` | Tracked parameters (paths, chunking, root heading), shared with `dvc.yaml` |
| `.env` | Per-machine and secret values only (key, bucket, MLflow URI, AWS profile) |
| `dvc.yaml`, `dvc.lock` | The pipeline: `profile → extract → repair → assemble → validate → chunk → embed, bm25 → evaluate_retrieval` |
| `src/raglaw/ingest/` | One module per stage, runnable as `python -m raglaw.ingest.<stage>` |
| `src/raglaw/retrieval/` | Query-time retrieval: tokenizer, article lookup, RRF, the hybrid retriever, the reranker, scoring and the `champion` model |
| `docs/reports/retrieval_misses.md`, `retrieval_comparison.md` | What the current config misses, and every retrieval variant against the baseline |
| `src/raglaw/schema.py`, `records.py` | Record models and their versioned file format |
| `src/raglaw/api/` | The FastAPI service: `/ask` and `/health` |
| `src/raglaw/rag.py`, `src/raglaw/prompts/` | The answer pipeline (retrieve, no-answer gate, prompt, citation check) and its versioned prompts |
| `src/raglaw/serving/models_app.py`, `src/raglaw/retrieval/remote.py` | The `models` service, and the API's HTTP clients for it |
| `src/raglaw/answer_eval.py`, `docs/reports/answer_eval.md` | Answer evaluation with RAGAS (faithfulness, answer relevancy, context precision and recall), and its report |
| `docs/reports/answer_comparison.md`, `ragas_arabic_scorecard.md` | Five answer configs compared, and which RAGAS scores to trust on Arabic answers |
| `docs/reports/answer_check.md` | 20 real questions answered end to end (`scripts/check_answers.py`) |
| `data/errata.yaml` | Owner-approved fixes for one-off source errors |
| `docs/reports/0_source_pdf_analysis.md` | The source facts (P1 to P26) and the rules (R1 to R23) built on them |
| `docs/reports/evaluation_data.md` | How the evaluation data was made, and its limits |
| `docs/normalization.md` | Arabic normalization rules, shared by ingestion and query time |
| `docs/metrics/` | Stage metrics read by `dvc metrics` |
| `data/gold/` | The gold sample and the retrieval eval set (see [Evaluation data](#evaluation-data)) |
| `compose.yaml`, `docker/api/`, `docker/models/`, `docker/compose.gpu.yml` | The containers: `api`, `models` and `ollama`, CPU by default, with a GPU override |
| `docker/compose.tracking.yml`, `docker/mlflow/` | The MLflow tracking server (Postgres and S3 behind it), for maintainers |
| `src/raglaw/retrieval/repealed.py`, `scripts/build_repealed.py`, `data/repealed/` | The repealed-range fallback: the 1948 text used for matching only, and the repeal records |

## Evaluation data

- **Gold sample** (`data/gold/articles_gold.json`): 20 articles transcribed from the printed pages, which `validate` compares the corpus against.
- **Retrieval eval set** (`data/gold/retrieval_eval.jsonl`): 142 Arabic and English questions, each with the articles that govern the answer; 20 are asked in both languages (`pair_id`).
- **Truth point:** both sets were made by an LLM, not a legal professional, and aren't spot-checked yet. Read [docs/reports/evaluation_data.md](docs/reports/evaluation_data.md) before relying on a score computed from them.

## Contributing

### Workflow

1. Branch off `main`, one branch per phase or feature (e.g., `phase-1-corpus`).
2. Write the tests first. The rules in `docs/reports/0_source_pdf_analysis.md` are the spec. Never change an expected value or a threshold to make a test pass. If the data contradicts a rule, stop and raise it with the evidence (page, row index and the extracted text).
3. Commit with a conventional prefix: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
4. Open a PR to `main`. CI must be green.

### What CI checks

| Job | Command | Needs |
| --- | --- | --- |
| lint | `pre-commit run --all-files` | dev group only |
| test | `pytest -m unit --cov` (80% coverage gate) | no S3 access |
| runtime | serves `/health` with `uv sync --no-default-groups` | API dependencies only |
| deliver | `dvc pull` → rebuild the index → retrieval and RAGAS on the 20-question CI set → quality gate (`scripts/ci_gate.py`) → export the production retriever → build both images; pushes them (`sha-<commit>` and `latest`) on merges | The self-hosted GPU runner (label `raglaw-gpu`) with `~/.config/raglaw/ci.env`; never runs for forks |

The gate fails a build when faithfulness on the CI set is below 0.75, or recall@5 on the tuning half loses more than one question against the base branch (`gate:` in `params.yaml`).

The runtime job fails if the API imports anything outside the runtime dependencies. Those include retrieval, the registry client (`mlflow-skinny`) and the LLM client, but not torch: keep `pymupdf`, `torch`, `sentence-transformers` and other pipeline imports out of `raglaw.api`, `raglaw.rag` and `raglaw.schema`. The models live in the `models` service.

### Test markers

| Marker | Runs on | Where |
| --- | --- | --- |
| `unit` | Synthetic inputs, including a PDF generated by `tests/conftest.py` | CI and locally |
| `profile` | The full PDF, via `docs/metrics/source_profile.json` | Locally, after `dvc pull` |
| `corpus` | The built corpus, via the `validate` metrics | Locally, after `dvc repro` |

`profile` and `corpus` tests skip themselves when the PDF isn't pulled. All pytest fixtures go in `tests/conftest.py`, never in test modules.

### Pipeline rules

- **Segment by table rows only.** Article boundaries come from the PDF's table rows, never from a regex over page or document text.
- **Repairs come from a rule or an errata entry.** No other edits to source text. A new errata entry needs owner approval and must match a whole line.
- **Surface anomalies, never absorb them.** Log each one with `log_anomaly` at WARNING, and count it in the stage's metrics.
- **Log, never `print`.** Ruff enforces this (`T20`).
- **Wrap every stage in `stage_run`.** Pass the input hash and the model it writes (`output_model=`), so its MLflow run records both.
- **List a stage's code in its `dvc.yaml` dependencies.** Include `src/raglaw/schema.py` for any stage that writes records, so a code change reruns it.
- **Keep data out of git.** Everything under `data/` is tracked by DVC, except `data/errata.yaml`. The pre-push hook runs `dvc push`.
- **Schema versions are automatic.** A model's version is a fingerprint of its fields, types and constraints (`Record.schema_fingerprint()`). Never set one by hand. Reading a file written under another fingerprint fails, and `dvc repro` rebuilds it.
- **Keep the analysis evidence byte-identical.** Changes to `src/raglaw/ingest/measure.py` must keep `docs/analysis/source_pdf/` (rewritten by `dvc repro profile`) passing `sha256sum -c docs/reports/0_source_pdf_analysis.sha256`. The `profile` tests check this.

### Configuration

- **New project parameter:** add it to `params.yaml` and to `Settings` in `src/raglaw/config.py`.
- **New per-machine or secret value:** add it to `.env.example` and `Settings`.

`tests/test_config.py` fails if a setting lives in both files or in neither.
