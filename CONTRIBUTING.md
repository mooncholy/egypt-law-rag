# Contributing

How to work on this repository: the workflow, where things live, the rules the pipeline follows, what CI checks, and the maintainer tasks. To run the project, start with the [README](README.md); its history is in [CHANGELOG.md](CHANGELOG.md).

## Workflow

1. Branch off `main`, one branch per phase or feature (e.g., `phase-1-corpus`).
2. Write the tests first. The rules in `docs/reports/0_source_pdf_analysis.md` are the spec. Never change an expected value or a threshold to make a test pass. If the data contradicts a rule, stop and raise it with the evidence (page, row index and the extracted text).
3. Commit with a conventional prefix: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`.
4. Open a PR to `main`. CI must be green.

## Repository layout

```text
egypt-law-rag/
├── compose.yaml            # api, models and ollama: the quickstart stack
├── docker/                 # Dockerfiles, and the GPU, vLLM, canary, monitoring and tracking overrides
├── dvc.yaml, dvc.lock      # the pipeline, PDF to index to evaluation
├── params.yaml             # tracked parameters, shared with dvc.yaml
├── .env.example            # per-machine and secret values
├── data/                   # DVC-tracked: raw PDF, interim rows, corpus, index, gold sets, repealed ranges
│   └── errata.yaml         # git-tracked: owner-approved fixes
├── src/raglaw/
│   ├── ingest/             # one module per pipeline stage
│   ├── retrieval/          # hybrid retriever, reranker, scoring, the registered champion, distillation
│   ├── text/               # Arabic normalization
│   ├── api/                # FastAPI: /ask, /ask/stream, /health, /metrics
│   ├── serving/            # the models service and the BentoML service
│   ├── monitoring/         # metrics, tracing, drift, the PII guardrail
│   ├── prompts/            # versioned answer prompts
│   └── rag.py, llm.py, answer_eval.py, gate.py, bench.py, config.py, schema.py, …
├── scripts/                # comparisons, benchmarks, distillation, the CI gate, re-indexing, trace scoring
├── loadtest/               # Locust at 50 users, and the usage sampler
├── docs/                   # serving and monitoring guides, source analysis, stage metrics, reports
├── reports/                # the load tests' Locust reports
└── tests/                  # pytest; every fixture in conftest.py
```

## Where things live

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
| `data/gold/` | The gold sample and the retrieval eval set (see [Limitations and honesty](README.md#limitations-and-honesty) and [evaluation_data.md](docs/reports/evaluation_data.md)) |
| `compose.yaml`, `docker/api/`, `docker/models/`, `docker/compose.gpu.yml` | The containers: `api`, `models` and `ollama`, CPU by default, with a GPU override |
| `docker/compose.tracking.yml`, `docker/mlflow/` | The MLflow tracking server (Postgres and S3 behind it), for maintainers |
| `src/raglaw/retrieval/repealed.py`, `scripts/build_repealed.py`, `data/repealed/` | The repealed-range fallback: the 1948 text used for matching only, and the repeal records |
| `src/raglaw/serving/bento_service.py` | The BentoML form of the service ([docs/serving.md](docs/serving.md)) |
| `src/raglaw/monitoring/` | Prometheus metrics, Langfuse tracing, query drift and the PII guardrail ([docs/monitoring.md](docs/monitoring.md)) |
| `docker/compose.vllm.yml`, `docker/compose.canary.yml`, `docker/compose.monitoring.yml`, `docker/monitoring/` | The opt-in overrides: vLLM, the nginx canary, and the monitoring stack with its provisioning |
| `loadtest/`, `reports/` | The Locust load test, and its reports |
| `src/raglaw/bench.py`, `scripts/bench_generator.py`, `scripts/distill_reranker.py` | The benchmark harness, the generator benchmark and the reranker distillation (Phase 13) |
| `src/raglaw/gate.py`, `scripts/ci_gate.py` | CI's quality gate |

## Configuration

- **New project parameter:** add it to `params.yaml` and to `Settings` in `src/raglaw/config.py`.
- **New per-machine or secret value:** add it to `.env.example` and `Settings`.

`tests/test_config.py` fails if a setting lives in both files or in neither.

## Pipeline rules

- **Segment by table rows only.** Article boundaries come from the PDF's table rows, never from a regex over page or document text.
- **Repairs come from a rule or an errata entry.** No other edits to source text. A new errata entry needs owner approval and must match a whole line.
- **Surface anomalies, never absorb them.** Log each one with `log_anomaly` at WARNING, and count it in the stage's metrics.
- **Log, never `print`.** Ruff enforces this (`T20`).
- **Wrap every stage in `stage_run`.** Pass the input hash and the model it writes (`output_model=`), so its MLflow run records both.
- **List a stage's code in its `dvc.yaml` dependencies.** Include `src/raglaw/schema.py` for any stage that writes records, so a code change reruns it.
- **Keep data out of git.** Everything under `data/` is tracked by DVC, except `data/errata.yaml`. The pre-push hook runs `dvc push`.
- **Schema versions are automatic.** A model's version is a fingerprint of its fields, types and constraints (`Record.schema_fingerprint()`). Never set one by hand. Reading a file written under another fingerprint fails, and `dvc repro` rebuilds it.
- **Keep the analysis evidence byte-identical.** Changes to `src/raglaw/ingest/measure.py` must keep `docs/analysis/source_pdf/` (rewritten by `dvc repro profile`) passing `sha256sum -c docs/reports/0_source_pdf_analysis.sha256`. The `profile` tests check this.

## Test markers

| Marker | Runs on | Where |
| --- | --- | --- |
| `unit` | Synthetic inputs, including a PDF generated by `tests/conftest.py` | CI and locally |
| `profile` | The full PDF, via `docs/metrics/source_profile.json` | Locally, after `dvc pull` |
| `corpus` | The built corpus, via the `validate` metrics | Locally, after `dvc repro` |

`profile` and `corpus` tests skip themselves when the PDF isn't pulled. All pytest fixtures go in `tests/conftest.py`, never in test modules.

## What CI checks

| Job | Command | Needs |
| --- | --- | --- |
| lint | `pre-commit run --all-files` | dev group only |
| test | `pytest -m unit --cov` (80% coverage gate) | no S3 access |
| runtime | serves `/health` with `uv sync --no-default-groups` | API dependencies only |
| deliver | `dvc pull` → rebuild the index when its inputs changed → retrieval, and RAGAS faithfulness on the 20-question CI set → quality gate (`scripts/ci_gate.py`) → export the production retriever → build both images; pushes them (`sha-<commit>` and `latest`) on merges | The self-hosted GPU runner (label `raglaw-gpu`) with `~/.config/raglaw/ci.env`; never runs for forks |

The gate fails a build when faithfulness on the CI set is below 0.75, or recall@5 on the tuning half loses more than one question against the base branch (`gate:` in `params.yaml`).

The runtime job fails if the API imports anything outside the runtime dependencies. Those include retrieval, the registry client (`mlflow-skinny`) and the LLM client, but not torch: keep `pymupdf`, `torch`, `sentence-transformers` and other pipeline imports out of `raglaw.api`, `raglaw.rag` and `raglaw.schema`. The models live in the `models` service.

### The self-hosted runner

The deliver job runs on a machine with the project's GPU, data access and models. To register one (Linux or WSL, as the user who has `uv`, `docker` and `gh` working):

```bash
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -fsSLo runner.tar.gz https://github.com/actions/runner/releases/download/v2.338.0/actions-runner-linux-x64-2.338.0.tar.gz
echo "af4b794c1bc41d73d40535e3fe092a39f9679cd8d965954c2aca25a05ca41d32  runner.tar.gz" | sha256sum -c
tar xzf runner.tar.gz
./config.sh --url https://github.com/mooncholy/egypt-law-rag \
  --token "$(gh api -X POST repos/mooncholy/egypt-law-rag/actions/runners/registration-token --jq .token)" \
  --name <machine> --labels raglaw-gpu --work _work --unattended
./run.sh    # foreground; WSL has no systemd by default
```

What the job expects on that machine:
- `~/.config/raglaw/ci.env`: the `.env` the job copies in (`RAGLAW_AWS_PROFILE`, the tracking URI, `RAGLAW_LLM_*`, `RAGLAW_JUDGE_*`).
- `~/.aws` with that profile, for `dvc pull`.
- The MLflow tracking server on `localhost:5000` (`docker/compose.tracking.yml`).
- Ollama serving the generator (`qwen2.5:7b-instruct-q8_0`) and the judge (`gemma3:12b`), with `OLLAMA_NUM_PARALLEL=4` so the judge takes RAGAS's 4 calls at once.
- `docker login` for the image push on merges.

## Maintainer tasks

### Build the images

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

### The tracking server

By default runs and the registry live in the local `mlflow.db`. The shared server keeps them in Postgres and proxies artifacts to the project bucket, so a client needs only its URL (Docker required):

```bash
docker compose --env-file .env -f docker/compose.tracking.yml up -d --build
# in .env: RAGLAW_MLFLOW_TRACKING_URI=http://localhost:5000, RAGLAW_MLFLOW_ARTIFACT_ROOT blank
```

The UI is on http://localhost:5000. The API serves whichever version the `production` alias names, read at startup: moving the alias (`promote`, or `set_registered_model_alias` for a rollback) and restarting the API changes the served config, with no code change and no rebuild.
