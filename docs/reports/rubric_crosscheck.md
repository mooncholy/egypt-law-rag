# Rubric and checklist cross-check (Phase 15)

*The MLOps Practitioner*'s 10-point rubric (Track B) and the Project 2 checklist items, line by line, against what this repository shows. "Owner" marks what only the owner can produce: screenshots of their own UI sessions, the peer review, and decisions.

## Rubric

| # | Area | Pass condition | Evidence | State |
| --- | --- | --- | --- | --- |
| 01 | Code and packaging | `pip install -e .` works; the model loads without errors | In a fresh venv (pip 26.2.1, Python 3.14), `pip install -e .` installs. In that venv the API imports, and production version 4 loads from the baked copy and retrieves Article 44 for "What is the age of majority?" (1,150 chunks) | Done |
| 02 | API endpoint | `curl /ask` returns an answer with article citations; Pydantic, `/health`, async handlers | README, "Quickstart": real `/ask` responses with `sources`; `/health` lists every component; `/ask` and `/ask/stream` are async | Done |
| 03 | Docker | `docker compose up && curl localhost:8000/ask` works on the reviewer's machine | `compose.yaml`; images on Docker Hub, pushed by CI on merge (run #44); answered on CPU and GPU here | Done here; **owner**: the Codespaces clean run |
| 04 | MLflow tracking | ≥ 5 runs compared in the UI; the best config in the registry, promoted to production | `retrieval` and `answers` experiments (dozens of runs); `civil-code-retriever` version 4 at `@production`, promoted by `champion promote` | Done; **owner**: the UI screenshot |
| 05 | DVC | `git checkout` + `dvc pull` + `dvc repro` reproduces the result | Every CI deliver run checks out fresh, pulls, rebuilds the index (`embed`, `bm25`) and re-scores retrieval, which must land within one question of the committed score (runs #43–#45 passed) | Index and retrieval: done in CI; **owner**: the full fresh-clone `dvc repro` from the PDF |
| 06 | CI/CD | lint → test → rebuild index → push image on every PR; RAGAS gate | `.github/workflows/ci.yml`: lint, tests (coverage 88%), the self-hosted deliver job with the RAGAS and recall gate; images pushed on merges | Done; **owner**: red/green screenshots |
| 07 | Serving | vLLM + BentoML; p95 documented; Locust report in `reports/` | `docker/compose.vllm.yml` (Qwen2.5-7B-Instruct-AWQ); `raglaw.serving.bento_service`; p95 at 50 users in the README's results and docs/serving.md (180 s Ollama, 101 s vLLM, RTX 5060 Ti); `reports/locust_50_*.html` | Done; **owner**: confirm the latency target |
| 08 | Monitoring | RAGAS faithfulness tracked; Grafana panel; alert threshold set; screenshot in README | `scripts/score_traces.py` puts faithfulness on sampled traces and pushes it; Grafana provisioned with the panel and the 0.80 alert | Done; **owner**: the Grafana, alert and Langfuse screenshots |
| 09 | Peer review | ≥ 300-word review of another project | – | **Owner**, outside this repository |
| 10 | README and architecture | 3-command setup, architecture diagram, changelog; runs without asking the author | README: the three commands, the Mermaid diagram; `CHANGELOG.md` per PR | Done |

## Project 2 checklist

| Item | Evidence | State |
| --- | --- | --- |
| Validated corpus | `validate` stage: C1–C16 pass, Arabic CER 0.03% (`docs/reports/corpus_validation.md`) | Done |
| Chunking experiments with faithfulness | Phase 9 runs in MLflow; faithfulness 0.868 with the chosen config | Done |
| Streaming `/ask` | `/ask/stream` (U38) | Done |
| Batch re-indexing | `scripts/reindex.py` (U39): 1150 → 1151 | Done |
| Canary config | `docker/compose.canary.yml`, nginx 95/5, version headers; stages in docs/serving.md | Done |
| AWQ 4-bit, faithfulness drop ≤ 0.03 | Measured; it dropped 0.052, so it isn't adopted (`docs/reports/optimization.md`) | Done (the rule decided against AWQ) |
| Langfuse | A trace per `/ask` with `retrieve`, `rerank` and `generate` spans, usage and metadata; scores attached by the scheduled evaluation | Done; **owner**: the trace screenshot |
| Cosine drift | U41; the eval set 1.000, family law 0.914, alert line 0.946 (`docs/reports/drift.md`) | Done |
| Token cost | `llm_tokens_total{model,type}`; the cost panel at the measured 44.2 tokens/s and an assumed GPU price | Done; **owner**: set the GPU price |
| PII guardrail | U42; 0 false positives on 2,298 Code texts, 0.046 ms each; a metric per decision | Done |
