# Build guidelines

The project was built to a phased plan, where each phase ended at a gate of numbered checks. The reports in `docs/reports/` cite that plan by phase number and by check ID (e.g., "C1–C16 pass", "C20 allows 0.02"). This page is the public part of the plan: the rules every phase followed, what each phase built, and what each cited ID asserts.

The source PDF's own IDs live elsewhere. [0_source_pdf_analysis.md](reports/0_source_pdf_analysis.md) lists the measured facts about the PDF (**P1–P35**), the rules built on them (**R1–R31**) and the errata entries (**E1–E33**). [normalization.md](normalization.md) defines the normalization rules (**N**, **F**).

## Ground rules

- **Segment by table rows only.** Article boundaries come from the PDF's table rows, never from a regex over page text (R1).
- **Repairs come from a rule or an errata entry.** Nothing else edits the source text (R10). Errata are in `data/errata.yaml`, approved by the owner.
- **No LLM calls in the pipeline.** The corpus and the index are built deterministically; an LLM appears only at answer time and as the RAGAS judge.
- **Surface anomalies, never absorb them.** Each anomaly is logged at WARNING with its page and row, and counted in the stage's metrics. An accepted exception is named in the report that allows it (e.g., Article 1022's empty Arabic cell under C8).
- **Tests are the spec.** An expected value, threshold or rule is never changed to make a test pass. Unit tests run on synthetic inputs, including a PDF generated at test time, so no source text is kept in git.
- **Tune on one half, report on the other (D15).** The eval questions are split by `id` into a tuning half, used to choose configs, and a held-out half, reported once for the chosen config.
- **Report negative results.** A measured regression with its reason is reported as such (e.g., AWQ and the distilled reranker in Phase 13).

## Phases

| Phase | What it built | Checks | Rubric rows | Reported in |
| --- | --- | --- | --- | --- |
| 0 | Package, config, logging, the DVC remote, MLflow, CI | – | – | – |
| 1 | `profile`: stops `dvc repro` when the PDF differs from the one analyzed | G1–G6 | 05 | `docs/metrics/source_profile.json` |
| 2 | `extract` and `repair`: table rows to text, glyph defects fixed, errata applied | – | 05 | `docs/metrics/extract.json`, `repair.json` |
| 3 | `assemble`: one record per article, with its heading path | U15 | 05 | `docs/reports/heading_tree.txt` |
| 4 | `validate`: re-measures the built corpus and compares it with the gold sample | C1–C14, C16 | 05 | [corpus_validation.md](reports/corpus_validation.md) |
| 5 | `chunk`: articles split at Arabic paragraph markers, never across articles | C15 | 05 | `docs/metrics/chunks.json` |
| 6 | `embed`, `bm25`, `evaluate_retrieval`: the hybrid index, scored before any LLM | C17, C18 | 04, 05 | [retrieval_comparison.md](reports/retrieval_comparison.md), [retrieval_misses.md](reports/retrieval_misses.md) |
| 7 | `/ask`: answers from retrieved articles only, with citations and a no-answer gate | C19 | 01, 02 | [answer_check.md](reports/answer_check.md) |
| 8 | The MLflow tracking server and the registry lifecycle (aliases) | – | 04 | README, Evidence |
| 9 | Answer evaluation with RAGAS, at least 5 configs compared on faithfulness | C20 | 04 | [answer_eval.md](reports/answer_eval.md), [answer_comparison.md](reports/answer_comparison.md), [ragas_arabic_scorecard.md](reports/ragas_arabic_scorecard.md) |
| 10 | Containers, Docker Hub images and the quickstart | – | 03, 10 | README, Quickstart |
| 11 | CI/CD with a faithfulness and recall gate | – | 06 | [CONTRIBUTING.md](../CONTRIBUTING.md#what-ci-checks) |
| 12 | Serving: vLLM, BentoML, streaming, the load test, batch re-indexing, the canary | U38, U39 | 07 | [serving.md](serving.md) |
| 13 | Optimization: AWQ 4-bit and a distilled reranker | – | – | [optimization.md](reports/optimization.md), [distillation.md](reports/distillation.md) |
| 14 | Monitoring: RAGAS trend, Langfuse, drift, token cost, the PII guardrail | U41, U42 | 08 | [monitoring.md](monitoring.md), [drift.md](reports/drift.md) |
| 15 | Release: README, architecture and the rubric cross-check | – | 01, 05, 10 | [rubric_crosscheck.md](reports/rubric_crosscheck.md) |

The rubric rows are the 10-point rubric (Track B) of *The MLOps Practitioner* handbook. Row 09, the peer review, sits outside the repository.

## Check IDs

The letter says what a check runs on:
- **G**: gates in `profile`, on the source PDF.
- **C**: checks on the real corpus, index or answers.
- **U**: unit tests, on synthetic inputs. Only the ones the reports cite are listed here; the rest are named in the test docstrings under `tests/`.

### Source gates (Phase 1)

| ID | Asserts |
| --- | --- |
| G1 | The PDF's md5 equals the md5 in `civil_code.pdf.dvc` |
| G2 | 170 pages, all with a text layer |
| G3 | With `lines_strict`, every page has exactly one table with 2 columns |
| G4 | The fonts are Arial-BoldMT, ArialMT, Calibri and Calibri-Bold |
| G5 | 1,463 detected rows and 1,464 tagged rows; only page 1 differs, by 1 |
| G6 | Raw defects: 3,500 lam-alef signatures, 1,167 right-to-left digit runs, 2 header typos, 6 same-line headers, 13 spaced `مادة` headers |

### Corpus checks (Phases 4 and 5)

| ID | Asserts |
| --- | --- |
| C1 | Article numbers are exactly 1 to 1,149, with no duplicates |
| C2 | Articles 54–80 and 389–417 have `is_repealed: true`, and no other record does |
| C3 | Every non-repealed record has matching English and Arabic article numbers |
| C4 | No anomaly rows are left by `assemble` |
| C5 | Articles + headings + continuations + anomalies = 1,463 rows |
| C6 | No line inside an article body fully matches the header form |
| C7 | Arabic paragraph markers run from `(١)` in sequence within every article |
| C8 | `text_ar` is non-empty for every record, and `heading_path` for every non-repealed record, except the accepted exceptions the report names |
| C9 | All 222 heading rows are consumed by the heading stack or logged as anomalies |
| C10 | Every defect count is 0 after repair |
| C11 | Each repair count equals its G6 baseline |
| C12 | No promulgation-law text, and Article 1's heading path starts with the root heading |
| C13 | Zero Arabic presentation forms, zero control characters other than `\n`, and at most 1% Latin letters in any `text_ar`; `normalize_arabic` is idempotent over the whole corpus |
| C14 | Against the gold sample: article-boundary precision and recall are 1.0, and the Arabic character error rate (`ar_cer`) is at most 0.5% (D2) |
| C15 | Every chunk carries `article_number` and is at most `max_chars` (1,000) characters |
| C16 | Two clean assemblies give the same `corpus_sha256` |

### Index and answer checks (Phases 6, 7 and 9)

| ID | Asserts |
| --- | --- |
| C17 | In `embed.json` and `bm25.json`, `chunks_indexed` equals the chunk count, and `truncated_chunks` is 0 |
| C18 | Two `evaluate_retrieval` runs on the same index give identical metrics |
| C19 | 20 real questions end to end: every `sources` entry has the citation format and names a retrieved article, and the out-of-scope questions get the no-answer reply or are listed |
| C20 | Two `evaluate_answers` runs at temperature 0 give overall faithfulness within 0.02 of each other, so the judge's noise is measured, not assumed |

### Unit tests cited by the reports

| ID | Asserts | Test |
| --- | --- | --- |
| U15 | `normalize_arabic` is idempotent (`f(f(x)) == f(x)`) on synthetic Arabic strings covering every rule in `normalization.md` | `tests/test_normalize.py` |
| U38 | A streamed answer arrives in more than one piece, and the final event carries `sources` | `tests/test_api.py` |
| U39 | Re-indexing a test document raises `documents_indexed` by its chunk count, and the document is retrievable by its number | `tests/test_reindex.py` |
| U41 | The drift score is near 1.0 on the eval set itself and falls on an off-topic set | `tests/test_drift.py` |
| U42 | The PII guardrail redacts a synthetic Egyptian national ID and phone number, and leaves a clean Arabic legal answer unchanged | `tests/test_pii.py` |

## Decisions the reports rely on

| ID | Decision | Answer |
| --- | --- | --- |
| D2 | The gold-sample `ar_cer` threshold (C14) | 0.5% |
| D15 | Tuning versus reporting | The eval questions split by `id` into a tuning half and a held-out half, stratified by `kind` |
| D17 | The recall@5 target | 0.9 on the tuning half (`evaluation.target_recall_at_5`), set by the owner after the 0.569 hybrid baseline. Not met |
| D20 | The generator | Qwen2.5-7B-Instruct in 8 bits (`qwen2.5:7b-instruct-q8_0`) with prompt v2, chosen over the 3B model on faithfulness (0.868 against 0.693) |
| D21 | The no-answer threshold | 0.1 on the rank-1 reranker score (`answer.no_answer_threshold`), set by the owner after counting the in-scope questions each threshold would refuse |
| D23 | The RAGAS judge and reference answers | A different model from the generator, to avoid self-preference (`gpt-oss:120b-cloud`). The reference answers were drafted by an LLM from the governing articles; their limits are in [evaluation_data.md](reports/evaluation_data.md) |
| D26 | How a container gets the retriever without the registry | The `api` image bakes in a copy of the `production` retriever, exported by alias at build, so it needs neither the tracking server nor AWS |
