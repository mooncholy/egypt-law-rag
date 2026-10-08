import json
from dataclasses import replace

import pytest

from raglaw.config import Evaluation, Search
from raglaw.ingest.evaluate_retrieval import (
    flatten_metrics,
    misses_report,
    run_evaluate_retrieval,
    run_params,
)
from raglaw.retrieval.retriever import HybridRetriever
from raglaw.retrieval.scoring import Retrieved, split_questions

SEARCH = Search(mode="hybrid", article_lookup=True, candidates=50, rrf_k=60, top_k=10)


def write_questions(path, questions):
    path.write_text(
        "".join(
            json.dumps(
                q.model_dump(exclude_none=True, by_alias=True), ensure_ascii=False
            )
            + "\n"
            for q in questions
        ),
        encoding="utf-8",
    )
    return path


def open_retriever(index, embedding):
    return HybridRetriever.from_index(
        index.dense_dir,
        index.bm25_dir,
        embedding=embedding,
        retrieval=index.retrieval,
        search=SEARCH,
        embeddings=index.embeddings,
    )


# --- Scoring a set of questions ---------------------------------------------------


@pytest.mark.unit
def test_every_question_is_scored_on_what_the_retriever_returned(search_scores):
    by_id = {s.question.id: s for s in search_scores}

    assert by_id["q001"].retrieved.articles[0] == 2
    assert by_id["q002"].retrieved.articles[0] == 3
    assert by_id["q003"].reciprocal_rank == 0.0
    assert by_id["q004"].reciprocal_rank is None  # out of scope


@pytest.mark.unit
def test_the_report_lists_misses_disagreeing_pairs_and_out_of_scope_hits(
    search_scores,
):
    report = misses_report(search_scores, "hybrid, tuning half")

    assert report.startswith("# Retrieval misses: hybrid, tuning half\n")
    assert "## Missed at 5: 1 of 4" in report
    assert "### q003 (rule, en, english)" in report
    assert "- Expected: 99 (not returned)" in report
    assert "## Pairs that disagree on the top article: 0 of 1" in report
    assert "## Out of scope: what came back first (1)" in report
    assert "- q004: Article 3" in report


@pytest.mark.unit
def test_arabic_text_in_the_report_sits_in_code_blocks(search_scores):
    """So right-to-left text never reorders the English around it."""
    first, twin = (s for s in search_scores if s.question.id in ("q001", "q005"))
    disagreeing = replace(twin, retrieved=Retrieved(articles=[1], top_scores={}))

    report = misses_report([first, disagreeing], "x")

    assert "## Pairs that disagree on the top article: 1 of 1" in report
    assert "```text\nالأهلية\n```" in report


@pytest.mark.unit
def test_a_range_of_missing_articles_is_listed_as_one(search_scores, make_question):
    """A repealed range expects 27 articles; the report names them as 54–80."""
    [miss] = [s for s in search_scores if s.question.id == "q003"]
    ranged = replace(
        miss,
        question=make_question("q003", [54, 55, 56, 2, 80], match="any"),
        ranks={54: None, 55: None, 56: None, 2: 1, 80: None},
        hits=dict.fromkeys(miss.hits, False),
    )

    report = misses_report([ranged], "x")

    assert "- Expected: 2 (rank 1); 54–56, 80 (not returned)" in report


# --- Metrics -----------------------------------------------------------------------


@pytest.mark.unit
def test_metrics_flatten_to_numeric_mlflow_names():
    metrics = {
        "split": "tuning",
        "overall": {"recall_at_5": 0.5, "questions": 4},
        "bilingual": {"same_top1": None, "by_kind": {"rule": {"pairs": 1}}},
    }

    assert flatten_metrics(metrics) == {
        "overall.recall_at_5": 0.5,
        "overall.questions": 4,
        "bilingual.by_kind.rule.pairs": 1,
    }


# --- The stage ---------------------------------------------------------------------


@pytest.mark.unit
def test_the_stage_scores_one_half_and_writes_metrics_and_report(
    tmp_path, search_index, embedding_config, search_questions
):
    eval_path = write_questions(tmp_path / "eval.jsonl", search_questions)
    tuning = split_questions(search_questions)["tuning"]

    metrics = run_evaluate_retrieval(
        eval_path,
        search_index.dense_dir,
        search_index.bm25_dir,
        tmp_path / "retrieval.json",
        tmp_path / "misses.md",
        embedding=embedding_config,
        retrieval=search_index.retrieval,
        search=SEARCH,
        evaluation=Evaluation(split="tuning", target_recall_at_5=0.9),
        embeddings=search_index.embeddings,
    )

    assert json.loads((tmp_path / "retrieval.json").read_text("utf-8")) == metrics
    assert metrics["split"] == "tuning"
    assert metrics["questions"] == len(tuning)
    assert metrics["overall"]["questions"] == sum(q.in_scope for q in tuning)
    assert metrics["mode"] == "hybrid"
    assert (tmp_path / "misses.md").read_text("utf-8").startswith("# Retrieval misses")


@pytest.mark.unit
def test_two_runs_on_one_index_give_identical_files(
    tmp_path, search_index, embedding_config, search_questions
):
    """C18 on the stub index; the corpus test checks the real one."""
    eval_path = write_questions(tmp_path / "eval.jsonl", search_questions)
    outputs = []
    for run in ("a", "b"):
        run_evaluate_retrieval(
            eval_path,
            search_index.dense_dir,
            search_index.bm25_dir,
            tmp_path / f"{run}.json",
            tmp_path / f"{run}.md",
            embedding=embedding_config,
            retrieval=search_index.retrieval,
            search=SEARCH,
            evaluation=Evaluation(split="heldout", target_recall_at_5=0.9),
            embeddings=search_index.embeddings,
        )
        outputs.append(
            (
                (tmp_path / f"{run}.json").read_bytes(),
                (tmp_path / f"{run}.md").read_bytes(),
            )
        )

    assert outputs[0] == outputs[1]


@pytest.mark.unit
def test_a_run_is_compared_by_its_whole_retrieval_config(tracking_settings):
    params = run_params(tracking_settings, chunks_sha256="0" * 64)

    assert params == {
        "strategy": "structural",
        "chunk_size": 1000,
        "chunk_overlap": 0,
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": tracking_settings.embedding.revision,
        "document_text": "both_with_headings",
        "bm25_tokenizer": "words",
        "bm25_stopwords": "none",
        "repealed": "per_article",
        "retrieval_mode": "hybrid",
        "article_lookup": True,
        "candidates": 50,
        "rrf_k": 60,
        "top_k": 10,
        "split": "tuning",
        "chunks_sha256": "0" * 64,
    }


# --- Corpus: the real index (C18) ------------------------------------------------------


@pytest.mark.corpus
def test_two_runs_on_the_real_index_give_identical_metrics(tmp_path, repo_root):
    """Loads bge-m3 and scores the tuning half twice (about a minute on CPU)."""
    from raglaw.config import Settings
    from raglaw.retrieval.embeddings import huggingface_embeddings

    settings = Settings(_env_file=None)
    index = repo_root / settings.paths.index_dir
    embeddings = huggingface_embeddings(settings.embedding)
    outputs = []
    for run in ("a", "b"):
        run_evaluate_retrieval(
            repo_root / settings.paths.gold_dir / "retrieval_eval.jsonl",
            index / "dense",
            index / "bm25",
            tmp_path / f"{run}.json",
            tmp_path / f"{run}.md",
            embedding=settings.embedding,
            retrieval=settings.retrieval,
            search=settings.search,
            evaluation=settings.evaluation,
            embeddings=embeddings,
        )
        outputs.append((tmp_path / f"{run}.json").read_bytes())

    assert outputs[0] == outputs[1]
