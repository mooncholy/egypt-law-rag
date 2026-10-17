"""Phase 13's one table: baseline, AWQ, the distilled reranker, and both.

Reads what was measured, on the same question sets:

- ``docs/metrics/answers.json`` and ``answers_awq.json``: RAGAS on the 65
  answer-eval questions with the 8-bit and the AWQ generator;
- ``docs/metrics/generator.json``: each generator on /ask's 54 prompts
  (``scripts/bench_generator.py``);
- ``docs/metrics/distillation.json``: each reranker on the 131 in-scope
  retrieval questions (``scripts/distill_reranker.py``).

and rewrites the table between the markers in ``docs/reports/optimization.md``,
leaving the text around it (the decision and the stopping rule) as written.
"""

import json
import re
from pathlib import Path

METRICS = Path("docs/metrics")
REPORT = Path("docs/reports/optimization.md")
START, END = "<!-- table:start -->", "<!-- table:end -->"
TEACHER, STUDENT = "teacher (bge-reranker-v2-m3)", "student, teacher's scores"


def load(name: str):
    path = METRICS / name
    return json.loads(path.read_text("utf-8")) if path.exists() else None


def _f(value, digits=3) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def main() -> None:
    answers = {"q8": load("answers.json"), "awq": load("answers_awq.json")}
    generators = {r["label"]: r for r in load("generator.json") or []}
    rerankers = {r["reranker"]: r for r in load("distillation.json") or []}
    gen_label = {"q8": "q8 (Ollama)", "awq": "AWQ (vLLM)"}
    rows = [
        ("Baseline", "q8", TEACHER),
        ("AWQ", "awq", TEACHER),
        ("Distilled reranker", "q8", STUDENT),
        ("Both", "awq", STUDENT),
    ]
    header = [
        "Row",
        "Generator",
        "Reranker",
        "Faithfulness",
        "Answer relevancy",
        "Context precision",
        "Context recall",
        "Retrieval recall@5",
        "Retrieval CP@5",
        "Rerank p95",
        "Generation p95",
        "Tokens/s",
        "Generator VRAM",
        "Hardware",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for name, gen, reranker in rows:
        g = generators.get(gen_label[gen], {})
        r = rerankers.get(reranker, {})
        # RAGAS ran with the teacher; a student row has no RAGAS of its own.
        a = (answers[gen] or {}).get("overall", {}) if reranker == TEACHER else {}
        lines.append(
            " | ".join(
                [
                    f"| {name}",
                    gen_label[gen],
                    "teacher, 568M" if reranker == TEACHER else "student, 118M",
                    _f(a.get("faithfulness")),
                    _f(a.get("answer_relevancy")),
                    _f(a.get("context_precision")),
                    _f(a.get("context_recall")),
                    _f(r.get("recall_at_5")),
                    _f(r.get("context_precision_at_5")),
                    f"{r['latency'][0]['p95_ms']:.0f} ms" if r else "–",
                    f"{g['p95_ms'] / 1000:.1f} s" if g else "–",
                    _f(g.get("tokens_per_second"), 1),
                    f"{g['vram_gib']:.2f} GiB" if g else "–",
                    g.get("hardware") or r.get("hardware") or "–",
                ]
            )
            + " |"
        )
    table = "\n".join(lines)
    text = REPORT.read_text("utf-8")
    text = re.sub(
        rf"{re.escape(START)}.*?{re.escape(END)}",
        f"{START}\n{table}\n{END}",
        text,
        flags=re.DOTALL,
    )
    REPORT.write_text(text, "utf-8")


if __name__ == "__main__":
    main()
