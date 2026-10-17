"""Time a generator on /ask's own prompts (Phase 13): latency, time to first token, tokens/sec.

Two steps, so the retrieval models are off the GPU while a generator is timed:

- ``prompts``: retrieve for each answer-eval question with the production
  retriever and keep the chat ``/ask`` would send (``AnswerPipeline.messages_for``;
  refused and repealed-range questions send none), in ``build/bench/prompts.json``.
- ``run``: send those chats to one OpenAI-compatible endpoint, streamed, with
  the harness (``raglaw.bench``: warmup, then timed calls cycling through the
  prompts), and add the row to ``docs/metrics/generator.json`` under ``--label``.

    uv run --no-group torch-cpu --group torch-gpu python scripts/bench_generator.py prompts
    uv run python scripts/bench_generator.py run --label q8-ollama \\
        --base-url http://localhost:11434/v1 --model qwen2.5:7b-instruct-q8_0 --vram-gib 8.9

``--vram-gib`` is read off ``nvidia-smi`` with the server loaded, minus the idle
reading: under WSL, nvidia-smi doesn't attribute memory to processes.
"""

import argparse
import asyncio
import json
import logging
import subprocess
import time
from pathlib import Path

from openai import OpenAI

from raglaw.answer_eval import read_answer_set
from raglaw.bench import benchmark
from raglaw.config import Settings

logger = logging.getLogger("bench_generator")

PROMPTS = Path("build/bench/prompts.json")
WARMUP, ITERATIONS = 5, 130  # two passes over the prompts


def step_prompts(settings: Settings) -> None:
    from raglaw.rag import AnswerPipeline
    from raglaw.retrieval.champion import load_champion
    from raglaw.retrieval.embeddings import huggingface_embeddings
    from raglaw.retrieval.rerank import CrossEncoderReranker

    gold = Path(settings.paths.gold_dir)
    items = read_answer_set(gold / "answer_eval.jsonl", gold / "retrieval_eval.jsonl")
    champion = load_champion(
        settings,
        embeddings=huggingface_embeddings(settings.embedding, settings.device),
        reranker=CrossEncoderReranker(settings.reranker, settings.device),
    )
    pipeline = AnswerPipeline(champion.retrieve, None, settings.answer)  # no LLM call
    prompts = []
    for item in items:
        messages = asyncio.run(pipeline.messages_for(item.question.question))
        if messages is not None:
            prompts.append({"id": item.question.id, "messages": messages})
    champion.close()
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(json.dumps(prompts, ensure_ascii=False, indent=1), "utf-8")
    logger.info(
        "%d of %d questions send a prompt -> %s", len(prompts), len(items), PROMPTS
    )


def _gpu() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"
    return out.stdout.strip().splitlines()[0]


def step_run(settings: Settings, args: argparse.Namespace) -> None:
    prompts = json.loads(PROMPTS.read_text("utf-8"))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)
    first_token, tokens, seconds = [], [], []

    def run(i: int) -> None:
        start = time.perf_counter()
        first, count = None, 0
        stream = client.chat.completions.create(
            model=args.model,
            messages=prompts[i % len(prompts)]["messages"],
            temperature=settings.answer.temperature,
            max_tokens=settings.answer.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        usage = None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage.completion_tokens
            if chunk.choices and chunk.choices[0].delta.content:
                first = first or time.perf_counter()
                count += 1
        if i >= WARMUP:
            first_token.append(((first or time.perf_counter()) - start) * 1000)
            tokens.append(usage or count)
            seconds.append(time.perf_counter() - start)

    timing = benchmark(run, warmup=WARMUP, iterations=args.iterations)
    first_token.sort()
    row = {
        "label": args.label,
        "model": args.model,
        "prompts": len(prompts),
        **timing.to_dict(),
        "ttft_p50_ms": first_token[len(first_token) // 2],
        "tokens_per_second": sum(tokens) / sum(seconds),
        "mean_completion_tokens": sum(tokens) / len(tokens),
        "vram_gib": args.vram_gib,
        "hardware": _gpu(),
        "concurrency": 1,
    }
    out = Path(settings.paths.metrics_dir) / "generator.json"
    rows = json.loads(out.read_text("utf-8")) if out.exists() else []
    rows = [r for r in rows if r["label"] != args.label] + [row]
    out.write_text(json.dumps(rows, indent=2) + "\n", "utf-8")
    logger.info(
        "%s: p50 %.0f ms, p95 %.0f ms, %.1f tokens/s",
        args.label,
        timing.p50_ms,
        timing.p95_ms,
        row["tokens_per_second"],
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="step", required=True)
    sub.add_parser("prompts")
    run = sub.add_parser("run")
    run.add_argument("--label", required=True)
    run.add_argument("--base-url", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--api-key", default="none")
    run.add_argument("--vram-gib", type=float, required=True)
    run.add_argument("--iterations", type=int, default=ITERATIONS)
    args = ap.parse_args()
    settings = Settings()
    if args.step == "prompts":
        step_prompts(settings)
    else:
        step_run(settings, args)


if __name__ == "__main__":
    main()
