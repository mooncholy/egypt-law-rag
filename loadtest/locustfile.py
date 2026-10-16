"""Load test for /ask (Phase 12): 50 users asking the eval set's real questions.

Each user picks a question at random from the retrieval eval set (142 Arabic
and English questions, out-of-scope ones included), so no single question is
repeated enough to hit a cache, and waits 1 to 3 s between requests; about 5%
of requests are /health. Needs ``dvc pull`` for the questions.

Run headless against a running stack:
    uv run locust -f loadtest/locustfile.py --headless -u 50 -r 5 -t 5m \\
        --host http://localhost:8000 --html reports/locust_50.html
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

QUESTIONS = [
    json.loads(line)["question"]
    for line in Path("data/gold/retrieval_eval.jsonl").read_text("utf-8").splitlines()
    if line.strip()
]


class Asker(HttpUser):
    wait_time = between(1, 3)

    @task(19)
    def ask(self) -> None:
        question = random.choice(QUESTIONS)
        # One name for every question, so the report aggregates them.
        self.client.post("/ask", json={"question": question}, name="/ask", timeout=180)

    @task(1)
    def health(self) -> None:
        self.client.get("/health", name="/health")
