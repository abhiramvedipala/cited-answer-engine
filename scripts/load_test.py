"""Asyncio load test for POST /ask: N concurrent users, each sends M sequential questions.

  uv run python -m scripts.load_test --users 20 --per-user 5 --url http://localhost:8000

Start the server with a high RATE_LIMIT, otherwise the per-IP limiter returns 429 (correctly).
Reports time-to-first-token and total latency p50/p95, error count and total cost.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from eval.metrics import percentile


async def one(client: httpx.AsyncClient, url: str, question: str) -> dict:
    t0 = time.perf_counter()
    first = None
    cost = 0.0
    try:
        async with client.stream("POST", f"{url}/ask", json={"question": question}) as r:
            if r.status_code != 200:
                return {"ok": False, "status": r.status_code}
            event = ""
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                elif line.startswith("data: "):
                    if event == "token" and first is None:
                        first = time.perf_counter() - t0
                    elif event == "done":
                        cost = json.loads(line[6:])["cost_usd"]
    except httpx.HTTPError as e:
        return {"ok": False, "status": type(e).__name__}
    return {
        "ok": True,
        "ttft": first or time.perf_counter() - t0,
        "total": time.perf_counter() - t0,
        "cost": cost,
    }


async def user(client, url, questions):
    return [await one(client, url, q) for q in questions]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--per-user", type=int, default=5)
    args = ap.parse_args()

    qs = [
        json.loads(x)["question"]
        for x in Path("eval/questions.jsonl").read_text().splitlines()
        if x
    ]
    rng = random.Random(1)
    plans = [rng.sample(qs, args.per_user) for _ in range(args.users)]
    async with httpx.AsyncClient(timeout=120) as client:
        t0 = time.perf_counter()
        results = [
            r
            for rs in await asyncio.gather(*(user(client, args.url, p) for p in plans))
            for r in rs
        ]
        wall = time.perf_counter() - t0
    ok = [r for r in results if r["ok"]]
    print(
        f"users={args.users} requests={len(results)} ok={len(ok)} "
        f"errors={len(results) - len(ok)} wall={wall:.1f}s throughput={len(ok) / wall:.2f}/s"
    )
    for key in ("ttft", "total"):
        xs = [r[key] for r in ok]
        print(f"{key}: p50={percentile(xs, 50):.2f}s p95={percentile(xs, 95):.2f}s")
    total = sum(r["cost"] for r in ok)
    print(f"cost: ${total:.4f} total, ${total / len(ok):.6f}/req")


if __name__ == "__main__":
    asyncio.run(main())
