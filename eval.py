"""Run the evaluation set (eval_set.json) against a deployed /ask endpoint.

Usage:  EVAL_KEY=... [EVAL_URL=https://...] python eval.py
"""

import json
import os
import time

import httpx

URL = os.environ.get("EVAL_URL", "http://127.0.0.1:8000") + "/ask"
KEY = os.environ["EVAL_KEY"]  # passed in from the environment, never hard-coded

# Phrases that count as a correct refusal for questions the reports can't answer.
NOT_FOUND_WORDS = ["not contain", "not found", "don't have", "do not have",
                   "not available", "no information", "does not include", "not in the reports"]


def check(case: dict, answer: str) -> bool:
    """Return True if the answer passes this test case."""
    text = answer.lower()
    if case.get("expect_not_found"):
        return any(phrase in text for phrase in NOT_FOUND_WORDS)
    return any(phrase in text for phrase in case["expect_any"])


def main() -> None:
    """Ask every question, print PASS/FAIL per case, then the score and average latency."""
    with open("eval_set.json") as f:
        cases = json.load(f)
    passed = 0
    latencies = []

    for case in cases:
        start = time.perf_counter()
        r = httpx.post(URL, json={"question": case["question"]},
                       headers={"x-api-key": KEY}, timeout=180)
        seconds = time.perf_counter() - start
        answer = r.json().get("answer", "")

        ok = check(case, answer)
        if ok:
            passed += 1
        latencies.append(seconds)

        print(f"{'PASS' if ok else 'FAIL'}  {seconds:5.1f}s  {case['question']}")
        if not ok:
            print("      got:", answer[:200])

    percent = passed / len(cases) * 100
    avg_latency = sum(latencies) / len(latencies)

    print()
    print(f"Score: {passed}/{len(cases)} ({percent:.0f}%)")
    print(f"Average latency: {avg_latency:.1f}s")


if __name__ == "__main__":
    main()
