import json, os, time
import httpx

URL = os.environ.get("EVAL_URL", "http://127.0.0.1:8000") + "/ask"
KEY = os.environ["EVAL_KEY"]          # pass the key from outside, never hard-code it

NOT_FOUND_WORDS = ["not contain", "not found", "don't have", "do not have",
                   "not available", "no information", "does not include", "not in the reports"]


def check(case: dict, answer: str) -> bool:
    """Return True if the answer passes this test case."""
    text = answer.lower()
    # TODO 1: if the case has "expect_not_found" -> pass if ANY phrase from NOT_FOUND_WORDS is in text
    #   hint: case.get("expect_not_found")   and   any(phrase in text for phrase in ...)
    if case.get("expect_not_found"):
        return any(phrase in text for phrase in NOT_FOUND_WORDS)

    # TODO 2: otherwise -> pass if ANY string from case["expect_any"] appears in the answer
    #   (same any(...) pattern)
    else:
        return any(phrase in text for phrase in case["expect_any"])


cases = json.load(open("eval_set.json"))
passed = 0
latencies = []

for case in cases:
    start = time.perf_counter()
    r = httpx.post(URL, json={"question": case["question"]},
                   headers={"x-api-key": KEY}, timeout=180)
    seconds = time.perf_counter() - start
    answer = r.json().get("answer", "")

    # TODO 3: call check(...), add 1 to passed if True, append seconds to latencies
    ok = check(case, answer)
    if ok:
        passed += 1
    latencies.append(seconds)

    print(f"{'PASS' if ok else 'FAIL'}  {seconds:5.1f}s  {case['question']}")
    if not ok:
        print("      got:", answer[:200])

# TODO 4: print the final score, e.g.  "Score: 9/11 (82%)",  and the average latency
percent = passed / len(cases) * 100                 # e.g. 9 / 11 * 100 = 81.8
avg_latency = sum(latencies) / len(latencies)       # total seconds / number of questions

print()                                             # empty line before the summary
print(f"Score: {passed}/{len(cases)} ({percent:.0f}%)")
print(f"Average latency: {avg_latency:.1f}s")