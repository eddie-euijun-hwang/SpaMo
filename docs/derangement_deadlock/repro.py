"""Reproduce the derangement() deadlock from utils/helpers.py.

The buggy implementation uses rejection sampling on element VALUES
(`lst[i] != shuffled[i]`). When the input has enough duplicates that
no value-derangement exists (Hall's condition violated), the
`while True:` loop spins forever.

This script:
  1. Defines the buggy `derangement_buggy` exactly as in utils/helpers.py.
  2. Defines `derangement_fixed` operating on indices instead of values.
  3. Runs both against a battery of inputs in subprocesses with a hard
     timeout, so we can prove the deadlock without actually hanging.
  4. Prints a verdict table.

Run:
    python docs/derangement_deadlock/repro.py
"""

from __future__ import annotations

import multiprocessing as mp
import random
import sys
import time
from typing import Any

TIMEOUT_SECONDS = 3.0


# ---------- the buggy version, copy-pasted from utils/helpers.py ----------
def derangement_buggy(lst):
    if len(lst) <= 1:
        return lst
    while True:
        shuffled = lst[:]
        random.shuffle(shuffled)
        if all(original != shuffled[i] for i, original in enumerate(lst)):
            return shuffled


# ---------- the fix: operate on indices, cap retries, deterministic fallback
def derangement_fixed(lst):
    if len(lst) <= 1:
        return lst
    n = len(lst)
    indices = list(range(n))
    for _ in range(1000):
        random.shuffle(indices)
        if all(indices[i] != i for i in range(n)):
            return [lst[i] for i in indices]
    return lst[1:] + lst[:1]  # cyclic shift — always a derangement on indices


# ---------- subprocess runner with hard timeout ----------
def _worker(fn_name: str, payload, q: mp.Queue):
    fn = {"buggy": derangement_buggy, "fixed": derangement_fixed}[fn_name]
    t0 = time.perf_counter()
    out = fn(payload)
    dt = time.perf_counter() - t0
    q.put((out, dt))


def run_with_timeout(fn_name: str, payload: list[Any], timeout: float):
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_worker, args=(fn_name, payload, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(1.0)
        if p.is_alive():
            p.kill()
            p.join()
        return ("HUNG", float("inf"))
    if q.empty():
        return ("CRASHED", 0.0)
    out, dt = q.get()
    return (out, dt)


# ---------- pathological cases ----------
CASES: list[tuple[str, list[Any]]] = [
    ("two equal strings", ["A", "A"]),
    ("majority duplicate", ["A", "A", "B"]),
    ("two empty strings + one value", ["", "", "X"]),
    ("three identical", ["A", "A", "A"]),
    ("4 elements, value count > n/2", ["A", "A", "A", "B"]),
    ("two empties at edge of batch", ["de1", "", ""]),
    # well-behaved cases to show fix doesn't regress
    ("all distinct, n=3 (OK)", ["A", "B", "C"]),
    ("all distinct, n=8 (OK)", list("ABCDEFGH")),
    ("singleton (no-op)", ["only"]),
    ("empty list (no-op)", []),
]


def fmt_case(payload):
    s = repr(payload)
    return s if len(s) <= 40 else s[:37] + "..."


def main():
    print(f"timeout per call: {TIMEOUT_SECONDS}s\n")
    header = f"{'case':<38} | {'input':<42} | {'buggy':<22} | {'fixed':<22}"
    print(header)
    print("-" * len(header))

    for label, payload in CASES:
        buggy_out, buggy_dt = run_with_timeout("buggy", payload, TIMEOUT_SECONDS)
        fixed_out, fixed_dt = run_with_timeout("fixed", payload, TIMEOUT_SECONDS)

        if buggy_out == "HUNG":
            buggy_str = f"HUNG (>{TIMEOUT_SECONDS:.0f}s)"
        else:
            buggy_str = f"ok in {buggy_dt * 1000:.2f}ms"

        if fixed_out == "HUNG":
            fixed_str = f"HUNG (>{TIMEOUT_SECONDS:.0f}s)"
        else:
            fixed_str = f"ok in {fixed_dt * 1000:.2f}ms"

        print(
            f"{label:<38} | {fmt_case(payload):<42} | {buggy_str:<22} | {fixed_str:<22}"
        )

    print()
    print("Buggy version hangs whenever no permutation has all positions differing")
    print(
        "by VALUE (Hall's condition: any value present in > n/2 positions blocks it)."
    )
    print("Fixed version checks position equality instead, which is always solvable")
    print("for n >= 2; cyclic-shift fallback bounds worst case.")


if __name__ == "__main__":
    # spawn (not fork) so we don't carry CUDA / Lightning state
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
