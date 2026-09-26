#!/usr/bin/env python3
"""Reproducer with a criterion for the split-KV composed-address fault (Xid 31).

What it reproduces
------------------
With the split-KV verify kernel enabled (SPEC_ATTN=1, i.e. VLLM_SPEC_DECODE_ATTN=1),
the FA2 backend and speculative decoding on, a long prompt followed by repeated
same-prefix sends faults the GPU inside run_fullgraph().replay(). The fault is
in _spec_attn_partial: its KV gather composed an address from strides that are frozen
at CUDA-graph capture, and nothing bounded the composed address, only the indices.

Criterion
---------
BEFORE the fix (pre-0d3ef11995 / this port): the server dies within round 14 of the
loop below, with 'illegal memory access' in the engine log and health leaving 200.
Observed five consecutive times, byte-identical stop point, on cmp170x CMP 170HX.

AFTER the fix: 0 crashes in >= 25 rounds (150 requests), health stays 200, and the
engine log contains no 'illegal memory access'. 25 rounds / 150 requests is the
number the fix was accepted against.

Usage
-----
    python3 bench/spec-attn-crash-repro.py --port 18001 [--rounds 25]
It sends real completions; it asserts only the crash criterion, not output quality
(that is bench_probe.py / bench_needle.py in bench/allover326/).
"""
import argparse
import json
import sys
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=18001)
ap.add_argument("--rounds", type=int, default=25)
ap.add_argument("--model", default=None)
args = ap.parse_args()

BASE = f"http://127.0.0.1:{args.port}"


def health():
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=10).read()
        return 200
    except Exception:
        return 0


def models():
    try:
        return json.load(urllib.request.urlopen(f"{BASE}/v1/models", timeout=10))["data"][0]["id"]
    except Exception:
        return None


WORDS = ("system kernel memory buffer thread process socket packet register cache "
         "pointer allocate schedule interrupt virtual physical address translate "
         "compile execute branch predict pipeline vector matrix tensor").split()


def prompt(n, seed):
    import random
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(int(n / 1.3)))


def send(n_chars, max_tokens=48, timeout=900):
    model = args.model or models()
    body = json.dumps({
        "model": model,
        "prompt": prompt(n_chars, 4242 + n_chars),
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,                    # keep every send comparable
        "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for _ in r:
            pass


def main():
    if health() != 200:
        sys.exit("server not healthy; start it first")
    print(f"# split-KV crash reproducer on port {args.port}, {args.rounds} rounds")
    print("# criterion: 0 crashes in all rounds, health 200 at the end")
    # 96k / 120k / 96k / 60k x3 per round: the 120k send is what re-uses the
    # prefix cache and lands the verify batch on the faulting path.
    seq = [96000, 120000, 96000, 60000, 60000, 60000]
    t0 = time.time()
    for r in range(1, args.rounds + 1):
        for n in seq:
            try:
                send(n)
            except Exception as e:
                print(f"r{r} chars={n} FAILED: {str(e)[:90]}")
                sys.exit(1)
            if health() != 200:
                print(f"r{r} chars={n} >>> SERVER DOWN (crash criterion hit)")
                sys.exit(1)
        print(f"  round {r} ok  ({time.time()-t0:.0f}s)")
    print("PASS: 0 crashes, health 200 throughout")


if __name__ == "__main__":
    main()
