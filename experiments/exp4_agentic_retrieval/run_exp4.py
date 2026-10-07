#!/usr/bin/env python3
"""
run_exp4.py - Experiment 4 driver: agentic retrieval (pull) vs static prompts.

Modes:
  phase0    P0-a/P0-b: tool-calling reliability per family (small, unconstrained
            safety caps, log-only) + P0-c budget/turn-cap calibration stats.
  translate full agentic runs: models x conditions (E4-A full tool set,
            E4-B lean: gloss + word_list), checkpointed per episode.
  score     (see score_exp4.py — run separately in the scoring env)

Usage (vLLM env):
  "$TMPDIR/vllm/bin/python" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode phase0
  "$TMPDIR/vllm/bin/python" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from experiments.common.prompt_variants import (  # noqa: E402
    FLORES_RESULTS_PATH, load_flores_records, load_references,
)
from experiments.common import checkpoint as ckpt  # noqa: E402
from experiments.common import vllm_runner  # noqa: E402
from experiments.exp4_agentic_retrieval.tool_server import (  # noqa: E402
    TOOLS_SCHEMA, LEAN_TOOLS, load_server,
)
from experiments.exp4_agentic_retrieval.agent_loop import (  # noqa: E402
    AgentRunner, episode_record,
)

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-9B",
    "google/gemma-4-E4B-it",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "meta-llama/Llama-3.1-8B-Instruct",
]

DEV_REF = os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CONDITIONS = {
    "E4-A": {"tools": TOOLS_SCHEMA},
    "E4-B": {"tools": LEAN_TOOLS},
}

# guardrails calibrated in Phase 0 (see DESIGN.md / ANALYSIS.md): log-only
# ceilings observed natural usage up to 32 calls; 24 covers selective agents
# and truncates only extreme over-fetchers (which is itself measured data)
DEFAULTS = {"max_calls": 24, "max_rounds": 8, "max_tokens": 512}
# phase-0 safety ceilings (log-only observation)
P0 = {"episodes": 30, "max_calls": 32, "max_rounds": 12, "max_tokens": 1024}


def results_path(model_id: str, condition: str, limit=None) -> str:
    tag = model_id.split("/")[-1].lower()
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{tag}_{condition.lower()}{lim}.results.jsonl")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["phase0", "translate"], required=True)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    choices=list(CONDITIONS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-calls", type=int, default=DEFAULTS["max_calls"])
    ap.add_argument("--max-rounds", type=int, default=DEFAULTS["max_rounds"])
    ap.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    ap.add_argument("--p0-episodes", type=int, default=P0["episodes"])
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=32768)
    return ap.parse_args()


def run_model_condition(args, model_id, condition, records, refs, server,
                        caps, tag="", n_total=None):
    n = n_total if n_total is not None else len(records)
    if args.limit and n_total is None:
        records, refs = records[:args.limit], refs[:args.limit]
        n = len(records)

    path = results_path(model_id, condition + (tag or ""), args.limit)
    completed = ckpt.load_completed(path)
    todo_idx = [i for i in range(n) if i not in completed]
    print(f"[exp4] {model_id} {condition}{tag}: {n - len(todo_idx)}/{n} "
          f"episodes done" + (f" — running {len(todo_idx)}" if todo_idx else ""),
          flush=True)
    if not todo_idx:
        return

    runner = AgentRunner(
        llm=_LLM, server=server, tools=CONDITIONS[condition]["tools"],
        model_id=model_id, **caps)

    writer = ckpt.JSONLCheckpointWriter(path)
    CHUNK = 128
    t0 = time.time()
    try:
        for start in range(0, len(todo_idx), CHUNK):
            batch_idx = todo_idx[start:start + CHUNK]
            eps = [runner.new_episode(i, records[i]["sentence"], refs[i])
                   for i in batch_idx]
            done = runner.run(eps)
            for ep in done:
                writer.write(episode_record(ep, model_id, condition))
            print(f"[exp4]   {model_id} {condition}: "
                  f"{start + len(batch_idx)}/{len(todo_idx)} episodes "
                  f"({time.time() - t0:.0f}s)", flush=True)
    finally:
        writer.close()


def phase0_report(model_id, path):
    stats = Counter()
    calls = rounds = n_eps = 0
    tool_mix = Counter()
    finish = Counter()
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            n_eps += 1
            calls += r["n_calls"]
            rounds += r["n_rounds"]
            tool_mix.update(r["tool_mix"])
            finish[r["finish"]] += 1
            stats["parse_failures"] += r["n_parse_failures"]
            stats["unresolved"] += r["n_unresolved"]
    import numpy as np
    call_counts = [json.loads(l)["n_calls"]
                   for l in open(path, encoding="utf-8")]
    round_counts = [json.loads(l)["n_rounds"]
                    for l in open(path, encoding="utf-8")]
    print(f"\n[phase0] {model_id}: {n_eps} episodes")
    print(f"  parseable episodes (>=1 completed turn): "
          f"{n_eps - stats['parse_failures']}/{n_eps}")
    print(f"  calls/episode: mean {calls/max(n_eps,1):.1f} "
          f"p50 {np.percentile(call_counts,50):.0f} "
          f"p95 {np.percentile(call_counts,95):.0f} max {max(call_counts, default=0)}")
    print(f"  rounds/episode: mean {rounds/max(n_eps,1):.1f} "
          f"p95 {np.percentile(round_counts,95):.0f} max {max(round_counts, default=0)}")
    print(f"  tool mix: {dict(tool_mix)}")
    print(f"  finish: {dict(finish)}  "
          f"parse_failures: {stats['parse_failures']}  "
          f"unresolved: {stats['unresolved']}")


_LLm = None


def main():
    global _LLM
    args = parse_args()

    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs)
    server = load_server()

    if args.mode == "phase0":
        caps = {"max_calls": P0["max_calls"], "max_rounds": P0["max_rounds"],
                "max_tokens": P0["max_tokens"], "context_limit": args.max_model_len}
        n0 = min(args.p0_episodes, len(records))
        recs0, refs0 = records[:n0], refs[:n0]
        for model_id in args.models:
            print(f"\n[phase0] ===== {model_id} (thinking OFF for Qwen, "
                  f"greedy, {n0} episodes, log-only caps) =====", flush=True)
            _LLM = vllm_runner.make_llm(model_id, max_model_len=args.max_model_len,
                                         gpu_memory_utilization=args.gpu_mem_util)
            run_model_condition(args, model_id, "E4-A", recs0, refs0, server,
                                caps, tag="_p0", n_total=n0)
            vllm_runner.free_llm(_LLM)
            _LLM = None
            phase0_report(model_id, results_path(model_id, "E4-A_p0"))
        return

    # translate mode
    caps = {"max_calls": args.max_calls, "max_rounds": args.max_rounds,
            "max_tokens": args.max_tokens, "context_limit": args.max_model_len}
    for model_id in args.models:
        print(f"\n[exp4] ===== {model_id} =====", flush=True)
        _LLM = vllm_runner.make_llm(model_id, max_model_len=args.max_model_len,
                                     gpu_memory_utilization=args.gpu_mem_util)
        for condition in args.conditions:
            run_model_condition(args, model_id, condition, records, refs,
                                server, caps)
        vllm_runner.free_llm(_LLM)
        _LLM = None


if __name__ == "__main__":
    main()
