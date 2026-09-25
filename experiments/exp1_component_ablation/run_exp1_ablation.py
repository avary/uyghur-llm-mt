#!/usr/bin/env python3
"""
run_exp1_ablation.py - Experiment 1: Component Ablation (2^3 = 8 conditions)

Research question: which pipeline component (syntax / morphology / dictionary)
contributes most to Uyghur->English MT quality, and do they interact?

Setup:
  - Anchor model: Qwen/Qwen3.5-9B (thinking disabled, greedy decoding)
  - Data: 997 FLORES-200 dev sentences, prompts rebuilt from the persisted
    analysis in stats/flores_dev/dev.results.jsonl (no re-parsing needed)
  - Conditions C0..C7, see experiments/common/prompt_variants.py

Usage (vLLM env):
  conda run -p "$TMPDIR/vllm" python experiments/exp1_component_ablation/run_exp1_ablation.py
  # smoke test (5 sentences, chrF++ only):
  conda run -p "$TMPDIR/vllm" python experiments/exp1_component_ablation/run_exp1_ablation.py --limit 5 --no-comet

Pause & resume: every completed sentence is flushed to
results/<condition>.results.jsonl; re-running the same command skips completed
work and continues where it stopped. Scoring runs in a separate phase so COMET
never competes with vLLM for GPU memory.

Phases:
  translate : generate missing translations (vLLM)
  score     : chrF++ + COMET over results (scoring venv, subprocess)
  all       : translate, free GPU, then score  [default]
"""

import argparse
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from experiments.common import checkpoint as ckpt
from experiments.common.prompt_variants import (
    CONDITIONS,
    FLORES_RESULTS_PATH,
    build_prompts_for_condition,
    get_condition,
    load_flores_records,
    load_references,
)
from experiments.common import vllm_runner

DEFAULT_MODEL = "Qwen/Qwen3.5-9B"
DEV_REF = os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCORING_PY = os.environ.get("SCORING_PY", os.path.join(os.environ.get("TMPDIR", ""), "scoring", "bin", "python"))
EVALUATOR_PY = os.path.join(_REPO_ROOT, "experiments", "common", "evaluator.py")


def parse_args():
    ap = argparse.ArgumentParser(description="Exp1: component ablation")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--conditions", nargs="+", default=[c.cid for c in CONDITIONS],
                    help="subset of C0..C7 (default: all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only use the first N sentences (smoke tests)")
    ap.add_argument("--phase", choices=["translate", "score", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=128,
                    help="generate+flush in chunks of N sentences")
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--scoring-python", default=SCORING_PY,
                    help="python of the scoring venv (sacrebleu + comet)")
    return ap.parse_args()


def results_path(cond_name: str, model_id: str, limit) -> str:
    """One results file per (condition, model, limit) combination."""
    tag = model_id.split("/")[-1].lower()
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{cond_name}_{tag}{lim}.results.jsonl")


def phase_translate(args, conds):
    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs), \
        f"{len(records)} analysis records vs {len(refs)} references"
    if args.limit:
        records, refs = records[:args.limit], refs[:args.limit]
    n = len(records)
    print(f"[exp1] {n} sentences | conditions: {[c.cid for c in conds]}")

    items = [{"id": i, "sentence": rec["sentence"], "reference": refs[i]}
             for i, rec in enumerate(records)]

    llm = vllm_runner.make_llm(
        args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    t0 = time.time()

    for cond in conds:
        path = results_path(cond.name, args.model, args.limit)
        completed = ckpt.load_completed(path)
        todo = ckpt.pending_items(items, completed)
        status = f"[exp1] {cond.cid} ({cond.name}): {len(completed)}/{n} done"
        if not todo:
            print(status + " — skipping (complete)")
            continue
        print(status + f" — generating {len(todo)}")

        # Build prompts only for pending sentences (dict-only conditions hit
        # the Schwarz DB during construction).
        pending_ids = {it["id"] for it in todo}
        prompts_all = build_prompts_for_condition(cond, records)
        id_to_prompt = {i: prompts_all[i] for i in pending_ids}

        writer = ckpt.JSONLCheckpointWriter(path)
        try:
            for start in range(0, len(todo), args.chunk_size):
                chunk = todo[start:start + args.chunk_size]
                gens = vllm_runner.chat_generate(
                    llm,
                    [id_to_prompt[it["id"]] for it in chunk],
                    max_tokens=args.max_tokens,
                )
                for it, gen in zip(chunk, gens):
                    rec = ckpt.new_record(
                        it["id"], args.model, cond.cid,
                        it["sentence"], it["reference"],
                        id_to_prompt[it["id"]],
                    )
                    writer.write(ckpt.finalize_record(rec, gen))
                done = len(completed) + min(start + args.chunk_size, len(todo))
                print(f"[exp1]   {cond.cid} progress: {done}/{n}")
        finally:
            writer.close()

    vllm_runner.free_llm(llm)
    print(f"[exp1] translate phase finished in {time.time() - t0:.1f}s")


def phase_score(args, conds):
    paths, labels = [], []
    for cond in conds:
        p = results_path(cond.name, args.model, args.limit)
        assert os.path.exists(p), f"missing results file: {p} (run translate first)"
        paths.append(p)
        labels.append(f"{cond.cid} {cond.description}")

    tag = args.model.split("/")[-1].lower()
    lim = f"_n{args.limit}" if args.limit else ""
    out_json = os.path.join(RESULTS_DIR, f"exp1_metrics{lim}.json")
    out_md = os.path.join(RESULTS_DIR, f"exp1_summary{lim}.md")

    cmd = [
        args.scoring_python, EVALUATOR_PY,
        "--results", *paths,
        "--labels", *labels,
        "--out", out_json,
        "--summary", out_md,
        "--title", f"Experiment 1 — Component Ablation ({args.model}, thinking off)",
        "--delta-label", "gain over C0 baseline",
    ]
    if args.no_comet:
        cmd.append("--no-comet")
    else:
        cmd += ["--comet-batch", str(args.comet_batch)]

    print(f"[exp1] scoring via {args.scoring_python}")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    conds = [get_condition(c) for c in args.conditions]

    if args.phase in ("translate", "all"):
        phase_translate(args, conds)
    if args.phase in ("score", "all"):
        phase_score(args, conds)


if __name__ == "__main__":
    main()
