#!/usr/bin/env python3
"""
run_exp2b_thinking.py - Tier-2 study of Experiment 2: Thinking Mode as an
Orthogonal Ablation

Research question: how much of the MT quality comes from *parameter scale* vs
*inference-time reasoning tokens*? Tier 1 (run_exp2_scaling.py) held the
reasoning regime fixed (thinking OFF) and varied scale; Tier 2 holds scale
fixed and toggles Qwen3.5's native thinking mode ON, then compares against the
already-measured thinking-OFF arm at the same size and condition.

    factor crossed : thinking OFF (Exp 2 results) x thinking ON (this run)
    sizes          : 9B (primary anchor), then one size down (4B) and up (27B)
    conditions     : C0 baseline, C7 full pipeline
    decoding       : greedy (temperature=0) everywhere, exactly as Exp 1/2,
                     so the ON-vs-OFF comparison changes exactly one variable.
                     Note: greedy makes reasoning loops MORE likely; loop rates
                     are a measured outcome here, not a confound.

Stop rule (user-defined): if a model exhausts its token budget in thinking mode
without ever closing the </think> block on a large fraction of sentences
("exhausted without an answer"), the size ladder stops — no smaller sizes are
attempted.

TWO SAMPLING ARMS (--sampling):
  greedy      : temperature=0 — decoding exactly matched to Tier 1/Exp 1, so
                the ONLY changed variable is thinking itself. Preliminary
                smoke tests show this arm non-terminates (the model never
                emits </think>) — the arm quantifies that fragility.
  recommended : temperature=0.6, top_p=0.95, top_k=20 — Qwen's official
                thinking-mode settings; represents what thinking actually
                delivers in practice. Stochastic by nature.

Differences vs Tier 1 runs:
  - max_tokens raised to 4096 (reasoning needs headroom; 512 sufficed OFF)
  - outputs are split into reasoning + answer; both are persisted, so thinking
    length, non-termination, and repetition loops can be analyzed offline

Usage (vLLM env):
  "$TMPDIR/vllm/bin/python" -u experiments/exp2b_thinking_mode/run_exp2b_thinking.py \
      --sampling greedy --models Qwen/Qwen3.5-9B
  "$TMPDIR/vllm/bin/python" -u experiments/exp2b_thinking_mode/run_exp2b_thinking.py \
      --sampling recommended
  # smoke test
  ... --limit 3 --no-comet --sampling recommended
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from typing import Dict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from experiments.common import checkpoint as ckpt
from experiments.common.prompt_variants import (
    FLORES_RESULTS_PATH,
    build_prompts_for_condition,
    get_condition,
    load_flores_records,
    load_references,
)
from experiments.common import vllm_runner

# Order: primary anchor first, then one size down (stop-rule check), then up.
DEFAULT_MODELS = [
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-27B",
]
DEFAULT_CONDITIONS = ["C0", "C7"]

# Thinking-off counterpart scores come from the Tier-1 experiment.
EXP2_METRICS = os.path.join(
    _REPO_ROOT, "experiments", "exp2_model_scaling", "results", "exp2_metrics.json")

DEV_REF = os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCORING_PY = os.environ.get("SCORING_PY", os.path.join(os.environ.get("TMPDIR", ""), "scoring", "bin", "python"))
EVALUATOR_PY = os.path.join(_REPO_ROOT, "experiments", "common", "evaluator.py")

# Stop rule: below this answered-rate we declare the size exhausted and stop
# the ladder (no smaller sizes are attempted anyway; this is a documented gate).
MIN_ANSWERED_RATE = 0.50

# Qwen3 thinking-mode recommended sampling (vendor documentation)
RECOMMENDED_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def parse_args():
    ap = argparse.ArgumentParser(description="Exp2 Tier-2: thinking mode ablation")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", choices=["translate", "score", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=16384,
                    help="generation budget incl. reasoning tokens "
                         "(Qwen3.5 thinks a LOT on Uyghur: ~5-6k tokens typical)")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--sampling", choices=["greedy", "recommended"],
                    default="greedy",
                    help="greedy = matched to Tier-1 decoding; "
                         "recommended = Qwen thinking settings (t=0.6, top_p=0.95, top_k=20)")
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--scoring-python", default=SCORING_PY)
    return ap.parse_args()


def results_path(model_id: str, cond_name: str, limit, sampling: str) -> str:
    tag = model_id.split("/")[-1].lower()
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{tag}_{cond_name}_think-{sampling}{lim}.results.jsonl")


def repetition_flag(text: str, gram: int = 12, thresh: int = 10) -> bool:
    """Crude degenerate-loop detector: any word n-gram repeated >= thresh times.

    Calibrated to flag true runaway repetition, not legitimate deliberation:
    Qwen3.5 thinking revisits candidate phrasings, so repeated n-grams occur
    naturally (a healthy 4k-token trace fires an 8-gram/5 detector spuriously).
    """
    words = text.split()
    if len(words) < gram:
        return False
    grams = [" ".join(words[i:i + gram]) for i in range(len(words) - gram + 1)]
    return max(Counter(grams).values()) >= thresh


def thinking_stats(path: str) -> Dict:
    """Thinking-specific quality/behavior stats for one results file."""
    n = n_answered = n_length = n_rep_reason = n_rep_answer = 0
    reason_chars = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n += 1
            if r.get("answered"):
                n_answered += 1
            if r.get("finish_reason") == "length":
                n_length += 1
            reasoning = r.get("reasoning") or ""
            reason_chars.append(len(reasoning))
            if reasoning and repetition_flag(reasoning):
                n_rep_reason += 1
            if r.get("translation") and repetition_flag(r["translation"]):
                n_rep_answer += 1
    reason_chars.sort()
    return {
        "n": n,
        "answered_rate": round(n_answered / n, 4) if n else None,
        "length_capped": n_length,
        "looping_reasoning": n_rep_reason,
        "looping_answer": n_rep_answer,
        "reasoning_chars_mean": round(sum(reason_chars) / n) if n else None,
        "reasoning_chars_p95": reason_chars[int(0.95 * n) - 1] if n else None,
        "reasoning_chars_max": reason_chars[-1] if n else None,
    }


def run_one_model(args, model_id, conds, records, refs):
    n = len(records)
    items = [{"id": i, "sentence": rec["sentence"], "reference": refs[i]}
             for i, rec in enumerate(records)]

    print(f"\n[exp2b] ===== {model_id} (thinking ON, {args.sampling}) =====", flush=True)
    llm = vllm_runner.make_llm(
        model_id,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    t0 = time.time()

    for cond in conds:
        path = results_path(model_id, cond.name, args.limit, args.sampling)
        completed = ckpt.load_completed(path)
        todo = ckpt.pending_items(items, completed)
        status = f"[exp2b] {model_id} {cond.cid} (think): {len(completed)}/{n} done"
        if not todo:
            print(status + " — skipping (complete)", flush=True)
            continue
        print(status + f" — generating {len(todo)}", flush=True)

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
                    enable_thinking=True,
                    **(RECOMMENDED_SAMPLING if args.sampling == "recommended"
                       else {"temperature": 0.0}),
                )
                for it, gen in zip(chunk, gens):
                    rec = ckpt.new_record(
                        it["id"], model_id, cond.cid,
                        it["sentence"], it["reference"],
                        id_to_prompt[it["id"]],
                        thinking=True,
                    )
                    writer.write(ckpt.finalize_record(rec, gen))
                done = len(completed) + min(start + args.chunk_size, len(todo))
                print(f"[exp2b]   {cond.cid} progress: {done}/{n}", flush=True)
        finally:
            writer.close()

    vllm_runner.free_llm(llm)
    print(f"[exp2b] {model_id} finished in {time.time() - t0:.1f}s (memory freed)",
          flush=True)

    # Stop rule: check answered rate of the just-finished model.
    for cond in conds:
        path = results_path(model_id, cond.name, args.limit, args.sampling)
        st = thinking_stats(path)
        rate = st["answered_rate"] or 0.0
        print(f"[exp2b]   {cond.cid} answered-rate: {rate:.1%} "
              f"(loops in reasoning: {st['looping_reasoning']}, "
              f"length-capped: {st['length_capped']})", flush=True)
        if rate < MIN_ANSWERED_RATE:
            print(f"[exp2b] !! STOP RULE TRIGGERED for {model_id} on {cond.cid}: "
                  f"answered rate {rate:.1%} < {MIN_ANSWERED_RATE:.0%}. "
                  f"Size ladder stops — do not probe smaller models.", flush=True)
            return False
    return True


def phase_translate(args, conds):
    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs)
    if args.limit:
        records, refs = records[:args.limit], refs[:args.limit]
    print(f"[exp2b] {len(records)} sentences | models: {args.models} (thinking ON)",
          flush=True)

    for model_id in args.models:
        ok = run_one_model(args, model_id, conds, records, refs)
        if not ok:
            print("[exp2b] stopping the size ladder (stop rule).", flush=True)
            break


def _load_exp2_scores() -> Dict:
    """Thinking-off Tier-1 scores keyed by '<model> <cid>'."""
    if not os.path.exists(EXP2_METRICS):
        return {}
    with open(EXP2_METRICS, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for r in data.get("runs", []):
        out[r["label"]] = r
    return out


def phase_score(args, conds):
    paths, labels = [], []
    for model_id in args.models:
        for cond in conds:
            p = results_path(model_id, cond.name, args.limit, args.sampling)
            assert os.path.exists(p), f"missing results file: {p}"
            paths.append(p)
            labels.append(f"{model_id.split('/')[-1]} {cond.cid} think-{args.sampling}")

    lim = f"_n{args.limit}" if args.limit else ""
    out_json = os.path.join(RESULTS_DIR, f"exp2b_metrics_{args.sampling}{lim}.json")
    out_md = os.path.join(RESULTS_DIR, f"exp2b_summary_{args.sampling}{lim}.md")

    cmd = [
        args.scoring_python, EVALUATOR_PY,
        "--results", *paths,
        "--labels", *labels,
        "--out", out_json,
        "--summary", out_md,
        "--title", f"Experiment 2, Tier 2 — Thinking ON ({args.sampling} sampling)",
        "--delta-label", "gain over first run (see per-model table below)",
    ]
    if args.no_comet:
        cmd.append("--no-comet")
    else:
        cmd += ["--comet-batch", str(args.comet_batch)]

    print(f"[exp2b] scoring via {args.scoring_python}", flush=True)
    subprocess.run(cmd, check=True)

    # ---- Post-process: thinking stats + Δ(think ON vs OFF) per model×cond ----
    with open(out_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    runs = metrics["runs"]
    tstats = {}
    for path in paths:
        tstats[os.path.basename(path)] = thinking_stats(path)
    metrics["thinking_stats"] = tstats
    metrics["sampling"] = args.sampling

    # Δ vs thinking OFF (Tier-1 exp2), keyed by "<model> <C0|C7>"
    off_scores = _load_exp2_scores()
    deltas = []
    for r in runs:
        label = r["label"]                       # e.g. "Qwen3.5-9B C0 think-greedy"
        base_label = label.replace(f" think-{args.sampling}", "")  # "Qwen3.5-9B C0"
        off = off_scores.get(base_label)
        if off is None:
            continue
        d = {
            "label": base_label,
            "think_chrfpp": r["chrfpp"], "nothink_chrfpp": off["chrfpp"],
            "delta_chrfpp": round(r["chrfpp"] - off["chrfpp"], 2),
        }
        if "comet" in r and "comet" in off:
            d["think_comet"] = r["comet"]
            d["nothink_comet"] = off["comet"]
            d["delta_comet"] = round(r["comet"] - off["comet"], 4)
        deltas.append(d)
    metrics["deltas_vs_thinking_off"] = deltas

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[exp2b] thinking stats + deltas written into {out_json}", flush=True)

    # Append tables to the markdown summary.
    has_comet = any("delta_comet" in d for d in deltas)
    lines = ["", "## Thinking-mode behavior (thinking ON runs)", "",
             "| Run | answered % | length-capped | reasoning loops | "
             "answer loops | reasoning chars (mean/p95/max) |",
             "|" + "|".join(["---"] * 6) + "|"]
    for p in paths:
        st = tstats[os.path.basename(p)]
        label = os.path.basename(p).replace(".results.jsonl", "")
        lines.append(
            f"| {label} | {100 * (st['answered_rate'] or 0):.1f}% | "
            f"{st['length_capped']} | {st['looping_reasoning']} | "
            f"{st['looping_answer']} | {st['reasoning_chars_mean']} / "
            f"{st['reasoning_chars_p95']} / {st['reasoning_chars_max']} |")

    lines += ["", "## Δ thinking ON − thinking OFF (per model × condition)", ""]
    dcols = ["Run", "chrF++ OFF→ON", "Δ chrF++"] + \
            (["COMET OFF→ON", "Δ COMET"] if has_comet else [])
    lines += ["| " + " | ".join(dcols) + " |",
              "|" + "|".join(["---"] * len(dcols)) + "|"]
    for d in deltas:
        row = [d["label"], f"{d['nothink_chrfpp']:.2f} → {d['think_chrfpp']:.2f}",
               ("+" if d["delta_chrfpp"] >= 0 else "") + f"{d['delta_chrfpp']:.2f}"]
        if has_comet:
            row += [f"{d['nothink_comet']:.4f} → {d['think_comet']:.4f}",
                    ("+" if d["delta_comet"] >= 0 else "") + f"{d['delta_comet']:.4f}"]
        lines.append("| " + " | ".join(row) + " |")

    with open(out_md, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2b] behavior + delta tables appended to {out_md}", flush=True)


def main():
    args = parse_args()
    conds = [get_condition(c) for c in args.conditions]

    if args.phase in ("translate", "all"):
        phase_translate(args, conds)
    if args.phase in ("score", "all"):
        phase_score(args, conds)


if __name__ == "__main__":
    main()
