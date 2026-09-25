#!/usr/bin/env python3
"""
run_exp2_scaling.py - Experiment 2: Model Scaling (5 dense Qwen3.5 sizes)

Research question: how does model scale interact with in-context linguistic
grounding? For each dense model size we compare:
    baseline (C0) : zero-shot raw sentence
    full     (C7) : full pipeline prompt (syntax + morphology + dictionary)
The gain Δ = full - baseline per size answers the "Equalizer" hypothesis
(small models benefit more) vs the "Context Overload" hypothesis (small models
drown in the analysis table).

Reasoning regime is held constant across sizes: Qwen3.5 thinking mode is
force-disabled everywhere, so parameter scale is the only manipulated variable.

Models (dense roster):
    Qwen/Qwen3.5-0.8B, 2B, 4B, 9B, 27B

Usage (vLLM env):
  conda run -p "$TMPDIR/vllm" python experiments/exp2_model_scaling/run_exp2_scaling.py
  # single model smoke test:
  conda run -p "$TMPDIR/vllm" python experiments/exp2_model_scaling/run_exp2_scaling.py \\
      --models Qwen/Qwen3.5-0.8B --limit 5 --no-comet

Pause & resume: results are appended per sentence to
results/<model>_<condition>.results.jsonl and flushed continuously; re-running
skips everything already on disk. The model roster loop also means an
interrupted 27B run never re-runs the smaller sizes.
"""

import argparse
import json
import os
import subprocess
import sys
import time

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

DEFAULT_MODELS = [
    "Qwen/Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3.5-27B",
]
DEFAULT_CONDITIONS = ["C0", "C7"]  # baseline vs full pipeline

DEV_REF = os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SCORING_PY = os.environ.get("SCORING_PY", os.path.join(os.environ.get("TMPDIR", ""), "scoring", "bin", "python"))
EVALUATOR_PY = os.path.join(_REPO_ROOT, "experiments", "common", "evaluator.py")


def parse_args():
    ap = argparse.ArgumentParser(description="Exp2: model scaling")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                    help="subset of C0..C7 (default: C0 C7)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", choices=["translate", "score", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--scoring-python", default=SCORING_PY)
    return ap.parse_args()


def results_path(model_id: str, cond_name: str, limit) -> str:
    tag = model_id.split("/")[-1].lower()
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{tag}_{cond_name}{lim}.results.jsonl")


def run_one_model(args, model_id, conds, records, refs):
    n = len(records)
    items = [{"id": i, "sentence": rec["sentence"], "reference": refs[i]}
             for i, rec in enumerate(records)]

    print(f"\n[exp2] ===== {model_id} =====")
    llm = vllm_runner.make_llm(
        model_id,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
    )
    t0 = time.time()

    for cond in conds:
        path = results_path(model_id, cond.name, args.limit)
        completed = ckpt.load_completed(path)
        todo = ckpt.pending_items(items, completed)
        status = f"[exp2] {model_id} {cond.cid} ({cond.name}): {len(completed)}/{n} done"
        if not todo:
            print(status + " — skipping (complete)")
            continue
        print(status + f" — generating {len(todo)}")

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
                        it["id"], model_id, cond.cid,
                        it["sentence"], it["reference"],
                        id_to_prompt[it["id"]],
                    )
                    writer.write(ckpt.finalize_record(rec, gen))
                done = len(completed) + min(start + args.chunk_size, len(todo))
                print(f"[exp2]   {cond.cid} progress: {done}/{n}")
        finally:
            writer.close()

    vllm_runner.free_llm(llm)
    print(f"[exp2] {model_id} finished in {time.time() - t0:.1f}s (memory freed)")


def phase_translate(args, conds):
    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs)
    if args.limit:
        records, refs = records[:args.limit], refs[:args.limit]
    print(f"[exp2] {len(records)} sentences | models: {args.models}")

    for model_id in args.models:
        run_one_model(args, model_id, conds, records, refs)


def phase_score(args, conds):
    paths, labels = [], []
    for model_id in args.models:
        for cond in conds:
            p = results_path(model_id, cond.name, args.limit)
            assert os.path.exists(p), f"missing results file: {p}"
            paths.append(p)
            labels.append(f"{model_id.split('/')[-1]} {cond.cid}")
        # blank separator entries handled by evaluator as-is (label ordering)

    lim = f"_n{args.limit}" if args.limit else ""
    out_json = os.path.join(RESULTS_DIR, f"exp2_metrics{lim}.json")
    out_md = os.path.join(RESULTS_DIR, f"exp2_summary{lim}.md")

    cmd = [
        args.scoring_python, EVALUATOR_PY,
        "--results", *paths,
        "--labels", *labels,
        "--out", out_json,
        "--summary", out_md,
        "--title", f"Experiment 2 — Model Scaling ({', '.join(m.split('/')[-1] for m in args.models)}, thinking off)",
        "--delta-label", "full pipeline (C7) gain over zero-shot (C0)",
    ]
    # Delta must be computed against each model's own baseline, not run #1.
    # Evaluator deltas vs. first run are a convenience only; the per-model
    # baseline deltas are recomputed below into a dedicated table.
    if args.no_comet:
        cmd.append("--no-comet")
    else:
        cmd += ["--comet-batch", str(args.comet_batch)]

    print(f"[exp2] scoring via {args.scoring_python}")
    subprocess.run(cmd, check=True)

    # Post-process: per-model Δ(full - baseline) table.
    with open(out_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)["runs"]
    per_model = []
    n_cond = len(conds)
    for i in range(0, len(metrics), n_cond):
        block = metrics[i:i + n_cond]
        base = next((r for r in block if r["label"].endswith("C0")), block[0])
        full = next((r for r in block if r["label"].endswith("C7")), None)
        if full is None:
            continue
        entry = {"model": base["label"].rsplit(" ", 1)[0],
                 "baseline": base, "full": full,
                 "delta_chrfpp": round(full["chrfpp"] - base["chrfpp"], 2)}
        if "comet" in base and "comet" in full:
            entry["delta_comet"] = round(full["comet"] - base["comet"], 4)
        per_model.append(entry)

    merged = dict(metrics=json.dumps({"runs": metrics}))  # keep raw in json
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"runs": metrics, "per_model_deltas": per_model}, f,
                  ensure_ascii=False, indent=2)
    print(f"[exp2] per-model deltas written into {out_json}")

    # Append delta table to the markdown summary.
    has_comet = any("delta_comet" in d for d in per_model)
    lines = ["", "## Δ per model (full C7 − baseline C0)", "",
             "| Model | Baseline chrF++ | Full chrF++ | Δ chrF++ |"
             + (" Baseline COMET | Full COMET | Δ COMET |" if has_comet else "")]
    lines.append("|" + "|".join(["---"] * (4 + (3 if has_comet else 0))) + "|")
    for d in per_model:
        row = [d["model"], f"{d['baseline']['chrfpp']:.2f}",
               f"{d['full']['chrfpp']:.2f}",
               ("+" if d["delta_chrfpp"] >= 0 else "") + f"{d['delta_chrfpp']:.2f}"]
        if has_comet:
            row += [f"{d['baseline']['comet']:.4f}", f"{d['full']['comet']:.4f}",
                    ("+" if d["delta_comet"] >= 0 else "") + f"{d['delta_comet']:.4f}"]
        lines.append("| " + " | ".join(row) + " |")
    with open(out_md, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp2] delta table appended to {out_md}")


def main():
    args = parse_args()
    conds = [get_condition(c) for c in args.conditions]

    if args.phase in ("translate", "all"):
        phase_translate(args, conds)
    if args.phase in ("score", "all"):
        phase_score(args, conds)


if __name__ == "__main__":
    main()
