#!/usr/bin/env python3
"""
run_exp3_cross_family.py - Experiment 3: Cross-Family Generalization Sweep

Research question: does the in-context linguistic pipeline (C7 full prompt)
help model families WITHOUT official Uyghur support — where the pipeline is
potentially the only source of linguistic knowledge? Qwen3.5 (official
Uyghur support) showed gains only at <=4B (Exp 2); if the "grounding replaces
missing parametric knowledge" hypothesis is right, unsupported families
should show significant C7-over-C0 gains at EVERY size.

Protocol: identical to Exp 2 Tier-1 so results are directly comparable:
  - C0 zero-shot vs C7 full pipeline prompt (from the persisted FLORES dev
    analysis), 997 sentences
  - greedy decoding (temperature=0), max_tokens=512, NO thinking mode
    (Instruct variants only; Ministral Reasoning and Qwen thinking excluded)
  - chrF++ + COMET scoring in the separate scoring env
  - per-model paired bootstrap CIs on ΔchrF++ (experiments/common/significance.py)
  - flush-per-record checkpoints; re-running resumes where it stopped

Roster (dense, ungated; Llama via unsloth full-precision mirrors of the
official weights since meta-llama repos require manual license approval):

  Ministral-3 (no Uyghur support):  3B, 8B, 14B   (Instruct-2512)
  Llama (no Uyghur support):        3.2-1B, 3.2-3B, 3.1-8B
  Gemma-4 (Uyghur exposure unclear): E2B, E4B, 12B, 31B   (it; 26B-A4B MoE skipped)

Usage (vLLM env):
  "$TMPDIR/vllm/bin/python" -u experiments/exp3_cross_family/run_exp3_cross_family.py
  # single-family smoke test:
  "$TMPDIR/vllm/bin/python" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
      --models unsloth/Llama-3.2-1B-Instruct --limit 5 --no-comet
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
    # family, size-ordered; smallest first so stop-rule style checks are cheap
    # NOTE: Ministral ships FP8 checkpoints natively (A100 lacks FP8 cores) —
    # we use the official *-BF16 variants instead. Llama via the official
    # gated meta-llama repos (HF token required); all Instruct.
    "meta-llama/Llama-3.2-1B-Instruct",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "meta-llama/Llama-3.2-3B-Instruct",
    "google/gemma-4-E2B-it",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "google/gemma-4-E4B-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-4-12B-it",
    "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    "google/gemma-4-31B-it",
]
DEFAULT_CONDITIONS = ["C0", "C7"]

DEV_REF = os.path.join(_REPO_ROOT, "flores200_dataset", "eng_Latn.dev")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
EXP2_METRICS = os.path.join(_REPO_ROOT, "experiments", "exp2_model_scaling",
                            "results", "exp2_metrics.json")
SCORING_PY = os.environ.get("SCORING_PY", os.path.join(os.environ.get("TMPDIR", ""), "scoring", "bin", "python"))
EVALUATOR_PY = os.path.join(_REPO_ROOT, "experiments", "common", "evaluator.py")

# pretty display names for the comparison table
FAMILY = {
    "meta-llama/Llama-3.2-1B-Instruct": ("Llama 3.2", "Llama-3.2-1B"),
    "meta-llama/Llama-3.2-3B-Instruct": ("Llama 3.2", "Llama-3.2-3B"),
    "meta-llama/Llama-3.1-8B-Instruct": ("Llama 3.1", "Llama-3.1-8B"),
    "mistralai/Ministral-3-3B-Instruct-2512-BF16": ("Ministral 3", "Ministral-3-3B"),
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": ("Ministral 3", "Ministral-3-8B"),
    "mistralai/Ministral-3-14B-Instruct-2512-BF16": ("Ministral 3", "Ministral-3-14B"),
    "google/gemma-4-E2B-it": ("Gemma 4", "Gemma-4-E2B"),
    "google/gemma-4-E4B-it": ("Gemma 4", "Gemma-4-E4B"),
    "google/gemma-4-12B-it": ("Gemma 4", "Gemma-4-12B"),
    "google/gemma-4-31B-it": ("Gemma 4", "Gemma-4-31B"),
}

# Mistral-3 weights hit a torch-inductor compile bug in this torch/vLLM combo
# ("auto_functionalized was not removed"); if it re-appears with the -BF16
# checkpoints, add the ids here to run them with enforce_eager.
EAGER_MODELS: set = set()


def parse_args():
    ap = argparse.ArgumentParser(description="Exp3: cross-family sweep")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--phase", choices=["translate", "score", "all"], default="all")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--chunk-size", type=int, default=128)
    ap.add_argument("--no-comet", action="store_true")
    ap.add_argument("--comet-batch", type=int, default=64)
    ap.add_argument("--scoring-python", default=SCORING_PY)
    ap.add_argument("--n-boot", type=int, default=5000,
                    help="paired bootstrap resamples for ΔchrF++ CIs")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="force enforce_eager for ALL models (default: only "
                         "architectures with known inductor bugs)")
    return ap.parse_args()


def model_tag(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def results_path(model_id: str, cond_name: str, limit) -> str:
    lim = f"_n{limit}" if limit else ""
    return os.path.join(RESULTS_DIR, f"{model_tag(model_id)}_{cond_name}{lim}.results.jsonl")


def run_one_model(args, model_id, conds, records, refs):
    n = len(records)
    items = [{"id": i, "sentence": rec["sentence"], "reference": refs[i]}
             for i, rec in enumerate(records)]

    print(f"\n[exp3] ===== {model_id} =====", flush=True)
    eager = args.enforce_eager or model_id in EAGER_MODELS
    if eager:
        print("[exp3]   enforce_eager=ON (inductor bug workaround)", flush=True)
    llm = vllm_runner.make_llm(
        model_id,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=eager,
    )
    t0 = time.time()

    for cond in conds:
        path = results_path(model_id, cond.name, args.limit)
        completed = ckpt.load_completed(path)
        todo = ckpt.pending_items(items, completed)
        status = f"[exp3] {model_id} {cond.cid} ({cond.name}): {len(completed)}/{n} done"
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
                )
                for it, gen in zip(chunk, gens):
                    rec = ckpt.new_record(
                        it["id"], model_id, cond.cid,
                        it["sentence"], it["reference"],
                        id_to_prompt[it["id"]],
                    )
                    writer.write(ckpt.finalize_record(rec, gen))
                done = len(completed) + min(start + args.chunk_size, len(todo))
                print(f"[exp3]   {cond.cid} progress: {done}/{n}", flush=True)
        finally:
            writer.close()

    vllm_runner.free_llm(llm)
    print(f"[exp3] {model_id} finished in {time.time() - t0:.1f}s (memory freed)",
          flush=True)


def phase_translate(args, conds):
    records = load_flores_records(FLORES_RESULTS_PATH)
    refs = load_references(DEV_REF)
    assert len(records) == len(refs)
    if args.limit:
        records, refs = records[:args.limit], refs[:args.limit]
    print(f"[exp3] {len(records)} sentences | {len(args.models)} models | "
          f"conditions: {[c.cid for c in conds]}", flush=True)

    for model_id in args.models:
        run_one_model(args, model_id, conds, records, refs)


def phase_score(args, conds):
    # score only models whose results exist (a failed/skipped download must
    # not poison the whole scoring phase)
    scored_models = []
    for model_id in args.models:
        if all(os.path.exists(results_path(model_id, c.name, args.limit))
               for c in conds):
            scored_models.append(model_id)
        else:
            print(f"[exp3] WARNING: incomplete results for {model_id} — excluded "
                  f"from scoring", flush=True)

    paths, labels = [], []
    for model_id in scored_models:
        for cond in conds:
            p = results_path(model_id, cond.name, args.limit)
            paths.append(p)
            fam, name = FAMILY.get(model_id, ("?", model_tag(model_id)))
            labels.append(f"{name} {cond.cid}")

    lim = f"_n{args.limit}" if args.limit else ""
    out_json = os.path.join(RESULTS_DIR, f"exp3_metrics{lim}.json")
    out_md = os.path.join(RESULTS_DIR, f"exp3_summary{lim}.md")

    cmd = [
        args.scoring_python, EVALUATOR_PY,
        "--results", *paths,
        "--labels", *labels,
        "--out", out_json,
        "--summary", out_md,
        "--title", "Experiment 3 — Cross-Family Sweep (C0 vs C7, greedy, no thinking)",
        "--delta-label", "C7 vs C0 within run list (see per-model table)",
    ]
    if args.no_comet:
        cmd.append("--no-comet")
    else:
        cmd += ["--comet-batch", str(args.comet_batch)]

    print(f"[exp3] scoring via {args.scoring_python}", flush=True)
    subprocess.run(cmd, check=True)

    # ---- post-processing: per-model Δ + paired bootstrap CIs ----------------
    # (bootstrap runs in the scoring env via subprocess; sacrebleu is not
    #  installed in the vLLM env)
    sig_specs = []
    sig_index = {}
    for model_id in args.models:
        fam, name = FAMILY.get(model_id, ("?", model_tag(model_id)))
        p0 = results_path(model_id, "c0_baseline", args.limit)
        p7 = results_path(model_id, "c7_full", args.limit)
        if os.path.exists(p0) and os.path.exists(p7):
            sig_index[name] = len(sig_specs)
            sig_specs.append({"a": p0, "b": p7, "name": f"{name} C7 vs C0",
                              "n_boot": args.n_boot})

    sig_results = []
    if sig_specs:
        sig_cli = os.path.join(_REPO_ROOT, "experiments", "common",
                               "significance_cli.py")
        spec_path = os.path.join(RESULTS_DIR, f".sig_specs{lim}.json")
        with open(spec_path, "w", encoding="utf-8") as f:
            json.dump(sig_specs, f)
        print(f"[exp3] paired bootstrap via {args.scoring_python} "
              f"({len(sig_specs)} comparisons, {args.n_boot} resamples each)",
              flush=True)
        proc = subprocess.run([args.scoring_python, sig_cli, spec_path],
                               capture_output=True, text=True)
        print(proc.stderr, end="", flush=True)
        proc.check_returncode()
        sig_results = json.loads(proc.stdout)

    with open(out_json, encoding="utf-8") as f:
        metrics = json.load(f)
    runs = {r["label"]: r for r in metrics["runs"]}

    per_model = []
    for model_id in scored_models:
        fam, name = FAMILY.get(model_id, ("?", model_tag(model_id)))
        if name not in sig_index:
            continue
        sig = sig_results[sig_index[name]]
        r0 = runs.get(f"{name} C0", {})
        r7 = runs.get(f"{name} C7", {})
        entry = {
            "model": name, "family": fam, "model_id": model_id,
            "c0_chrfpp": sig["score_a"], "c7_chrfpp": sig["score_b"],
            "delta_chrfpp": sig["delta"],
            "ci_lo": sig["ci_lo"], "ci_hi": sig["ci_hi"],
            "p_value": sig["p_value"],
            "significant": sig["p_value"] < 0.05,
        }
        if "comet" in r0 and "comet" in r7:
            entry["c0_comet"] = r0["comet"]
            entry["c7_comet"] = r7["comet"]
            entry["delta_comet"] = round(r7["comet"] - r0["comet"], 4)
        per_model.append(entry)
        print(f"[exp3]   {name}: ΔchrF++(C7−C0) = {sig['delta']:+.2f} "
              f"[{sig['ci_lo']:+.2f}, {sig['ci_hi']:+.2f}] p={sig['p_value']:.4f}",
              flush=True)

    metrics["per_model_deltas"] = per_model

    # ---- Qwen reference rows from Exp 2 ---------------------------------------
    qwen_rows = []
    if os.path.exists(EXP2_METRICS):
        with open(EXP2_METRICS, encoding="utf-8") as f:
            exp2 = json.load(f)
        for d in exp2.get("per_model_deltas", []):
            qwen_rows.append(d)  # has model, baseline, full, delta_chrfpp, delta_comet
    metrics["qwen_reference"] = qwen_rows

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[exp3] per-model deltas + Qwen reference written into {out_json}",
          flush=True)

    # ---- Markdown: cross-family comparison table ------------------------------
    has_comet = any("delta_comet" in d for d in per_model)
    lines = ["", "## Cross-family comparison — Δ(C7 − C0) per model",
             "", "Paired bootstrap 95% CIs on chrF++ (resampled sentence sets).",
             ""]
    cols = ["Family", "Model", "C0 chrF++", "C7 chrF++", "Δ chrF++ [95% CI]", "p"] + \
           (["Δ COMET"] if has_comet else [])
    lines += ["| " + " | ".join(cols) + " |",
              "|" + "|".join(["---"] * len(cols)) + "|"]
    for d in per_model:
        star = " ✱" if d["significant"] else ""
        row = [d["family"], d["model"], f"{d['c0_chrfpp']:.2f}", f"{d['c7_chrfpp']:.2f}",
               f"{d['delta_chrfpp']:+.2f} [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]{star}",
               f"{d['p_value']:.4f}"]
        if has_comet and "delta_comet" in d:
            row.append(("+" if d["delta_comet"] >= 0 else "") + f"{d['delta_comet']:.4f}")
        lines.append("| " + " | ".join(row) + " |")

    if qwen_rows:
        lines += ["", "### Reference: Qwen3.5 (official Uyghur support) from Exp 2", ""]
        qcols = ["Model", "C0 chrF++", "C7 chrF++", "Δ chrF++"] + \
                (["Δ COMET"] if any("delta_comet" in q for q in qwen_rows) else [])
        lines += ["| " + " | ".join(qcols) + " |",
                  "|" + "|".join(["---"] * len(qcols)) + "|"]
        for q in qwen_rows:
            row = [q["model"],
                   f"{q['baseline']['chrfpp']:.2f}", f"{q['full']['chrfpp']:.2f}",
                   ("+" if q["delta_chrfpp"] >= 0 else "") + f"{q['delta_chrfpp']:.2f}"]
            if "delta_comet" in q and "delta_comet" in qcols:
                row.append(("+" if q["delta_comet"] >= 0 else "") + f"{q['delta_comet']:.4f}")
            lines.append("| " + " | ".join(row) + " |")

    with open(out_md, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp3] comparison table appended to {out_md}", flush=True)


def main():
    args = parse_args()
    conds = [get_condition(c) for c in args.conditions]

    if args.phase in ("translate", "all"):
        phase_translate(args, conds)
    if args.phase in ("score", "all"):
        phase_score(args, conds)


if __name__ == "__main__":
    main()
