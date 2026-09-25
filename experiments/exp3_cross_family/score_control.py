"""
score_control.py - Anchor-vs-knowledge decomposition for the Exp 3 control runs.

Context (see ANALYSIS.md, "Honest confound"): weak-family C7 gains may partly
come from *output anchoring* — the structured prompt makes outputs clean —
rather than from the linguistic knowledge itself.

Design: C1 (syntax-only) and C2 (morphology-only) provide the same table-format
anchor as C7 but NO English dictionary glosses. For each control model:

    Δ(C1−C0), Δ(C2−C0)  = anchor effect (+ whatever syntax/morph alone adds)
    Δ(C7−C0)            = total effect
    Δ(C7−C1), Δ(C7−C2)  = knowledge effect (glosses on top of the anchor)

Models: Ministral-3-8B (biggest honest rambler) and Gemma-4-E2B (biggest Δ).
Run in the SCORING env after the control translate phase:

  "$TMPDIR/scoring/bin/python" experiments/exp3_cross_family/score_control.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../common")
from significance import paired_bootstrap_chrf  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MODELS = [
    ("ministral-3-8b-instruct-2512-bf16", "Ministral-3-8B"),
    ("gemma-4-e2b-it", "Gemma-4-E2B"),
]
CONDS = [("c0_baseline", "C0"), ("c1_syntax", "C1"),
         ("c2_morph", "C2"), ("c7_full", "C7")]
PAIRS = [("C0", "C1"), ("C0", "C2"), ("C0", "C7"), ("C1", "C7"), ("C2", "C7")]

ECHO_MARKERS = ("### Uyghur Source Sentence:", "### English Translation:",
                "Clause Architecture")


def load_run(path):
    hyps, refs, srcs = [], [], []
    n_len = n_echo = 0
    toks = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = (r.get("translation") or "").strip()
            hyps.append(t)
            refs.append(r["reference"].strip())
            srcs.append(r["sentence"].strip())
            n_len += r.get("finish_reason") == "length"
            n_echo += any(m in t for m in ECHO_MARKERS)
            toks += r.get("completion_tokens") or 0
    return {"hyps": hyps, "refs": refs, "srcs": srcs, "n": len(hyps),
            "len_capped": n_len, "echo": n_echo, "avg_out_tok": toks // max(len(hyps), 1)}


def main(n_boot=5000):
    # ---- load runs ----------------------------------------------------------
    runs = {}
    for tag, name in MODELS:
        for cond, cname in CONDS:
            path = os.path.join(RESULTS, f"{tag}_{cond}.results.jsonl")
            assert os.path.exists(path), f"missing {path} — run translate first"
            runs[(name, cname)] = load_run(path)
            print(f"[control] loaded {name} {cname} (n={runs[(name, cname)]['n']})")

    # ---- chrF++ (corpus) ----------------------------------------------------
    chrf = CHRF(word_order=2)
    scores = {}
    for (name, cname), run in runs.items():
        scores[(name, cname)] = chrf.corpus_score(run["hyps"], [run["refs"]]).score

    # ---- COMET (single model load, all runs) --------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    comet_scores = {}
    for (name, cname), run in runs.items():
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(run["srcs"], run["hyps"], run["refs"])]
        out = comet_model.predict(data, batch_size=64, gpus=1, progress_bar=False)
        comet_scores[(name, cname)] = float(out.system_score)
        print(f"[control] {name} {cname}: chrF++ {scores[(name, cname)]:.2f} "
              f"COMET {comet_scores[(name, cname)]:.4f}")
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- paired bootstrap ----------------------------------------------------
    print(f"[control] paired bootstrap ({len(MODELS) * len(PAIRS)} pairs, "
          f"{n_boot} resamples)")
    path_of = {(name, cname): os.path.join(RESULTS, f"{tag}_{cond}.results.jsonl")
               for tag, name in MODELS for cond, cname in CONDS}
    deltas = {}
    for _, name in MODELS:
        for a, b in PAIRS:
            sa, sb, d, lo, hi, p = paired_bootstrap_chrf(
                path_of[(name, a)], path_of[(name, b)], n_boot=n_boot)
            deltas[(name, a, b)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[control]   {name} {b}−{a}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.4f}")

    # ---- report ---------------------------------------------------------------
    out = {"scores": {f"{n}|{c}": {"chrfpp": round(scores[(n, c)], 2),
                                   "comet": round(comet_scores[(n, c)], 4)}
                      for _, n in MODELS for _, c in CONDS},
           "deltas": {f"{n}|{b}−{a}": {"delta": round(v["delta"], 2),
                                        "ci_lo": round(v["lo"], 2),
                                        "ci_hi": round(v["hi"], 2),
                                        "p": round(v["p"], 4)}
                      for (n, a, b), v in deltas.items()},
           "behavior": {f"{n}|{c}": {"len_capped": runs[(n, c)]["len_capped"],
                                      "echo": runs[(n, c)]["echo"],
                                      "avg_out_tok": runs[(n, c)]["avg_out_tok"]}
                        for _, n in MODELS for _, c in CONDS}}
    with open(os.path.join(RESULTS, "control_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    lines = ["", "## Control run — anchor vs knowledge decomposition", "",
             "C1 (syntax-only) / C2 (morph-only) carry the table-format anchor but no "
             "glosses;", "Δ(C1/C2−C0) = anchor effect, Δ(C7−C1/C2) = knowledge effect.",
             "", "### Scores", "",
             "| Model | Cond | chrF++ | COMET | len-cap | echo | avg out tok |",
             "|" + "|".join(["---"] * 7) + "|"]
    for _, name in MODELS:
        for _, cname in CONDS:
            b = out["behavior"][f"{name}|{cname}"]
            lines.append(
                f"| {name} | {cname} | {scores[(name, cname)]:.2f} | "
                f"{comet_scores[(name, cname)]:.4f} | {b['len_capped']} | "
                f"{b['echo']} | {b['avg_out_tok']} |")

    lines += ["", "### Paired bootstrap deltas (chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |",
              "|" + "|".join(["---"] * 4) + "|"]
    for _, name in MODELS:
        for a, b in PAIRS:
            v = deltas[(name, a, b)]
            star = " ✱" if v["p"] < 0.05 else ""
            lines.append(f"| {name} | {b} − {a} | {v['delta']:+.2f} "
                         f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")

    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[control] appended control tables to ANALYSIS.md; "
          f"raw data in results/control_results.json")


if __name__ == "__main__":
    main()
