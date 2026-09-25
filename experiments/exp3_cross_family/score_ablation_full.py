"""
score_ablation_full.py - Full 2^3 component ablation (C0-C7) for the two
knowledge-poor control models (Ministral-3-8B, Gemma-4-E2B).

Completes the Exp-1 factorial in an unsupported-family regime: all 8 conditions
per model, chrF++ + COMET, paired bootstrap Δ vs C0 for every condition, and the
Exp-1-style summary tables appended to ANALYSIS.md.

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" experiments/exp3_cross_family/score_ablation_full.py
"""

import json
import os
import sys

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

MODELS = [
    ("ministral-3-8b-instruct-2512-bf16", "Ministral-3-8B"),
    ("gemma-4-e2b-it", "Gemma-4-E2B"),
]
CONDS = [
    ("c0_baseline", "C0 baseline (zero-shot)"),
    ("c1_syntax", "C1 + syntax"),
    ("c2_morph", "C2 + morphology"),
    ("c3_dict", "C3 + dictionary"),
    ("c4_syntax_morph", "C4 + syntax + morph"),
    ("c5_syntax_dict", "C5 + syntax + dict"),
    ("c6_morph_dict", "C6 + morph + dict"),
    ("c7_full", "C7 full pipeline"),
]

ECHO_MARKERS = ("### Uyghur Source Sentence:", "### English Translation:",
                "Clause Architecture")


def load_run(path):
    hyps, refs, srcs = [], [], []
    n_len = 0
    toks = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = (r.get("translation") or "").strip()
            hyps.append(t)
            refs.append(r["reference"].strip())
            srcs.append(r["sentence"].strip())
            n_len += r.get("finish_reason") == "length"
            toks += r.get("completion_tokens") or 0
    return {"hyps": hyps, "refs": refs, "srcs": srcs, "n": len(hyps),
            "len_capped": n_len, "avg_out_tok": toks // max(len(hyps), 1)}


def bootstrap_vs_c0(stats_c0, stats_x, n_boot, seed=12345):
    """Paired bootstrap Δ(x − C0) from precomputed segment-stat arrays."""
    rng = np.random.default_rng(seed)
    n = stats_c0.shape[0]
    base = _corpus_from_stats(stats_c0)
    score = _corpus_from_stats(stats_x)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = _corpus_from_stats(stats_x[idx]) - _corpus_from_stats(stats_c0[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return base, score, score - base, lo, hi, p


def main(n_boot=5000):
    chrf = CHRF(word_order=2)

    # ---- load all runs -------------------------------------------------------
    runs, stats = {}, {}
    for tag, name in MODELS:
        for cond, _ in CONDS:
            path = os.path.join(RESULTS, f"{tag}_{cond}.results.jsonl")
            assert os.path.exists(path), f"missing {path} — run translate first"
            run = load_run(path)
            runs[(name, cond)] = run
            stats[(name, cond)] = np.array(
                [_segment_stats(h, r) for h, r in zip(run["hyps"], run["refs"])])
            print(f"[ablation] loaded {name} {cond} (n={run['n']})")

    # ---- chrF++ (validate our stats against corpus_score) ---------------------
    scores = {}
    for key, run in runs.items():
        scores[key] = chrf.corpus_score(run["hyps"], [run["refs"]]).score
        assert abs(scores[key] - _corpus_from_stats(stats[key])) < 1e-9, key

    # ---- COMET (one model load for all runs) ----------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    comet_scores = {}
    for key, run in runs.items():
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(run["srcs"], run["hyps"], run["refs"])]
        comet_scores[key] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[ablation] {key[0]} {key[1]}: chrF++ {scores[key]:.2f} "
              f"COMET {comet_scores[key]:.4f}")
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- paired bootstrap: every condition vs C0 ------------------------------
    deltas = {}
    for _, name in MODELS:
        for cond, _ in CONDS[1:]:
            b, s, d, lo, hi, p = bootstrap_vs_c0(
                stats[(name, "c0_baseline")], stats[(name, cond)], n_boot)
            deltas[(name, cond)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[ablation]   {name} {cond}−C0: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"p={p:.4f}", flush=True)

    # ---- persist ----------------------------------------------------------------
    out = {
        "scores": {f"{n}|{c}": {"chrfpp": round(scores[(n, c)], 2),
                                "comet": round(comet_scores[(n, c)], 4)}
                   for _, n in MODELS for c, _ in CONDS},
        "deltas_vs_c0": {f"{n}|{c}": {"delta": round(v["delta"], 2),
                                       "ci_lo": round(v["lo"], 2),
                                       "ci_hi": round(v["hi"], 2),
                                       "p": round(v["p"], 4)}
                         for (n, c), v in deltas.items()},
        "behavior": {f"{n}|{c}": {"len_capped": runs[(n, c)]["len_capped"],
                                  "avg_out_tok": runs[(n, c)]["avg_out_tok"]}
                     for _, n in MODELS for c, _ in CONDS},
    }
    with open(os.path.join(RESULTS, "exp3_ablation_full.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # ---- markdown tables ---------------------------------------------------------
    lines = ["", "## Full component ablation (C0-C7) — knowledge-poor regime",
             "", "Mirrors Exp 1 (Qwen3.5-9B) on Ministral-3-8B and Gemma-4-E2B.",
             ""]
    for _, name in MODELS:
        lines += [f"### {name}", "",
                  "| Cond | chrF++ | COMET | Δ chrF++ vs C0 [95% CI] | p | len-cap | avg out tok |",
                  "|" + "|".join(["---"] * 7) + "|"]
        for cond, cname in CONDS:
            s = scores[(name, cond)]
            c = comet_scores[(name, cond)]
            b = out["behavior"][f"{name}|{cond}"]
            if cond == "c0_baseline":
                lines.append(f"| {cname} | {s:.2f} | {c:.4f} | — | — | "
                             f"{b['len_capped']} | {b['avg_out_tok']} |")
            else:
                v = deltas[(name, cond)]
                star = " ✱" if v["p"] < 0.05 else ""
                lines.append(f"| {cname} | {s:.2f} | {c:.4f} | "
                             f"{v['delta']:+.2f} [{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | "
                             f"{v['p']:.4f} | {b['len_capped']} | {b['avg_out_tok']} |")
        lines.append("")

    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(RESULTS, "exp3_ablation_full.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[ablation] tables appended to ANALYSIS.md; "
          f"raw data in results/exp3_ablation_full.json + .md")


if __name__ == "__main__":
    main()
