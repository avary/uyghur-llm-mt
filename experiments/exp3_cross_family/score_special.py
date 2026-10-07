"""
score_special.py - Analysis for the two follow-up studies (conditions S, C3L).

Study 1 (segmentation hypothesis): S = the exact C0 prompt with the sentence
presented as a numbered token list (no roles/lemmas/glosses/guidance).
Contrasts: S−C0 (pure presentation effect) and S−C1 (presentation vs the
best lean single component).

Study 2 (lemma-keyed dictionary): C3L = dict-only table identical to C3 but
glosses from the pipeline's LEMMA-keyed lookups (80.4% coverage) instead of
surface-form lookups (44.0%). Contrasts: C3L−C3 and C3L−C1.

Incidental third contrast: Qwen C3 (re-run on the post-441236a gloss format)
minus Qwen C3 from Exp 1 (pre-441236a format) = the presentation effect of the
improved multi-sense gloss rendering on a knowledge-rich model.

NOTE on data versions: dev.results.jsonl was regenerated in commit 441236a
(Sep 24) with numbered-sense glosses. Exp 1/2 ran on the pre-commit file,
Exp 3 on the post-commit file. Within-study contrasts here are all run on the
post-commit file (Qwen's C3 reference is re-run for exactly this reason), so
every contrast below is version-consistent.

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" experiments/exp3_cross_family/score_special.py
"""

import json
import os
import sys

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.join(HERE, "results")
E1 = os.path.join(HERE, "..", "exp1_component_ablation", "results")
FLORES_ANALYSIS = os.path.join(HERE, "..", "..", "stats", "flores_dev", "dev.results.jsonl")
SURFACE_CACHE = os.path.join(HERE, "..", "common", ".surface_dict_cache.json")

# name -> {cond_label: results file}
RUNS = {
    "Ministral-3-8B": {
        "C0": f"{E3}/ministral-3-8b-instruct-2512-bf16_c0_baseline.results.jsonl",
        "C1": f"{E3}/ministral-3-8b-instruct-2512-bf16_c1_syntax.results.jsonl",
        "C3": f"{E3}/ministral-3-8b-instruct-2512-bf16_c3_dict.results.jsonl",
        "S":  f"{E3}/ministral-3-8b-instruct-2512-bf16_s_segmented.results.jsonl",
        "C3L": f"{E3}/ministral-3-8b-instruct-2512-bf16_c3_dict_lemma.results.jsonl",
    },
    "Gemma-4-E2B": {
        "C0": f"{E3}/gemma-4-e2b-it_c0_baseline.results.jsonl",
        "C1": f"{E3}/gemma-4-e2b-it_c1_syntax.results.jsonl",
        "C3": f"{E3}/gemma-4-e2b-it_c3_dict.results.jsonl",
        "S":  f"{E3}/gemma-4-e2b-it_s_segmented.results.jsonl",
        "C3L": f"{E3}/gemma-4-e2b-it_c3_dict_lemma.results.jsonl",
    },
    "Qwen3.5-9B": {
        "C0": f"{E1}/c0_baseline_qwen3.5-9b.results.jsonl",
        "C1": f"{E1}/c1_syntax_qwen3.5-9b.results.jsonl",
        "C3old": f"{E1}/c3_dict_qwen3.5-9b.results.jsonl",  # pre-441236a glosses
        "C3": f"{E3}/qwen3.5-9b_c3_dict.results.jsonl",    # re-run, post-441236a
        "S":  f"{E3}/qwen3.5-9b_s_segmented.results.jsonl",
        "C3L": f"{E3}/qwen3.5-9b_c3_dict_lemma.results.jsonl",
    },
    "Llama-3.2-3B": {
        "C0": f"{E3}/llama-3.2-3b-instruct_c0_baseline.results.jsonl",
        "C1": f"{E3}/llama-3.2-3b-instruct_c1_syntax.results.jsonl",
        "C7": f"{E3}/llama-3.2-3b-instruct_c7_full.results.jsonl",
        "S":  f"{E3}/llama-3.2-3b-instruct_s_segmented.results.jsonl",
    },
    "Llama-3.2-1B": {
        "C0": f"{E3}/llama-3.2-1b-instruct_c0_baseline.results.jsonl",
        "C1": f"{E3}/llama-3.2-1b-instruct_c1_syntax.results.jsonl",
        "C7": f"{E3}/llama-3.2-1b-instruct_c7_full.results.jsonl",
        "S":  f"{E3}/llama-3.2-1b-instruct_s_segmented.results.jsonl",
    },
}

CONTRASTS = [
    # (model(s), a, b, description)
    ("ALL",        "C0", "S",  "S − C0 (presentation only)"),
    ("SEG_REF",    "C1", "S",  "S − C1 (presentation vs roles)"),
    ("LEMMA",      "C3", "C3L", "C3L − C3 (lemma-keyed dict)"),
    ("LEMMA_REF",  "C1", "C3L", "C3L − C1 (lemma-dict vs roles)"),
    ("QWEN",       "C3old", "C3", "C3(new gloss fmt) − C3(old fmt)"),
    ("LLAMA1B",    "C0", "C1",  "C1 − C0 (lean prompt at the capacity floor)"),
]


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
            n_echo += t.startswith(("Here is", "Here's", "Sure", "Of course"))
            toks += r.get("completion_tokens") or 0
    return {"hyps": hyps, "refs": refs, "srcs": srcs, "n": len(hyps),
            "len_capped": n_len, "preamble": n_echo,
            "avg_out_tok": toks // max(len(hyps), 1)}


def bootstrap_between(stats_a, stats_b, n_boot, seed=12345):
    rng = np.random.default_rng(seed)
    n = stats_a.shape[0]
    sa, sb = _corpus_from_stats(stats_a), _corpus_from_stats(stats_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (_corpus_from_stats(stats_b[idx])
                     - _corpus_from_stats(stats_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return sa, sb, sb - sa, lo, hi, p


def lookup_analysis():
    """Lemma- vs surface-keyed dictionary coverage on FLORES dev."""
    cache = json.load(open(SURFACE_CACHE, encoding="utf-8"))
    n = lh = sh = both = 0
    examples = []
    with open(FLORES_ANALYSIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for t in r["per_token"]:
                if t["pos"] == "Punctuation":
                    continue
                n += 1
                is_lh = t["has_dict"]
                sc = cache.get(t["form"])
                is_sh = sc is not None and sc["gloss"] is not None
                lh += is_lh
                sh += is_sh
                both += is_lh and is_sh
                if is_lh and not is_sh and len(examples) < 3:
                    examples.append((t["form"], t["lemma"], t["gloss"]))
    return {"n": n, "lemma_hit": lh, "surface_hit": sh, "both": both,
            "examples": examples}


def main(n_boot=5000):
    chrf = CHRF(word_order=2)

    # ---- load all runs ------------------------------------------------------
    runs, stats = {}, {}
    for name, conds in RUNS.items():
        for cl, path in conds.items():
            assert os.path.exists(path), f"missing {path} — run translate first"
            run = load_run(path)
            runs[(name, cl)] = run
            stats[(name, cl)] = np.array(
                [_segment_stats(h, r) for h, r in zip(run["hyps"], run["refs"])])
            print(f"[special] loaded {name} {cl} (n={run['n']})")

    # ---- chrF++ (validate stats path) ---------------------------------------
    scores = {}
    for key, run in runs.items():
        scores[key] = chrf.corpus_score(run["hyps"], [run["refs"]]).score
        assert abs(scores[key] - _corpus_from_stats(stats[key])) < 1e-9, key

    # ---- COMET ---------------------------------------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    comet_scores = {}
    for key, run in runs.items():
        data = [{"src": s, "mt": h, "ref": r}
                for s, h, r in zip(run["srcs"], run["hyps"], run["refs"])]
        comet_scores[key] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[special] {key[0]} {key[1]}: chrF++ {scores[key]:.2f} "
              f"COMET {comet_scores[key]:.4f}", flush=True)
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- paired bootstrap contrasts ------------------------------------------
    contrasts = []
    # S − C0 for all models
    for name in RUNS:
        contrasts.append((name, "C0", "S", "S − C0 (presentation only)"))
    # S − C1 where C1 exists
    for name in RUNS:
        if "C1" in RUNS[name] and "S" in RUNS[name]:
            contrasts.append((name, "C1", "S", "S − C1 (presentation vs roles)"))
    # lemma-dict contrasts
    for name in ["Ministral-3-8B", "Gemma-4-E2B", "Qwen3.5-9B"]:
        contrasts.append((name, "C3", "C3L", "C3L − C3 (lemma-keyed dict)"))
        contrasts.append((name, "C1", "C3L", "C3L − C1 (lemma-dict vs roles)"))
    # Qwen gloss-format effect
    contrasts.append(("Qwen3.5-9B", "C3old", "C3",
                      "C3(new gloss fmt) − C3(old fmt)"))
    # capacity floor: lean prompt at 1B
    contrasts.append(("Llama-3.2-1B", "C0", "C1", "C1 − C0 (floor: lean prompt)"))
    contrasts.append(("Llama-3.2-1B", "C1", "C7", "C7 − C1 (floor: full prompt)"))

    deltas = {}
    for name, a, b, desc in contrasts:
        sa, sb, d, lo, hi, p = bootstrap_between(
            stats[(name, a)], stats[(name, b)], n_boot)
        deltas[(name, a, b)] = {"desc": desc, "delta": d, "lo": lo, "hi": hi, "p": p}
        print(f"[special]   {name} {desc}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
              f"p={p:.4f}", flush=True)

    # ---- offline lookup analysis ----------------------------------------------
    look = lookup_analysis()

    # ---- persist ---------------------------------------------------------------
    out = {
        "scores": {f"{n}|{c}": {"chrfpp": round(scores[(n, c)], 2),
                                 "comet": round(comet_scores[(n, c)], 4)}
                   for n in RUNS for c in RUNS[n]},
        "behavior": {f"{n}|{c}": {"len_capped": runs[(n, c)]["len_capped"],
                                   "preamble": runs[(n, c)]["preamble"],
                                   "avg_out_tok": runs[(n, c)]["avg_out_tok"]}
                     for n in RUNS for c in RUNS[n]},
        "contrasts": {f"{n}|{b}−{a}": {"desc": v["desc"],
                                        "delta": round(v["delta"], 2),
                                        "ci_lo": round(v["lo"], 2),
                                        "ci_hi": round(v["hi"], 2),
                                        "p": round(v["p"], 4)}
                      for (n, a, b), v in deltas.items()},
        "lookup_analysis": {
            "n_content_tokens": look["n"],
            "lemma_hit_rate": round(look["lemma_hit"] / look["n"], 4),
            "surface_hit_rate": round(look["surface_hit"] / look["n"], 4),
            "both_hit_rate": round(look["both"] / look["n"], 4),
            "lemma_only_rate": round((look["lemma_hit"] - look["both"]) / look["n"], 4),
            "surface_only_rate": round((look["surface_hit"] - look["both"]) / look["n"], 4),
            "lemma_only_examples": [
                {"form": f, "lemma": l, "gloss": g} for f, l, g in look["examples"]],
        },
    }
    with open(os.path.join(E3, "exp3_special.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown ----------------------------------------------------------------
    lines = ["", "## Follow-up studies — segmentation (S) and lemma-keyed dictionary (C3L)", ""]
    for name in RUNS:
        lines += [f"### {name}", "",
                  "| Cond | chrF++ | COMET | len-cap | preamble | avg out tok |",
                  "|" + "|".join(["---"] * 6) + "|"]
        for cl in RUNS[name]:
            s, c = scores[(name, cl)], comet_scores[(name, cl)]
            b = out["behavior"][f"{name}|{cl}"]
            lines.append(f"| {cl} | {s:.2f} | {c:.4f} | {b['len_capped']} | "
                         f"{b['preamble']} | {b['avg_out_tok']} |")
        lines.append("")

    lines += ["### Paired bootstrap contrasts (chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |",
              "|" + "|".join(["---"] * 4) + "|"]
    for (name, a, b), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        lines.append(f"| {name} | {v['desc']} | {v['delta']:+.2f} "
                     f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")

    la = out["lookup_analysis"]
    lines += ["", "### Dictionary lookup coverage (FLORES dev, 17,159 content tokens)",
              "",
              f"- lemma-keyed hit rate: **{100*la['lemma_hit_rate']:.1f}%** "
              f"(near-advisory extra 9.2%)",
              f"- surface-keyed hit rate: **{100*la['surface_hit_rate']:.1f}%**",
              f"- every surface hit is also a lemma hit "
              f"(surface-only: {100*la['surface_only_rate']:.1f}%; "
              f"lemma-only: +{100*la['lemma_only_rate']:.1f} points)",
              "",
              "Lemma-only examples (surface lookup missed, lemma lookup hit):"]
    for ex in la["lemma_only_examples"]:
        lines.append(f"  - `{ex['form']}` → lemma `{ex['lemma']}` → \"{ex['gloss']}\"")

    with open(os.path.join(E3, "exp3_special.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[special] tables appended to ANALYSIS.md; raw data in "
          f"results/exp3_special.json + .md")


if __name__ == "__main__":
    main()
