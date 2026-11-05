"""
score_exp4_think.py - Exp 4 follow-up: does thinking mode improve agentic
retrieval (tool selection) even though it hurt direct translation (Exp 2b)?

Conditions compared (Qwen3.5-9B, Gemma-4-E4B):
  C0    zero-shot static           (greedy)
  C3L   lean lemma-keyed push      (greedy)
  E4-A  agentic, thinking OFF      (greedy)
  E4-AS agentic, thinking OFF      (recommended sampling — deconfounds sampling)
  E4-AT agentic, thinking ON       (recommended sampling)

Clean contrasts: E4-AT − E4-AS = thinking effect; E4-AS − E4-A = sampling
effect. Behavior: calls/episode, tool mix, think length, length caps,
budget-forcing, calibration precision/recall (H4).

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" experiments/exp4_agentic_retrieval/score_exp4_think.py
"""

import json
import os
import sys

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402
from score_exp4 import behavior, calibration, needed_words  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "results")
E1 = os.path.join(HERE, "..", "exp1_component_ablation", "results")
E3 = os.path.join(HERE, "..", "exp3_cross_family", "results")
FLORES_ANALYSIS = os.path.join(HERE, "..", "..", "stats", "flores_dev", "dev.results.jsonl")

MODELS = [
    ("qwen3.5-9b", "Qwen3.5-9B",
     f"{E1}/c0_baseline_qwen3.5-9b.results.jsonl",
     f"{E3}/qwen3.5-9b_c3_dict_lemma.results.jsonl"),
    ("gemma-4-e4b-it", "Gemma-4-E4B",
     f"{E3}/gemma-4-e4b-it_c0_baseline.results.jsonl",
     f"{E3}/gemma-4-e4b-it_c3_dict_lemma.results.jsonl"),
]
CONDS = ["C0", "C3L", "E4-A", "E4-AS", "E4-AT"]
CONTRASTS = [
    ("E4-A", "E4-AS", "E4-AS − E4-A (sampling effect)"),
    ("E4-AS", "E4-AT", "E4-AT − E4-AS (thinking effect, clean)"),
    ("E4-A", "E4-AT", "E4-AT − E4-A (thinking+sampling combined)"),
    ("C3L", "E4-AT", "E4-AT − C3L (thinking pull vs best push)"),
]


def load_run(path):
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
    return hyps, refs


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


def main(n_boot=5000):
    chrf = CHRF(word_order=2)
    records = [json.loads(l) for l in open(FLORES_ANALYSIS, encoding="utf-8")]

    runs, recs_map = {}, {}
    for tag, name, c0_path, c3l_path in MODELS:
        for cond, path in [("C0", c0_path), ("C3L", c3l_path),
                           ("E4-A", f"{E4}/{tag}_e4-a.results.jsonl"),
                           ("E4-AS", f"{E4}/{tag}_e4-as.results.jsonl"),
                           ("E4-AT", f"{E4}/{tag}_e4-at.results.jsonl")]:
            assert os.path.exists(path), f"missing {path}"
            hyps, refs = load_run(path)
            stats = np.array([_segment_stats(h, r) for h, r in zip(hyps, refs)])
            runs[(name, cond)] = (hyps, refs, stats)
            if cond.startswith("E4"):
                recs_map[(name, cond)] = [json.loads(l)
                                          for l in open(path, encoding="utf-8")]
        print(f"[think-score] loaded {name}: C0/C3L/E4-A/E4-AS/E4-AT")

    # ---- COMET ---------------------------------------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    comet_scores = {}
    for key, (hyps, refs, stats) in runs.items():
        data = [{"src": records[i]["sentence"], "mt": h, "ref": r}
                for i, (h, r) in enumerate(zip(hyps, refs))]
        comet_scores[key] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[think-score] {key[0]} {key[1]}: "
              f"chrF++ {chrf.corpus_score(hyps, [refs]).score:.2f} "
              f"COMET {comet_scores[key]:.4f}", flush=True)
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- contrasts ------------------------------------------------------------
    deltas = {}
    for tag, name, _, _ in MODELS:
        for a, b, desc in CONTRASTS:
            sa, sb, d, lo, hi, p = bootstrap_between(
                runs[(name, a)][2], runs[(name, b)][2], n_boot)
            deltas[(name, a, b, desc)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[think-score] {name} {desc}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"p={p:.4f}", flush=True)

    # ---- behavior + calibration -------------------------------------------------
    beh, cal = {}, {}
    for tag, name, c0_path, _ in MODELS:
        c0_hyps = runs[(name, "C0")][0]
        needed = needed_words(c0_hyps, records, None)
        for cond in ("E4-A", "E4-AS", "E4-AT"):
            recs = recs_map[(name, cond)]
            beh[f"{name}|{cond}"] = behavior(recs)
            cal[f"{name}|{cond}"] = calibration(
                recs, needed, None)
            beh[f"{name}|{cond}"]["think_chars_mean"] = (
                sum(r.get("think_chars", 0) for r in recs) // len(recs))
            beh[f"{name}|{cond}"]["length_caps"] = sum(
                r.get("n_length_caps", 0) for r in recs)

    with open(os.path.join(E4, "exp4_think_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "scores": {f"{n}|{c}": {"chrfpp": round(
                           chrf.corpus_score(runs[(n, c)][0],
                                             [runs[(n, c)][1]]).score, 2),
                           "comet": round(comet_scores[(n, c)], 4)}
                       for _, n, _, _ in MODELS for c in CONDS},
            "contrasts": {f"{n}|{b}−{a}": {"desc": d,
                                            "delta": round(v["delta"], 2),
                                            "ci_lo": round(v["lo"], 2),
                                            "ci_hi": round(v["hi"], 2),
                                            "p": round(v["p"], 4)}
                          for (n, a, b, d), v in deltas.items()},
            "behavior": beh, "calibration": cal,
        }, f, indent=2, ensure_ascii=False)

    # ---- markdown ----------------------------------------------------------------
    lines = ["", "## Exp 4 follow-up — thinking mode in the agentic loop", "",
             "Sampling note: thinking mode requires recommended sampling "
             "(t=0.6, top_p=0.95, top_k=20); E4-AS (thinking OFF, same "
             "sampling) deconfounds the sampling change.", ""]
    lines += ["### Quality", "", "| Model | Cond | chrF++ | COMET |",
              "|" + "---|" * 4]
    for _, name, _, _ in MODELS:
        for c in CONDS:
            h, r, _ = runs[(name, c)]
            lines.append(f"| {name} | {c} | "
                         f"{chrf.corpus_score(h, [r]).score:.2f} | "
                         f"{comet_scores[(name, c)]:.4f} |")
    lines += ["", "### Contrasts (paired bootstrap, chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |", "|" + "---|" * 4]
    for (n, a, b, desc), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        lines.append(f"| {n} | {desc} | {v['delta']:+.2f} "
                     f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")
    lines += ["", "### Behavior", "",
              "| Model | Cond | ep w/ calls | calls/ep | rounds/ep | "
              "budget-forced | think chars/ep | length-caps |",
              "|" + "---|" * 8]
    for _, name, _, _ in MODELS:
        for c in ("E4-A", "E4-AS", "E4-AT"):
            b = beh[f"{name}|{c}"]
            lines.append(f"| {name} | {c} | {b['episodes_with_calls']}/{b['n']} "
                         f"| {b['calls_mean']} | {b['rounds_mean']} "
                         f"| {b['finish'].get('budget_forced', 0)} "
                         f"| {b['think_chars_mean']} | {b['length_caps']} |")
    lines += ["", "Tool mix:"]
    for _, name, _, _ in MODELS:
        for c in ("E4-AS", "E4-AT"):
            lines.append(f"  - {name} {c}: {beh[f'{name}|{c}']['tool_mix']}")
    lines += ["", "### Calibration (H4)", "",
              "| Model | Cond | gloss fetches | recall | precision |",
              "|" + "---|" * 5]
    for _, name, _, _ in MODELS:
        for c in ("E4-A", "E4-AS", "E4-AT"):
            cl = cal[f"{name}|{c}"]
            lines.append(f"| {name} | {c} | {cl['gloss_fetches']} "
                         f"| {cl['recall']} | {cl['precision']} |")

    with open(os.path.join(E4, "exp4_think_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[think-score] wrote exp4_think_metrics.json + exp4_think_summary.md "
          "+ ANALYSIS.md")


if __name__ == "__main__":
    main()
