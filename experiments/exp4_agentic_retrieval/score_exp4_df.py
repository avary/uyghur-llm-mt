"""
score_exp4_df.py - Exp 4 draft-first protocol (E4-DF) analysis.

E4-DF: system-prompt instruction — (1) draft translation on its own,
(2) retrieve ONLY for uncertainties, (3) final translation. Thinking off,
greedy, same caps as E4-A → E4-DF − E4-A is the clean protocol contrast.

New metrics unlocked by the draft:
  - Δ(final − draft) per episode: retrieval value given a draft, separated
    from framing/anchor effects (the draft shares the framing, lacks the info)
  - draft vs C0: framing effect without information (same content as E4-B's
    zero-call anchor, but per-episode and universal)
  - calibration against the DRAFT (fetched words vs words the draft got
    wrong) — a much better-grounded needed-words heuristic than C0-based

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" experiments/exp4_agentic_retrieval/score_exp4_df.py
"""

import json
import os
import sys
import re

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402
from score_exp4 import behavior, needed_words, ECHO_MARKERS  # noqa: E402

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
     f"{E3}/gemma-4-e4b-it_c3_dict_lemma.results.jsonl")
]

CONTRASTS = [
    ("E4-A", "E4-DF", "E4-DF − E4-A (draft-first protocol)"),
    ("C0", "E4-DF", "E4-DF − C0"),
    ("C3L", "E4-DF", "E4-DF − C3L (vs best push)"),
]


def load_run(path):
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
    return hyps, refs


def sent_chrf(hyp, ref):
    return CHRF(word_order=2).sentence_score(hyp, [ref]).score


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


def draft_calibration(recs, records):
    """Needed-words from the DRAFT (words the draft likely got wrong), not C0:
    content tokens whose first-sense gloss keyword is absent from the draft."""
    fetched_needed = needed_count = fetched_count = 0
    for r in recs:
        draft = (r.get("draft") or "").lower()
        fetched = {c["resolved"] for c in r["call_log"]
                   if c["tool"] == "get_gloss" and c.get("resolved")}
        need = set()
        for t in records[r["id"]]["per_token"]:
            if t["pos"] == "Punctuation" or not t["has_dict"]:
                continue
            g = t["gloss"]
            senses = g.split(": ", 1)[1] if ": " in g else g
            first = re.split(r"/\s*\d+\.|;|,", senses)[0].strip(" .")
            first = re.sub(r"\[.*?\]", "", first).strip()
            kw = first.split()[0].lower() if first else ""
            if len(kw) >= 3 and kw not in draft:
                need.add(t["id"])
        fetched_needed += len(fetched & need)
        needed_count += len(need)
        fetched_count += len(fetched)
    return {"gloss_fetches": fetched_count, "needed_tokens": needed_count,
            "recall": round(fetched_needed / needed_count, 3) if needed_count else None,
            "precision": round(fetched_needed / fetched_count, 3) if fetched_count else None}


def main(n_boot=5000):
    chrf = CHRF(word_order=2)
    records = [json.loads(l) for l in open(FLORES_ANALYSIS, encoding="utf-8")]

    runs, df_recs = {}, {}
    for tag, name, c0_path, c3l_path in MODELS:
        for cond, path in [("C0", c0_path), ("C3L", c3l_path),
                           ("E4-A", f"{E4}/{tag}_e4-a.results.jsonl"),
                           ("E4-DF", f"{E4}/{tag}_e4-df.results.jsonl")]:
            assert os.path.exists(path), f"missing {path}"
            hyps, refs = load_run(path)
            stats = np.array([_segment_stats(h, r) for h, r in zip(hyps, refs)])
            runs[(name, cond)] = (hyps, refs, stats)
        df_recs[name] = [json.loads(l)
                         for l in open(f"{E4}/{tag}_e4-df.results.jsonl",
                                       encoding="utf-8")]
        print(f"[df-score] loaded {name}: C0/C3L/E4-A/E4-DF")

    # ---- COMET -----------------------------------------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    comet_scores = {}
    for key, (hyps, refs, stats) in runs.items():
        data = [{"src": records[i]["sentence"], "mt": h, "ref": r}
                for i, (h, r) in enumerate(zip(hyps, refs))]
        comet_scores[key] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[df-score] {key[0]} {key[1]}: "
              f"chrF++ {chrf.corpus_score(hyps, [refs]).score:.2f} "
              f"COMET {comet_scores[key]:.4f}", flush=True)
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- contrasts ---------------------------------------------------------------
    deltas = {}
    for tag, name, _, _ in MODELS:
        for a, b, desc in CONTRASTS:
            sa, sb, d, lo, hi, p = bootstrap_between(
                runs[(name, a)][2], runs[(name, b)][2], n_boot)
            deltas[(name, a, b, desc)] = {"delta": d, "lo": lo, "hi": hi, "p": p}
            print(f"[df-score] {name} {desc}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"p={p:.4f}", flush=True)

    # ---- draft analysis -------------------------------------------------------------
    draft_stats = {}
    for tag, name, _, _ in MODELS:
        recs = df_recs[name]
        n = len(recs)
        has_draft = sum(1 for r in recs if (r.get("draft") or "").strip())
        d_chrf, f_chrf, dd = [], [], []
        changed = 0
        for r in recs:
            draft = (r.get("draft") or "").strip()
            final = (r.get("translation") or "").strip()
            if draft:
                d_chrf.append(sent_chrf(draft, r["reference"]))
                f_chrf.append(sent_chrf(final, r["reference"]))
                dd.append(f_chrf[-1] - d_chrf[-1])
                changed += draft != final
        draft_stats[name] = {
            "episodes": n,
            "has_draft": has_draft,
            "draft_vs_final_changed": changed,
            "draft_chrF_mean": round(float(np.mean(d_chrf)), 2) if d_chrf else None,
            "final_chrF_mean": round(float(np.mean(f_chrf)), 2) if f_chrf else None,
            "delta_final_minus_draft_mean": round(float(np.mean(dd)), 2) if dd else None,
            "delta_final_minus_draft_p50": round(float(np.percentile(dd, 50)), 2) if dd else None,
            "P(final_better)": round(float(np.mean([x > 0.5 for x in dd])), 3) if dd else None,
            "P(final_worse)": round(float(np.mean([x < -0.5 for x in dd])), 3) if dd else None,
        }
        draft_stats[name]["calibration_vs_draft"] = draft_calibration(recs, records)
        draft_stats[name]["behavior"] = behavior(recs)
        print(f"[df-score] {name}: draft {draft_stats[name]['draft_chrF_mean']} "
              f"→ final {draft_stats[name]['final_chrF_mean']} "
              f"(Δ {draft_stats[name]['delta_final_minus_draft_mean']}, "
              f"changed {changed}/{has_draft})", flush=True)

    # ---- persist --------------------------------------------------------------------
    with open(os.path.join(E4, "exp4_df_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "scores": {f"{n}|{c}": {"chrfpp": round(
                           chrf.corpus_score(runs[(n, c)][0],
                                             [runs[(n, c)][1]]).score, 2),
                           "comet": round(comet_scores[(n, c)], 4)}
                       for _, n, _, _ in MODELS
                       for c in ("C0", "C3L", "E4-A", "E4-DF")},
            "contrasts": {f"{n}|{b}−{a}": {"desc": d,
                                            "delta": round(v["delta"], 2),
                                            "ci_lo": round(v["lo"], 2),
                                            "ci_hi": round(v["hi"], 2),
                                            "p": round(v["p"], 4)}
                          for (n, a, b, d), v in deltas.items()},
            "draft": draft_stats,
        }, f, indent=2, ensure_ascii=False)

    # ---- markdown -----------------------------------------------------------------------
    lines = ["", "## Exp 4 follow-up — draft-first protocol (E4-DF)", "",
             "Instruction (system prompt): draft on your own → check ONLY "
             "uncertainties with tools → final translation. Thinking off, "
             "greedy, same caps as E4-A.", ""]
    lines += ["### Quality", "", "| Model | Cond | chrF++ | COMET |",
              "|" + "---|" * 4]
    for _, name, _, _ in MODELS:
        for c in ("C0", "C3L", "E4-A", "E4-DF"):
            h, r, _ = runs[(name, c)]
            lines.append(f"| {name} | {c} | "
                         f"{chrf.corpus_score(h, [r]).score:.2f} | "
                         f"{comet_scores[(name, c)]:.4f} |")
    lines += ["", "### Contrasts (paired bootstrap, chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |", "|" + "---|" * 4]
    for (n, a, b, d), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        lines.append(f"| {n} | {d} | {v['delta']:+.2f} "
                     f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")
    lines += ["", "### Draft analysis (per-episode, sentence chrF)", "",
              "| Model | has draft | changed | draft chrF | final chrF | "
              "Δ(fin−dr) mean | P(better) | P(worse) |",
              "|" + "---|" * 8]
    for _, name, _, _ in MODELS:
        d = draft_stats[name]
        hd = f"{d['has_draft']}/{d['episodes']}"
        lines.append(f"| {name} | {hd} | {d['draft_vs_final_changed']} "
                     f"| {d['draft_chrF_mean']} | {d['final_chrF_mean']} "
                     f"| {d['delta_final_minus_draft_mean']} "
                     f"| {d['P(final_better)']} | {d['P(final_worse)']} |")
    lines += ["", "### Calibration vs draft + behavior (E4-DF)", ""]
    for _, name, _, _ in MODELS:
        d = draft_stats[name]
        c = d["calibration_vs_draft"]
        b = d["behavior"]
        lines.append(
            f"- **{name}**: gloss fetches {c['gloss_fetches']}, "
            f"recall {c['recall']}, precision {c['precision']}; "
            f"calls/ep {b['calls_mean']}, budget-forced "
            f"{b['finish'].get('budget_forced', 0)}, "
            f"episodes w/ calls {b['episodes_with_calls']}/{b['n']}, "
            f"parse-fail {b['n_parse_failures']}, "
            f"tool mix {b['tool_mix']}")

    with open(os.path.join(E4, "exp4_df_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[df-score] wrote exp4_df_metrics.json + exp4_df_summary.md + ANALYSIS.md")


if __name__ == "__main__":
    main()
