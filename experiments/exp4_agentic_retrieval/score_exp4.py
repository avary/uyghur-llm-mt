"""
score_exp4.py - Exp 4 analysis: agentic retrieval (pull) vs static prompts.

Per model x condition (E4-A full tools, E4-B lean):
  - quality: chrF++ + COMET, paired bootstrap vs C0, C3L, and E4-A vs E4-B
  - behavior: calls/episodes, rounds, tool mix, addressing mode, resolution
    paths, miss-path trichotomy (accept / lemma / near-match), budget-forced
    rate, protocol failures, final-answer degeneration
  - cost: context tokens vs static prompt sizes
  - metacognition (H4): precision/recall of fetched glosses against a
    "needed words" heuristic (content tokens whose first-sense gloss keyword
    is absent from the model's own C0 translation — lexical heuristic,
    documented as such)

Run in the SCORING env after the translate phase:
  "$TMPDIR/scoring/bin/python" experiments/exp4_agentic_retrieval/score_exp4.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
from significance import _segment_stats, _corpus_from_stats  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E4 = os.path.join(HERE, "results")
E1 = os.path.join(HERE, "..", "exp1_component_ablation", "results")
E3 = os.path.join(HERE, "..", "exp3_cross_family", "results")
FLORES_ANALYSIS = os.path.join(HERE, "..", "..", "stats", "flores_dev", "dev.results.jsonl")

MODELS = [
    ("Qwen/Qwen3.5-9B", "qwen3.5-9b", "Qwen3.5-9B",
     f"{E1}/c0_baseline_qwen3.5-9b.results.jsonl",
     f"{E3}/qwen3.5-9b_c3_dict_lemma.results.jsonl"),
    ("google/gemma-4-E4B-it", "gemma-4-e4b-it", "Gemma-4-E4B",
     f"{E3}/gemma-4-e4b-it_c0_baseline.results.jsonl",
     f"{E3}/gemma-4-e4b-it_c3_dict_lemma.results.jsonl"),
    ("mistralai/Ministral-3-8B-Instruct-2512-BF16",
     "ministral-3-8b-instruct-2512-bf16", "Ministral-3-8B",
     f"{E3}/ministral-3-8b-instruct-2512-bf16_c0_baseline.results.jsonl",
     f"{E3}/ministral-3-8b-instruct-2512-bf16_c3_dict_lemma.results.jsonl"),
    ("meta-llama/Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct", "Llama-3.1-8B",
     f"{E3}/llama-3.1-8b-instruct_c0_baseline.results.jsonl",
     f"{E3}/llama-3.1-8b-instruct_c3_dict_lemma.results.jsonl"),
]

CONTRASTS = [
    ("C0", "E4-A", "E4-A − C0 (agency vs nothing)"),
    ("C3L", "E4-A", "E4-A − C3L (pull vs best push)"),
    ("E4-B", "E4-A", "E4-A − E4-B (offering more tools)"),
]

ECHO_MARKERS = ("### Uyghur Source Sentence:", "### English Translation:",
                "Clause Architecture")


def load_run(path):
    hyps, refs = [], []
    n_len = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t = (r.get("translation") or "").strip()
            hyps.append(t)
            refs.append(r["reference"].strip())
            n_len += r.get("finish_reason") == "length"
    return hyps, refs, n_len


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


def behavior(recs):
    out = {}
    n = len(recs)
    calls = [r["n_calls"] for r in recs]
    rounds = [r["n_rounds"] for r in recs]
    mix, fin, res = Counter(), Counter(), Counter()
    addr = Counter()
    tri = Counter()
    tok_in = tok_out = 0
    degenerate = 0
    for r in recs:
        mix.update(r["tool_mix"])
        fin[r["finish"]] += 1
        addr.update(r["addressing"])
        for c in r["call_log"]:
            res[c.get("resolution")] += 1
            if c["tool"] == "get_gloss" and c.get("result") == "miss":
                # trichotomy: what did the agent do AFTER this miss?
                tri["misses"] += 1
        tok_in += r["tokens_in"]
        tok_out += r["tokens_out"]
        t = r.get("translation") or ""
        degenerate += bool(any(m in t for m in ECHO_MARKERS)
                            or r["finish"] in ("failed", "context_overflow"))
    # miss-path trichotomy from message sequence: after a gloss miss, next
    # call to same token = lemma | near_matches | nothing (accepted)
    for r in recs:
        for i, c in enumerate(r["call_log"]):
            if c["tool"] == "get_gloss" and c.get("result") == "miss" and c.get("resolved"):
                tok = c["resolved"]
                follow = None
                for c2 in r["call_log"][i + 1:]:
                    if c2.get("resolved") == tok:
                        follow = c2["tool"]
                        break
                if follow == "get_lemma":
                    tri["explored_lemma"] += 1
                elif follow == "get_near_matches":
                    tri["explored_near"] += 1
                elif follow is None:
                    tri["accepted_unknown"] += 1
    out.update({
        "n": n, "calls_mean": round(float(np.mean(calls)), 1),
        "calls_p50": float(np.percentile(calls, 50)),
        "calls_max": max(calls),
        "episodes_with_calls": sum(c > 0 for c in calls),
        "rounds_mean": round(float(np.mean(rounds)), 1),
        "finish": dict(fin), "tool_mix": dict(mix), "resolution": dict(res),
        "addressing": dict(addr),
        "n_parse_failures": sum(r["n_parse_failures"] for r in recs),
        "n_skipped_calls": sum(r["n_skipped_calls"] for r in recs),
        "n_unresolved": sum(r["n_unresolved"] for r in recs),
        "tokens_in_mean": tok_in // n, "tokens_out_mean": tok_out // n,
        "degenerate_finals": degenerate,
    })
    if tri.get("misses"):
        out["miss_path"] = {k: round(v / tri["misses"], 3)
                            for k, v in tri.items() if k != "misses"}
        out["miss_path"]["n_misses"] = tri["misses"]
    return out


def needed_words(rec_c0_hyps, records, model_tag):
    """Heuristic: per sentence, content tokens whose first-sense gloss keyword
    is absent from the model's own C0 translation. Returns sets of needed
    token ids per sentence."""
    import re
    needed = {}
    for i, rec in enumerate(records):
        toks = rec["per_token"]
        hyp = (rec_c0_hyps[i] or "").lower()
        need = []
        for t in toks:
            if t["pos"] == "Punctuation" or not t["has_dict"]:
                continue
            g = t["gloss"]
            senses = g.split(": ", 1)[1] if ": " in g else g
            first = re.split(r"/\s*\d+\.|;|,", senses)[0].strip(" .")
            first = re.sub(r"\[.*?\]", "", first).strip()
            kw = first.split()[0].lower() if first else ""
            if len(kw) >= 3 and kw not in hyp:
                need.append(t["id"])
        needed[i] = need
    return needed


def calibration(recs, needed, n_content_by_sent):
    """Precision/recall of fetched gloss tokens vs needed tokens."""
    fetched_needed = needed_count = fetched_count = 0
    for r in recs:
        fetched = {c["resolved"] for c in r["call_log"]
                   if c["tool"] == "get_gloss" and c.get("resolved")}
        need = set(needed[r["id"]])
        fetched_needed += len(fetched & need)
        needed_count += len(need)
        fetched_count += len(fetched)
    return {
        "gloss_fetches": fetched_count,
        "needed_tokens": needed_count,
        "recall": round(fetched_needed / needed_count, 3) if needed_count else None,
        "precision": round(fetched_needed / fetched_count, 3) if fetched_count else None,
    }


def main(n_boot=5000):
    chrf = CHRF(word_order=2)
    records = [json.loads(l) for l in open(FLORES_ANALYSIS, encoding="utf-8")]
    n_content_by_sent = [sum(1 for t in r["per_token"] if t["pos"] != "Punctuation")
                         for r in records]

    lines = ["", "## Experiment 4 — agentic retrieval: results", ""]

    # ---- load all runs -----------------------------------------------------
    runs = {}  # (model_name, cond) -> (hyps, refs, stats, recs)
    for model_id, tag, name, c0_path, c3l_path in MODELS:
        for cond, path in [("C0", c0_path), ("C3L", c3l_path),
                           ("E4-A", f"{E4}/{tag}_e4-a.results.jsonl"),
                           ("E4-B", f"{E4}/{tag}_e4-b.results.jsonl")]:
            assert os.path.exists(path), f"missing {path}"
            hyps, refs, n_len = load_run(path)
            stats = np.array([_segment_stats(h, r) for h, r in zip(hyps, refs)])
            recs = None
            if cond.startswith("E4"):
                recs = [json.loads(l) for l in open(path, encoding="utf-8")]
            runs[(name, cond)] = (hyps, refs, stats, recs)
        print(f"[exp4-score] loaded {name}: C0/C3L/E4-A/E4-B")

    # ---- quality + COMET -----------------------------------------------------
    from comet import download_model, load_from_checkpoint
    ckpt = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(ckpt)
    scores, comet_scores = {}, {}
    for (name, cond), (hyps, refs, stats, recs) in runs.items():
        scores[(name, cond)] = chrf.corpus_score(hyps, [refs]).score
        data = [{"src": records[i]["sentence"], "mt": h, "ref": r}
                for i, (h, r) in enumerate(zip(hyps, refs))]
        comet_scores[(name, cond)] = float(comet_model.predict(
            data, batch_size=64, gpus=1, progress_bar=False).system_score)
        print(f"[exp4-score] {name} {cond}: chrF++ {scores[(name, cond)]:.2f} "
              f"COMET {comet_scores[(name, cond)]:.4f}", flush=True)
    del comet_model
    import torch
    torch.cuda.empty_cache()

    # ---- contrasts ------------------------------------------------------------
    deltas = {}
    for model_id, tag, name, _, _ in MODELS:
        for a, b, desc in CONTRASTS:
            sa, sb, d, lo, hi, p = bootstrap_between(
                runs[(name, a)][2], runs[(name, b)][2], n_boot)
            deltas[(name, a, b, desc)] = {"delta": d, "lo": lo, "hi": hi,
                                          "p": p}
            print(f"[exp4-score] {name} {desc}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}] "
                  f"p={p:.4f}", flush=True)

    # ---- behavior + calibration -------------------------------------------------
    beh, cal = {}, {}
    for model_id, tag, name, c0_path, _ in MODELS:
        c0_hyps = runs[(name, "C0")][0]
        needed = needed_words(c0_hyps, records, None)
        for cond in ("E4-A", "E4-B"):
            recs = runs[(name, cond)][3]
            beh[f"{name}|{cond}"] = behavior(recs)
            cal[f"{name}|{cond}"] = calibration(recs, needed, n_content_by_sent)

    # ---- persist -----------------------------------------------------------------
    out = {
        "scores": {f"{n}|{c}": {"chrfpp": round(scores[(n, c)], 2),
                                 "comet": round(comet_scores[(n, c)], 4)}
                   for _, _, n, _, _ in MODELS
                   for c in ("C0", "C3L", "E4-A", "E4-B")},
        "contrasts": {f"{n}|{b}−{a}": {"desc": d, "delta": round(v["delta"], 2),
                                        "ci_lo": round(v["lo"], 2),
                                        "ci_hi": round(v["hi"], 2),
                                        "p": round(v["p"], 4)}
                      for (n, a, b, d), v in deltas.items()},
        "behavior": beh, "calibration": cal,
    }
    with open(os.path.join(E4, "exp4_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # ---- markdown ------------------------------------------------------------------
    lines += ["### Quality", "",
              "| Model | Cond | chrF++ | COMET |", "|" + "---|" * 4]
    for _, _, name, _, _ in MODELS:
        for c in ("C0", "C3L", "E4-A", "E4-B"):
            s, cm = scores[(name, c)], comet_scores[(name, c)]
            lines.append(f"| {name} | {c} | {s:.2f} | {cm:.4f} |")
    lines += ["", "### Contrasts (paired bootstrap, chrF++)", "",
              "| Model | Contrast | Δ [95% CI] | p |", "|" + "---|" * 4]
    for (n, a, b, desc), v in deltas.items():
        star = " ✱" if v["p"] < 0.05 else ""
        lines.append(f"| {n} | {desc} | {v['delta']:+.2f} "
                     f"[{v['lo']:+.2f}, {v['hi']:+.2f}]{star} | {v['p']:.4f} |")

    lines += ["", "### Agent behavior", "",
              "| Model | Cond | ep w/ calls | calls/ep | rounds/ep | "
              "budget-forced | parse-fail | skipped | unresolved | tokens in/out |",
              "|" + "---|" * 10]
    for _, _, name, _, _ in MODELS:
        for cond in ("E4-A", "E4-B"):
            b = beh[f"{name}|{cond}"]
            lines.append(f"| {name} | {cond} | {b['episodes_with_calls']}/{b['n']} "
                         f"| {b['calls_mean']} | {b['rounds_mean']} "
                         f"| {b['finish'].get('budget_forced', 0)} "
                         f"| {b['n_parse_failures']} | {b['n_skipped_calls']} "
                         f"| {b['n_unresolved']} "
                         f"| {b['tokens_in_mean']}/{b['tokens_out_mean']} |")
    lines += ["", "Tool mix (E4-A):"]
    for _, _, name, _, _ in MODELS:
        b = beh[f"{name}|E4-A"]
        lines.append(f"  - {name}: {b['tool_mix']}  "
                     f"(addressing: {b['addressing']})")
        if "miss_path" in b:
            lines.append(f"    miss-path: {b['miss_path']}")

    lines += ["", "### Metacognition calibration (H4, lexical heuristic)", "",
              "| Model | Cond | gloss fetches | needed tokens | recall | precision |",
              "|" + "---|" * 6]
    for _, _, name, _, _ in MODELS:
        for cond in ("E4-A", "E4-B"):
            c = cal[f"{name}|{cond}"]
            lines.append(f"| {name} | {cond} | {c['gloss_fetches']} "
                         f"| {c['needed_tokens']} | {c['recall']} | {c['precision']} |")

    with open(os.path.join(E4, "exp4_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(HERE, "ANALYSIS.md"), "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[exp4-score] wrote exp4_metrics.json + exp4_summary.md + ANALYSIS.md")


if __name__ == "__main__":
    main()
