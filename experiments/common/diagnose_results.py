"""
diagnose_results.py - Post-hoc diagnostics for the FLORES-200 experiments.

Answers three questions from data already on disk:

1. MEMORIZATION: are any translations near-verbatim copies of the FLORES
   references (which would indicate the uig->eng dev pairs leaked into
   pretraining)?
2. SIGNIFICANCE: which observed differences are real? Exact paired bootstrap
   over sentence-level chrF++ statistics (sacrebleu's own segment statistics,
   re-aggregated over resampled indices).
3. WHERE-DOES-IT-HELP: per-sentence chrF++ delta (C7 - C0) vs the sentence's
   dictionary coverage - does the pipeline info help where it is informative?

Run in the scoring env:
  "$TMPDIR/scoring/bin/python" experiments/common/diagnose_results.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sacrebleu.metrics import CHRF

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E1 = os.path.join(_REPO, "exp1_component_ablation", "results")
E2 = os.path.join(_REPO, "exp2_model_scaling", "results")
FLORES_ANALYSIS = os.path.join(_REPO, "..", "stats", "flores_dev", "dev.results.jsonl")

chrf = CHRF(word_order=2)


def load(path):
    hyps, refs, sids = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
            sids.append(r["id"])
    return hyps, refs, sids


# ---------------------------------------------------------------------------
# 1. Memorization: near-verbatim reference overlap
# ---------------------------------------------------------------------------
def memorization_table(paths_labels):
    print("\n" + "=" * 100)
    print("1. MEMORIZATION CHECK — how close do outputs get to the FLORES references?")
    print("   (a model that memorized dev pairs would show a fat tail at chrF>=95)")
    print("=" * 100)
    print(f"{'run':<44} {'mean':>6} {'max':>6} {'=100':>5} {'>=95':>5} {'>=90':>5} {'>=80':>5}")
    for path, label in paths_labels:
        hyps, refs, _ = load(path)
        exact = 0
        buckets = Counter()
        scores = []
        for h, r in zip(hyps, refs):
            s = chrf.sentence_score(h, [r]).score
            scores.append(s)
            exact += int(h == r)
            for b in (95, 90, 80):
                buckets[b] += int(s >= b)
        print(f"{label:<44} {np.mean(scores):>6.2f} {max(scores):>6.2f} "
              f"{exact:>5} {buckets[95]:>5} {buckets[90]:>5} {buckets[80]:>5}")


# ---------------------------------------------------------------------------
# 2. Exact paired bootstrap on corpus chrF++
# ---------------------------------------------------------------------------
def segment_stats(hyp, ref):
    """sacrebleu's own per-segment statistics: [hyp, ref, match] x 8 orders."""
    ref_kwargs = chrf._extract_reference_info([ref])
    return chrf._compute_segment_statistics(hyp, ref_kwargs)


def corpus_chrf_from_stats(stats):  # stats: (n, 24) array
    agg = stats.sum(axis=0)
    return chrf._compute_score_from_stats(list(agg)).score


def paired_bootstrap(path_a, path_b, label_a, label_b, n_boot=10000, seed=12345):
    hyps_a, refs_a, _ = load(path_a)
    hyps_b, refs_b, _ = load(path_b)
    assert refs_a == refs_b, "runs must share references/ids"

    stats_a = np.array([segment_stats(h, r) for h, r in zip(hyps_a, refs_a)])
    stats_b = np.array([segment_stats(h, r) for h, r in zip(hyps_b, refs_b)])

    # validate our re-aggregation reproduces the corpus score exactly
    va, vb = corpus_chrf_from_stats(stats_a), corpus_chrf_from_stats(stats_b)
    sa = chrf.corpus_score(hyps_a, [refs_a]).score
    sb = chrf.corpus_score(hyps_b, [refs_b]).score
    assert abs(va - sa) < 1e-9 and abs(vb - sb) < 1e-9, "stat re-aggregation broken"

    rng = np.random.default_rng(seed)
    n = len(refs_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = (corpus_chrf_from_stats(stats_b[idx])
                     - corpus_chrf_from_stats(stats_a[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    print(f"{label_b} vs {label_a}: "
          f"Δ={vb - sa:+.2f} chrF++ "
          f"[95% CI {lo:+.2f}, {hi:+.2f}], p≈{p:.4f} "
          f"{'*' if p < 0.05 else '(n.s.)'}")


# ---------------------------------------------------------------------------
# 3. Where does the info help? delta vs dictionary coverage
# ---------------------------------------------------------------------------
def coverage_analysis():
    print("\n" + "=" * 100)
    print("3. WHERE-DOES-IT-HELP — per-sentence ΔchrF++ (C7-C0) vs dictionary coverage")
    print("   (if info helps, Δ should grow as coverage of rare words grows)")
    print("=" * 100)

    # sentence id -> fraction of content tokens with a dictionary hit
    cov = {}
    with open(FLORES_ANALYSIS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            toks = [t for t in r["per_token"] if t["pos"] != "Punctuation"]
            cov[r["sentence"] and len(cov)] = None  # placeholder, keyed by id below
    cov = {}
    with open(FLORES_ANALYSIS, encoding="utf-8") as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            toks = [t for t in r["per_token"] if t["pos"] != "Punctuation"]
            cov[i] = (sum(1 for t in toks if t["has_dict"]) / len(toks)) if toks else 1.0

    def deltas(model_tag, c0_file, c7_file):
        h0, r0, ids0 = load(c0_file)
        h7, r7, _ = load(c7_file)
        d = np.array([
            chrf.sentence_score(h, [r]).score - chrf.sentence_score(g, [r]).score
            for h, g, r in zip(h7, h0, r0)
        ])
        c = np.array([cov[i] for i in ids0])
        return d, c, (h0, h7, r0, ids0)

    def report(tag, d, c):
        from scipy import stats as sps  # scipy present in scoring env? fallback below
        print(f"\n{tag}: n={len(d)}, mean Δ={d.mean():+.2f}")
        qs = np.percentile(c, [25, 50, 75])
        for name, mask in [("Q1 low cov", c <= qs[0]),
                           ("Q2", (c > qs[0]) & (c <= qs[1])),
                           ("Q3", (c > qs[1]) & (c <= qs[2])),
                           ("Q4 high cov", c > qs[2])]:
            print(f"   {name:<12} (cov≤{mask.sum() and round(c[mask].mean(), 2)}): "
                  f"mean ΔchrF++ = {d[mask].mean():+.2f}")

    for model, e2base in [
        ("9B", os.path.join(E2, "qwen3.5-9b_c0_baseline.results.jsonl")),
    ]:
        d, c, _ = deltas(model,
                         os.path.join(E1, "c0_baseline_qwen3.5-9b.results.jsonl"),
                         os.path.join(E1, "c7_full_qwen3.5-9b.results.jsonl"))
        report(f"[Exp1 9B C7 vs C0]", d, c)
        # correlation
        try:
            from scipy.stats import spearmanr
            rho, p = spearmanr(c, d)
            print(f"   Spearman(coverage, Δ) = {rho:+.3f} (p={p:.3f})")
        except ImportError:
            # pearson fallback
            rho = np.corrcoef(c, d)[0, 1]
            print(f"   Pearson(coverage, Δ) = {rho:+.3f}")

    for model in ["qwen3.5-0.8b", "qwen3.5-4b", "qwen3.5-27b"]:
        d, c, _ = deltas(model,
                         os.path.join(E2, f"{model}_c0_baseline.results.jsonl"),
                         os.path.join(E2, f"{model}_c7_full.results.jsonl"))
        report(f"[Exp2 {model} C7 vs C0]", d, c)


if __name__ == "__main__":
    memorization_table([
        (os.path.join(E2, "qwen3.5-0.8b_c0_baseline.results.jsonl"), "0.8B C0 baseline"),
        (os.path.join(E2, "qwen3.5-2b_c0_baseline.results.jsonl"), "2B C0 baseline"),
        (os.path.join(E2, "qwen3.5-4b_c0_baseline.results.jsonl"), "4B C0 baseline"),
        (os.path.join(E2, "qwen3.5-9b_c0_baseline.results.jsonl"), "9B C0 baseline"),
        (os.path.join(E2, "qwen3.5-27b_c0_baseline.results.jsonl"), "27B C0 baseline"),
        (os.path.join(E1, "c7_full_qwen3.5-9b.results.jsonl"), "9B C7 full"),
    ])

    print("\n" + "=" * 100)
    print("2. PAIRED BOOTSTRAP SIGNIFICANCE (10k resamples, exact corpus chrF++)")
    print("=" * 100)
    c0 = os.path.join(E1, "c0_baseline_qwen3.5-9b.results.jsonl")
    for cid, name in [("c3_dict", "C3 dict"),
                      ("c2_morph", "C2 morph"),
                      ("c1_syntax", "C1 syntax"),
                      ("c7_full", "C7 full")]:
        paired_bootstrap(c0, os.path.join(E1, f"{name.split()[0]}_qwen3.5-9b.results.jsonl"
                             if False else f"{cid}_qwen3.5-9b.results.jsonl"),
                         "C0", cid)
    for model in ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3.5-4b",
                  "qwen3.5-9b", "qwen3.5-27b"]:
        paired_bootstrap(
            os.path.join(E2, f"{model}_c0_baseline.results.jsonl"),
            os.path.join(E2, f"{model}_c7_full.results.jsonl"),
            f"{model} C0", f"{model} C7")

    coverage_analysis()
