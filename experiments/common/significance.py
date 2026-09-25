"""
significance.py - Exact paired bootstrap for corpus chrF++ differences.

Uses sacrebleu's own per-segment statistics ([hyp, ref, match] triplets per
n-gram order) so the resampled corpus score is the exact corpus chrF++ of the
resampled sentence set (no approximation). 10k resamples take ~1 min per pair.

Runs in the scoring env.
"""

from typing import Tuple

import numpy as np
from sacrebleu.metrics import CHRF

_chrf = CHRF(word_order=2)


def _load(path):
    import json
    hyps, refs = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            hyps.append((r.get("translation") or "").strip())
            refs.append(r["reference"].strip())
    return hyps, refs


def _segment_stats(hyp, ref):
    ref_kwargs = _chrf._extract_reference_info([ref])
    return _chrf._compute_segment_statistics(hyp, ref_kwargs)


def _corpus_from_stats(stats):
    agg = stats.sum(axis=0)
    return _chrf._compute_score_from_stats(list(agg)).score


def paired_bootstrap_chrf(path_a: str, path_b: str,
                          n_boot: int = 5000,
                          seed: int = 12345) -> Tuple[float, float, float, float, float]:
    """Paired bootstrap for corpus chrF++ (B − A).

    Returns (score_a, score_b, delta, ci_lo, ci_hi, p_two_sided).
    Sentences are matched by order; the two files must contain the same ids.
    """
    hyps_a, refs_a = _load(path_a)
    hyps_b, refs_b = _load(path_b)
    assert len(hyps_a) == len(hyps_b) and refs_a == refs_b, \
        "paired files must be aligned (same ids/references)"

    stats_a = np.array([_segment_stats(h, r) for h, r in zip(hyps_a, refs_a)])
    stats_b = np.array([_segment_stats(h, r) for h, r in zip(hyps_b, refs_b)])

    sa = _corpus_from_stats(stats_a)
    sb = _corpus_from_stats(stats_b)

    rng = np.random.default_rng(seed)
    n = len(refs_a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = _corpus_from_stats(stats_b[idx]) - _corpus_from_stats(stats_a[idx])

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return sa, sb, sb - sa, lo, hi, p
