"""
significance_cli.py - CLI wrapper for paired_bootstrap_chrf.

Runs in the SCORING env (sacrebleu is not installed in the vLLM env).
Reads a JSON list of comparison specs on argv, prints a JSON list of results.

Usage:
  scoring_python significance_cli.py [spec.json]
  spec: [{"a": "path_a.jsonl", "b": "path_b.jsonl", "name": "Model C7 vs C0",
          "n_boot": 5000}, ...]
If no spec file is given, reads one JSON spec from stdin.
"""

import json
import sys

from significance import paired_bootstrap_chrf


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], encoding="utf-8") as f:
            specs = json.load(f)
    else:
        specs = json.load(sys.stdin)

    results = []
    for s in specs:
        sa, sb, delta, lo, hi, p = paired_bootstrap_chrf(
            s["a"], s["b"], n_boot=s.get("n_boot", 5000))
        results.append({
            "name": s.get("name", f"{s['b']} vs {s['a']}"),
            "score_a": round(sa, 2), "score_b": round(sb, 2),
            "delta": round(delta, 2),
            "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
            "p_value": round(p, 4),
        })
        print(f"[significance] {s.get('name', '')}: Δ={delta:+.2f} "
              f"[{lo:+.2f}, {hi:+.2f}] p={p:.4f}", file=sys.stderr, flush=True)

    json.dump(results, sys.stdout)


if __name__ == "__main__":
    main()
