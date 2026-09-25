"""
evaluator.py - Automated MT scoring: chrF++ (sacrebleu) + COMET (wmt22-comet-da).

IMPORTANT - runs in the dedicated scoring venv, NOT the vLLM env:
    $TMPDIR/scoring/bin/python experiments/common/evaluator.py \\
        --results run1.jsonl run2.jsonl --labels "run1" "run2" \\
        --out metrics.json --summary summary.md

COMET checkpoint: Unbabel/wmt22-comet-da (cached under $HF_HOME).
Translations that are empty or look like error placeholders are counted as
failures and scored as empty strings (they drag the corpus score down, which is
the honest treatment for a missing system output).
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional

ERROR_PAT = re.compile(r"^\[(Translation Error|API Key)", re.IGNORECASE)


def is_bad_translation(text: Optional[str]) -> bool:
    if text is None:
        return True
    t = text.strip()
    return (not t) or bool(ERROR_PAT.match(t))


def load_run(path: str) -> Dict:
    """Load one results.jsonl -> {ids, hyps, refs, srcs, n_failed, records}."""
    ids, hyps, refs, srcs, n_failed = [], [], [], [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hyp = rec.get("translation")
            if is_bad_translation(hyp):
                hyp = ""
                n_failed += 1
            else:
                hyp = hyp.strip()
            ids.append(rec["id"])
            hyps.append(hyp)
            refs.append(rec["reference"])
            srcs.append(rec["sentence"])
    return {"ids": ids, "hyps": hyps, "refs": refs, "srcs": srcs,
            "n_failed": n_failed, "path": path}


def score_chrf(hyps: List[str], refs: List[str]) -> float:
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score


def score_comet(srcs: List[str], hyps: List[str], refs: List[str],
                batch_size: int = 64, gpus: int = 1,
                model_name: str = "Unbabel/wmt22-comet-da") -> float:
    from comet import download_model, load_from_checkpoint
    ckpt = download_model(model_name)
    model = load_from_checkpoint(ckpt)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
    out = model.predict(data, batch_size=batch_size, gpus=gpus,
                        progress_bar=False)
    del model
    return float(out.system_score)


def score_run(run: Dict, use_comet: bool, comet_batch: int) -> Dict:
    res = {
        "path": run["path"],
        "label": run.get("label", os.path.basename(run["path"])),
        "n": len(run["hyps"]),
        "n_failed": run["n_failed"],
    }
    res["chrfpp"] = round(score_chrf(run["hyps"], run["refs"]), 2)
    if use_comet:
        res["comet"] = round(
            score_comet(run["srcs"], run["hyps"], run["refs"],
                        batch_size=comet_batch),
            4,
        )
    return res


# ---------------------------------------------------------------------------
# Summary rendering
# ---------------------------------------------------------------------------
def _fmt(v, nd=2):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def write_markdown_summary(runs: List[Dict], title: str, out_path: str,
                           delta_label: str, delta_base: int):
    has_comet = any("comet" in r for r in runs)
    cols = ["Run", "n", "failed", "chrF++"] + (["COMET"] if has_comet else [])
    lines = [f"# {title}", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for i, r in enumerate(runs):
        row = [r.get("label", os.path.basename(r["path"])), str(r["n"]),
               str(r["n_failed"]), _fmt(r.get("chrfpp"))]
        if has_comet:
            row.append(_fmt(r.get("comet"), 4))
        lines.append("| " + " | ".join(row) + " |")

    # Delta table (e.g. gain over baseline, or pipeline gain per model)
    lines += ["", f"## Δ ({delta_label})", ""]
    dcols = ["Run", "Δ chrF++"] + (["Δ COMET"] if has_comet else [])
    lines += ["| " + " | ".join(dcols) + " |",
              "|" + "|".join(["---"] * len(dcols)) + "|"]
    base = runs[delta_base]
    for r in runs:
        d_chrf = r.get("chrfpp") - base.get("chrfpp") if r.get("chrfpp") is not None else None
        row = [r.get("label", os.path.basename(r["path"])),
               ("+" if d_chrf >= 0 else "") + _fmt(d_chrf)]
        if has_comet:
            d_c = r.get("comet") - base.get("comet") if r.get("comet") is not None else None
            row.append(("+" if d_c >= 0 else "") + _fmt(d_c, 4))
        lines.append("| " + " | ".join(row) + " |")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Score MT results (chrF++ / COMET)")
    ap.add_argument("--results", nargs="+", required=True,
                    help="results.jsonl files to score")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display label per results file")
    ap.add_argument("--out", required=True, help="output metrics JSON")
    ap.add_argument("--summary", default=None, help="output Markdown summary")
    ap.add_argument("--title", default="MT Evaluation Results")
    ap.add_argument("--delta-label", default="vs. first run",
                    help="label for the delta table")
    ap.add_argument("--no-comet", action="store_true",
                    help="skip COMET (chrF++ only)")
    ap.add_argument("--comet-batch", type=int, default=64)
    args = ap.parse_args()

    labels = args.labels or [os.path.basename(p) for p in args.results]
    assert len(labels) == len(args.results)

    use_comet = not args.no_comet
    runs = []
    for path, label in zip(args.results, labels):
        print(f"[evaluator] loading {path}")
        run = load_run(path)
        run["label"] = label
        print(f"[evaluator]   {len(run['hyps'])} records, "
              f"{run['n_failed']} failed/empty -> scoring...")
        runs.append(score_run(run, use_comet, args.comet_batch))
        print(f"[evaluator]   chrF++ = {runs[-1]['chrfpp']}"
              + (f", COMET = {runs[-1]['comet']}" if use_comet else ""))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"runs": runs}, f, ensure_ascii=False, indent=2)
    print(f"[evaluator] wrote {args.out}")

    if args.summary:
        write_markdown_summary(runs, args.title, args.summary,
                               delta_label=args.delta_label, delta_base=0)
        print(f"[evaluator] wrote {args.summary}")


if __name__ == "__main__":
    main()
