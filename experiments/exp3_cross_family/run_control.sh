#!/bin/bash
# run_control.sh - Anchor-vs-knowledge control run for Exp 3 (self-contained for
# a batch job; fresh-$TMPDIR safe).
#
# Generates C1 (syntax-only) and C2 (morph-only) for Ministral-3-8B and
# Gemma-4-E2B — table-format anchor, no glosses — then decomposes the C7 gain:
#     Δ(C1/C2 − C0) = anchor effect
#     Δ(C7 − C0)    = total effect (already measured in the main sweep)
#     Δ(C7 − C1/C2) = knowledge effect
#
# Results: appended to experiments/exp3_cross_family/ANALYSIS.md
#          + results/control_results.json
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp3_cross_family/run_control.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

# --- env bootstrap (fresh $TMPDIR in batch jobs) -----------------------------
"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[control] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[control] scoring env OK')"

# --- control translate phase (C1 + C2; C0/C7 already on disk) ---------------
"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models mistralai/Ministral-3-8B-Instruct-2512-BF16 google/gemma-4-E2B-it \
    --conditions C1 C2 \
    --phase translate

# --- scoring + decomposition (scoring env; COMET + paired bootstrap) --------
"${TMPDIR}/scoring/bin/python" -u experiments/exp3_cross_family/score_control.py

echo "[control] done — see ANALYSIS.md (control section) + results/control_results.json"
