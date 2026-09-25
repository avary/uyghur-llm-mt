#!/bin/bash
# run_ablation_full.sh - Full 2^3 component ablation (C0-C7) on the two
# knowledge-poor control models (Ministral-3-8B, Gemma-4-E2B).
# Self-contained for a batch job (fresh-$TMPDIR safe).
#
# C0/C1/C2/C7 already exist from the main sweep + control run; this generates
# the missing C3/C4/C5/C6, then scores the complete factorial (chrF++ + COMET +
# paired bootstrap Δ vs C0) and appends Exp-1-style tables to ANALYSIS.md.
#
# Outputs: results/exp3_ablation_full.{json,md} + ANALYSIS.md section.
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp3_cross_family/run_ablation_full.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

# --- env bootstrap (fresh $TMPDIR in batch jobs) -----------------------------
"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[ablation] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[ablation] scoring env OK')"

# --- generate missing conditions C3-C6 (C0/C1/C2/C7 already on disk) --------
"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models mistralai/Ministral-3-8B-Instruct-2512-BF16 google/gemma-4-E2B-it \
    --conditions C3 C4 C5 C6 \
    --phase translate

# --- complete factorial scoring (scoring env) --------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp3_cross_family/score_ablation_full.py

echo "[ablation] done — see ANALYSIS.md (full ablation section) + results/exp3_ablation_full.md"
