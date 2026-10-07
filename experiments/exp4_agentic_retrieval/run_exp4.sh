#!/bin/bash
# run_exp4.sh - Experiment 4: agentic retrieval vs static prompts.
# Self-contained for a batch job (fresh-$TMPDIR safe).
#
# 1. Static baseline C3L runs for the two models that lack them
#    (Gemma-E4B, Llama-3.1-8B) via the proven exp3 driver.
# 2. Agentic runs: 4 models x {E4-A full tools, E4-B lean} x 997 episodes,
#    calibrated caps (24 calls / 8 rounds / 512 tokens), greedy, thinking off.
# 3. Scoring + behavior + calibration analysis (scoring env).
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp4_agentic_retrieval/run_exp4.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

# --- env bootstrap (fresh $TMPDIR in batch jobs) -----------------------------
"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[exp4] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[exp4] scoring env OK')"

# --- 1. static C3L baselines (Gemma-E4B, Llama-8B) ---------------------------
"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models google/gemma-4-E4B-it meta-llama/Llama-3.1-8B-Instruct \
    --conditions C3L --phase translate

# --- 2. agentic runs (E4-A full tools, E4-B lean) ----------------------------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --max-calls 24 --max-rounds 8 --max-tokens 512

# --- 3. analysis (scoring env) ------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp4_agentic_retrieval/score_exp4.py

echo "[exp4] done — see experiments/exp4_agentic_retrieval/ANALYSIS.md + results/exp4_summary.md"
