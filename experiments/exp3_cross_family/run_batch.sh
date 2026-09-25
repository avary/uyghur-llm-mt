#!/bin/bash
# run_batch.sh - Self-contained Exp 3 entry point for a BATCH job.
#
# A batch job gets a fresh $TMPDIR: the cluster-provisioned vLLM conda env
# ($TMPDIR/vllm) is present, but the scoring venv and the pixtral patch are
# not (they were created in an interactive session's TMPDIR). This script
# recreates them idempotently, then runs the full sweep + scoring.
#
# Persistent across sessions: model weights + COMET checkpoint (HF_HOME on
# vault), and all experiment results (repo directory).
#
# Usage (batch job script):
#   bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp3_cross_family/run_batch.sh
#
# Or interactively:
#   nohup bash .../run_batch.sh > exp3_batch.log 2>&1 &

set -eu

cd "$(dirname "$0")/../../"   # repo root

# --- 0. environment ---------------------------------------------------------
PY="${TMPDIR:?TMPDIR must be set (cluster-provisioned)}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # so vLLM EngineCore finds ninja
export TMPDIR   # used by the run scripts to locate the scoring python

# --- 1. pixtral import patch (transformers 5.x vs vLLM 0.29) ----------------
"$PY" experiments/patches/patch_vllm_pixtral.py

# --- 2. scoring venv (sacrebleu + unbabel-comet; separate from vLLM env) ----
if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[exp3-batch] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[exp3-batch] scoring env OK')"

# --- 3. sweep + scoring ------------------------------------------------------
bash experiments/exp3_cross_family/run_all.sh

echo "[exp3-batch] all done — results in experiments/exp3_cross_family/results/"
