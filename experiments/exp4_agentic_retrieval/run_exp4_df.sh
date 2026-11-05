#!/bin/bash
# run_exp4_df.sh - Exp 4 follow-up: draft-first protocol (E4-DF).
# Self-contained for a batch job (fresh-$TMPDIR safe).
#
# System-prompt instruction: (1) draft translation on its own, (2) call tools
# ONLY for uncertainties, (3) final translation. Thinking off, greedy, same
# caps as E4-A → E4-DF − E4-A is the clean protocol contrast. All 4 models.
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp4_agentic_retrieval/run_exp4_df.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"
export TMPDIR

"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[df] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[df] scoring env OK')"

# --- draft-first runs (all 4 models, thinking off, E4-A-matched caps) --------
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models Qwen/Qwen3.5-9B google/gemma-4-E4B-it \
    --conditions E4-DF \
    --max-calls 24 --max-rounds 8 --max-model-len 16384

# --- analysis (scoring env) ----------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp4_agentic_retrieval/score_exp4_df.py

echo "[df] done — see ANALYSIS.md (draft-first section) + results/exp4_df_summary.md"
