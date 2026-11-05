#!/bin/bash
# run_exp4_think.sh - Exp 4 follow-up: thinking mode in the agentic loop
# (Qwen3.5-9B + Gemma-4-E4B only). Self-contained for a batch job.
#
#   E4-AT: thinking ON  + recommended sampling (t=0.6, top_p=0.95, top_k=20)
#   E4-AS: thinking OFF + recommended sampling (control: isolates the sampling
#          change so E4-AT vs E4-A is deconfounded)
# 8192 tokens per assistant turn (thinking headroom); context limit 65536.
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp4_agentic_retrieval/run_exp4_think.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"
export TMPDIR

"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[think] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unabel-comet "setuptools<81" 2>/dev/null || \
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[think] scoring env OK')"

# --- agentic runs: E4-AT (thinking) + E4-AS (sampling control) ---------------
# Proportional call budget (alpha=1: one call per content word, floor 4) —
# sentence-adaptive instead of the flat 24; identical policy in both arms so
# the E4-AT vs E4-AS thinking contrast stays clean. 16k tokens per turn.
"$PY" -u experiments/exp4_agentic_retrieval/run_exp4.py --mode translate \
    --models Qwen/Qwen3.5-9B google/gemma-4-E4B-it \
    --conditions E4-AT E4-AS \
    --budget-policy proportional --budget-alpha 1.0 --budget-floor 4 \
    --max-calls 24 --max-rounds 8 --max-model-len 65536

# --- analysis (scoring env) ---------------------------------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp4_agentic_retrieval/score_exp4_think.py

echo "[think] done — see ANALYSIS.md (thinking follow-up section) + results/exp4_think_summary.md"
