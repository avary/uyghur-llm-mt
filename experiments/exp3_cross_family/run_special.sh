#!/bin/bash
# run_special.sh - Follow-up studies: segmentation condition (S) and
# lemma-keyed dictionary condition (C3L). Self-contained for a batch job.
#
# Study 1 (S): exact C0 prompt with the sentence as a numbered token list —
#   tests whether clean word separation alone explains the weak-model gains.
# Study 2 (C3L): dict-only table, glosses from lemma-keyed lookups (80.4%
#   coverage) instead of surface-form lookups (44.0%) — tests how much the
#   lemmatization unlocks the dictionary.
#
# Run matrix:
#   Llama-3.2-1B : S, C1   (C1 = capacity-floor test with the lean prompt)
#   Llama-3.2-3B : S, C1   (third unsupported family; no singles existed)
#   Ministral-8B : S, C3L
#   Gemma-4-E2B  : S, C3L
#   Qwen3.5-9B   : S, C3, C3L  (C3 re-run: post-441236a gloss format, so the
#                               C3L−C3 contrast is version-consistent; also
#                               yields old-vs-new gloss-format effect)
#
# Usage:  bash /home/hpc/b279bb/b279bb29/uyghur-llm-mt/experiments/exp3_cross_family/run_special.sh

set -eu

cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
UV="${TMPDIR}/vllm/bin/uv"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # vLLM EngineCore needs ninja
export TMPDIR

# --- env bootstrap (fresh $TMPDIR in batch jobs) -----------------------------
"$PY" experiments/patches/patch_vllm_pixtral.py

if [ ! -x "${TMPDIR}/scoring/bin/python" ]; then
  echo "[special] creating scoring venv at \$TMPDIR/scoring"
  "$UV" venv "${TMPDIR}/scoring" --python 3.12
  "$UV" pip install --python "${TMPDIR}/scoring/bin/python" \
      sacrebleu unbabel-comet "setuptools<81"
fi
"${TMPDIR}/scoring/bin/python" -c "import comet, sacrebleu; print('[special] scoring env OK')"

# --- translate phase (S / C1 / C3 / C3L; existing conditions auto-skipped) --
"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
    --conditions S C1 --phase translate

"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models mistralai/Ministral-3-8B-Instruct-2512-BF16 google/gemma-4-E2B-it \
    --conditions S C3L --phase translate

"$PY" -u experiments/exp3_cross_family/run_exp3_cross_family.py \
    --models Qwen/Qwen3.5-9B \
    --conditions S C3 C3L --phase translate

# --- scoring + bootstrap contrasts (scoring env) ----------------------------
"${TMPDIR}/scoring/bin/python" -u experiments/exp3_cross_family/score_special.py

echo "[special] done — see ANALYSIS.md (follow-up section) + results/exp3_special.md"
