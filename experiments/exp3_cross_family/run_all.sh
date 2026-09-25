#!/bin/bash
# run_all.sh - Exp 3 orchestration: per-model (download -> translate), then
# score everything that exists. A failed download/run skips to the next model.
# Usage: nohup bash experiments/exp3_cross_family/run_all.sh > $TMPDIR/exp3_run.log 2>&1 &

set -u
cd "$(dirname "$0")/../../"   # repo root

PY="${TMPDIR:?TMPDIR must be set}/vllm/bin/python"
SCRIPT="experiments/exp3_cross_family/run_exp3_cross_family.py"
export PATH="${TMPDIR}/vllm/bin:${PATH}"   # EngineCore needs ninja on PATH

# re-apply env fixups (the $TMPDIR vLLM env is recreated every session)
"$PY" experiments/patches/patch_vllm_pixtral.py || exit 1

MODELS=(
  meta-llama/Llama-3.2-1B-Instruct
  mistralai/Ministral-3-3B-Instruct-2512-BF16
  meta-llama/Llama-3.2-3B-Instruct
  google/gemma-4-E2B-it
  mistralai/Ministral-3-8B-Instruct-2512-BF16
  google/gemma-4-E4B-it
  meta-llama/Llama-3.1-8B-Instruct
  google/gemma-4-12B-it
  mistralai/Ministral-3-14B-Instruct-2512-BF16
  google/gemma-4-31B-it
)

for m in "${MODELS[@]}"; do
  echo "[exp3-orchestrator] downloading $m"
  if ! "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"; then
    echo "[exp3-orchestrator] DOWNLOAD FAILED for $m — skipping"
    continue
  fi
  echo "[exp3-orchestrator] running $m"
  if ! "$PY" -u "$SCRIPT" --models "$m" --phase translate; then
    echo "[exp3-orchestrator] RUN FAILED for $m — skipping"
  fi
done

echo "[exp3-orchestrator] scoring phase"
"$PY" -u "$SCRIPT" --phase score
echo "[exp3-orchestrator] done"
