#!/bin/bash
# Evaluate an HF-format checkpoint directory (e.g. a GRPO checkpoint-N) with the exact
# FORGE/ES Countdown eval: vLLM greedy, 512 tokens, countdown_eval (2000 prompts).
#   bash scripts/eval_hf_ckpt.sh <ckpt_dir> <gpu_index> [expname]
set -uo pipefail
CKPT="${1:?ckpt dir}"; GPU="${2:?gpu index}"; EXP="${3:-eval-$(basename "$CKPT")}"
[[ "$GPU" == *0* ]] && { echo "GPU 0 is banned"; exit 2; }
cd /data/liyan/es-at-scale
source .venv/bin/activate
export HF_HOME=/data/liyan/hf-cache TOKENIZERS_PARALLELISM=false VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1
export RAY_TMPDIR=/data/liyan/ray_tmp NCCL_P2P_DISABLE=1
export FORGE_MASTER_PORT=$(( 30000 + RANDOM % 25000 ))
python es_at_scale/train_grzo_surrogate.py --model-name "$CKPT" --task countdown \
  --n-iterations 0 --eval-freq 1 --max-tokens 512 --n-vllm-engines 1 --use-gpus "$GPU" \
  --logging none --experiment-name "$EXP" --output-directory /data/liyan/runs/grpo-eval 2>&1 \
  | grep -E "answer_acc|Traceback|Error" | tail -3
