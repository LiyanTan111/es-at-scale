#!/bin/bash
# FORGE model-scaling lane (slot 2), v2 after the 7B OOM smoke.
# Root cause: vLLM's KV reservation (util 0.7) left no room for the GPU-resident
# master copy on 7B+. Memory budget per 40GB A100:
#   7B TP1 util 0.45: vLLM 17.8G (14.2 model + 3.6 KV) + master 14.2 + 2x fp32
#                     noise temp ~2.2 + overhead  -> fits, keeps 4 engines
#   8B TP2 util 0.6 : per GPU vLLM 23.7G (8 shard + 15.7 KV) + master shard 8
#                     + noise ~1.1                -> comfortable, 2 engines
# Stages: 3-iter smokes on 7B and 8B -> gate chain submission; then 0.5B full
# budget (defaults: 4 engines, util 0.7).
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

FORGE_ENV="TASK=countdown TRAIN_DS=datasets/train/countdown \
EVAL_DS=datasets/evaluation_suite/countdown/ EVAL_SUBSETS= \
SIGMA=1e-3 LR=5e-4 DELTA_NORM=zscore MIN_DIRS=64 \
DAPO_TARGET=8 DAPO_DRAW=32 TEMP=1.0 G=8 B=8 \
CD_REWARD=binary EVAL_FREQ=25 MAXTOK=512 SEED=42 \
OUT=/pscratch/sd/l/liyantan/runs/grzo"

echo "[LANE G] smoke: smoke512b-qwen7b (TP2 x2 engines, util 0.6 — TP1 structurally impossible: 14.2 model + 14.2 master + 4.5 vLLM workspace + 2 noise + 1.8 ctx > 40G)"
env $FORGE_ENV MODEL=Qwen/Qwen2.5-7B-Instruct EXPNAME=smoke512b-qwen7b ITERS=3 \
    ENGINES=2 GPUS_PER_ENGINE=2 GPU_MEM_UTIL=0.6 bash scripts/relay_grzos.sh

echo "[LANE G] smoke: smoke512b-llama8b (TP2 x2 engines, util 0.55)"
env $FORGE_ENV MODEL=meta-llama/Llama-3.1-8B-Instruct EXPNAME=smoke512b-llama8b ITERS=3 \
    ENGINES=2 GPUS_PER_ENGINE=2 GPU_MEM_UTIL=0.55 bash scripts/relay_grzos.sh

echo "[LANE G] smokes done (check logs/relay/smoke512b-*.log before submitting 7B/8B chains)"

echo "[LANE G] starting forge512-qwen0p5b full-budget relay"
env $FORGE_ENV MODEL=Qwen/Qwen2.5-0.5B-Instruct EXPNAME=forge512-qwen0p5b ITERS=4000 \
    bash scripts/relay_grzos.sh
echo "[LANE G] ALL DONE"
