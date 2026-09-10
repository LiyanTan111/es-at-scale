#!/bin/bash
# Waits for Lane A (3B pair) to finish, then takes over its interactive slot to
# run the qwen7b full-budget extension in parallel with Lane B's llama-8B (the
# long pole). Lane B will later skip qwen7b via relay_es.sh's is_done check.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
until grep -q "ALL DONE" logs/relay/lane_A_full.driver.log 2>/dev/null; do sleep 300; done
echo "[LANE C] Lane A finished; starting qwen7b relay"
MODEL=Qwen/Qwen2.5-7B-Instruct EXPNAME=es512-qwen7b ITERS=500 \
  TASK=countdown TRAIN_DS=datasets/train/countdown \
  EVAL_DS=datasets/evaluation_suite/countdown/ EVAL_SUBSETS= \
  SIGMA=1e-3 POP=30 B=200 EVAL_FREQ=25 MAXTOK=512 \
  OUT=/pscratch/sd/l/liyantan/runs/grzo \
  bash scripts/relay_es.sh
echo "[LANE C] ALL DONE"
