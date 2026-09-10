#!/bin/bash
# Waits for Lane C (qwen7b, last replication run) to finish, then keeps that
# interactive slot on paper-critical FORGE work: margin-tiebreak @ 512,
# full 3M-rollout budget. Same expname as the f512mtb regular chain (held);
# whatever progress lands here is banked for the chain via --resume.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
until grep -q "\[LANE C\] ALL DONE" logs/relay/lane_C_qwen7b.driver.log 2>/dev/null; do sleep 300; done
echo "[LANE G] Lane C finished; starting forge512-1p5b-mtb relay"
MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=forge512-1p5b-mtb ITERS=4000 \
  TASK=countdown TRAIN_DS=datasets/train/countdown \
  EVAL_DS=datasets/evaluation_suite/countdown/ EVAL_SUBSETS= \
  SIGMA=1e-3 LR=5e-4 DELTA_NORM=zscore MIN_DIRS=64 \
  DAPO_TARGET=8 DAPO_DRAW=32 TEMP=1.0 G=8 B=8 \
  CD_REWARD=margin-tiebreak MARGIN_TAU=0.2 \
  EVAL_FREQ=25 MAXTOK=512 SEED=42 \
  OUT=/pscratch/sd/l/liyantan/runs/grzo \
  bash scripts/relay_grzos.sh
echo "[LANE G] ALL DONE"
