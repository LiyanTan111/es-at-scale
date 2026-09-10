#!/bin/bash
# When the anneal lane frees its interactive slot, run the FORGE-MATH duel arm
# there instead of waiting days in the regular queue. Same expname as the
# held fmathmtb regular chain -> progress banked, resumable either way.
# FORGE-MATH equal budget: 3300 iters (300 already done) ~= 900k generations
# = the budget ES already spent on math for no gain.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
until grep -q "DONE at iter 4000" logs/relay/lane_F2_anneal.driver.log 2>/dev/null; do sleep 300; done
echo "[LANE M] anneal slot free; starting FORGE-MATH mtb relay"
MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=grzos-math-mtb-tau05 ITERS=3300 \
  TASK=math TRAIN_DS=datasets/train/math_lvl3to5_8k \
  EVAL_DS=datasets/evaluation_suite/math/ EVAL_SUBSETS=math500 \
  MATH_REWARD=margin-tiebreak MARGIN_TAU=0.5 \
  SIGMA=1e-3 LR=5e-4 DELTA_NORM=zscore MIN_DIRS=64 \
  DAPO_TARGET=8 DAPO_DRAW=32 TEMP=1.0 G=8 B=8 \
  EVAL_FREQ=25 MAXTOK=1024 SEED=42 \
  OUT=/pscratch/sd/l/liyantan/runs/grzo \
  bash scripts/relay_grzos.sh
echo "[LANE M] ALL DONE"
