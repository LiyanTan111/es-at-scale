#!/bin/bash
# Math early-window significance seeds (revised paper story): the one
# surviving quantitative FORGE win is the early-budget window on MATH
# (peak +3.4pp @ ~81k gens while ES sits at base). Two extra seeds x 500
# iters (~136k gens) quantify seed variance of that peak; the equal-budget
# ES comparison point comes from the existing es-orig-math full curve.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

MATH_ENV="TASK=math TRAIN_DS=datasets/train/math_lvl3to5_8k \
EVAL_DS=datasets/evaluation_suite/math/ EVAL_SUBSETS=math500 \
MATH_REWARD=margin-tiebreak MARGIN_TAU=0.5 \
SIGMA=1e-3 LR=5e-4 DELTA_NORM=zscore MIN_DIRS=64 \
DAPO_TARGET=8 DAPO_DRAW=32 TEMP=1.0 G=8 B=8 \
EVAL_FREQ=25 MAXTOK=1024 ITERS=500 \
OUT=/pscratch/sd/l/liyantan/runs/grzo MODEL=Qwen/Qwen2.5-1.5B-Instruct"

for S in 43 44; do
    echo "[LANE Q] math-mtb seed $S"
    env $MATH_ENV SEED=$S EXPNAME=grzos-math-mtb-s$S bash scripts/relay_grzos.sh
done
echo "[LANE Q] ALL DONE"
