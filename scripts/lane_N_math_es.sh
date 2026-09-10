#!/bin/bash
# When the 1B lane frees its interactive slot, run the ES-MATH duel arm there.
# ES-MATH to 500 iters = 3M generations = 3.3x the budget FORGE-MATH gets --
# the "even at 3x budget ES has no math trend" anchor. Same expname as held
# esmath regular chain.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
until grep -q "DONE at iter 4000" logs/relay/lane_H_llama1b.driver.log 2>/dev/null; do sleep 300; done
echo "[LANE N] 1B slot free; starting ES-MATH relay"
MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=es-orig-math-1p5b-pop30-bs200 ITERS=500 \
  TASK=math TRAIN_DS=datasets/train/math_lvl3to5_8k \
  EVAL_DS=datasets/evaluation_suite/math/ EVAL_SUBSETS=math500 \
  SIGMA=1e-3 POP=30 B=200 EVAL_FREQ=25 MAXTOK=1024 SEED=42 \
  OUT=/pscratch/sd/l/liyantan/runs/grzo \
  bash scripts/relay_es.sh
echo "[LANE N] ALL DONE"
