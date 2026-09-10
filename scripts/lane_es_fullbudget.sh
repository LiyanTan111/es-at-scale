#!/bin/bash
# Extend every ES-512 replication run to the paper's FULL budget: 3M rollouts
# = 500 iterations (user directive 2026-07-18: paper-grade, 100% budget,
# identical comparisons). Resume machinery picks each model up from its latest
# checkpoint; already-finished 500-iter models are skipped by is_done.
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
LANE="${LANE:?set LANE=A|B}"

run_es() {  # model expname (ITERS fixed at 500)
    echo "[LANE $LANE] ==== starting $2 ($1 -> 500 iters) ===="
    MODEL="$1" EXPNAME="$2" ITERS=500 \
      TASK=countdown TRAIN_DS=datasets/train/countdown \
      EVAL_DS=datasets/evaluation_suite/countdown/ EVAL_SUBSETS= \
      SIGMA=1e-3 POP=30 B=200 EVAL_FREQ=25 MAXTOK=512 \
      OUT=/pscratch/sd/l/liyantan/runs/grzo \
      bash scripts/relay_es.sh
    echo "[LANE $LANE] ==== finished $2 ===="
}

if [[ "$LANE" == "A" ]]; then
    run_es Qwen/Qwen2.5-3B-Instruct         es512-qwen3b
    run_es meta-llama/Llama-3.2-3B-Instruct es512-llama3b
else
    run_es meta-llama/Llama-3.1-8B-Instruct es512-llama8b
    run_es Qwen/Qwen2.5-7B-Instruct         es512-qwen7b
fi
echo "[LANE $LANE] ALL DONE"
