#!/bin/bash
# ES-paper replication campaign lane: runs a sequence of ES-original Countdown
# experiments back-to-back on interactive relays. Each relay_es.sh call blocks
# (allocating/resuming across 4h walls) until that experiment completes, then
# the next starts. Launch one lane per interactive slot.
#
#   LANE=A nohup bash scripts/lane_es_paper.sh > logs/relay/lane_A.driver.log 2>&1 &
set -uo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

LANE="${LANE:?set LANE=A|B}"

# PROTOCOL: MAXTOK=512 matching the ES paper exactly (user directive 2026-07-17:
# every paper-matching number must use the paper's setting). The earlier 1024
# runs (es-cd-*) are retained as internal-protocol data only.
run_es() {  # model expname iters
    echo "[LANE $LANE] ==== starting $2 ($1, $3 iters) ===="
    MODEL="$1" EXPNAME="$2" ITERS="$3" \
      TASK=countdown TRAIN_DS=datasets/train/countdown \
      EVAL_DS=datasets/evaluation_suite/countdown/ EVAL_SUBSETS= \
      SIGMA=1e-3 POP=30 B=200 EVAL_FREQ=25 MAXTOK=512 \
      OUT=/pscratch/sd/l/liyantan/runs/grzo \
      bash scripts/relay_es.sh
    echo "[LANE $LANE] ==== finished $2 ===="
}

if [[ "$LANE" == "A" ]]; then
    run_es Qwen/Qwen2.5-0.5B-Instruct       es512-qwen0p5b 500
    run_es meta-llama/Llama-3.2-1B-Instruct es512-llama1b  500
    run_es Qwen/Qwen2.5-3B-Instruct         es512-qwen3b   400
    run_es meta-llama/Llama-3.2-3B-Instruct es512-llama3b  400
else
    run_es Qwen/Qwen2.5-1.5B-Instruct       es512-qwen1p5b 500
    run_es Qwen/Qwen2.5-7B-Instruct         es512-qwen7b   300
    run_es meta-llama/Llama-3.1-8B-Instruct es512-llama8b  300
fi
echo "[LANE $LANE] ALL DONE"
