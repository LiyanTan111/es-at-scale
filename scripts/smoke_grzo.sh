#!/bin/bash
# GRZO smoke test: run a few iterations on 1 interactive node (4x A100) to confirm
# the code runs end-to-end, evals produce an accuracy number, and per-step
# reward/advantage variance is logged. NOT a method go/no-go -- just a code check.
#
# Usage (from an interactive alloc):
#   JID=<jobid> NODE=<node> bash scripts/smoke_grzo.sh [single_point|two_point]
set -uo pipefail

MODE="${1:-single_point}"
JID="${JID:?set JID to the interactive job id}"
NODE="${NODE:?set NODE to the allocated node}"

REPO=/global/cfs/cdirs/m4788/liyantan/es-at-scale
VENV=/pscratch/sd/l/liyantan/es-at-scale/.venv
OUT=$SCRATCH/runs/grzo-smoke
LOGDIR=$REPO/logs
mkdir -p "$LOGDIR" "$OUT"

srun --jobid="$JID" --nodes=1 --nodelist="$NODE" --gpus=4 --ntasks=1 \
     --cpus-per-task=128 --exclusive \
     bash -c "
        module load python/3.12-26.1.0
        source $VENV/bin/activate
        cd $REPO
        export HF_HOME=/pscratch/sd/l/liyantan/hf_cache
        export TOKENIZERS_PARALLELISM=false
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        python es_at_scale/train_grzo.py \
            --model-name Qwen/Qwen2.5-0.5B-Instruct \
            --grzo-mode $MODE \
            --group-size 8 \
            --batch-size 8 \
            --mini-batch-size 8 \
            --sigma 1e-3 \
            --n-iterations 6 \
            --eval-freq 5 \
            --max-tokens 512 \
            --n-vllm-engines 4 \
            --use-gpus 0,1,2,3 \
            --logging none \
            --output-directory $OUT
     " 2>&1 | tee "$LOGDIR/smoke_grzo_${MODE}.log"
