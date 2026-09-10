#!/bin/bash
# Validate the two new pieces before committing hours to a relay:
#   1. sharded eval  -> base 0.5B eval must match the known single-engine numbers
#                       (pass@1 0.0434 / answer_acc 0.0000)
#   2. save latest + --resume -> second run must resume from the saved iteration.
set -uo pipefail

JID="${JID:?}"; NODE="${NODE:?}"
REPO=/global/cfs/cdirs/m4788/liyantan/es-at-scale
VENV=/pscratch/sd/l/liyantan/es-at-scale/.venv
OUT=$SCRATCH/runs/grzo-val
LOG=$REPO/logs/validate_grzo.log
mkdir -p "$REPO/logs"
rm -rf "$OUT"   # clean slate so --resume genuinely starts fresh the first time

srun --jobid="$JID" --nodes=1 --nodelist="$NODE" --gpus=4 --ntasks=1 \
     --cpus-per-task=128 --exclusive \
     bash -c "
        module load python/3.12-26.1.0
        source $VENV/bin/activate
        cd $REPO
        export HF_HOME=/pscratch/sd/l/liyantan/hf_cache
        export TOKENIZERS_PARALLELISM=false
        export VLLM_ENABLE_V1_MULTIPROCESSING=0

        COMMON='--model-name Qwen/Qwen2.5-0.5B-Instruct --grzo-mode single_point \
                --group-size 8 --batch-size 8 --mini-batch-size 8 --sigma 1e-3 \
                --n-iterations 3 --eval-freq 2 --max-tokens 512 --n-vllm-engines 4 \
                --use-gpus 0,1,2,3 --logging none --output-directory $OUT \
                --experiment-name grzo-val'

        echo '########## RUN 1: FRESH ##########'
        python es_at_scale/train_grzo.py \$COMMON --resume

        echo '########## RUN 2: RESUME ##########'
        python es_at_scale/train_grzo.py \$COMMON --resume
     " 2>&1 | tee "$LOG"
