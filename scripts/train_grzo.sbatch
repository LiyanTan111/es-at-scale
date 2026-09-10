#!/bin/bash
# GRZO-RLVR real run (single node, 4x A100). Env-var driven; override at submit.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen2.5-0.5B-Instruct,MODE=single_point,SIGMA=1e-3 \
#          scripts/train_grzo.sbatch
#
#SBATCH -A m4788_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 12:00:00
#SBATCH -J grzo
#SBATCH -o logs/grzo_%j.out
#SBATCH -e logs/grzo_%j.err

set -uo pipefail
module load python/3.12-26.1.0
source /pscratch/sd/l/liyantan/es-at-scale/.venv/bin/activate
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

export HF_HOME=/pscratch/sd/l/liyantan/hf_cache
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export PYTHONUNBUFFERED=1   # sbatch runs python w/o a tty -> unbuffer so .out flushes live

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
MODE="${MODE:-single_point}"
REWARD_MODE="${REWARD_MODE:-shaped}"
SIGMA="${SIGMA:-1e-3}"
G="${G:-8}"
B="${B:-8}"
ITERS="${ITERS:-400}"
EVAL_FREQ="${EVAL_FREQ:-25}"
MAXTOK="${MAXTOK:-1024}"
SEED="${SEED:-42}"
OUT="${OUT:-$SCRATCH/runs/grzo}"
EXPNAME="${EXPNAME:-}"          # fixed name enables --resume across requeues

python es_at_scale/train_grzo.py \
    --model-name "$MODEL" \
    --grzo-mode "$MODE" \
    --reward-mode "$REWARD_MODE" \
    --group-size "$G" \
    --batch-size "$B" \
    --mini-batch-size "$B" \
    --sigma "$SIGMA" \
    --n-iterations "$ITERS" \
    --eval-freq "$EVAL_FREQ" \
    --max-tokens "$MAXTOK" \
    --n-vllm-engines 4 \
    --use-gpus 0,1,2,3 \
    --seed "$SEED" \
    --logging wandb \
    --wandb-project grzo-rlvr \
    --save-best-models \
    ${EXPNAME:+--experiment-name "$EXPNAME"} \
    --resume \
    --output-directory "$OUT"
