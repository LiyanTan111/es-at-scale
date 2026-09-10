#!/bin/bash
# Regular-queue 512-protocol campaign (user directive 2026-07-19: everything on
# regular must be citable = exact paper protocol + full 3M-rollout budget).
#   ES chains   : 500 iters  (paper budget), seeds 43/44  -> E3 seed variance
#   FORGE chains: 4000 iters (~3M rollouts, equal budget), seeds 43/44 +
#                 margin-tiebreak + tau-anneal variants
# Each chain = N x 24h chunks linked with --dependency=afterany, all --resume.
set -euo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

ENVP='module load python/3.12-26.1.0 && source /pscratch/sd/l/liyantan/es-at-scale/.venv/bin/activate && export HF_HOME=/pscratch/sd/l/liyantan/hf_cache VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false'

submit_chain() {  # <chain-name> <n-chunks> <python-cmd>
    local name="$1" n="$2" cmd="$3" dep="" jid
    for i in $(seq 1 "$n"); do
        jid=$(sbatch --parsable -A m4788_g -C gpu -q regular -N 1 \
                     --gpus-per-node=4 -c 128 -t 24:00:00 \
                     -J "${name}-${i}" -o "logs/${name}_${i}_%j.out" \
                     ${dep:+--dependency=afterany:$dep} \
                     --wrap "$ENVP && $cmd")
        echo "  $name-$i -> $jid${dep:+ (after $dep)}"
        dep="$jid"
    done
}

ES_BASE="python es_at_scale/train_es_relay.py --model-name Qwen/Qwen2.5-1.5B-Instruct \
--sigma 1e-3 --population-size 30 --task countdown \
--train-dataset datasets/train/countdown --eval-dataset datasets/evaluation_suite/countdown/ \
--batch-size 200 --mini-batch-size 200 --n-iterations 500 --eval-freq 25 --max-tokens 512 \
--n-vllm-engines 4 --use-gpus 0,1,2,3 --logging wandb --wandb-project grzo-rlvr \
--save-best-models --output-directory /pscratch/sd/l/liyantan/runs/grzo --resume"

FORGE_BASE="python es_at_scale/train_grzo_surrogate.py --model-name Qwen/Qwen2.5-1.5B-Instruct \
--sigma 1e-3 --lr 5e-4 --delta-norm zscore --min-directions 64 \
--dapo-target-groups 8 --dapo-draw 32 --rollout-temperature 1.0 --group-size 8 \
--batch-size 8 --mini-batch-size 8 --n-iterations 4000 --eval-freq 25 --max-tokens 512 \
--n-vllm-engines 4 --use-gpus 0,1,2,3 --logging wandb --wandb-project grzo-rlvr \
--save-best-models --output-directory /pscratch/sd/l/liyantan/runs/grzo --resume"

echo "== E3 seed variance: ES @ paper protocol =="
submit_chain es512s43 2 "$ES_BASE --seed 43 --experiment-name es512-qwen1p5b-s43"
submit_chain es512s44 2 "$ES_BASE --seed 44 --experiment-name es512-qwen1p5b-s44"

echo "== E3 seed variance: FORGE @ equal budget =="
submit_chain f512s43 5 "$FORGE_BASE --seed 43 --experiment-name forge512-1p5b-s43"
submit_chain f512s44 5 "$FORGE_BASE --seed 44 --experiment-name forge512-1p5b-s44"

echo "== FORGE reward variants @ 512, equal budget =="
submit_chain f512mtb 5 "$FORGE_BASE --seed 42 --countdown-reward margin-tiebreak \
--margin-tau 0.2 --experiment-name forge512-1p5b-mtb"
submit_chain f512ann 5 "$FORGE_BASE --seed 42 --countdown-reward margin-anneal \
--margin-tau 0.2 --margin-tau-end 0.02 --lr-schedule cosine \
--experiment-name forge512-1p5b-anneal"
