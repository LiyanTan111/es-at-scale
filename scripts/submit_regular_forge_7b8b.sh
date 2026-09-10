#!/bin/bash
# FORGE countdown matrix, 7B/8B rows (regular queue). TP2 x2 engines — TP1 is
# structurally impossible on 40GB A100s for 7B+ (model + master copy + vLLM
# workspace + noise + CUDA ctx > 40G). util 0.6 validated by smoke512b-qwen7b.
# Usage: bash submit_regular_forge_7b8b.sh [7b|8b|both]
set -euo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale
WHICH="${1:-7b}"

ENVP='module load python/3.12-26.1.0 && source /pscratch/sd/l/liyantan/es-at-scale/.venv/bin/activate && export HF_HOME=/pscratch/sd/l/liyantan/hf_cache VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false GRZO_GPU_MEM_UTIL=0.6'

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

forge_cmd() {  # <model> <expname>
    echo "python es_at_scale/train_grzo_surrogate.py --model-name $1 \
--sigma 1e-3 --lr 5e-4 --delta-norm zscore --min-directions 64 \
--dapo-target-groups 8 --dapo-draw 32 --rollout-temperature 1.0 --group-size 8 \
--batch-size 8 --mini-batch-size 8 --n-iterations 4000 --eval-freq 25 --max-tokens 512 \
--n-vllm-engines 2 --n-gpu-per-vllm-engine 2 --use-gpus 0,1,2,3 --seed 42 \
--logging wandb --wandb-project grzo-rlvr \
--save-best-models --output-directory /pscratch/sd/l/liyantan/runs/grzo --resume \
--experiment-name $2"
}

if [[ "$WHICH" == "7b" || "$WHICH" == "both" ]]; then
    echo "== FORGE matrix: Qwen-7B =="
    submit_chain f512q7b 5 "$(forge_cmd Qwen/Qwen2.5-7B-Instruct forge512-qwen7b)"
fi
if [[ "$WHICH" == "8b" || "$WHICH" == "both" ]]; then
    echo "== FORGE matrix: Llama-8B =="
    submit_chain f512l8b 6 "$(forge_cmd meta-llama/Llama-3.1-8B-Instruct forge512-llama8b)"
fi
