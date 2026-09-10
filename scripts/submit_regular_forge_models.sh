#!/bin/bash
# FORGE model-scaling matrix on the regular queue (user directive 2026-07-19:
# FORGE must run every experiment in the ES-at-Scale paper -> the 7-model
# Countdown table needs FORGE rows, full 3M-rollout budget, 512 protocol).
# Interactive covers 1.5B (running) + 0.5B (tonight); this queues 1B + both 3Bs.
# 7B/8B chains are submitted separately after the memory smokes pass.
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

forge_cmd() {  # <model> <expname>
    echo "python es_at_scale/train_grzo_surrogate.py --model-name $1 \
--sigma 1e-3 --lr 5e-4 --delta-norm zscore --min-directions 64 \
--dapo-target-groups 8 --dapo-draw 32 --rollout-temperature 1.0 --group-size 8 \
--batch-size 8 --mini-batch-size 8 --n-iterations 4000 --eval-freq 25 --max-tokens 512 \
--n-vllm-engines 4 --use-gpus 0,1,2,3 --seed 42 --logging wandb --wandb-project grzo-rlvr \
--save-best-models --output-directory /pscratch/sd/l/liyantan/runs/grzo --resume \
--experiment-name $2"
}

echo "== FORGE model matrix: 1B + 3B pair =="
submit_chain f512l1b 3 "$(forge_cmd meta-llama/Llama-3.2-1B-Instruct forge512-llama1b)"
submit_chain f512q3b 5 "$(forge_cmd Qwen/Qwen2.5-3B-Instruct forge512-qwen3b)"
submit_chain f512l3b 5 "$(forge_cmd meta-llama/Llama-3.2-3B-Instruct forge512-llama3b)"
