#!/bin/bash
# FORGE conciseness campaign (ES-paper Section 4.2 analog), regular queue.
# Paper ES budget: 1000 iters x pop30 x 2 prompts = 60k generations.
# FORGE equal budget: B=2 x G=8 = 16 gens/step -> 3750 iters = 60k generations.
# 4 seeds to mirror the paper's 4-run reliability analysis (Table 2).
# 7B layout: TP2 x2 engines, util 0.6 (validated by smoke512b-qwen7b 2026-07-19).
set -euo pipefail
cd /global/cfs/cdirs/m4788/liyantan/es-at-scale

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

concise_cmd() {  # <seed> <expname>
    echo "python es_at_scale/train_grzo_surrogate.py --model-name Qwen/Qwen2.5-7B-Instruct \
--task conciseness --train-dataset datasets/train/conciseness \
--eval-dataset datasets/evaluation_suite/conciseness \
--sigma 1e-3 --lr 5e-4 --delta-norm zscore --min-directions 64 \
--dapo-target-groups 0 --rollout-temperature 1.0 --group-size 8 \
--batch-size 2 --mini-batch-size 8 --n-iterations 3750 --eval-freq 25 --max-tokens 100 \
--n-vllm-engines 2 --n-gpu-per-vllm-engine 2 --use-gpus 0,1,2,3 --seed $1 --logging wandb --wandb-project grzo-rlvr \
--save-best-models --output-directory /pscratch/sd/l/liyantan/runs/grzo --resume \
--experiment-name $2"
}

echo "== FORGE conciseness x 4 seeds (7B, equal 60k-generation budget) =="
submit_chain fcons42 2 "$(concise_cmd 42 forge-concise-7b-s42)"
submit_chain fcons43 2 "$(concise_cmd 43 forge-concise-7b-s43)"
submit_chain fcons44 2 "$(concise_cmd 44 forge-concise-7b-s44)"
submit_chain fcons45 2 "$(concise_cmd 45 forge-concise-7b-s45)"
