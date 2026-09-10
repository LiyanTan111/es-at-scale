#!/bin/bash
# Interactive-relay driver for the ES-original positive control.
set -uo pipefail

MODEL="${MODEL:?set MODEL}"
EXPNAME="${EXPNAME:?set EXPNAME (fixed -- resume keys on it)}"
TASK="${TASK:-countdown}"
TRAIN_DS="${TRAIN_DS:-datasets/train/countdown}"
EVAL_DS="${EVAL_DS:-datasets/evaluation_suite/countdown/}"
EVAL_SUBSETS="${EVAL_SUBSETS:-}"
SIGMA="${SIGMA:-1e-3}"
POP="${POP:-30}"
B="${B:-200}"
ITERS="${ITERS:-150}"; EVAL_FREQ="${EVAL_FREQ:-10}"
MAXTOK="${MAXTOK:-1024}"; SEED="${SEED:-42}"
WALL="${WALL:-04:00:00}"
OUT="${OUT:-$SCRATCH/runs/grzo}"

REPO=/global/cfs/cdirs/m4788/liyantan/es-at-scale
VENV=/pscratch/sd/l/liyantan/es-at-scale/.venv
LOGDIR="$REPO/logs/relay"; mkdir -p "$LOGDIR"
PROG="$OUT/$EXPNAME/latest/progress.json"

last_iter() {
    [[ -f "$PROG" ]] || { echo -1; return; }
    local n
    n=$(grep -oE '"iteration"[: ]+[0-9]+' "$PROG" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "${n:--1}"
}
is_done() { [[ "$(last_iter)" -ge $((ITERS - 1)) ]]; }

echo "[RELAY $EXPNAME] start: model=$MODEL sigma=$SIGMA pop=$POP B=$B iters=$ITERS wall=$WALL"

while true; do
    if is_done; then echo "[RELAY $EXPNAME] DONE at iter $(last_iter)"; break; fi
    echo "[RELAY $EXPNAME] resume-from iter $(last_iter); allocating $WALL node..."
    out=$(salloc --no-shell -A m4788_g -C gpu -q "${QOS:-interactive}" -t "$WALL" -N 1 \
                 --gpus-per-node=4 -c 128 -J "relay-$EXPNAME" 2>&1)
    JID=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | awk '{print $NF}')
    if [[ -z "$JID" ]]; then echo "[RELAY $EXPNAME] salloc failed: $out"; sleep 120; continue; fi

    ws=$(date +%s); ok=1
    while true; do
        st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
        [[ "$st" == "RUNNING" ]] && break
        if (( $(date +%s) - ws > ${PEND_PATIENCE:-900} )); then
            echo "[RELAY $EXPNAME] not RUNNING in 900s (state=$st); cancel+retry"
            scancel "$JID" 2>/dev/null; ok=0; break
        fi
        sleep 10
    done
    [[ "$ok" -eq 0 ]] && { sleep 30; continue; }

    NODE=$(scontrol show hostnames "$(squeue -j "$JID" -h -o '%N' | tr -d ' ')" | head -1)
    echo "[RELAY $EXPNAME] RUNNING jid=$JID node=$NODE"

    srun --jobid="$JID" --nodes=1 --nodelist="$NODE" --gpus=4 --ntasks=1 \
         --cpus-per-task=128 --exclusive \
         bash -c "
            module load python/3.12-26.1.0
            source $VENV/bin/activate
            cd $REPO
            export HF_HOME=/pscratch/sd/l/liyantan/hf_cache
            export TOKENIZERS_PARALLELISM=false
            export VLLM_ENABLE_V1_MULTIPROCESSING=0
            export PYTHONUNBUFFERED=1
            python es_at_scale/train_es_relay.py \
                --model-name $MODEL --sigma $SIGMA --population-size $POP \
                --task $TASK --train-dataset $TRAIN_DS --eval-dataset $EVAL_DS \
                ${EVAL_SUBSETS:+--eval-subsets $EVAL_SUBSETS} \
                --batch-size $B --mini-batch-size $B \
                --n-iterations $ITERS --eval-freq $EVAL_FREQ --max-tokens $MAXTOK \
                --n-vllm-engines 4 --use-gpus 0,1,2,3 --seed $SEED \
                --logging wandb --wandb-project grzo-rlvr --save-best-models \
                --experiment-name $EXPNAME --output-directory $OUT --resume
         " >> "$LOGDIR/$EXPNAME.log" 2>&1

    echo "[RELAY $EXPNAME] srun returned; releasing $JID"
    scancel "$JID" 2>/dev/null || true
    sleep 20
done
