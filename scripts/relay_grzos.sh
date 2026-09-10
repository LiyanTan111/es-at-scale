#!/bin/bash
# Interactive-relay driver for an Angle B (GRZO-on-surrogate) run.
# Same relay pattern as relay_grzo.sh but drives train_grzo_surrogate.py.
#
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=grzos-... LR=5e-4 \
#     nohup bash scripts/relay_grzos.sh > logs/relay/<name>.driver.log 2>&1 & disown
set -uo pipefail

MODEL="${MODEL:?set MODEL}"
EXPNAME="${EXPNAME:?set EXPNAME (fixed, no timestamp -- resume keys on it)}"
TASK="${TASK:-countdown}"
TRAIN_DS="${TRAIN_DS:-datasets/train/countdown}"
EVAL_DS="${EVAL_DS:-datasets/evaluation_suite/countdown/}"
EVAL_SUBSETS="${EVAL_SUBSETS:-}"
SIGMA="${SIGMA:-1e-3}"
LR="${LR:-5e-4}"
TEMP="${TEMP:-1.0}"
DELTA_NORM="${DELTA_NORM:-zscore}"
MIN_DIRS="${MIN_DIRS:-1}"
DAPO_TARGET="${DAPO_TARGET:-0}"
DAPO_DRAW="${DAPO_DRAW:-0}"
PAIRS_PER_DIR="${PAIRS_PER_DIR:-1}"
DIRS_PER_STEP="${DIRS_PER_STEP:-0}"
CD_REWARD="${CD_REWARD:-binary}"
MATH_REWARD="${MATH_REWARD:-binary}"
MARGIN_TAU="${MARGIN_TAU:-0.1}"
MARGIN_TAU_END="${MARGIN_TAU_END:-0.02}"
LR_SCHEDULE="${LR_SCHEDULE:-const}"
G="${G:-8}"; B="${B:-8}"
ITERS="${ITERS:-300}"; EVAL_FREQ="${EVAL_FREQ:-25}"
MAXTOK="${MAXTOK:-1024}"; SEED="${SEED:-42}"
ENGINES="${ENGINES:-4}"; GPUS_PER_ENGINE="${GPUS_PER_ENGINE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
ANCHOR_RATCHET="${ANCHOR_RATCHET:-0}"
RATCHET_DROP="${RATCHET_DROP:-0.02}"; RATCHET_PATIENCE="${RATCHET_PATIENCE:-2}"
RATCHET_WARMUP="${RATCHET_WARMUP:-0}"
REPLAY_FRAC="${REPLAY_FRAC:-0}"; REPLAY_ADV="${REPLAY_ADV:-1.0}"
REPLAY_CAP_PER_PROMPT="${REPLAY_CAP_PER_PROMPT:-4}"; REPLAY_MAX="${REPLAY_MAX:-512}"
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

echo "[RELAY $EXPNAME] start: model=$MODEL sigma=$SIGMA lr=$LR T=$TEMP G=$G B=$B iters=$ITERS wall=$WALL"

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
            export GRZO_GPU_MEM_UTIL=$GPU_MEM_UTIL
            export ANCHOR_RATCHET=$ANCHOR_RATCHET
            export RATCHET_DROP=$RATCHET_DROP RATCHET_PATIENCE=$RATCHET_PATIENCE
            export RATCHET_WARMUP=$RATCHET_WARMUP
            export REPLAY_FRAC=$REPLAY_FRAC REPLAY_ADV=$REPLAY_ADV
            export REPLAY_CAP_PER_PROMPT=$REPLAY_CAP_PER_PROMPT REPLAY_MAX=$REPLAY_MAX
            python es_at_scale/train_grzo_surrogate.py \
                --model-name $MODEL --sigma $SIGMA --lr $LR \
                --task $TASK --train-dataset $TRAIN_DS --eval-dataset $EVAL_DS \
                ${EVAL_SUBSETS:+--eval-subsets $EVAL_SUBSETS} \
                --delta-norm $DELTA_NORM --min-directions $MIN_DIRS \
                --dapo-target-groups $DAPO_TARGET --dapo-draw $DAPO_DRAW \
                --pairs-per-direction $PAIRS_PER_DIR --directions-per-step $DIRS_PER_STEP \
                --countdown-reward $CD_REWARD --math-reward $MATH_REWARD \
                --margin-tau $MARGIN_TAU --margin-tau-end $MARGIN_TAU_END \
                --lr-schedule $LR_SCHEDULE \
                --rollout-temperature $TEMP --group-size $G \
                --batch-size $B --mini-batch-size $B \
                --n-iterations $ITERS --eval-freq $EVAL_FREQ --max-tokens $MAXTOK \
                --n-vllm-engines $ENGINES --n-gpu-per-vllm-engine $GPUS_PER_ENGINE \
                --use-gpus 0,1,2,3 --seed $SEED \
                --logging wandb --wandb-project grzo-rlvr --save-best-models \
                --experiment-name $EXPNAME --output-directory $OUT --resume
         " >> "$LOGDIR/$EXPNAME.log" 2>&1

    echo "[RELAY $EXPNAME] srun returned (finished or wall hit); releasing $JID"
    scancel "$JID" 2>/dev/null || true
    sleep 20
done
