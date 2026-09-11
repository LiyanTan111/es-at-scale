#!/bin/bash
# Local (no-scheduler) driver for a FORGE run on lambda-scalar.
# Port of scripts/relay_grzos.sh with the Slurm layer removed: same env-var
# config, same python flags; runs train_grzo_surrogate.py with --resume in a
# restart loop until latest/progress.json reports the final iteration.
#
# Launch (survives terminal death):
#   MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=forge2-cd-v2-local \
#     setsid bash scripts/local_forge.sh > logs/local/forge2-cd-v2-local.driver.log 2>&1 &
# Stop:  kill -TERM $(cat $OUT/$EXPNAME/train.pid)     # driver PID: driver.pid
#   (never `pkill -f train_grzo_surrogate` -- that also kills the reward Pool workers)
#
# GPU rules on this box: GPU 0 is BANNED (temperature fault); max 4 cards.
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
GPUS="${GPUS:-1,2,3,4}"                      # physical nvidia-smi indices
GPUS_PER_ENGINE="${GPUS_PER_ENGINE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
ANCHOR_RATCHET="${ANCHOR_RATCHET:-0}"
RATCHET_DROP="${RATCHET_DROP:-0.02}"; RATCHET_PATIENCE="${RATCHET_PATIENCE:-2}"
RATCHET_WARMUP="${RATCHET_WARMUP:-0}"
REPLAY_FRAC="${REPLAY_FRAC:-0}"; REPLAY_ADV="${REPLAY_ADV:-1.0}"
REPLAY_CAP_PER_PROMPT="${REPLAY_CAP_PER_PROMPT:-4}"; REPLAY_MAX="${REPLAY_MAX:-512}"
LOGGING="${LOGGING:-wandb}"                  # wandb | none
RETRY_MAX="${RETRY_MAX:-20}"; RETRY_SLEEP="${RETRY_SLEEP:-60}"
OUT="${OUT:-/data/liyan/runs/grzo}"

REPO=/data/liyan/es-at-scale

# Run from a per-run SNAPSHOT of this script. bash reads scripts incrementally, so editing
# scripts/local_*.sh while a driver is running corrupts that driver (observed 2026-09-10:
# "syntax error near unexpected token `done'" at the end of a finished ES run). SETUP.md P12.
if [[ -z "${_RUNNER_SNAPSHOT:-}" ]]; then
    mkdir -p "$OUT/$EXPNAME"
    cp "$0" "$OUT/$EXPNAME/driver.sh"
    _RUNNER_SNAPSHOT=1 exec bash "$OUT/$EXPNAME/driver.sh" "$@"
fi
VENV=$REPO/.venv
LOGDIR="$REPO/logs/local"; mkdir -p "$LOGDIR" "$OUT/$EXPNAME"
PROG="$OUT/$EXPNAME/latest/progress.json"

# --- guards -----------------------------------------------------------------
if [[ ",$GPUS," == *",0,"* ]]; then
    echo "[LOCAL $EXPNAME] REFUSING: GPU 0 is banned on this machine (temperature fault)."; exit 2
fi
NGPU=$(tr ',' '\n' <<<"$GPUS" | grep -c .)
if (( NGPU > 4 )); then echo "[LOCAL $EXPNAME] REFUSING: >4 GPUs requested ($GPUS)."; exit 2; fi
ENGINES="${ENGINES:-$(( NGPU / GPUS_PER_ENGINE ))}"
SMI=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits) || { echo "[LOCAL $EXPNAME] REFUSING: nvidia-smi failed"; exit 2; }
for g in $(tr ',' ' ' <<<"$GPUS"); do
    used=$(awk -F', *' -v g="$g" '$1==g {print $2}' <<<"$SMI")
    used=${used:-NA}
    echo "[LOCAL $EXPNAME] GPU $g memory.used=${used} MiB"
    if [[ "$used" == "NA" ]] || (( used > 2000 )); then
        echo "[LOCAL $EXPNAME] REFUSING: GPU $g busy or unreadable (${used} MiB) -- someone else's job?"; exit 2
    fi
done

last_iter() {
    [[ -f "$PROG" ]] || { echo -1; return; }
    local n
    n=$(grep -oE '"iteration"[: ]+[0-9]+' "$PROG" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "${n:--1}"
}
is_done() { [[ "$(last_iter)" -ge $((ITERS - 1)) ]]; }

echo $$ > "$OUT/$EXPNAME/driver.pid"

# Unique NCCL rendezvous port per run (see es_trainer: FORGE_MASTER_PORT). Reserve a small
# range (port .. port+GPUS_PER_ENGINE) that no listener currently uses.
pick_port() {
    local p tries=0
    while (( tries < 50 )); do
        p=$(( 30000 + RANDOM % 25000 ))
        if ! ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":($p|$((p+1))|$((p+2))|$((p+3)))$"; then echo "$p"; return; fi
        tries=$((tries+1))
    done
    echo 0
}
export FORGE_MASTER_PORT="${FORGE_MASTER_PORT:-$(pick_port)}"
echo "[LOCAL $EXPNAME] FORGE_MASTER_PORT=$FORGE_MASTER_PORT"

# Cross-socket NCCL groups (GPUs 0-3 = NUMA node 0, 4-7 = node 1) crash with
# "illegal memory access" in init_inter_engine_group unless PCIe P2P is disabled
# (SETUP.md P10). Apply automatically when the GPU set spans both sockets.
# 2026-09-10: the same crash also hit the same-socket pair (1,2) intermittently (2 of 3
# starts), so P2P is disabled for every run; the update phase is ~10% of a step.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
echo "[LOCAL $EXPNAME] NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
export RAY_TMPDIR="${RAY_TMPDIR:-/data/liyan/ray_tmp}"; mkdir -p "$RAY_TMPDIR"
CHILD=""
forward() { echo "[LOCAL $EXPNAME] driver got signal; forwarding TERM to trainer ${CHILD:-?}"; [[ -n "$CHILD" ]] && kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null; exit 130; }
trap forward TERM INT HUP

echo "[LOCAL $EXPNAME] start $(date): model=$MODEL gpus=$GPUS engines=$ENGINES sigma=$SIGMA lr=$LR G=$G B=$B iters=$ITERS"

attempt=0
while true; do
    if is_done; then echo "[LOCAL $EXPNAME] DONE at iter $(last_iter) $(date)"; break; fi
    if (( attempt >= RETRY_MAX )); then echo "[LOCAL $EXPNAME] giving up after $attempt attempts"; exit 1; fi
    attempt=$((attempt + 1))
    echo "[LOCAL $EXPNAME] attempt $attempt: resume-from iter $(last_iter) $(date)"

    (
        source "$VENV/bin/activate"
        cd "$REPO"
        export HF_HOME=/data/liyan/hf-cache
        export TOKENIZERS_PARALLELISM=false
        export VLLM_ENABLE_V1_MULTIPROCESSING=0
        export PYTHONUNBUFFERED=1
        export GRZO_GPU_MEM_UTIL=$GPU_MEM_UTIL
        export ANCHOR_RATCHET=$ANCHOR_RATCHET
        export RATCHET_DROP=$RATCHET_DROP RATCHET_PATIENCE=$RATCHET_PATIENCE
        export RATCHET_WARMUP=$RATCHET_WARMUP
        export REPLAY_FRAC=$REPLAY_FRAC REPLAY_ADV=$REPLAY_ADV
        export REPLAY_CAP_PER_PROMPT=$REPLAY_CAP_PER_PROMPT REPLAY_MAX=$REPLAY_MAX
        exec python es_at_scale/train_grzo_surrogate.py \
            --model-name "$MODEL" --sigma "$SIGMA" --lr "$LR" \
            --task "$TASK" --train-dataset "$TRAIN_DS" --eval-dataset "$EVAL_DS" \
            ${EVAL_SUBSETS:+--eval-subsets $EVAL_SUBSETS} \
            --delta-norm "$DELTA_NORM" --min-directions "$MIN_DIRS" \
            --dapo-target-groups "$DAPO_TARGET" --dapo-draw "$DAPO_DRAW" \
            --pairs-per-direction "$PAIRS_PER_DIR" --directions-per-step "$DIRS_PER_STEP" \
            --countdown-reward "$CD_REWARD" --math-reward "$MATH_REWARD" \
            --margin-tau "$MARGIN_TAU" --margin-tau-end "$MARGIN_TAU_END" \
            --lr-schedule "$LR_SCHEDULE" \
            --rollout-temperature "$TEMP" --group-size "$G" \
            --batch-size "$B" --mini-batch-size "$B" \
            --n-iterations "$ITERS" --eval-freq "$EVAL_FREQ" --max-tokens "$MAXTOK" \
            --n-vllm-engines "$ENGINES" --n-gpu-per-vllm-engine "$GPUS_PER_ENGINE" \
            --use-gpus "$GPUS" --seed "$SEED" \
            --logging "$LOGGING" --wandb-project grzo-rlvr --save-best-models \
            --experiment-name "$EXPNAME" --output-directory "$OUT" --resume
    ) >> "$LOGDIR/$EXPNAME.log" 2>&1 &
    CHILD=$!
    echo "$CHILD" > "$OUT/$EXPNAME/train.pid"
    wait "$CHILD"; rc=$?
    echo "[LOCAL $EXPNAME] trainer exited rc=$rc at iter $(last_iter) $(date)"
    is_done && continue
    (( rc == 0 )) && { echo "[LOCAL $EXPNAME] rc=0 but not done -- check log"; }
    sleep "$RETRY_SLEEP"
done
