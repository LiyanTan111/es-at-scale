#!/bin/bash
# Interactive-relay driver for one GRZO run. Loops: allocate a fresh 4h
# interactive node -> resume-train from latest/ until the wall hits or the run
# finishes -> re-allocate. Survives the 4h interactive wall via checkpoint resume.
#
# Launch detached so it outlives the Claude/ssh session:
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct EXPNAME=grzo-0p5b-sp-s1e-3 \
#     nohup bash scripts/relay_grzo.sh > logs/relay/grzo-0p5b-sp-s1e-3.driver.log 2>&1 & disown
set -uo pipefail

MODEL="${MODEL:?set MODEL}"
EXPNAME="${EXPNAME:?set EXPNAME (fixed, no timestamp -- resume keys on it)}"
MODE="${MODE:-single_point}"
REWARD_MODE="${REWARD_MODE:-shaped}"
SIGMA="${SIGMA:-1e-3}"
G="${G:-8}"; B="${B:-8}"
ITERS="${ITERS:-400}"; EVAL_FREQ="${EVAL_FREQ:-25}"
MAXTOK="${MAXTOK:-1024}"; SEED="${SEED:-42}"
WALL="${WALL:-04:00:00}"
OUT="${OUT:-$SCRATCH/runs/grzo}"

REPO=/global/cfs/cdirs/m4788/liyantan/es-at-scale
VENV=/pscratch/sd/l/liyantan/es-at-scale/.venv
LOGDIR="$REPO/logs/relay"; mkdir -p "$LOGDIR"
PROG="$OUT/$EXPNAME/latest/progress.json"

last_iter() {
    # progress.json is {"iteration": N}. Parse with grep so the login-node driver
    # needs no python/venv.
    [[ -f "$PROG" ]] || { echo -1; return; }
    local n
    n=$(grep -oE '"iteration"[: ]+[0-9]+' "$PROG" 2>/dev/null | grep -oE '[0-9]+' | head -1)
    echo "${n:--1}"
}
is_done() { [[ "$(last_iter)" -ge $((ITERS - 1)) ]]; }

echo "[RELAY $EXPNAME] start: model=$MODEL mode=$MODE sigma=$SIGMA G=$G B=$B iters=$ITERS wall=$WALL"

while true; do
    if is_done; then echo "[RELAY $EXPNAME] DONE at iter $(last_iter)"; break; fi

    echo "[RELAY $EXPNAME] resume-from iter $(last_iter); allocating $WALL node..."
    out=$(salloc --no-shell -A m4788_g -C gpu -q interactive -t "$WALL" -N 1 \
                 --gpus-per-node=4 -c 128 -J "relay-$EXPNAME" 2>&1)
    JID=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | awk '{print $NF}')
    if [[ -z "$JID" ]]; then echo "[RELAY $EXPNAME] salloc failed: $out"; sleep 120; continue; fi

    # Poll for RUNNING (runbook gotcha 6.4 -- Granted != ready).
    ws=$(date +%s)
    ok=1
    while true; do
        st=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
        [[ "$st" == "RUNNING" ]] && break
        if (( $(date +%s) - ws > 900 )); then
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
            python es_at_scale/train_grzo.py \
                --model-name $MODEL --grzo-mode $MODE --reward-mode $REWARD_MODE --group-size $G \
                --batch-size $B --mini-batch-size $B --sigma $SIGMA \
                --n-iterations $ITERS --eval-freq $EVAL_FREQ --max-tokens $MAXTOK \
                --n-vllm-engines 4 --use-gpus 0,1,2,3 --seed $SEED \
                --logging wandb --wandb-project grzo-rlvr --save-best-models \
                --experiment-name $EXPNAME --output-directory $OUT --resume
         " >> "$LOGDIR/$EXPNAME.log" 2>&1

    echo "[RELAY $EXPNAME] srun returned (finished or wall hit); releasing $JID"
    scancel "$JID" 2>/dev/null || true
    sleep 20
done
