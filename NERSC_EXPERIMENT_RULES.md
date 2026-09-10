# NERSC Perlmutter — Practical Runbook

> Handoff doc for a Claude window running experiments on NERSC. Everything below
> is empirically verified. If something contradicts what you observe live, trust
> the observation.

---

## 1. Environment quick facts

| Item | Value |
|---|---|
| Cluster | NERSC Perlmutter |
| Account | `m4788_g` |
| Login node | `ssh perlmutter.nersc.gov` (or use SSHFS/VSCode-SSH) |
| Constraint | `-C gpu` (each GPU node = 4× A100 40GB) |
| Scratch (fast) | `/pscratch/sd/<initial>/<user>/` — use for models, data, results |
| CFS (project, backed up) | `/global/cfs/cdirs/m4788/<user>/` — use for code + archived checkpoints |
| Home (slow, backed up) | `/global/homes/<initial>/<user>/` — use for configs, dotfiles |
| Python module | `module load python/3.12-26.1.0` |
| Recommended venv layout | `python -m venv /pscratch/.../.venv` (keeps deps off $HOME) |

**Storage workflow — code on CFS, run on SCRATCH, archive back to CFS**:
Keep code on CFS (persistent, project-shared, backed up). Write training output
to `$SCRATCH` because it's the fast parallel filesystem — do NOT train directly
into CFS. When a run finishes, copy only the important checkpoints back to CFS,
since `$SCRATCH` is purged after 90 days of inactivity (see 6.13).

```bash
# Code lives on CFS
cd /global/cfs/cdirs/m4788/<user>/<project>

# Train output to SCRATCH (fast parallel FS)
python train.py --output_dir $SCRATCH/runs/exp-001

# Done — archive the important checkpoint back to CFS (survives SCRATCH purge)
cp -r $SCRATCH/runs/exp-001/best_model /global/cfs/cdirs/m4788/<user>/models/exp-001-best
```

**Session setup boilerplate**:
```bash
module load python/3.12-26.1.0
source /pscratch/sd/<initial>/<user>/<project>/.venv/bin/activate
# Any project-specific env exports here (HF_HOME, W&B, etc.)
```

---

## 2. QOS layout & resource strategy

| QOS | Max concurrent | Wall | Best for | Node request |
|---|---|---|---|---|
| **premium** (`-q premium`) | **5** | 24h | Long single-node runs, one-shot evals | `-N 1 --gpus-per-node=4` |
| **interactive** (`-q interactive`) | **2** | 4h | Multi-node parallel dev/debug/relay | `-N {1..8} --gpus-per-node=4` |
| **regular** (`-q regular`) | ~unlimited | 24h | Overflow when other queues full; long PD wait | `-N 1 --gpus-per-node=4` |
| **debug** (`-q debug`) | 1 | 30 min | Very short smoke tests | `-N 1 --gpus-per-node=4` |
| **shared** (`-q shared`) | many | 24h | Single-GPU jobs (auto-packs multiple users on a node) | `--gpus-per-node=1` + `-c 32` |

**Standing allocation strategy for parallel sweeps**:
- Use **all 5 premium slots** for independent long single-node runs
- Use **both interactive 4-node allocs** for parallel N-task-at-once relays with auto-resume
- Reserve regular for overflow (12+h PD wait is common)

Total peak: `5 premium × 1 node + 2 interactive × 4 nodes = 13 nodes = 52 A100 GPUs`.

**Charging**: interactive is billed at same rate as regular; premium is 2× more expensive. Check your allocation with `sqs` or NERSC IRIS portal.

---

## 3. Smoke test protocol (mandatory before any big sweep)

**Rule**: never `sbatch` a 20+h run without first confirming loss decreases / no crash on a short interactive run. Debug on interactive; queue on sbatch.

```bash
# 1. Grab 1-node interactive for smoke
salloc --no-shell -A <account> -C gpu -q interactive -t 1:00:00 -N 1 \
    --gpus-per-node=4 -c 128 -J my-smoke

# 2. Get the JID + node
JID=$(squeue --me -q gpu_interactive -h -o "%i" | head -1)
NODE=$(squeue -j $JID -h -o "%N" | tr -d ' ')

# 3. Run smoke via srun
srun --jobid=$JID --nodes=1 --nodelist=$NODE --gpus=4 --ntasks=1 \
     --cpus-per-task=128 --exclusive \
     bash -c "
        module load python/3.12-26.1.0 && source .venv/bin/activate
        # short training / test command
        torchrun --standalone --nnodes=1 --nproc-per-node=4 --master-port=31201 \
            train.py --max_steps 50 ...
     " 2>&1 | tee logs/smoke.log

# 4. Verify pass criteria (loss present, no Traceback, etc.)
grep -aoE "'loss': [0-9.]+" logs/smoke.log | head -20
grep -aE "Traceback|Error|OOM|RuntimeError" logs/smoke.log
```

**Pass criteria** (customize to your task):
- N loss samples logged in expected range
- No `Traceback` / `OOM` / `RuntimeError` in log
- Any model-specific sanity check (e.g. tokenizer loaded, correct # trainable params)

**Common smoke bugs caught here**: OOM (batch too big), missing HF_TOKEN, wrong path, config mismatch. Fix on interactive (fast iteration) before wasting a 24h premium sbatch.

---

## 4. Sbatch pattern (single-node, long runs)

**Template**: keep a single `.sbatch` template driven by env vars. Sample skeleton:

```bash
#!/bin/bash
#SBATCH -A <account>
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 128
#SBATCH -t 24:00:00
#SBATCH -J my-job
#SBATCH -o logs/ddp_%j.out
#SBATCH -e logs/ddp_%j.err

set -euo pipefail
module load python/3.12-26.1.0
source /pscratch/sd/<initial>/<user>/<project>/.venv/bin/activate
# any HF_HOME / W&B / TOKENIZERS_PARALLELISM exports

# Env-var-driven params — override at submit time
TASK="${TASK:-default_task}"
LR="${LR:-1e-4}"
STEPS="${STEPS:-20000}"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nnodes=1 --nproc-per-node=4 \
    train.py \
    --task $TASK --learning_rate $LR --max_steps $STEPS ...
```

**Submit with env overrides**:
```bash
sbatch --qos=premium -t 24:00:00 \
  --export=ALL,TASK=RTE,LR=1e-4,STEPS=20000,RUN_NAME=my-tag \
  scripts/my_template.sbatch
```

**Auto-resume from checkpoint**: If your training framework supports `resume_from_checkpoint` (HuggingFace Trainer does by default when `--output_dir` contains `checkpoint-N/`), just re-submit the same job — it picks up automatically.

---

## 5. Interactive relay pattern (N-task-parallel × 4h × auto-resume)

**Use when**: you have M tasks to run and don't want to wait M × 20h sequential. Interactive relay parallelizes N tasks per 4-node alloc, and a background watcher re-allocates when the 4h wall time expires.

### 5.1 Manual bootstrap (first iter)

```bash
# 1. Allocate N-node × 4h
out=$(salloc --no-shell -A <account> -C gpu -q interactive -t 4:00:00 -N 4 \
    --gpus-per-node=4 -c 128 -J relay-boot 2>&1)
JID=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | awk '{print $NF}')

# 2. WAIT FOR RUNNING (not just Granted — see gotcha 6.4)
until [[ "$(squeue -j $JID -h -o '%T' 2>/dev/null)" == "RUNNING" ]]; do sleep 5; done
NODES=($(scontrol show hostnames $(squeue -j $JID -h -o '%N' 2>/dev/null | tr -d ' ')))

# 3. Launch M srun tasks (one per node, each 4 GPUs DDP)
for i in {0..3}; do
    task=${TASKS[$i]}; tag=${TAGS[$i]}
    srun --jobid=$JID --nodes=1 --nodelist=${NODES[$i]} --gpus=4 --ntasks=1 \
         --cpus-per-task=128 --exclusive \
         bash -c "
            module load python/3.12-26.1.0 && source .venv/bin/activate
            torchrun --standalone --nnodes=1 --nproc-per-node=4 --master-port=$((31500+i)) \
                train.py --task $task ...
         " >> logs/long_run/${tag}.log 2>&1 &
done
```

### 5.2 Auto-resumer (relaunch relay after alloc ends)

```bash
# Watcher: sleep until alloc ends, then start relay runner script
nohup bash -c "
    until ! squeue -j $JID -h 2>/dev/null | grep -q .; do sleep 60; done
    nohup bash scripts/relay_runner.sh >> logs/relay.log 2>&1 &
    disown \$!
" > /tmp/resumer.log 2>&1 &
disown $!
```

### 5.3 Relay runner script skeleton

```bash
#!/bin/bash
set -u
TASKS=(
    "task_1_args..."
    "task_2_args..."
    ...
)

max_ckpt() {
    # Return highest checkpoint-N number in a given output_dir
    local d="$1" max=0
    if [[ -d "$d" ]]; then
        for c in "$d"/checkpoint-*; do
            [[ -d "$c" ]] || continue
            local n="${c##*/checkpoint-}"
            [[ "$n" =~ ^[0-9]+$ ]] || continue
            (( n > max )) && max=$n
        done
    fi
    echo "$max"
}

MAX_STEPS=20000
while true; do
    # Check pending TASKS (those with max_ckpt < MAX_STEPS)
    pending=()
    for entry in "${TASKS[@]}"; do
        # parse tag from entry, check ckpt
        last=$(max_ckpt "path/to/results/$tag")
        (( last >= MAX_STEPS )) || pending+=("$entry")
    done
    [[ ${#pending[@]} -eq 0 ]] && break

    # New salloc
    out=$(salloc --no-shell -A <account> -C gpu -q interactive -t 4:00:00 \
        -N ${#pending[@]:0:4} --gpus-per-node=4 -c 128 -J relay 2>&1)
    JID=$(echo "$out" | grep -oE 'Granted job allocation [0-9]+' | awk '{print $NF}')
    [[ -z "$JID" ]] && { echo "salloc failed; retry"; sleep 120; continue; }

    # POLL for RUNNING state (see gotcha 6.4)
    wait_start=$(date +%s)
    while true; do
        state=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
        [[ "$state" == "RUNNING" ]] && break
        (( $(date +%s) - wait_start > 300 )) && { scancel "$JID"; continue 2; }
        sleep 10
    done
    NODES=($(scontrol show hostnames $(squeue -j $JID -h -o '%N' | tr -d ' ')))

    # Launch srun per node
    pids=()
    for i in "${!pending[@]}"; do
        (( i >= ${#NODES[@]} )) && break
        srun --jobid=$JID --nodes=1 --nodelist=${NODES[$i]} ... &
        pids+=($!)
    done

    # Wait for all pids or alloc-end
    while true; do
        any_alive=0
        for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && any_alive=1 && break; done
        (( any_alive == 0 )) && break
        state=$(squeue -j "$JID" -h -o '%T' 2>/dev/null)
        [[ -z "$state" || "$state" != "RUNNING" ]] && break
        sleep 60
    done
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    scancel "$JID"; sleep 20
done
```

---

## 6. Gotchas (learned the hard way)

### 6.1 `grep -a` is mandatory for training logs
Progress bars (tqdm, HF Trainer) emit escape sequences that make grep treat log files as binary → silent skip. **Always use `grep -a`** when parsing any training log.
```bash
grep -aoE "'loss': [0-9.]+" logs/train.log    # works
grep    -oE "'loss': [0-9.]+" logs/train.log  # returns nothing
```

### 6.2 stdout vs stderr split
`torchrun`/HF Trainer send different messages to different streams:
- `.out` typically has `'loss':`, `'eval_loss':` (Trainer's Python `print`)
- `.err` typically has tqdm progress bars + Python `logger` messages

When looking for loss, check both; when looking for step, `.err` usually has tqdm markers.

### 6.3 Wall time timeouts kill mid-eval
If an sbatch expires while doing end-of-training generation eval, partial results are in the log but no `eval_result.json`. Give generous `-t` (e.g. `24:00:00` for a `1:30:00` training + `2:00:00` eval).

### 6.4 `salloc` polling for RUNNING (NOT Granted)
`salloc` returns "Granted job allocation N" **before** nodes are actually ready. If you immediately query `scontrol show job N`, `NodeList=(null)` because state is still PENDING/CONFIGURING. **Poll for RUNNING state**:

```bash
# WRONG
salloc ...; sleep 5; nodelist=$(scontrol show job $JID | grep NodeList=)  # empty → break

# RIGHT
salloc ...
until [[ "$(squeue -j $JID -h -o '%T' 2>/dev/null)" == "RUNNING" ]]; do sleep 10; done
nodelist=$(squeue -j $JID -h -o '%N' | tr -d ' ')
```

### 6.5 `COMPLETING` state can hang for 30+ min after alloc ends
When an interactive alloc hits wall time, Slurm can hang in `COMPLETING` from 30 sec to 30+ min. squeue still shows the job. **Do not force-kill with `scancel --signal=KILL`** — you get "Invalid job id" and it clears nothing.

Just wait, or manually submit a new alloc if in a hurry. If your auto-resumer waits for `squeue` to return empty, it may be blocked. Sometimes worth killing the watcher and manually re-allocating.

### 6.6 Bash `TASKS` variable is read once at script start
If you edit a runner script's `TASKS` array while it's running, **the edit does NOT propagate** — bash reads TASKS to memory when script starts. To change TASKS mid-relay: kill the runner, edit, restart.

### 6.7 Premium `QOSMaxSubmitJobPerUserLimit = 5`
Sixth `sbatch --qos=premium` fails with:
```
QOSMaxSubmitJobPerUserLimit: Batch job submission failed
```
Overflow to `regular` (long PD wait) or wait for a slot.

### 6.8 Wandb creates a new run per training-script init
Each `torchrun train.py` creates a fresh wandb run. In a K-iter relay, each task gets K wandb runs (fragmented curves). To combine, extract loss/eval curves from local logs (append mode is your friend), then upload as a single new run via `wandb.log()`.

### 6.9 `--overwrite_output_dir` doesn't delete checkpoints
Despite the name, HF Trainer's `--overwrite_output_dir` only clears the output directory of non-checkpoint state. It **explicitly resumes from `last_checkpoint`** if any exists. To force fresh start: `rm -rf $OUT_DIR/checkpoint-*` before submit.

### 6.10 `--save_total_limit` deletes older checkpoints
`--save_total_limit N` keeps only N most recent ckpts. Old ones (`ckpt-4000`, `ckpt-8000`) get deleted when newer ones are written. If you want to eval mid-training checkpoints, either raise `save_total_limit` or eval immediately.

### 6.11 NCCL init hang on multi-node
For multi-node DDP, single-node `torchrun --standalone` won't work. Use:
```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1)
export MASTER_PORT=29500
torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID \
    --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT ...
```
And launch with `srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 bash -c "torchrun ..."`.

### 6.12 A100 40GB memory budget
Common per-GPU allocations on Perlmutter's A100 40GB:
- 8B params fp16 + Adam32 states + activations → OOM (need FSDP or 80GB)
- 8B params fp16 + inference/ZO → ~24-30 GB (fits)
- 13B params fp16 + ZO → ~30-35 GB (tight but fits)
- 30B params fp16 → doesn't fit on single GPU (need tensor parallel)

### 6.12b `sbatch` python stdout is block-buffered → looks hung when it isn't
A plain `sbatch` runs `python` with **no tty**, so Python stdout is *block*-buffered
(flushes only every ~4-8 KB), not line-buffered. A healthy training run can then
show a **frozen `.out` file for many minutes** while it's actually progressing —
easy to misread as an NCCL/engine hang and kill prematurely. `srun`-launched runs
(interactive relays) usually don't show this because srun line-buffers.

**Fix**: `export PYTHONUNBUFFERED=1` (or `python -u`) in the sbatch so `.out`
flushes live. **Before declaring a run hung**, check for side-effect artifacts that
bypass stdout buffering (a written `eval_output/*.json`, a checkpoint dir, wandb
step count, or the `.out` file mtime) — if any advanced past engine init, it's a
buffering artifact, not a hang.

### 6.13 File system pitfalls
- `/pscratch/` is **purged** after 90 days of inactivity — archive important checkpoints to CFS (`/global/cfs/cdirs/m4788/`) when a run finishes (see storage workflow in §1)
- Home has small quota (~40 GB) — don't put model weights or datasets there
- CFS (`/global/cfs/cdirs/m4788/`) is the persistent, backed-up project FS — good for code + archived checkpoints, but slower than `$SCRATCH` for hot training I/O; don't train directly into it
- `/tmp` on compute nodes is ephemeral and small — use `$TMPDIR` if you need scratch space per-job

---

## 7. Common commands (cheat sheet)

```bash
# Queue overview
squeue --me -o "%.12i %.10q %.10T %.5D %.10L %.45j" | head -15

# By QOS
squeue --me -q gpu_premium -h | wc -l       # 0-5
squeue --me -q gpu_interactive -h | wc -l   # 0-2

# Job accounting (after job finishes)
sacct -j $JID -X -o JobID,State,Start,End,Elapsed -n

# Cancel jobs
scancel JID1 JID2 ...
scancel --me --qos=interactive     # cancel all my interactive jobs

# See running srun children of an interactive alloc
ps -ef | grep "srun.*--jobid=$JID" | grep -v grep

# Get node list of an alloc
scontrol show hostnames $(squeue -j $JID -h -o '%N')

# Check reason a PD job isn't starting
squeue -j $JID -o "%.12i %.10T %r"    # %r shows Reason

# Priority / estimated start
sqs                                    # queue state
squeue -j $JID -o "%S"                 # estimated start (rough)

# Node info
sinfo -N -l -p regular_milan_ss11      # CPU nodes
sinfo -N -l -C gpu                     # GPU nodes
sinfo --long                           # includes MAINT reservations

# Storage
du -sh /pscratch/sd/<initial>/<user>/*   # scratch usage
myquota                                  # home quota

# NERSC status
cat /etc/motd                          # system message
```

**Log parsing helpers**:
```bash
JID=12345678
# Latest step
grep -aoE "[0-9]+/[0-9]+" logs/ddp_$JID.err | tail -1
# All loss values
grep -aoE "'loss': [0-9.]+" logs/ddp_$JID.out
# Eval loss timeline
grep -aoE "'eval_loss': [0-9.]+" logs/ddp_$JID.out
# End-of-training metric
grep -aoE "INFO - \{'(accuracy|f1|...)':[^}]*\}" logs/ddp_$JID.err | tail -1
```

---

## 8. Debugging playbook

**Job stuck in PD forever**:
```bash
squeue -j $JID -o "%r"    # Reason column
sacct -j $JID -o Reason -n
```
- `QOSMaxSubmitJobPerUserLimit` → hit the 5-premium cap, submit to different QOS
- `Resources` → wait (busy cluster)
- `Priority` → someone else ahead — nothing to do but wait or bump priority
- `ReqNodeNotAvail, UnavailableNodes:...` → NERSC maintenance window coming

**Job dies immediately (RUNNING → COMPLETED in <1 min)**:
```bash
tail -20 logs/ddp_$JID.err   # look for Traceback
```
Common causes:
- OOM (reduce batch, model, or sequence length)
- Missing env var (HF_TOKEN, WANDB_API_KEY, etc.)
- Wrong module path (`ModuleNotFoundError` — venv not sourced)
- Config error (`tokenizer_config`, dtype mismatch)
- Permission denied (data on `$HOME` — move to `/pscratch/`)

**Model load takes >5 min**: Usually caused by wrong `device_map` for multi-GPU. Verify `LOCAL_RANK` is set (torchrun does this automatically). For very large models, first load may be slow due to sharded checkpoint reading — subsequent loads use HF cache.

**Loss is NaN**: LR too high, gradient explosion, or fp16 overflow. Halve LR, or use bf16 / mixed precision.

**Loss oscillates but doesn't decrease**: Might be expected for your algorithm (e.g., ZO methods have this pattern). Trust the eval metric, not the training loss trajectory.

**Auto-resume not picking up ckpt**: Check that `--output_dir` matches an existing dir with `checkpoint-N/` subdirs. HF Trainer resumes from `get_last_checkpoint(output_dir)`. If dir has stale files, `ls -la $OUT_DIR/checkpoint-N/*.safetensors` should show valid weights.

**Wandb runs fragmented (N iters = N runs per task)**:
1. Extract loss/eval_loss from log with `grep -aoE`
2. Dump to CSV
3. Upload with `wandb.init()` + `wandb.log()` as a new "combined" run

---

## 9. NERSC monthly maintenance

**Every ~4th Wed** (roughly), NERSC does a full-day maintenance window (usually 06:00-22:00 PDT). All jobs PD; running jobs killed if their wall time extends into the window. Check:
- https://www.nersc.gov/live-status/motd/ for announced windows
- `sinfo --long` for `MAINT` reservations
- `/etc/motd` on login node

Before maintenance:
- Save all in-flight ckpts (small `--save_steps` interval)
- Don't submit long sbatches close to the window
- Post-maintenance, previously-queued PD sbatches start automatically

---

## 10. Sanity checklist for a fresh setup

```bash
# 1. Environment
module list                            # python/3.12-26.1.0 loaded?
source /pscratch/.../.venv/bin/activate
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
# Expected: True, 4

# 2. Account access
sqs                                    # your allocation shown?

# 3. Slurm partitions
sinfo -h | head -5                     # gpu partition available?

# 4. Storage
myquota                                # home not near full?
df -h /pscratch/sd/<initial>/<user>/   # scratch has room?

# 5. Smoke sbatch (2 min)
sbatch --qos=debug -t 0:05:00 --wrap="hostname; nvidia-smi -L"
# check logs after ~1 min
```

If all 5 pass, you're ready to submit real jobs.

---

## 11. Useful docs

- NERSC Perlmutter user guide: https://docs.nersc.gov/systems/perlmutter/
- Slurm quick reference: https://docs.nersc.gov/jobs/
- QOS charging rates: https://docs.nersc.gov/jobs/policy/#qos-limits-and-charges
- MOTD / status page: https://www.nersc.gov/live-status/motd/
- IRIS portal (allocation usage): https://iris.nersc.gov/

---

*Adapt paths, account, and QOS names as needed. If experiments give different behavior than documented, live measurements always win.*
