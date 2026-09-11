# ES-at-Scale — Environment Setup Notes

Portable notes for standing up this repo's environment (Python 3.12 + vLLM), the
pitfalls hit, and how to reproduce on a fresh cluster. Written while setting up on
one box; the GPU smoke test is intended to run on the target cluster.

## TL;DR quickstart (fresh cluster)

Assumes Linux, NVIDIA GPU(s) with a recent driver (CUDA 12.x), and a **large-capacity
disk** — torch + vLLM wheels plus model/HF caches are tens of GB, so keep everything
**off a nearly-full home partition**. Uses [`uv`](https://docs.astral.sh/uv/) to fetch
Python 3.12 without touching the system Python.

```bash
# 0. Point all heavy caches at a big disk (NOT home).
export BIG=/path/to/big/disk                 # e.g. /data/<you>
export UV_CACHE_DIR=$BIG/.cache/uv
export UV_PYTHON_INSTALL_DIR=$BIG/.uv/python
export HF_HOME=$BIG/hf-cache

# 1. Python 3.12 venv (README requires 3.12; system default is often 3.10).
uv venv --python 3.12 $BIG/es-at-scale/.venv
source $BIG/es-at-scale/.venv/bin/activate

# 2. Core install (pulls torch 2.8 + vLLM 0.11 + transformers 4.57.6).
uv pip install -e .

# 3. Extras: math grading + logging.
uv pip install -e ".[math,logging]"
#   (equivalently: uv pip install math-verify pylatexenc latex2sympy2_extended wandb)

# 4. Sanity check (no GPU needed).
python -c "import torch, vllm, transformers, ray, datasets; \
print('torch', torch.__version__, '| vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
```

If step 2 fails on `pip`/build backends, see pitfall **P1** below.

## Verified versions (Python 3.12.13)

Installed and import-verified on 2026-07-01:

| package | version |  | package | version |
|---|---|---|---|---|
| torch | 2.8.0 |  | datasets | 5.0.0 |
| vllm | 0.11.0 |  | numpy | 2.2.6 |
| transformers | 4.57.6 |  | accelerate | 1.14.0 |
| ray | 2.56.0 |  | math-verify | 0.9.0 |
| wandb | 0.28.0 |  | latex2sympy2-extended | 1.11.0 |

CPU-only end-to-end check of the reward paths passed:
`boxed_reward_fn` (correct→1.0, wrong→0.0) and `countdown_reward_fn` (→1.1). The vLLM
**GPU** smoke test (model load + a generation) was not run on the original box — no free
card there — and should be run first on the target cluster.

## Pitfalls

### P1 — `futures` dependency breaks install on Python 3.12  ✅ fixed in this repo
`setup.py` listed `futures` unconditionally. `futures` is a **Python-2-only** backport
of the stdlib `concurrent.futures`; on Py3 its build fails with
`This backport is meant only for Python 2.`, and because one dep fails the **entire**
`pip install` rolls back (torch/vLLM never land).
**Fix (already applied):** guarded to Python 2 → `"futures; python_version < '3'"`.

### P2 — Disk / caches must live off home
torch + vLLM wheels and model downloads are large. Keep the venv, `UV_CACHE_DIR`,
`UV_PYTHON_INSTALL_DIR`, and `HF_HOME` on a big disk. On the original box, home had only
~42 GB free (98% full) — installing there would have failed mid-way.

### P3 — Python version
Only Python 3.10 was available system-wide; the README requires 3.12.
`uv venv --python 3.12` provisions a standalone 3.12 build; no root / no system changes.

### P4 — Pre-existing `SyntaxWarning`s (non-blocking)
`es_at_scale/reward_function/math_grader.py` uses non-raw regex strings (e.g. `"\d"`,
`"\^"`), which emit `SyntaxWarning: invalid escape sequence` on Python 3.12. Harmless
today (warnings only), but they'll become errors in a future Python. Left as-is; convert
those literals to raw strings (`r"..."`) if you want them gone.

## GPU / running notes

- **Hard constraint: at most 2 GPUs on the working cluster.** README examples default to
  8 GPUs (`--use-gpus "0,..,7" --n-vllm-engines 8`) — always scale that down.
- Layouts (see README "Notes"):
  - small model, 2 members in parallel: `--use-gpus "0,1" --n-vllm-engines 2 --n-gpu-per-vllm-engine 1`
  - big model (tensor-parallel): `--use-gpus "0,1" --n-vllm-engines 1 --n-gpu-per-vllm-engine 2`
  - eval-only: `--n-iterations 0` (runs on `engines[0]`); add `--logging none` to skip wandb.
- `n_vllm_engines` affects throughput only, not the ES result: each iteration runs
  `ceil(population_size / n_vllm_engines)` sequential rounds.
- Datasets for the countdown + math eval suites are already committed under `datasets/`.

## First thing to run on the new cluster (GPU smoke test)

Pick two GPUs that are actually free (`nvidia-smi`), then eval-only on a small model:

```bash
python es_at_scale/train.py \
  --task countdown \
  --model-name "Qwen/Qwen2.5-1.5B-Instruct" \
  --eval-dataset "datasets/evaluation_suite/countdown" \
  --max-tokens 512 --n-iterations 0 \
  --n-vllm-engines 1 --use-gpus "<free_idx>" \
  --output-directory "./experiments/" --experiment-name "smoke-eval" \
  --logging none
```

Then a short training run (few iterations, ≤2 GPUs) to confirm the ES update loop.

## Status checklist

- [x] Python 3.12 venv on a big disk
- [x] Core install (torch / vLLM / transformers / ray / datasets)
- [x] Import verification on Python 3.12
- [x] Extras (math-verify etc.) installed + import-verified
- [x] Reward/grader paths validated end-to-end (CPU)
- [ ] GPU smoke test (eval-only) — run on target cluster
- [ ] Short training run on ≤2 GPUs — run on target cluster

---

## lambda-scalar (this box) — 2026-09-10 onward

Same physical machine as the original setup box (hostname `lambda-scalar`, 8× H100 NVL,
**no scheduler** — plain processes). The NERSC scripts (`scripts/relay_*.sh`,
`submit_regular_*.sh`, `train_grzo.sbatch`) hardcode Slurm + `/pscratch` paths and
**must not be used here**; use `scripts/local_forge.sh` instead (same env-var knobs).

**GPU rules (hard):**
- **GPU 0 is permanently banned** (hardware temperature fault). Never put it in `--use-gpus`.
- **At most 4 cards.** Default `GPUS=1,2,3,4`.
- Shared box (~30 users). `local_forge.sh` refuses to start if a requested card already has
  >2 GB in use. Check `nvidia-smi` first anyway.

**Launch / stop:**
```bash
source env.sh                                  # venv + HF_HOME on /data
MODEL=Qwen/Qwen2.5-1.5B-Instruct EXPNAME=forge2-cd-v2-local ITERS=4000 EVAL_FREQ=25 \
  MIN_DIRS=64 DAPO_TARGET=8 DAPO_DRAW=32 REPLAY_FRAC=0.5 \
  ANCHOR_RATCHET=1 RATCHET_DROP=0.03 RATCHET_PATIENCE=3 RATCHET_WARMUP=400 MAXTOK=512 \
  setsid bash scripts/local_forge.sh > logs/local/forge2-cd-v2-local.driver.log 2>&1 &
# training log: logs/local/<EXPNAME>.log ; runs/ckpts: /data/liyan/runs/grzo/<EXPNAME>/
kill -TERM $(cat /data/liyan/runs/grzo/<EXPNAME>/train.pid)     # graceful stop
```

### P5 — `setsid nohup cmd &` gives the wrong PID
`$!` after `setsid ... &` is the setsid/shell wrapper, not the python process (it exits at
once, so a watcher on it concludes the run "died silently" while it is actually loading).
`local_forge.sh` writes the real trainer PID to `<run>/train.pid` and the driver PID to
`<run>/driver.pid` — use those.

### P6 — Never `pkill -f train_grzo_surrogate`
The trainer's reward-grading `multiprocessing.Pool(8)` workers share the driver's command
line, so a name-based pkill also kills them (and any other run's driver). Worse, a
`pgrep -f`/`pkill -f` pattern typed into an interactive command also matches *that command's
own shell* (its cmdline contains the pattern) — use the `[t]rain_grzo` bracket trick if you
must search by name. Kill only the PID from `train.pid`; the trainer's SIGTERM handler tears
down its Ray actors.

### P7 — Ray dashboard `opentelemetry` messages are benign
`dashboard.log` prints INFO lines like "Module ... cannot be loaded ... No module named
'opentelemetry'". Harmless (dashboard disabled); ignore.

### P9 — Two runs starting at the same time crash in `init_inter_engine_group`
Symptom: `CUDA error: an illegal memory access was encountered` inside
`init_inter_engine_group` on both runs, ~1 min after launch. Cause: both trainers call
vLLM's `get_open_port()` within seconds, receive the same free port, and their NCCL
inter-engine rendezvous cross. Fix: `es_trainer.py` honours `FORGE_MASTER_PORT`;
`scripts/local_forge.sh` assigns a unique unused port per run. If launching by hand, set
`FORGE_MASTER_PORT` yourself or stagger launches by ≥2 minutes.

### P10 — Cross-socket 2-GPU NCCL group crashes at init
GPUs 0–3 are on NUMA node 0, GPUs 4–7 on node 1 (`nvidia-smi topo -m`). A 2-engine run on
a pair that spans sockets (e.g. 3,4) dies in `init_inter_engine_group` with
`CUDA error: an illegal memory access` — solo, with unique ports, every time. Same-socket
pairs (1,2) mostly work but the same crash hit (1,2) on 2 of 3 starts, so it is not purely a
cross-socket issue — PCIe P2P between non-NVLink GPUs looks flaky on this box (NVLink pairs:
(0,1) (2,3) (4,6) (5,7)). The 4-engine group (1,2,3,4) worked 3/3. Fix: `NCCL_P2P_DISABLE=1`
(NCCL falls back to shared-memory transport) — verified on (3,4) for FORGE and ES; both runners
now set it for **every** run (update phase is ~10% of a step, so the cost is small).
Lane policy: Lane A = (1,2), Lane B = (3,4).

### P11 — Root partition at 95%: move Ray's temp dir off /tmp
The raylet logs `... is over 95% full ... Object creation will fail if spilling is required`
because `/tmp` is on `/` (95% full). `es_trainer.py` honours `RAY_TMPDIR` (passed to
`ray.init(_temp_dir=...)`); both runners export `RAY_TMPDIR=/data/liyan/ray_tmp`.

### P12 — Never edit a runner script while a driver is executing it
bash reads a script file incrementally; patching `scripts/local_*.sh` while a driver was
running produced `syntax error near unexpected token 'done'` when that driver reached the end
of its loop (the ES run itself was unaffected; only the driver's DONE bookkeeping was lost).
Both runners now copy themselves to `<run>/driver.sh` and re-exec from the snapshot.

### Status on this box
- [x] venv reused (Python 3.12.13, all deps incl. scipy/wandb import; new trainer modules import)
- [x] HF hub reachable; `HF_HOME=/data/liyan/hf-cache`; wandb creds in `~/.netrc`
- [x] fork `main` pulled; `setup/py312-env` (futures fix) merged back into `main` and pushed
- [x] `scripts/local_forge.sh` written; GPU-0 / busy-card guards tested
- [x] FORGE smoke via `scripts/local_forge.sh` (Qwen2.5-0.5B, 4 engines on GPUs 1–4,
      3 iters, eval every 2): rc=0, `DONE at iter 3`, 135 s wall (≈90 s engine start,
      ≈1.3 s/iter, ≈9 s per 2000-sample eval), GPUs fully released, no stray processes.
      Base 0.5B answer_acc 0.0000 matches the NERSC validate number.

### P8 — `[INFO] Received signal 15, cleaning up...` at shutdown is benign
Printed (several times) *after* `-- Training completed! --`. It comes from the reward
`multiprocessing.Pool` workers: they inherit the driver's SIGTERM handler and the Pool
sends them SIGTERM on exit. Not a crash. A driver that sits in `do_wait` for minutes after
completion, however, means a Pool worker was killed externally (see P6) — kill -9 it.
**Update 2026-09-11:** the hang also happened on a clean 4000-iter run: the Pool workers ran
`cleanup()` inside the inherited SIGTERM handler, hit `AttributeError: no attribute 'engines'`
and never exited. Fixed in `es_trainer._handle_exit` (workers `os._exit(0)`; cleanup wrapped).
Runs started before the fix may still need `kill -TERM <train.pid>` after completion.
