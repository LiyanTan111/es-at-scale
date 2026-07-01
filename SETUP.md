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
