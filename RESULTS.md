# RESULTS ledger (paper store) — one row per finished run; append the day it finishes

Hardware for all rows unless noted: lambda-scalar, H100 NVL 95 GB, GPUs 1–4.
Budget columns: `gens` = cumulative rollouts generated (train), `iters`, `wall` = wall-clock h.
`best` / `final` = Countdown answer_acc (0/1) on countdown_eval (2000, greedy, 512 tok) unless task says otherwise.

| run (EXPNAME) | task | model | recipe | seed | engines | iters | gens | wall (h) | best (iter) | final | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forge2-cd-1p5b-h100-s42 | countdown | Qwen2.5-1.5B-Instruct | FORGE v2 | 42 | 2 (1,2) | 4000 | — | — | — | — | **F1, in flight since 2026-09-10 12:11 PT** (wandb ngqa6knz) |
| es512-cd-1p5b-h100-s42 | countdown | Qwen2.5-1.5B-Instruct | ES paper | 42 | 2 (3,4) | 500 | 3.0M | ~6.2 | 22.8% (470) | **22.6%** | ⚠️ far below NERSC 37.9% (same vLLM 0.11, FA, config; only engines 2 vs 4 + H100). Eval noise ruled out (fixed-θ repeat std 0.0000; population std 0.016). UNVERIFIED as a reference until a 4-engine H100 rerun. |

## Reference rows imported from NERSC (A100-40GB, 4 engines) — see PROJECT_STATUS.md §2

| run | task | model | recipe | seed | iters | best | final | notes |
|---|---|---|---|---|---|---|---|---|
| es512-qwen1p5b | countdown | Qwen2.5-1.5B-Instruct | ES paper | 42 | 500 | — | 37.9% | 102% of paper |
| es512 s43 / s44 | countdown | Qwen2.5-1.5B-Instruct | ES paper | 43/44 | 500 | — | 34.5% / 33.6% | ES 3-seed 35.3 ± 2.2 |
| forge2-cd-assault-v2 | countdown | Qwen2.5-1.5B-Instruct | FORGE v2 | 42 | 1574/4000 (unfinished) | 10.45% (1475) | — | replay+ratchet, no collapse |
| forge2-cd-assault | countdown | Qwen2.5-1.5B-Instruct | FORGE v1 cosine | 42 | 3374/4000 | — | ~6.8% | frozen |
| forge512-1p5b-zs | countdown | Qwen2.5-1.5B-Instruct | FORGE v0 binary | 42 | 4000 | 15.75% (2475) | 7.0% | collapse −55% |

## Speed probes (throwaway runs, timing only)

| probe | model | engines | s/iter (steady) | notes |
|---|---|---|---|---|
| probe-1p5b-v2-2eng | 1.5B v2 | 2 (GPUs 1,2) | 15–19 (mean 17.9) | rollout ~12 s, score ~5.5 s, update ~1.5 s; eval ≈ 50 s |
| probe-1p5b-v2-speed | 1.5B v2 | 4 | 11.5–12.5 (mean 13.0 incl. eval step) | H100, GPUs 1–4; eval of 2000 prompts ≈ 28 s; base acc 1.60%/1.75% (NERSC 1.80%/1.40%) ✓; A100 ref 17.7 |
