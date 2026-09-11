# RESULTS ledger (paper store) — one row per finished run; append the day it finishes

Hardware for all rows unless noted: lambda-scalar, H100 NVL 95 GB, GPUs 1–4.
Budget columns: `gens` = cumulative rollouts generated (train), `iters`, `wall` = wall-clock h.
`best` / `final` = Countdown answer_acc (0/1) on countdown_eval (2000, greedy, 512 tok) unless task says otherwise.

| run (EXPNAME) | task | model | recipe | seed | engines | iters | gens | wall (h) | best (iter) | final | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| forge2-cd-1p5b-h100-s42 | countdown | Qwen2.5-1.5B-Instruct | FORGE v2 | 42 | 2 (1,2) | 4000 | — | — | — | — | 4000 | 2.3M | 19.7 | **9.25% (3700)** | 8.2% | F1 done 2026-09-11 07:40; ratchet @624, @1599; G2 FAIL (<15%) |
| forge-raw-N96-lr4e-5-s42 | countdown | 1.5B | FORGE raw δ/2σ, N=96, lr 4e-5, no ratchet | 42 | 1 (4) | 60/400 (stopped) | — | ~1 | 2.25% (25) | 1.2% | P1 smoke: raw estimator trains without error (~24–60 s/iter on 1 engine); stopped to give GPU 4 to E1.8 |
| forge-raw-N384-lr6.4e-4-s43 | countdown | 1.5B | FORGE raw, N=384, lr 6.4e-4, no ratchet | 43 | 1 (4) | 200 | ~0.1M | 3.5 | **10.85% (125)** | 6.0% | replicates fast climb; decays after peak without ratchet |
| forge-raw-N384-lr2.6e-3-s42 | countdown | 1.5B | FORGE raw, N=384, lr 2.6e-3 | 42 | 1 (2) | 200 | — | — | 0% | immediate collapse |
| forge-raw-N96-lr3.2e-4-s42 | countdown | 1.5B | FORGE raw, N=96, lr 3.2e-4 | 42 | 1 (1) | 200 | — | 1.5 | 2.05% (100) | 0.8% | degrades below base: η_max(96) < 3.2e-4 |
| forge-raw-N96-lr1.6e-4-s43 | countdown | 1.5B | FORGE raw, N=96, lr 1.6e-4 | 43 | 1 (1) | 200 | ~0.13M | 1.6 | 7.2% (175) | 6.1% | replicates s42 |
| forge3-cd-1p5b-rawN384-h100-s42 | countdown | 1.5B | **F2**: raw, N=384, lr 6.4e-4, DAPO 8/32, replay 0.5, ratchet | 42 | 2 (1,4) | 4000 | — | — | — | — | **launched 2026-09-11 11:35 PT** |
| forge-raw-N384-lr6.4e-4-s42 | countdown | 1.5B | FORGE raw, N=384 (k=4), lr 6.4e-4, no ratchet | 42 | 1 (4) | 200 | ~0.09M | 3.4 | **11.0% (200)**, 9.0% (75) | 11.0% | η_max grows with N: N=96 collapses at this lr |
| forge-raw-N384-lr1.6e-4-s42 | countdown | 1.5B | FORGE raw, N=384, lr 1.6e-4, no ratchet | 42 | 1 (4) | 200 | ~0.12M | 3.1 | 6.7% (200) | 6.7% | N alone at fixed lr: no gain vs N=96 |
| forge-raw-N96-lr1.6e-4-s42 | countdown | 1.5B | FORGE raw, N=96, lr 1.6e-4, no ratchet | 42 | 1 (4) | 200 | ~0.13M | 1.5 | **8.2% (200)** | 8.2% | lr sweep; fastest climb seen so far (F1 z-score needs ~1100 iters for 8%) |
| forge-raw-N96-lr6.4e-4-s42 | countdown | 1.5B | FORGE raw, N=96, lr 6.4e-4, no ratchet | 42 | 1 (4) | 200 | ~0.13M | 1.5 | 5.3% (25) | 0.05% | collapse after ~40 iters (η_max < 6.4e-4 at N=96) |
| forge-raw-N96-lr4e-5-s42b | countdown | 1.5B | FORGE raw, N=96, lr 4e-5, no ratchet | 42 | 1 (4) | 200 | ~0.13M | 1.6 | 2.3% (25) | 1.9% | too small |
| grpo-cd-1p5b-h100-s42 | countdown | Qwen2.5-1.5B-Instruct | GRPO (TRL 0.29, loss grpo, β=0, lr 1e-6, 64 prompts×8/step, vLLM colocate) | 42 | 1 GPU (3) | 2000 steps | 1.0M | ~19.5 (35 s/step) | — | — | **E1.6, in flight since 2026-09-10 19:00 PT** (wandb ctcmdgya→relaunch); HF ckpts every 100 steps, evaluate with scripts/eval_hf_ckpt.sh |
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

## E1.5 gradient alignment (2026-09-10, Qwen2.5-1.5B base, 1 H100) — `results/align_1p5b_base.json`

⟨ĝ,g_BP⟩ ≈ ‖g_BP‖² (unbiased) at all N; cos(ĝ,g_BP) = √(N/d) (hybrid) / ¼√(N/d) (per-example): 6e-5 @N=96 → 2e-4 @N=1024. Same-rollout agreement at noise floor. See METHOD_REVIEW R3.

## GRPO reference — checkpoint evaluations (2026-09-11 01:00; same eval as FORGE/ES: countdown_eval 2000, greedy, 512 tok)

| step | generations | answer_acc |
|---|---|---|
| 100 | 51k | 40.25% |
| 200 | 102k | 41.95% |
| 300 | 154k | 44.30% |
| 400 | 205k | 45.50% |
| 500 | 256k | 45.60% |
| 600 | 307k | 45.90% |
| 700 | 358k | 46.05% |
| 800 | 410k | 45.15% |
| 1000 | 512k | 46.20% |
| 1200 | 614k | 46.95% |
| 1400 | 717k | 46.10% |

Base 1.75%. Train-set reward mean at step 669 = 0.68. TRL 0.29 GRPOTrainer: loss_type grpo, β=0, lr 1e-6 const, 64 prompts × 8, T=1, 512 tok, vLLM colocate.
