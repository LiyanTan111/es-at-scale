# FORGE: Forward-Only RLVR with GRZO Estimators — Project Status & Handoff

**Last updated: 2026-07-24 (written for onboarding a new machine/session).**
This is the authoritative English summary of everything done so far.
Raw lab notebook (mixed-language, verdict-by-verdict): `EXPERIMENT_LOG.md`.
Novelty check: `LITERATURE_NOVELTY.md`. Original design doc: `ZO_RLVR_ROADMAP_v2.md`.
NERSC operational runbook: `NERSC_EXPERIMENT_RULES.md`.

---

## 1. What FORGE is

**Goal**: run RLVR fine-tuning (the GRPO family) with **no backpropagation** —
no stored activations, no gradients, no optimizer states. If a machine can
*serve* a model, it can *train* it.

**Method (3-layer construction)**:

```
Layer 0  True objective (scored, never trained on): J(θ) = E[1{answer correct}]
Layer 1  Margin smoothing (from our TCAD yield paper): r = exp(−relative_miss/τ)
         Variants: binary (pure 0/1), margin-tiebreak (binary + 0.2·margin), τ-anneal
Layer 2  Group advantage (from GRPO): Â = (r − group mean)/group std, G=8 per prompt
Layer 3  Surrogate + two-point ZO (GRZO):
         L(θ) = −Σ Â · log π_θ(y|x)/|y|
         δ_j = L(θ+σε_j) − L(θ−σε_j)   (antithetic, common random numbers)
         θ ← θ − (lr/N) Σ c_j ε_j      (seed-regenerated noise, no noise storage)
Guard    Spearman(margin, binary) on eval = calibration gate (anti-reward-hacking)
```

Key identity: at the sampling point, ∇L equals the true policy gradient
(REINFORCE), so the surrogate is gradient-exact; ZO adds only O(σ²) smoothing bias.

**Cost asymmetry vs ES**: evaluating a perturbation = one batched prefill
(re-reading existing rollouts), not new generation. FORGE generates 64–740
rollouts/step; ES (paper recipe) generates 6,000/step.

Baseline/competitor: "Evolution Strategies at Scale" (arXiv 2509.24372); this
repo is a fork of its official code. ES recipe: pop=30, σ=1e-3, α=5e-4,
batch=200 prompts, greedy decoding, shaped reward. Paper budget = 3M sample
evaluations = 500 ES iterations. Protocol for anything citable:
max_tokens=512 for Countdown, identical budget, identical eval (2000-sample
greedy answer accuracy).

---

## 2. Scoreboard (all full-budget = 3M rollouts, Qwen2.5-1.5B-Instruct, Countdown unless noted)

| Method | Peak | Final | Notes |
|---|---|---|---|
| ES (our replication, seed 42) | — | **37.9%** | 102% of paper's 37.3% |
| ES seeds 43/44 | — | 34.5 / 33.6% | **ES 3-seed = 35.3 ± 2.2%** → target zone 33–38% |
| FORGE v0 binary (old recipe) | 15.75% @2475 | 7.0% | climbs then collapses −55% |
| FORGE v0 margin-tiebreak | 10.15% @1100 | 7.7% | no collapse, lower peak |
| FORGE v0 τ-anneal | 9.75% @825 | 2.15% | worst; homotopy failed at full budget |
| **FORGE v2 "assault" (new recipe)** | **10.45% @1475 (in flight, 1574/4000)** | — | replay+ratchet; no collapse so far; band 8.6–10.45 |

**MATH (math500, base 54.0%)**: ES = no trend at ANY budget (3M gens, 30-eval
mean 53.6%). FORGE v0 peaks +2.3pp avg across 3 seeds (~57%) at ~80k gens then
degrades below base. Anti-decay fixes (below) hold FORGE at/above base
indefinitely (54.9% mean over late window vs controls' 49–52%), but ~57% looks
like a structural ceiling for this setup.

**Conciseness (ES-paper §4.2)**: FORGE has structurally zero signal — model
always hits the max_tokens cap → zero within-group length variance → no
advantage signal. ES's parameter-space exploration is immune. Documented as a
scope finding ("FORGE needs within-group diversity; ES doesn't").

**Model-scale matrix (FORGE v0, full budget)**: 0.5B best 1.25% (no "spark":
base-incapable model → no active groups → starvation); Llama-1B best 8.05%;
1.5B best 15.75%. ES beats FORGE at every scale on Countdown.

---

## 3. The central discovery arc

1. **v1 falsified**: ZO directly on raw 0/1 reward with greedy decoding
   optimizes a piecewise-constant objective (a.e. zero gradient) — same
   pathology as circuit yield. Dead on arrival; kept as motivation.
2. **Angle B works**: ZO on the GRPO surrogate loss is the viable construction
   (novel per our literature check — closest prior art: GRZO 2606.02857
   (SFT-only), ZO-PG-RLHF 2409.17401, MeZO 2305.17333).
3. **Update starvation** on sparse tasks fixed by prioritized DAPO resampling
   (`--dapo-target-groups 8 --dapo-draw 32`; half of redraws from known-active
   prompts). ~4–5% natural group-active rate on Countdown-1.5B.
4. **Margin smoothing** (TCAD transplant) is the only budget-axis win anywhere:
   3.15% @ 5.6k generations while ES is still at base (~20× cheaper than our
   binary variant). Pure margin at scale gets **reward-hacked** (margin↑ acc↓,
   Spearman gate 0.49→0.31 — the gate's first live kill); margin-tiebreak is
   the hack-proof form.
5. **The disease: late-run degradation.** EVERY FORGE run climbs → peaks →
   decays (Countdown 15.75→7; MATH 57.4→33). Gate healthy throughout ⇒ not
   hacking. Mechanism hypothesis (twice independently supported): z-scored
   coefficients force constant-magnitude updates; near a peak the
   signal-to-noise collapses and fixed-size steps become a random walk that
   drifts off the peak. Compounded by policy-entropy shrinkage.
6. **The cures (validated on the MATH fast-testbed, then Countdown)**:
   - **Anchor ratchet** (`ANCHOR_RATCHET=1`): keep a stable `best/` checkpoint;
     if eval drops >RATCHET_DROP below best for RATCHET_PATIENCE consecutive
     evals, restore best weights on all engines + refresh masters + halve the
     update scale (floor 0.125). Turned MATH climb-then-decay into a permanent
     hold ≥ base (first time ever). Needs RATCHET_WARMUP (~400) so a lucky
     early eval doesn't become the anchor.
   - **Cosine lr**: also prevents decay but freezes late progress —
     **evidence-driven schedule (ratchet) beats preset schedule (cosine)**;
     the in-flight race (v1 cosine vs v2 const+ratchet) confirms v2 ahead ~2pp
     at equal iteration.
   - **Success replay** (`REPLAY_FRAC=0.5`): bank every correct rollout
     (in-memory, per-prompt cap 4, global cap 512); inject replayed pairs as
     fixed-advantage (+1) surrogate jobs each step. On Countdown this **3×'d
     time-to-8%** (800 vs 2400 iters). This is a FORGE-only capability — ES
     has no per-example objective to replay into.
   - Rejected by ablation: σ ∈ {5e-4, 2e-3} (1e-3 stays), hybrid-XL N64×k8
     (shared-exam averaging reduces exam noise, not direction noise).

**Current best recipe ("FORGE v2")**: binary reward + zscore + prioritized
DAPO (8/32) + min-directions 64 + replay 0.5 + anchor ratchet (drop 0.03,
patience 3, warmup 400) + constant lr 5e-4 + σ 1e-3.

---

## 4. What is running / queued right now

- `forge2-cd-assault-v2` (the leader): Countdown full-budget 4000 iters,
  FORGE v2 recipe. At 1574/4000: best 10.45%, band stable, 3 ratchet
  triggers, no collapse. **This is the run that answers "can FORGE reach ES's
  33–38% zone".**
- `forge2-cd-assault` (v1, cosine arm of the race): 3374/4000, frozen ~6.8%.
  Race verdict nearly final; keep to completion as the controlled comparison.
- Both relaunched via relay drivers on **QOS=overrun** (free tier) because
  **the m4788 GPU allocation is exhausted** (salloc rejects: repo balance
  ~0.16 node-hours). Overrun only runs when the cluster has idle capacity —
  it may sit PENDING for a long time.
- Two ES-seed regular jobs completed just before the money ran out (that's
  where the 34.5/33.6 came from).

## 5. Immediate next steps (when compute exists again)

1. Finish `forge2-cd-assault-v2` → if it enters 15%+ still holding, extend
   (the ratchet makes longer budgets monotone-ish, so budget is pure upside).
2. Tune the ratchet for climb-vs-hold: current settings may cap the climb
   (lr floor 0.125 after repeated triggers). Ideas: reset lr_scale to 1.0
   after N clean evals above the anchor; or widen RATCHET_DROP once past 10%.
3. Replay strength ablation on Countdown (0.25 / 0.5 / 1.0) + REPLAY_ADV
   scaling (currently fixed +1; consider group-normalized).
4. FORGE v2 seeds 43/44 on Countdown → 3-seed vs ES's 35.3±2.2.
5. MATH with v2 full recipe (replay should help less there — dense signal —
   but ratchet holds the band; target: push past the 57% ceiling).
6. If v2 stalls below ~15%: next-tier ideas — entropy floor / temperature
   bump when within-group reward std collapses; direction-consistency-gated
   updates; G=16; curriculum from replay buffer (train on solved prompts'
   neighbors).

## 6. Compute situation (CRITICAL for new machine)

- **NERSC m4788 GPU node-hours: EXHAUSTED** (~53k+ of 62,289 used; AY2026
  runs to 2027-01-19; no automatic mid-year refresh). Our usage: 5,672 (~9%).
- Options: (a) PI requests a supplemental allocation from NERSC (project
  demonstrably used its time — good case); (b) run on `QOS=overrun` (free,
  preemptible, may queue indefinitely); (c) another repo/account via
  `-A <repo>`; (d) another cluster entirely — the code is portable: only
  needs 4× ~40GB GPUs, vLLM, Ray (see §7).
- Relay scripts accept `QOS=` and `PEND_PATIENCE=` env vars.
  All Slurm-facing bits live in `scripts/relay_grzos.sh` / `relay_es.sh` —
  porting to another scheduler means rewriting only the salloc/srun wrapper.

## 7. Code map (all FORGE code is ours; upstream = ES paper's repo)

```
es_at_scale/trainer/grzo_surrogate_trainer.py  FORGE core: rollout→advantage→
    two-point surrogate scoring→accumulate→seed-regen update. DAPO resampling,
    replay buffer, hybrid m>1 scoring, τ-homotopy, lr schedule.
es_at_scale/trainer/grzo_trainer.py            Base: sharded eval, Spearman gate,
    best-ckpt saving, ANCHOR RATCHET (in eval_step), relay resume (latest/ +
    progress.json incl. ratchet state), save-latest throttling.
es_at_scale/utils/worker_extension.py          In-engine weight ops: master-copy
    exact perturb/restore (bf16-drift fix), fp32 seed-regen updates, chunked
    fp32 apply (7B OOM fix), TP-rank-suffixed checkpoint save/load (TP2 fix),
    atomic saves (tmp+rename).
es_at_scale/trainer/es_relay_trainer.py        ES original + relay/resume.
es_at_scale/train_grzo_surrogate.py            FORGE entry point (all flags).
es_at_scale/train_es_relay.py                  ES entry point (countdown/math).
es_at_scale/reward_function/countdown_grader.py  binary/margin/margin-tiebreak/
                                                  anneal rewards + parse.
es_at_scale/reward_function/math_grader.py     boxed 0/1 + margin-tiebreak.
es_at_scale/reward_function/conciseness_grader.py  length reward (paper §4.2).
scripts/relay_grzos.sh, relay_es.sh            4h-relay drivers (salloc loop,
    resume via latest/progress.json; env-driven config incl. QOS, ratchet,
    replay). Launch with setsid, NOT nohup (survives session death).
scripts/submit_regular_*.sh                    sbatch chain builders (24h chunks,
    --dependency=afterany, --resume).
scripts/build_conciseness_dataset.py           paper Tables 4/5 verbatim.
```

**Env knobs** (read by trainer at process start; relay scripts pass them):
`ANCHOR_RATCHET, RATCHET_DROP, RATCHET_PATIENCE, RATCHET_WARMUP,
REPLAY_FRAC, REPLAY_ADV, REPLAY_CAP_PER_PROMPT, REPLAY_MAX,
GRZO_GPU_MEM_UTIL, SAVE_LATEST_FREQ`.

**Numerical landmines already fixed (do not regress)**:
1. bf16 in-place perturb→restore drifts ≈ half the ZO signal → master-copy
   scheme, restore is bitwise.
2. Scoring-side and update-side noise must be the SAME fp32 vector.
3. 7B+ on 40GB A100: TP1 impossible with master copy (model 14.2 + master
   14.2 + vLLM workspace 4.5 + noise 2 + ctx 1.8 > 40G) → TP2 ×2 engines,
   util 0.6, chunked fp32 apply.
4. TP>1: checkpoints are per-rank shards (`.rank0/.rank1` suffix).
5. Checkpoint writes are atomic (tmp+rename) — wall-time SIGKILL used to
   truncate them.
6. Save-latest throttled to ≥300s — unthrottled 3GB/74s writes tripped Ray's
   95% host-RAM OOM killer.
7. `PYTHONUNBUFFERED=1` everywhere (sbatch stdout block-buffering looks like
   a hang).
8. vLLM `max_model_len=4096` (Llama-3.1's 131k default demands 16GB KV).

## 8. Run artifacts

- Training runs + checkpoints: `/pscratch/sd/l/liyantan/runs/grzo/<expname>/`
  (`latest/` = resume point incl. ratchet state; `checkpoints/` = best-acc;
  `best/` = ratchet anchor). SCRATCH purges after 90 days.
- Durable archives (CFS): `/global/cfs/cdirs/m4788/liyantan/models/
  {grzo-baselines,grzo-angleb}/` — all completed finals/bests.
- Logs: `logs/relay/<expname>.log` (training), `logs/relay/lane_*.driver.log`
  (relay drivers), `logs/*.out` (sbatch chunks). Eval curves are grep-able:
  `grep "answer_acc (0/1)" logs/relay/<exp>.log`.
- W&B: project `grzo-rlvr` (user liyan_tan) — every run, live. Group by run
  name (each relay hop is a separate wandb run with the same name).

## 9. Paper framing (as of the "make it work" pivot)

The user's directive (2026-07-21): goal is to MAKE FORGE WIN, not to write a
negative-results paper. Current narrative assets if a paper is written now:
- The efficiency/domain map: FORGE wins low-budget sparse (20× cheaper to
  first signal) and is the only method with any MATH signal; ES wins
  full-budget sparse; conciseness shows the diversity-spark boundary.
- The degradation pathology + cures (ratchet/replay) — general contributions
  for ZO-on-LLM training.
- The calibration-gate methodology (live hack detection) from the TCAD paper.
- Full 7-model ES replication (validates our baseline: 1.5B at 102% of paper).
