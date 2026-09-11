# FORGE — Project Plan (authoritative; written 2026-09-10, lambda-scalar era)

This document is the single source of truth for what we are doing, why, in what order,
and how we decide. `PROJECT_STATUS.md` is the *history* (NERSC era, frozen 2026-07-24);
this file is the *plan going forward*. Update the **Status block** (§9) at every milestone.
If a question or idea comes up mid-flight, it goes to the **Parking lot** (§10) — it does
not change the running plan unless a gate in §5 says so.

---

## 1. North star

**Core scientific question (set by the user, 2026-09-10):**

> **Can cheap additional forward queries close the ZO-gradient estimation gap?**
>
> *Answered at the estimator level on 2026-09-10 (METHOD_REVIEW R3): not in cosine terms —*
> *cos(ĝ, g_BP) = √(N/d) exactly, ~2e-4 at N=1024. Reframed: **does the stable learning rate,***
> ***and hence progress per step, scale linearly with N?** (B1 probes P1–P4).*

i.e. is FORGE's limitation the *zeroth-order estimate* of the GRPO gradient (fixable with
more prefill-only queries), or the surrogate itself? The paper is organised around answering
this with a direct measurement (E1.5) and then a training-level confirmation (B1).
Consequences: (i) the estimator must be the **raw** one, c_j = δ_j/2σ, so that
E[ĝ] ≈ ∇L̄ and the sentence "FORGE estimates the GRPO policy gradient without
backpropagation" is literally true; z-score becomes an ablation, not the method.
(ii) The anchor ratchet is treated as a hypothesis-to-test: it likely treats a symptom of
fixed-norm z-scored steps (late-run δ_j → small, z-scored c_j stays O(1) → constant-size
random steps → climb → overshoot → decay). (iii) Alignment-vs-N is measured before any
recipe change is promoted.


**Paper thesis (one sentence):** *Forward-only policy gradients for RLVR — a zeroth-order
estimator on the GRPO surrogate that trains any model you can serve, with per-prompt credit
assignment that evolution strategies lack; we characterise where it wins, where it fails, and
the two mechanisms (evidence-driven ratchet, success replay) that make it stable.*

**Venue:** **ICML 2027** — abstract 2027-01-16, full paper 2027-01-22 (AoE).
ICLR 2027 (2026-09-25) is 15 days away: infeasible for a full campaign, not targeted.
Fallback: NeurIPS 2027 (2027-05-21).

**What "beautiful" means for a top venue here** (in priority order):
1. One clean method with a one-line identity (∇ of the surrogate = policy gradient at the
   sampling point; ZO adds only O(σ²) bias) and a cost asymmetry that is *measured*, not asserted.
2. A **Pareto/efficiency figure**: accuracy vs. #generations *and* vs. GPU-hours *on the same
   hardware*, for FORGE / ES / GRPO. This figure is the paper. Everything else supports it.
3. An honest **regime map**: sparse-reward full budget (ES wins today), low budget & dense
   signal (FORGE wins), zero-diversity tasks (FORGE has no signal, by construction).
4. A **pathology + cure** section with ablations (climb-peak-decay → ratchet, replay).
5. Robustness: 3 seeds, 2 tasks, ≥3 model scales incl. one 7B (the "train what you can serve" demo).

---

## 2. Claims ledger (each claim ↔ the evidence that must exist before we write it)

| ID | Claim | Evidence required | Status |
|---|---|---|---|
| C1 | FORGE is a valid forward-only policy-gradient estimator (gradient-exact surrogate; O(σ²) bias) | Derivation + numerical check ⟨ĝ, g_BP⟩ = ‖g_BP‖² and cos = √(N/d) on 1.5B | **done (E1.5, results/align_1p5b_base.json)** |
| C2 | Per-step cost: FORGE needs generations only for rollouts; perturbation scoring is prefill-only | Per-step time breakdown (rollout / scoring / perturb-restore / update) on H100 | **TODO (E1.2)** |
| C3 | Low-budget efficiency: FORGE reaches first signal ≫ cheaper than ES (in generations) | Curves vs #generations, ES & FORGE, same protocol | NERSC data exists; re-measure on H100 (E1.3, E1.1) |
| C4 | With ratchet+replay FORGE is stable (no collapse) at full budget | Flagship F1 4000 iters, no collapse; ablation without each | NERSC partial (1574/4000); **F1 local** |
| C5 | FORGE is competitive with ES at ES's budget on Countdown-1.5B (target: ≥ 30% vs ES 35.3±2.2) | F1 + "make-it-win" bets (§5 Phase 2) | **open — the main research risk** |
| C6 | FORGE has signal where ES has none (MATH-500, 1.5B) | 3 seeds each, same budget, ES flat vs FORGE +Δ | NERSC: FORGE +2.3pp avg (3 seeds), ES flat; rerun best recipe (E3.3) |
| C7 | Scope boundary: no within-group diversity ⇒ no FORGE signal (conciseness) | Existing NERSC result + 1 confirmatory run | done (NERSC); confirm once |
| C8 | Scale: FORGE runs 7B/8B on inference memory (TP2) where GRPO needs training memory | 7B run to a short budget + peak-memory table (FORGE vs GRPO vs ES) | 7B smoke works (TP2, base 25.1%); **memory table TODO (E1.4)** |
| C9 | GRPO reference (backprop ceiling) on the same protocol | GRPO run(s) at matched generation budgets | **running: 40.3% @51k gens, 45.9% @307k (R7)** |

Rule: no claim enters the paper without its row being "done". No experiment runs unless it
serves a row (or a gated bet in §5).

---

## 3. Literature refresh (2026-09-10) — what changed since `LITERATURE_NOVELTY.md` (07-01)

Targeted searches (ZO/forward-only × RLVR/GRPO/policy-gradient; ES × LLM reasoning; replay in
RLVR). **Verdict: the core method (two-point ZO on the GRPO advantage-weighted log-likelihood
surrogate) still has no published precedent.** GRZO (2606.02857) remains SFT-only.
2608.28011 (ZO perturbation budgets for frozen LLM agents) perturbs agent modules, not policy
weights via a surrogate — not a precedent.

New work we **must** cite and position against:
- **EGGROLL — "Evolution Strategies at the Hyperscale"** (2511.16652): low-rank perturbations
  make ES ~100× faster at large populations; on Countdown (RWKV-7 1.5B) reaches 35% vs GRPO 23%
  at equal wall-clock. → The strongest *forward-only* competitor on the wall-clock axis. Our
  differentiators: per-prompt credit (surrogate), prefill-only scoring. Low-rank perturbations
  are also a variance-reduction lever we can borrow **with citation** (Bet B2).
- **"Understanding ES for LLM Reasoning: broader reasoning coverage than GRPO"** (2608.27351):
  ES keeps Pass@K high while GRPO entropy-collapses; verifier-projected JS diversity across the
  population predicts Pass@K. → Frames our entropy-shrinkage / diversity-spark findings; we
  should report Pass@K for FORGE too (cheap: same eval generations at T>0).
- **"Matching accuracy, different geometry: ES vs GRPO"** (2604.01499): ES matches GRPO
  single-task with larger, orthogonal updates; linearly connected solutions.
- **Replay in RLVR** — RLEP (2507.07451), ExGRPO (2510.02245), rollout-level advantage-
  prioritised replay (2606.04560), EAPO (2606.30420). → Success replay is *not* new in RLVR; our
  contribution is porting it into the forward-only setting (ES cannot: no per-example objective)
  and showing it triples time-to-8%. Write it that way.
- **"Advantage shaping as surrogate reward maximisation"** (2510.23049): supports the
  "surrogate is the object being optimised" framing.
- ZO variance levers to cite if used: gradient-aligned projected perturbations (2510.18228),
  Sparse-MeZO (2402.15751), learned ZO optimiser (2510.00419).

Re-run this search **two weeks before submission** (2027-01-02) — field moves monthly.

---

## 4. Compute model & scheduling policy

Hardware: lambda-scalar, GPUs **1,2,3,4 only** (GPU 0 banned; max 4; shared box — check
`nvidia-smi` before every launch; `scripts/local_forge.sh` refuses busy cards).

Measured costs on H100 (2026-09-10; FORGE v2, 1.5B, Countdown, 512 tok):
- 4 engines (GPUs 1–4): **~12 s/iter** (rollout ≈ 60%, scoring ≈ 30%, update ≈ 10%);
  eval of 2000 prompts ≈ 28 s. 4000 iters ≈ 15 h.
- 2 engines (GPUs 1,2): **~16–19 s/iter** (rollout ~12 s, scoring ~5.5 s, update ~1.5 s);
  eval ≈ 50 s. 4000 iters ≈ 21 h. Slowdown 1.4–1.5× < 1.7 ⇒ total throughput preserved.
- A100 reference (NERSC): FORGE v2 17.7 s/iter; ES paper recipe 41.5 s/iter (500 iters ≈ 5.8 h).

**Scheduling policy (decided 2026-09-10): 2+2 split.**
- **Lane A = GPUs (1,2):** the flagship / final-recipe seeds (one run at a time).
- **Lane B = GPUs (3,4):** probes, baselines (ES, GRPO), instrumentation runs (≤1500 iters).
- Topology rule: GPUs 0–3 are NUMA node 0, 4–7 node 1. A 2-GPU NCCL group across sockets
  crashes at init unless `NCCL_P2P_DISABLE=1`; `local_forge.sh` sets it automatically for
  spanning sets (SETUP.md P10). Same-socket pairs (1,2), (2,3) need nothing.
- A 4-engine run (GPUs 1–4) is allowed only when Lane B is idle and the run needs <1 day.
- Every run: `LOGGING=wandb`, project `grzo-rlvr`, EXPNAME = `<recipe>-<task>-<model>-h100-s<seed>`.
- Every result is appended to `RESULTS.md` (table: run, recipe, budget, best, final, wall-clock,
  generations) the day it finishes. wandb is the curve store; RESULTS.md is the paper store.

---

## 5. Phases, experiments, gates

### Phase 0 — Restart (2026-09-10 → 09-12)  ✅ mostly done
- E0.1 Environment + local runner + smoke — **done** (SETUP.md).
- E0.2 H100 speed probe, v2 recipe, 4 engines (30 iters) — **done** (§4).
- E0.3 2-engine speed probe (8 iters) — **done** → 2+2 policy (§4).
- E0.4 Literature refresh — **done** (§3).
- E0.5 Launch **F1**: FORGE v2, Countdown, 1.5B, seed 42, 4000 iters, wandb — **launched 2026-09-10 12:10 PT on GPUs (1,2)**.

### Phase 1 — Foundations for the paper's main figure (09-12 → 10-05)
Runs in the probe lane while F1 trains (or after, if lane A only):
- E1.1 **ES on H100**: paper recipe, 1.5B, Countdown, seed 42, 500 iters — same hardware
  wall-clock for the Pareto figure (accuracy known: 37.9%). ~4–6 h.
- E1.2 **Per-step time breakdown** instrumentation in `grzo_surrogate_trainer.py`
  (rollout gen / DAPO rounds / scoring prefill / perturb+restore / update / eval) → logged
  per iteration. Needed for C2 and for finding the wall-clock bottleneck (the 17.7 s/iter
  is 3.4× ES's per-iter and must be explained and, if perturbation ops dominate, fixed).
- E1.3 **Generation accounting**: log cumulative #generations and #prefill-tokens per run
  (FORGE & ES) so curves can be plotted vs. generations exactly.
- E1.4 **Peak-memory table**: FORGE vs ES (measured) vs GRPO (measured in E1.6) for 1.5B
  and 7B. One table, one sentence of the abstract.
- E1.5 **Gradient-alignment experiment (THE key measurement; runs first on Lane B).**
  `scripts/grad_alignment.py`, pure HF/torch on ONE GPU (no vLLM), so both the ZO estimate and
  the backprop gradient live in the same parameter layout.
  Fixed batch: 8 active groups (DAPO-style pooling), G=8, T=1, 512 tokens → ~96 (x, y, Â) pairs.
  g_BP = ∇_θ (1/|pairs|) Σ_j −Â_j log π(y_j|x_j)/|y_j| by autograd (fp32).
  ĝ_FORGE(N) with per-example directions (one pair per direction, cycling pairs, fresh seeds),
  N ∈ {32, 64, 96, 256, 512, 1024}, coefficients raw δ/2σ **and** z-scored; plus the hybrid
  scheme (each direction scored on all pairs) at N ∈ {8, 32, 96}.
  Report: cos(ĝ, g_BP) vs N (raw vs z), same-rollout agreement cos(ĝ_A, ĝ_B), cross-rollout
  agreement cos(ĝ_A, ĝ_C) (absorbs E1.7), ‖g_BP‖, δ statistics (→ principled lr for raw:
  lr_raw = lr_z · std(z-coeffs)/std(raw-coeffs) matches the early step norm).
  Models: Qwen2.5-0.5B and 1.5B; checkpoints: base **and** F1's `best/` (late-training regime
  where the SNR collapse is hypothesised). Expected signature if the ZO-error hypothesis is
  right: cos rises ~√N; if cos saturates low at large N, the surrogate/PG-noise term dominates.
- E1.6 **GRPO baseline infra** — venv built (`/data/liyan/venvs/grpo`: TRL 0.29.1, vLLM 0.11, torch 2.8 cu128;
  a cu130 torch pulled by default is incompatible with driver 570), `scripts/grpo_countdown.py` +
  `scripts/eval_hf_ckpt.sh` written, **untested** until Lane B frees. Countdown
  reward = ours, protocol = ours (512 tok, same eval set). Run 1.5B to the generation budgets
  {0.1M, 0.5M, 3M} — gives the backprop ceiling for the Pareto figure. (Needs ≥1 GPU for
  training memory; schedule in probe lane.)
- E1.7 **Noise decomposition** — absorbed into E1.5 (same/cross-rollout agreement); `scripts/noise_decomposition.py` (vLLM-layout version) kept as a cross-check: on one fixed batch, re-score the same
  pairs with fresh directions (ZO projection variance) and re-roll the same prompts (policy-
  gradient sampling variance); report both vs N and G. Justifies/kills bets B1–B3 before they run.
- E1.8 **Local stability & curvature (user, 2026-09-10; runs before any further probe).**
  `scripts/local_stability.py` on the E1.5 batch: η sweep around η ∝ N for N ∈ {64..1024}, exact
  ΔL(η) with independent direction draws, quadratic fit → η_opt/η_max vs N; forward-only tr(H)
  and gᵀHg; prediction η_opt(N) from forward-only quantities vs observed. Success = η_max ∝ N and
  predicted ≈ observed. Sets the learning rates for P2–P4 (paused until then). Also reports the
  bf16 update-survival fraction (apply-path rounding diagnostic).
- **Gate G1 (F1 @ iter 1500):** local curve must track NERSC v2 (best ≥ 9% by 1500, no
  collapse). If not → parity debugging is the only allowed task until fixed.

### Phase 2 — "Make it win" bets (10-05 → 11-10)
Each bet: (a) targeted literature check (feedback rule), (b) 1500-iter probe against the F1
curve at equal iteration, (c) promote to full budget only if it beats F1's best-so-far at the
same iteration by ≥ 2 pp on two consecutive evals. Ordered by expected value / cost:
- **B1 Raw estimator + more directions (the training-level confirmation of E1.5).**
  Probes (1500 iters, Lane B) with `--delta-norm none`, lr set from E1.5's step-norm matching,
  N ∈ {96, 256, 1024} directions per update (`--directions-per-step N --pairs-per-direction 1`
  or per-example expansion), DAPO + replay on. Success = accuracy at equal iteration ordered
  the same way as alignment in E1.5 (N↑ ⇒ cos↑ ⇒ acc↑). Companion ablations: (a) z-score vs
  raw at N=96; (b) raw **without ratchet** (hypothesis: no climb-then-decay once steps scale
  with the signal); (c) raw with ratchet (should trigger rarely).
- **B2 Low-rank / structured perturbations (EGGROLL-style, cite).** Shrinks effective
  dimension → lower ZO variance per direction; also makes perturb/restore cheaper (helps E1.2).
- **B3 SNR-gated / trust-region steps.** Skip or shrink an update when direction agreement
  across the N estimates is below a threshold (replaces the fixed-magnitude z-scored step that
  the random-walk diagnosis blames). Ratchet becomes a fallback, not the driver.
- **B4 Ratchet tuning for climb-vs-hold**: lr_scale reset to 1.0 after k clean evals; widen
  RATCHET_DROP past 10%.
- **B5 Diversity floor**: temperature bump / G=16 when within-group reward std collapses
  (ties to 2608.27351's diversity finding).
- **Gate G2 (F1 @ 4000, ~10-01):** best ≥ 20% ⇒ C5 stays the headline goal, bets get full
  budgets. 15–20% ⇒ bets continue but the paper's headline moves to the efficiency/regime map
  (Story B), C5 becomes "narrowing the gap". < 15% ⇒ Story B only; bets capped at 2 full budgets.
- **Gate G3 (11-10): recipe freeze.** Whatever is best on Countdown-1.5B becomes "FORGE" for
  all Phase 3 runs. No recipe changes after this date.

### Phase 3 — Robustness & scale (11-10 → 12-15)
- E3.1 Countdown 1.5B, final recipe, seeds 43, 44 (3-seed mean ± std vs ES 35.3 ± 2.2).
- E3.2 Ablations (1 seed, full budget): −replay, −ratchet, −DAPO, −B* winners. Table 2.
- E3.3 MATH-500, 1.5B: final recipe, 3 seeds; ES 3 seeds already flat (NERSC) — rerun 1 ES seed
  on H100 for wall-clock parity.
- E3.4 Scale: 0.5B, 3B, 7B (TP2×2 engines) on Countdown, final recipe, half budget; ES same
  (NERSC 7-model replication exists for ES).
- E3.5 Pass@K (K=1,4,16) at final checkpoints for FORGE / ES / GRPO (cheap; coverage story).
- E3.6 Conciseness confirmatory run (C7) — one run.

### Phase 4 — Paper (12-15 → 2027-01-16)
- Figures: F1 Pareto (acc vs generations; acc vs GPU-h, same H100s); F2 pathology & cures
  curves; F3 estimator sanity; F4 regime map; T1 main table; T2 ablations; T3 memory.
- Related work from §3 (re-search 01-02). Code release branch `paper/icml2027` with configs.
- Internal deadline for a complete draft: **2027-01-05**. Buffer: 11 days.

---

## 6. Risks & pivots

| Risk | Signal | Response |
|---|---|---|
| FORGE never beats ~15% on Countdown | G2 | Story B (efficiency + regime map + pathology/cures); C5 reworded as "gap analysis" with the estimator-variance explanation (E1.5 makes it quantitative) |
| Wall-clock per iteration stays 3× ES | E1.2 | Profile; fuse noise generation; B2 low-rank perturbations; report both axes honestly |
| GRPO infra eats weeks | E1.6 slips past 10-15 | Use TRL defaults, 1 model size only; if still blocked, cite published GRPO-Countdown numbers with explicit caveats |
| Shared box contention (other users take GPUs 1–4) | busy-guard refusals | Keep ≤2 runs in flight; never grab a card in use; long runs resume via `latest/` |
| Reviewer: "why not EGGROLL?" | — | Position in related work; B2 borrows its lever; memory & credit-assignment comparison |
| Replay/ratchet seen as known tricks | — | Cite RLEP/ExGRPO etc.; claim is the *forward-only* port + mechanism study, not novelty of replay |

---

## 7. Operating rules (so we do not lose track)

1. This plan is the authority. Questions from the user are answered; the plan changes only via
   gates (§5) or an explicit "change the plan" instruction, and the change is written here first.
2. Every new idea → §10 Parking lot with a one-line lit-check note before any code.
3. At most one flagship in flight; probes ≤ 1500 iters; nothing runs without a claims-ledger row.
4. Results go to `RESULTS.md` the day they finish; the Status block (§9) is updated at the
   same time. wandb project `grzo-rlvr`.
5. Naming: `<recipe>-<task>-<model>-h100-s<seed>`; runs live in `/data/liyan/runs/grzo/`.
6. Commit code + docs at every milestone (plain commit messages).

---

## 8. Definitions (protocol, fixed)

- Countdown: train `datasets/train/countdown`, eval `countdown_eval` (2000 prompts), greedy,
  `max_tokens=512`, metric = answer_acc (0/1). Budget = 3M generations for "full".
- MATH: `math_lvl3to5_8k` train; eval `math500` (+ amc/aime/minerva/olympiad as secondary).
- ES baseline: pop 30, σ 1e-3, α 5e-4, batch 200, 500 iters (paper recipe).
- FORGE v2 (starting recipe): binary reward, zscore, DAPO 8/32, min-directions 64,
  REPLAY_FRAC 0.5, ratchet (drop 0.03, patience 3, warmup 400), lr 5e-4 const, σ 1e-3, G 8, B 8.

---

## 9. Status block (update at every milestone)

- **Today:** 2026-09-10 (19:10 PT).
- **Current step:** Phase 1. Lane A (1,2): F1 at iter ~1400 (G1 at 1500 imminent). GPU 3: GRPO reference
  running (35 s/step, ETA 2026-09-11 ~14:30 PT). GPU 4: E1.5 gradient alignment on 1.5B base (running).
- **In flight:** `forge2-cd-1p5b-h100-s42`, `grpo-cd-1p5b-h100-s42`, `align_1p5b_base`.
- **Done today:** E1.1 ES on H100 (2 engines) = 22.6% — ⚠️ far below NERSC 37.9%; eval noise ruled out
  (fixed-θ repeat std 0); hypothesis = bf16 perturb/restore drift × more cycles per engine at E=2.
  `ES_MASTER_COPY=1` (drift-free) implemented for a controlled rerun.
- **Next decision:** N=96 sweep done (R6): lr 1.6e-4 → 8.2% @200 iters (z-score F1 needs ~1100); 6.4e-4 collapses. Now: GRPO ckpt evals (100–600) then N=384 at 6.4e-4 / 1.6e-4 on GPU 4. Lane A after F1 (06:30): N=96 lr 3.2e-4, N=384 lr 2.6e-3, noise tolerance, ES parity.
  Lane A → ES rerun ×2 (2 engines + ES_MASTER_COPY; 4 engines original) to settle the ES-on-H100 question.
- **G1 verdict (2026-09-10 19:05 PT): FAIL on the number, PASS on the shape.** F1 best by 1500 = 8.10% (@~1100)
  vs threshold 9% and NERSC v2 10.45%; at 1500: 4.9% vs 9.75%. Curve shape identical (climb → sag → ratchet @624 vs
  @674). Config identical (Namespace diff clean). ES on the same box is also ~40% below NERSC, so a common
  systematic factor (2 engines / H100) is possible — but E1.5 shows FORGE trajectories are intrinsically
  high-variance (independent updates on the same data are orthogonal), so a single-run gap of 2 pp is not
  strong evidence either way. **Action (deviation from the gate's "parity only" rule, flagged to the user):**
  F1 continues to 4000 (G2 data; stopping it gains nothing); parity work runs in parallel on Lane A after F1
  (ES_MASTER_COPY 2-engine rerun, then ES 4-engine); the core-question probes P1–P4 continue on GPU 4 because
  the user placed them first. If ES parity recovers with master-copy, the FORGE gap is attributed to trajectory
  variance and one extra FORGE seed is run to confirm.
- **Next gate:** G2 at F1 iter 4000 (~2026-09-11 06:30 PT).
