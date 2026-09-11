# METHOD_REVIEW — a running critical review of FORGE (started 2026-09-10)

Purpose: the user asked for continuous, results-driven scrutiny of the method itself —
including whether the GRZO estimator belongs in it and whether the whole construction is
sound. Each entry: what the evidence says, what it implies, what we change (if anything).
Newest entries at the bottom. Nothing here changes the plan by itself; changes go through
`PROJECT_PLAN.md` gates.

---

## R1 (2026-09-10) — Code-level review of the estimator, before any new H100 data

**What the code does (`grzo_surrogate_trainer.train_step`).** Per step: B=8 prompts × G=8
rollouts (DAPO re-draws until 8 groups have reward variance); advantages Â per group; one
*independent* Gaussian direction ε_j per (prompt, rollout) pair with |Â|>0, plus replayed
pairs with Â=+1; two prefill passes per pair at θ±σε_j give
δ_j = −Â_j·(log π_{θ+σε_j}(y_j) − log π_{θ−σε_j}(y_j))/|y_j|; coefficients c_j = zscore(δ)
across the ~96 jobs; update θ ← θ − lr·(1/N)·Σ_j c_j ε_j once N ≥ 64 directions are pending.

**1. Is it a valid estimator?** Yes. With raw coefficients δ_j/(2σ) the update is
(1/N)Σ_j⟨∇L_j,ε_j⟩ε_j, whose expectation is ∇L̄ (mean surrogate over pairs) up to O(σ²).
The "GRZO" choice — one direction per example — is a legitimate, cheap way to obtain N =
#pairs directions for 2·#pairs prefills. The REINFORCE identity makes ∇L̄ the policy
gradient at the sampling point. The construction is sound.

**2. Where the signal goes.** Two *nested* noise sources, and only one of them is cheap to
reduce:
- (a) **Policy-gradient sampling noise**: L̄ is built from 64 rollouts/step (ES measures its
  objective with 6,000 generations/step). More rollouts cost generations — exactly the cost
  we are trying to avoid. Prefill-cheap scoring does **nothing** for this term.
- (b) **ZO projection noise**: each term ⟨∇L_j,ε⟩ε has variance ∝ d·‖∇L_j‖² with d ≈ 1.5e9;
  per-step SNR ∝ √(N/d_eff). With N≈96 this is tiny; progress comes from drift accumulating
  over thousands of steps (MeZO regime). This term *is* cheap to reduce: more directions =
  more prefills only.
The NERSC ablation "hybrid-XL N64×k8 did not help" says averaging each direction over 8
pairs (reducing between-pair heterogeneity) is not the bottleneck ⇒ (b) dominates today.
**Implication:** bet B1 (N: 96 → 256 → 1024 directions, each scored on one random pair) is
the *right first lever*, and it is the one place where the cost asymmetry genuinely favours
FORGE. B2 (low-rank/structured ε, cf. EGGROLL) attacks the same term through d_eff.

**3. The z-score is the weakest design decision.** zscore(δ) across directions discards the
magnitude that makes ZO self-limiting near an optimum (raw δ/(2σ) → 0 as ‖∇L‖ → 0).
It turns every update into a fixed-norm step ≈ lr·√(d/N) (≈ 2.0 in L2 for lr 5e-4, N 96,
d 1.5e9 — about 5e-5 RMS per parameter, every update, regardless of signal). That is
literally the "random walk near the peak" pathology; the ratchet is a patch over it. The
team moved to z-score because raw coefficients gave "update starvation" — but that was
confounded with min_directions accumulation and 4–5% active groups at the time. A
signal-proportional step (raw δ/(2σ) with a properly re-tuned lr; or raw with RMS/Adam-style
normalisation of the *accumulated update across parameters*, not across directions; or
SNR-gated scaling = bet B3) is the principled fix and should be re-tested with DAPO on.

**4. The cost story must be stated per unit of progress, not per step.** At equal generation
budget (3M) ES is at 35% and FORGE at ~10%: "94× fewer rollouts per step" is not an
efficiency claim by itself. The defensible claims are (i) low-budget: first signal 20× cheaper
(exists), (ii) prefill-only scoring lets N grow cheaply (B1 will show whether that buys
accuracy), (iii) inference-level memory / any-serve-able-model (E1.4, C8). Wall-clock is
also currently *worse* than ES per iteration (12 s vs. ~28 s H100-est. per ES iter, but ES
needs 500 iters vs. our 4000: 14 h vs. ~4 h). The per-phase timers added today (E1.2)
will show whether the 12 s is rollout, scoring, or the 96×(perturb, score, restore) RPC
chain (GPU utilisation during the probe read 0–4%, which points at RPC/CPU overhead —
a systems fix, not a method limit).

**5. Is GRZO's per-example perturbation "reasonable"?** As an estimator, yes. As a design
*default*, it silently sets N = #pairs and ties the number of directions to the rollout
budget. Decoupling N from #pairs (score extra directions on re-used pairs) is essential and
is already supported by the hybrid path (`--directions-per-step N --pairs-per-direction 1`).

**6. What would falsify the method (so we know when to stop).** If B1 (N=1024) and B2
(low-rank) together do not move the Countdown-1.5B peak above ~15–20% at the ES budget, the
projection-noise explanation is confirmed and the paper becomes the efficiency/regime-map
story (Story B) with a *measured* noise decomposition as its analytical core.

**Actions adopted (via plan):** add **E1.7 noise decomposition** to Phase 1 — on one fixed
batch, (a) re-score the same pairs with fresh directions to measure Var_(b); (b) re-roll the
same prompts to measure Var_(a); report both vs. N and G. Cheap (minutes), and it is the
figure that justifies every bet in Phase 2. Also: re-test raw coefficients with DAPO+replay as
part of B3 (not a new bet; folded in).

**Parking-lot idea (needs lit check before any code):** *partial first-order hybrid* — the LM
head's exact gradient is available from the forward pass (softmax − one-hot) ⊗ h with no
activation storage for the body; ZO for the body + exact gradient for the head. cf. ElasticZO
(2501.04287). Would attack (b) for the parameters that matter most for token probabilities.

## R2 (2026-09-10, user) — z-score discards the gradient magnitude; measure alignment directly

The user sharpened R1 into the project's core question. Two points, adopted verbatim into
the plan:

1. **z-score(δ) removes the absolute magnitude of the estimate.** Late in training all δ_j
   become small, but as long as they differ *relatively*, the z-scored coefficients stay
   O(1) and the model keeps taking fixed-size steps → climb → overshoot/noise → decay, and
   the ratchet pulls it back. So the ratchet is most likely **treating a symptom introduced
   by the z-score update, not something FORGE needs.** With the raw estimator
   c_j = δ_j/(2σ) we have E_ε[c_j ε_j] ≈ ∇L_j and hence E[ĝ] ≈ ∇L̄ — the clean statement
   "FORGE estimates the GRPO policy gradient without backpropagation" holds; with z-score it
   does not.
2. **Measure cos(ĝ_FORGE(N), g_BP) directly** on a small model (0.5B or smaller), allowing
   one backward pass to get g_BP = ∇_θ L, for N ∈ {32, 64, 96, 256, 512, 1024}. If
   N↑ ⇒ alignment↑ ⇒ training improves, the mechanism is nailed: the main limitation is
   ZO estimation error, not the surrogate. This is far stronger than "256 directions beats
   96 by 2 pp". Placed at the very front of the plan (E1.5).

**Core question (boxed):** *Can cheap additional forward queries close the ZO-gradient
estimation gap?* If N = 256/1024 significantly improves both alignment and final
performance, the project turns from "an interesting forward-only RL trick" into a clear
scientific story.

Actions: E1.5 implemented as `scripts/grad_alignment.py` (pure HF/torch, one GPU; also
measures same-/cross-rollout agreement and the δ statistics needed to set a principled
raw-estimator learning rate); B1 redefined as raw estimator + N sweep + no-ratchet ablation.

## R3 (2026-09-10 evening) — E1.5 result: the estimator is exactly textbook ZO; alignment ∝ √(N/d)

Setup: Qwen2.5-1.5B base (d = 1.54e9), one fixed batch of 8 active Countdown groups
(16 pairs with |Â|>0 in rollout R0, 8 in R1), g_BP by autograd (fp32), ĝ(N) from seed-
regenerated directions on one H100. `results/align_1p5b_base.json`.

| scheme | N | cos(ĝ, g_BP) | predicted | ⟨ĝ, g_BP⟩ | ‖g_BP‖² | ‖ĝ‖ |
|---|---|---|---|---|---|---|
| per-example | 32 | 4.3e-5 | ¼·√(N/d) = 3.6e-5 | 14.5 | 14.87 | 8.7e4 |
| per-example | 96 | 5.9e-5 | 6.2e-5 | 9.7 | 14.87 | 4.2e4 |
| per-example | 256 | 4.3e-5 | 1.0e-4 | 5.2 | 14.87 | 3.1e4 |
| per-example | 512 | 1.8e-4 | 1.4e-4 | 17.9 | 14.87 | 2.5e4 |
| per-example | 1024 | 2.1e-4 | 2.0e-4 | 14.0 | 14.87 | 1.7e4 |
| hybrid (all pairs) | 8 | 4.0e-5 | √(N/d) = 7.2e-5 | 3.6 | 14.87 | 2.4e4 |
| hybrid | 32 | 1.7e-4 | 1.4e-4 | 20.7 | 14.87 | 3.1e4 |
| hybrid | 96 | 2.5e-4 | 2.5e-4 | 15.0 | 14.87 | 1.6e4 |

**Findings.**
1. **Unbiasedness confirmed quantitatively (claim C1 done):** ⟨ĝ, g_BP⟩ ≈ ‖g_BP‖² = 14.9 at every N
   (scatter is the expected ‖g‖²/√N), for both schemes. The implementation estimates the GRPO
   policy gradient correctly.
2. **Alignment is astronomically small and follows theory exactly:** cos ≈ √(N/d) for the
   hybrid scheme (2.48e-4 observed vs 2.49e-4 predicted at N=96) and ≈ (‖ḡ‖/rms‖g_j‖)·√(N/d)
   ≈ ¼·√(N/d) for the per-example scheme (16 nearly-orthogonal per-pair gradients). Going
   from N=96 to 1024 raises cos from 6e-5 to 2e-4. To reach cos = 0.01 one would need
   N ≈ 1.5e5 directions per step — ~1500× today's prefill budget.
3. **Same-rollout agreement cos(ĝ_A, ĝ_B) is at the noise floor (|·| < 5e-4) at every N**, as
   theory predicts ((N/d)·(‖ḡ‖/rms‖g_j‖)² ≈ 4e-8): two independent FORGE updates on the same
   data are essentially orthogonal. Cross-rollout agreement likewise.
4. **z-score does not change the direction** (cos_z ≡ cos_raw to 4 decimals): raw coefficients
   δ_j/2σ = ⟨g_j, ε_j⟩ are already zero-mean, so z-scoring is a per-step rescaling. Its only
   effect is the step norm: fixed lr·√(d/N) (z) vs. lr·rms‖g_j‖·√(d/N) (raw, shrinks with the
   gradient). The user's critique stands exactly in that form.

**Answer to the core question at the estimator level: no.** Cheap extra forward queries
cannot make a full-parameter Gaussian ZO estimate *point* along the gradient at d = 1.5e9;
the cosine gap is set by d, and N reduces it only as √N.

**Why anything trains at all, and what N really buys.** Progress does not require alignment:
E⟨ĝ, g⟩ = ‖ḡ‖² holds at any N, so the first-order loss decrease per step, lr·‖ḡ‖², is the
same as gradient descent with the same lr. The cost of the noise is second-order:
lr²·ĝᵀHĝ ≈ lr²·(rms‖g_j‖²/N)·tr(H). Hence the **stable learning rate scales linearly with N**
(and inversely with tr H), and progress per step at the stability limit ∝ N. This is the
MeZO/ES regime (convergence governed by the Hessian's effective rank, not by cosine), and
it is the precise, testable form of the core question:

> **Reframed core question: does FORGE's stable learning rate — and therefore its progress
> per step — scale linearly with the number of directions N?**

If yes, N = 4–16× (cheap prefills) gives 4–16× fewer steps to any accuracy level, which is
the "make it win" lever; if no, curvature along random directions (tr H) is the wall and
structured/low-rank directions (B2) or fewer effective dimensions are the only way out.

**Immediate consequence for recipe design.** With raw coefficients the step-norm-matched lr
(matching z-score's early step norm lr·√(d/N)) is ≈ 3.5–4.7e-5 for N≈96 per-example
(4e-5 used in probe P1). Under the linear-scaling hypothesis, N=384 admits ≈ 1.6e-4.

**Probes (Phase 2 / B1, redefined):** P1 raw N=96 lr 4e-5 (control: does raw train without
starving, no ratchet); P2 raw N=384 lr 1.6e-4 (the bet); P3 raw N=384 lr 4e-5 (N alone);
P4 raw N=96 lr 1.6e-4 (lr alone; expected to destabilise). 400 iters each, 1 engine, eval
every 25; metric = train reward slope and eval acc at equal iteration vs F1 (z-score).

## R4 (2026-09-10, user) — The mature story, and the experiment that nails the mechanism

The user's reading of E1.5, adopted as the project's framing:

> **FORGE does not reconstruct the gradient direction.** It takes **an unbiased but extremely
> noisy policy-gradient step**, and the role of N is **not to recover the gradient but to
> reduce the second-order noise penalty, thereby allowing a larger learning rate.**

The old story ("ZO is noisy, add directions until it looks like backprop") is dead — E1.5
shows it cannot happen at d = 1.5e9. The ZO-native story is: progress per step ∝ η‖ḡ‖²
minus a curvature penalty ∝ η²·(mean‖g_j‖²/N)·tr(H); so η_stable ∝ N and
iterations-to-target ∝ 1/N.

**E1.8 — local stability & curvature (the user's design, `scripts/local_stability.py`):**
on the *same* fixed batch and surrogate as E1.5, for N ∈ {64, 96, 256, 512, 1024}, sweep η
around η ∝ N with several independent direction draws, and measure the *exact*
ΔL = L(θ − η ĝ) − L(θ). Fit ΔL = −aη + bη² → η_opt = a/2b, η_max = a/b. Plot η_max vs N.
Then estimate tr(H) **forward-only** from random second differences
(L(θ+hu) + L(θ−hu) − 2L(θ))/h², E_u[uᵀHu] = tr(H), plus gᵀHg/‖g‖² along the true gradient,
and predict η_opt(N) = ‖ḡ‖² / (mean‖g_j‖²·tr(H)/N + ḡᵀHḡ) with mean‖g_j‖² = E[(δ/2σ)²]
(also forward-only). Closure to aim for:

    measured tr(H)  ⇒  predicted η_max(N)  ≈  observed η_max(N),   η_max ∝ N.

This is cheaper (minutes), cleaner, and more decisive than 20-hour training probes; P2–P4
are paused until E1.8 sets their learning rates. A side diagnostic is included: the training
code applies updates as `p_bf16.add_(u.to(bf16))`, so sub-ulp components of the update are
rounded away; E1.8 reports the fraction of the update norm that survives at each (N, η).

## R5 (2026-09-10, late) — E1.8 result: the batch surrogate is *not* a stability criterion

`results/stability_1p5b_base.json` (1.5B base, same 16-pair batch as E1.5, fp32, 3 direction
draws per N, η grid centred at η ∝ N spanning ×1/8…×16).

| N | ⟨ĝ,g⟩ | mean‖g_j‖² (fwd) | fit a | fit b | η_opt (fit) | best grid η (= top of grid) | ΔL there |
|---|---|---|---|---|---|---|---|
| 64 | 14.5 | 188 | 14.7 | 1924 | 3.8e-3 | 4.3e-4 | −0.006 |
| 96 | 11.6 | 214 | 12.0 | 763 | 7.9e-3 | 6.4e-4 | −0.007 |
| 256 | 12.0 | 200 | 12.1 | 641 | 9.5e-3 | 1.7e-3 | −0.019 |
| 512 | 13.3 | 208 | 13.3 | 549 | 1.2e-2 | 3.4e-3 | −0.039 |
| 1024 | 15.0 | 227 | 14.7 | 641 | 1.2e-2 | 6.8e-3 | −0.071 |

with ‖g‖² = 14.87, gᵀHg/‖g‖² = +69, and **tr(H) = −2.1e3 ± 0.4e3 (h = 5e-4), −1.8e3 ± 0.35e3 (h = 1e-3)**
(individual uᵀHu from −5.4e3 to +0.9e3).

**What happened.**
1. ⟨ĝ, g⟩ = ‖g‖² again at every N (unbiasedness, second independent confirmation), and the
   forward-only second moment E[(δ/2σ)²] = mean‖g_j‖² ≈ 200 is stable across N — these
   forward-only quantities are reliable.
2. **The surrogate loss has *negative* mean curvature along random directions.** Random
   parameter noise *lowers* L on this batch. Mechanism (checked with a split of L into
   positive- and negative-advantage terms): 14 of the 16 pairs carry Â < 0 (one success per
   group ⇒ seven failures at Â ≈ −0.38 each), and random noise makes every specific sequence
   less likely; for Â < 0 terms that *decreases* −Â·log π. The surrogate rewards "unlearning
   everything", and noise does that for free. Hence the quadratic-penalty model
   b = (mean‖g_j‖² tr(H)/N + gᵀHg)/2 predicts b < 0 — meaningless — while the *observed* b is
   positive and roughly N-independent (≈ 550–760 for N ≥ 96, close to gᵀHg/2 = 514): the
   curvature along ĝ is dominated by the signal component, not by the noise.
3. **The η grid was too low at every N**: the best grid point was always the top of the grid,
   and the parabola fits extrapolate η_opt ≈ 4e-3…1e-2 — 100–300× the training learning
   rate (raw-equivalent 4e-5; z-score's step norm ≈ 2.0 in L2 corresponds to ≈ 4e-5 here).
   At those η the step norm is ≈ 300 in L2 (≈ 40% relative change of a typical weight): the
   batch surrogate keeps decreasing while the model is being destroyed. The "best grid η ∝ N"
   pattern is an artefact of the grid centre scaling with N.

**Conclusion.** The single-batch surrogate decrease is not a proxy for training stability
in RLVR: (i) its curvature along random directions is negative (noise looks like progress),
(ii) per-batch surrogate gradients are nearly orthogonal across batches (E1.5:
cos(g_R0, g_R1) = −0.02), so a step's value can only be judged on the *objective* (accuracy),
not on another batch's surrogate. This is a genuine difference from the SFT setting where
MeZO-style local analysis works (convex-ish cross-entropy, positive tr H). It also warns
against any batch-loss-based line search / step-size adaptation for FORGE.

**What replaces it (decisive, still cheap):** the *training-level* sweep the user wants at the
end: raw estimator, N ∈ {96 (k=1), 384 (k=4)} × η ∈ {4e-5, 1.6e-4, 6.4e-4} (N=384 also 2.6e-3),
200 iterations each, eval every 25 (train reward slope + eval acc). Hypothesis η_stable ∝ N ⇒
N=384's best η ≈ 4× N=96's and its iterations-to-target ≈ ¼. Launched tonight on GPU 4 (N=96
series); N=384 series on Lane A after F1. A second local measurement that *is* meaningful:
the objective's noise tolerance — greedy train-prompt accuracy F(θ + h·u) vs h (deterministic,
ES-style fitness): how far θ can move randomly before accuracy degrades. This bounds the
cumulative random-walk displacement a training run can afford and, with the per-step noise
norm η·rms‖g_j‖·√(d/N), gives an η–N–horizon relation (to be done with the vLLM trainer
machinery; ~1 h).
