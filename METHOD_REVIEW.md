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
