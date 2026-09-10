# Angle B novelty search (2026-07-01)

**Target method (Angle B):** forward-only zeroth-order **two-point** estimate of the
GRPO **advantage-weighted log-likelihood surrogate** `L(θ) = −Σ_{i,g} Â_{i,g}·log π_θ(y_{i,g}|x_i)`,
where rollouts + group-relative `Â` come from GRPO exactly, and only ∇L is estimated
forward-only via parameter perturbations + teacher-forcing forward passes (read log π
via vLLM `prompt_logprobs`). No backprop, no critic/ref, no importance sampling.

## Verdict
The **exact** method appears **novel / unpublished**: no work found that runs ZO on the
GRPO advantage-weighted log-likelihood surrogate for LLMs/RLVR. The pieces exist
separately; the combination (ZO estimator on the RL surrogate loss, on LLMs) does not.
Confidence: high on the SFT/ES lines (full-text checked); medium on the RLHF-ZO-PG line
(abstract-level).

## Closest prior art (what a reviewer would cite)
1. **GRZO — Group-Relative Zeroth-Order Optimization** (arXiv 2606.02857). Closest
   *methodologically* + name collision: per-example perturbation + group-relative
   normalization of ZO loss differences `aᵢ=δᵢ/(s+ε)`. **SFT-ONLY** — GLUE/SuperGLUE/
   SQuAD/DROP, cross-entropy; no RL/RLVR/advantage/policy-gradient (only a "GRPO-style
   normalization" borrow). *Angle B = feed an RL policy-gradient surrogate through this
   estimator.* This is the direct ancestor of our estimator; our novelty is the objective.
2. **Zeroth-Order Policy Gradient for RLHF without Reward Inference** (arXiv 2409.17401).
   Closest *conceptually*: ZO policy-gradient for RLHF. But estimates the **value-function
   difference from preferences** then ZO-approximates the PG — preference/value-based, NOT
   the teacher-forcing log-likelihood surrogate; theory + small stochastic environments
   (not LLM-scale), not RLVR, not GRPO groups.
3. **MeZO — Fine-Tuning LMs with Just Forward Passes** (arXiv 2305.17333). Foundational
   forward-only ZO for LLM fine-tuning (SFT). The lineage our estimator sits in.

## Contrast line (perturb-for-exploration, NOT forward-only update — Angle A cousins)
- **PSN-RLVR — Parameter-Space Noise for RLVR** (arXiv 2602.02555): perturb params for
  trajectory-level exploration, but **backprop update + truncated importance sampling**.
- **Adaptive Layerwise Perturbation** (arXiv 2603.19470): learnable perturbations + single
  importance ratio, off-policy correction. Both still backprop + need IS — opposite of B.
- **ES-at-Scale** (base repo): ES/ZO on **raw reward**, m=200 aggregation, no per-prompt
  credit. Our v1 was the m=1 sparse-binary degenerate case (falsified).
- **Training-Free GRPO** (arXiv 2510.08191): no parameter updates at all (in-context). N/A.

## Convergence-speed risk levers (ZO variance vs effective dim; low-rank/subspace ZO)
- Sparse MeZO (2402.15751), LoZO (low-rank ZO), Adaptive Bayesian Subspace Optimizer
  (2601.01452), Learnable Direction Sampling (2602.13659), Variance-reduced ZO / MeZO-SVRG
  (2404.08080), FZOO (2506.09034), Learning a ZO Optimizer (2510.00419).
  → if convergence too slow, pull the effective-dimension lever (subspace / low-rank / more
  directions), consistent with roadmap §8.4.

## Residual gaps
- 2409.17401 characterized at abstract level (OpenReview page was behind a bot check; used
  arXiv abstract). Worth a full-text pass before writing a related-work section.
- Field moves fast (GRZO itself is 2606); re-run this search near submission.
