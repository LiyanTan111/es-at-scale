# The Two Methods, Explained: Evolution Strategies (ES) vs. FORGE

Self-contained technical explanation of the two training methods compared in this
repository. Written 2026-09-10 for hand-off to other readers/agents. Everything here is
derived from the code in this repo (`es_at_scale/`) and the project documents
(`PROJECT_STATUS.md`, `PROJECT_PLAN.md`, `METHOD_REVIEW.md`, `ZO_RLVR_ROADMAP_v2.md`).

---

## 0. The shared setting: RLVR without backpropagation

**Task family: RLVR** (reinforcement learning with verifiable rewards). A language model
π_θ(y|x) is given a prompt x, produces a completion y, and a deterministic *verifier*
returns a reward R(x, y) — for the Countdown task, 1 if the `<answer>…</answer>` expression
uses each given number exactly once and evaluates to the target, else 0. The objective is

    J(θ) = E_x E_{y ~ π_θ(·|x)} [ R(x, y) ].

Standard RLVR (GRPO, PPO, DAPO, …) maximises J with **backpropagation**: it needs stored
activations, gradients and optimizer states, i.e. 3–12× the memory of inference.

**Both methods here are backprop-free.** They only ever *run* the model (generation or
teacher-forcing forward passes) and perturb its weights. If a machine can serve the model
with vLLM, it can train it with either method. Both hold the full parameter vector θ
(no LoRA, no low-rank adapters); for Qwen2.5-1.5B, d ≈ 1.5 × 10⁹.

**Shared infrastructure** (`es_at_scale/trainer/es_trainer.py`, `utils/worker_extension.py`):
- Ray launches E vLLM engines (one per GPU; tensor-parallel for ≥7B). Each engine holds a
  full copy of θ in bf16.
- Weight perturbations are **never stored**: a perturbation ε ~ N(0, I_d) is identified by
  an integer seed and regenerated on the GPU (`torch.randn(..., generator=seed)`), tensor by
  tensor, whenever needed. This is what makes full-parameter ES/ZO feasible at 1.5B–72B.
- An NCCL group across engines lets engine 0 broadcast updated weights to all others.
- Countdown protocol (from the ES paper): train set = 200 prompts
  (`datasets/train/countdown`), eval set = 2000 held-out prompts (`countdown_eval`),
  greedy decoding at eval, `max_tokens = 512`. The metric is **answer_acc**: fraction of
  eval prompts solved exactly (0/1). Base Qwen2.5-1.5B-Instruct: ≈ 1.6–1.8%.
- "Full budget" = 3 M generated completions during training (= 500 ES iterations).

---

## 1. Method A — Evolution Strategies at Scale (ES) — the baseline

Paper: *Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning*
(Qiu et al., arXiv 2509.24372). Code: `EvolutionStrategiesTrainer` (`es_trainer.py`),
entry `train_es_relay.py` (adds checkpoint/resume; algorithm unchanged).

### 1.1 What it optimises
ES treats the whole training pipeline as a black box **fitness function**

    F(θ) = (1/m) Σ_{i=1..m} R(x_i, greedy_θ(x_i)),   m = 200 (the entire train set),

evaluated with **greedy decoding** (temperature 0). Note this is *not* J(θ): it is the
accuracy of the deterministic greedy policy. The reward used for training on Countdown is
the paper's *shaped* reward `0.1·format + answer` (format = 1 if `<think>…</think>\n<answer>…</answer>`
is well-formed); eval reports the pure answer_acc.

### 1.2 One iteration (population size N = 30, σ = 10⁻³, α = σ/2 = 5 × 10⁻⁴)
For each of N seeds s_i (drawn deterministically from the iteration index):
1. **Perturb**: on an idle engine, θ ← θ + σ ε(s_i), ε regenerated from s_i in bf16, in place.
2. **Evaluate**: generate greedy completions for all m = 200 prompts; grade; F_i = mean reward.
3. **Restore**: θ ← θ − σ ε(s_i) (regenerate the same noise, subtract).
Population members are spread across engines (E engines evaluate E members in parallel;
⌈N/E⌉ sequential rounds).

Then the **update** (`update_weights_from_seeds`), with z-scored fitnesses
z_i = (F_i − mean F) / (std F + 1e-8) (`utils/reward_shaping.z_score`):

    θ ← θ + (α / N) Σ_i z_i · ε(s_i)          (accumulated in fp32, cast to bf16 once)

followed by broadcasting θ from engine 0 to all engines. This is OpenAI-ES / NES with
fitness shaping: the update is the population-weighted average direction. Because the
coefficients are z-scored, **every update has (approximately) the same norm** regardless
of how much the fitness actually varies.

### 1.3 Cost per iteration
- Generations: N × m = 30 × 200 = **6,000 completions** (the dominant cost).
- Weight ops: 2N noise regenerations of all d parameters (perturb + restore) + N for the
  update.
- Measured: A100-40GB ×4 engines: 41.5 s/iter; H100 ×2 engines: ~40–47 s/iter.
  Full budget (500 iters, 3 M generations) ≈ 5.8 h (A100 ×4) / ≈ 6.5 h (H100 ×2).

### 1.4 Why it works, and what it cannot do
- **Why it works despite a 0/1 reward and a 1.5B-dim search space**: each fitness F_i
  averages 200 prompts, so F(θ) is a *nearly continuous* function of θ even though every
  single R is 0/1 — the averaging supplies the smoothness that a single 0/1 outcome lacks.
  ZO variance still scales with the *effective* dimension of the fine-tuning landscape,
  which for LLM fine-tuning is far smaller than d (the same reason MeZO works).
- **Robustness**: greedy decoding + parameter-space exploration + z-scoring make ES
  insensitive to reward scale, immune to length/entropy collapse, and (per the paper)
  less prone to reward hacking than PPO/GRPO.
- **No per-prompt credit assignment**: every direction ε_i is scored by *one scalar* (the
  mean over 200 prompts). ES cannot tell which prompts a perturbation helped. It cannot
  replay individual successful answers (there is no per-example objective to replay into).
- **Every fitness evaluation costs full generation**: widening the population (more
  directions) or the batch (less noisy fitness) both cost 200 generations per unit.

### 1.5 Known results (this repo's replication, NERSC A100)
Countdown, Qwen2.5-1.5B-Instruct, full budget: **37.9%** (seed 42; paper 37.3%), 34.5% and
33.6% (seeds 43/44) → 35.3 ± 2.2%. Full 7-model replication (0.5B–8B, Qwen & Llama) done.
On MATH-500 (1.5B): no improvement at any budget. On the paper's conciseness task: works
(parameter-space exploration needs no output diversity).

---

## 2. Method B — FORGE (Forward-Only RLVR with GRZO Estimators) — our method

Code: `GRZOSurrogateTrainer` (`trainer/grzo_surrogate_trainer.py`, subclass of
`GRZOTrainer` in `grzo_trainer.py`, subclass of the ES trainer), entry
`train_grzo_surrogate.py`. Design doc: `ZO_RLVR_ROADMAP_v2.md`.

### 2.1 The idea in one sentence
Keep GRPO's rollouts, verifier rewards and group-relative advantages exactly; replace
*only the backprop step* by a zeroth-order (two-point, finite-difference) estimate of the
gradient of GRPO's **surrogate loss**, which needs nothing but forward passes.

### 2.2 The three-layer construction
```
Layer 0  True objective (measured at eval, never optimised directly):
             J(θ) = E[ 1{answer correct} ]
Layer 1  Reward: binary r = R(x,y) ∈ {0,1}  (v2; margin-shaped variants exist)
Layer 2  Group advantage (GRPO): for each prompt x_i, sample G = 8 completions at T = 1,
             Â_{i,g} = (r_{i,g} − mean_g r_i) / (std_g r_i + 1e-8);  groups with std = 0 are dropped
Layer 3  Surrogate loss and its forward-only gradient:
             L(θ) = − Σ_{i,g} Â_{i,g} · log π_θ(y_{i,g} | x_i) / |y_{i,g}|
             ∇L estimated by two-point ZO (below)
```

**Key identity (why the surrogate is the right object).** At the parameters that generated
the samples, ∇_θ L(θ) is exactly the REINFORCE / GRPO policy-gradient estimate
(∇J ≈ E[Â ∇ log π]). So estimating ∇L is estimating the policy gradient. The ZO estimate of
∇L is unbiased up to an O(σ²) smoothing term (standard two-point analysis). The
`/|y|` length normalisation makes L the *per-sequence-mean* GRPO loss (`loss_type="grpo"` in
TRL terms); `--no-length-norm` gives the sum.

### 2.3 The GRZO estimator (per-example perturbations)
Every (prompt, completion) pair j with Â_j ≠ 0 becomes a **scoring job** with its own
random direction ε_j (seed s_j). For each job:

    ℓ_j^± = − Â_j · log π_{θ ± σ ε_j}(y_j | x_j) / |y_j|
    δ_j   = ℓ_j^+ − ℓ_j^−            (antithetic / common random numbers)

`log π_{θ±σε}(y|x)` is read from vLLM in **one prefill pass** (`prompt_logprobs` on the
concatenated x‖y token ids — teacher forcing, no generation). Weights are set to θ ± σε_j
from a bf16 **master copy** in fp32 arithmetic (`perturb_from_master`) so the restore is
bitwise exact (an early bf16 in-place perturb→restore drifted by ≈ half the signal).

Estimator per direction: δ_j / (2σ) ≈ ⟨∇L_j, ε_j⟩ where L_j is the pair's own loss term.
Update over the N pending directions:

    coefficients  c_j = zscore(δ)         (v2 default, `--delta-norm zscore`)
                  c_j = δ_j / (2σ)        (raw, `--delta-norm none`)
    θ ← θ − (lr / N) Σ_j c_j ε(s_j),     lr = 5 × 10⁻⁴, applied when N ≥ min_directions = 64

(`update_weights_from_seeds_fp32`, noise regenerated from the same seeds in fp32, so the
update uses the identical vectors that were scored). Afterwards engine 0 broadcasts θ
and every engine refreshes its master copy.

With raw coefficients, E[update] ∝ (1/N) Σ_j ∇L_j = ∇L̄: an unbiased estimate of the
policy gradient. The z-score variant (chosen empirically in July) instead applies a
**fixed-norm** step every update, like ES's fitness shaping; see §3 and
`METHOD_REVIEW.md` for why this is the method's weakest design decision.

"Hybrid" mode (`--directions-per-step N --pairs-per-direction k`) scores each of N fresh
directions on a shared mini-batch of k pairs instead of one pair each (ES-style averaging
inside the estimator). Tested in July (N64×k8): no gain — averaging over pairs reduces
between-pair heterogeneity, which is not the dominant noise.

### 2.4 The v2 recipe's three additions (why they exist)
1. **Prioritised DAPO resampling** (`--dapo-target-groups 8 --dapo-draw 32`): with B = 8
   prompts and a base model that solves only ~4–5% of Countdown prompts, most groups are
   all-0 (advantage 0, no signal). After grading, if fewer than 8 groups are "active"
   (reward std > 0), draw 32 more prompts (half from prompts known to have produced active
   groups before, half uniform), roll them out, and pool until 8 active groups exist (max 4
   rounds). Typical: 2–3 rollout rounds per step, ~96 scoring jobs.
2. **Success replay** (`REPLAY_FRAC=0.5`, cap 4 per prompt, 512 total): every correct
   completion is banked (in memory); each step, replay_frac × (#on-policy jobs) banked pairs
   are added as scoring jobs with fixed advantage +1 (`REPLAY_ADV`). On sparse tasks a lucky
   solve would otherwise shape one update and be discarded. Measured: 3× faster
   time-to-8% on Countdown. Only possible because FORGE has a per-example objective; ES
   has none. (Replay itself is known in RLVR — RLEP, ExGRPO, EAPO — the novelty is the
   forward-only port.)
3. **Anchor ratchet** (`ANCHOR_RATCHET=1`, drop 0.03, patience 3, warmup 400): keep a
   `best/` checkpoint; if eval drops more than 0.03 below the best for 3 consecutive evals
   (after iteration 400), restore the best weights on all engines and halve the update
   scale (floor 0.125). Turns "climb → peak → decay" into "climb → hold". This is an
   *evidence-driven* schedule; it beat a preset cosine schedule in a controlled race.

Full v2 hyper-parameters: binary reward, z-score, DAPO 8/32, min-directions 64, replay
0.5, ratchet (0.03 / 3 / 400), lr 5e-4 constant, σ 1e-3, G 8, B 8, T 1.0, 512 tokens.

### 2.5 Cost per iteration (H100, 2 engines, measured 2026-09-10)
- Generations: B × G = 64 on the first rollout round, + 32 × 8 per DAPO round → typically
  **~600 completions/step** (vs ES 6,000).
- Scoring: 2 prefill passes per job (~96 jobs → ~192 prefills of ~800 tokens); ≈ 5 s.
- Weight ops: 2 perturbations per job (from master) + N noise regenerations for the update.
- Measured breakdown: rollout ≈ 8.6–12 s, scoring ≈ 5.5 s, update ≈ 1.5 s → **~15–19 s/iter**
  (≈ 12 s with 4 engines). Full budget as run = 4000 iterations ≈ 18–21 h (2 engines).
- The asymmetry the method is built on: *evaluating a direction costs a prefill, not a
  generation* — so the number of directions can grow at ~5 s per 96, while ES pays 200
  generations per direction.

### 2.6 What happened when v1 tried to skip the surrogate (why Layer 3 exists)
v1 applied ZO directly to the raw 0/1 reward of a *greedy* completion with one prompt per
direction. With greedy decoding, the objective is J_greedy(θ) = E_x[R(greedy_θ(x))] — a
**piecewise-constant** function of θ whose gradient is zero almost everywhere; all signal
sits on decision boundaries. Every direction whose ±σ perturbations flip nothing yields
δ = 0 ("dead groups", 60–90% of them). It never rose above the base model. The same
pathology appears in circuit-yield optimisation (the authors' TCAD work), and the fix is the
same: optimise a continuous surrogate whose gradient matches the objective's.

### 2.7 Known results (NERSC A100, 4 engines)
Countdown, 1.5B, full budget: v0 (no replay/ratchet) peaked 15.75% then collapsed to 7%;
**v2 reached 10.45% at 1574/4000 with no collapse** when compute ran out; the local H100
rerun (F1) is in progress. MATH-500: FORGE +2.3 pp over base (3 seeds) where ES is flat;
with the ratchet it holds ≥ base indefinitely. Conciseness: zero signal by construction
(all completions hit the token cap → no within-group variance → all advantages 0).
Low budget: first Countdown signal at ~5.6k generations, ~20× fewer than ES.

---

## 3. Side-by-side

| | **ES (baseline)** | **FORGE (ours)** |
|---|---|---|
| Objective optimised | F(θ): mean greedy accuracy over 200 prompts (shaped reward) | L(θ): GRPO surrogate; ∇L = policy gradient of J at the sampling point |
| Decoding during training | greedy (T = 0) | sampled (T = 1), G = 8 per prompt |
| What a "direction" is scored on | 200 full generations (one scalar fitness) | 1 (prompt, completion) pair, 2 prefill passes |
| Directions per update | 30 | ~96 (≥ 64 accumulated) |
| Generations per step | 6,000 | ~600 (64 + DAPO redraws) |
| Credit assignment | none (population-level) | per prompt and per completion (advantages) |
| Coefficient normalisation | z-score over the population | z-score over directions (v2) or raw δ/2σ |
| Update | θ += (α/N) Σ z_i ε_i | θ −= (lr/N) Σ c_j ε_j |
| Perturbation arithmetic | bf16 in place, perturb/restore | fp32 from bf16 master copy, bitwise restore |
| Stabilisers | none needed | DAPO resampling, success replay, anchor ratchet |
| Needs output diversity? | no | yes (all-equal groups carry no signal) |
| Per-iteration wall-clock (H100) | ~40–47 s (2 eng.) | ~15–19 s (2 eng.) |
| Iterations to full budget | 500 | 4000 |
| Countdown-1.5B @ 3 M generations | 35.3 ± 2.2% | ~10% (in progress) |
| MATH-500-1.5B | no signal | +2–3 pp, holds ≥ base |
| Memory | inference-level | inference-level + one bf16 master copy per engine |

### 3.1 The honest picture of where FORGE stands (see `METHOD_REVIEW.md`)
FORGE's estimate of ∇L carries **two nested noise sources**: (a) policy-gradient sampling
noise (64 rollouts/step build L̄; ES measures its objective with 6,000 generations/step),
and (b) zeroth-order projection noise, whose per-step signal-to-noise ratio scales like
√(N / d_eff) with N ≈ 96 directions. July ablations indicate (b) dominates. (b) is exactly
the term FORGE can reduce cheaply (more directions = more prefills), which is the first
lever in the current plan (B1: N = 96 → 256 → 1024). The z-score normalisation converts
that noise into fixed-size steps, producing the observed climb-peak-decay; the ratchet
patches the symptom. At equal generation budget ES is currently far ahead on Countdown;
FORGE's defensible advantages today are low-budget efficiency, per-prompt credit (signal
on MATH where ES has none), and inference-level memory.

---

## 4. Glossary
- **RLVR** — RL with verifiable rewards (a deterministic checker gives the reward).
- **GRPO** — Group Relative Policy Optimisation: G samples per prompt, advantage = group
  z-score of rewards, no critic.
- **DAPO** — a GRPO variant; here we borrow only its *dynamic sampling* (drop all-equal
  groups, keep sampling until enough informative ones).
- **ES / NES / OpenAI-ES** — evolution strategies: estimate an ascent direction from
  fitnesses of randomly perturbed parameters; z-scoring the fitnesses is "fitness shaping".
- **ZO (zeroth-order)** — gradient estimation from function values only; two-point:
  (f(θ+σε) − f(θ−σε)) / 2σ · ε.
- **MeZO / GRZO** — memory-efficient ZO fine-tuning of LLMs (MeZO: one direction per
  step, seed-regenerated; GRZO: per-example directions with group-relative normalisation).
- **Teacher forcing / prefill** — feeding a known token sequence through the model to read
  its log-probabilities; no sampling.
- **Master copy** — a bf16 copy of θ kept on each engine so perturbations are applied as
  master ± σε in fp32 and restored exactly.
- **Anchor ratchet** — restore-best-on-decline with learning-rate halving.
- **answer_acc** — fraction of the 2000 eval prompts solved exactly (greedy decoding).

## 5. Where to look in the code
- ES step: `es_trainer.py: train_step`, `evaluate_population_on_batch`;
  `worker_extension.py: perturb_self_weights / restore_self_weights / update_weights_from_seeds`.
- FORGE step: `grzo_surrogate_trainer.py: train_step` (rollout → DAPO → advantages → replay →
  jobs → `_score_jobs_two_point` → z-score → `update_weights_from_seeds_fp32`);
  `grzo_trainer.py: eval_step` (sharded eval, ratchet), `_save_latest` (resume).
- Rewards: `reward_function/countdown_grader.py` (`countdown_reward_fn` shaped,
  `countdown_answer_only_reward_fn` binary), `math_grader.py`.
- Runners on this machine: `scripts/local_forge.sh`, `scripts/local_es.sh`;
  GRPO reference: `scripts/grpo_countdown.py`.
