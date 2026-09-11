"""Angle B: GRZO on the GRPO surrogate loss (ZO-RLVR roadmap v2, section 4).

v1 (GRZOTrainer) differenced the RAW 0/1 reward under greedy decoding -- falsified:
that objective J_greedy(theta) = E_x[R(greedy_theta(x))] is piecewise constant, its
gradient is zero a.e., and dead groups are inevitable. This trainer keeps GRPO's
rollout and group-relative advantage EXACTLY, and replaces only the backprop step
with a two-point zeroth-order estimate of the CONTINUOUS surrogate loss

    L(theta) = -(1/|D|) sum_{i,g} A_hat_{i,g} * log pi_theta(y_{i,g} | x_i),

whose gradient at the sampling point equals the policy gradient.

Per training step:
  1. ROLLOUT (= GRPO): sample G responses per prompt from pi_theta (T=1, vLLM),
     grade 0/1, group-normalize within each prompt's G -> A_hat_{i,g}.
     Degenerate groups (all-same reward) get A_hat = 0 -> their pairs are SKIPPED
     (DAPO-style data filter; this is data-level sparsity, it no longer kills the
     estimator like v1's estimator-level dead groups).
  2. SCORE: for each active pair (x_i, y_{i,g}) draw an independent seed s;
     teacher-forcing forward at theta +/- sigma*eps(s) reads
     logpi_pm = log pi_{theta +/- sigma*eps}(y|x) via vLLM prompt_logprobs
     (token-id prompts, zero re-tokenization). The pair's ZO signal is
         delta = -A_hat * (logpi_plus - logpi_minus)     [~ directional deriv of L]
     with logpi optionally length-normalized (mean per-token logprob) so long
     sequences don't dominate.
  3. UPDATE: z-score delta across active pairs (GRZO-layer normalization --
     distinct from the GRPO-layer reward normalization in step 1), then
         theta <- theta - lr * (1/N_act) sum_j deltanorm_j * eps(s_j)
     via seed regeneration (fp32 noise, identical vector to scoring side).
     Broadcast, refresh per-engine master copies.

Perturbation uses the master-copy scheme (perturb_from_master /
restore_from_master): bitwise-exact restore, no bf16 round-trip drift.

Forward-only throughout: no backprop, no activations, no critic/ref, no
importance sampling. Scoring forwards are prefill-only (max_tokens=1).
"""

import numpy as np
import ray
import torch
from vllm import SamplingParams

from es_at_scale.trainer.grzo_trainer import GRZOTrainer


class GRZOSurrogateTrainer(GRZOTrainer):
    def __init__(self, *args, rollout_temperature=1.0, lr=1e-6,
                 normalize_by_length=True, delta_norm="zscore",
                 min_directions=1, dapo_target_groups=0, dapo_max_rounds=4,
                 dapo_draw=0, pairs_per_direction=1, directions_per_step=0,
                 lr_schedule="const", anneal_reward_fn=None,
                 anneal_tau=None, dir_multiplier=1, **kwargs):
        super().__init__(*args, **kwargs)
        # N directions per step decoupled from #pairs: every scoring job (pair) is
        # replicated dir_multiplier times with fresh direction seeds (per-example
        # scheme, pairs cycled). N = dir_multiplier x #pairs. 1 = legacy behaviour.
        self.dir_multiplier = int(dir_multiplier)
        self.rollout_temperature = float(rollout_temperature)
        self.lr = float(lr)
        self.normalize_by_length = bool(normalize_by_length)
        assert delta_norm in ("zscore", "none")
        self.delta_norm = delta_norm
        # Accumulate scored directions across steps and update only once >= this
        # many are collected. Small per-step N (8-16 active pairs) makes z-scored
        # fixed-size updates noise-dominated; accumulating restores the N~64
        # geometry that v1 validated. 1 = update every active step (no change).
        self.min_directions = int(min_directions)
        self._pending_seeds = []
        self._pending_coeffs = []
        # DAPO-style dynamic resampling: after grading, if fewer than
        # dapo_target_groups prompt-groups are active (non-degenerate), draw more
        # prompts from the train set and roll them out too, up to dapo_max_rounds
        # extra rollout rounds. Rollouts are cheap (~6s); update starvation from
        # ~22% active-step density is the bottleneck this removes. 0 = off.
        self.dapo_target_groups = int(dapo_target_groups)
        self.dapo_max_rounds = int(dapo_max_rounds)
        self.dapo_draw = int(dapo_draw)  # prompts per resample round (0 -> 4*B)
        self._dapo_pool = None  # lazily bound to the train dataset
        # Prioritized resampling: dataset idx -> times this prompt produced an
        # active (non-degenerate) group. The model can only solve a small subset
        # of the train set (~4-5% group-active rate); random redraws mostly miss.
        # Half of each DAPO draw comes from known-active prompts, half explores.
        self._prompt_stats = {}
        # Hybrid (direction-level m>1): each direction's surrogate delta is the
        # MEAN over k pairs (shared across all directions per step, CRN-style,
        # like ES's shared batch) instead of a single pair. Imports ES's
        # m-averaging variance reduction while keeping per-prompt advantages.
        # pairs_per_direction=1 (+ directions_per_step=0) = legacy behavior.
        self.pairs_per_direction = int(pairs_per_direction)
        self.directions_per_step = int(directions_per_step)
        # lr schedule: every completed run showed late-run instability/decay at
        # constant lr; cosine anneals the step size to ~0 by num_iterations.
        assert lr_schedule in ("const", "cosine")
        self.lr_schedule = lr_schedule
        # tau homotopy (continuation method): start with a wide margin tau
        # (dense gradient, fast start) and anneal tau -> tau_end so the margin
        # reward smoothly degrades toward the binary indicator (exactness
        # requirement hardens; hacking window closes). anneal_reward_fn is the
        # RAW margin fn accepting tau=; anneal_tau=(tau0, tau_end).
        self.anneal_reward_fn = anneal_reward_fn
        self.anneal_tau = anneal_tau
        # Master copies on every engine (weights are final here: checkpoint, if
        # any, was loaded in the parent __init__ before this line).
        ray.get([
            e.collective_rpc.remote("save_master_weights", args=())
            for e in self.engines
        ])
        print("[GRZO-S] master weights saved on all engines")

    # ------------------------------------------------------------------ #
    # Grading helper: per-rollout 0/1 rewards (not the per-prompt mean that
    # _postprocess_outputs computes).
    # ------------------------------------------------------------------ #
    def _grade_rollouts(self, outputs, targets):
        """outputs: list of RequestOutput (len B, each with G completions).
        Returns rewards float[B][G] using the trainer's reward_function via the
        timeout pool."""
        from multiprocessing import TimeoutError as MpTimeout
        pending, rewards = [], []
        for i, out in enumerate(outputs):
            row = []
            for g, comp in enumerate(out.outputs):
                res = self.mp_pool.apply_async(self.task, (comp.text, targets[i]))
                row.append(res)
            pending.append(row)
        for row in pending:
            rrow = []
            for res in row:
                try:
                    _, r = res.get(timeout=self.reward_function_timeout)
                    rrow.append(float(r))
                except MpTimeout:
                    rrow.append(0.0)
            rewards.append(rrow)
        return rewards

    # ------------------------------------------------------------------ #
    # Teacher-forcing scoring of one batch of jobs at theta +/- sigma*eps.
    # ------------------------------------------------------------------ #
    def _score_jobs_two_point(self, jobs):
        """jobs: list of dicts with keys seed, xy_ids (full token ids), n_resp.
        Round-robin over engines; per job: perturb(+) -> logpi+, perturb(-) ->
        logpi-. Returns (logpi_plus, logpi_minus) lists aligned to jobs.
        No restore between jobs (perturb overwrites from master)."""
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        E = self.n_vllm_engines
        lp_plus = [0.0] * len(jobs)
        lp_minus = [0.0] * len(jobs)

        def read_logpi(request_output, xy_ids, n_resp):
            pls = request_output.prompt_logprobs
            total = 0.0
            for pos in range(len(xy_ids) - n_resp, len(xy_ids)):
                entry = pls[pos]
                tid = xy_ids[pos]
                if entry and tid in entry:
                    total += float(entry[tid].logprob)
                elif entry:
                    total += float(list(entry.values())[0].logprob)
            return total

        for start in range(0, len(jobs), E):
            chunk = jobs[start:start + E]
            for sign_name, negate in (("plus", False), ("minus", True)):
                ray.get([
                    self.engines[k].collective_rpc.remote(
                        "perturb_from_master",
                        args=(int(job["seed"]), self.sigma, negate),
                    )
                    for k, job in enumerate(chunk)
                ])
                handles = [
                    self.engines[k].generate.remote(
                        [{"prompt_token_ids": job["xy_ids"]}], sp, use_tqdm=False
                    )
                    for k, job in enumerate(chunk)
                ]
                results = ray.get(handles)
                for k, job in enumerate(chunk):
                    val = read_logpi(results[k][0], job["xy_ids"], job["n_resp"])
                    if self.normalize_by_length and job["n_resp"] > 0:
                        val /= job["n_resp"]
                    if sign_name == "plus":
                        lp_plus[start + k] = val
                    else:
                        lp_minus[start + k] = val

        # Leave engines at theta for the update/broadcast that follows.
        ray.get([
            e.collective_rpc.remote("restore_from_master", args=())
            for e in self.engines
        ])
        return lp_plus, lp_minus

    # ------------------------------------------------------------------ #
    # Hybrid scoring: each direction evaluated on a SHARED k-pair mini-batch.
    # ------------------------------------------------------------------ #
    def _score_directions_on_pairs(self, dir_seeds, pairs):
        """For each direction seed, compute delta = mean over `pairs` of
        -adv * (logpi+ - logpi-), with both signs evaluated at theta +/- sigma*eps.
        All k pair prompts go through ONE batched generate call per sign.
        Returns list of deltas aligned to dir_seeds."""
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        E = self.n_vllm_engines
        deltas = [0.0] * len(dir_seeds)

        def read_logpi(request_output, xy_ids, n_resp):
            pls = request_output.prompt_logprobs
            total = 0.0
            for pos in range(len(xy_ids) - n_resp, len(xy_ids)):
                entry = pls[pos]
                tid = xy_ids[pos]
                if entry and tid in entry:
                    total += float(entry[tid].logprob)
                elif entry:
                    total += float(list(entry.values())[0].logprob)
            if self.normalize_by_length and n_resp > 0:
                total /= n_resp
            return total

        batch_prompts = [{"prompt_token_ids": p["xy_ids"]} for p in pairs]

        for start in range(0, len(dir_seeds), E):
            chunk = dir_seeds[start:start + E]
            per_sign = {}
            for sign_name, negate in (("plus", False), ("minus", True)):
                ray.get([
                    self.engines[k].collective_rpc.remote(
                        "perturb_from_master",
                        args=(int(seed), self.sigma, negate),
                    )
                    for k, seed in enumerate(chunk)
                ])
                handles = [
                    self.engines[k].generate.remote(batch_prompts, sp, use_tqdm=False)
                    for k, _ in enumerate(chunk)
                ]
                per_sign[sign_name] = ray.get(handles)

            for k in range(len(chunk)):
                acc = 0.0
                for i, pair in enumerate(pairs):
                    lp_p = read_logpi(per_sign["plus"][k][i], pair["xy_ids"], pair["n_resp"])
                    lp_m = read_logpi(per_sign["minus"][k][i], pair["xy_ids"], pair["n_resp"])
                    acc += -pair["adv"] * (lp_p - lp_m)
                deltas[start + k] = acc / max(1, len(pairs))

        ray.get([
            e.collective_rpc.remote("restore_from_master", args=())
            for e in self.engines
        ])
        return deltas

    # ------------------------------------------------------------------ #
    # DAPO resampling helpers.
    # ------------------------------------------------------------------ #
    def _draw_dapo_batch(self, n, rng):
        """Draw n (templated prompt, target, dataset idx) from the train set.
        Half prioritized from known-active prompts, half uniform exploration."""
        if self._dapo_pool is None:
            self._dapo_pool = self.train_dataloader.dataset
        known = [i for i, hits in self._prompt_stats.items() if hits > 0]
        idxs = []
        k = min(len(known), n // 2)
        if k > 0:
            idxs.extend(int(i) for i in rng.choice(known, size=k, replace=False))
        idxs.extend(int(i) for i in
                    rng.integers(0, len(self._dapo_pool), size=n - len(idxs)))
        batch = [self._dapo_pool[i] for i in idxs]
        prompts, targets = self.train_dataloader.collate_fn(batch)
        return [self.template(p) for p in prompts], list(targets), idxs

    # ------------------------------------------------------------------ #
    # Training step.
    # ------------------------------------------------------------------ #
    def train_step(self, iteration, seeds, input_text, target_text):
        G = self.group_size
        rng = np.random.default_rng((self.global_seed or 42) + iteration)
        # E1.2/E1.3 instrumentation: per-phase wall time + cumulative generations
        # (paper cost accounting). Print/log only; no effect on the update.
        import time as _time
        _t = {"start": _time.perf_counter()}
        if not hasattr(self, "_gens_total"):
            self._gens_total = 0
            self._score_tokens_total = 0

        # tau homotopy: exponential anneal tau0 -> tau_end over the run; swap
        # the grading task to the current tau (module-level fn + partial stays
        # picklable for the timeout pool).
        if self.anneal_reward_fn is not None and self.anneal_tau:
            import functools
            tau0, tau_end = self.anneal_tau
            frac = min(1.0, iteration / max(1, self.num_iterations))
            tau_t = float(tau0 * (tau_end / tau0) ** frac)
            self.task = functools.partial(self.anneal_reward_fn, tau=tau_t)
            self._tau_t = tau_t

        def rollout_and_grade(prompts, targets, round_idx):
            sp = SamplingParams(
                n=G,
                seed=(self.global_seed or 42) + iteration + round_idx * 1000003,
                temperature=self.rollout_temperature,
                top_p=1.0,
                max_tokens=self.max_tokens,
            )
            outs = self._sharded_generate(prompts, sp)
            self._gens_total += len(prompts) * G
            rews = np.array(self._grade_rollouts(outs, targets))
            return list(zip(outs, rews))

        # ---- 1) Rollout like GRPO; optionally DAPO-resample until enough
        #         non-degenerate groups are pooled.
        entries = rollout_and_grade(input_text, target_text, 0)
        rollout_rounds = 1
        if self.dapo_target_groups > 0:
            pooled = [e for e in entries if e[1].std() > 1e-12]
            draw_n = self.dapo_draw or 4 * len(input_text)
            while (len(pooled) < self.dapo_target_groups
                   and rollout_rounds <= self.dapo_max_rounds):
                xp, xt, xi = self._draw_dapo_batch(draw_n, rng)
                extra = rollout_and_grade(xp, xt, rollout_rounds)
                for j, e in enumerate(extra):
                    if e[1].std() > 1e-12:
                        self._prompt_stats[xi[j]] = self._prompt_stats.get(xi[j], 0) + 1
                        pooled.append(e)
                rollout_rounds += 1
            if pooled:
                entries = pooled[: max(self.dapo_target_groups, len(input_text))]

        _t["rollout"] = _time.perf_counter()
        B = len(entries)
        outputs = [e[0] for e in entries]
        rewards = np.stack([e[1] for e in entries])  # [B,G]

        # GRPO-layer normalization: advantage within each prompt's group.
        adv = np.zeros_like(rewards)
        active_group = np.zeros(B, dtype=bool)
        for i in range(B):
            std = rewards[i].std()
            if std > 1e-12:  # DAPO-style: all-same groups contribute nothing
                adv[i] = (rewards[i] - rewards[i].mean()) / (std + 1e-8)
                active_group[i] = True

        # ---- 1b) Success replay (STaR-style, REPLAY_FRAC>0 enables): bank every
        # correct rollout; replayed pairs enter the surrogate as fixed positive-
        # advantage jobs. On sparse-reward tasks a lucky solve normally shapes
        # ONE update and is discarded -- the buffer turns it into a standing
        # teacher. ES has no per-example objective and cannot copy this.
        # Buffer is in-memory only (rebuilds across relay hops).
        import os as _os
        replay_frac = float(_os.environ.get("REPLAY_FRAC", "0"))
        if replay_frac > 0:
            if not hasattr(self, "_success_buf"):
                self._success_buf = {}   # prompt-ids-hash -> list[(xy_ids, n_resp)]
            cap_pp = int(_os.environ.get("REPLAY_CAP_PER_PROMPT", "4"))
            buf_max = int(_os.environ.get("REPLAY_MAX", "512"))
            thresh = float(_os.environ.get("REPLAY_SUCCESS_THRESH", "0.999"))
            for i in range(B):
                x_ids = list(outputs[i].prompt_token_ids)
                key = hash(tuple(x_ids))
                for g in range(G):
                    if rewards[i, g] < thresh:
                        continue
                    y_ids = list(outputs[i].outputs[g].token_ids)
                    if not y_ids:
                        continue
                    slot = self._success_buf.setdefault(key, [])
                    xy = x_ids + y_ids
                    if all(e[0] != xy for e in slot):
                        slot.append((xy, len(y_ids)))
                        if len(slot) > cap_pp:
                            slot.pop(0)
            while sum(len(v) for v in self._success_buf.values()) > buf_max:
                # Drop the oldest entry of the largest slot.
                k_big = max(self._success_buf, key=lambda k: len(self._success_buf[k]))
                self._success_buf[k_big].pop(0)
                if not self._success_buf[k_big]:
                    del self._success_buf[k_big]

        # ---- 2) Build scoring jobs for pairs with nonzero advantage.
        jobs = []
        for i in range(B):
            if not active_group[i]:
                continue
            x_ids = list(outputs[i].prompt_token_ids)
            for g in range(G):
                if abs(adv[i, g]) < 1e-12:
                    continue
                y_ids = list(outputs[i].outputs[g].token_ids)
                if len(y_ids) == 0:
                    continue
                jobs.append({
                    "seed": int(rng.integers(0, 2 ** 30)),
                    "xy_ids": x_ids + y_ids,
                    "n_resp": len(y_ids),
                    "adv": float(adv[i, g]),
                })

        n_onpolicy = len(jobs)
        n_replay = 0
        if replay_frac > 0 and jobs and getattr(self, "_success_buf", None):
            pool = [e for slot in self._success_buf.values() for e in slot]
            n_replay = min(len(pool), max(1, int(replay_frac * len(jobs))))
            radv = float(_os.environ.get("REPLAY_ADV", "1.0"))
            for j in rng.choice(len(pool), size=n_replay, replace=False):
                xy, n_resp = pool[int(j)]
                jobs.append({
                    "seed": int(rng.integers(0, 2 ** 30)),
                    "xy_ids": xy,
                    "n_resp": n_resp,
                    "adv": radv,
                })

        if self.dir_multiplier > 1 and jobs:
            jobs = [dict(j, seed=int(rng.integers(0, 2 ** 30)))
                    for _ in range(self.dir_multiplier) for j in jobs]
        n_act = len(jobs)
        reward_mean = float(rewards.mean())
        frac_active_groups = float(active_group.mean())

        if n_act == 0:
            print(f"[GRZO-S] iter {iteration} | B={B} G={G} | reward mean="
                  f"{reward_mean:.4f} | NO active pairs (all groups degenerate) "
                  f"-- skipping update")
            if self.logging == "wandb":
                self.wandb.log({
                    "global_step": iteration,
                    "train/grzos/reward/mean": reward_mean,
                    "train/grzos/frac_active_groups": 0.0,
                    "train/grzos/active_pairs": 0,
                }, commit=True)
            return

        # ---- 3) Two-point ZO scoring of the surrogate.
        _t["prep"] = _time.perf_counter()
        self._score_tokens_total += 2 * sum(len(j["xy_ids"]) for j in jobs)
        hybrid = (self.pairs_per_direction > 1 or self.directions_per_step > 0)
        if hybrid:
            # Direction-level m>1: N fresh direction seeds, each scored as the
            # MEAN over a shared k-pair mini-batch (ES-style averaging).
            k = min(self.pairs_per_direction, len(jobs)) or len(jobs)
            sel = rng.choice(len(jobs), size=k, replace=False)
            shared_pairs = [jobs[int(i)] for i in sel]
            n_dirs = self.directions_per_step or len(jobs)
            upd_seeds = [int(rng.integers(0, 2 ** 30)) for _ in range(n_dirs)]
            deltas = np.array(
                self._score_directions_on_pairs(upd_seeds, shared_pairs)
            )
        else:
            lp_plus, lp_minus = self._score_jobs_two_point(jobs)
            upd_seeds = [int(job["seed"]) for job in jobs]
            deltas = np.array([
                -job["adv"] * (lp_plus[j] - lp_minus[j])
                for j, job in enumerate(jobs)
            ])

        _t["score"] = _time.perf_counter()
        # GRZO-layer normalization.
        if self.delta_norm == "zscore" and n_act > 1 and deltas.std() > 1e-12:
            # Unit-variance coeffs: fixed-magnitude updates regardless of signal
            # strength (v1-style). Noise-dominated when N per update is small.
            coeffs = (deltas - deltas.mean()) / (deltas.std() + 1e-8)
        else:
            # Raw two-point directional-derivative scale: delta/(2*sigma). Steps
            # shrink automatically when the signal is weak; strong-|delta|
            # directions dominate the update.
            coeffs = deltas / (2.0 * self.sigma)

        # ---- 4) Accumulate directions; update once >= min_directions collected:
        #          theta <- theta - lr * (1/N) sum coeff_j eps_j
        self._pending_seeds.extend(upd_seeds)
        self._pending_coeffs.extend(float(c) for c in coeffs)
        n_pending = len(self._pending_seeds)
        applied = False
        if n_pending >= self.min_directions:
            lr_t = self.lr
            if self.lr_schedule == "cosine":
                import math as _math
                frac = min(1.0, iteration / max(1, self.num_iterations))
                lr_t = self.lr * 0.5 * (1.0 + _math.cos(_math.pi * frac))
            # Anchor-ratchet decay (see GRZOTrainer.eval_step): halved on each
            # restore-from-best; 1.0 unless ANCHOR_RATCHET is active.
            lr_t *= getattr(self, "ratchet_lr_scale", 1.0)
            ray.get(
                self.engines[0].collective_rpc.remote(
                    "update_weights_from_seeds_fp32",
                    args=(self._pending_seeds, self._pending_coeffs,
                          -lr_t, n_pending),
                )
            )
            ray.get([
                e.collective_rpc.remote("broadcast_all_weights", args=(0,))
                for e in self.engines
            ])
            ray.get([
                e.collective_rpc.remote("save_master_weights", args=())
                for e in self.engines
            ])
            torch.cuda.synchronize()
            self._pending_seeds, self._pending_coeffs = [], []
            applied = True

        _t["update"] = _time.perf_counter()
        t_roll = _t["rollout"] - _t["start"]
        t_score = _t["score"] - _t["prep"]
        t_upd = _t["update"] - _t["score"]
        t_step = _t["update"] - _t["start"]

        gen_lens = float(np.mean([
            len(c.token_ids) for out in outputs for c in out.outputs
        ]))
        print(
            f"[GRZO-S] iter {iteration} | B={B} G={G} | reward mean={reward_mean:.4f} "
            f"| active_groups={frac_active_groups:.2f} pairs={n_act}"
            f"{f' (replay {n_replay})' if n_replay else ''} "
            f"| delta mean={deltas.mean():+.2e} std={deltas.std():.2e} "
            f"| len={gen_lens:.0f} rounds={rollout_rounds} "
            f"| {'UPDATED n=' + str(n_pending) if applied else 'pending=' + str(n_pending)}"
        )
        print(
            f"[GRZO-T] iter {iteration} | rollout={t_roll:.1f}s score={t_score:.1f}s "
            f"update={t_upd:.1f}s step={t_step:.1f}s | gens_total={self._gens_total} "
            f"score_tokens_total={self._score_tokens_total}"
        )

        if self.logging == "wandb":
            self.wandb.log({
                "global_step": iteration,
                "train/grzos/reward/mean": reward_mean,
                "train/grzos/reward/var": float(rewards.var()),
                "train/grzos/frac_active_groups": frac_active_groups,
                "train/grzos/active_pairs": int(n_act),
                "train/grzos/replay_pairs": int(n_replay),
                "train/grzos/replay_buffer": int(
                    sum(len(v) for v in getattr(self, "_success_buf", {}).values())),
                "train/grzos/delta/mean": float(deltas.mean()),
                "train/grzos/delta/std": float(deltas.std()),
                "train/grzos/gen_len/mean": gen_lens,
                "train/grzos/sigma": float(self.sigma),
                "train/grzos/lr": float(self.lr),
                "train/grzos/rollout_temperature": self.rollout_temperature,
                "cost/t_rollout_s": t_roll,
                "cost/t_score_s": t_score,
                "cost/t_update_s": t_upd,
                "cost/t_step_s": t_step,
                "cost/gens_total": int(self._gens_total),
                "cost/score_tokens_total": int(self._score_tokens_total),
            }, commit=True)
