"""GRZO-RLVR trainer: zeroth-order optimization as an optimizer for RLVR.

This subclasses ``EvolutionStrategiesTrainer`` and changes ONLY the update rule
(and the evaluation that feeds it). The engine launch, vLLM plumbing, seed-based
in-place perturb/restore, the fp32 ``update_weights_from_seeds`` accumulator, and
the reward postprocessing are all reused unchanged.

GRZO vs ES
----------
ES (parent) evaluates each of ``population_size`` perturbations on the WHOLE batch
of prompts, averages the reward per perturbation, z-scores across perturbations,
and updates ``theta += (alpha / P) * sum_n z_n * eps_n``.

GRZO ties each perturbation to a single prompt and normalizes group-relative:

  For a step with B prompts, each prompt p gets G independent perturbations.
  For each (prompt p, perturbation g):
    1. draw an independent seed s_{p,g}
    2. perturb the FULL model in place: theta <- theta + sigma * eps(s_{p,g})
    3. GREEDY-decode a response for prompt p (all variation comes from the
       parameter perturbation, not token sampling -> on-policy in perturbation
       space, trajectory-level exploration for free)
    4. reward R_{p,g} = verify(prompt_p, response)
    5. restore: theta <- theta - sigma * eps(s_{p,g})
  Group-relative advantage, normalized WITHIN each prompt's group of G (the
  baseline controls for prompt difficulty, the ZO analog of GRPO's per-prompt
  group):
    A_{p,g} = (R_{p,g} - mean_g R_{p,.}) / (std_g R_{p,.} + 1e-8)
  Update over all N = B * G pairs:
    theta <- theta + (alpha / N) * sum_{p,g} A_{p,g} * eps(s_{p,g})

The last line is exactly the parent's ``update_weights_from_seeds`` with per-pair
seeds and coefficients -- the noise regeneration recipe is identical, so the eps
used in evaluation and in the update are guaranteed to match.

Two modes (``grzo_mode``)
-------------------------
- ``single_point``: one evaluation per (prompt, perturbation), group-relative
  baseline. Default, the GRZO-native form.
- ``two_point``: antithetic -- evaluate each pair at both theta+sigma*eps and
  theta-sigma*eps and use the difference (R^+ - R^-) as the raw signal before
  group normalization. Lower variance, 2x forward cost.

Relay support
-------------
Eval is sharded across all engines (``_sharded_generate``) so the 2000-sample
Countdown eval uses 4 GPUs instead of 1. A ``latest/`` checkpoint + progress.json
is written at every eval so an interactive-relay run can resume across the 4h
wall-time boundary (see ``fit`` + ``start_iteration``).

No critic, no reference model, no KL, no backprop, no activation storage, no
importance sampling. sigma is absorbed into ``alpha`` (the learning rate), exactly
as in the parent ES implementation.
"""

import json
import os
import time

import numpy as np
import ray
import torch
from vllm import SamplingParams

from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer


class GRZOTrainer(EvolutionStrategiesTrainer):
    def __init__(self, *args, grzo_mode="single_point", group_size=8,
                 start_iteration=0, **kwargs):
        assert grzo_mode in ("single_point", "two_point"), grzo_mode
        # The parent tracks ``population_size``; for GRZO that role is played by
        # the per-prompt group size G, so keep them in sync for reused bookkeeping.
        kwargs["population_size"] = group_size
        super().__init__(*args, **kwargs)
        self.grzo_mode = grzo_mode
        self.group_size = int(group_size)
        self.start_iteration = int(start_iteration)

    # ------------------------------------------------------------------ #
    # Evaluation: one prompt per (prompt, perturbation) job.
    # ------------------------------------------------------------------ #
    def _evaluate_grzo_jobs(self, jobs, sampling_params):
        """Evaluate a list of jobs, each ``dict(seed, prompt, target, sign)``.

        Jobs are scheduled ``n_vllm_engines`` at a time. Within a chunk, engine i
        perturbs with its job's seed (negated when ``sign < 0`` for the two_point
        minus branch), greedy-decodes ONLY that job's single prompt, then restores
        by applying the inverse perturbation. Returns a reward list aligned to
        ``jobs``.
        """
        rewards = [0.0] * len(jobs)
        E = self.n_vllm_engines

        for start in range(0, len(jobs), E):
            chunk = jobs[start:start + E]

            # 1) Perturb: theta +/- sigma*eps on each engine in the chunk.
            ray.get([
                self.engines[i].collective_rpc.remote(
                    "perturb_self_weights",
                    args=(int(job["seed"]), self.sigma, bool(job["sign"] < 0)),
                )
                for i, job in enumerate(chunk)
            ])

            # 2) Greedy-decode the single paired prompt on each engine.
            handles = [
                self.engines[i].generate.remote(
                    [job["prompt"]], sampling_params, use_tqdm=False
                )
                for i, job in enumerate(chunk)
            ]
            outputs_per_engine = ray.get(handles)

            # 3) Restore: apply the inverse perturbation (opposite sign).
            ray.get([
                self.engines[i].collective_rpc.remote(
                    "perturb_self_weights",
                    args=(int(job["seed"]), self.sigma, bool(job["sign"] > 0)),
                )
                for i, job in enumerate(chunk)
            ])

            # 4) Score (single prompt -> avg_reward is that prompt's reward).
            for i, job in enumerate(chunk):
                metrics = self._postprocess_outputs(
                    outputs_per_engine[i], [job["target"]]
                )
                rewards[start + i] = float(metrics["avg_reward"])

        return rewards

    # ------------------------------------------------------------------ #
    # Training step: per-prompt group-relative ZO update.
    # ------------------------------------------------------------------ #
    def train_step(self, iteration, seeds, input_text, target_text):
        # ``seeds`` from the parent loop is ignored: GRZO draws its own
        # independent seed per (prompt, perturbation) pair.
        B = len(input_text)
        G = self.group_size
        two_point = (self.grzo_mode == "two_point")

        rng = np.random.default_rng((self.global_seed or 42) + iteration)
        seed_grid = rng.integers(0, 2 ** 30, size=(B, G), dtype=np.int64)

        sampling_params = SamplingParams(
            n=1,
            seed=(self.global_seed or 42) + iteration,
            temperature=self.train_temperature,  # 0.0 -> greedy
            top_p=self.train_top_p,
            max_tokens=self.max_tokens,
        )

        # Build the flat job list. For two_point each pair yields a + and - job.
        jobs = []
        for b in range(B):
            for g in range(G):
                s = int(seed_grid[b, g])
                jobs.append({"b": b, "g": g, "sign": +1, "seed": s,
                             "prompt": input_text[b], "target": target_text[b]})
                if two_point:
                    jobs.append({"b": b, "g": g, "sign": -1, "seed": s,
                                 "prompt": input_text[b], "target": target_text[b]})

        rewards = self._evaluate_grzo_jobs(jobs, sampling_params)

        Rp = np.zeros((B, G), dtype=np.float64)
        Rm = np.zeros((B, G), dtype=np.float64)
        for job, r in zip(jobs, rewards):
            if job["sign"] > 0:
                Rp[job["b"], job["g"]] = r
            else:
                Rm[job["b"], job["g"]] = r

        raw = (Rp - Rm) if two_point else Rp

        # Group-relative normalization WITHIN each prompt's group of G.
        adv = np.zeros((B, G), dtype=np.float64)
        for b in range(B):
            row = raw[b]
            adv[b] = (row - row.mean()) / (row.std() + 1e-8)

        all_seeds = [int(seed_grid[b, g]) for b in range(B) for g in range(G)]
        all_coeffs = [float(adv[b, g]) for b in range(B) for g in range(G)]
        N = B * G

        # ---- metrics (spec: reward mean + per-step reward/advantage variance) ----
        reward_mean = float(Rp.mean())
        reward_var = float(Rp.var())
        raw_var = float(raw.var())
        adv_var = float(adv.var())
        # Fraction of prompts whose group is degenerate (all-equal reward -> zero
        # signal): a high value means the group baseline gives no gradient.
        dead_groups = float(np.mean([raw[b].std() < 1e-12 for b in range(B)]))

        print(
            f"[GRZO/{self.grzo_mode}] iter {iteration} | B={B} G={G} "
            f"| reward mean={reward_mean:.4f} var={reward_var:.4f} "
            f"| raw_var={raw_var:.4f} adv_var={adv_var:.4f} dead_groups={dead_groups:.2f}"
        )

        if self.logging == "wandb":
            self.wandb.log({
                "global_step": iteration,
                "train/grzo/mode_two_point": int(two_point),
                "train/grzo/group_size": int(G),
                "train/grzo/prompts_per_step": int(B),
                "train/grzo/reward/mean": reward_mean,
                "train/grzo/reward/var": reward_var,
                "train/grzo/reward/plus_mean": float(Rp.mean()),
                "train/grzo/reward/minus_mean": float(Rm.mean()) if two_point else 0.0,
                "train/grzo/raw_signal/var": raw_var,
                "train/grzo/advantage/var": adv_var,
                "train/grzo/dead_group_frac": dead_groups,
                "train/es/sigma": float(self.sigma),
                "train/es/alpha": float(self.alpha),
            }, commit=True)

        # ---- update: theta += (alpha / N) * sum A_i eps(seed_i), then broadcast ----
        ray.get(
            self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(all_seeds, all_coeffs, self.alpha, N),
            )
        )
        ray.get([
            e.collective_rpc.remote("broadcast_all_weights", args=(0,))
            for e in self.engines
        ])
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ #
    # Sharded generation: split prompts across all engines (eval only; the
    # engines all hold identical weights here, and decoding is greedy).
    # ------------------------------------------------------------------ #
    def _sharded_generate(self, prompts, sampling_params):
        n = len(prompts)
        E = self.n_vllm_engines
        bounds = [(k * n) // E for k in range(E + 1)]
        handles, spans = [], []
        for k in range(E):
            s, e = bounds[k], bounds[k + 1]
            if s >= e:
                continue
            handles.append(
                self.engines[k].generate.remote(prompts[s:e], sampling_params, use_tqdm=False)
            )
            spans.append((s, e))
        results = ray.get(handles)
        outputs = [None] * n
        for (s, e), res in zip(spans, results):
            for j, o in enumerate(res):
                outputs[s + j] = o
        return outputs

    # ------------------------------------------------------------------ #
    # Eval: sharded generation; log shaped pass@1 AND pure answer-accuracy
    # (the honest task metric).
    # ------------------------------------------------------------------ #
    def eval_step(self, iteration):
        to_log = {"eval-iteration": iteration}
        mean_pass1, mean_answer_acc = [], []

        for name, eval_loader in self.eval_dataloader_dict.items():
            # Gather all templated prompts/targets for the dataset, then generate
            # once across all engines.
            all_prompts, all_targets = [], []
            for input_text, target_text in eval_loader:
                all_prompts.extend(self.template(i) for i in input_text)
                all_targets.extend(list(target_text))

            sampling_params = SamplingParams(
                n=1,
                seed=(self.global_seed or 42) + iteration,
                temperature=0.0,
                top_p=1.0,
                max_tokens=self.max_tokens,
            )
            outputs = self._sharded_generate(all_prompts, sampling_params)
            metrics = self._postprocess_outputs(outputs, all_targets, eval=True)

            count = len(metrics["rewards"])
            pass1 = float(metrics["avg_reward"]) if count else 0.0
            # Pure answer accuracy. Countdown's fmt dict carries "answer_reward"
            # (reward itself is shaped); graders whose reward is already binary
            # 0/1 (e.g. math boxed_reward_fn) have no such key -- fall back to
            # the raw reward.
            answer_acc = (
                float(np.mean([
                    r["format"].get("answer_reward", r["reward"])
                    if isinstance(r.get("format"), dict) else r["reward"]
                    for r in metrics["results"]
                ]))
                if metrics["results"] else 0.0
            )

            # Spearman calibration gate (TCAD-style): when the training reward
            # is a shaped/margin surrogate (reward != binary answer), verify it
            # still RANKS samples like the true 0/1 signal. A collapse of this
            # correlation is the reward-hacking alarm.
            spearman_gate = None
            if metrics["results"]:
                rr = np.array([float(r["reward"]) for r in metrics["results"]])
                bb = np.array([
                    float(r["format"].get("answer_reward", r["reward"]))
                    if isinstance(r.get("format"), dict) else float(r["reward"])
                    for r in metrics["results"]
                ])
                if not np.allclose(rr, bb) and rr.std() > 0 and bb.std() > 0:
                    try:
                        from scipy.stats import spearmanr
                        spearman_gate = float(spearmanr(rr, bb).statistic)
                    except Exception:
                        spearman_gate = None
            mean_pass1.append(pass1)
            mean_answer_acc.append(answer_acc)

            gate_str = (f" -- spearman_gate: {spearman_gate:.3f}"
                        if spearman_gate is not None else "")
            print(f"{name} -- eval pass@1 (shaped): {pass1:.4f} -- "
                  f"answer_acc (0/1): {answer_acc:.4f} --{gate_str}")

            to_log.update({
                "global_step": iteration,
                f"eval/{name}/pass@1/mean": pass1,
                f"eval/{name}/answer_acc/mean": answer_acc,
            })
            if spearman_gate is not None:
                to_log[f"eval/{name}/spearman_gate"] = spearman_gate
            fn = f"{self.logging_dir}/eval-output/model_eval_task{name}_iteration{iteration + 1}.json"
            print(f"saving model outputs at {fn}")
            json.dump(metrics["results"], open(fn, "w"), indent=4)

        avg_pass1 = float(np.mean(mean_pass1)) if mean_pass1 else 0.0
        avg_answer_acc = float(np.mean(mean_answer_acc)) if mean_answer_acc else 0.0
        to_log.update({
            "eval/avgpass@1/mean": avg_pass1,
            "eval/avg_answer_acc/mean": avg_answer_acc,
        })

        if self.logging == "wandb":
            self.wandb.log(to_log, commit=True)

        # Best-model saving keyed on the true task metric (answer accuracy).
        if self.save_best_models and avg_answer_acc > self.best_avg:
            self.best_avg = avg_answer_acc
            model_path = f"{self.logging_dir}/checkpoints/{self.experiment_name}-acc{avg_answer_acc}"
            os.makedirs(model_path, exist_ok=True)
            ray.get(
                self.engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk",
                    args=(f"{model_path}/pytorch_model.pth",),
                )
            )
            # Stable path for the anchor ratchet to restore from.
            bd = f"{self.logging_dir}/best"
            os.makedirs(bd, exist_ok=True)
            ray.get(
                self.engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk",
                    args=(f"{bd}/pytorch_model.pth",),
                )
            )
            self._ratchet_bad_evals = 0

        # Anchor ratchet (ANCHOR_RATCHET=1): every observed FORGE run climbs,
        # peaks, then degrades (constant-magnitude z-scored steps keep walking
        # once the signal-to-noise collapses near a good region). When eval sits
        # >RATCHET_DROP below the best for RATCHET_PATIENCE consecutive evals,
        # restore the best weights and halve the update scale -- turning
        # climb-then-decay into a ratchet.
        # RATCHET_WARMUP: don't arm before this iteration -- a lucky early eval
        # otherwise becomes the anchor and normal climb-phase variance triggers
        # repeated restores that crush lr at the start (observed on
        # forge2-math-hold: lr 0.25x by iter 174).
        if (os.environ.get("ANCHOR_RATCHET", "0") == "1"
                and self.save_best_models and self.best_avg > 0
                and iteration >= int(os.environ.get("RATCHET_WARMUP", "0"))):
            drop = float(os.environ.get("RATCHET_DROP", "0.02"))
            patience = int(os.environ.get("RATCHET_PATIENCE", "2"))
            if avg_answer_acc < self.best_avg - drop:
                self._ratchet_bad_evals = getattr(self, "_ratchet_bad_evals", 0) + 1
            else:
                self._ratchet_bad_evals = 0
            if self._ratchet_bad_evals >= patience:
                best_f = f"{self.logging_dir}/best/pytorch_model.pth"
                if os.path.exists(best_f) or os.path.exists(f"{best_f}.rank0"):
                    self.ratchet_lr_scale = max(
                        0.125, getattr(self, "ratchet_lr_scale", 1.0) * 0.5)
                    ray.get([
                        e.collective_rpc.remote(
                            "load_weights_from_disk", args=(best_f,))
                        for e in self.engines
                    ])
                    ray.get([
                        e.collective_rpc.remote("save_master_weights", args=())
                        for e in self.engines
                    ])
                    if hasattr(self, "_pending_seeds"):
                        self._pending_seeds, self._pending_coeffs = [], []
                    self._ratchet_bad_evals = 0
                    print(f"[RATCHET] iter {iteration}: eval {avg_answer_acc:.4f} "
                          f"< best {self.best_avg:.4f} - {drop}; restored best, "
                          f"lr_scale -> {self.ratchet_lr_scale}")

    # ------------------------------------------------------------------ #
    # Relay checkpointing: write latest/ weights + progress so a fresh
    # interactive alloc can resume across the wall-time boundary.
    # ------------------------------------------------------------------ #
    def _save_latest(self, iteration):
        self._last_latest_save_t = time.time()
        d = f"{self.logging_dir}/latest"
        os.makedirs(d, exist_ok=True)
        ray.get(
            self.engines[0].collective_rpc.remote(
                "save_self_weights_to_disk", args=(f"{d}/pytorch_model.pth",),
            )
        )
        # Ratchet state must survive relay hops: without it, best_avg resets to
        # 0 each 4h hop and the ratchet silently re-baselines on post-hop evals
        # (observed on probe-cd-replay: late sag with no second trigger).
        json.dump({
            "iteration": int(iteration),
            "best_avg": float(getattr(self, "best_avg", 0.0)),
            "ratchet_lr_scale": float(getattr(self, "ratchet_lr_scale", 1.0)),
        }, open(f"{d}/progress.json", "w"))
        print(f"[CKPT] latest saved at iteration {iteration} -> {d}")

    # ------------------------------------------------------------------ #
    # Training loop with resume support (overrides parent fit).
    # ------------------------------------------------------------------ #
    def fit(self):
        start = int(getattr(self, "start_iteration", 0))
        # How often to write latest/ between evals (env-tunable; picked up
        # automatically at the next relay hop since each hop re-execs python).
        self._save_latest_freq = int(os.environ.get("SAVE_LATEST_FREQ", "10"))
        self._last_latest_save_t = 0.0

        # Fresh run: baseline eval before any training. On resume, the base model
        # was already evaluated in the earlier alloc, so skip it.
        if start == 0:
            self.eval_step(iteration=0)

        if self.num_iterations == 0:
            self.cleanup()
            if self.logging == "wandb":
                try:
                    self.wandb.finish()
                except Exception:
                    pass
            print("-- Evaluation completed! --")
            return

        iteration = start
        epoch, done = 0, False
        if start > 0:
            print(f"-- Resuming from iteration {start} --")
            # Restore ratchet state across relay hops (see _save_latest).
            try:
                prog = json.load(open(f"{self.logging_dir}/latest/progress.json"))
                if float(prog.get("best_avg", 0.0)) > float(getattr(self, "best_avg", 0.0)):
                    self.best_avg = float(prog["best_avg"])
                self.ratchet_lr_scale = float(prog.get("ratchet_lr_scale", 1.0))
                print(f"[RATCHET] restored state: best_avg={self.best_avg:.4f} "
                      f"lr_scale={self.ratchet_lr_scale}")
            except Exception:
                pass

        while not done:
            for input_text, target_text in self.train_dataloader:
                input_text = [self.template(i) for i in input_text]
                print(f"\n\n=== Epoch {epoch + 1}; Iteration {iteration + 1} ===")
                t0 = time.time()

                self.train_step(iteration, None, input_text, target_text)

                if ((iteration + 1) % self.eval_freq) == 0 and (iteration > 0):
                    self.eval_step(iteration=iteration)
                    self._save_latest(iteration)
                elif (((iteration + 1) % self._save_latest_freq) == 0
                      and time.time() - self._last_latest_save_t >= 300):
                    # Decoupled from eval: relay hops lose at most
                    # _save_latest_freq-1 iterations of work, not eval_freq-1.
                    # Time-throttled to >=300s apart: on fast runs (~7s/iter)
                    # un-throttled 10-iter saves wrote 3GB every 74s and the
                    # page-cache buildup tripped Ray's 95% host-RAM kill
                    # (observed 2026-07-19 on forge512-1p5b-mtb).
                    self._save_latest(iteration)

                print(f"=== Iteration {iteration + 1} finished in {time.time() - t0:.2f}s ===\n")

                iteration += 1
                if iteration > self.num_iterations:
                    done = True
                    break
            epoch += 1
            if done:
                break

        # Final weights: both a stable latest/ (for a would-be further resume)
        # and the conventional final checkpoint dir.
        self._save_latest(iteration - 1)
        final_model_path = f"{self.logging_dir}/checkpoint-grzo_iteration_{self.num_iterations}"
        os.makedirs(final_model_path, exist_ok=True)
        ray.get(
            self.engines[0].collective_rpc.remote(
                "save_self_weights_to_disk",
                args=(f"{final_model_path}/pytorch_model.pth",),
            )
        )
        print(f"Final model weights saved to {final_model_path}.")

        self.cleanup()
        if self.logging == "wandb":
            try:
                self.wandb.finish()
            except Exception:
                pass
        print("-- Training completed! --")
