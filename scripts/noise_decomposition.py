"""E1.7 — Noise decomposition of the FORGE update estimate (METHOD_REVIEW R1).

On ONE fixed prompt batch, measure how much of a FORGE update is signal vs. noise:
  (b) ZO projection noise : two independent direction draws on the SAME rollouts
      -> cos(u_A, u_B)   ("same-rollout agreement")
  (a) policy-gradient sampling noise : directions drawn on a DIFFERENT rollout of the
      same prompts -> cos(u_A, u_C)   ("cross-rollout agreement")
swept over the number of directions N (each direction scored on one pair, i.e. the
GRZO scheme with N decoupled from #pairs). Both raw (delta/2sigma) and z-scored
coefficients are reported (training uses z-score).

Usage (Lane B, GPUs 3,4):
  python scripts/noise_decomposition.py --use-gpus 3,4 --n-vllm-engines 2 \
      --n-list 96,384,1536 --out /data/liyan/runs/grzo-smoke/noise/noise_1p5b.json
"""
import argparse, json, os, time
import numpy as np
import ray
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from vllm import SamplingParams

from es_at_scale.trainer.grzo_surrogate_trainer import GRZOSurrogateTrainer
from es_at_scale.train_grzo_surrogate import countdown_collate_fn, set_seed
from es_at_scale.reward_function.countdown_grader import countdown_answer_only_reward_fn
from es_at_scale.template_function.apply_template import countdown_template


def build_jobs(outputs, rewards, rng):
    """Replicates train_step step 2: one job per (prompt, rollout) with |adv|>0."""
    B, G = rewards.shape
    jobs = []
    for i in range(B):
        std = rewards[i].std()
        if std <= 1e-12:
            continue
        adv = (rewards[i] - rewards[i].mean()) / (std + 1e-8)
        x_ids = list(outputs[i].prompt_token_ids)
        for g in range(G):
            if abs(adv[g]) < 1e-12:
                continue
            y_ids = list(outputs[i].outputs[g].token_ids)
            if not y_ids:
                continue
            jobs.append({"seed": int(rng.integers(0, 2 ** 30)), "xy_ids": x_ids + y_ids,
                         "n_resp": len(y_ids), "adv": float(adv[g])})
    return jobs


def expand(jobs, n, rng):
    """N directions, each scored on one pair: cycle pairs, fresh seed per direction."""
    order = rng.permutation(len(jobs))
    out = []
    while len(out) < n:
        for k in order:
            if len(out) >= n:
                break
            out.append(dict(jobs[k], seed=int(rng.integers(0, 2 ** 30))))
    return out


def score(trainer, jobs):
    lp_p, lp_m = trainer._score_jobs_two_point(jobs)
    delta = np.array([-j["adv"] * (lp_p[k] - lp_m[k]) for k, j in enumerate(jobs)])
    raw = delta / (2.0 * trainer.sigma)
    z = (delta - delta.mean()) / (delta.std() + 1e-8) if delta.std() > 1e-12 else raw
    return delta, raw, z


def cos_stats(trainer, seeds_a, ca, seeds_b, cb):
    r = ray.get(trainer.engines[0].collective_rpc.remote(
        "direction_stats", args=(seeds_a, [float(c) for c in ca], seeds_b, [float(c) for c in cb])))
    d = r[0] if isinstance(r, list) else r
    return d["dot"] / (np.sqrt(d["na2"] * d["nb2"]) + 1e-30), d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--use-gpus", default="3,4")
    ap.add_argument("--n-vllm-engines", type=int, default=2)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-groups", type=int, default=8)
    ap.add_argument("--dapo-draw", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--n-list", default="96,384,1536")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/liyan/runs/grzo-smoke/noise/noise.json")
    ap.add_argument("--train-dataset", default="datasets/train/countdown")
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ds = next(iter(load_from_disk(args.train_dataset).values()))
    train_dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=countdown_collate_fn, shuffle=True)
    tr = GRZOSurrogateTrainer(
        rollout_temperature=1.0, lr=5e-4, delta_norm="zscore", min_directions=64,
        dapo_target_groups=args.target_groups, dapo_draw=args.dapo_draw,
        grzo_mode="single_point", group_size=args.group_size, start_iteration=0,
        model_name=args.model_name, checkpoint=None, sigma=args.sigma, alpha=5e-4,
        reward_shaping="z-scores", num_iterations=0, max_tokens=args.max_tokens,
        batch_size=args.batch_size, mini_batch_size=args.batch_size,
        reward_function=countdown_answer_only_reward_fn, template_function=countdown_template,
        train_dataloader=train_dl, eval_dataloader_dict={}, eval_freq=10 ** 9,
        n_vllm_engines=args.n_vllm_engines, n_gpu_per_vllm_engine=1, logging="none",
        global_seed=args.seed, use_gpus=args.use_gpus, experiment_name="noise-decomp",
        wandb_project=None, save_best_models=False, reward_function_timeout=10,
        output_directory=os.path.dirname(args.out),
    )
    rng = np.random.default_rng(args.seed)
    G = args.group_size

    def rollout(prompts, targets, seed):
        sp = SamplingParams(n=G, seed=seed, temperature=1.0, top_p=1.0, max_tokens=args.max_tokens)
        outs = tr._sharded_generate(prompts, sp)
        return outs, np.array(tr._grade_rollouts(outs, targets))

    # ---- fixed prompt set with >= target active groups (DAPO-style pooling) ----
    prompts, targets = next(iter(train_dl))
    prompts = [tr.template(p) for p in prompts]; targets = list(targets)
    P, T = [], []
    rounds = 0
    while len(P) < args.target_groups and rounds < 12:
        if rounds > 0:
            prompts, targets, _ = tr._draw_dapo_batch(args.dapo_draw, rng)
        outs, rews = rollout(prompts, targets, 1000 + rounds)
        for i in range(len(prompts)):
            if rews[i].std() > 1e-12 and len(P) < args.target_groups:
                P.append(prompts[i]); T.append(targets[i])
        rounds += 1
        print(f"[NOISE] pooling round {rounds}: active so far {len(P)}/{args.target_groups}")
    print(f"[NOISE] fixed prompt set: {len(P)} prompts")

    # ---- two independent rollouts of the same prompts ----
    outs0, rews0 = rollout(P, T, 2001)
    outs1, rews1 = rollout(P, T, 2002)
    jobs0 = build_jobs(outs0, rews0, rng)
    jobs1 = build_jobs(outs1, rews1, rng)
    print(f"[NOISE] pairs: R0={len(jobs0)} R1={len(jobs1)} | reward mean R0={rews0.mean():.3f} R1={rews1.mean():.3f}")
    results = {"model": args.model_name, "sigma": args.sigma, "G": G, "n_prompts": len(P),
               "pairs_R0": len(jobs0), "pairs_R1": len(jobs1), "rows": []}

    for N in [int(x) for x in args.n_list.split(",")]:
        t0 = time.time()
        jA = expand(jobs0, N, rng); jB = expand(jobs0, N, rng); jC = expand(jobs1, N, rng)
        dA, rA, zA = score(tr, jA); dB, rB, zB = score(tr, jB); dC, rC, zC = score(tr, jC)
        t_score = time.time() - t0
        sA = [j["seed"] for j in jA]; sB = [j["seed"] for j in jB]; sC = [j["seed"] for j in jC]
        cos_same_raw, _ = cos_stats(tr, sA, rA, sB, rB)
        cos_same_z, _ = cos_stats(tr, sA, zA, sB, zB)
        cos_cross_raw, _ = cos_stats(tr, sA, rA, sC, rC)
        cos_cross_z, _ = cos_stats(tr, sA, zA, sC, zC)
        row = {"N": N, "cos_same_rollout_raw": cos_same_raw, "cos_same_rollout_z": cos_same_z,
               "cos_cross_rollout_raw": cos_cross_raw, "cos_cross_rollout_z": cos_cross_z,
               "delta_mean": float(dA.mean()), "delta_std": float(dA.std()),
               "delta_abs_mean": float(np.abs(dA).mean()),
               "t_score_3sets_s": t_score, "t_total_s": time.time() - t0}
        results["rows"].append(row)
        print(f"[NOISE] N={N:5d} | same-rollout cos raw={cos_same_raw:+.4f} z={cos_same_z:+.4f} "
              f"| cross-rollout cos raw={cos_cross_raw:+.4f} z={cos_cross_z:+.4f} "
              f"| delta std={dA.std():.2e} | {time.time()-t0:.0f}s")
        json.dump(results, open(args.out, "w"), indent=2)

    print(f"[NOISE] wrote {args.out}")
    tr.cleanup()


if __name__ == "__main__":
    main()
