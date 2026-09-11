"""Objective noise tolerance: how far can theta move in a random direction before accuracy
degrades?  F(theta + h u) for u ~ N(0, I) (seed-regenerated inside the vLLM engine, fp32 from
the bf16 master), h in a grid, several seeds; F = greedy accuracy on the 200 train prompts
(ES fitness: shaped reward and answer_acc) and optionally on an eval subset.

Gives the random-walk displacement budget r* (per-parameter RMS h and L2 norm h*sqrt(d)) at
which accuracy drops by a given fraction. With the per-step noise norm of a FORGE update,
eta * rms|g_j| * sqrt(d/N), this bounds how many steps a run can take at a given (eta, N).

  CUDA_VISIBLE_DEVICES=1 python scripts/noise_tolerance.py --use-gpus 1 --h-list 1e-4,2.5e-4,5e-4,1e-3,2e-3,4e-3,8e-3 --seeds 3
"""
import argparse, json, os, sys
import numpy as np, ray
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torch.utils.data import DataLoader
from datasets import load_from_disk
from vllm import SamplingParams
from es_at_scale.trainer.es_trainer import EvolutionStrategiesTrainer
from es_at_scale.train_grzo_surrogate import countdown_collate_fn
from es_at_scale.reward_function.countdown_grader import countdown_reward_fn
from es_at_scale.template_function.apply_template import countdown_template


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--use-gpus", default="1")
    ap.add_argument("--h-list", default="1e-4,2.5e-4,5e-4,1e-3,2e-3,4e-3,8e-3")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--eval-n", type=int, default=500, help="also evaluate on the first n eval prompts (0 = skip)")
    ap.add_argument("--out", default="/data/liyan/runs/grzo-smoke/noise/noise_tolerance_1p5b.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    train = next(iter(load_from_disk("datasets/train/countdown").values()))
    dl = DataLoader(train, batch_size=200, collate_fn=countdown_collate_fn, shuffle=False)
    tr = EvolutionStrategiesTrainer(
        model_name=args.model_name, checkpoint=args.checkpoint, sigma=1e-3, alpha=5e-4, population_size=1,
        reward_shaping="z-scores", num_iterations=0, max_tokens=args.max_tokens, batch_size=200, mini_batch_size=200,
        reward_function=countdown_reward_fn, template_function=countdown_template, train_dataloader=dl,
        eval_dataloader_dict={}, eval_freq=10 ** 9, n_vllm_engines=1, n_gpu_per_vllm_engine=1, logging="none",
        use_gpus=args.use_gpus, global_seed=42, output_directory=os.path.dirname(args.out),
        experiment_name="noise-tol", save_best_models=False, reward_function_timeout=10)
    ray.get(tr.engines[0].collective_rpc.remote("save_master_weights", args=()))
    prompts, targets = next(iter(dl)); prompts = [tr.template(p) for p in prompts]; targets = list(targets)
    sets = {"train200": (prompts, targets)}
    if args.eval_n > 0:
        ev = load_from_disk("datasets/evaluation_suite/countdown")["countdown_eval"].select(range(args.eval_n))
        ep, et = countdown_collate_fn([ev[i] for i in range(len(ev))])
        sets[f"eval{args.eval_n}"] = ([tr.template(p) for p in ep], list(et))
    sp = SamplingParams(n=1, seed=42, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    def measure():
        out = {}
        for name, (P, T) in sets.items():
            o = ray.get(tr.engines[0].generate.remote(P, sp, use_tqdm=False))
            m = tr._postprocess_outputs(o, T)
            out[name] = {"shaped": m["avg_reward"], "acc": float(np.mean([r > 0.999 for r in m["rewards"]]))}
        return out

    base = measure(); print(f"[TOL] h=0: {base}")
    res = {"model": args.model_name, "ckpt": args.checkpoint, "base": base, "rows": []}
    for h in [float(x) for x in args.h_list.split(",")]:
        for s in range(args.seeds):
            seed = 777000 + s
            ray.get(tr.engines[0].collective_rpc.remote("perturb_from_master", args=(seed, h, False)))
            m = measure()
            ray.get(tr.engines[0].collective_rpc.remote("restore_from_master", args=()))
            res["rows"].append({"h": h, "seed": seed, **{k: v for k, v in m.items()}})
            print(f"[TOL] h={h:g} seed={s}: " + " | ".join(f"{k}: acc {v['acc']:.4f} shaped {v['shaped']:.4f}" for k, v in m.items()))
        json.dump(res, open(args.out, "w"), indent=2)
    print(f"[TOL] wrote {args.out}")
    tr.cleanup()


if __name__ == "__main__":
    main()
