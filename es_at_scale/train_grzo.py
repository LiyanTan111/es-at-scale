# GRZO-RLVR entry point: zeroth-order optimization as an optimizer for RLVR.
#
# Mirrors train.py but wires the GRZOTrainer (per-example / per-prompt-group ZO
# update) instead of the ES population-weighted update. Countdown task only for
# the prototype go/no-go. Reuses the repo's on-disk 200-train / 2000-eval split
# and countdown grader unchanged.
#
# Interpretation of the shared knobs for GRZO:
#   --batch-size   -> B, number of distinct prompts drawn per training step
#   --group-size   -> G, number of independent perturbations per prompt (the
#                     group over which the advantage baseline is computed)
# Each step performs B*G (single_point) or 2*B*G (two_point) single-prompt
# greedy rollouts, then one seed-based ZO update over all B*G pairs.

import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from es_at_scale.trainer.grzo_trainer import GRZOTrainer


def countdown_collate_fn(batch):
    prompts = [item["context"] for item in batch]
    targets = [{"numbers": item["numbers"], "target": item["target"]} for item in batch]
    return prompts, targets


def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)


def main():
    parser = argparse.ArgumentParser(description="GRZO-RLVR training (Countdown).")

    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=-1,
                        help="Learning rate. Defaults to sigma/2 when < 0 (sigma is "
                             "absorbed into alpha, as in ES-at-Scale).")
    parser.add_argument("--grzo-mode", type=str, default="single_point",
                        choices=["single_point", "two_point"])
    parser.add_argument("--reward-mode", type=str, default="shaped",
                        choices=["shaped", "binary"],
                        help="shaped: 0.1*format + answer (repo default). "
                             "binary: pure 0/1 answer correctness (no format bonus).")
    parser.add_argument("--group-size", type=int, default=8,
                        help="G: independent perturbations per prompt (baseline group).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="B: distinct prompts drawn per training step.")
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--train-dataset", type=str, default="datasets/train/countdown")
    parser.add_argument("--eval-dataset", type=str,
                        default="datasets/evaluation_suite/countdown/")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--logging", type=str, default="wandb")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpus", type=str, default="0,1,2,3")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument("--output-directory", type=str, default="./experiments/")
    parser.add_argument("--save-best-models", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="grzo-rlvr")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from {output_dir}/{experiment_name}/latest/ if "
                             "present (for interactive-relay runs across the 4h wall). "
                             "Requires a fixed --experiment-name.")

    args = parser.parse_args()
    print(args)

    from es_at_scale.reward_function.countdown_grader import (
        countdown_reward_fn, countdown_answer_only_reward_fn,
    )
    from es_at_scale.template_function.apply_template import countdown_template
    template_function = countdown_template
    collate = countdown_collate_fn
    # Module-level fns (picklable for the reward-timeout multiprocessing pool).
    reward_function = (countdown_answer_only_reward_fn
                       if args.reward_mode == "binary" else countdown_reward_fn)

    alpha = args.alpha
    if alpha == -1.0:
        alpha = args.sigma / 2

    set_seed(args.seed)

    # Train loader: B prompts per step (existing 200-sample countdown split).
    for _, dataset in load_from_disk(args.train_dataset).items():
        train_dataloader = DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate, shuffle=True,
        )

    # Eval loader: existing 2000-sample split, size-weighted pass@1.
    eval_dataloader_dict = {}
    if args.eval_dataset and os.path.exists(args.eval_dataset):
        for task_name, dataset in load_from_disk(args.eval_dataset).items():
            eval_dataloader_dict[task_name] = DataLoader(
                dataset, batch_size=args.mini_batch_size, shuffle=False, collate_fn=collate,
            )

    experiment_name = args.experiment_name or (
        f"grzo-{args.grzo_mode}-{args.reward_mode}"
        f"-sigma{args.sigma}-alpha{alpha}-G{args.group_size}-B{args.batch_size}"
        f"-tkn{args.max_tokens}-model{str(args.model_name).split('/')[-1]}"
        f"-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"-- Running: {experiment_name} --")

    # Resume support for interactive relays: pick up latest/ weights + iteration.
    start_iteration = 0
    checkpoint = args.checkpoint
    if args.resume:
        latest_dir = f"{args.output_directory}/{experiment_name}/latest"
        prog_f = f"{latest_dir}/progress.json"
        ckpt_f = f"{latest_dir}/pytorch_model.pth"
        if os.path.exists(prog_f) and os.path.exists(ckpt_f):
            import json as _json
            last_done = int(_json.load(open(prog_f))["iteration"])
            start_iteration = last_done + 1
            checkpoint = ckpt_f
            print(f"[RESUME] found latest at iter {last_done}; "
                  f"resuming from iter {start_iteration}, ckpt={ckpt_f}")
        else:
            print(f"[RESUME] no latest checkpoint under {latest_dir}; starting fresh")

    trainer = GRZOTrainer(
        grzo_mode=args.grzo_mode,
        group_size=args.group_size,
        start_iteration=start_iteration,
        model_name=args.model_name,
        checkpoint=checkpoint,
        sigma=args.sigma,
        alpha=alpha,
        reward_shaping="z-scores",
        num_iterations=args.n_iterations,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        reward_function=reward_function,
        template_function=template_function,
        train_dataloader=train_dataloader,
        eval_dataloader_dict=eval_dataloader_dict,
        eval_freq=args.eval_freq,
        n_vllm_engines=args.n_vllm_engines,
        n_gpu_per_vllm_engine=args.n_gpu_per_vllm_engine,
        logging=args.logging,
        global_seed=args.seed,
        use_gpus=args.use_gpus,
        experiment_name=experiment_name,
        wandb_project=args.wandb_project,
        save_best_models=args.save_best_models,
        reward_function_timeout=args.reward_function_timeout,
        output_directory=args.output_directory,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
