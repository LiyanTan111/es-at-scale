# ES-original positive control entry (Countdown, shaped reward — the paper's
# native recipe), with relay resume. See es_relay_trainer.py.

import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from es_at_scale.trainer.es_relay_trainer import ESRelayTrainer


def countdown_collate_fn(batch):
    prompts = [item["context"] for item in batch]
    targets = [{"numbers": item["numbers"], "target": item["target"]} for item in batch]
    return prompts, targets


def math_collate_fn(batch):
    prompts = [item["problem"] for item in batch]
    targets = [item["answer"] for item in batch]
    return prompts, targets


def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main():
    parser = argparse.ArgumentParser(description="ES-original positive control (relay).")
    parser.add_argument("--task", type=str, default="countdown",
                        choices=["countdown", "math"])
    parser.add_argument("--eval-subsets", type=str, default="",
                        help="Comma-separated eval subset names to keep (empty = all).")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=-1)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--mini-batch-size", type=int, default=200)
    parser.add_argument("--n-iterations", type=int, default=150)
    parser.add_argument("--eval-freq", type=int, default=10)
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
    parser.add_argument("--resume", action="store_true")

    args = parser.parse_args()
    print(args)

    if args.task == "math":
        # Repo's native math recipe: boxed_reward_fn is already binary 0/1.
        from es_at_scale.reward_function.math_grader import boxed_reward_fn as reward_function
        from es_at_scale.template_function.apply_template import qwen_math_template as template_function
        collate = math_collate_fn
    else:
        # Countdown native recipe: shaped 0.1*format + answer.
        from es_at_scale.reward_function.countdown_grader import countdown_reward_fn as reward_function
        from es_at_scale.template_function.apply_template import countdown_template as template_function
        collate = countdown_collate_fn

    alpha = args.alpha
    if alpha == -1.0:
        alpha = args.sigma / 2

    set_seed(args.seed)

    for _, dataset in load_from_disk(args.train_dataset).items():
        train_dataloader = DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate,
            shuffle=True,
        )

    keep_subsets = {s.strip() for s in args.eval_subsets.split(",") if s.strip()}
    eval_dataloader_dict = {}
    if args.eval_dataset and os.path.exists(args.eval_dataset):
        for task_name, dataset in load_from_disk(args.eval_dataset).items():
            if keep_subsets and task_name not in keep_subsets:
                continue
            eval_dataloader_dict[task_name] = DataLoader(
                dataset, batch_size=args.mini_batch_size, shuffle=False,
                collate_fn=collate,
            )
    print(f"eval subsets: {list(eval_dataloader_dict.keys())}")

    experiment_name = args.experiment_name or (
        f"es-orig-{args.task}-sigma{args.sigma}-alpha{alpha}-pop{args.population_size}"
        f"-bs{args.batch_size}-model{str(args.model_name).split('/')[-1]}"
        f"-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"-- Running: {experiment_name} --")

    start_iteration = 0
    checkpoint = args.checkpoint
    if args.resume:
        latest_dir = f"{args.output_directory}/{experiment_name}/latest"
        prog_f, ckpt_f = f"{latest_dir}/progress.json", f"{latest_dir}/pytorch_model.pth"
        if os.path.exists(prog_f) and os.path.exists(ckpt_f):
            import json as _json
            last_done = int(_json.load(open(prog_f))["iteration"])
            start_iteration = last_done + 1
            checkpoint = ckpt_f
            print(f"[RESUME] resuming from iter {start_iteration}, ckpt={ckpt_f}")
        else:
            print(f"[RESUME] no latest checkpoint under {latest_dir}; starting fresh")

    trainer = ESRelayTrainer(
        grzo_mode="single_point",            # unused; parent bookkeeping
        group_size=args.population_size,     # maps to population_size in parent
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
