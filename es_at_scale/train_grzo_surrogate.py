# Angle B entry point: GRZO on the GRPO surrogate loss (Countdown, binary reward).
#
# Rollout + group advantage exactly as GRPO; only the backprop step is replaced
# by a two-point ZO estimate of L(theta) = -sum A_hat * log pi_theta(y|x).
# See es_at_scale/trainer/grzo_surrogate_trainer.py and ZO_RLVR_ROADMAP_v2.md.
#
# Knobs:
#   --batch-size B  prompts per step; --group-size G rollouts per prompt (GRPO group)
#   --sigma         ZO smoothing radius (small; NOT an exploration knob here)
#   --lr            learning rate for the ZO update (1/2sigma absorbed)
#   --rollout-temperature  sampling T for rollouts (default 1.0, the GRPO regime)

import argparse
from datetime import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk

from es_at_scale.trainer.grzo_surrogate_trainer import GRZOSurrogateTrainer


def countdown_collate_fn(batch):
    prompts = [item["context"] for item in batch]
    targets = [{"numbers": item["numbers"], "target": item["target"]} for item in batch]
    return prompts, targets


def math_collate_fn(batch):
    prompts = [item["problem"] for item in batch]
    targets = [item["answer"] for item in batch]
    return prompts, targets


def conciseness_collate_fn(batch):
    prompts = [item["input"] for item in batch]
    targets = [item["target"] for item in batch]
    return prompts, targets


def identity_template(prompt):
    # Conciseness protocol (ES paper archive script): raw prompt, no chat template.
    return prompt


def set_seed(seed_value=42):
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main():
    parser = argparse.ArgumentParser(description="GRZO-on-surrogate (Angle B) training.")

    parser.add_argument("--task", type=str, default="countdown",
                        choices=["countdown", "math", "conciseness"],
                        help="Selects reward fn, prompt template, and collate. "
                             "countdown/math use a binary 0/1 answer reward; "
                             "conciseness uses the ES-paper length reward "
                             "-|len(y)-len(s)| with raw prompts (no template).")
    parser.add_argument("--countdown-reward", type=str, default="binary",
                        choices=["binary", "margin", "margin-tiebreak", "margin-anneal"],
                        help="binary: 0/1. margin: exp(-rel_miss/tau) (gameable at "
                             "scale). margin-tiebreak: binary + 0.2*margin -- "
                             "exactness dominates, margin only orders the wrongs. "
                             "margin-anneal: pure margin with tau homotopy "
                             "tau -> margin-tau-end over the run.")
    parser.add_argument("--math-reward", type=str, default="binary",
                        choices=["binary", "margin-tiebreak"],
                        help="math task reward: binary boxed 0/1, or tiebreak "
                             "margin over numeric answers.")
    parser.add_argument("--margin-tau-end", type=float, default=0.02,
                        help="Final tau for margin-anneal homotopy.")
    parser.add_argument("--margin-tau", type=float, default=0.1,
                        help="Margin sharpness: relative miss where reward ~ 0.37.")
    parser.add_argument("--lr-schedule", type=str, default="const",
                        choices=["const", "cosine"],
                        help="cosine anneals lr to 0 over n-iterations (late-run "
                             "instability fix).")
    parser.add_argument("--eval-subsets", type=str, default="",
                        help="Comma-separated eval subset names to keep (e.g. "
                             "'math500'). Empty = all subsets in --eval-dataset.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--group-size", type=int, default=8,
                        help="G: rollouts per prompt (GRPO group).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="B: prompts per training step.")
    parser.add_argument("--mini-batch-size", type=int, default=8)
    parser.add_argument("--no-length-norm", action="store_true",
                        help="Use summed (not per-token mean) logprob in the surrogate.")
    parser.add_argument("--delta-norm", type=str, default="zscore",
                        choices=["zscore", "none"],
                        help="zscore: unit-variance coeffs (fixed-size steps). "
                             "none: raw delta/(2*sigma) directional-derivative scale.")
    parser.add_argument("--min-directions", type=int, default=1,
                        help="Accumulate scored directions across steps; apply the "
                             "update only once this many are collected.")
    parser.add_argument("--dapo-target-groups", type=int, default=0,
                        help="DAPO dynamic resampling: re-draw prompts until this many "
                             "non-degenerate groups are pooled per step (0 = off).")
    parser.add_argument("--dapo-max-rounds", type=int, default=4,
                        help="Max extra rollout rounds per step for DAPO resampling.")
    parser.add_argument("--dapo-draw", type=int, default=0,
                        help="Prompts drawn per resample round (0 = 4*batch-size). "
                             "Half prioritized from known-active prompts, half uniform.")
    parser.add_argument("--pairs-per-direction", type=int, default=1,
                        help="Hybrid: each direction's delta = mean over this many "
                             "shared pairs (ES-style m-averaging). 1 = legacy.")
    parser.add_argument("--directions-per-step", type=int, default=0,
                        help="Hybrid: fresh direction seeds per step (0 = one per pair).")
    parser.add_argument("--n-iterations", type=int, default=300)
    parser.add_argument("--eval-freq", type=int, default=25)
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
                        help="Resume from {output_dir}/{experiment_name}/latest/ "
                             "(requires fixed --experiment-name).")

    args = parser.parse_args()
    print(args)

    # Binary (answer-only) rewards for both tasks: the RLVR setting (roadmap 8.2).
    anneal_reward_fn, anneal_tau = None, None
    if args.task == "math":
        from es_at_scale.reward_function.math_grader import (
            boxed_reward_fn, math_margin_tiebreak_reward_fn,
        )
        from es_at_scale.template_function.apply_template import qwen_math_template
        if args.math_reward == "margin-tiebreak":
            import functools
            reward_function = functools.partial(
                math_margin_tiebreak_reward_fn, tau=args.margin_tau)
        else:
            reward_function = boxed_reward_fn      # already pure 0/1
        template_function = qwen_math_template
        collate = math_collate_fn
    elif args.task == "conciseness":
        from es_at_scale.reward_function.conciseness_grader import (
            conciseness_reward_fn,
        )
        reward_function = conciseness_reward_fn
        template_function = identity_template   # paper uses raw prompts
        collate = conciseness_collate_fn
    else:
        from es_at_scale.reward_function.countdown_grader import (
            countdown_answer_only_reward_fn, countdown_margin_reward_fn,
        )
        from es_at_scale.template_function.apply_template import countdown_template
        if args.countdown_reward == "margin":
            import functools
            reward_function = functools.partial(
                countdown_margin_reward_fn, tau=args.margin_tau)
        elif args.countdown_reward == "margin-anneal":
            import functools
            reward_function = functools.partial(
                countdown_margin_reward_fn, tau=args.margin_tau)
            anneal_reward_fn = countdown_margin_reward_fn
            anneal_tau = (args.margin_tau, args.margin_tau_end)
        elif args.countdown_reward == "margin-tiebreak":
            import functools
            from es_at_scale.reward_function.countdown_grader import (
                countdown_margin_tiebreak_reward_fn,
            )
            reward_function = functools.partial(
                countdown_margin_tiebreak_reward_fn, tau=args.margin_tau)
        else:
            reward_function = countdown_answer_only_reward_fn
        template_function = countdown_template
        collate = countdown_collate_fn

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
        f"grzos-{args.task}-sigma{args.sigma}-lr{args.lr}-T{args.rollout_temperature}"
        f"-G{args.group_size}-B{args.batch_size}"
        f"-tkn{args.max_tokens}-model{str(args.model_name).split('/')[-1]}"
        f"-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    print(f"-- Running: {experiment_name} --")

    start_iteration = 0
    checkpoint = args.checkpoint
    if args.resume:
        latest_dir = f"{args.output_directory}/{experiment_name}/latest"
        prog_f, ckpt_f = f"{latest_dir}/progress.json", f"{latest_dir}/pytorch_model.pth"
        # TP>1 runs save shard-per-rank files (pytorch_model.pth.rank0, ...).
        if os.path.exists(prog_f) and (
            os.path.exists(ckpt_f) or os.path.exists(f"{ckpt_f}.rank0")
        ):
            import json as _json
            last_done = int(_json.load(open(prog_f))["iteration"])
            start_iteration = last_done + 1
            checkpoint = ckpt_f
            print(f"[RESUME] resuming from iter {start_iteration}, ckpt={ckpt_f}")
        else:
            print(f"[RESUME] no latest checkpoint under {latest_dir}; starting fresh")

    trainer = GRZOSurrogateTrainer(
        rollout_temperature=args.rollout_temperature,
        lr=args.lr,
        normalize_by_length=not args.no_length_norm,
        delta_norm=args.delta_norm,
        min_directions=args.min_directions,
        dapo_target_groups=args.dapo_target_groups,
        dapo_max_rounds=args.dapo_max_rounds,
        dapo_draw=args.dapo_draw,
        pairs_per_direction=args.pairs_per_direction,
        directions_per_step=args.directions_per_step,
        lr_schedule=args.lr_schedule,
        anneal_reward_fn=anneal_reward_fn,
        anneal_tau=anneal_tau,
        grzo_mode="single_point",       # unused by the surrogate step; parent field
        group_size=args.group_size,
        start_iteration=start_iteration,
        model_name=args.model_name,
        checkpoint=checkpoint,
        sigma=args.sigma,
        alpha=args.lr,                  # parent bookkeeping; surrogate uses self.lr
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
