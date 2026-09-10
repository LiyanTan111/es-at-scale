"""E1.6 — GRPO (backprop) reference on Countdown with the FORGE/ES protocol.

Runs in the separate GRPO venv (/data/liyan/venvs/grpo: TRL 0.29 + vLLM 0.11 colocate).
Same train set (datasets/train/countdown, 200 prompts), same prompt text (the dataset's
`context` field is the full prompt, as countdown_template is the identity), same
0/1 reward (countdown_answer_only_reward_fn), max_completion_length 512, T=1.0, G=8.
Loss = per-sequence-mean log-prob weighted by group-normalised advantage
(loss_type="grpo", beta=0): the same objective FORGE's surrogate estimates.

Checkpoints are HF directories -> evaluate with scripts/eval_hf_ckpt.sh (vLLM greedy on
countdown_eval, the exact FORGE/ES eval). Cumulative generations per optimizer step =
generation_batch_size (default 512).

  source /data/liyan/venvs/grpo/bin/activate
  CUDA_VISIBLE_DEVICES=3 python scripts/grpo_countdown.py --max-steps 2000 \
      --exp grpo-cd-1p5b-h100-s42
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_from_disk
from trl import GRPOConfig, GRPOTrainer
from es_at_scale.reward_function.countdown_grader import countdown_answer_only_reward_fn


def countdown_reward(completions, numbers, target, **kwargs):
    out = []
    for c, n, t in zip(completions, numbers, target):
        text = c if isinstance(c, str) else c[0]["content"]
        _, r = countdown_answer_only_reward_fn(text, {"numbers": n, "target": t})
        out.append(float(r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train-dataset", default="datasets/train/countdown")
    ap.add_argument("--exp", default="grpo-cd-1p5b-h100-s42")
    ap.add_argument("--out-root", default="/data/liyan/runs/grpo")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--prompts-per-step", type=int, default=64)   # x G completions per optimizer step
    ap.add_argument("--micro-batch", type=int, default=64)        # completions per forward/backward
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--loss-type", default="grpo")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--vllm-mem", type=float, default=0.3)
    ap.add_argument("--wandb-project", default="grzo-rlvr")
    ap.add_argument("--logging", default="wandb")
    args = ap.parse_args()

    ds = next(iter(load_from_disk(args.train_dataset).values()))
    ds = ds.map(lambda ex: {"prompt": ex["context"]}, remove_columns=[c for c in ds.column_names if c not in ("numbers", "target")])
    gen_bs = args.prompts_per_step * args.num_generations
    assert gen_bs % args.micro_batch == 0
    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    cfg = GRPOConfig(
        output_dir=f"{args.out_root}/{args.exp}", run_name=args.exp, seed=args.seed,
        num_generations=args.num_generations, max_completion_length=args.max_completion_length,
        temperature=1.0, top_p=1.0, beta=0.0, loss_type=args.loss_type, scale_rewards="group",
        epsilon=0.2, num_iterations=1,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=gen_bs // args.micro_batch,
        generation_batch_size=gen_bs,
        learning_rate=args.lr, lr_scheduler_type="constant", warmup_steps=0,
        max_steps=args.max_steps, bf16=True, gradient_checkpointing=True,
        logging_steps=1, save_steps=args.save_steps, save_total_limit=None,
        report_to=[args.logging] if args.logging != "none" else [],
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=1024 + args.max_completion_length,
        log_completions=False,
    )
    trainer = GRPOTrainer(model=args.model_name, reward_funcs=countdown_reward, args=cfg, train_dataset=ds)
    trainer.train()
    trainer.save_model(f"{args.out_root}/{args.exp}/final")


if __name__ == "__main__":
    main()
