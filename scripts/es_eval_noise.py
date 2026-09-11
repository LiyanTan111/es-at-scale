"""Diagnostic: how noisy is an ES fitness evaluation on this hardware?

ES ranks 30 perturbed models by their greedy accuracy on the 200 train prompts; the
population fitness std is ~0.015-0.02. If the *evaluation noise* of a fixed model (batching /
kernel non-determinism, bf16 perturb->restore drift) is comparable, the ES signal is drowned.

Measures, with one vLLM engine:
  A) same theta, K repeats, no perturbation            -> pure evaluation noise
  B) sigma=1e-3, SAME seed, K repeats (perturb/restore) -> eval noise + drift
  C) sigma=1e-3, K DIFFERENT seeds                     -> population fitness std (signal+noise)
  D) after the K cycles of B/C, fitness at theta again -> drift of the restore path

  CUDA_VISIBLE_DEVICES=3 python scripts/es_eval_noise.py --use-gpus 3 --repeats 5
"""
import argparse, os, sys, time
import numpy as np
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
    ap.add_argument("--use-gpus", default="3")
    ap.add_argument("--n-vllm-engines", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--train-dataset", default="datasets/train/countdown")
    args = ap.parse_args()

    ds = next(iter(load_from_disk(args.train_dataset).values()))
    dl = DataLoader(ds, batch_size=200, collate_fn=countdown_collate_fn, shuffle=False)
    tr = EvolutionStrategiesTrainer(
        model_name=args.model_name, checkpoint=None, sigma=args.sigma, alpha=args.sigma / 2,
        population_size=30, reward_shaping="z-scores", num_iterations=0, max_tokens=args.max_tokens,
        batch_size=200, mini_batch_size=200, reward_function=countdown_reward_fn,
        template_function=countdown_template, train_dataloader=dl, eval_dataloader_dict={},
        eval_freq=10 ** 9, n_vllm_engines=args.n_vllm_engines, n_gpu_per_vllm_engine=1,
        logging="none", use_gpus=args.use_gpus, global_seed=42,
        output_directory="/data/liyan/runs/grzo-smoke/es-noise", experiment_name="es-noise",
        save_best_models=False, reward_function_timeout=10)
    prompts, targets = next(iter(dl)); prompts = [tr.template(p) for p in prompts]; targets = list(targets)
    sp = SamplingParams(n=1, seed=42, temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    def fitness_at_theta():
        outs = __import__("ray").get(tr.engines[0].generate.remote(prompts, sp, use_tqdm=False))
        m = tr._postprocess_outputs(outs, targets)
        return m["avg_reward"], float(np.mean([r > 0.999 for r in m["rewards"]]))

    def fitness_perturbed(seed):
        perf, _ = tr.evaluate_population_on_batch([seed], prompts, targets, sp)
        return perf[seed]["avg_reward"]

    print("[NOISE-ES] A) fixed theta, repeated evaluation")
    A = [fitness_at_theta() for _ in range(args.repeats)]
    print("   shaped:", [f"{a[0]:.4f}" for a in A], "| answer_acc:", [f"{a[1]:.4f}" for a in A])
    A_std = float(np.std([a[0] for a in A]))

    print("[NOISE-ES] B) sigma=%g, SAME seed, repeated perturb->eval->restore" % args.sigma)
    B = [fitness_perturbed(12345) for _ in range(args.repeats)]
    print("   shaped:", [f"{b:.4f}" for b in B])

    print("[NOISE-ES] C) sigma=%g, DIFFERENT seeds (population spread)" % args.sigma)
    C = [fitness_perturbed(100 + k) for k in range(args.repeats * 2)]
    print("   shaped:", [f"{c:.4f}" for c in C])

    print("[NOISE-ES] D) theta after %d perturb/restore cycles" % (len(B) + len(C)))
    D = [fitness_at_theta() for _ in range(2)]
    print("   shaped:", [f"{d[0]:.4f}" for d in D], "| answer_acc:", [f"{d[1]:.4f}" for d in D])

    print(f"[NOISE-ES] SUMMARY: eval-noise std(A)={A_std:.4f} | same-seed std(B)={np.std(B):.4f} "
          f"| population std(C)={np.std(C):.4f} | mean(A)={np.mean([a[0] for a in A]):.4f} mean(D)={np.mean([d[0] for d in D]):.4f}")
    tr.cleanup()


if __name__ == "__main__":
    main()
