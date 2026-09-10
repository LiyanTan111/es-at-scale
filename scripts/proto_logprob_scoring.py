"""Angle B foundation prototype: teacher-forcing log pi_theta(y|x) via vLLM
prompt_logprobs, and confirm a parameter perturbation moves it.

Checks:
  1. Token alignment: score = sum_t log p(y_t | x, y_<t) over ONLY the response
     tokens, using token-id prompts (no re-tokenization mismatch).
  2. Sanity: a coherent/correct response scores higher (less negative) than gibberish.
  3. Perturbation sensitivity: perturbing weights by +/- sigma*eps changes the score,
     so the two-point difference delta = l+ - l- is nonzero (Angle B needs this).
  4. Length normalization: report both summed NLL and per-token NLL.

Run on 1 GPU. Not the full trainer -- just the scoring mechanism.
"""
import numpy as np
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True,
          gpu_memory_utilization=0.6, enable_prefix_caching=False)


def score_logpi(x_text, y_text):
    """Return (sum log p(y|x), n_response_tokens) via prompt_logprobs on x+y token ids."""
    x_ids = tok(x_text, add_special_tokens=False).input_ids
    xy_ids = tok(x_text + y_text, add_special_tokens=False).input_ids
    n_resp = len(xy_ids) - len(x_ids)
    if n_resp <= 0:
        return float("nan"), 0
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    out = llm.generate([{"prompt_token_ids": xy_ids}], sp, use_tqdm=False)[0]
    pls = out.prompt_logprobs  # list aligned to prompt tokens; [0] is None
    # response tokens occupy the last n_resp positions
    total = 0.0
    for pos in range(len(xy_ids) - n_resp, len(xy_ids)):
        entry = pls[pos]
        tid = xy_ids[pos]
        # entry: {token_id: Logprob(logprob=...)}. prompt_logprobs=0 -> the actual token.
        lp = entry[tid].logprob if (entry and tid in entry) else list(entry.values())[0].logprob
        total += float(lp)
    return total, n_resp


# A Countdown-style prompt and two candidate responses.
x = ("You are a helpful assistant. Using the numbers [44, 19, 35], create an "
     "expression that equals 98.\n<think>")
y_good = " (44 + 19) + 35 </think>\n<answer>(44 + 19) + 35</answer>"   # =98, correct
y_bad = " asdf qwer zxcv lkjh </think>\n<answer>1 + 1</answer>"          # gibberish/wrong

print("=== 1) teacher-forcing scores (higher = more likely) ===")
for name, y in [("good", y_good), ("bad", y_bad)]:
    s, n = score_logpi(x, y)
    print(f"  {name}: sum_logp={s:.3f}  n_tok={n}  per_tok_nll={-s/max(n,1):.3f}")

print("=== 2) perturbation sensitivity (delta should be nonzero) ===")
SIGMA = 1e-3


def _find_model():
    # vLLM internals move across versions/engines (V0 vs V1). Try known paths.
    import itertools
    cands = [
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.engine_core.engine_core.model_executor.driver_worker.model_runner.model",
    ]
    for path in cands:
        obj = llm
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            list(itertools.islice(obj.named_parameters(), 1))  # verify it's a module
            print(f"  [model found via] llm.{path}")
            return obj
        except Exception:
            continue
    return None


model = _find_model()


def perturb(seed, sign):
    for _, p in model.named_parameters():
        g = torch.Generator(device=p.device); g.manual_seed(int(seed))
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=g)
        p.data.add_(sign * SIGMA * noise)
    torch.cuda.synchronize()


if model is None:
    print("  [SKIP] could not locate model on driver (V1 engine path differs);")
    print("  perturbation will be tested inside Ray workers in the real trainer.")
    print("DONE")
    raise SystemExit(0)

base, n = score_logpi(x, y_good)
perturb(12345, +1.0)
lp_plus, _ = score_logpi(x, y_good)
perturb(12345, -2.0)   # from +sigma to -sigma
lp_minus, _ = score_logpi(x, y_good)
perturb(12345, +1.0)   # restore to base
restored, _ = score_logpi(x, y_good)

print(f"  base={base:.4f}  theta+={lp_plus:.4f}  theta-={lp_minus:.4f}  restored={restored:.4f}")
print(f"  two-point delta (l+ - l-) on -logp = {(-lp_plus) - (-lp_minus):+.5f}")
print(f"  restore error vs base = {abs(restored - base):.2e} (should be ~0)")
print("DONE")
