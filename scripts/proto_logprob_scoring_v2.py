"""Angle B prototype v2: master-copy perturbation with EXACT restore.

v1 finding: in-place bf16 add/sub perturb-restore drifts weights by ~ulp per add;
restore error in log-prob (0.46) was ~half the two-point signal (0.91). Unusable
for Angle B's small-signal estimator.

v2 scheme: keep a GPU-resident bf16 master copy of theta. Perturb by
p.copy_((master_fp32 + sign*sigma*noise_fp32).to(bf16)) -- one deterministic
rounding from master; restore by p.copy_(master) -- bitwise exact.

Checks:
  0. Scoring determinism: same weights -> identical score twice (control).
  1. Restore exactness: score(base) == score(after perturb+restore) bitwise.
  2. Delta reproducibility: two-point delta computed twice must match exactly.
  3. Report delta magnitude for a few seeds (signal scale for the trainer).
"""
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SIGMA = 1e-3

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", enforce_eager=True,
          gpu_memory_utilization=0.6, enable_prefix_caching=False)
model = llm.llm_engine.model_executor.driver_worker.model_runner.model


def score_logpi(x_text, y_text):
    x_ids = tok(x_text, add_special_tokens=False).input_ids
    xy_ids = tok(x_text + y_text, add_special_tokens=False).input_ids
    n_resp = len(xy_ids) - len(x_ids)
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    out = llm.generate([{"prompt_token_ids": xy_ids}], sp, use_tqdm=False)[0]
    pls = out.prompt_logprobs
    total = 0.0
    for pos in range(len(xy_ids) - n_resp, len(xy_ids)):
        entry = pls[pos]; tid = xy_ids[pos]
        lp = entry[tid].logprob if (entry and tid in entry) else list(entry.values())[0].logprob
        total += float(lp)
    return total, n_resp


x = ("You are a helpful assistant. Using the numbers [44, 19, 35], create an "
     "expression that equals 98.\n<think>")
y = " (44 + 19) + 35 </think>\n<answer>(44 + 19) + 35</answer>"

print("=== 0) scoring determinism at fixed weights ===")
s1, _ = score_logpi(x, y); s2, _ = score_logpi(x, y)
print(f"  score1={s1:.6f} score2={s2:.6f} |diff|={abs(s1-s2):.2e} (want 0)")

print("=== master copy (GPU bf16) ===")
master = {n: p.detach().clone() for n, p in model.named_parameters()}
mem = sum(v.numel() * v.element_size() for v in master.values()) / 2**30
print(f"  master size: {mem:.2f} GiB")


def perturb_from_master(seed, sign):
    for n, p in model.named_parameters():
        g = torch.Generator(device=p.device); g.manual_seed(int(seed))
        noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=g)
        p.data.copy_((master[n].to(torch.float32)
                      + float(sign) * SIGMA * noise).to(p.dtype))
    torch.cuda.synchronize()


def restore_master():
    for n, p in model.named_parameters():
        p.data.copy_(master[n])
    torch.cuda.synchronize()


print("=== 1) restore exactness ===")
base, _ = score_logpi(x, y)
perturb_from_master(12345, +1)
lp_p, _ = score_logpi(x, y)
restore_master()
r1, _ = score_logpi(x, y)
print(f"  base={base:.6f} theta+={lp_p:.6f} restored={r1:.6f}")
print(f"  restore |err|={abs(r1-base):.2e} (want exactly 0)")

print("=== 2) two-point delta reproducibility + 3) magnitudes ===")
for seed in (12345, 777, 31415):
    deltas = []
    for rep in range(2):
        perturb_from_master(seed, +1); lp_plus, _ = score_logpi(x, y)
        perturb_from_master(seed, -1); lp_minus, _ = score_logpi(x, y)
        restore_master()
        deltas.append((-lp_plus) - (-lp_minus))  # delta on the NLL
    match = "EXACT" if deltas[0] == deltas[1] else f"DIFF {abs(deltas[0]-deltas[1]):.2e}"
    print(f"  seed={seed}: delta={deltas[0]:+.5f} (repeat: {match})")

print("DONE")
