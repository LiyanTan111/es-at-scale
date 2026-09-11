"""E1.5 — Gradient alignment: does FORGE's zeroth-order estimate point along the true
GRPO gradient, and does alignment grow with the number of forward queries N?

Pure HF/torch on ONE GPU (no vLLM), so the ZO estimate g_hat and the backprop gradient
g_BP share one parameter layout. On one fixed batch of (x, y, A) pairs:

  g_BP      = grad_theta (1/P) sum_j  -A_j * log pi(y_j|x_j)/|y_j|          (autograd, fp32)
  g_hat(N)  = (1/N) sum_{j<N} c_j eps_j,   eps_j ~ N(0, I) regenerated from seeds
              per-example scheme: direction j scored on ONE pair (pairs cycled), 2 forwards
              hybrid scheme:      direction j scored on ALL pairs (mean), 2 batched forwards
              c_j = delta_j/(2 sigma)  (raw)   or   zscore(delta)   (z, training default)
              delta_j = -A_j [ log pi_{theta+sigma eps}(y|x) - log pi_{theta-sigma eps}(y|x) ]/|y|

Reports, per N:  cos(g_hat, g_BP) raw/z;  same-rollout agreement cos(g_hat_A, g_hat_B);
cross-rollout agreement cos(g_hat_A, g_hat_C) (C uses a second rollout of the same prompts);
delta statistics and a step-norm-matched learning rate for the raw estimator.

  CUDA_VISIBLE_DEVICES=4 python scripts/grad_alignment.py --model-name Qwen/Qwen2.5-0.5B-Instruct \
      --out /data/liyan/runs/grzo-smoke/align/align_0p5b_base.json
Optional: --vllm-ckpt <pytorch_model.pth> loads a FORGE checkpoint (vLLM layout, fused qkv /
gate_up) into the HF model.
"""
import argparse, json, math, os, sys, time, copy
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from es_at_scale.reward_function.countdown_grader import countdown_answer_only_reward_fn


# --------------------------------------------------------------------------- #
# checkpoint loading (vLLM fused layout -> HF)
# --------------------------------------------------------------------------- #
def load_vllm_ckpt_into_hf(model, path):
    sd = torch.load(path, map_location="cpu")
    cfg = model.config
    hd = cfg.hidden_size // cfg.num_attention_heads
    q_n, kv_n = cfg.num_attention_heads * hd, cfg.num_key_value_heads * hd
    inter = cfg.intermediate_size
    n_copied = 0
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in sd:
                src = sd[name]
            elif ".self_attn." in name and any(k in name for k in ("q_proj", "k_proj", "v_proj")):
                fused = sd[name.replace("q_proj", "qkv_proj").replace("k_proj", "qkv_proj").replace("v_proj", "qkv_proj")]
                if "q_proj" in name: src = fused[:q_n]
                elif "k_proj" in name: src = fused[q_n:q_n + kv_n]
                else: src = fused[q_n + kv_n:q_n + 2 * kv_n]
            elif ".mlp." in name and ("gate_proj" in name or "up_proj" in name):
                fused = sd[name.replace("gate_proj", "gate_up_proj").replace("up_proj", "gate_up_proj")]
                src = fused[:inter] if "gate_proj" in name else fused[inter:2 * inter]
            elif name == "lm_head.weight" and "model.embed_tokens.weight" in sd and getattr(cfg, "tie_word_embeddings", False):
                src = sd["model.embed_tokens.weight"]
            else:
                raise KeyError(f"no source for HF param {name}")
            assert src.shape == p.shape, (name, src.shape, p.shape)
            p.copy_(src.to(p.dtype)); n_copied += 1
    print(f"[ALIGN] loaded vLLM ckpt {path}: {n_copied} tensors")


# --------------------------------------------------------------------------- #
# data / rollouts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rollout(model, tok, prompts, G, max_new, seed, device):
    tok.padding_side = "left"
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    torch.manual_seed(seed)
    # Qwen2.5-Instruct's generation_config carries repetition_penalty=1.1, top_k=20, top_p=0.8;
    # vLLM's explicit SamplingParams (used in training) apply none of these. Override all of
    # them so the HF rollouts match the training distribution (checked: 0/24 vs 3/24 active groups).
    out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                         repetition_penalty=1.0, max_new_tokens=max_new, num_return_sequences=G,
                         pad_token_id=tok.pad_token_id)
    L = enc["input_ids"].shape[1]
    gens = out[:, L:]
    res = []  # per prompt: list of (y_ids, text)
    for i in range(len(prompts)):
        x_ids = enc["input_ids"][i][enc["attention_mask"][i].bool()].tolist()
        row = []
        for g in range(G):
            y = gens[i * G + g].tolist()
            if tok.eos_token_id in y:
                y = y[: y.index(tok.eos_token_id) + 1]
            y = [t for t in y if t != tok.pad_token_id] if tok.pad_token_id != tok.eos_token_id else y
            row.append((x_ids, y, tok.decode(y, skip_special_tokens=True)))
        res.append(row)
    return res


def grade(rows, targets):
    R = np.zeros((len(rows), len(rows[0])))
    if os.environ.get("ALIGN_FAKE_REWARDS") == "1":   # code-path smoke test only
        return (np.random.default_rng(0).random(R.shape) < 0.5).astype(float)
    for i, row in enumerate(rows):
        for g, (_, _, text) in enumerate(row):
            _, r = countdown_answer_only_reward_fn(text, targets[i]); R[i, g] = float(r)
    return R


def build_pairs(rows, R):
    pairs = []
    for i, row in enumerate(rows):
        std = R[i].std()
        if std <= 1e-12: continue
        adv = (R[i] - R[i].mean()) / (std + 1e-8)
        for g, (x, y, _) in enumerate(row):
            if abs(adv[g]) < 1e-12 or not y: continue
            pairs.append({"x": x, "y": y, "adv": float(adv[g])})
    return pairs


# --------------------------------------------------------------------------- #
# log-prob evaluation
# --------------------------------------------------------------------------- #
def batch_mean_logprob(model, pairs, device, pad_id, need_grad=False):
    """Per-pair mean log pi(y|x) over the y tokens. Right-padded batch."""
    seqs = [p["x"] + p["y"] for p in pairs]
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    ymask = torch.zeros((len(seqs), T), dtype=torch.bool)
    for i, (p, s) in enumerate(zip(pairs, seqs)):
        ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        ymask[i, len(p["x"]):len(s)] = True
    ids, att, ymask = ids.to(device), att.to(device), ymask.to(device)
    ctx = torch.enable_grad() if need_grad else torch.no_grad()
    with ctx:
        logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
        lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = ymask[:, 1:].float()
        mean_lp = (lp * m).sum(1) / m.sum(1).clamp(min=1)
    return mean_lp


def bp_gradient(model_bf16, pairs, device, pad_id, chunk):
    """g_BP = grad of (1/P) sum_j -A_j * mean_logprob_j, computed on an fp32 copy."""
    m32 = copy.deepcopy(model_bf16).float(); m32.train(False)
    for p in m32.parameters(): p.requires_grad_(True); p.grad = None
    P = len(pairs)
    for s in range(0, P, chunk):
        ch = pairs[s:s + chunk]
        mlp = batch_mean_logprob(m32, ch, device, pad_id, need_grad=True)
        adv = torch.tensor([p["adv"] for p in ch], device=device)
        loss = (-(adv * mlp)).sum() / P
        loss.backward()
    g = {n: p.grad.detach().clone() for n, p in m32.named_parameters()}
    del m32; torch.cuda.empty_cache()
    return g


# --------------------------------------------------------------------------- #
# ZO machinery (seed-regenerated noise, same convention as worker_extension)
# --------------------------------------------------------------------------- #
class ZO:
    def __init__(self, model, sigma, device):
        self.model, self.sigma, self.device = model, sigma, device
        self.params = [(n, p) for n, p in model.named_parameters()]
        self.master = [p.detach().clone() for _, p in self.params]

    def _noise(self, p, seed):
        gen = torch.Generator(device=p.device); gen.manual_seed(int(seed))
        return torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)

    @torch.no_grad()
    def perturb(self, seed, sign):
        for (_, p), m in zip(self.params, self.master):
            p.copy_((m.float() + sign * self.sigma * self._noise(p, seed)).to(p.dtype))

    @torch.no_grad()
    def restore(self):
        for (_, p), m in zip(self.params, self.master): p.copy_(m)

    @torch.no_grad()
    def inner_with_noise(self, seed, vecs):
        """<eps(seed), v> for each dict of fp32 tensors in vecs; also ||eps||^2."""
        outs = [0.0] * len(vecs); n2 = 0.0
        for (name, p), _ in zip(self.params, self.master):
            e = self._noise(p, seed).view(-1)
            n2 += float(torch.dot(e, e))
            for k, v in enumerate(vecs):
                if v is not None:
                    outs[k] += float(torch.dot(e, v[name].view(-1)))
        return outs, n2

    @torch.no_grad()
    def materialize(self, seeds, coeffs):
        """g_hat = (1/N) sum c_j eps_j as a dict of fp32 tensors."""
        acc = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in self.params}
        for s, c in zip(seeds, coeffs):
            for n, p in self.params:
                acc[n].add_(self._noise(p, s), alpha=float(c))
        for n in acc: acc[n].div_(len(seeds))
        return acc


def vnorm(g): return math.sqrt(sum(float(torch.dot(t.view(-1), t.view(-1))) for t in g.values()))
def vdot(a, b): return sum(float(torch.dot(a[n].view(-1), b[n].view(-1))) for n in a)
def vcos(a, b): return vdot(a, b) / (vnorm(a) * vnorm(b) + 1e-30)


def score_per_example(zo, pairs, N, rng, device, pad_id, gbp):
    """Direction j scored on one pair (cycling). Returns seeds, deltas, <eps,gBP>, ||eps||^2."""
    order = rng.permutation(len(pairs)); idx = [int(order[j % len(pairs)]) for j in range(N)]
    seeds = [int(rng.integers(0, 2 ** 30)) for _ in range(N)]
    deltas, a_bp, n2s = [], [], []
    for j in range(N):
        pr = pairs[idx[j]]
        zo.perturb(seeds[j], +1.0); lp_p = float(batch_mean_logprob(zo.model, [pr], device, pad_id)[0])
        zo.perturb(seeds[j], -1.0); lp_m = float(batch_mean_logprob(zo.model, [pr], device, pad_id)[0])
        deltas.append(-pr["adv"] * (lp_p - lp_m))
        (a,), n2 = zo.inner_with_noise(seeds[j], [gbp]); a_bp.append(a); n2s.append(n2)
    zo.restore()
    return seeds, np.array(deltas), np.array(a_bp), np.array(n2s)


def score_hybrid(zo, pairs, N, rng, device, pad_id, gbp, chunk):
    """Direction j scored on ALL pairs (mean over pairs), batched forwards."""
    seeds = [int(rng.integers(0, 2 ** 30)) for _ in range(N)]
    adv = np.array([p["adv"] for p in pairs])
    deltas, a_bp, n2s = [], [], []
    for j in range(N):
        vals = {}
        for sign in (+1.0, -1.0):
            zo.perturb(seeds[j], sign)
            lps = []
            for s in range(0, len(pairs), chunk):
                lps.extend(batch_mean_logprob(zo.model, pairs[s:s + chunk], device, pad_id).tolist())
            vals[sign] = np.array(lps)
        deltas.append(float(np.mean(-adv * (vals[1.0] - vals[-1.0]))))
        (a,), n2 = zo.inner_with_noise(seeds[j], [gbp]); a_bp.append(a); n2s.append(n2)
    zo.restore()
    return seeds, np.array(deltas), np.array(a_bp), np.array(n2s)


def coeffs(deltas, sigma):
    raw = deltas / (2.0 * sigma)
    z = (deltas - deltas.mean()) / (deltas.std() + 1e-8) if deltas.std() > 1e-12 else raw.copy()
    return raw, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--vllm-ckpt", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--train-dataset", default="datasets/train/countdown")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--target-groups", type=int, default=8)
    ap.add_argument("--dapo-draw", type=int, default=32)
    ap.add_argument("--max-pool-rounds", type=int, default=12)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--n-list", default="32,64,96,256,512,1024")
    ap.add_argument("--hybrid-n-list", default="8,32,96")
    ap.add_argument("--bp-chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr-z", type=float, default=5e-4, help="training lr used with z-score (for step-norm matching)")
    ap.add_argument("--out", default="/data/liyan/runs/grzo-smoke/align/align.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device).eval()
    if args.vllm_ckpt: load_vllm_ckpt_into_hf(model, args.vllm_ckpt)
    for k, v in dict(temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0).items():
        setattr(model.generation_config, k, v)
    d = sum(p.numel() for p in model.parameters())
    print(f"[ALIGN] model {args.model_name} d={d/1e6:.1f}M device={device}")

    ds = next(iter(load_from_disk(args.train_dataset).values()))
    G = args.group_size

    # ---- fixed prompt set with >= target active groups (DAPO-style pooling) ----
    P_prompts, P_targets, seen = [], [], set()
    rounds = 0
    while len(P_prompts) < args.target_groups and rounds < args.max_pool_rounds:
        n = args.batch_size if rounds == 0 else args.dapo_draw
        idxs = [int(i) for i in rng.choice(len(ds), size=n, replace=False) if int(i) not in seen]
        seen.update(idxs)
        for s in range(0, len(idxs), 8):   # generate in chunks of 8 prompts (x G)
            sub = idxs[s:s + 8]
            prompts = [ds[i]["context"] for i in sub]
            targets = [{"numbers": ds[i]["numbers"], "target": ds[i]["target"]} for i in sub]
            rows = rollout(model, tok, prompts, G, args.max_new_tokens, 1000 + rounds * 100 + s, device)
            R = grade(rows, targets)
            for i in range(len(sub)):
                if R[i].std() > 1e-12 and len(P_prompts) < args.target_groups:
                    P_prompts.append(prompts[i]); P_targets.append(targets[i])
        rounds += 1
        print(f"[ALIGN] pooling round {rounds}: active {len(P_prompts)}/{args.target_groups}")
    assert P_prompts, "no active groups found"

    # ---- two independent rollouts of the fixed prompts ----
    rows0 = []; rows1 = []
    for s in range(0, len(P_prompts), 8):
        rows0 += rollout(model, tok, P_prompts[s:s + 8], G, args.max_new_tokens, 2001 + s, device)
        rows1 += rollout(model, tok, P_prompts[s:s + 8], G, args.max_new_tokens, 2002 + s, device)
    R0, R1 = grade(rows0, P_targets), grade(rows1, P_targets)
    pairs0, pairs1 = build_pairs(rows0, R0), build_pairs(rows1, R1)
    assert pairs0 and pairs1, f"no active pairs after re-rollout (R0={len(pairs0)}, R1={len(pairs1)})"
    print(f"[ALIGN] prompts={len(P_prompts)} pairs R0={len(pairs0)} R1={len(pairs1)} reward R0={R0.mean():.3f} R1={R1.mean():.3f}")
    torch.save({"pairs0": pairs0, "pairs1": pairs1, "prompts": P_prompts, "targets": P_targets},
               args.out.replace(".json", "_pairs.pt"))

    # ---- backprop gradients ----
    t0 = time.time()
    g0 = bp_gradient(model, pairs0, device, tok.pad_token_id, args.bp_chunk)
    g1 = bp_gradient(model, pairs1, device, tok.pad_token_id, args.bp_chunk)
    print(f"[ALIGN] g_BP: |g0|={vnorm(g0):.4e} |g1|={vnorm(g1):.4e} cos(g0,g1)={vcos(g0,g1):+.4f} ({time.time()-t0:.0f}s)")
    zo = ZO(model, args.sigma, device)
    res = {"model": args.model_name, "ckpt": args.vllm_ckpt, "d": d, "sigma": args.sigma, "G": G,
           "n_prompts": len(P_prompts), "pairs0": len(pairs0), "pairs1": len(pairs1),
           "gbp_norm0": vnorm(g0), "gbp_norm1": vnorm(g1), "cos_gbp0_gbp1": vcos(g0, g1),
           "per_example": [], "hybrid": []}

    def analyse(scheme, N, sA, sB, sC):
        (seedsA, dA, aA, n2A), (seedsB, dB, aB, n2B), (seedsC, dC, aC, n2C) = sA, sB, sC
        row = {"N": N, "delta_mean": float(dA.mean()), "delta_std": float(dA.std()),
               "delta_abs_mean": float(np.abs(dA).mean()), "eps_norm2_mean": float(n2A.mean()),
               "per_dir_cos_eps_gbp_mean": float(np.mean(aA / np.sqrt(n2A) / (vnorm(g0) + 1e-30)))}
        for tag, cf in (("raw", lambda dd: coeffs(dd, args.sigma)[0]), ("z", lambda dd: coeffs(dd, args.sigma)[1])):
            cA, cB, cC = cf(dA), cf(dB), cf(dC)
            gA = zo.materialize(seedsA, cA); gB = zo.materialize(seedsB, cB)
            row[f"cos_bp_{tag}"] = vcos(gA, g0)
            row[f"cos_bp_{tag}_B"] = vcos(gB, g0)
            row[f"cos_AB_{tag}"] = vcos(gA, gB)
            row[f"ghat_norm_{tag}"] = vnorm(gA)
            row[f"coeff_std_{tag}"] = float(np.std(cA))
            del gB
            gC = zo.materialize(seedsC, cC)
            row[f"cos_AC_{tag}"] = vcos(gA, gC)          # cross-rollout (A on R0, C on R1)
            row[f"cos_C_bp1_{tag}"] = vcos(gC, g1)
            del gA, gC; torch.cuda.empty_cache()
        row["lr_raw_stepnorm_matched"] = args.lr_z * row["coeff_std_z"] / (row["coeff_std_raw"] + 1e-30)
        return row

    for N in [int(x) for x in args.n_list.split(",") if x]:
        t0 = time.time()
        sA = score_per_example(zo, pairs0, N, rng, device, tok.pad_token_id, g0)
        sB = score_per_example(zo, pairs0, N, rng, device, tok.pad_token_id, g0)
        sC = score_per_example(zo, pairs1, N, rng, device, tok.pad_token_id, g1)
        row = analyse("per_example", N, sA, sB, sC); row["t_s"] = time.time() - t0
        res["per_example"].append(row)
        print(f"[ALIGN][per-example] N={N:5d} | cos(g_hat,g_BP) raw={row['cos_bp_raw']:+.4f} z={row['cos_bp_z']:+.4f} "
              f"| same-rollout AB raw={row['cos_AB_raw']:+.4f} z={row['cos_AB_z']:+.4f} "
              f"| cross-rollout AC raw={row['cos_AC_raw']:+.4f} | delta std={row['delta_std']:.2e} | {row['t_s']:.0f}s")
        json.dump(res, open(args.out, "w"), indent=2)

    for N in [int(x) for x in args.hybrid_n_list.split(",") if x]:
        t0 = time.time()
        sA = score_hybrid(zo, pairs0, N, rng, device, tok.pad_token_id, g0, args.bp_chunk)
        sB = score_hybrid(zo, pairs0, N, rng, device, tok.pad_token_id, g0, args.bp_chunk)
        sC = score_hybrid(zo, pairs1, N, rng, device, tok.pad_token_id, g1, args.bp_chunk)
        row = analyse("hybrid", N, sA, sB, sC); row["t_s"] = time.time() - t0
        res["hybrid"].append(row)
        print(f"[ALIGN][hybrid]      N={N:5d} | cos(g_hat,g_BP) raw={row['cos_bp_raw']:+.4f} z={row['cos_bp_z']:+.4f} "
              f"| same-rollout AB raw={row['cos_AB_raw']:+.4f} | cross-rollout AC raw={row['cos_AC_raw']:+.4f} | {row['t_s']:.0f}s")
        json.dump(res, open(args.out, "w"), indent=2)

    print(f"[ALIGN] wrote {args.out}")


if __name__ == "__main__":
    main()
