"""E1.8 — Local stability & curvature: does the stable learning rate scale with N?
(User's proposal, 2026-09-10.)

On the SAME fixed batch/surrogate L as E1.5 (pairs saved by grad_alignment.py), everything
in fp32 on one GPU, no vLLM:

  1. L(theta) exactly (batched teacher forcing).
  2. Forward-only curvature: tr(H) = E_u[u^T H u], u ~ N(0, I), via
         (L(theta + h u) + L(theta - h u) - 2 L(theta)) / h^2   (M random u, two step sizes h)
     plus the curvature along the true gradient, g^T H g / |g|^2 (g by autograd, fp32).
  3. For N in {64, 96, 256, 512, 1024}: R independent FORGE estimates g_hat (raw coefficients,
     per-example scheme, pairs cycled), and for each an eta grid around eta ∝ N:
         Delta L(eta) = L(theta - eta g_hat) - L(theta)      (exact, fp32)
     Quadratic fit Delta L = -a eta + b eta^2  ->  eta_opt = a/2b, eta_max = a/b.
  4. Prediction from forward-only quantities only:
         a_pred = |g_bar|^2  (E<g_hat, g>),   b_pred = ( mean_j|g_j|^2 · tr(H) / N + g_bar^T H g_bar ) / 2
     where mean_j|g_j|^2 = E[(delta_j / 2 sigma)^2] (raw-coefficient second moment: forward-only).
  5. bf16 diagnostic: fraction of the update norm that survives `p.add_(u.to(bf16))` on bf16
     weights (the training code's apply path) for each (N, eta).

  CUDA_VISIBLE_DEVICES=4 python scripts/local_stability.py \
      --pairs /data/liyan/runs/grzo-smoke/align/align_1p5b_base_pairs.pt \
      --out /data/liyan/runs/grzo-smoke/align/stability_1p5b_base.json
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from grad_alignment import batch_mean_logprob, load_vllm_ckpt_into_hf


class Model32:
    """fp32 model with a master copy; helpers to set theta = master + v and evaluate L."""
    def __init__(self, model, pairs, pad_id, device, chunk):
        self.model, self.pairs, self.pad, self.dev, self.chunk = model, pairs, pad_id, device, chunk
        self.params = [(n, p) for n, p in model.named_parameters()]
        self.master = [p.detach().clone() for _, p in self.params]
        self.adv = torch.tensor([p["adv"] for p in pairs], device=device)

    @torch.no_grad()
    def loss(self):
        lps = []
        for s in range(0, len(self.pairs), self.chunk):
            lps.append(batch_mean_logprob(self.model, self.pairs[s:s + self.chunk], self.dev, self.pad))
        lp = torch.cat(lps)
        return float((-(self.adv * lp)).mean())

    @torch.no_grad()
    def set_from_master(self, vec=None, scale=0.0):
        for (n, p), m in zip(self.params, self.master):
            if vec is None or scale == 0.0: p.copy_(m)
            else: p.copy_(m + scale * vec[n])

    @torch.no_grad()
    def set_from_seed(self, seed, scale):
        for (n, p), m in zip(self.params, self.master):
            gen = torch.Generator(device=p.device); gen.manual_seed(int(seed))
            p.copy_(m + scale * torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen))

    def grad(self):
        for _, p in self.params: p.requires_grad_(True); p.grad = None
        P = len(self.pairs)
        for s in range(0, P, self.chunk):
            ch = self.pairs[s:s + self.chunk]
            lp = batch_mean_logprob(self.model, ch, self.dev, self.pad, need_grad=True)
            adv = torch.tensor([p["adv"] for p in ch], device=self.dev)
            ((-(adv * lp)).sum() / P).backward()
        g = {n: p.grad.detach().clone() for n, p in self.params}
        for _, p in self.params: p.grad = None; p.requires_grad_(False)
        return g

    # ---- ZO scoring (per-example, raw coefficients), fp32 ----
    @torch.no_grad()
    def score(self, N, sigma, rng):
        order = rng.permutation(len(self.pairs)); idx = [int(order[j % len(self.pairs)]) for j in range(N)]
        seeds = [int(rng.integers(0, 2 ** 30)) for _ in range(N)]
        deltas = []
        for j in range(N):
            pr = self.pairs[idx[j]]
            self.set_from_seed(seeds[j], +sigma); lp_p = float(batch_mean_logprob(self.model, [pr], self.dev, self.pad)[0])
            self.set_from_seed(seeds[j], -sigma); lp_m = float(batch_mean_logprob(self.model, [pr], self.dev, self.pad)[0])
            deltas.append(-pr["adv"] * (lp_p - lp_m))
        self.set_from_master()
        return seeds, np.array(deltas)

    @torch.no_grad()
    def materialize(self, seeds, coeffs):
        acc = {n: torch.zeros_like(p) for n, p in self.params}
        for s, c in zip(seeds, coeffs):
            for n, p in self.params:
                gen = torch.Generator(device=p.device); gen.manual_seed(int(s))
                acc[n].add_(torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen), alpha=float(c))
        for n in acc: acc[n].div_(len(seeds))
        return acc


def vnorm2(g): return sum(float(torch.dot(t.view(-1), t.view(-1))) for t in g.values())
def vdot(a, b): return sum(float(torch.dot(a[n].view(-1), b[n].view(-1))) for n in a)


@torch.no_grad()
def bf16_survival(m32, vec, scale):
    """Training applies p_bf16.add_(u.to(bf16)). Fraction of |u| that survives rounding."""
    num = 0.0; den = 0.0
    for (n, p), m in zip(m32.params, m32.master):
        u = scale * vec[n]
        pb = m.to(torch.bfloat16)
        applied = (pb + u.to(torch.bfloat16)).float() - pb.float()
        num += float(torch.dot(applied.view(-1), applied.view(-1))); den += float(torch.dot(u.view(-1), u.view(-1)))
    return math.sqrt(num / (den + 1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--vllm-ckpt", default=None)
    ap.add_argument("--pairs", required=True, help="*_pairs.pt from grad_alignment.py (uses pairs0)")
    ap.add_argument("--sigma", type=float, default=1e-3)
    ap.add_argument("--n-list", default="64,96,256,512,1024")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--eta96", type=float, default=4e-5, help="grid centre at N=96; centre scales ∝ N")
    ap.add_argument("--eta-factors", default="0.125,0.25,0.5,1,2,4,8,16")
    ap.add_argument("--curv-samples", type=int, default=16)
    ap.add_argument("--curv-h", default="5e-4,1e-3")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = torch.device(os.environ.get("STAB_DEVICE", "cuda")); rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32).to(dev).eval()
    if args.vllm_ckpt: load_vllm_ckpt_into_hf(model, args.vllm_ckpt)
    pairs = torch.load(args.pairs)["pairs0"]
    m32 = Model32(model, pairs, tok.pad_token_id, dev, args.chunk)
    d = sum(p.numel() for _, p in m32.params)
    L0 = m32.loss()
    print(f"[STAB] model {args.model_name} d={d/1e6:.1f}M pairs={len(pairs)} L(theta)={L0:.6f}")

    # ---- true gradient and curvature along it ----
    t0 = time.time(); g = m32.grad(); gn2 = vnorm2(g)
    gu = {n: t / math.sqrt(gn2) for n, t in g.items()}
    h = 1e-3
    m32.set_from_master(gu, +h); Lp = m32.loss(); m32.set_from_master(gu, -h); Lm = m32.loss(); m32.set_from_master()
    gHg_unit = (Lp + Lm - 2 * L0) / h ** 2            # g^T H g / |g|^2
    print(f"[STAB] |g|^2={gn2:.4f}  g^T H g/|g|^2={gHg_unit:.4e}  ({time.time()-t0:.0f}s)")

    # ---- forward-only tr(H) via random directions ----
    trH = {}
    for hs in [float(x) for x in args.curv_h.split(",")]:
        vals = []
        for k in range(args.curv_samples):
            seed = 900000 + k
            m32.set_from_seed(seed, +hs); Lp = m32.loss(); m32.set_from_seed(seed, -hs); Lm = m32.loss()
            vals.append((Lp + Lm - 2 * L0) / hs ** 2)            # u^T H u, |u|^2 ≈ d
        m32.set_from_master()
        trH[hs] = (float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))))
        print(f"[STAB] tr(H) via h={hs:g}: {trH[hs][0]:.4e} ± {trH[hs][1]:.1e}  (uHu samples: min {min(vals):.3e} max {max(vals):.3e})")
    trH_est = trH[max(trH)][0]

    res = {"model": args.model_name, "ckpt": args.vllm_ckpt, "d": d, "pairs": len(pairs), "L0": L0,
           "g_norm2": gn2, "gHg_over_g2": gHg_unit, "trH": {str(k): v for k, v in trH.items()}, "rows": []}
    factors = [float(x) for x in args.eta_factors.split(",")]

    for N in [int(x) for x in args.n_list.split(",")]:
        t0 = time.time()
        eta_c = args.eta96 * N / 96.0
        etas = [eta_c * f for f in factors]
        dL = np.zeros((args.reps, len(etas))); surv = np.zeros((args.reps, len(etas)))
        c2, ghat_n2, dot_g = [], [], []
        for r in range(args.reps):
            seeds, deltas = m32.score(N, args.sigma, rng)
            coeffs = deltas / (2 * args.sigma)
            c2.append(float(np.mean(coeffs ** 2)))                 # forward-only mean_j |g_j|^2
            ghat = m32.materialize(seeds, coeffs)
            ghat_n2.append(vnorm2(ghat)); dot_g.append(vdot(ghat, g))
            for k, eta in enumerate(etas):
                m32.set_from_master(ghat, -eta); dL[r, k] = m32.loss() - L0
                surv[r, k] = bf16_survival(m32, ghat, -eta) if k in (0, 3, 6) else np.nan
            m32.set_from_master(); del ghat; (torch.cuda.empty_cache() if dev.type == 'cuda' else None)
        mean_dL = dL.mean(0)
        # quadratic fit dL = -a eta + b eta^2 on points with eta <= first positive-going region (all points)
        X = np.stack([-np.array(etas), np.array(etas) ** 2], 1)
        a, b = np.linalg.lstsq(X, mean_dL, rcond=None)[0]
        eta_opt_obs = a / (2 * b) if b > 0 else float("nan"); eta_max_obs = a / b if b > 0 else float("nan")
        best_k = int(np.argmin(mean_dL))
        # prediction (forward-only quantities): a_pred = |g_bar|^2 ≈ <g_hat,g>; b_pred = (c2 trH/N + gHg)/2
        gHg = gHg_unit * gn2
        a_pred = gn2; b_pred = 0.5 * (np.mean(c2) * trH_est / N + gHg)
        row = {"N": N, "etas": etas, "dL_mean": mean_dL.tolist(), "dL_std": dL.std(0).tolist(),
               "bf16_survival": np.nanmean(surv, 0).tolist(),
               "fit_a": float(a), "fit_b": float(b), "eta_opt_obs": float(eta_opt_obs), "eta_max_obs": float(eta_max_obs),
               "eta_best_grid": etas[best_k], "dL_best_grid": float(mean_dL[best_k]),
               "mean_gj2_forward": float(np.mean(c2)), "ghat_norm2_mean": float(np.mean(ghat_n2)),
               "dot_ghat_g_mean": float(np.mean(dot_g)),
               "a_pred": float(a_pred), "b_pred": float(b_pred),
               "eta_opt_pred": float(a_pred / (2 * b_pred)), "eta_max_pred": float(a_pred / b_pred),
               "t_s": time.time() - t0}
        res["rows"].append(row)
        print(f"[STAB] N={N:5d} | <ghat,g>={np.mean(dot_g):7.2f} (|g|^2={gn2:.2f}) | mean|g_j|^2(fwd)={np.mean(c2):8.2f} "
              f"| fit a={a:.3f} b={b:.3e} -> eta_opt={eta_opt_obs:.2e} eta_max={eta_max_obs:.2e} "
              f"| pred eta_opt={a_pred/(2*b_pred):.2e} eta_max={a_pred/b_pred:.2e} | best grid eta={etas[best_k]:.2e} dL={mean_dL[best_k]:+.4f} | {time.time()-t0:.0f}s")
        print("       dL(eta): " + " ".join(f"{e:.1e}:{v:+.4f}" for e, v in zip(etas, mean_dL)))
        print("       bf16 update survival at eta grid[0,3,6]: " + " ".join(f"{v:.2f}" for v in np.nanmean(surv, 0)[[0, 3, 6]]))
        json.dump(res, open(args.out, "w"), indent=2)
    print(f"[STAB] wrote {args.out}")


if __name__ == "__main__":
    main()
