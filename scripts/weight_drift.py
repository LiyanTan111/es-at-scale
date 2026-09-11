"""Measure how far a FORGE/ES checkpoint (vLLM layout .pth) has moved from the base HF weights:
per-parameter RMS and L2 of (theta_ckpt - theta_base), overall and by tensor family."""
import argparse, math, os, sys, re
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoModelForCausalLM

def fuse_hf_to_vllm(sd, cfg):
    hd = cfg.hidden_size // cfg.num_attention_heads
    out = {}
    layers = {}
    for k, v in sd.items():
        m = re.match(r"(model\.layers\.\d+\.)(self_attn\.(q|k|v)_proj|mlp\.(gate|up)_proj)\.(weight|bias)", k)
        if m:
            layers.setdefault((m.group(1), "qkv" if m.group(3) else "gate_up", m.group(5)), {})[m.group(3) or m.group(4)] = v
        else:
            out[k] = v
    for (pre, kind, wb), parts in layers.items():
        if kind == "qkv": out[f"{pre}self_attn.qkv_proj.{wb}"] = torch.cat([parts["q"], parts["k"], parts["v"]], 0)
        else: out[f"{pre}mlp.gate_up_proj.{wb}"] = torch.cat([parts["gate"], parts["up"]], 0)
    return out

ap = argparse.ArgumentParser(); ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct"); ap.add_argument("ckpts", nargs="+")
args = ap.parse_args()
base = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.bfloat16)
bsd = fuse_hf_to_vllm({k: v for k, v in base.state_dict().items()}, base.config)
for ck in args.ckpts:
    sd = torch.load(ck, map_location="cpu")
    tot2 = 0.0; n = 0; fam = {}
    for k, v in sd.items():
        b = bsd.get(k)
        if b is None:
            if k == "lm_head.weight" and "model.embed_tokens.weight" in bsd: b = bsd["model.embed_tokens.weight"]
            else: print("  (no base match:", k, ")"); continue
        d = (v.float() - b.float()); s2 = float((d * d).sum()); tot2 += s2; n += d.numel()
        f = "embed" if "embed" in k or "lm_head" in k else ("norm" if "norm" in k else ("attn" if "attn" in k else ("mlp" if "mlp" in k else "other")))
        fam.setdefault(f, [0.0, 0])[0] += s2; fam[f][1] += d.numel()
    print(f"{ck}\n  overall: L2={math.sqrt(tot2):.2f}  RMS/param={math.sqrt(tot2/n):.3e}  (n={n/1e6:.0f}M)")
    for f, (s2, m) in fam.items(): print(f"    {f:6s} RMS={math.sqrt(s2/m):.3e}")
