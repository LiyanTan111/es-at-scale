import gc
import time
import torch
import random
import numpy as np


def _stateless_init_process_group(
    master_address, master_port, rank, world_size, device
):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, 
        port=master_port, 
        rank=rank, 
        world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)


class WorkerExtension:
    """
    The class for vLLM's worker to inherit from.
    """

    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def save_self_initial_weights(
        self,
    ):
        """Save a copy of itself in the CPU memory."""
        self.initial_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self.initial_weights[name] = p.detach().clone().cpu()
        print("Initial weights saved.")


    def restore_self_weights(self, seed, sigma):
        self._set_seed(seed)
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    
    def init_inter_engine_group(self, master_address: str, master_port: int,
                            engine_idx: int, num_engines: int):
            # TP rank within the engine (0..tp_size-1)
            tp_rank = getattr(self, "rank", None)
            tp_size = getattr(self, "world_size", None)

            if tp_rank is None or tp_size is None:
                tp_rank = getattr(self, "local_rank", 0)
                tp_size = getattr(getattr(self, "parallel_config", None), "tensor_parallel_size", None)
                if tp_size is None:
                    raise RuntimeError("Could not determine TP rank/size from vLLM worker attributes.")

            # IMPORTANT: create a group PER tp_rank across engines
            # So: rank is engine_idx, world_size is num_engines
            # Use a different port per tp_rank so each group has its own rendezvous
            port = int(master_port) + int(tp_rank)

            self.inter_pg = _stateless_init_process_group(
                master_address,
                port,
                rank=int(engine_idx),
                world_size=int(num_engines),
                device=self.device,
            )
            return True
    
    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(
                p, src=int(src_rank), stream=torch.cuda.current_stream()
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward

        for _, p in self.model_runner.model.named_parameters():
            # float32
            update_accumulator = torch.zeros_like(p.data, dtype=torch.float32)

            for i, (seed) in enumerate(seeds):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))

                # Generate noise (in native precision, usually float16/bfloat16)
                noise = torch.randn(
                    p.shape, dtype=p.dtype, device=p.device, generator=gen
                )

                # FIXED: Convert noise to float32 BEFORE multiplication.
                # Previous code: noise.to(torch.float16) * coeffs[i]
                # This caused the tiny update signal (1e-5) to be truncated by FP16 precision limits.
                term = noise.to(torch.float32) * coeffs[i]

                # Accumulate in FP32
                update_accumulator.add_(term)

            # multiply with scale (alpha/population_size) once at the end to preserve precision
            scale = float(alpha) / float(population_size)
            update_accumulator.mul_(scale)
            # Apply final update to weight (cast back to model dtype at the very end)
            p.data.add_(update_accumulator.to(p.dtype))

            del update_accumulator

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        """
        Add noise(seed) scaled by sigma_or_scale * coeff (or subtract when negate=True).
        - For exploration:  perturb_self_weights(seed, SIGMA, 1.0, False)
          and restore with restore_self_weights(seed, SIGMA) as before.
        - For ES update:   perturb_self_weights(seed, 1.0, coeff, False)
          where coeff = ALPHA/POPULATION_SIZE * norm_reward.
        """
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0

        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"Weights changed with: sign={sign}; scale={sign * scale}.")

    # ------------------------------------------------------------------ #
    # Master-copy perturbation (Angle B / GRZO-surrogate).
    #
    # In-place bf16 add/sub perturb-restore drifts weights by ~1 ulp per add
    # ((a+b)-b != a in floating point); measured drift was ~half the two-point
    # log-prob signal, which corrupts a small-signal ZO estimator. Instead keep
    # a GPU-resident master copy of theta: perturb = one deterministic rounding
    # from master, restore = bitwise-exact copy of master.
    #
    # Noise is generated in fp32 here AND in update_weights_from_seeds_fp32 --
    # the scoring-side and update-side noise must be the identical vector.
    # ------------------------------------------------------------------ #

    # FORGE_FP32_MASTER=1 (parking-lot B6, evidence METHOD_REVIEW R10): keep the master copy
    # in fp32 and apply updates to it, so sub-half-ulp components of an update are not
    # rounded away by the bf16 serving weights. Serving weights p are always master -> bf16.
    # Protocol per update: engine 0 update_weights_from_seeds_fp32 (master32 += delta,
    # p = master32) -> broadcast_all_weights (p) -> broadcast_master32 (fp32 master) ->
    # save_master_weights (no-op while the master is fresh). load_weights_from_disk marks the
    # master stale so the next save_master_weights rebuilds it from the loaded weights.
    @property
    def _fp32_master(self):
        import os as _os
        return _os.environ.get("FORGE_FP32_MASTER", "0") == "1"

    def save_master_weights(self):
        """Snapshot current weights as the GPU-resident master copy (bf16 clone, or fp32
        when FORGE_FP32_MASTER=1; in that mode an existing fresh master is kept)."""
        if self._fp32_master and getattr(self, "master_weights", None) is not None \
                and not getattr(self, "_master_stale", True):
            return True
        dtype = torch.float32 if self._fp32_master else None
        self.master_weights = {
            name: (p.detach().to(dtype).clone() if dtype else p.detach().clone())
            for name, p in self.model_runner.model.named_parameters()
        }
        self._master_stale = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def broadcast_master32(self, src_rank: int):
        """Broadcast the fp32 master copy across engines (FORGE_FP32_MASTER mode)."""
        if not self._fp32_master:
            return True
        for name, _ in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(
                self.master_weights[name], src=int(src_rank), stream=torch.cuda.current_stream()
            )
        self._master_stale = False
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    # fp32 chunk size for the elementwise apply loops below (64M elems = 256MB).
    # Whole-tensor `(master.to(fp32) + s*noise).to(bf16)` holds THREE fp32
    # temporaries of the largest layer at once (~7GB transient for a 7B embed),
    # which OOMs 40GB cards next to the vLLM reservation + master copy. The
    # noise tensor itself must stay whole (RNG sequence must match the update
    # side exactly), but the scale/add/round can run in chunks — the per-element
    # fp32 operations and their order are unchanged, so results are bitwise
    # identical to the whole-tensor expression.
    _APPLY_CHUNK = 1 << 26

    def perturb_from_master(self, seed, sigma, negate=False):
        """Set weights to master +/- sigma*eps(seed), computed in fp32 with a
        single rounding from master. No restore needed between jobs -- the next
        perturb_from_master overwrites from master again."""
        sign = -1.0 if negate else 1.0
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=torch.float32, device=p.device,
                                generator=gen)
            noise.mul_(sign * float(sigma))
            master = self.master_weights[name]
            if p.data.is_contiguous() and master.is_contiguous():
                p_flat, m_flat, n_flat = (p.data.view(-1), master.view(-1),
                                          noise.view(-1))
                for s in range(0, p_flat.numel(), self._APPLY_CHUNK):
                    e = min(s + self._APPLY_CHUNK, p_flat.numel())
                    tmp = m_flat[s:e].to(torch.float32)
                    tmp.add_(n_flat[s:e])
                    p_flat[s:e].copy_(tmp.to(p.dtype))
                    del tmp
            else:
                p.data.copy_((master.to(torch.float32) + noise).to(p.dtype))
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def restore_from_master(self):
        """Bitwise-exact restore of theta from the master copy."""
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self.master_weights[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def update_weights_from_seeds_fp32(self, seeds, coeffs, alpha, population_size):
        """Like update_weights_from_seeds, but noise is generated in fp32 so it
        matches perturb_from_master's noise exactly. theta += (alpha/N)*sum c_i*eps_i.
        Call save_master_weights afterwards (post-broadcast) to refresh masters."""
        for _name, p in self.model_runner.model.named_parameters():
            acc = torch.zeros_like(p.data, dtype=torch.float32)
            for i, seed in enumerate(seeds):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=torch.float32, device=p.device,
                                    generator=gen)
                # In-place scale: same fp32 multiply as `noise * c`, without the
                # third full-size temporary (bitwise-identical result).
                noise.mul_(float(coeffs[i]))
                acc.add_(noise)
                del noise
            acc.mul_(float(alpha) / float(population_size))
            if self._fp32_master and getattr(self, "master_weights", None) is not None:
                m = self.master_weights[_name]
                m.add_(acc)                      # exact fp32 accumulation
                p.data.copy_(m)                  # serving weights = master -> bf16
            elif p.data.is_contiguous():
                p_flat, a_flat = p.data.view(-1), acc.view(-1)
                for s in range(0, p_flat.numel(), self._APPLY_CHUNK):
                    e = min(s + self._APPLY_CHUNK, p_flat.numel())
                    p_flat[s:e].add_(a_flat[s:e].to(p.dtype))
            else:
                p.data.add_(acc.to(p.dtype))
            del acc
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def direction_stats(self, seeds_a, coeffs_a, seeds_b, coeffs_b):
        """E1.7 noise decomposition helper. For two seed/coefficient sets, form the
        (unscaled) update directions u_a = sum_i c_i eps_i and u_b = sum_j c_j eps_j
        with the SAME fp32 seed-regenerated noise as update_weights_from_seeds_fp32,
        and return their inner product and squared norms accumulated over all
        parameters (this TP rank's shard). cos(u_a, u_b) = dot / sqrt(na2 * nb2)
        measures how much of an update estimate is signal vs. direction noise.
        Never modifies weights."""
        dot = 0.0; na2 = 0.0; nb2 = 0.0
        for _, p in self.model_runner.model.named_parameters():
            ua = torch.zeros_like(p.data, dtype=torch.float32)
            for seed, c in zip(seeds_a, coeffs_a):
                gen = torch.Generator(device=p.device); gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
                noise.mul_(float(c)); ua.add_(noise); del noise
            ub = torch.zeros_like(p.data, dtype=torch.float32)
            for seed, c in zip(seeds_b, coeffs_b):
                gen = torch.Generator(device=p.device); gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=torch.float32, device=p.device, generator=gen)
                noise.mul_(float(c)); ub.add_(noise); del noise
            dot += float(torch.dot(ua.view(-1), ub.view(-1)).item())
            na2 += float(torch.dot(ua.view(-1), ua.view(-1)).item())
            nb2 += float(torch.dot(ub.view(-1), ub.view(-1)).item())
            del ua, ub
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return {"dot": dot, "na2": na2, "nb2": nb2}

    def _tp_rank_suffix(self):
        """Per-rank filename suffix under tensor parallelism.

        With TP>1 each worker holds only its parameter SHARD, and
        collective_rpc runs on every rank -- a shared filepath means ranks
        race on the same file (observed 2026-07-21: FileNotFoundError when
        rank B's os.replace ran after rank A consumed the tmp) and the
        surviving file is one rank's shard, not the model. Each rank gets its
        own file instead; resume re-loads shard-per-rank under the same TP
        layout. TP=1 keeps the legacy un-suffixed path.
        """
        try:
            from vllm.distributed import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )
            if get_tensor_model_parallel_world_size() > 1:
                return f".rank{get_tensor_model_parallel_rank()}"
        except Exception:
            pass
        return ""

    def save_self_weights_to_disk(self, filepath):
        """Save the current model weights to disk (atomically).

        Relay hops get SIGKILLed at the wall-time limit; a plain torch.save
        overwrite caught mid-write leaves a truncated file that poisons every
        subsequent resume (observed 2026-07-19: 335MB of 3.1GB). Write to a
        temp file and rename -- rename is atomic on the same filesystem.
        """
        import os as _os
        filepath = f"{filepath}{self._tp_rank_suffix()}"
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        tmp = f"{filepath}.tmp"
        torch.save(state_dict_to_save, tmp)
        _os.replace(tmp, filepath)
        print(f"Model weights saved to {filepath}.")

    def load_weights_from_disk(self, filepath):
        # map_location MUST be cpu: loading the full state dict straight to GPU
        # stacks a second copy of the weights on top of vLLM's reservation and
        # OOMs for 7B+ models (28GB vLLM + 14GB ckpt > 40GB). Stream per-param.
        filepath = f"{filepath}{self._tp_rank_suffix()}"
        state_dict = torch.load(filepath, map_location="cpu")
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(state_dict[name])
        self._master_stale = True
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
