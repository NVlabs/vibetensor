#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import sys
import time
from pathlib import Path

import numpy as np

import vibetensor
import vibetensor.torch as vt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.minigpt_shakespeare.common import (  # noqa: E402
    ShakespeareConfig,
    load_shakespeare_dataset,
    make_batch_numpy,
    write_metrics_json,
)
from examples.minigpt_shakespeare.vbt_io import (  # noqa: E402
    extract_vbt_weights,
    load_checkpoint,
    load_vbt_weights_,
    save_checkpoint,
)
from vibe_kernels.ops import Transformer  # noqa: E402


from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class VbtParamRef:
    name: str
    op: object
    key: str

    @property
    def weight(self):
        return self.op.weights[self.key]

    @property
    def grad(self):
        return self.op.grads.get(self.key)

    def set_weight(self, new_weight) -> None:
        self.op.weights[self.key] = new_weight

    @property
    def do_weight_decay(self) -> bool:
        return len(tuple(int(s) for s in self.weight.sizes)) >= 2


def iter_vbt_params(model: Transformer) -> list[VbtParamRef]:
    out: list[VbtParamRef] = []

    out.append(VbtParamRef("embedding.weight", model.embedding, "weight"))

    for i in range(int(model.n_layers)):
        blk = model.blocks[i]
        out.append(VbtParamRef(f"blocks.{i}.norm1.weight", blk.norm1, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.wq.weight", blk.wq, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.wk.weight", blk.wk, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.wv.weight", blk.wv, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.wo.weight", blk.wo, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.norm2.weight", blk.norm2, "weight"))
        out.append(VbtParamRef(f"blocks.{i}.ffn.w1", blk.ffn, "w1"))
        out.append(VbtParamRef(f"blocks.{i}.ffn.w2", blk.ffn, "w2"))

    out.append(VbtParamRef("final_norm.weight", model.final_norm, "weight"))
    out.append(VbtParamRef("lm_head.weight", model.lm_head, "weight"))

    return out


def vbt_zero_grads(model: Transformer) -> None:
    model.embedding.zero_grad()
    for blk in model.blocks:
        blk.norm1.zero_grad()
        blk.wq.zero_grad()
        blk.wk.zero_grad()
        blk.wv.zero_grad()
        blk.wo.zero_grad()
        blk.norm2.zero_grad()
        blk.ffn.zero_grad()
    model.final_norm.zero_grad()
    model.lm_head.zero_grad()


class VbtAdamW:
    def __init__(
        self,
        params: Iterable[VbtParamRef],
        *,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        grad_clamp: float | None = None,
    ) -> None:
        self.params = list(params)
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.grad_clamp = None if grad_clamp is None else float(grad_clamp)

        self.step_num = 0

        self._exp_avg: dict[str, object] = {}
        self._exp_avg_sq: dict[str, object] = {}

        for p in self.params:
            w = p.weight
            shape = [int(s) for s in w.sizes]
            self._exp_avg[p.name] = vt.zeros(shape).cuda()
            self._exp_avg_sq[p.name] = vt.zeros(shape).cuda()

    def step(self) -> None:
        self.step_num += 1

        beta1, beta2 = self.betas
        bias_correction1 = 1.0 - (beta1 ** int(self.step_num))
        bias_correction2 = 1.0 - (beta2 ** int(self.step_num))

        step_size = self.lr / bias_correction1
        inv_bias_correction2 = 1.0 / bias_correction2

        beta1_t = vt.full([1], beta1).cuda()
        one_minus_beta1_t = vt.full([1], 1.0 - beta1).cuda()
        beta2_t = vt.full([1], beta2).cuda()
        one_minus_beta2_t = vt.full([1], 1.0 - beta2).cuda()

        eps_t = vt.full([1], self.eps).cuda()
        neg_step_size_t = vt.full([1], -float(step_size)).cuda()
        inv_bc2_t = vt.full([1], float(inv_bias_correction2)).cuda()
        neg_lr_wd_t = vt.full([1], -float(self.lr * self.weight_decay)).cuda()

        for p in self.params:
            g = p.grad
            if g is None:
                continue

            if self.grad_clamp is not None and self.grad_clamp > 0:
                g = g.clamp(-float(self.grad_clamp), float(self.grad_clamp))

            exp_avg = self._exp_avg[p.name]
            exp_avg_sq = self._exp_avg_sq[p.name]

            exp_avg = exp_avg * beta1_t + g * one_minus_beta1_t
            g2 = g * g
            exp_avg_sq = exp_avg_sq * beta2_t + g2 * one_minus_beta2_t

            self._exp_avg[p.name] = exp_avg
            self._exp_avg_sq[p.name] = exp_avg_sq

            denom = (exp_avg_sq * inv_bc2_t).sqrt() + eps_t
            update = exp_avg * denom.reciprocal()

            w = p.weight
            if p.do_weight_decay and self.weight_decay != 0.0:
                w = w + w * neg_lr_wd_t

            w = w + update * neg_step_size_t
            p.set_weight(w)


def _vbt_to_numpy_cpu(vbt_tensor) -> np.ndarray:
    t_cpu = vbt_tensor.cpu()
    return np.from_dlpack(t_cpu).reshape(tuple(int(s) for s in t_cpu.sizes))


def _clip_grad_norm_(
    model: Transformer,
    max_norm: float,
    *,
    eps: float = 1e-6,
) -> None:
    """In-place global L2 grad norm clipping for VBT grads."""

    if not (float(max_norm) > 0.0):
        return

    grads: list[tuple[object, str, object]] = []
    total_sq = None

    for p in iter_vbt_params(model):
        g = p.grad
        if g is None:
            continue
        grads.append((p.op, p.key, g))
        sq = (g * g).sum()
        total_sq = sq if total_sq is None else (total_sq + sq)

    if total_sq is None:
        return

    max_norm_t = vt.full([1], float(max_norm)).cuda()
    eps_t = vt.full([1], float(eps)).cuda()
    total_norm = total_sq.sqrt()
    clip_coef = max_norm_t * (total_norm + eps_t).reciprocal()
    clip_coef = clip_coef.clamp(0.0, 1.0)

    for op, key, g in grads:
        op.grads[key] = g * clip_coef


def _init_vbt_weights(
    model: Transformer,
    *,
    vocab_size: int,
    cfg: ShakespeareConfig,
) -> None:
    """Initialize model weights from `cfg.seed`.

    Uses a normal distribution (mean=0, std=0.02) for weight matrices and ones
    for normalization scale vectors.
    """

    rng = np.random.default_rng(int(cfg.seed))

    def _normal(shape: tuple[int, ...], *, std: float = 0.02) -> np.ndarray:
        return rng.normal(loc=0.0, scale=float(std), size=shape).astype(np.float32)

    dim = int(cfg.dim)
    layers = int(cfg.layers)

    weights: dict[str, np.ndarray] = {}
    weights["embedding.weight"] = _normal((int(vocab_size), dim))

    for i in range(layers):
        weights[f"blocks.{i}.norm1.weight"] = np.ones((dim,), dtype=np.float32)
        weights[f"blocks.{i}.wq.weight"] = _normal((dim, dim))
        weights[f"blocks.{i}.wk.weight"] = _normal((dim, dim))
        weights[f"blocks.{i}.wv.weight"] = _normal((dim, dim))
        weights[f"blocks.{i}.wo.weight"] = _normal((dim, dim))
        weights[f"blocks.{i}.norm2.weight"] = np.ones((dim,), dtype=np.float32)
        weights[f"blocks.{i}.ffn.w1"] = _normal((dim, 4 * dim))
        weights[f"blocks.{i}.ffn.w2"] = _normal((4 * dim, dim))

    weights["final_norm.weight"] = np.ones((dim,), dtype=np.float32)
    weights["lm_head.weight"] = _normal((dim, int(vocab_size)))

    load_vbt_weights_(model, weights)


def greedy_generate_vbt(
    model: Transformer,
    prompt_ids: np.ndarray,
    *,
    max_new_tokens: int,
    max_seq_len: int,
) -> np.ndarray:
    seq = prompt_ids.astype(np.int64, copy=True)
    for _ in range(int(max_new_tokens)):
        ctx = seq[-int(max_seq_len) :]
        toks_full = np.empty((int(max_seq_len),), dtype=np.int64)
        toks_full[: ctx.shape[0]] = ctx
        if ctx.shape[0] < int(max_seq_len):
            toks_full[ctx.shape[0] :] = 0
        toks_v = vt.from_numpy(toks_full[None, :]).cuda()
        logits_v = model.fwd(toks_v)
        logits_np = _vbt_to_numpy_cpu(logits_v).reshape(1, int(max_seq_len), -1)
        last_pos = int(ctx.shape[0] - 1)
        next_id = int(np.argmax(logits_np[0, last_pos, :], axis=-1))
        seq = np.concatenate([seq, np.array([next_id], dtype=np.int64)], axis=0)
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--num-step", type=int, default=None, help="Number of training steps to run")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)

    # AdamW
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.1)

    # Gradient norm clipping. Set <=0 to disable.
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Elementwise grad clamp (NOT norm-clip). Disabled by default; set >0 to enable.
    ap.add_argument("--grad-clamp", type=float, default=0.0)

    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)

    ap.add_argument("--manual-attn-bwd", action="store_true", help="Use slow manual attention backward instead of Triton")

    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--metrics-json", type=str, default=None)

    ap.add_argument("--show-samples", action="store_true")
    ap.add_argument("--sample-len", type=int, default=200)
    ap.add_argument("--prompt", type=str, default="to be or")

    ap.add_argument("--ckpt-path", type=str, default=None)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None)

    args = ap.parse_args()

    cfg = ShakespeareConfig()
    if args.max_seq_len is not None:
        cfg = replace(cfg, max_seq_len=int(args.max_seq_len))
    if args.dim is not None:
        cfg = replace(cfg, dim=int(args.dim))
    if args.heads is not None:
        cfg = replace(cfg, heads=int(args.heads))
    if args.layers is not None:
        cfg = replace(cfg, layers=int(args.layers))

    iters = int(cfg.iters if args.iters is None else args.iters)
    if args.num_step is not None:
        iters = int(args.num_step)
    batch_size = int(cfg.batch_size if args.batch_size is None else args.batch_size)
    log_every = int(cfg.log_every if args.log_every is None else args.log_every)

    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        raise SystemExit("CUDA is required for VibeTensor")

    use_triton_attn_bwd = not bool(args.manual_attn_bwd)

    np.random.seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))
    vt.manual_seed(int(cfg.seed))

    dataset = load_shakespeare_dataset(data_dir=args.data_dir) if args.data_dir else load_shakespeare_dataset()
    vocab_size = int(dataset.vocab_size)
    split_note = "no split" if dataset.train is dataset.val else "train/val split"
    print(
        f"Shakespeare: vocab_size={vocab_size} train={dataset.train.shape[0]} val={dataset.val.shape[0]} ({split_note})"
    )

    model = Transformer(
        vocab_size=int(vocab_size),
        dim=int(cfg.dim),
        n_heads=int(cfg.heads),
        n_layers=int(cfg.layers),
        max_seq_len=int(cfg.max_seq_len),
        causal=True,
        use_triton_attn_bwd=bool(use_triton_attn_bwd),
    )

    # If we're not resuming, initialize weights deterministically from cfg.seed.
    if not args.resume_from:
        _init_vbt_weights(model, vocab_size=int(vocab_size), cfg=cfg)


    grad_clamp = None
    if float(args.grad_clamp) > 0:
        grad_clamp = float(args.grad_clamp)

    opt = VbtAdamW(
        iter_vbt_params(model),
        lr=float(args.lr),
        betas=(float(args.betas[0]), float(args.betas[1])),
        eps=float(args.eps),
        weight_decay=float(args.weight_decay),
        grad_clamp=grad_clamp,
    )

    start_step = 1
    if args.resume_from:
        ckpt = load_checkpoint(args.resume_from, map_location="cpu")
        if ckpt.get("format") != "vibetorch-minigpt-shakespeare-v1":
            raise SystemExit(f"Unexpected checkpoint format: {ckpt.get('format')}")

        ckpt_cfg = ckpt.get("config", {})
        # Basic config consistency checks.
        for k_name, want, got in (
            ("vocab_size", int(vocab_size), int(ckpt_cfg.get("vocab_size", -1))),
            ("max_seq_len", int(cfg.max_seq_len), int(ckpt_cfg.get("max_seq_len", -1))),
            ("dim", int(cfg.dim), int(ckpt_cfg.get("dim", -1))),
            ("heads", int(cfg.heads), int(ckpt_cfg.get("heads", -1))),
            ("layers", int(cfg.layers), int(ckpt_cfg.get("layers", -1))),
        ):
            if int(want) != int(got):
                raise SystemExit(f"Checkpoint {k_name} mismatch: ckpt={got} current={want}")

        load_vbt_weights_(model, ckpt["weights"])

        opt_state = ckpt.get("optimizer")
        if isinstance(opt_state, dict):
            opt.step_num = int(opt_state.get("step_num", int(ckpt.get("step", 0))))

            exp_avg = opt_state.get("exp_avg", {})
            exp_avg_sq = opt_state.get("exp_avg_sq", {})
            if isinstance(exp_avg, dict) and isinstance(exp_avg_sq, dict):
                def _to_f32_np(x: object) -> np.ndarray:
                    arr = np.asarray(x)
                    if arr.dtype == object:
                        raise TypeError(f"Unsupported optimizer state type: {type(x)}")
                    if arr.dtype != np.float32:
                        arr = arr.astype("float32")
                    return arr

                for name, t_cpu in exp_avg.items():
                    opt._exp_avg[name] = vt.from_numpy(_to_f32_np(t_cpu)).cuda()
                for name, t_cpu in exp_avg_sq.items():
                    opt._exp_avg_sq[name] = vt.from_numpy(_to_f32_np(t_cpu)).cuda()
        else:
            opt.step_num = int(ckpt.get("step", 0))

        rng_state = ckpt.get("rng_state")
        if isinstance(rng_state, dict):
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state

        start_step = int(opt.step_num) + 1
        print(f"Resumed from {args.resume_from}: step_num={int(opt.step_num)}", flush=True)

    run_meta = {
        "script": "examples/minigpt_shakespeare/train_vbt_shakespeare.py",
        "config": {
            "seed": int(cfg.seed),
            "vocab_size": int(vocab_size),
            "max_seq_len": int(cfg.max_seq_len),
            "dim": int(cfg.dim),
            "heads": int(cfg.heads),
            "layers": int(cfg.layers),
            "batch_size": int(batch_size),
            "optimizer": "AdamW",
            "lr": float(args.lr),
            "betas": [float(args.betas[0]), float(args.betas[1])],
            "eps": float(args.eps),
            "weight_decay": float(args.weight_decay),
            "grad_clip": float(args.grad_clip),
            "grad_clamp": None if grad_clamp is None else float(grad_clamp),
            "iters": int(iters),
            "log_every": int(log_every),
            "attn_bwd": "triton" if use_triton_attn_bwd else "manual",
            "init": "seeded_normal" if not args.resume_from else "resume",
            "show_samples": bool(args.show_samples),
        },
    }

    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    metrics_steps: list[dict] = []

    def _save_ckpt(step_completed: int) -> None:
        if not args.ckpt_path:
            return

        ckpt_base = Path(args.ckpt_path)

        # Optimizer state (CPU numpy arrays) so we can resume without restarting.
        exp_avg_cpu: dict[str, np.ndarray] = {}
        exp_avg_sq_cpu: dict[str, np.ndarray] = {}
        for name, t in opt._exp_avg.items():
            exp_avg_cpu[str(name)] = _vbt_to_numpy_cpu(t).copy()
        for name, t in opt._exp_avg_sq.items():
            exp_avg_sq_cpu[str(name)] = _vbt_to_numpy_cpu(t).copy()

        weights_cpu = extract_vbt_weights(model)

        ckpt = {
            "format": "vibetorch-minigpt-shakespeare-v1",
            "step": int(step_completed),
            "config": {
                "seed": int(cfg.seed),
                "vocab_size": int(vocab_size),
                "max_seq_len": int(cfg.max_seq_len),
                "dim": int(cfg.dim),
                "heads": int(cfg.heads),
                "layers": int(cfg.layers),
            },
            "tokenizer": dataset.tokenizer.to_config(),
            "weights": weights_cpu,
            "optimizer": {
                "step_num": int(opt.step_num),
                "lr": float(opt.lr),
                "betas": [float(opt.betas[0]), float(opt.betas[1])],
                "eps": float(opt.eps),
                "weight_decay": float(opt.weight_decay),
                "grad_clamp": None if opt.grad_clamp is None else float(opt.grad_clamp),
                "exp_avg": exp_avg_cpu,
                "exp_avg_sq": exp_avg_sq_cpu,
            },
            "rng_state": rng.bit_generator.state,
        }

        save_checkpoint(ckpt_base, ckpt)
        print(f"[ckpt] step={int(step_completed)} -> {ckpt_base}", flush=True)

    t0 = time.time()
    ckpt_every = max(0, int(args.ckpt_every)) if args.ckpt_path else 0
    for step in range(int(start_step), iters + 1):
        step_t0 = time.perf_counter()

        tokens_np, targets_np = make_batch_numpy(
            dataset.train,
            block_size=int(cfg.max_seq_len),
            batch_size=int(batch_size),
            rng=rng,
        )
        tokens_v = vt.from_numpy(tokens_np.astype(np.int64)).cuda()
        targets_v = vt.from_numpy(targets_np.astype(np.int64)).cuda()

        logits_v = model.fwd(tokens_v)
        loss_v = float(model.compute_loss(logits_v, targets_v))

        if not math.isfinite(loss_v):
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            logits_np = _vbt_to_numpy_cpu(logits_v)
            finite = np.isfinite(logits_np)
            finite_frac = float(finite.mean())
            max_abs = float(np.max(np.abs(logits_np))) if logits_np.size else float("nan")
            mn = float(np.min(logits_np)) if logits_np.size else float("nan")
            mx = float(np.max(logits_np)) if logits_np.size else float("nan")
            print(
                f"Non-finite loss at step {step}/{iters} | loss={loss_v} iter_ms={iter_ms:.2f}",
                flush=True,
            )
            print(
                f"  logits: finite_frac={finite_frac:.6f} min={mn:.6g} max={mx:.6g} max_abs={max_abs:.6g}",
                flush=True,
            )
            metrics_steps.append({"step": int(step), "loss": float(loss_v), "ppl": None, "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})
            break

        model.bwd()

        if float(args.grad_clip) > 0.0:
            _clip_grad_norm_(model, float(args.grad_clip))
        opt.step()
        vbt_zero_grads(model)

        if step % int(log_every) == 0 or step == 1 or step == iters:
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            ppl = float(math.exp(min(20.0, loss_v)))
            print(f"Step {step:4d}/{iters} | loss={loss_v:.4f} ppl={ppl:.2f} iter_ms={iter_ms:.2f}", flush=True)

            if bool(args.show_samples):
                prompt = args.prompt
                prompt_ids = dataset.tokenizer.encode(prompt)
                gen = greedy_generate_vbt(
                    model,
                    prompt_ids,
                    max_new_tokens=int(args.sample_len),
                    max_seq_len=int(cfg.max_seq_len),
                )
                print("--- sample ---")
                print(dataset.tokenizer.decode(gen))
                print("--------------", flush=True)

            metrics_steps.append({"step": int(step), "loss": float(loss_v), "ppl": ppl, "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})

        if args.ckpt_path and (step == iters or (ckpt_every > 0 and step % ckpt_every == 0)):
            _save_ckpt(int(step))

    print(f"Done in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
