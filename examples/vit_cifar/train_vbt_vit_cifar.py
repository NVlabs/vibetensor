#!/usr/bin/env python3

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

import os

import vibetensor
import vibetensor.torch as vt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.vit_cifar.common import (
    VitCifarConfig as Config,
    cifar10_normalize_,
    cosine_lr,
    iter_minibatches,
    load_cifar10_numpy,
    num_minibatches,
    resolve_training_limits,
    write_metrics_json,
)

from vibe_kernels.ops import RMSNorm, Linear, Attention
from vibe_kernels.activation import vbt_native as act_kernel
from vibe_kernels.loss import vbt_native as loss_kernel
from vibe_kernels.tensor_ops import vbt_native as tensor_ops
from vibe_kernels.indexing.vbt_native import argmax as vbt_argmax


class VbtAdamW:
    def __init__(
        self,
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr = float(lr)
        self.betas = (float(betas[0]), float(betas[1]))
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_num = 0
        self._exp_avg: dict[str, object] = {}
        self._exp_avg_sq: dict[str, object] = {}

    @staticmethod
    def _do_weight_decay(w) -> bool:
        # Apply weight decay to matrices/embeddings; skip 1D norm-like params.
        return len(tuple(int(s) for s in w.sizes)) >= 2

    def step(self, named_params_and_grads) -> None:
        self.step_num += 1
        beta1, beta2 = self.betas

        bias_correction1 = 1.0 - (beta1 ** int(self.step_num))
        bias_correction2 = 1.0 - (beta2 ** int(self.step_num))
        step_size = self.lr / bias_correction1
        inv_bc2 = 1.0 / bias_correction2

        beta1_t = vt.full([1], float(beta1)).cuda()
        one_minus_beta1_t = vt.full([1], float(1.0 - beta1)).cuda()
        beta2_t = vt.full([1], float(beta2)).cuda()
        one_minus_beta2_t = vt.full([1], float(1.0 - beta2)).cuda()

        eps_t = vt.full([1], float(self.eps)).cuda()
        neg_step_size_t = vt.full([1], -float(step_size)).cuda()
        inv_bc2_t = vt.full([1], float(inv_bc2)).cuda()
        neg_lr_wd_t = vt.full([1], -float(self.lr * self.weight_decay)).cuda()

        for name, w, g in named_params_and_grads:
            if g is None:
                continue

            n = str(name)
            exp_avg = self._exp_avg.get(n)
            exp_avg_sq = self._exp_avg_sq.get(n)
            if exp_avg is None or exp_avg_sq is None:
                shape = [int(s) for s in w.sizes]
                exp_avg = vt.zeros(shape).cuda()
                exp_avg_sq = vt.zeros(shape).cuda()

            exp_avg = exp_avg * beta1_t + g * one_minus_beta1_t
            g2 = g * g
            exp_avg_sq = exp_avg_sq * beta2_t + g2 * one_minus_beta2_t

            self._exp_avg[n] = exp_avg
            self._exp_avg_sq[n] = exp_avg_sq

            denom = (exp_avg_sq * inv_bc2_t).sqrt() + eps_t
            update = exp_avg * denom.reciprocal()

            if self.weight_decay != 0.0 and self._do_weight_decay(w):
                w.add_(w * neg_lr_wd_t)

            w.add_(update * neg_step_size_t)


def numpy_to_vbt(arr, device="cuda:0"):
    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    t = vt.from_numpy(arr)
    return t.cuda() if device.startswith("cuda") else t


def vbt_to_numpy(tensor):
    return np.from_dlpack(tensor.cpu()).reshape(tuple(int(s) for s in tensor.sizes))


def _cat_dim1(a, b):
    a_t = a.permute([1, 0, 2]).contiguous()
    b_t = b.permute([1, 0, 2]).contiguous()
    out_t = tensor_ops.cat([a_t, b_t], dim=0)
    return out_t.permute([1, 0, 2]).contiguous()


class CrossEntropyLoss:
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = int(ignore_index)
        self._cache = None

    def fwd(self, logits, targets):
        loss_val, cache = loss_kernel.cross_entropy_with_cache(logits, targets, ignore_index=self.ignore_index)
        self._cache = cache
        return float(loss_val)

    def bwd(self):
        cache = self._cache
        if cache is None:
            raise RuntimeError("CrossEntropy backward called before forward")
        return loss_kernel.cross_entropy_backward(cache, 1.0)


class MLP:

    def __init__(self, dim: int, hidden_dim: int):
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.fc1 = Linear(self.dim, self.hidden_dim)
        self.fc2 = Linear(self.hidden_dim, self.dim)

        self._h1_pre = None

    def zero_grad(self):
        self.fc1.zero_grad()
        self.fc2.zero_grad()

    def fwd(self, x, *, train: bool):
        h1_pre = self.fc1.fwd(x)
        self._h1_pre = h1_pre
        h1_act = act_kernel.gelu(h1_pre)
        return self.fc2.fwd(h1_act)

    def bwd(self, grad_out):
        dh1_act = self.fc2.bwd(grad_out)
        dh1 = act_kernel.gelu_backward(dh1_act, self._h1_pre)
        return self.fc1.bwd(dh1)


class ViTBlock:
    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_dim: int,
    ):
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads

        self.norm1 = RMSNorm(self.dim)
        self.wq = Linear(self.dim, self.dim)
        self.wk = Linear(self.dim, self.dim)
        self.wv = Linear(self.dim, self.dim)
        self.attn = Attention(self.heads, self.head_dim, causal=False)
        self.wo = Linear(self.dim, self.dim)

        self.norm2 = RMSNorm(self.dim)
        self.mlp = MLP(self.dim, mlp_dim)

        self._b = 0
        self._t = 0

    def zero_grad(self):
        for op in (self.norm1, self.wq, self.wk, self.wv, self.wo, self.norm2):
            op.zero_grad()
        self.mlp.zero_grad()

    def fwd(self, x, *, train: bool):
        b, t, d = (int(x.sizes[0]), int(x.sizes[1]), int(x.sizes[2]))
        self._b = b
        self._t = t

        x2 = x.reshape([b * t, d])
        x_norm = self.norm1.fwd(x2)

        q = self.wq.fwd(x_norm).reshape([b, t, self.heads, self.head_dim]).permute([0, 2, 1, 3])
        k = self.wk.fwd(x_norm).reshape([b, t, self.heads, self.head_dim]).permute([0, 2, 1, 3])
        v = self.wv.fwd(x_norm).reshape([b, t, self.heads, self.head_dim]).permute([0, 2, 1, 3])

        attn_out = self.attn.fwd(q, k, v)
        attn_flat = attn_out.permute([0, 2, 1, 3]).reshape([b * t, d])
        o = self.wo.fwd(attn_flat).reshape([b, t, d])
        x = x + o

        x2 = x.reshape([b * t, d])
        x_norm2 = self.norm2.fwd(x2)
        h2 = self.mlp.fwd(x_norm2, train=train).reshape([b, t, d])
        return x + h2

    def bwd(self, grad_out):
        b, t, d = self._b, self._t, self.dim

        dx = grad_out

        d_h2_flat = dx.reshape([b * t, d])
        d_norm2 = self.mlp.bwd(d_h2_flat)
        d_pre_norm2 = self.norm2.bwd(d_norm2).reshape([b, t, d])
        dx = dx + d_pre_norm2

        d_o_flat = dx.reshape([b * t, d])
        d_attn_flat = self.wo.bwd(d_o_flat)
        d_attn = d_attn_flat.reshape([b, t, self.heads, self.head_dim]).permute([0, 2, 1, 3])

        dq, dk, dv = self.attn.bwd(d_attn)
        dq_flat = dq.permute([0, 2, 1, 3]).reshape([b * t, d])
        dk_flat = dk.permute([0, 2, 1, 3]).reshape([b * t, d])
        dv_flat = dv.permute([0, 2, 1, 3]).reshape([b * t, d])

        dx_qkv = self.wq.bwd(dq_flat)
        dx_qkv = dx_qkv + self.wk.bwd(dk_flat)
        dx_qkv = dx_qkv + self.wv.bwd(dv_flat)

        d_pre_norm1 = self.norm1.bwd(dx_qkv).reshape([b, t, d])
        return dx + d_pre_norm1


class ViTVBT:
    def __init__(
        self,
        *,
        patch_size: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        dropout: float,
        emb_dropout: float,
        num_classes: int,
        n_cls_tokens: int = 16,
        seed: int,
        rng: np.random.Generator,
    ):
        self.patch_size = int(patch_size)
        self.dim = int(dim)
        self.depth = int(depth)
        self.heads = int(heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout_p = float(dropout)
        self.emb_dropout_p = float(emb_dropout)
        self.num_classes = int(num_classes)
        self.n_cls_tokens = int(n_cls_tokens)

        self.grid = 32 // self.patch_size
        self.n_patches = self.grid * self.grid
        self.patch_dim = 3 * self.patch_size * self.patch_size

        vt.manual_seed(int(seed))

        self.patch_proj = Linear(self.patch_dim, self.dim)
        self.cls_token = (vt.randn([1, self.n_cls_tokens, self.dim]) * 0.02).cuda()
        self.pos_embedding = (vt.randn([1, self.n_patches + self.n_cls_tokens, self.dim]) * 0.02).cuda()

        self.blocks = [ViTBlock(self.dim, self.heads, self.mlp_dim) for _ in range(self.depth)]

        self.final_norm = RMSNorm(self.dim)
        self.head = Linear(self.dim, self.num_classes)

        self.loss_fn = CrossEntropyLoss()

        cls_mask = np.zeros([self.n_patches + self.n_cls_tokens], dtype=np.float32)
        cls_mask[: self.n_cls_tokens] = 1.0 / max(1, self.n_cls_tokens)
        self._cls_mask = numpy_to_vbt(cls_mask, "cuda:0")

        self._cached_tokens = None
        self._cached_b = 0

        self._grad_cls = None
        self._grad_pos = None

    def zero_grad(self):
        self.patch_proj.zero_grad()
        for blk in self.blocks:
            blk.zero_grad()
        self.final_norm.zero_grad()
        self.head.zero_grad()
        self._grad_cls = None
        self._grad_pos = None

    def patchify(self, x):
        b, c, h, w = (int(x.sizes[0]), int(x.sizes[1]), int(x.sizes[2]), int(x.sizes[3]))
        ps = self.patch_size
        gh = h // ps
        gw = w // ps
        x = x.reshape([b, c, gh, ps, gw, ps]).permute([0, 2, 4, 1, 3, 5])
        return x.reshape([b, gh * gw, c * ps * ps])

    def fwd_logits(self, x_img, *, train: bool):
        b = int(x_img.sizes[0])
        n = self.n_patches
        d = self.dim

        patches = self.patchify(x_img)
        p_flat = patches.reshape([b * n, self.patch_dim])
        x = self.patch_proj.fwd(p_flat).reshape([b, n, d])

        cls = self.cls_token.expand([b, self.n_cls_tokens, d])
        x = _cat_dim1(cls, x)

        pos = self.pos_embedding.expand([1, n + self.n_cls_tokens, d]).expand([b, n + self.n_cls_tokens, d])
        x = x + pos

        for blk in self.blocks:
            x = blk.fwd(x, train=train)

        self._cached_tokens = x
        self._cached_b = b

        cls_tok = x.narrow(1, 0, self.n_cls_tokens).sum(dim=1) * vt.full([1], 1.0 / max(1, self.n_cls_tokens)).cuda()
        cls_norm = self.final_norm.fwd(cls_tok)
        return self.head.fwd(cls_norm)

    def loss_fwd(self, logits, targets):
        return self.loss_fn.fwd(logits, targets)

    def bwd(self):
        dlogits = self.loss_fn.bwd()

        b = int(self._cached_b)
        d = self.dim
        t = self.n_patches + self.n_cls_tokens

        dcls_norm = self.head.bwd(dlogits)
        dcls = self.final_norm.bwd(dcls_norm)

        mask = self._cls_mask.unsqueeze(0).unsqueeze(2).expand([b, t, d])
        dx = dcls.unsqueeze(1).expand([b, t, d]) * mask

        for blk in reversed(self.blocks):
            dx = blk.bwd(dx)

        self._grad_pos = dx.sum(dim=0, keepdim=True)
        dcls_tok = dx.narrow(1, 0, self.n_cls_tokens)
        self._grad_cls = dcls_tok.sum(dim=0, keepdim=True)

        dpatch = dx.narrow(1, self.n_cls_tokens, self.n_patches).reshape([b * self.n_patches, d])
        _ = self.patch_proj.bwd(dpatch)

    def iter_params_and_grads(self):
        yield self.cls_token, self._grad_cls
        yield self.pos_embedding, self._grad_pos

        for _name, w in self.patch_proj.weights.items():
            yield w, self.patch_proj.grads.get(_name)

        for blk in self.blocks:
            for op in (blk.norm1, blk.wq, blk.wk, blk.wv, blk.wo, blk.norm2):
                for _name, w in op.weights.items():
                    yield w, op.grads.get(_name)
            for op in (blk.mlp.fc1, blk.mlp.fc2):
                for _name, w in op.weights.items():
                    yield w, op.grads.get(_name)

        for _name, w in self.final_norm.weights.items():
            yield w, self.final_norm.grads.get(_name)

        for _name, w in self.head.weights.items():
            yield w, self.head.grads.get(_name)

    def iter_named_params_and_grads(self):
        yield "cls_token", self.cls_token, self._grad_cls
        yield "pos_embedding", self.pos_embedding, self._grad_pos

        for k in sorted(self.patch_proj.weights.keys()):
            yield f"patch_proj.{k}", self.patch_proj.weights[k], self.patch_proj.grads.get(k)

        for i, blk in enumerate(self.blocks):
            for op_name, op in (
                ("norm1", blk.norm1),
                ("wq", blk.wq),
                ("wk", blk.wk),
                ("wv", blk.wv),
                ("wo", blk.wo),
                ("norm2", blk.norm2),
            ):
                for k in sorted(op.weights.keys()):
                    yield f"blocks.{i}.{op_name}.{k}", op.weights[k], op.grads.get(k)

            for op_name, op in (("mlp.fc1", blk.mlp.fc1), ("mlp.fc2", blk.mlp.fc2)):
                for k in sorted(op.weights.keys()):
                    yield f"blocks.{i}.{op_name}.{k}", op.weights[k], op.grads.get(k)

        for k in sorted(self.final_norm.weights.keys()):
            yield f"final_norm.{k}", self.final_norm.weights[k], self.final_norm.grads.get(k)

        for k in sorted(self.head.weights.keys()):
            yield f"head.{k}", self.head.weights[k], self.head.grads.get(k)

    def load_weights_(self, weights: dict[str, np.ndarray]) -> None:
        self.cls_token = vt.from_numpy(weights["cls_token"].astype(np.float32)).cuda()
        self.pos_embedding = vt.from_numpy(weights["pos_embedding"].astype(np.float32)).cuda()

        for k in sorted(self.patch_proj.weights.keys()):
            arr = weights[f"patch_proj.{k}"].astype(np.float32)
            self.patch_proj.weights[k] = vt.from_numpy(arr).cuda()

        for i, blk in enumerate(self.blocks):
            for op_name, op in (
                ("norm1", blk.norm1),
                ("wq", blk.wq),
                ("wk", blk.wk),
                ("wv", blk.wv),
                ("wo", blk.wo),
                ("norm2", blk.norm2),
            ):
                for k in sorted(op.weights.keys()):
                    arr = weights[f"blocks.{i}.{op_name}.{k}"].astype(np.float32)
                    op.weights[k] = vt.from_numpy(arr).cuda()

            for op_name, op in (("mlp.fc1", blk.mlp.fc1), ("mlp.fc2", blk.mlp.fc2)):
                for k in sorted(op.weights.keys()):
                    arr = weights[f"blocks.{i}.{op_name}.{k}"].astype(np.float32)
                    op.weights[k] = vt.from_numpy(arr).cuda()

        for k in sorted(self.final_norm.weights.keys()):
            arr = weights[f"final_norm.{k}"].astype(np.float32)
            self.final_norm.weights[k] = vt.from_numpy(arr).cuda()

        for k in sorted(self.head.weights.keys()):
            arr = weights[f"head.{k}"].astype(np.float32)
            self.head.weights[k] = vt.from_numpy(arr).cuda()


def _save_checkpoint(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def _load_checkpoint(path: str | Path) -> dict:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--metrics-json", type=str, default=None)
    ap.add_argument("--num-step", type=int, default=None, help="Run this many training steps then exit")
    ap.add_argument("--ckpt-path", type=str, default=None)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--resume-from", type=str, default=None)
    args, _unknown = ap.parse_known_args()

    cfg = Config()

    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        print("ERROR: CUDA not available for VibeTensor", flush=True)
        sys.exit(1)

    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    patch_size = int(cfg.patch_size)
    dim = int(cfg.dim)
    depth = int(cfg.depth)
    heads = int(cfg.heads)
    mlp_dim = int(cfg.dim) * int(cfg.mlp_ratio)
    dropout = float(cfg.dropout)
    emb_dropout = float(cfg.emb_dropout)

    model = ViTVBT(
        patch_size=patch_size,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
        num_classes=cfg.num_classes,
        seed=cfg.seed,
        rng=rng,
    )
    use_triton_bwd = True
    for blk in model.blocks:
        blk.attn.use_triton_bwd = use_triton_bwd

    train_x_raw, train_y, test_x_raw, test_y = load_cifar10_numpy(cfg.data_root)
    print(f"Train: {train_x_raw.shape}, Test: {test_x_raw.shape}", flush=True)
    opt = VbtAdamW(lr=float(cfg.lr), betas=(0.9, 0.999), eps=1e-8, weight_decay=float(cfg.weight_decay))

    max_epochs, max_batches, sched_epochs = resolve_training_limits(epochs=cfg.epochs)
    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    metrics_epochs: list[dict] = []
    run_meta = {
        "script": "examples/vit_cifar/train_vbt_vit_cifar.py",
        "config": {
            "seed": int(cfg.seed),
            "batch_size": int(cfg.batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "weight_decay": float(cfg.weight_decay),
            "patch_size": int(cfg.patch_size),
            "dim": int(cfg.dim),
            "depth": int(cfg.depth),
            "heads": int(cfg.heads),
            "mlp_ratio": int(cfg.mlp_ratio),
            "dropout": float(cfg.dropout),
            "emb_dropout": float(cfg.emb_dropout),
            "num_classes": int(cfg.num_classes),
        },
        "limits": {"max_epochs": int(max_epochs), "max_batches": (int(max_batches) if max_batches is not None else None)},
        "use_triton_bwd": bool(use_triton_bwd),
    }

    print(
        f"Config: patch={patch_size} dim={dim} depth={depth} heads={heads} mlp_dim={mlp_dim} "
        f"dropout={dropout} emb_dropout={emb_dropout} batch={cfg.batch_size} epochs={max_epochs} "
        f"lr={cfg.lr} wd={cfg.weight_decay}",
        flush=True,
    )

    ckpt_path = Path(args.ckpt_path) if args.ckpt_path else None
    ckpt_every = int(args.ckpt_every) if ckpt_path is not None else 0

    start_epoch = 1
    steps_done = 0
    if args.resume_from:
        ckpt = _load_checkpoint(args.resume_from)
        if ckpt.get("format") != "vibetorch-vit-cifar-v1":
            raise SystemExit(f"Unexpected checkpoint format: {ckpt.get('format')}")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        steps_done = int(ckpt.get("steps_done", 0))

        rng_state = ckpt.get("rng_state")
        if isinstance(rng_state, dict):
            rng = np.random.default_rng()
            rng.bit_generator.state = rng_state

        model.load_weights_(ckpt["weights"])

        opt_state = ckpt.get("optimizer", {})
        opt.step_num = int(opt_state.get("step_num", 0))
        exp_avg = opt_state.get("exp_avg", {})
        exp_avg_sq = opt_state.get("exp_avg_sq", {})
        if isinstance(exp_avg, dict):
            for name, arr in exp_avg.items():
                if isinstance(arr, np.ndarray):
                    opt._exp_avg[str(name)] = vt.from_numpy(arr.astype(np.float32)).cuda()
        if isinstance(exp_avg_sq, dict):
            for name, arr in exp_avg_sq.items():
                if isinstance(arr, np.ndarray):
                    opt._exp_avg_sq[str(name)] = vt.from_numpy(arr.astype(np.float32)).cuda()

        print(f"[resume] epoch={start_epoch} steps_done={steps_done}", flush=True)

    t0 = time.time()
    target_steps = None if args.num_step is None else int(args.num_step)
    tqdm_disable = bool(int(os.environ.get("TQDM_DISABLE", "0")))
    stop_training = False
    for epoch in range(int(start_epoch), max_epochs + 1):
        lr = cosine_lr(cfg.lr, epoch, sched_epochs)
        opt.lr = float(lr)

        epoch_t0 = time.time()
        print(f"Epoch {epoch}/{max_epochs} (lr={lr:.6g})", flush=True)

        train_loss_sum = 0.0
        train_loss_batches = 0
        train_correct = 0
        train_total = 0

        total_batches = num_minibatches(int(train_x_raw.shape[0]), cfg.batch_size)
        if max_batches is not None:
            total_batches = min(total_batches, int(max_batches))

        b_i = 0
        train_iter = iter_minibatches(train_x_raw, train_y, cfg.batch_size, rng, augment=True)
        for xb, yb in tqdm(
            train_iter,
            total=total_batches,
            desc=f"Train {epoch}/{max_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=tqdm_disable,
        ):
            b_i += 1
            if max_batches is not None and b_i > max_batches:
                break

            cifar10_normalize_(xb)
            x = numpy_to_vbt(xb.astype(np.float32), "cuda:0")
            y = numpy_to_vbt(yb.astype(np.int64), "cuda:0").reshape([-1])

            model.zero_grad()

            logits = model.fwd_logits(x, train=True)
            loss = model.loss_fwd(logits, y)
            if not (loss == loss):
                print(f"  NaN loss at batch {b_i}", flush=True)
                break

            model.bwd()
            opt.step(model.iter_named_params_and_grads())
            steps_done += 1
            if target_steps is not None and steps_done >= target_steps:
                print(
                    f"num_step={target_steps} reached; stopping early | last_loss={float(loss):.4f}",
                    flush=True,
                )
                stop_training = True
                break

            train_loss_sum += float(loss)
            train_loss_batches += 1
            preds = vbt_argmax(logits, dim=1)
            preds_np = vbt_to_numpy(preds)
            train_correct += int((preds_np == yb).sum())
            train_total += int(yb.shape[0])

        test_correct = 0
        test_total = 0
        test_loss_sum = 0.0
        test_loss_batches = 0

        total_test_batches = num_minibatches(int(test_x_raw.shape[0]), cfg.eval_batch_size)
        test_iter = iter_minibatches(test_x_raw, test_y, cfg.eval_batch_size, rng, augment=False)
        for xb, yb in tqdm(
            test_iter,
            total=total_test_batches,
            desc=f"Test  {epoch}/{max_epochs}",
            unit="batch",
            dynamic_ncols=True,
            leave=False,
            disable=tqdm_disable,
        ):
            cifar10_normalize_(xb)
            x = numpy_to_vbt(xb.astype(np.float32), "cuda:0")
            y = numpy_to_vbt(yb.astype(np.int64), "cuda:0").reshape([-1])

            logits = model.fwd_logits(x, train=False)
            loss = model.loss_fwd(logits, y)
            test_loss_sum += float(loss)
            test_loss_batches += 1

            preds = vbt_argmax(logits, dim=1)
            preds_np = vbt_to_numpy(preds)
            test_correct += int((preds_np == yb).sum())
            test_total += int(yb.shape[0])

        train_acc = train_correct / max(1, train_total)
        test_acc = test_correct / max(1, test_total)
        train_loss = train_loss_sum / max(1, train_loss_batches)
        test_loss = test_loss_sum / max(1, test_loss_batches)
        epoch_dt = time.time() - epoch_t0

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}"
            f" | Epoch time: {epoch_dt:.2f}s",
            flush=True,
        )

        metrics_epochs.append(
            {
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "test_loss": float(test_loss),
                "test_acc": float(test_acc),
                "epoch_time_s": float(epoch_dt),
            }
        )
        if metrics_path is not None:
            write_metrics_json(metrics_path, {"run": run_meta, "epochs": metrics_epochs})

        if ckpt_path is not None and (ckpt_every > 0) and (epoch % ckpt_every == 0 or epoch == max_epochs):
            weights_cpu: dict[str, np.ndarray] = {}
            for name, w, _g in model.iter_named_params_and_grads():
                weights_cpu[str(name)] = vbt_to_numpy(w).astype(np.float32, copy=True)

            exp_avg_cpu: dict[str, np.ndarray] = {}
            exp_avg_sq_cpu: dict[str, np.ndarray] = {}
            for name, t in opt._exp_avg.items():
                exp_avg_cpu[str(name)] = vbt_to_numpy(t).astype(np.float32, copy=True)
            for name, t in opt._exp_avg_sq.items():
                exp_avg_sq_cpu[str(name)] = vbt_to_numpy(t).astype(np.float32, copy=True)

            _save_checkpoint(
                ckpt_path,
                {
                    "format": "vibetorch-vit-cifar-v1",
                    "epoch": int(epoch),
                    "steps_done": int(steps_done),
                    "rng_state": rng.bit_generator.state,
                    "weights": weights_cpu,
                    "optimizer": {
                        "step_num": int(opt.step_num),
                        "exp_avg": exp_avg_cpu,
                        "exp_avg_sq": exp_avg_sq_cpu,
                    },
                },
            )
            print(f"[ckpt] epoch={int(epoch)} -> {ckpt_path}", flush=True)

        if stop_training:
            break

    dt = time.time() - t0
    print(f"Final Test Acc: {test_acc:.4f}", flush=True)
    print(f"Training time: {dt:.2f}s ({max_epochs} epochs)", flush=True)


if __name__ == "__main__":
    main()
