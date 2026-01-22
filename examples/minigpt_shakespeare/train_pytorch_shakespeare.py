#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import replace
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.minigpt_shakespeare.common import (  # noqa: E402
    ShakespeareConfig,
    load_shakespeare_dataset,
    make_batch_numpy,
    write_metrics_json,
)


def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    w32 = weight.float()
    inv_rms = torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x32 * inv_rms * w32


def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, s, d = q.shape
    hd2 = d // 2
    pos = torch.arange(s, device=q.device, dtype=torch.long).view(1, 1, s).expand(b, h, s)
    cos_e = cos[pos]
    sin_e = sin[pos]

    q1, q2 = q[..., :hd2], q[..., hd2:]
    k1, k2 = k[..., :hd2], k[..., hd2:]

    q_out1 = q1 * cos_e + q2 * sin_e
    q_out2 = -q1 * sin_e + q2 * cos_e
    k_out1 = k1 * cos_e + k2 * sin_e
    k_out2 = -k1 * sin_e + k2 * cos_e

    return torch.cat([q_out1, q_out2], dim=-1), torch.cat([k_out1, k_out2], dim=-1)


def _init_normal_(p: torch.Tensor, *, std: float = 0.02) -> None:
    with torch.no_grad():
        p.normal_(mean=0.0, std=float(std))


class _VbtStyleBlock(nn.Module):
    def __init__(self, *, dim: int, heads: int, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.head_dim = self.dim // self.heads

        self.norm1_weight = nn.Parameter(torch.ones((self.dim,), dtype=torch.float32))
        self.wq = nn.Parameter(torch.empty((self.dim, self.dim), dtype=torch.float32))
        self.wk = nn.Parameter(torch.empty((self.dim, self.dim), dtype=torch.float32))
        self.wv = nn.Parameter(torch.empty((self.dim, self.dim), dtype=torch.float32))
        self.wo = nn.Parameter(torch.empty((self.dim, self.dim), dtype=torch.float32))

        self.norm2_weight = nn.Parameter(torch.ones((self.dim,), dtype=torch.float32))
        self.w1 = nn.Parameter(torch.empty((self.dim, 4 * self.dim), dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty((4 * self.dim, self.dim), dtype=torch.float32))

        _init_normal_(self.wq)
        _init_normal_(self.wk)
        _init_normal_(self.wv)
        _init_normal_(self.wo)
        _init_normal_(self.w1)
        _init_normal_(self.w2)

        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        h = self.heads
        hd = self.head_dim

        residual1 = x
        x_flat = x.reshape(b * s, d)
        x_norm = _rmsnorm(x_flat, self.norm1_weight).reshape(b * s, d)

        q = x_norm @ self.wq
        k = x_norm @ self.wk
        v = x_norm @ self.wv
        q = q.reshape(b, s, h, hd).permute(0, 2, 1, 3)
        k = k.reshape(b, s, h, hd).permute(0, 2, 1, 3)
        v = v.reshape(b, s, h, hd).permute(0, 2, 1, 3)

        q, k = _apply_rope(q, k, self.rope_cos, self.rope_sin)

        scale = 1.0 / math.sqrt(hd)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale)
        attn_flat = attn.permute(0, 2, 1, 3).reshape(b * s, d)
        o = (attn_flat @ self.wo).reshape(b, s, d)
        x = residual1 + o

        residual2 = x
        x_flat2 = x.reshape(b * s, d)
        x_norm2 = _rmsnorm(x_flat2, self.norm2_weight).reshape(b * s, d)
        h1_pre = x_norm2 @ self.w1
        h1_act = F.gelu(h1_pre, approximate="tanh")
        h2 = (h1_act @ self.w2).reshape(b, s, d)
        x = residual2 + h2
        return x


class VbtStyleTransformer(nn.Module):
    def __init__(self, *, vocab_size: int, dim: int, heads: int, layers: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.heads = int(heads)
        self.layers = int(layers)
        self.max_seq_len = int(max_seq_len)

        if self.dim % self.heads != 0:
            raise ValueError("dim must be divisible by heads")
        head_dim = self.dim // self.heads
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        self.embedding_weight = nn.Parameter(torch.empty((self.vocab_size, self.dim), dtype=torch.float32))
        self.final_norm_weight = nn.Parameter(torch.ones((self.dim,), dtype=torch.float32))
        self.lm_head_weight = nn.Parameter(torch.empty((self.dim, self.vocab_size), dtype=torch.float32))

        _init_normal_(self.embedding_weight)
        _init_normal_(self.lm_head_weight)

        hd2 = head_dim // 2
        freq_indices = torch.arange(hd2, dtype=torch.float32)
        inv_freq = (10000.0 ** (freq_indices / float(hd2))).reciprocal().view(1, hd2)
        pos = torch.arange(self.max_seq_len, dtype=torch.float32).view(self.max_seq_len, 1)
        angles = pos * inv_freq
        rope_cos = torch.cos(angles)
        rope_sin = torch.sin(angles)

        self.blocks = nn.ModuleList(
            [_VbtStyleBlock(dim=self.dim, heads=self.heads, rope_cos=rope_cos, rope_sin=rope_sin) for _ in range(self.layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, s = tokens.shape
        if s > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")
        x = F.embedding(tokens, self.embedding_weight)  # [B,S,D]
        for blk in self.blocks:
            x = blk(x)
        x_flat = x.reshape(b * s, self.dim)
        x_norm = _rmsnorm(x_flat, self.final_norm_weight)
        return x_norm @ self.lm_head_weight  # [B*S, V]


@torch.no_grad()
def greedy_generate(model: VbtStyleTransformer, prompt: torch.Tensor, *, max_new_tokens: int) -> torch.Tensor:
    model.eval()
    out = prompt
    for _ in range(int(max_new_tokens)):
        x = out[:, -int(model.max_seq_len) :]
        b, t = x.shape
        logits_flat = model(x)
        logits = logits_flat.reshape(b, t, -1)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out = torch.cat([out, next_id], dim=1)
    return out


def _build_adamw(model: nn.Module, *, lr: float, betas: tuple[float, float], eps: float, weight_decay: float) -> torch.optim.Optimizer:
    """AdamW with GPT-style weight decay filtering.

    Commonly exclude biases / norm scale parameters from weight decay.
    Here we approximate that rule by excluding all <2D parameters.
    """

    decay: list[torch.Tensor] = []
    no_decay: list[torch.Tensor] = []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=float(lr),
        betas=betas,
        eps=float(eps),
    )


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)

    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--metrics-json", type=str, default=None)
    ap.add_argument("--show-samples", action="store_true")
    ap.add_argument("--sample-len", type=int, default=200)
    ap.add_argument("--prompt", type=str, default="to be or")
    args, _unknown = ap.parse_known_args()

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
    batch_size = int(cfg.batch_size if args.batch_size is None else args.batch_size)
    log_every = int(cfg.log_every if args.log_every is None else args.log_every)

    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    rng = np.random.default_rng(int(cfg.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset = load_shakespeare_dataset(data_dir=args.data_dir) if args.data_dir else load_shakespeare_dataset()
    vocab_size = int(dataset.vocab_size)
    split_note = "no split" if dataset.train is dataset.val else "train/val split"
    print(
        f"Shakespeare: vocab_size={vocab_size} train={dataset.train.shape[0]} val={dataset.val.shape[0]} ({split_note})"
    )

    model = VbtStyleTransformer(
        vocab_size=vocab_size,
        dim=int(cfg.dim),
        heads=int(cfg.heads),
        layers=int(cfg.layers),
        max_seq_len=int(cfg.max_seq_len),
    ).to(device=device, dtype=torch.float32)

    optimizer = _build_adamw(
        model,
        lr=float(args.lr),
        betas=(float(args.betas[0]), float(args.betas[1])),
        eps=float(args.eps),
        weight_decay=float(args.weight_decay),
    )

    run_meta = {
        "script": "examples/minigpt_shakespeare/train_pytorch_shakespeare.py",
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
            "iters": int(iters),
            "log_every": int(log_every),
            "eval_every": int(args.eval_every),
        },
    }

    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    metrics_steps: list[dict] = []

    t0 = time.time()
    for step in range(1, iters + 1):
        step_t0 = time.perf_counter()

        tokens_np, targets_np = make_batch_numpy(
            dataset.train,
            block_size=int(cfg.max_seq_len),
            batch_size=int(batch_size),
            rng=rng,
        )
        tokens = torch.from_numpy(tokens_np).to(device=device, dtype=torch.long)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits_flat = model(tokens)
        loss = F.cross_entropy(logits_flat, targets.reshape(-1), reduction="mean")

        if not torch.isfinite(loss).all():
            print(f"Non-finite loss at step {step}/{iters}: {loss}", flush=True)
            break

        loss.backward()

        if float(args.grad_clip) and float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))

        optimizer.step()

        if step % int(log_every) == 0 or step == 1 or step == iters:
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            loss_f = float(loss.detach().cpu().item())
            ppl = float(math.exp(min(20.0, loss_f)))
            print(f"Step {step:4d}/{iters} | loss={loss_f:.4f} ppl={ppl:.2f} iter_ms={iter_ms:.2f}", flush=True)

            if bool(args.show_samples):
                prompt = args.prompt
                prompt_ids = dataset.tokenizer.encode(prompt)
                prompt_t = torch.from_numpy(prompt_ids[None, :]).to(device=device, dtype=torch.long)
                gen_ids = greedy_generate(model, prompt_t, max_new_tokens=int(args.sample_len))[0].detach().cpu().numpy()
                print("--- sample ---")
                print(dataset.tokenizer.decode(gen_ids))
                print("--------------", flush=True)

            metrics_steps.append({"step": int(step), "loss": loss_f, "ppl": ppl, "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})

        if int(args.eval_every) and step % int(args.eval_every) == 0:
            with torch.no_grad():
                tokens_np, targets_np = make_batch_numpy(
                    dataset.val,
                    block_size=int(cfg.max_seq_len),
                    batch_size=int(batch_size),
                    rng=rng,
                )
                tokens = torch.from_numpy(tokens_np).to(device=device, dtype=torch.long)
                targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.long)
                logits_flat = model(tokens)
                val_loss = F.cross_entropy(logits_flat, targets.reshape(-1), reduction="mean")
                val_loss_f = float(val_loss.detach().cpu().item())
                print(f"  val_loss={val_loss_f:.4f}", flush=True)

    print(f"Done in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
