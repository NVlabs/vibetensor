#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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


def _pair(t):
    if isinstance(t, tuple):
        return t
    return (t, t)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        *,
        use_bias: bool = True,
        gelu_approx: str = "none",
    ):
        super().__init__()
        if gelu_approx not in ("none", "tanh"):
            raise ValueError("gelu_approx must be 'none' or 'tanh'")
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=use_bias),
            nn.GELU(approximate=gelu_approx),
            nn.Linear(hidden_dim, dim, bias=use_bias),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0, *, use_bias: bool = True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if project_out:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim, bias=use_bias),
            )
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        b, n, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        h = self.heads
        q = q.view(b, n, h, -1).transpose(1, 2)
        k = k.view(b, n, h, -1).transpose(1, 2)
        v = v.view(b, n, h, -1).transpose(1, 2)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        *,
        rmsnorm_eps: float = 1e-8,
        use_bias: bool = True,
        gelu_approx: str = "none",
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        nn.RMSNorm(dim, eps=rmsnorm_eps),
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, use_bias=use_bias),
                        nn.RMSNorm(dim, eps=rmsnorm_eps),
                        FeedForward(dim, mlp_dim, dropout=dropout, use_bias=use_bias, gelu_approx=gelu_approx),
                    ]
                )
            )

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: int,
        patch_size: int,
        num_classes: int,
        dim: int,
        depth: int,
        heads: int,
        mlp_dim: int,
        pool: str = "cls",
        n_cls_tokens: int = 16,
        channels: int = 3,
        dim_head: int | None = None,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        patch_layout: str = "torch",
        rmsnorm_eps: float = 1e-8,
        use_bias: bool = True,
        gelu_approx: str = "none",
    ):
        super().__init__()
        self.dim = int(dim)
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)

        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        if pool not in {"cls", "mean"}:
            raise ValueError("pool must be either 'cls' or 'mean'")

        if dim_head is None:
            if dim % heads != 0:
                raise ValueError("dim must be divisible by heads")
            dim_head = dim // heads

        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.n_cls_tokens = int(n_cls_tokens)
        self.patch_layout = str(patch_layout)
        if self.patch_layout not in ("torch", "vbt"):
            raise ValueError("patch_layout must be 'torch' or 'vbt'")

        self.to_patch_embedding = nn.Linear(patch_dim, dim, bias=use_bias)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + self.n_cls_tokens, dim))
        self.cls_token = nn.Parameter(torch.randn(1, self.n_cls_tokens, dim))

        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            rmsnorm_eps=rmsnorm_eps,
            use_bias=use_bias,
            gelu_approx=gelu_approx,
        )
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.RMSNorm(dim, eps=rmsnorm_eps),
            nn.Linear(dim, num_classes, bias=use_bias),
        )

    def forward(self, img):
        b, c, h, w = img.shape
        ph = self.patch_height
        pw = self.patch_width
        gh = h // ph
        gw = w // pw
        if self.patch_layout == "vbt":
            x = img.view(b, c, gh, ph, gw, pw).permute(0, 2, 4, 1, 3, 5).contiguous()
        else:
            x = img.view(b, c, gh, ph, gw, pw).permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(b, gh * gw, ph * pw * c)
        x = self.to_patch_embedding(x)

        cls_tokens = self.cls_token.expand(b, self.n_cls_tokens, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, : x.shape[1]]

        x = self.transformer(x)
        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            x = x[:, : self.n_cls_tokens].mean(dim=1)
        return self.mlp_head(x)


def _train_epoch(model, opt, *, train_x_raw, train_y, cfg: Config, rng, epoch: int, max_epochs: int, max_batches):
    lr = cosine_lr(cfg.lr, epoch, cfg.epochs)
    for pg in opt.param_groups:
        pg["lr"] = lr
    device = next(model.parameters()).device

    model.train()
    loss_sum = 0.0
    loss_batches = 0
    correct = 0
    total = 0

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
    ):
        b_i += 1
        if max_batches is not None and b_i > max_batches:
            break
        cifar10_normalize_(xb)
        x = torch.from_numpy(xb).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(yb).to(device=device, dtype=torch.long)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

        loss_sum += float(loss.detach().cpu())
        loss_batches += 1
        pred = logits.detach().argmax(dim=1)
        correct += int((pred == y).sum().cpu())
        total += y.numel()

    loss_avg = loss_sum / max(1, loss_batches)
    acc = correct / max(1, total)
    return loss_avg, acc, lr


@torch.no_grad()
def _eval_epoch(model, *, test_x_raw, test_y, cfg: Config, rng, epoch: int, max_epochs: int):
    model.eval()
    device = next(model.parameters()).device
    loss_sum = 0.0
    loss_batches = 0
    correct = 0
    total = 0

    total_batches = num_minibatches(int(test_x_raw.shape[0]), cfg.eval_batch_size)
    test_iter = iter_minibatches(test_x_raw, test_y, cfg.eval_batch_size, rng, augment=False)
    for xb, yb in tqdm(
        test_iter,
        total=total_batches,
        desc=f"Test  {epoch}/{max_epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    ):
        cifar10_normalize_(xb)
        x = torch.from_numpy(xb).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(yb).to(device=device, dtype=torch.long)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss_sum += float(loss.detach().cpu())
        loss_batches += 1
        pred = logits.detach().argmax(dim=1)
        correct += int((pred == y).sum().cpu())
        total += y.numel()

    loss_avg = loss_sum / max(1, loss_batches)
    acc = correct / max(1, total)
    return loss_avg, acc


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--metrics-json", type=str, default=None)
    args, _unknown = ap.parse_known_args()

    cfg = Config()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    train_x_raw, train_y, test_x_raw, test_y = load_cifar10_numpy(cfg.data_root)
    print(f"Train: {train_x_raw.shape}, Test: {test_x_raw.shape}", flush=True)

    patch_size = int(cfg.patch_size)
    dim = int(cfg.dim)
    depth = int(cfg.depth)
    heads = int(cfg.heads)
    mlp_dim = int(cfg.dim) * int(cfg.mlp_ratio)
    dropout = float(cfg.dropout)
    emb_dropout = float(cfg.emb_dropout)

    model = ViT(
        image_size=32,
        patch_size=patch_size,
        num_classes=cfg.num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)

    max_epochs, max_batches, sched_epochs = resolve_training_limits(epochs=cfg.epochs)
    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    metrics_epochs: list[dict] = []
    run_meta = {
        "script": "examples/vit_cifar/train_pytorch_vit_cifar.py",
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
    }

    print(
        f"Config: patch={patch_size} dim={dim} depth={depth} heads={heads} mlp_dim={mlp_dim} "
        f"dropout={dropout} emb_dropout={emb_dropout} batch={cfg.batch_size} epochs={max_epochs} "
        f"lr={cfg.lr} wd={cfg.weight_decay}",
        flush=True,
    )

    for epoch in range(1, max_epochs + 1):
        lr = cosine_lr(cfg.lr, epoch, sched_epochs)
        epoch_t0 = time.time()
        print(f"Epoch {epoch}/{max_epochs} (lr={lr:.6g})", flush=True)

        train_loss, train_acc, lr_used = _train_epoch(
            model,
            opt,
            train_x_raw=train_x_raw,
            train_y=train_y,
            cfg=cfg,
            rng=rng,
            epoch=epoch,
            max_epochs=max_epochs,
            max_batches=max_batches,
        )
        test_loss, test_acc = _eval_epoch(
            model,
            test_x_raw=test_x_raw,
            test_y=test_y,
            cfg=cfg,
            rng=rng,
            epoch=epoch,
            max_epochs=max_epochs,
        )
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

    dt = time.time() - t0
    print(f"Final Test Acc: {test_acc:.4f}", flush=True)
    print(f"Training time: {dt:.2f}s ({max_epochs} epochs)", flush=True)


if __name__ == "__main__":
    main()
