#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

import vibetensor
import vibetensor.torch as vt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.minigpt_reverse.common import ReverseConfig, decode_tokens, make_batch_numpy, write_metrics_json
from vibe_kernels.ops import Transformer
from vibe_kernels.indexing.vbt_native import argmax as vbt_argmax


def _vbt_to_numpy_cpu(vbt_tensor) -> np.ndarray:
    t_cpu = vbt_tensor.cpu()
    return np.from_dlpack(t_cpu).reshape(tuple(int(s) for s in t_cpu.sizes))


def greedy_generate_vbt(
    model: Transformer,
    prompt_np: np.ndarray,
    *,
    max_new_tokens: int,
    eos_id: int,
    max_seq_len: int,
    pad_id: int,
) -> np.ndarray:
    seq = prompt_np.astype(np.int64, copy=True)
    for _ in range(int(max_new_tokens)):
        if seq.shape[0] > int(max_seq_len):
            raise RuntimeError("generation exceeded max_seq_len")
        toks_full = np.full((int(max_seq_len),), int(pad_id), dtype=np.int64)
        toks_full[: seq.shape[0]] = seq
        toks_v = vt.from_numpy(toks_full[None, :]).cuda()
        logits_v = model.fwd(toks_v)  # [B*S, V], with S=max_seq_len
        logits_np = _vbt_to_numpy_cpu(logits_v).reshape(1, int(max_seq_len), -1)
        last_pos = int(seq.shape[0] - 1)
        next_id = int(np.argmax(logits_np[0, last_pos, :], axis=-1))
        seq = np.concatenate([seq, np.array([next_id], dtype=np.int64)], axis=0)
        if next_id == int(eos_id):
            break
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--num-step", type=int, default=None, help="Number of training steps to run")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--show-samples", action="store_true")
    ap.add_argument("--manual-attn-bwd", action="store_true", help="Use slow manual attention backward instead of Triton")
    ap.add_argument("--metrics-json", type=str, default=None)
    args, _unknown = ap.parse_known_args()

    cfg = ReverseConfig()
    iters = int(cfg.iters if args.iters is None else args.iters)
    if args.num_step is not None:
        iters = int(args.num_step)
    batch_size = int(cfg.batch_size if args.batch_size is None else args.batch_size)
    log_every = int(cfg.log_every if args.log_every is None else args.log_every)
    show_samples = bool(args.show_samples)

    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        raise SystemExit("CUDA is required for VibeTensor")

    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    vt.manual_seed(int(cfg.seed))

    lr = 0.1
    grad_clip = 1.0

    use_triton_attn_bwd = not bool(args.manual_attn_bwd)

    model = Transformer(
        vocab_size=int(cfg.vocab_size),
        dim=int(cfg.dim),
        n_heads=int(cfg.heads),
        n_layers=int(cfg.layers),
        max_seq_len=int(cfg.max_seq_len),
        causal=True,
        use_triton_attn_bwd=use_triton_attn_bwd,
    )

    run_meta = {
        "script": "examples/minigpt_reverse/train_vbt_reverse.py",
        "config": {
            "seed": int(cfg.seed),
            "max_n": int(cfg.max_n),
            "alphabet_size": int(cfg.alphabet_size),
            "vocab_size": int(cfg.vocab_size),
            "max_seq_len": int(cfg.max_seq_len),
            "dim": int(cfg.dim),
            "heads": int(cfg.heads),
            "layers": int(cfg.layers),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(cfg.weight_decay),
            "grad_clip": float(grad_clip),
            "iters": int(iters),
            "log_every": int(log_every),
            "show_samples": bool(show_samples),
        },
        "task_format": "<BOS> a1..an <SEP> an..a1 <EOS>",
    }

    metrics_path = Path(args.metrics_json) if args.metrics_json else None
    metrics_steps: list[dict] = []

    t0 = time.time()
    for step in range(1, iters + 1):
        step_t0 = time.perf_counter()
        tokens_np, targets_np, lengths_np = make_batch_numpy(cfg=cfg, rng=rng, batch_size=batch_size)
        tokens_v = vt.from_numpy(tokens_np.astype(np.int64)).cuda()
        targets_v = vt.from_numpy(targets_np.astype(np.int64)).cuda()

        logits_v = model.fwd(tokens_v)  # [B*S, V]
        loss_v = float(model.compute_loss(logits_v, targets_v))
        model.bwd()
        model.update(float(lr))

        preds_v = vbt_argmax(logits_v, dim=1).reshape([int(batch_size), int(cfg.max_seq_len)])
        preds_np = _vbt_to_numpy_cpu(preds_v)
        mask = targets_np != -100
        denom = int(mask.sum())
        tok_acc = float(((preds_np == targets_np) & mask).sum() / max(1, denom))

        w_np = _vbt_to_numpy_cpu(model.lm_head.weights["weight"])
        if not np.isfinite(w_np).all():
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            w_finite_frac = float(np.isfinite(w_np).mean())
            w_max_abs = float(np.max(np.abs(w_np))) if w_np.size else float("nan")
            print(
                f"Non-finite weights after update at step {step}/{iters} | iter_ms={iter_ms:.2f}",
                flush=True,
            )
            print(
                f"  lm_head.weight: finite_frac={w_finite_frac:.6f} max_abs={w_max_abs:.6g}",
                flush=True,
            )
            metrics_steps.append({"step": int(step), "loss": float(loss_v), "token_acc": float(tok_acc), "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})
            break

        if not np.isfinite(loss_v):
            logits_np = _vbt_to_numpy_cpu(logits_v)
            finite = np.isfinite(logits_np)
            finite_frac = float(finite.mean())
            max_abs = float(np.max(np.abs(logits_np))) if logits_np.size else float("nan")
            mn = float(np.min(logits_np)) if logits_np.size else float("nan")
            mx = float(np.max(logits_np)) if logits_np.size else float("nan")
            w_finite_frac = float(np.isfinite(w_np).mean())
            w_max_abs = float(np.max(np.abs(w_np))) if w_np.size else float("nan")
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            print(
                f"Non-finite loss at step {step}/{iters} | loss={loss_v} tok_acc={tok_acc:.4f} iter_ms={iter_ms:.2f}",
                flush=True,
            )
            print(
                f"  logits: finite_frac={finite_frac:.6f} min={mn:.6g} max={mx:.6g} max_abs={max_abs:.6g}",
                flush=True,
            )
            print(
                f"  lm_head.weight: finite_frac={w_finite_frac:.6f} max_abs={w_max_abs:.6g}",
                flush=True,
            )
            metrics_steps.append({"step": int(step), "loss": float(loss_v), "token_acc": float(tok_acc), "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})
            break

        if step % int(log_every) == 0 or step == 1 or step == iters:
            iter_ms = (time.perf_counter() - step_t0) * 1000.0
            print(f"Step {step:4d}/{iters} | loss={loss_v:.4f} tok_acc={tok_acc:.4f} iter_ms={iter_ms:.2f}", flush=True)
            if show_samples:
                ex_n = int(lengths_np[0])
                prompt_np = tokens_np[0, : (1 + ex_n + 1)]
                expected = tokens_np[0, : (2 * ex_n + 3)]
                max_new = int(ex_n) + 1
                gen = greedy_generate_vbt(
                    model,
                    prompt_np,
                    max_new_tokens=max_new,
                    eos_id=cfg.eos_id,
                    max_seq_len=cfg.max_seq_len,
                    pad_id=cfg.pad_id,
                )
                print(f"  prompt:   {decode_tokens(prompt_np, cfg=cfg)}", flush=True)
                print(f"  generate: {decode_tokens(gen, cfg=cfg)}", flush=True)
                print(f"  expect:   {decode_tokens(expected, cfg=cfg)}", flush=True)
            metrics_steps.append({"step": int(step), "loss": float(loss_v), "token_acc": float(tok_acc), "iter_ms": float(iter_ms)})
            if metrics_path is not None:
                write_metrics_json(metrics_path, {"run": run_meta, "steps": metrics_steps})

    print(f"Done in {time.time() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
