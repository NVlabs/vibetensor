#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import vibetensor
import vibetensor.torch as vt

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.minigpt_shakespeare.common import ShakespeareBPE  # noqa: E402
from examples.minigpt_shakespeare.vbt_io import load_checkpoint, load_vbt_weights_  # noqa: E402
from vibe_kernels.ops import Transformer  # noqa: E402


def _logits_last_position(
    model: Transformer,
    ctx_ids: np.ndarray,
    *,
    max_seq_len: int,
    pad_id: int,
) -> np.ndarray:
    ctx = ctx_ids[-int(max_seq_len) :]
    if ctx.size == 0:
        raise ValueError("empty context")

    toks_full = np.full((int(max_seq_len),), int(pad_id), dtype=np.int64)
    toks_full[: ctx.shape[0]] = ctx

    toks_v = vt.from_numpy(toks_full[None, :]).cuda()
    last_pos = int(ctx.shape[0] - 1)

    with vt.no_grad():
        logits_v = model.fwd(toks_v).reshape([1, int(max_seq_len), -1])
        logits_row_v = logits_v.select(0, 0).select(0, last_pos)  # [V]
        logits_cpu = logits_row_v.cpu()
        return np.from_dlpack(logits_cpu).reshape((int(logits_cpu.sizes[0]),))


def _sample_next_id(logits_np: np.ndarray, *, temperature: float, top_k: int) -> int:
    if float(temperature) <= 0:
        return int(np.argmax(logits_np, axis=-1))

    x = logits_np.astype(np.float32, copy=False)
    if float(temperature) != 1.0:
        x = x / np.float32(float(temperature))

    if int(top_k) > 0:
        k = min(int(top_k), int(x.shape[0]))
        ix = np.argpartition(x, -k)[-k:]
        v = x[ix]
        v = v - np.max(v)
        probs = np.exp(v).astype(np.float64, copy=False)
        probs = probs / max(1e-20, float(probs.sum()))
        next_local = int(np.random.choice(np.arange(k, dtype=np.int64), p=probs))
        return int(ix[next_local])

    return int(np.argmax(x, axis=-1))


def generate(
    model: Transformer,
    prompt_ids: np.ndarray,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_k: int,
    pad_id: int,
) -> np.ndarray:
    out = prompt_ids.astype(np.int64, copy=True)

    for _ in range(int(max_new_tokens)):
        logits_np = _logits_last_position(model, out, max_seq_len=int(max_seq_len), pad_id=int(pad_id))
        next_id = _sample_next_id(logits_np, temperature=float(temperature), top_k=int(top_k))

        out = np.concatenate([out, np.array([next_id], dtype=np.int64)], axis=0)

    return out


def stream_generate(
    model: Transformer,
    tok: ShakespeareBPE,
    prompt_ids: np.ndarray,
    *,
    max_new_tokens: int,
    max_seq_len: int,
    temperature: float,
    top_k: int,
    pad_id: int,
    delay_s: float,
) -> np.ndarray:
    import codecs
    import time

    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    for tid in prompt_ids.tolist():
        sys.stdout.write(decoder.decode(tok.token_bytes(int(tid)), final=False))
    sys.stdout.flush()

    out = prompt_ids.astype(np.int64, copy=True)

    for _ in range(int(max_new_tokens)):
        logits_np = _logits_last_position(model, out, max_seq_len=int(max_seq_len), pad_id=int(pad_id))
        next_id = _sample_next_id(logits_np, temperature=float(temperature), top_k=int(top_k))

        out = np.concatenate([out, np.array([next_id], dtype=np.int64)], axis=0)

        piece = decoder.decode(tok.token_bytes(int(next_id)), final=False)
        if piece:
            sys.stdout.write(piece)
            sys.stdout.flush()

        time.sleep(float(delay_s))

    tail = decoder.decode(b"", final=True)
    if tail:
        sys.stdout.write(tail)

    sys.stdout.write("\n")
    sys.stdout.flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="to be or")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--next-only", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--delay", type=float, default=0.02)
    args = ap.parse_args()

    if not vibetensor._C._has_cuda or vibetensor._C._cuda_device_count() == 0:
        raise SystemExit("CUDA is required for VibeTensor")

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    if ckpt.get("format") != "vibetorch-minigpt-shakespeare-v1":
        raise SystemExit(f"Unexpected checkpoint format: {ckpt.get('format')}")

    cfg = ckpt["config"]
    vocab_size = int(cfg["vocab_size"])
    max_seq_len = int(cfg["max_seq_len"])

    tok_cfg = ckpt.get("tokenizer")
    if tok_cfg is None:
        raise SystemExit("Checkpoint missing tokenizer config (expected BPE tokenizer)")
    tok = ShakespeareBPE.from_config(tok_cfg)
    pad_id = int(tok.pad_id)

    if int(tok.vocab_size) != int(vocab_size):
        raise SystemExit(f"Tokenizer vocab_size mismatch: ckpt={vocab_size} tokenizer={tok.vocab_size}")

    model = Transformer(
        vocab_size=int(vocab_size),
        dim=int(cfg["dim"]),
        n_heads=int(cfg["heads"]),
        n_layers=int(cfg["layers"]),
        max_seq_len=int(max_seq_len),
        causal=True,
    )
    load_vbt_weights_(model, ckpt["weights"])

    prompt_ids = tok.encode(args.prompt)

    if bool(args.next_only):
        logits_np = _logits_last_position(model, prompt_ids, max_seq_len=max_seq_len, pad_id=pad_id)
        next_id = int(np.argmax(logits_np, axis=-1))
        piece = tok.decode([next_id])
        print(f"next_token_id={next_id} next_piece={repr(piece)}")
        return

    if bool(args.stream):
        stream_generate(
            model,
            tok,
            prompt_ids,
            max_new_tokens=int(args.max_new_tokens),
            max_seq_len=int(max_seq_len),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            pad_id=int(pad_id),
            delay_s=float(args.delay),
        )
        return

    out = generate(
        model,
        prompt_ids,
        max_new_tokens=int(args.max_new_tokens),
        max_seq_len=int(max_seq_len),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        pad_id=pad_id,
    )
    sys.stdout.write(tok.decode(out) + "\n")


if __name__ == "__main__":
    main()
