#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReverseConfig:
    seed: int = 123

    max_n: int = 16
    alphabet_size: int = 32

    bos_id: int = 0
    sep_id: int = 1
    eos_id: int = 2
    pad_id: int = 3

    dim: int = 128
    heads: int = 4
    layers: int = 2

    batch_size: int = 64
    weight_decay: float = 0.0

    iters: int = 2000
    log_every: int = 100

    @property
    def vocab_size(self) -> int:
        return 4 + int(self.alphabet_size)

    @property
    def token_offset(self) -> int:
        return 4

    @property
    def max_seq_len(self) -> int:
        # Task format length: <BOS> a1..an <SEP> an..a1 <EOS> = 2*max_n + 3.
        # VibeTensor attention kernels require power-of-two tiling for some configs,
        # so we round up to the next power of 2 for the model's max_seq_len.
        need = 2 * int(self.max_n) + 3
        p2 = 1
        while p2 < need:
            p2 *= 2
        return int(p2)


def make_batch_numpy(*, cfg: ReverseConfig, rng: np.random.Generator, batch_size: int | None = None):
    bsz = int(cfg.batch_size if batch_size is None else batch_size)
    s = int(cfg.max_seq_len)

    tokens = np.full((bsz, s), cfg.pad_id, dtype=np.int64)
    targets = np.full((bsz, s), -100, dtype=np.int64)
    lengths = np.empty((bsz,), dtype=np.int64)

    for i in range(bsz):
        n = int(rng.integers(1, cfg.max_n + 1))
        a = rng.integers(0, cfg.alphabet_size, size=(n,), dtype=np.int64) + cfg.token_offset

        seq = np.empty((1 + n + 1 + n + 1,), dtype=np.int64)
        seq[0] = cfg.bos_id
        seq[1 : 1 + n] = a
        seq[1 + n] = cfg.sep_id
        seq[1 + n + 1 : 1 + n + 1 + n] = a[::-1]
        seq[1 + n + 1 + n] = cfg.eos_id

        tokens[i, : seq.shape[0]] = seq
        lengths[i] = n

        sep_pos = 1 + n
        for t in range(sep_pos, seq.shape[0] - 1):
            targets[i, t] = seq[t + 1]

    return tokens, targets, lengths


def make_prompt_numpy(*, cfg: ReverseConfig, rng: np.random.Generator, batch_size: int = 1):
    tokens, _targets, lengths = make_batch_numpy(cfg=cfg, rng=rng, batch_size=batch_size)
    prompts = []
    for i in range(int(batch_size)):
        n = int(lengths[i])
        prompt = tokens[i, : (1 + n + 1)]  # <BOS> a..a <SEP>
        prompts.append(prompt)
    return prompts, lengths


def decode_tokens(tokens: list[int] | np.ndarray, *, cfg: ReverseConfig) -> str:
    ids = [int(x) for x in (tokens.tolist() if isinstance(tokens, np.ndarray) else tokens)]
    out = []
    for t in ids:
        if t == cfg.bos_id:
            out.append("<BOS>")
        elif t == cfg.sep_id:
            out.append("<SEP>")
        elif t == cfg.eos_id:
            out.append("<EOS>")
        elif t == cfg.pad_id:
            out.append("<PAD>")
        elif t >= cfg.token_offset:
            out.append(str(t - cfg.token_offset))
        else:
            out.append(f"<{t}>")
    return " ".join(out)


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
