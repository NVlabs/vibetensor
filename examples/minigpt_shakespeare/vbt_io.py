from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

import vibetensor.torch as vt

from vibe_kernels.ops import Transformer


def _vbt_to_numpy_cpu(vbt_tensor) -> np.ndarray:
    t_cpu = vbt_tensor.cpu()
    return np.from_dlpack(t_cpu).reshape(tuple(int(s) for s in t_cpu.sizes))


def extract_vbt_weights(model: Transformer) -> dict[str, np.ndarray]:
    """Return a CPU numpy snapshot of the VBT Transformer weights."""

    out: dict[str, np.ndarray] = {}

    def _snap(name: str, w) -> None:
        out[name] = _vbt_to_numpy_cpu(w).copy()

    _snap("embedding.weight", model.embedding.weights["weight"])
    for i in range(int(model.n_layers)):
        blk = model.blocks[i]
        _snap(f"blocks.{i}.norm1.weight", blk.norm1.weights["weight"])
        _snap(f"blocks.{i}.wq.weight", blk.wq.weights["weight"])
        _snap(f"blocks.{i}.wk.weight", blk.wk.weights["weight"])
        _snap(f"blocks.{i}.wv.weight", blk.wv.weights["weight"])
        _snap(f"blocks.{i}.wo.weight", blk.wo.weights["weight"])
        _snap(f"blocks.{i}.norm2.weight", blk.norm2.weights["weight"])
        _snap(f"blocks.{i}.ffn.w1", blk.ffn.weights["w1"])
        _snap(f"blocks.{i}.ffn.w2", blk.ffn.weights["w2"])

    _snap("final_norm.weight", model.final_norm.weights["weight"])
    _snap("lm_head.weight", model.lm_head.weights["weight"])
    return out


def load_vbt_weights_(model: Transformer, weights: dict[str, Any]) -> None:
    """In-place load of weights into a fresh VBT Transformer."""

    def _to_numpy(t: Any) -> np.ndarray:
        if isinstance(t, np.ndarray):
            return t
        arr = np.asarray(t)
        if not isinstance(arr, np.ndarray) or arr.dtype == object:
            raise TypeError(f"Unsupported weight type: {type(t)}")
        return arr

    def _to_vt(t: Any):
        arr = _to_numpy(t)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        return vt.from_numpy(arr).cuda()

    model.embedding.weights["weight"] = _to_vt(weights["embedding.weight"])
    for i in range(int(model.n_layers)):
        blk = model.blocks[i]
        blk.norm1.weights["weight"] = _to_vt(weights[f"blocks.{i}.norm1.weight"])
        blk.wq.weights["weight"] = _to_vt(weights[f"blocks.{i}.wq.weight"])
        blk.wk.weights["weight"] = _to_vt(weights[f"blocks.{i}.wk.weight"])
        blk.wv.weights["weight"] = _to_vt(weights[f"blocks.{i}.wv.weight"])
        blk.wo.weights["weight"] = _to_vt(weights[f"blocks.{i}.wo.weight"])
        blk.norm2.weights["weight"] = _to_vt(weights[f"blocks.{i}.norm2.weight"])
        blk.ffn.weights["w1"] = _to_vt(weights[f"blocks.{i}.ffn.w1"])
        blk.ffn.weights["w2"] = _to_vt(weights[f"blocks.{i}.ffn.w2"])

    model.final_norm.weights["weight"] = _to_vt(weights["final_norm.weight"])
    model.lm_head.weights["weight"] = _to_vt(weights["lm_head.weight"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, Any]:
    del map_location
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(4)
        f.seek(0)
        if head.startswith(b"PK\x03\x04"):
            raise ValueError(
                "This checkpoint looks like a Torch zip-format file. "
                "This example no longer depends on torch; please re-save checkpoints using the updated script."
            )
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected checkpoint type: {type(obj)}")
    return obj
