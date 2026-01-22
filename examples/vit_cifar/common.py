from __future__ import annotations

import json
import os
import pickle
import tarfile
from dataclasses import dataclass
from pathlib import Path
import urllib.request

import numpy as np

_CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_CIFAR10_ARCHIVE_NAME = "cifar-10-python.tar.gz"
_CIFAR10_EXTRACTED_DIR = "cifar-10-batches-py"


@dataclass(frozen=True)
class VitCifarConfig:
    data_root: str = str(Path(__file__).resolve().parents[2] / "tmp" / "data" / "CIFAR10")
    seed: int = 42
    batch_size: int = 256
    eval_batch_size: int = 256
    epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 0.0

    patch_size: int = 4
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    emb_dropout: float = 0.1
    num_classes: int = 10


def _unpickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="bytes")


def ensure_cifar10(root_dir: str | Path) -> Path:
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    extracted = root_dir / _CIFAR10_EXTRACTED_DIR
    if (extracted / "batches.meta").is_file():
        return extracted

    archive_path = root_dir / _CIFAR10_ARCHIVE_NAME
    if not archive_path.is_file():
        tmp_path = archive_path.with_suffix(".tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        with urllib.request.urlopen(_CIFAR10_URL) as r, tmp_path.open("wb") as out:
            while True:
                chunk = r.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if not tmp_path.is_file():
            raise RuntimeError(f"Download failed; missing temp file {tmp_path}")
        if tmp_path.stat().st_size == 0:
            raise RuntimeError(f"Download failed; empty temp file {tmp_path}")
        tmp_path.replace(archive_path)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(root_dir)

    if not (extracted / "batches.meta").is_file():
        raise RuntimeError(f"Failed to prepare CIFAR-10 under {extracted}")
    return extracted


def load_cifar10_numpy(root_dir: str | Path):
    """Return (train_images, train_labels, test_images, test_labels).

    - images: float32 in [0,1], NCHW, shape [N, 3, 32, 32]
    - labels: int64, shape [N]
    """
    extracted = ensure_cifar10(root_dir)

    train_x = []
    train_y = []
    for i in range(1, 6):
        d = _unpickle(extracted / f"data_batch_{i}")
        x = d[b"data"].reshape(-1, 3, 32, 32)
        y = np.array(d[b"labels"], dtype=np.int64)
        train_x.append(x)
        train_y.append(y)

    train_x = np.concatenate(train_x, axis=0).astype(np.float32) / 255.0
    train_y = np.concatenate(train_y, axis=0)

    d = _unpickle(extracted / "test_batch")
    test_x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    test_y = np.array(d[b"labels"], dtype=np.int64)

    return train_x, train_y, test_x, test_y


def cifar10_normalize_(x: np.ndarray):
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)
    x -= mean
    x /= std
    return x


def augment_cifar10_batch(x: np.ndarray, rng: np.random.Generator):
    n, _c, h, w = x.shape
    pad = 4
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="reflect")

    out = np.empty_like(x)
    for i in range(n):
        top = int(rng.integers(0, 2 * pad + 1))
        left = int(rng.integers(0, 2 * pad + 1))
        crop = x_pad[i, :, top : top + h, left : left + w]
        if bool(rng.integers(0, 2)):
            crop = crop[:, :, ::-1]

        b = float(rng.uniform(0.8, 1.2))
        crop = crop * b
        mean = crop.mean(axis=(1, 2), keepdims=True)
        cscale = float(rng.uniform(0.8, 1.2))
        crop = (crop - mean) * cscale + mean

        if bool(rng.integers(0, 2)):
            cut = int(rng.integers(8, 17))
            cy = int(rng.integers(0, h))
            cx = int(rng.integers(0, w))
            y0 = max(0, cy - cut // 2)
            y1 = min(h, y0 + cut)
            x0 = max(0, cx - cut // 2)
            x1 = min(w, x0 + cut)
            crop[:, y0:y1, x0:x1] = 0.0

        crop = np.clip(crop, 0.0, 1.0)
        out[i] = crop
    return out


def write_metrics_json(path: str | Path, payload: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def build_2d_sincos_pos_embed(grid_h: int, grid_w: int, dim: int) -> np.ndarray:
    if dim % 4 != 0:
        raise ValueError("dim must be divisible by 4 for 2D sincos")
    gy, gx = np.meshgrid(np.arange(grid_h, dtype=np.float32), np.arange(grid_w, dtype=np.float32), indexing="ij")
    pos = np.stack([gy.reshape(-1), gx.reshape(-1)], axis=1)  # [N,2]
    omega = np.arange(dim // 4, dtype=np.float32) / (dim // 4)
    omega = 1.0 / (10000**omega)  # [dim/4]

    out = []
    for i in range(2):
        p = pos[:, i : i + 1] * omega.reshape(1, -1)
        out.append(np.sin(p))
        out.append(np.cos(p))
    return np.concatenate(out, axis=1).astype(np.float32)  # [N, dim]


def iter_minibatches(x: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator, *, augment: bool):
    n = x.shape[0]
    indices = rng.permutation(n)
    for i in range(0, n, batch_size):
        j = indices[i : i + batch_size]
        xb = x[j]
        yb = y[j]
        if augment:
            xb = augment_cifar10_batch(xb, rng)
        yield xb, yb


def num_minibatches(n: int, batch_size: int) -> int:
    return (int(n) + int(batch_size) - 1) // int(batch_size)


def cosine_lr(base_lr: float, epoch: int, sched_epochs: int) -> float:
    import math

    t = float(epoch - 1) / max(1, int(sched_epochs))
    return float(base_lr) * 0.5 * (1.0 + math.cos(math.pi * t))


def _env_int(keys: list[str], default: int | None) -> int | None:
    for k in keys:
        v = os.environ.get(k)
        if v is None or v == "":
            continue
        return int(v)
    return default


def resolve_training_limits(*, epochs: int):
    max_epochs = _env_int(["VIT_MAX_EPOCHS", "VBT_MAX_EPOCHS", "TORCH_MAX_EPOCHS"], int(epochs))
    max_batches = _env_int(["VIT_MAX_BATCHES", "VBT_MAX_BATCHES", "TORCH_MAX_BATCHES"], None)
    sched_epochs = _env_int(["VIT_SCHED_EPOCHS", "VBT_SCHED_EPOCHS", "TORCH_SCHED_EPOCHS"], int(epochs))
    return int(max_epochs), (int(max_batches) if max_batches is not None else None), int(sched_epochs)
