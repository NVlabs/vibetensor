from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any
import urllib.request

from collections import Counter
import re

import numpy as np

_SHAKESPEARE_URL = "https://www.gutenberg.org/files/100/100-0.txt"
_SHAKESPEARE_FILE = "shakespeare.txt"


@dataclass(frozen=True)
class ShakespeareConfig:
    # Repro
    seed: int = 42

    # Data/model
    max_seq_len: int = 128  # block size; keep power-of-2 for kernel-friendly tiling

    dim: int = 256
    heads: int = 8
    layers: int = 6

    # Training defaults (scripts can override via CLI)
    batch_size: int = 64
    iters: int = 5000
    log_every: int = 1000


def default_data_dir() -> Path:
    # <repo>/tmp/data/shakespeare_full
    return Path(__file__).resolve().parents[2] / "tmp" / "data" / "shakespeare_full"


def default_tokenizer_dir() -> Path:
    # <repo>/tmp/tokenizers/shakespeare_full_bpe
    return Path(__file__).resolve().parents[2] / "tmp" / "tokenizers" / "shakespeare_full_bpe"


def ensure_shakespeare(data_dir: str | Path, *, url: str = _SHAKESPEARE_URL) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / _SHAKESPEARE_FILE
    if out_path.is_file() and out_path.stat().st_size > 0:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    req = urllib.request.Request(url, headers={"User-Agent": "vibetorch-minigpt-shakespeare"})
    with urllib.request.urlopen(req) as r, tmp_path.open("wb") as out:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
        out.flush()
        os.fsync(out.fileno())

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {tmp_path}")
    os.replace(tmp_path, out_path)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Failed to prepare Shakespeare dataset under {out_path}")
    return out_path


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte<->unicode reversible mapping.

    This gives a reversible mapping for all 256 byte values.
    """

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


class ShakespeareBPE:
    """Trainable byte-level BPE tokenizer for Shakespeare.

    - Trains on the configured Shakespeare corpus (when missing) and saves a tokenizer JSON.
    - Pure-Python, no external tokenizer dependencies.
    - Reversible byte-level encoding (exact decode) via GPT-2 byte<->unicode mapping.

    Vocab layout:
      ids [0..255]    : raw bytes (byte alphabet)
      ids [256..]     : merged symbols (trained merges)

    Note: id 0 corresponds to byte 0 which never appears in UTF-8 text, so we
    use it as `pad_id` for right-padding.
    """

    _PAT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+")

    def __init__(self, *, vocab_size: int, merges: list[tuple[str, str]]):
        if int(vocab_size) < 300:
            raise ValueError("vocab_size too small for byte-level BPE")
        self.vocab_size = int(vocab_size)
        self.merges = list(merges)

        self._byte_encoder = _bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        # Merge ranks: lower rank = apply earlier.
        self._bpe_ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._bpe_cache: dict[str, tuple[str, ...]] = {}

        # Build vocab id <-> token mapping.
        id_to_token: list[str] = []
        for b in range(256):
            id_to_token.append(self._byte_encoder[b])

        # Add merged symbols in merge order.
        for a, b in self.merges:
            if len(id_to_token) >= int(self.vocab_size):
                break
            id_to_token.append(a + b)

        self._id_to_token = id_to_token
        self._token_to_id = {tok: i for i, tok in enumerate(self._id_to_token)}

    @property
    def pad_id(self) -> int:
        return 0

    def to_config(self, *, include_merges: bool = True) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "type": "shakespeare_bpe",
            "version": 1,
            "vocab_size": int(self.vocab_size),
        }
        if include_merges:
            cfg["merges"] = [[a, b] for (a, b) in self.merges]
        return cfg

    @staticmethod
    def from_config(cfg: dict[str, Any]) -> "ShakespeareBPE":
        if cfg.get("type") != "shakespeare_bpe":
            raise ValueError(f"Unsupported tokenizer config: {cfg}")
        merges_raw = cfg.get("merges")
        if merges_raw is None:
            raise ValueError("Tokenizer config missing merges")
        merges = [(str(a), str(b)) for a, b in merges_raw]
        return ShakespeareBPE(vocab_size=int(cfg["vocab_size"]), merges=merges)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = self.to_config(include_merges=True)
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def load(path: str | Path) -> "ShakespeareBPE":
        path = Path(path)
        cfg = json.loads(path.read_text(encoding="utf-8"))
        return ShakespeareBPE.from_config(cfg)

    @classmethod
    def train_from_text(
        cls,
        text: str,
        *,
        vocab_size: int,
        min_pair_freq: int = 2,
        verbose: bool = True,
    ) -> "ShakespeareBPE":
        if verbose:
            print(f"[tokenizer] training ShakespeareBPE vocab_size={int(vocab_size)} ...", flush=True)

        byte_encoder = _bytes_to_unicode()

        # Pre-tokenize to GPT-2-like pieces (keeps leading spaces on word tokens).
        pieces = cls._PAT.findall(text)
        counter: Counter[str] = Counter()
        for p in pieces:
            p_u = "".join(byte_encoder[b] for b in p.encode("utf-8"))
            counter[p_u] += 1

        # Vocabulary for BPE training: word(tuple of symbols) -> freq
        words: dict[tuple[str, ...], int] = {tuple(k): int(v) for k, v in counter.items()}

        merges_target = int(vocab_size) - 256
        merges: list[tuple[str, str]] = []

        # Deterministic BPE training (lexicographic tie-break).
        cur_min_freq = int(min_pair_freq)
        for it in range(int(merges_target)):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for w, freq in words.items():
                if len(w) < 2:
                    continue
                prev = w[0]
                for ch in w[1:]:
                    pair_counts[(prev, ch)] += freq
                    prev = ch

            if not pair_counts:
                if verbose:
                    print(f"[tokenizer] stopped early at {len(merges)} merges (no pairs)", flush=True)
                break

            best_count = max(pair_counts.values())
            if best_count < cur_min_freq:
                if cur_min_freq > 1:
                    cur_min_freq = 1
                    if verbose:
                        print(f"[tokenizer] relaxing min_pair_freq to 1 (best_count={best_count})", flush=True)
                    # retry this iteration
                    continue
                if verbose:
                    print(f"[tokenizer] stopped early at {len(merges)} merges (best_count={best_count})", flush=True)
                break

            # Deterministic best pair: max count, then lexicographic.
            best_pairs = [pair for pair, c in pair_counts.items() if c == best_count]
            best_pair = min(best_pairs)
            merges.append(best_pair)

            # Merge across all words.
            a, b = best_pair
            merged_sym = a + b
            new_words: dict[tuple[str, ...], int] = {}
            for w, freq in words.items():
                if len(w) < 2:
                    new_words[w] = new_words.get(w, 0) + freq
                    continue
                out: list[str] = []
                i = 0
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        out.append(merged_sym)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                wt = tuple(out)
                new_words[wt] = new_words.get(wt, 0) + freq
            words = new_words

            if verbose and (it + 1) % 100 == 0:
                print(f"[tokenizer] merges {it + 1}/{merges_target} (top_pair_count={best_count})", flush=True)

        if verbose:
            print(f"[tokenizer] done: merges={len(merges)} vocab_size={int(vocab_size)}", flush=True)

        return cls(vocab_size=int(vocab_size), merges=merges)

    def _bpe(self, token: str) -> tuple[str, ...]:
        cached = self._bpe_cache.get(token)
        if cached is not None:
            return cached

        word = tuple(token)
        if len(word) <= 1:
            out = (token,)
            self._bpe_cache[token] = out
            return out

        pairs = _get_pairs(word)
        if not pairs:
            out = (token,)
            self._bpe_cache[token] = out
            return out

        while True:
            bigram = min(pairs, key=lambda p: self._bpe_ranks.get(p, 10**18))
            if bigram not in self._bpe_ranks:
                break

            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])

                if j < len(word) - 1 and word[j] == first and word[j + 1] == second:
                    new_word.append(first + second)
                    i = j + 2
                else:
                    new_word.append(word[j])
                    i = j + 1

            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)

        self._bpe_cache[token] = word
        return word

    def encode(self, text: str) -> np.ndarray:
        ids: list[int] = []
        for piece in self._PAT.findall(text):
            piece_u = "".join(self._byte_encoder[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(piece_u):
                tid = self._token_to_id.get(tok)
                if tid is None:
                    # Shouldn't happen for trained merges + byte alphabet coverage.
                    tid = int(self.pad_id)
                ids.append(int(tid))
        return np.asarray(ids, dtype=np.int64)

    def token_bytes(self, token_id: int) -> bytes:
        tok = self._id_to_token[int(token_id)]
        return bytes([self._byte_decoder[c] for c in tok])

    def decode_bytes(self, ids: list[int] | np.ndarray) -> bytes:
        if isinstance(ids, np.ndarray):
            ids_list = ids.tolist()
        else:
            ids_list = ids

        # Build the byte-level string first, then map back to raw bytes.
        s = "".join(self._id_to_token[int(i)] for i in ids_list)
        return bytes([self._byte_decoder[c] for c in s])

    def decode(self, ids: list[int] | np.ndarray) -> str:
        return self.decode_bytes(ids).decode("utf-8", errors="replace")


@dataclass(frozen=True)
class ShakespeareDataset:
    train: np.ndarray  # int64 tokens
    val: np.ndarray  # int64 tokens
    tokenizer: ShakespeareBPE

    @property
    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)


def load_shakespeare_dataset(
    *,
    data_dir: str | Path = default_data_dir(),
    url: str = _SHAKESPEARE_URL,
    train_frac: float = 1.0,
    vocab_size: int = 4096,
    tokenizer_dir: str | Path = default_tokenizer_dir(),
) -> ShakespeareDataset:
    """Load Shakespeare and tokenize with a trained byte-level BPE tokenizer.

    Default dataset: Project Gutenberg ebook #100 (Complete Works of William Shakespeare).

    - If the tokenizer JSON does not exist, it will be trained on the Shakespeare corpus and saved.
    - Tokenized ids are cached as a `.npy` under the dataset directory.
    - By default (`train_frac=1.0`), no train/val split is performed: all tokens are used for
      training and `dataset.val` is set to the full token stream as well.
    """

    data_dir = Path(data_dir)
    path = ensure_shakespeare(data_dir, url=url)

    tokenizer_dir = Path(tokenizer_dir)
    tok_path = tokenizer_dir / f"shakespeare_full_bpe_v{int(vocab_size)}.json"

    text: str | None = None
    if tok_path.is_file() and tok_path.stat().st_size > 0:
        tok = ShakespeareBPE.load(tok_path)
    else:
        text = path.read_text(encoding="utf-8")
        tok = ShakespeareBPE.train_from_text(text, vocab_size=int(vocab_size), verbose=True)
        tok.save(tok_path)

    cache_tag = f"shakespeare_full_bpe_v{int(tok.vocab_size)}"
    cache_path = data_dir / f"tokens_{cache_tag}.npy"

    if cache_path.is_file() and cache_path.stat().st_size > 0:
        data = np.load(cache_path)
    else:
        if text is None:
            text = path.read_text(encoding="utf-8")
        data = tok.encode(text)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        with tmp.open("wb") as f:
            np.save(f, data)
        os.replace(tmp, cache_path)

    train_frac_f = float(train_frac)
    if not (0.0 < train_frac_f <= 1.0):
        raise ValueError(f"train_frac must be in (0, 1], got {train_frac!r}")

    # Default behavior for this example: use *all* data for training.
    # Keep `val` pointing at the same stream so optional eval runs still work.
    if train_frac_f >= 1.0:
        train = data
        val = data
    else:
        n = int(train_frac_f * int(data.shape[0]))
        train = data[:n]
        val = data[n:]

    return ShakespeareDataset(train=train, val=val, tokenizer=tok)


def make_batch_numpy(
    data: np.ndarray,
    *,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tokens/targets for next-token prediction.

    tokens:  [B, S]
    targets: [B, S] where targets[b, t] = tokens[b, t+1] in the raw stream.

    We sample contiguous windows of length S+1 from the token stream.
    """

    s = int(block_size)
    b = int(batch_size)

    if int(data.shape[0]) <= s + 1:
        raise ValueError(f"dataset too small for block_size={s}: len={int(data.shape[0])}")

    starts = rng.integers(0, int(data.shape[0]) - (s + 1), size=(b,), dtype=np.int64)
    tokens = np.empty((b, s), dtype=np.int64)
    targets = np.empty((b, s), dtype=np.int64)
    for i, st in enumerate(starts.tolist()):
        chunk = data[int(st) : int(st) + (s + 1)]
        tokens[i, :] = chunk[:-1]
        targets[i, :] = chunk[1:]
    return tokens, targets


def write_metrics_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)
