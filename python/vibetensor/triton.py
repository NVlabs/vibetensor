# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple, Dict, List, Union

# NOTE: This module must be importable without torch or triton installed.
# provide just registration scaffolding and CPU redispatch behavior.

from vibetensor import _C as _C  # C++ bindings
import vibetensor.torch as vt  # torch-free overlay (safe import)

# DLPack device type codes (reuse DLPack definitions)
_KDLCPU = 1
_KDLCUDA = 2
_KDLCUDAMANAGED = 13

import os as _os
import shutil as _shutil
import atexit as _atexit
import threading as _threading
from collections import OrderedDict as _OrderedDict
import inspect as _inspect


# Cache types
_CodeKey = Tuple[str, Tuple[int, int], str, int, str, Tuple[Tuple[str, int], ...], int, int]
# (opname, cc, triton_version, kernel_id, signature, normalized_meta_items, threads_per_block, num_stages)
_ModuleKey = Tuple[int, _CodeKey]  # (device_index, code_key)

# Capacity: env VBT_TRITON_CACHE_SIZE, default 64, minimum 1
def _cache_capacity() -> int:
    try:
        v = int(_os.getenv("VBT_TRITON_CACHE_SIZE", "64"))
        return max(1, v)
    except Exception:
        return 64

# LRU stores
_code_cache: _OrderedDict[_CodeKey, Tuple[bytes, str, int]] = _OrderedDict()  # (ptx, entry, shared_bytes)
_module_cache: _OrderedDict[_ModuleKey, Tuple[int, int]] = _OrderedDict()  # (mod_handle, func_handle)
# Track modules currently in use to avoid eviction during launches
_module_in_use: Dict[_ModuleKey, int] = {}

# Counters
_code_hits = 0
_code_misses = 0
_module_hits = 0
_module_misses = 0

# Locks
_global_lock = _threading.RLock()
_compile_locks: Dict[_CodeKey, _threading.Lock] = {}
_module_locks: Dict[_ModuleKey, _threading.Lock] = {}


def _kernel_identity(kernel: Any) -> int:
    # Fallback to object id; Triton JITFunction is stable in-process
    return id(kernel)


def _normalized_meta_items(meta: Optional[Dict[str, Any]]) -> Tuple[Tuple[str, int], ...]:
    m = dict(meta or {})
    items: List[Tuple[str, int]] = []
    for k in sorted(m.keys()):
        items.append((str(k), int(m[k])))
    return tuple(items)


def _code_key(opname: str, cc: Tuple[int, int], triton_version: str, kernel: Any, signature: str, meta: Optional[Dict[str, Any]], tpb: int, num_stages_key: int) -> _CodeKey:
    return (
        str(opname),
        (int(cc[0]), int(cc[1])),
        str(triton_version),
        int(_kernel_identity(kernel)),
        str(signature),
        _normalized_meta_items(meta),
        int(tpb),
        int(num_stages_key),
    )


def _lru_get_code(k: _CodeKey) -> Optional[Tuple[bytes, str, int]]:
    global _code_hits, _code_misses
    with _global_lock:
        val = _code_cache.get(k)
        if val is not None:
            _code_cache.move_to_end(k)
            _code_hits += 1
            return val
        _code_misses += 1
        return None


def _lru_put_code(k: _CodeKey, v: Tuple[bytes, str, int]) -> None:
    cap = _cache_capacity()
    with _global_lock:
        _code_cache[k] = v
        _code_cache.move_to_end(k)
        # Code cache doesn't require explicit destruction on eviction, but prune per-key locks
        while len(_code_cache) > cap:
            old_k, _ = _code_cache.popitem(last=False)
            _compile_locks.pop(old_k, None)


def _lru_get_module(k: _ModuleKey) -> Optional[Tuple[int, int]]:
    global _module_hits, _module_misses
    with _global_lock:
        val = _module_cache.get(k)
        if val is not None:
            _module_cache.move_to_end(k)
            _module_hits += 1
            return val
        _module_misses += 1
        return None


def _lru_put_module(k: _ModuleKey, v: Tuple[int, int]) -> None:
    cap = _cache_capacity()
    with _global_lock:
        _module_cache[k] = v
        _module_cache.move_to_end(k)
        # Evict least-recently used entries; skip ones currently in-use
        if len(_module_cache) > cap:
            attempts = 0
            while len(_module_cache) > cap and attempts < len(_module_cache) + 1:
                attempts += 1
                old_k, old_v = _module_cache.popitem(last=False)
                if _module_in_use.get(old_k, 0) > 0:
                    # Reinsert at end and skip eviction for now
                    _module_cache[old_k] = old_v
                    _module_cache.move_to_end(old_k)
                    continue
                try:
                    _C._cuda_module_unload(old_v[0])  # type: ignore[attr-defined]
                except Exception:
                    pass


def _flush_module_cache() -> None:
    with _global_lock:
        for k, (mod_h, _) in list(_module_cache.items()):
            try:
                _C._cuda_module_unload(mod_h)  # type: ignore[attr-defined]
            except Exception:
                pass
            _module_cache.pop(k, None)
        _module_in_use.clear()


def _maybe_set_triton_ptxas_path() -> None:
    """Ensure Triton uses a recent PTXAS when available.

    Mirrors torch._inductor.runtime.compile_tasks._set_triton_ptxas_path but,
    for VibeTensor, prefers any `ptxas` found on PATH when TRITON_PTXAS_PATH
    is not already set.
    """
    if _os.getenv("TRITON_PTXAS_PATH") is not None:
        return
    try:
        ptxas = _shutil.which("ptxas")  # type: ignore[attr-defined]
    except Exception:
        return
    if not ptxas:
        return
    try:
        if _os.path.isfile(ptxas) and _os.access(ptxas, _os.X_OK):
            _os.environ["TRITON_PTXAS_PATH"] = ptxas
    except Exception:
        # Best-effort only; fall back to Triton's default resolution.
        return


# NOTE: Avoid atexit callbacks into native extension to prevent teardown hazards
# in some CI environments. Tests can explicitly call clear_cache() if needed.
# _atexit.register(_flush_module_cache)


@dataclass(frozen=True)
class State:
    name: str
    cc: Tuple[int, int]
    threads_per_block: int
    effective_shared: int
    triton_version: str


def _is_cpu_tensor_like(obj: Any) -> bool:
    dev = getattr(obj, "device", None)
    if not isinstance(dev, (tuple, list)) or len(dev) < 2:
        return False
    try:
        dev_type = int(dev[0])
    except Exception:
        return False
    return dev_type == _KDLCPU


def _is_cuda_tensor_like(obj: Any) -> Tuple[bool, Optional[int]]:
    dev = getattr(obj, "device", None)
    if not isinstance(dev, (tuple, list)) or len(dev) < 2:
        return (False, None)
    try:
        dev_type = int(dev[0])
        dev_index = int(dev[1])
    except Exception:
        return (False, None)
    if dev_type in (_KDLCUDA, _KDLCUDAMANAGED):
        return (True, dev_index)
    return (False, None)


def _numel(sizes: Sequence[int]) -> int:
    n = 1
    for s in sizes:
        si = int(s)
        if si < 0:
            raise ValueError("size must be >= 0")
        if si == 0:
            return 0
        n *= si
    return n


def _assert_dense_contiguous_zero_offset(t: Any) -> None:
    try:
        sizes = tuple(int(s) for s in getattr(t, "sizes"))
        strides = tuple(int(s) for s in getattr(t, "strides"))
        off = int(getattr(t, "storage_offset"))
    except Exception:
        raise TypeError("expected a vibetensor._C.Tensor with sizes/strides/storage_offset")
    # Contiguous check: last stride == 1 and stride[i] == prod(sizes[i+1:])
    expected = [0] * len(sizes)
    acc = 1
    for i in range(len(sizes) - 1, -1, -1):
        expected[i] = acc
        acc *= sizes[i] if sizes[i] != 0 else 1
    if tuple(strides) != tuple(expected):
        raise ValueError("tensor must be dense contiguous (row-major)")
    if off != 0:
        raise ValueError("tensor storage_offset must be 0")


def _normalize_meta(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    m: Dict[str, Any] = dict(meta or {})
    # Basic validation for supported constexpr keys
    allowed = {"BLOCK_SIZE", "BLOCK_M", "BLOCK_N", "BLOCK_K"}
    for k in list(m.keys()):
        if k not in allowed:
            raise ValueError(f"unsupported meta key: {k}")
        try:
            m[k] = int(m[k])
        except Exception:
            raise TypeError(f"meta['{k}'] must be an int")
        if m[k] <= 0:
            raise ValueError(f"meta['{k}'] must be > 0")
    # Enforce 1D or 2D only
    if ("BLOCK_M" in m) != ("BLOCK_N" in m):
        raise ValueError("meta must include both BLOCK_M and BLOCK_N for 2D kernels")
    return m


def _ptr_token_to_dtype(tok: str) -> str:
    t = tok.lstrip("*").lower()
    # Accept fp32/float32 synonyms; fp16/float16; bf16/bfloat16
    if t in ("fp32", "float32"): return "float32"
    if t in ("fp16", "float16"): return "float16"
    if t in ("bf16", "bfloat16"): return "bfloat16"
    if t in ("int32", "i32"): return "int32"
    if t in ("int64", "i64"): return "int64"
    if t in ("bool", "b8", "u8"): return "bool"
    # Default: return normalized token to surface a clear error downstream
    return t


def _expand_grid_dims(g: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(g) == 1:
        return (int(g[0]), 1, 1)
    if len(g) == 2:
        return (int(g[0]), int(g[1]), 1)
    if len(g) >= 3:
        return (int(g[0]), int(g[1]), int(g[2]))
    raise ValueError("grid_fn must return a tuple of 1 to 3 ints")


def _call_grid_callable(
    grid: Callable[..., Any],
    st: State,
    inputs: Sequence[_C.Tensor],
    meta: Dict[str, Any],
) -> Tuple[int, ...]:
    """Invoke a user-provided grid callable.

    Supports either grid(meta) or grid(state, inputs, meta) arities.
    """
    try:
        sig = _inspect.signature(grid)
        params = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                _inspect.Parameter.POSITIONAL_ONLY,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        nargs = len(params)
    except Exception:
        nargs = None

    if nargs == 1:
        g = grid(dict(meta))
    else:
        g = grid(st, inputs, dict(meta))

    if not isinstance(g, (tuple, list)):
        raise TypeError("grid callable must return a tuple or list of ints")

    return tuple(int(x) for x in g)


def _threads_per_block(num_warps: Optional[int]) -> int:
    nw = int(num_warps) if num_warps is not None else 4
    if nw <= 0:
        raise ValueError("num_warps must be > 0")
    return nw * 32


def cache_stats() -> Dict[str, int]:
    """
    Return a snapshot of Triton cache counters and sizes.
    Keys: code_hits, code_misses, module_hits, module_misses, code_size, module_size, capacity
    """
    with _global_lock:
        return {
            "code_hits": _code_hits,
            "code_misses": _code_misses,
            "module_hits": _module_hits,
            "module_misses": _module_misses,
            "code_size": len(_code_cache),
            "module_size": len(_module_cache),
            "capacity": _cache_capacity(),
        }


def reset_cache_stats() -> None:
    """Reset cache hit/miss counters to zero (sizes unaffected)."""
    global _code_hits, _code_misses, _module_hits, _module_misses
    with _global_lock:
        _code_hits = _code_misses = _module_hits = _module_misses = 0


def clear_cache() -> None:
    """Clear Triton caches and unload all cached modules."""
    with _global_lock:
        _code_cache.clear()
        _compile_locks.clear()
    _flush_module_cache()


def _count_entry_params(ptx: bytes, entry: str) -> int:
    """Return number of .param entries in the PTX .entry for the given symbol."""
    try:
        s = ptx.decode("utf-8", errors="ignore")
        i = s.find(f".entry {entry}")
        if i < 0:
            return 0
        lb = s.find("(", i)
        rb = s.find(")", lb)
        if lb < 0 or rb < 0:
            return 0
        header = s[lb:rb]
        # Count occurrences of '.param' in the parameter list
        return header.count(".param")
    except Exception:
        return 0


def register(
    opname: str,
    kernel: Callable[..., Any],
    *,
    signature: str | None = None,
    grid_fn: Optional[Callable[[State, Sequence[_C.Tensor], Dict[str, Any]], Tuple[int, int, int]]] = None,
    meta: Optional[Dict[str, Any]] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
    shared_mem: Optional[int] = None,
    allow_broadcast: bool = False,
    allow_hetero_shapes: bool = False,
    args_fn: Optional[Callable[[State, Sequence[_C.Tensor], Dict[str, Any]], List[int]]] = None,
    out_shape_fn: Optional[Callable[[State, Sequence[_C.Tensor], Dict[str, Any]], Sequence[int]]] = None,
    grid: Optional[Union[Sequence[int], Callable[..., Tuple[int, int, int]]]] = None,
) -> None:
    """
    Install a boxed Python override for an operator name backed by a
    ``@triton.jit`` kernel.

    Behavior:
      - CPU tensors: redispatch to the base kernel unchanged.
      - CUDA tensors: compile the Triton kernel to PTX (cached), load/resolve a
        CUDA module/function (cached), and launch on the current VibeTensor CUDA
        stream via ``_C._cuda_launch``.

    Notes:
      - ``signature`` is required for CUDA execution (e.g. ``"*fp32,*fp32,*fp32,i32"``).
      - Dynamic (extern) shared memory defaults to Triton's compiled requirement
        (``compiled.metadata.shared``) unless ``shared_mem`` is explicitly provided.
    """
    if not isinstance(opname, str):
        raise TypeError("opname must be a string")
    if not callable(kernel):
        raise TypeError("kernel must be callable")
    if signature is not None and not isinstance(signature, str):
        raise TypeError("signature must be a string or None")
    if grid_fn is not None and not callable(grid_fn):
        raise TypeError("grid_fn must be callable or None")
    if meta is not None and not isinstance(meta, dict):
        raise TypeError("meta must be a dict or None")
    if num_warps is not None and not isinstance(num_warps, int):
        raise TypeError("num_warps must be an int or None")
    if num_stages is not None and not isinstance(num_stages, int):
        raise TypeError("num_stages must be an int or None")
    if num_stages is not None and int(num_stages) <= 0:
        raise ValueError("num_stages must be > 0")
    if shared_mem is not None and not isinstance(shared_mem, int):
        raise TypeError("shared_mem must be an int or None")
    if not isinstance(allow_broadcast, bool):
        raise TypeError("allow_broadcast must be a bool")
    if not isinstance(allow_hetero_shapes, bool):
        raise TypeError("allow_hetero_shapes must be a bool")
    if args_fn is not None and not callable(args_fn):
        raise TypeError("args_fn must be callable or None")
    if out_shape_fn is not None and not callable(out_shape_fn):
        raise TypeError("out_shape_fn must be callable or None")
    if grid is not None and not (callable(grid) or isinstance(grid, (tuple, list))):
        raise TypeError("grid must be a tuple/list or callable or None")

    # Pre-parse static pieces for CUDA path
    normalized_meta = _normalize_meta(meta)
    sig_tokens: List[str] = []
    if signature is not None:
        sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]
    explicit_grid = grid

    def _override_fn(*args: Any) -> Any:
        # No inputs: redispatch to base
        if len(args) == 0:
            return _C._redispatch_boxed_current(*args)
        first = args[0]
        # CPU path: redispatch to base kernel unchanged
        if _is_cpu_tensor_like(first):
            return _C._redispatch_boxed_current(*args)
        # CUDA path
        is_cuda, dev_index = _is_cuda_tensor_like(first)
        if is_cuda:
            if dev_index is None:
                raise RuntimeError("invalid CUDA device index")
            # Validate all inputs are CUDA tensors on the same device and contiguous
            inputs: List[Any] = list(args)
            if len(inputs) == 0:
                raise TypeError("expected at least one tensor argument")
            # Ensure all are VBT tensors with same device and dtype; shape equality optionally enforced
            base_sizes = tuple(int(s) for s in getattr(inputs[0], "sizes"))
            base_dtype = str(getattr(inputs[0], "dtype"))
            for t in inputs:
                ok, di = _is_cuda_tensor_like(t)
                if not ok or int(di) != int(dev_index):
                    raise TypeError("all inputs must be CUDA tensors on the same device")
                _assert_dense_contiguous_zero_offset(t)
                if not allow_hetero_shapes:
                    if tuple(int(s) for s in getattr(t, "sizes")) != base_sizes:
                        if not allow_broadcast:
                            raise ValueError("all inputs must have the same shape (broadcast not implemented)")
                        else:
                            raise NotImplementedError("allow_broadcast=True is not implemented")
                if str(getattr(t, "dtype")) != base_dtype:
                    raise TypeError("all inputs must have the same dtype")

            # Enforce pointer dtype expectations from signature
            out_dtype = base_dtype
            if sig_tokens:
                ptr_types = [tok for tok in sig_tokens if tok.strip().startswith("*")]
                if ptr_types:
                    # Input pointers (all but last) must match base dtype
                    expected_in = _ptr_token_to_dtype(ptr_types[0])
                    if expected_in != base_dtype:
                        raise TypeError("input tensor dtype does not match signature")
                    expected_out = _ptr_token_to_dtype(ptr_types[-1])
                    # Allow heterogeneous output dtype (e.g., cast kernels)
                    out_dtype = expected_out

            # Zero-sized fast path: allocate and return (use base_sizes)
            if _numel(base_sizes) == 0:
                return _C._cuda_empty(list(base_sizes), out_dtype, int(dev_index))  # type: ignore[attr-defined]

            # Determine arch and threading state
            cc = _C._cuda_device_cc(int(dev_index))  # type: ignore[attr-defined]
            try:
                import triton  # type: ignore
                triton_version = str(getattr(triton, "__version__", ""))
            except Exception:
                raise NotImplementedError("vibetensor.triton requires Triton; install 'triton' package")
            tpb = _threads_per_block(num_warps)
            eff_shmem = int(shared_mem) if shared_mem is not None else 0
            stages_key = -1 if num_stages is None else int(num_stages)
            st = State(name="kernel", cc=(int(cc[0]), int(cc[1])), threads_per_block=tpb, effective_shared=eff_shmem, triton_version=triton_version)

            # Compile kernel to PTX (with code cache)
            if signature is None:
                raise TypeError("signature is required for CUDA kernels in vibetensor.triton.register")
            ck = _code_key(opname, st.cc, st.triton_version, kernel, signature, normalized_meta, tpb, stages_key)

            # Per-key compile lock to avoid duplicate compilations
            with _global_lock:
                lock = _compile_locks.get(ck)
                if lock is None:
                    lock = _threading.Lock()
                    _compile_locks[ck] = lock
            with lock:
                cached = _lru_get_code(ck)
                if cached is None:
                    ptx, entry, compiled_shmem = _compile_to_ptx(
                        kernel,
                        signature=signature,
                        meta=normalized_meta,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    _lru_put_code(ck, (ptx, entry, int(compiled_shmem)))
                else:
                    ptx, entry, compiled_shmem = cached

            # Triton may require dynamic (extern) shared memory for some kernels.
            # Use the compiled requirement by default; allow explicit override.
            if shared_mem is None:
                eff_shmem = int(compiled_shmem)

            # Refresh State so user callables see the effective shared mem.
            st = State(
                name="kernel",
                cc=(int(cc[0]), int(cc[1])),
                threads_per_block=tpb,
                effective_shared=int(eff_shmem),
                triton_version=triton_version,
            )

            # Compute grid from explicit grid, grid_fn, or meta
            if explicit_grid is not None:
                # Highest precedence: explicit grid parameter. It is never rewritten
                # or reoriented by meta-derived policies.
                if callable(explicit_grid):
                    g = _call_grid_callable(explicit_grid, st, inputs, dict(normalized_meta))
                else:
                    g = tuple(int(x) for x in explicit_grid)  # type: ignore[arg-type]
                launch_grid = _expand_grid_dims(tuple(g))
            elif grid_fn is not None:
                g = grid_fn(st, inputs, dict(normalized_meta))
                launch_grid = _expand_grid_dims(g)
            else:
                # Derive commonly-used 1D/2D grids from meta
                if "BLOCK_SIZE" in normalized_meta:
                    # If args_fn provided, prefer user n from args; else numel of first input
                    n = _numel(base_sizes)
                    bs = int(normalized_meta["BLOCK_SIZE"])
                    gx = (n + bs - 1) // bs
                    launch_grid = (gx, 1, 1)
                elif "BLOCK_M" in normalized_meta and "BLOCK_N" in normalized_meta:
                    # 2D launch: prefer output shape if provided
                    out_M: Optional[int] = None
                    out_N: Optional[int] = None
                    if out_shape_fn is not None:
                        try:
                            out_sz = tuple(int(s) for s in out_shape_fn(st, inputs, dict(normalized_meta)))
                            if len(out_sz) >= 2:
                                out_M = int(out_sz[-2])
                                out_N = int(out_sz[-1])
                        except Exception:
                            out_M = out_N = None
                    if out_M is None or out_N is None:
                        # Fallback inference: M from first input, N from second input if present
                        try:
                            a_sizes = tuple(int(s) for s in getattr(inputs[0], "sizes"))
                        except Exception:
                            a_sizes = base_sizes
                        m_candidate: Optional[int] = None
                        n_candidate: Optional[int] = None
                        if len(a_sizes) >= 2:
                            m_candidate = int(a_sizes[-2])
                        elif len(a_sizes) == 1:
                            m_candidate = int(a_sizes[0])

                        b_sizes: Tuple[int, ...] = ()
                        if len(inputs) >= 2:
                            try:
                                b_sizes = tuple(int(s) for s in getattr(inputs[1], "sizes"))
                            except Exception:
                                b_sizes = ()
                        if len(b_sizes) >= 1:
                            n_candidate = int(b_sizes[-1])
                        elif len(a_sizes) >= 1:
                            n_candidate = int(a_sizes[-1])

                        if out_M is None:
                            out_M = m_candidate
                        if out_N is None:
                            out_N = n_candidate
                    if out_M is None or out_N is None:
                        raise ValueError(
                            "cannot infer 2D grid from inputs; provide grid or grid_fn, or an out_shape_fn that returns (..., M, N)"
                        )
                    bm = int(normalized_meta["BLOCK_M"])
                    bn = int(normalized_meta["BLOCK_N"])
                    mode = _os.getenv("VBT_TRITON_GRID_ORIENTATION", "legacy").lower()
                    if mode == "pytorch":
                        gx = (int(out_M) + bm - 1) // bm
                        gy = (int(out_N) + bn - 1) // bn
                    else:
                        # legacy: N-first orientation
                        gx = (int(out_N) + bn - 1) // bn
                        gy = (int(out_M) + bm - 1) // bm
                    launch_grid = (gx, gy, 1)
                else:
                    raise ValueError("grid_fn is required when meta lacks BLOCK_SIZE or BLOCK_M/BLOCK_N")

            block = (tpb, 1, 1)

            # Build argument list from signature: map pointer args to inputs and output (last pointer)
            ptr_positions = [i for i, tok in enumerate(sig_tokens) if tok.lstrip().startswith("*")]
            if not ptr_positions:
                raise ValueError("signature must include at least one pointer (output)")
            # Pointers: all but last come from inputs in order; last is output we allocate (shape from out_shape_fn or first input)
            num_input_ptrs = len(ptr_positions) - 1
            if num_input_ptrs > len(inputs):
                raise ValueError("not enough input tensors for signature pointers")

            # Determine output shape (default base_sizes) and allocate
            out_sizes = tuple(base_sizes)
            if out_shape_fn is not None:
                try:
                    out_sizes = tuple(int(s) for s in out_shape_fn(st, inputs, dict(normalized_meta)))
                except Exception as e:
                    raise RuntimeError(f"out_shape_fn failed: {e}")
            out = _C._cuda_empty(list(out_sizes), out_dtype, int(dev_index))  # type: ignore[attr-defined]

            # Scalars: prefer args_fn when provided
            scalars: List[Any] = []
            scalar_tokens = [tok for tok in sig_tokens if not tok.strip().startswith("*")]
            if args_fn is not None:
                try:
                    provided = list(args_fn(st, inputs, dict(normalized_meta)))
                except Exception as e:
                    raise RuntimeError(f"args_fn failed: {e}")
                if len(provided) != len(scalar_tokens):
                    raise ValueError("args_fn must return exactly the number of scalar tokens in signature")
                # Cast ints (i32/i64); other types currently unsupported
                for tok, val in zip(scalar_tokens, provided):
                    t = tok.strip().lower()
                    if t in ("i32","int32","i64","int64"):
                        scalars.append(int(val))
                    elif t in ("fp32", "float32", "float"):
                        scalars.append(float(val))
                    else:
                        raise ValueError(f"unsupported scalar token in signature: {tok}")
            else:
                # Default: support a single trailing i32/i64 'n' equal to numel(out)
                for tok in scalar_tokens:
                    t = tok.strip().lower()
                    if t in ("i32","int32"):
                        nval = int(_numel(out_sizes))
                        if nval < 0 or nval > 0x7FFFFFFF:
                            raise ValueError("i32 scalar out of range for signature")
                        scalars.append(nval)
                    elif t in ("i64","int64"):
                        scalars.append(int(_numel(out_sizes)))
                    else:
                        raise ValueError(f"unsupported scalar token in signature: {tok}")

            # Compose argument list in signature order: for pointer tokens, take from inputs then out
            argv: List[Any] = []
            in_iter = iter(inputs)
            remaining_input_ptrs = num_input_ptrs
            for tok in sig_tokens:
                if tok.strip().startswith("*"):
                    if remaining_input_ptrs > 0:
                        argv.append(next(in_iter))
                        remaining_input_ptrs -= 1
                    else:
                        argv.append(out)
                else:
                    # Append scalar in order of appearance
                    if not scalars:
                        raise ValueError("internal error: scalar list underflow")
                    argv.append(scalars.pop(0))

            # Triton >=3.5 adds extra scratch pointer params even when size==0; detect via PTX
            total_params = _count_entry_params(ptx, entry)
            extra = max(0, total_params - len(sig_tokens))
            if extra:
                argv.extend([None] * extra)

            # Resolve/load function from module cache, then launch on current VBT stream
            stream_handle = vt._cuda_stream_handle_current()
            if stream_handle is None:
                # Best-effort: fall back to device current stream helper
                stream_handle = int(_C._cuda_stream_handle_current_for_device(int(dev_index)))  # type: ignore[attr-defined]

            mk: _ModuleKey = (int(dev_index), ck)
            # Per-module-key lock to avoid duplicate loads
            with _global_lock:
                mlock = _module_locks.get(mk)
                if mlock is None:
                    mlock = _threading.Lock()
                    _module_locks[mk] = mlock
            with mlock:
                handles = _lru_get_module(mk)
                if handles is None:
                    # Double-check under lock then load module and resolve function
                    cached2 = _module_cache.get(mk)
                    if cached2 is None:
                        mod_h = _C._cuda_module_load_ptx(ptx)  # type: ignore[attr-defined]
                        fn_h = _C._cuda_module_get_function(mod_h, entry)  # type: ignore[attr-defined]
                        _lru_put_module(mk, (int(mod_h), int(fn_h)))
                        handles = (int(mod_h), int(fn_h))
                    else:
                        handles = cached2

            # Take a short lease to prevent eviction during launch
            with _global_lock:
                _module_in_use[mk] = _module_in_use.get(mk, 0) + 1
                func_handle = int(handles[1])
            try:
                if _os.getenv("VBT_TRITON_DEBUG", "0") == "1":
                    try:
                        types = [type(x).__name__ for x in argv]
                        ivalues = [int(x) for x in argv if isinstance(x, int)]
                        print(f"[vt_triton] argv_types={types} ints={ivalues}")
                    except Exception:
                        pass
                _C._cuda_launch(func_handle, launch_grid, block, eff_shmem, int(stream_handle), argv)  # type: ignore[attr-defined]
            finally:
                with _global_lock:
                    cnt = _module_in_use.get(mk, 0)
                    if cnt <= 1:
                        _module_in_use.pop(mk, None)
                    else:
                        _module_in_use[mk] = cnt - 1
            return out

        # Unknown device/type → redispatch
        return _C._redispatch_boxed_current(*args)

    # Try-register the override; duplicate registrations are no-ops
    try:
        _C._register_boxed_python_override(opname, _override_fn)
    except Exception as e:
        msg = str(e)
        if msg.startswith("duplicate CPU impl (boxed): ") and opname in msg:
            return
        raise


def _compile_to_ptx(
    kernel: Any,
    *,
    signature: str,
    meta: Optional[Dict[str, Any]] = None,
    num_warps: Optional[int] = None,
    num_stages: Optional[int] = None,
) -> Tuple[bytes, str, int]:
    """
    Compile a @triton.jit kernel to PTX using Triton's compiler backends, without
    importing torch. Returns (ptx_bytes, entry_name, shared_mem_bytes).

    - signature: explicit ABI, e.g., "*fp32,*fp32,*fp32,i32" (last pointer is output)
    - meta: constexpr values like {"BLOCK_SIZE": 1024}
    - num_warps: optional override (defaults to backend default)
    """
    _maybe_set_triton_ptxas_path()
    try:
        import triton  # type: ignore
    except Exception:
        raise NotImplementedError("vibetensor.triton requires Triton; install 'triton' package")

    # Build a signature mapping for ASTSource: name -> type or "constexpr"
    sig_tokens = [tok.strip() for tok in signature.split(",") if tok.strip()]

    # Attempt to introspect param names and constexpr flags from Triton JITFunction
    arg_names = getattr(kernel, "arg_names", None)
    params = getattr(kernel, "params", None)
    if not isinstance(arg_names, (list, tuple)) or params is None:
        raise TypeError("kernel must be a @triton.jit function (JITFunction)")

    sig_map: Dict[str, str] = {}
    idx = 0
    for i, p in enumerate(params):
        name = arg_names[i]
        is_constexpr = bool(getattr(p, "is_constexpr", False))
        if is_constexpr:
            sig_map[name] = "constexpr"
        else:
            if idx >= len(sig_tokens):
                raise ValueError("signature arity does not match kernel parameters")
            sig_map[name] = sig_tokens[idx]
            idx += 1
    # Enforce exact arity: no extra tokens in signature
    if idx != len(sig_tokens):
        raise ValueError("signature has extra tokens that do not match kernel parameters")

    consts = dict(meta or {})

    # Prefer modern Triton compile via ASTSource + target backend
    modern_available = True
    try:
        from triton.compiler.compiler import ASTSource, make_backend  # type: ignore
        from triton.runtime import driver  # type: ignore
    except Exception:
        modern_available = False

    if modern_available:
        target = driver.active.get_current_target()
        # Build AST from the kernel + signature/consts; specialization empty for v1
        if _os.getenv("VBT_TRITON_DEBUG", "0") == "1":
            try:
                print(f"[vt_triton] sig_map={sig_map} consts={consts}")
            except Exception:
                pass
        src = ASTSource(kernel, sig_map, consts, {})
        # Triton 3.5+ expects a plain dict for options
        opt_dict: Dict[str, Any] = {}
        if num_warps is not None:
            opt_dict["num_warps"] = int(num_warps)
        if num_stages is not None:
            opt_dict["num_stages"] = int(num_stages)
        # Prefer a conservative arch by default, but pick a PTX ISA version that
        # is accepted by ptxas for the active target.
        opt_dict.setdefault("arch", "sm80")
        if "ptx_version" not in opt_dict:
            arch_raw = getattr(target, "arch", None)
            arch_num = None

            if isinstance(arch_raw, int):
                arch_num = arch_raw
            elif isinstance(arch_raw, str):
                s = arch_raw.strip().lower()
                if s.startswith("sm"):
                    s = s[2:]
                if s.startswith("_"):
                    s = s[1:]
                digits = ""
                for ch in s:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits:
                    try:
                        arch_num = int(digits)
                    except Exception:
                        arch_num = None

            if arch_num is not None:
                # PTX ISA requirements (empirically validated via CUDA 13 ptxas):
                # - sm100a: >= 8.6
                # - sm103a: >= 8.8
                if arch_num >= 103:
                    opt_dict["ptx_version"] = 88
                elif arch_num >= 100:
                    opt_dict["ptx_version"] = 86
                else:
                    opt_dict["ptx_version"] = 80
            else:
                opt_dict["ptx_version"] = 80
        try:
            compiled = triton.compile(src, target=target, options=opt_dict)  # type: ignore[arg-type]
        except TypeError as e:
            # Older Triton versions may not support num_stages; retry without it.
            if "num_stages" in str(e):
                opt_dict.pop("num_stages", None)
                compiled = triton.compile(src, target=target, options=opt_dict)  # type: ignore[arg-type]
            else:
                raise

        asm = getattr(compiled, "asm", None)
        if not isinstance(asm, dict) or "ptx" not in asm:
            raise RuntimeError("Triton compile produced no PTX")
        ptx = asm["ptx"]
        if isinstance(ptx, bytes):
            ptx_bytes: bytes = ptx
        else:
            ptx_bytes = str(ptx).encode("utf-8")

        md = getattr(compiled, "metadata", None)

        # Entry name from metadata if available, fallback to function name
        entry = getattr(compiled, "name", None)
        if not isinstance(entry, str):
            if isinstance(md, dict):
                entry = md.get("name")  # type: ignore[assignment]
            else:
                entry = getattr(md, "name", None)
        if not isinstance(entry, str) or not entry:
            entry = getattr(kernel, "__name__", "kernel")

        shared_bytes = 0
        try:
            if isinstance(md, dict):
                shared_bytes = int(md.get("shared", 0) or 0)
            else:
                shared_bytes = int(getattr(md, "shared", 0) or 0)
        except Exception:
            shared_bytes = 0

        return ptx_bytes, entry, shared_bytes

    # Fallback to legacy compile paths where available
    # Try top-level triton.compile taking (fn, ...)
    try:
        triton_compile = getattr(triton, "compile")  # type: ignore[attr-defined]
    except Exception as e2:  # pragma: no cover - extremely old Triton
        raise RuntimeError(f"Triton compile API unavailable: {e2}")

    kwargs: Dict[str, Any] = {}
    if num_warps is not None:
        kwargs["num_warps"] = int(num_warps)
    if num_stages is not None:
        kwargs["num_stages"] = int(num_stages)
    if consts:
        kwargs["constants"] = consts
    # Prefer passing signature string; some Triton versions reject dicts
    try:
        compiled = triton_compile(kernel, signature=signature, device="cuda", **kwargs)  # type: ignore[call-arg]
    except TypeError as e2:  # pragma: no cover - extremely old Triton
        # Older Triton may not accept num_stages; retry without it.
        if "num_stages" in str(e2):
            kwargs.pop("num_stages", None)
            compiled = triton_compile(kernel, signature=signature, device="cuda", **kwargs)  # type: ignore[call-arg]
        else:
            raise RuntimeError(f"Triton compile API unavailable: {e2}")
    asm = getattr(compiled, "asm", None)
    if isinstance(asm, dict) and "ptx" in asm:
        ptx_val = asm["ptx"]
        ptx_bytes = ptx_val if isinstance(ptx_val, bytes) else str(ptx_val).encode("utf-8")
    elif isinstance(compiled, (str, bytes)):
        ptx_bytes = compiled if isinstance(compiled, bytes) else compiled.encode("utf-8")
    else:
        raise RuntimeError("Triton legacy compile did not yield PTX")
    md = getattr(compiled, "metadata", None)
    shared_bytes = 0
    try:
        if isinstance(md, dict):
            shared_bytes = int(md.get("shared", 0) or 0)
        else:
            shared_bytes = int(getattr(md, "shared", 0) or 0)
    except Exception:
        shared_bytes = 0

    entry = getattr(compiled, "name", None) or getattr(kernel, "__name__", "kernel")
    return ptx_bytes, entry, shared_bytes
