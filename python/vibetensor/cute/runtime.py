# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib as _hashlib
import struct as _struct
import threading as _threading
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from vibetensor import _C as _C
import vibetensor.torch as vt

# DLPack device type codes (reuse DLPack definitions)
_KDLCPU = 1
_KDLCUDA = 2
_KDLCUDAMANAGED = 13

_PTR_SIZE = _struct.calcsize("P")


def _as_non_bool_int(x: Any, *, api_name: str) -> int:
    if isinstance(x, bool):
        raise TypeError(f"{api_name}: expected an int, not bool")
    return int(x)


@dataclass(frozen=True)
class CuteParamSpec:
    """A single CUDA kernel parameter spec (one `.param`).

    This is intentionally expressed at the `cuLaunchKernel` parameter level.
    """

    kind: str  # "tensor_ptr" | "memref" | "scalar" | "device_ptr" | "bytes"
    dtype: Optional[str] = None
    rank: Optional[int] = None
    index_width: int = 64
    size: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise TypeError("CuteParamSpec.kind must be a non-empty str")

        k = self.kind
        if k not in {"tensor_ptr", "memref", "scalar", "device_ptr", "bytes"}:
            raise ValueError(f"CuteParamSpec.kind must be a supported kind, got {k!r}")

        if k == "scalar":
            if not isinstance(self.dtype, str) or not self.dtype:
                raise TypeError("CuteParamSpec.scalar requires dtype=str")
            if self.dtype not in {
                "i32",
                "i64",
                "u8",
                "u16",
                "u32",
                "u64",
                "f32",
                "f64",
            }:
                raise ValueError(f"CuteParamSpec.scalar dtype is not supported: {self.dtype!r}")
            if self.rank is not None:
                raise TypeError("CuteParamSpec.scalar must not set rank")
            if self.size is not None:
                raise TypeError("CuteParamSpec.scalar must not set size")

        elif k == "memref":
            if self.dtype is not None:
                raise TypeError("CuteParamSpec.memref must not set dtype")
            if self.size is not None:
                raise TypeError("CuteParamSpec.memref must not set size")
            if self.rank is None:
                raise TypeError("CuteParamSpec.memref requires rank")
            r = _as_non_bool_int(self.rank, api_name="CuteParamSpec.memref rank")
            if r < 0:
                raise ValueError("CuteParamSpec.memref rank must be >= 0")
            if int(self.index_width) != 64:
                raise ValueError("CuteParamSpec.memref only supports index_width=64")

        elif k == "bytes":
            if self.dtype is not None:
                raise TypeError("CuteParamSpec.bytes must not set dtype")
            if self.rank is not None:
                raise TypeError("CuteParamSpec.bytes must not set rank")
            if self.size is None:
                raise TypeError("CuteParamSpec.bytes requires size")
            n = _as_non_bool_int(self.size, api_name="CuteParamSpec.bytes size")
            if n < 1:
                # `_cuda_launch_checked` disallows size==0 (sizes are [1, 4096]).
                raise ValueError("CuteParamSpec.bytes size must be >= 1")
            if n > 4096:
                raise ValueError("CuteParamSpec.bytes size must be <= 4096")

        else:
            # tensor_ptr / device_ptr
            if self.dtype is not None:
                raise TypeError(f"CuteParamSpec.{k} must not set dtype")
            if self.rank is not None:
                raise TypeError(f"CuteParamSpec.{k} must not set rank")
            if self.size is not None:
                raise TypeError(f"CuteParamSpec.{k} must not set size")

    def expected_param_size(self) -> int:
        """Return the expected by-value parameter size in bytes."""
        k = self.kind
        if k in {"tensor_ptr", "device_ptr"}:
            return int(_PTR_SIZE)

        if k == "scalar":
            dt = str(self.dtype)
            if dt in {"u8"}:
                return 1
            if dt in {"u16"}:
                return 2
            if dt in {"i32", "u32", "f32"}:
                return 4
            if dt in {"i64", "u64", "f64"}:
                return 8
            raise ValueError(f"unsupported scalar dtype: {dt!r}")

        if k == "memref":
            r = int(self.rank or 0)
            if int(self.index_width) != 64:
                raise ValueError("memref only supports index_width=64")
            return 24 + 16 * r

        if k == "bytes":
            return int(self.size or 0)

        raise ValueError(f"unsupported kind: {k!r}")


@dataclass(frozen=True)
class CuteKernelArtifact:
    cubin: bytes
    kernel: str
    params: Tuple[CuteParamSpec, ...]
    func_attrs: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cubin, (bytes, bytearray, memoryview)):
            raise TypeError("CuteKernelArtifact.cubin must be bytes-like")
        object.__setattr__(self, "cubin", bytes(self.cubin))

        if not isinstance(self.kernel, str) or not self.kernel:
            raise TypeError("CuteKernelArtifact.kernel must be a non-empty str")

        if not isinstance(self.params, (tuple, list)):
            raise TypeError("CuteKernelArtifact.params must be a sequence of CuteParamSpec")
        ps = tuple(self.params)
        for p in ps:
            if not isinstance(p, CuteParamSpec):
                raise TypeError("CuteKernelArtifact.params items must be CuteParamSpec")
        object.__setattr__(self, "params", ps)

        if not isinstance(self.func_attrs, (tuple, list)):
            raise TypeError("CuteKernelArtifact.func_attrs must be a sequence of (int,int) pairs")
        fa: List[Tuple[int, int]] = []
        for item in self.func_attrs:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("CuteKernelArtifact.func_attrs items must be (attribute, value)")
            a = _as_non_bool_int(item[0], api_name="CuteKernelArtifact.func_attrs attribute")
            v = _as_non_bool_int(item[1], api_name="CuteKernelArtifact.func_attrs value")
            fa.append((int(a), int(v)))
        object.__setattr__(self, "func_attrs", tuple(fa))

    @property
    def expected_param_sizes(self) -> Tuple[int, ...]:
        return tuple(int(p.expected_param_size()) for p in self.params)


@dataclass
class _CachedFunction:
    func_handle: int
    attrs_applied: bool = False


_ModuleKey = Tuple[int, str]  # (device_index, sha256(cubin))
_FunctionKey = Tuple[int, str, str]  # (device_index, sha256(cubin), kernel_name)

_cache_lock = _threading.RLock()
_module_cache: dict[_ModuleKey, int] = {}  # mod_handle
_function_cache: dict[_FunctionKey, _CachedFunction] = {}


def clear_cache() -> None:
    """Unload any cached CUDA modules loaded via CuteKernel."""
    with _cache_lock:
        mods = list(_module_cache.values())
        _module_cache.clear()
        _function_cache.clear()

    for mod_h in mods:
        try:
            _C._cuda_module_unload(int(mod_h))  # type: ignore[attr-defined]
        except Exception:
            # Best-effort only.
            pass


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


def _first_cuda_device_index(args: Sequence[object], specs: Sequence[CuteParamSpec]) -> int:
    for a, s in zip(args, specs):
        if s.kind not in {"tensor_ptr", "memref"}:
            continue
        ok, idx = _is_cuda_tensor_like(a)
        if ok and idx is not None:
            return int(idx)
    raise ValueError("CuteKernel.launch: expected at least one CUDA tensor arg to infer device")


def _resolve_stream_handle(dev_index: int, stream: Optional[int]) -> int:
    if stream is not None:
        return _as_non_bool_int(stream, api_name="CuteKernel.launch stream")

    h_opt = vt._cuda_stream_handle_current()
    if h_opt is not None:
        return int(h_opt)

    try:
        return int(_C._cuda_stream_handle_current_for_device(int(dev_index)))  # type: ignore[attr-defined]
    except Exception:
        return 0


class CuteKernel:
    """A thin runtime wrapper for launching CuTeDSL-exported kernels.

    This runtime does **not** import or depend on CuTeDSL at call time.
    """

    def __init__(self, art: CuteKernelArtifact):
        if not isinstance(art, CuteKernelArtifact):
            raise TypeError("CuteKernel: expected a CuteKernelArtifact")
        self._art = art
        self._expected_sizes = art.expected_param_sizes
        self._code_key = _hashlib.sha256(art.cubin).hexdigest()

    @property
    def artifact(self) -> CuteKernelArtifact:
        return self._art

    @property
    def expected_param_sizes(self) -> Tuple[int, ...]:
        return self._expected_sizes

    def _get_or_load_module(self, dev_index: int) -> int:
        key: _ModuleKey = (int(dev_index), str(self._code_key))

        with _cache_lock:
            hit = _module_cache.get(key)
            if hit is not None:
                return int(hit)

        # Load outside the global lock (driver calls can be slow).
        mod_h = int(_C._cuda_module_load_ptx(self._art.cubin))  # type: ignore[attr-defined]

        with _cache_lock:
            # Another thread could have populated it while we loaded; prefer existing.
            existing = _module_cache.get(key)
            if existing is not None:
                try:
                    _C._cuda_module_unload(int(mod_h))  # type: ignore[attr-defined]
                except Exception:
                    pass
                return int(existing)

            _module_cache[key] = int(mod_h)
            return int(mod_h)

    def _get_or_load_function(self, dev_index: int) -> _CachedFunction:
        mod_h = self._get_or_load_module(dev_index)
        key: _FunctionKey = (int(dev_index), str(self._code_key), str(self._art.kernel))

        with _cache_lock:
            hit = _function_cache.get(key)
            if hit is not None:
                return hit

        fn_h = int(_C._cuda_module_get_function(mod_h, self._art.kernel))  # type: ignore[attr-defined]
        entry = _CachedFunction(func_handle=int(fn_h), attrs_applied=False)

        with _cache_lock:
            # Another thread could have populated it while we loaded; prefer existing.
            existing = _function_cache.get(key)
            if existing is not None:
                return existing

            _function_cache[key] = entry
            return entry

    def _maybe_apply_func_attrs(self, fn: _CachedFunction) -> None:
        if fn.attrs_applied:
            return

        set_attr = getattr(_C, "_cuda_func_set_attribute", None)
        if not callable(set_attr):
            if self._art.func_attrs:
                raise NotImplementedError(
                    "_cuda_func_set_attribute is required to apply CuteKernelArtifact.func_attrs"
                )
            return

        drv_ver = getattr(_C, "_cuda_driver_version", None)
        if callable(drv_ver):
            dv = int(drv_ver())
            if dv >= 11080:
                # Opt-in for newer cluster behavior when supported.
                attr_code = getattr(_C, "_cuda_func_attribute_non_portable_cluster_size_allowed", None)
                if attr_code is not None:
                    ac = int(attr_code)
                    if ac != 0:
                        set_attr(int(fn.func_handle), ac, 1)

        for a, v in self._art.func_attrs:
            set_attr(int(fn.func_handle), int(a), int(v))

        fn.attrs_applied = True

    def launch(
        self,
        *args: object,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shmem_bytes: int = 0,
        stream: Optional[int] = None,
        record_stream: bool = True,
    ) -> None:
        if len(args) != len(self._art.params):
            raise ValueError(
                f"CuteKernel.launch: expected {len(self._art.params)} args, got {len(args)}"
            )

        dev_index = _first_cuda_device_index(args, self._art.params)

        # Validate all tensor-like args are CUDA tensors on the same device.
        for a, spec in zip(args, self._art.params):
            if spec.kind not in {"tensor_ptr", "memref"}:
                continue
            ok, idx = _is_cuda_tensor_like(a)
            if not ok or idx is None:
                raise TypeError(
                    f"CuteKernel.launch: expected a CUDA tensor for {spec.kind} param"
                )
            if int(idx) != int(dev_index):
                raise ValueError(
                    "CuteKernel.launch: all tensor args must be on the same CUDA device"
                )

        stream_handle = _resolve_stream_handle(dev_index, stream)

        # Resolve and cache module + function.
        fn = self._get_or_load_function(dev_index)
        self._maybe_apply_func_attrs(fn)

        # Pack args for _cuda_launch_checked.
        packed: List[object] = []
        tensor_args: List[object] = []
        for spec, a in zip(self._art.params, args):
            k = spec.kind
            if k == "tensor_ptr":
                packed.append(a)
                tensor_args.append(a)
            elif k == "memref":
                if spec.rank is None:
                    raise ValueError("CuteKernel.launch: memref param requires rank")
                desc = _C._cuda_arg_memref(  # type: ignore[attr-defined]
                    a,
                    rank=int(spec.rank),
                    index_width=int(spec.index_width),
                    allow_empty_for_grid0=False,
                )
                packed.append(desc)
                tensor_args.append(a)
            elif k == "scalar":
                dt = str(spec.dtype)
                if dt == "i32":
                    packed.append(_C._cuda_arg_i32(_as_non_bool_int(a, api_name="i32")))  # type: ignore[attr-defined]
                elif dt == "i64":
                    packed.append(_C._cuda_arg_i64(_as_non_bool_int(a, api_name="i64")))  # type: ignore[attr-defined]
                elif dt == "u8":
                    packed.append(_C._cuda_arg_u8(_as_non_bool_int(a, api_name="u8")))  # type: ignore[attr-defined]
                elif dt == "u16":
                    packed.append(_C._cuda_arg_u16(_as_non_bool_int(a, api_name="u16")))  # type: ignore[attr-defined]
                elif dt == "u32":
                    packed.append(_C._cuda_arg_u32(_as_non_bool_int(a, api_name="u32")))  # type: ignore[attr-defined]
                elif dt == "u64":
                    packed.append(_C._cuda_arg_u64(_as_non_bool_int(a, api_name="u64")))  # type: ignore[attr-defined]
                elif dt == "f32":
                    packed.append(_C._cuda_arg_f32(float(a)))  # type: ignore[attr-defined]
                elif dt == "f64":
                    packed.append(_C._cuda_arg_f64(float(a)))  # type: ignore[attr-defined]
                else:
                    raise ValueError(f"CuteKernel.launch: unsupported scalar dtype: {dt!r}")
            elif k == "device_ptr":
                packed.append(_C._cuda_arg_device_ptr(_as_non_bool_int(a, api_name="device_ptr")))  # type: ignore[attr-defined]
            elif k == "bytes":
                packed.append(_C._cuda_arg_bytes(a))  # type: ignore[attr-defined]
            else:
                raise ValueError(f"CuteKernel.launch: unsupported kind: {k!r}")

        if record_stream:
            for t in tensor_args:
                _C._cuda_record_stream(t, int(stream_handle))  # type: ignore[attr-defined]

        _C._cuda_launch_checked(  # type: ignore[attr-defined]
            int(fn.func_handle),
            tuple(int(x) for x in grid),
            tuple(int(x) for x in block),
            int(shmem_bytes),
            int(stream_handle),
            packed,
            expected_param_sizes=list(self._expected_sizes),
            strict=True,
        )
