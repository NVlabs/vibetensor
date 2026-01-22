# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ._about import __version__  # noqa: F401
from .ops import ops  # noqa: F401
from . import library as library  # noqa: F401
from . import _C as _C  # noqa: F401


def _patch_numpy_from_dlpack_writable() -> None:
    """Ensure numpy.from_dlpack() returns writable arrays when possible.

    NumPy 2.x marks arrays created via from_dlpack as read-only by default
    for many providers (including torch and VibeTensor), which breaks
    patterns that expect hook arguments to be mutable views. We follow the
    DLPack ownership contract and simply flip the writable flag on the
    returned array.
    """

    try:
        import numpy as _np  # type: ignore[import]
    except Exception:  # pragma: no cover - numpy is a test dependency
        return

    orig = getattr(_np, "from_dlpack", None)
    if orig is None:
        return
    # Avoid double-patching
    if getattr(orig, "__vbt_patched__", False):  # type: ignore[attr-defined]
        return

    def _vbt_from_dlpack(x, /, *args, **kwargs):  # type: ignore[override]
        # NumPy's from_dlpack expects a provider object with __dlpack__ /
        # __dlpack_device__. Many producers (including vibetensor.torch.to_dlpack)
        # return a raw PyCapsule instead, so wrap it.
        if type(x).__name__ == "PyCapsule":
            def _capsule_device(cap):
                # Best-effort extraction of (device_type, device_id) from a DLPack
                # capsule. This keeps NumPy error paths sane for non-CPU capsules
                # (NumPy only supports CPU DLPack) and avoids lying about devices.
                try:
                    import ctypes as _ctypes
                except Exception:
                    return (1, 0)

                PyCapsule_GetPointer = _ctypes.pythonapi.PyCapsule_GetPointer
                PyCapsule_GetPointer.argtypes = [_ctypes.py_object, _ctypes.c_char_p]
                PyCapsule_GetPointer.restype = _ctypes.c_void_p

                class _DLDevice(_ctypes.Structure):
                    _fields_ = [
                        ("device_type", _ctypes.c_int),
                        ("device_id", _ctypes.c_int),
                    ]

                class _DLTensor(_ctypes.Structure):
                    _fields_ = [
                        ("data", _ctypes.c_void_p),
                        ("device", _DLDevice),
                    ]

                class _DLManagedTensor(_ctypes.Structure):
                    _fields_ = [("dl_tensor", _DLTensor)]

                ptr = None
                try:
                    ptr = PyCapsule_GetPointer(cap, b"dltensor")
                except Exception:
                    # Some consumers rename capsules after consumption.
                    try:
                        ptr = PyCapsule_GetPointer(cap, b"used_dltensor")
                    except Exception:
                        ptr = None
                if not ptr:
                    return (1, 0)

                mt = _ctypes.cast(ptr, _ctypes.POINTER(_DLManagedTensor)).contents
                dev = mt.dl_tensor.device
                return (int(dev.device_type), int(dev.device_id))

            class _CapsuleProvider:
                __slots__ = ("_cap", "_device")

                def __init__(self, cap):
                    self._cap = cap
                    self._device = _capsule_device(cap)

                def __dlpack__(self, *a, **k):  # type: ignore[misc]
                    return self._cap

                def __dlpack_device__(self, *a, **k):  # type: ignore[misc]
                    return self._device

            x = _CapsuleProvider(x)

        arr = orig(x, *args, **kwargs)
        try:
            # Always return a writable array; copy to decouple from providers
            arr = arr.copy()
            arr.setflags(write=True)
        except Exception:
            pass
        return arr

    setattr(_vbt_from_dlpack, "__vbt_patched__", True)
    _np.from_dlpack = _vbt_from_dlpack  # type: ignore[assignment]


def _patch_autograd_no_grad() -> None:
    """Patch `_C.autograd.no_grad` / `.enable_grad` to Python helpers.

    The C++ binding exposes RAII guards, but Python semantics are owned by
    :mod:`vibetensor.autograd`; this indirection keeps behavior centralized and
    avoids nanobind context-manager edge cases.
    """

    ag = getattr(_C, "autograd", None)
    if ag is None:
        return

    try:
        import vibetensor.autograd as A  # local import to avoid cycles
    except Exception:
        return

    try:
        ag.no_grad = A.no_grad  # type: ignore[attr-defined]
        ag.enable_grad = A.enable_grad  # type: ignore[attr-defined]
    except Exception:
        # Best-effort patch; if it fails, leave original bindings in place.
        pass


def _patch_autograd_inference_mode() -> None:
    """Patch `_C.autograd.inference_mode` to Python helper if available."""
    ag = getattr(_C, "autograd", None)
    if ag is None:
        return

    # Require inference helpers to exist; otherwise degrade to no-op.
    if not hasattr(ag, "is_inference_mode_enabled") or not hasattr(ag, "_set_inference_mode_enabled"):
        return

    try:
        import vibetensor.autograd as A  # local import to avoid cycles
    except Exception:
        return

    try:
        ag.inference_mode = A.inference_mode  # type: ignore[attr-defined]
    except Exception:
        # Best-effort patch; if it fails, leave original binding in place.
        pass


def _patch_tensor_autograd_surface() -> None:
    """Install PyTorch-like Tensor autograd attributes on _C.Tensor.

    This wires .grad, .requires_grad, .is_leaf, and .grad_fn to the
    underlying C++ methods while keeping the core TensorImpl type simple.
    """

    tensor_cls = getattr(_C, "Tensor", None)
    ag = getattr(_C, "autograd", None)
    if tensor_cls is None or ag is None:
        return

    _clear_grad = getattr(ag, "_clear_tensor_grad", None)

    # Capture original methods before installing descriptors so we can
    # delegate without recursion even after we override the attributes.
    _orig_grad = getattr(tensor_cls, "grad", None)
    if not callable(_orig_grad):
        _orig_grad = None

    _orig_requires_grad = getattr(tensor_cls, "requires_grad", None)
    if not callable(_orig_requires_grad):
        _orig_requires_grad = None

    _orig_set_requires_grad = getattr(tensor_cls, "set_requires_grad", None)
    if not callable(_orig_set_requires_grad):
        _orig_set_requires_grad = None

    class _GradProxy:
        """Lightweight wrapper so that x.grad and x.grad() both work.

        - Attribute access and DLPack protocol are forwarded to the
          underlying Tensor, so code expecting a Tensor still behaves
          as before.
        - Calling the proxy (x.grad()) returns the underlying Tensor
          object, preserving the legacy method surface.
        """

        __slots__ = ("_t",)

        def __init__(self, t):
            self._t = t

        def __call__(self):
            return self._t

        def __getattr__(self, name):
            return getattr(self._t, name)

        def __dlpack__(self, *args, **kwargs):  # type: ignore[misc]
            return self._t.__dlpack__(*args, **kwargs)

        def __dlpack_device__(self, *args, **kwargs):  # type: ignore[misc]
            return self._t.__dlpack_device__(*args, **kwargs)

        def __repr__(self) -> str:  # pragma: no cover - trivial
            return repr(self._t)

    class _GradDescriptor:
        def __get__(self, instance, owner=None):  # type: ignore[override]
            if instance is None:
                return self
            # Prefer the original grad() method if it exists; fall back to
            # grad_tensor() which is guaranteed not to be shadowed.
            if _orig_grad is not None:
                result = _orig_grad(instance)
            else:
                grad_tensor = getattr(instance, "grad_tensor", None)
                if grad_tensor is None:
                    raise AttributeError("Tensor.grad is not available on this build")
                result = grad_tensor()
            if result is None:
                return None
            return _GradProxy(result)

        def __set__(self, instance, value):  # type: ignore[override]
            if value is not None:
                raise AttributeError(
                    "Tensor.grad: can only be set to None; use backward() to populate grads"
                )
            if _clear_grad is None:
                raise AttributeError("Tensor.grad clearing is not available on this build")
            _clear_grad(instance)

    class _RequiresGradDescriptor:
        def __get__(self, instance, owner=None):  # type: ignore[override]
            if instance is None:
                return self
            if _orig_requires_grad is None:
                raise AttributeError("Tensor.requires_grad is not available on this build")
            return bool(_orig_requires_grad(instance))

        def __set__(self, instance, value):  # type: ignore[override]
            if _orig_set_requires_grad is None:
                raise AttributeError("Tensor.requires_grad setter is not available on this build")
            _orig_set_requires_grad(instance, bool(value))

    try:
        setattr(tensor_cls, "grad", _GradDescriptor())
    except Exception:
        # Best-effort: if this fails, keep the method surface only.
        pass

    try:
        setattr(tensor_cls, "requires_grad", _RequiresGradDescriptor())
    except Exception:
        pass

    # Properties for is_leaf and grad_fn backed by C++ helpers.
    try:
        is_leaf_method = getattr(tensor_cls, "is_leaf", None)
        if callable(is_leaf_method):
            def _is_leaf_prop(self, _m=is_leaf_method):  # type: ignore[misc]
                return bool(_m(self))

            setattr(tensor_cls, "is_leaf", property(_is_leaf_prop))
    except Exception:
        pass

    try:
        grad_fn_handle = getattr(tensor_cls, "_grad_fn_handle", None)
        if callable(grad_fn_handle):
            def _grad_fn_prop(self, _h=grad_fn_handle):  # type: ignore[misc]
                return _h(self)

            setattr(tensor_cls, "grad_fn", property(_grad_fn_prop))
    except Exception:
        pass

    # `.shape` parity alias for `.sizes`.
    try:
        if not hasattr(tensor_cls, "shape") and hasattr(tensor_cls, "sizes"):
            def _shape_prop(self):  # type: ignore[misc]
                return tuple(int(s) for s in getattr(self, "sizes", ()))

            setattr(tensor_cls, "shape", property(_shape_prop))
    except Exception:
        pass

    # Minimal Tensor.item() helper for single-element tensors.
    try:
        if not hasattr(tensor_cls, "item"):
            def _item(self):  # type: ignore[misc]
                # NOTE: Avoid DLPack→torch for CUDA tensors in the common case.
                # Raw DLPack capsules do not carry stream semantics, and consumers
                # may observe stale values when producers run on non-default streams.
                # For correctness we prefer a synchronous D2H copy when NumPy is
                # available, and fall back to a conservative torch + device sync
                # path otherwise.

                KDLCPU = 1
                KDLCUDA = 2
                KDLCUDAMANAGED = 13

                dev = getattr(self, "device", None)
                dev_type = None
                dev_idx = 0
                if isinstance(dev, tuple) and len(dev) >= 1:
                    dev_type = int(dev[0])
                    if len(dev) >= 2:
                        dev_idx = int(dev[1])

                # Preflight: avoid large D2H copies for non-scalar tensors.
                sizes = None
                try:
                    sizes = getattr(self, "sizes", None)
                except Exception:
                    sizes = None

                is_single = None
                if sizes is not None:
                    try:
                        is_single = True
                        for s in sizes:
                            if int(s) != 1:
                                is_single = False
                                break
                    except Exception:
                        is_single = None

                if is_single is False:
                    raise RuntimeError(
                        "Tensor.item() is only supported for tensors with a single element in VibeTensor"
                    )

                # Optional NumPy (preferred).
                try:
                    import numpy as _np  # type: ignore[import]
                except Exception:
                    _np = None  # type: ignore[assignment]

                if dev_type in (KDLCUDA, KDLCUDAMANAGED):
                    if _np is not None and hasattr(_C, "_cuda_d2h_copy_numpy_sync"):
                        try:
                            arr = _C._cuda_d2h_copy_numpy_sync(self)
                            if arr.size != 1:
                                raise RuntimeError(
                                    "Tensor.item() is only supported for tensors with a single element in VibeTensor"
                                )
                            return arr.item()
                        except Exception:
                            # Fall back to torch path (e.g. dtypes not representable in
                            # NumPy like bfloat16, or layout constraints).
                            pass

                    # Fallback: torch path (with conservative sync to avoid stale reads).
                    try:
                        import torch as _torch  # type: ignore[import]
                    except Exception as e:
                        raise RuntimeError(
                            "Tensor.item() on CUDA tensors requires numpy or torch to be installed"
                        ) from e

                    synced = False
                    try:
                        _torch.cuda.synchronize(dev_idx)
                        synced = True
                    except Exception:
                        try:
                            _torch.cuda.synchronize()
                            synced = True
                        except Exception as e:
                            raise RuntimeError(
                                "Tensor.item(): torch.cuda.synchronize() failed"
                            ) from e

                    if not synced:
                        raise RuntimeError("Tensor.item(): torch.cuda.synchronize() failed")

                    cap = _C._to_dlpack(self)
                    t = _torch.utils.dlpack.from_dlpack(cap)
                    if t.numel() != 1:
                        raise RuntimeError(
                            "Tensor.item() is only supported for tensors with a single element in VibeTensor"
                        )
                    return t.item()

                if dev_type == KDLCPU or dev_type is None:
                    # CPU tensor: prefer NumPy view when available, else torch DLPack.
                    if _np is not None:
                        try:
                            maybe_numpy = getattr(self, "numpy", None)
                            if callable(maybe_numpy):
                                arr = maybe_numpy()
                            else:
                                cap = _C._to_dlpack(self)
                                arr = _np.from_dlpack(cap)

                            if arr.size != 1:
                                raise RuntimeError(
                                    "Tensor.item() is only supported for tensors with a single element in VibeTensor"
                                )
                            return arr.item()
                        except Exception:
                            # Fall back to torch path for dtypes not supported by NumPy
                            # (e.g. bfloat16) or other conversion failures.
                            pass

                    try:
                        import torch as _torch  # type: ignore[import]
                    except Exception as e:
                        raise RuntimeError(
                            "Tensor.item() requires torch or numpy to be installed"
                        ) from e

                    cap = _C._to_dlpack(self)
                    t = _torch.utils.dlpack.from_dlpack(cap)
                    if t.numel() != 1:
                        raise RuntimeError(
                            "Tensor.item() is only supported for tensors with a single element in VibeTensor"
                        )
                    return t.item()

                raise RuntimeError("Tensor.item(): unsupported device")

            setattr(tensor_cls, "item", _item)
    except Exception:
        pass


_patch_numpy_from_dlpack_writable()
_patch_autograd_no_grad()
_patch_autograd_inference_mode()
_patch_tensor_autograd_surface()


def _register_cleanup_handlers() -> None:
    """Register atexit handlers to clear C++ registries holding Python objects.

    Static C++ maps (g_autograd_py, g_overrides) store nb::object references.
    If destroyed after Python finalization, Py_DECREF causes segfault.
    Clear them before interpreter shutdown via atexit.
    """
    import atexit

    def _cleanup():
        # Clear autograd backward registry
        reset_autograd = getattr(_C, "_reset_autograd_py", None)
        if callable(reset_autograd):
            try:
                reset_autograd()
            except Exception:
                pass

        # Clear Python override registry
        reset_overrides = getattr(_C, "_reset_boxed_python_overrides", None)
        if callable(reset_overrides):
            try:
                reset_overrides()
            except Exception:
                pass

    atexit.register(_cleanup)


_register_cleanup_handlers()
