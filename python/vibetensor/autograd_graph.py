# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, IO, Dict, NamedTuple
import sys as _sys
import contextlib
import threading

from vibetensor import _C as _C

_Tensor = _C.Tensor


def _get_ag():
    """Return `_C.autograd` submodule or raise.

    Graph inspection helpers are only available when autograd is compiled in;
    this helper centralizes the fallback behavior.
    """

    ag = getattr(_C, "autograd", None)
    if ag is None:
        raise RuntimeError(
            "vibetensor.autograd.graph: _C.autograd is unavailable in this build"
        )
    return ag


def _require_tensor(value: Any, *, arg_name: str) -> _Tensor:
    if not isinstance(value, _C.Tensor):
        raise TypeError(
            f"vibetensor.autograd.graph.{arg_name}: expected a VibeTensor `_C.Tensor` "
            f"got {type(value)!r}"
        )
    return value


class Node:
    """Read-only view of a VibeTensor autograd Node.

    Wraps a `_C.autograd.GradFn` handle and exposes a minimal inspection
    surface. Equality and hashing are by underlying Node identity derived
    from a debug identifier provided by the C++ core.

    Currently, Node objects are **purely observational**:
    - They cannot register hooks.
    - They cannot trigger execution.
    """

    __slots__ = ("_handle", "_debug_id")

    def __init__(self, handle: Any) -> None:
        ag = _get_ag()
        gradfn_type = getattr(ag, "GradFn", None)
        if gradfn_type is None or not isinstance(handle, gradfn_type):
            raise TypeError(
                "vibetensor.autograd.graph.Node: expected a _C.autograd.GradFn "
                f"handle, got {type(handle)!r}"
            )
        self._handle = handle
        meta = ag._grad_fn_debug_metadata(handle)
        self._debug_id = int(meta.get("debug_id", 0))

    # ------------------------------------------------------------------
    # Public inspection surface
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """Return C++ `Node::name()` as a string."""

        return str(self._handle.name)

    @property
    def next_functions(self) -> tuple[tuple["Node | None", int], ...]:
        """Outgoing edges as `(child_node_or_None, input_nr)` tuples.

        `None` is used when the underlying edge has a null `fn`.
        """

        ag = _get_ag()
        edges = ag._grad_fn_next_edges(self._handle)
        out: list[tuple[Node | None, int]] = []
        for child_handle, input_nr in edges:
            if child_handle is None:
                out.append((None, int(input_nr)))
            else:
                out.append((Node(child_handle), int(input_nr)))
        return tuple(out)

    def metadata(self) -> Dict[str, object]:
        """Return a small, stable metadata dictionary for debugging.

        At minimum this contains:

        - ``{"num_inputs": int, "has_input_meta": bool, "debug_id": int}``
        """

        ag = _get_ag()
        raw = ag._grad_fn_debug_metadata(self._handle)
        # Ensure a plain dict with builtin types
        return {str(k): raw[k] for k in raw.keys()}

    # ------------------------------------------------------------------
    # Identity semantics
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover - repr stability only
        return f"Node(name={self.name!r}, debug_id={self._debug_id})"

    def __hash__(self) -> int:
        return hash(self._debug_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self._debug_id == other._debug_id

    # ------------------------------------------------------------------
    # Unsupported hook APIs (parity note only)
    # ------------------------------------------------------------------
    def register_hook(self, hook: Any):  # pragma: no cover - simple error path
        raise NotImplementedError(
            "Node.register_hook is not supported; "
            "use Tensor.register_hook or register_multi_grad_hook instead."
        )

    def register_prehook(self, hook: Any):  # pragma: no cover - simple error path
        raise NotImplementedError(
            "Node.register_prehook is not supported."
        )


class GradientEdge(NamedTuple):
    node: Node
    output_nr: int


def get_gradient_edge(tensor: Any) -> GradientEdge:
    """Return the gradient edge for a tensor.

    Raises:
      TypeError  – if ``tensor`` is not a VibeTensor ``_C.Tensor``.
      RuntimeError – if ``tensor`` does not require grad or has no AutogradMeta.
    """

    t = _require_tensor(tensor, arg_name="tensor")
    if not bool(getattr(t, "requires_grad", False)):
        raise RuntimeError(
            "vibetensor.autograd.graph.get_gradient_edge: tensor does not "
            "require grad"
        )

    ag = _get_ag()
    handle, output_nr = ag._graph_get_gradient_edge(t)
    return GradientEdge(node=Node(handle), output_nr=int(output_nr))


def _root_node(root: Node | GradientEdge | Any) -> Node:
    """Normalize ``root`` argument for traversal helpers."""

    if isinstance(root, Node):
        return root
    if isinstance(root, GradientEdge):
        return root.node
    if isinstance(root, _C.Tensor):
        edge = get_gradient_edge(root)
        return edge.node
    raise TypeError(
        "vibetensor.autograd.graph.iter_nodes: root must be a Node, "
        "GradientEdge, or VibeTensor `_C.Tensor`"
    )


def iter_nodes(
    root: Node | GradientEdge | Any,
    *,
    max_depth: int | None = None,
) -> Iterator[Node]:
    """Depth-first traversal over the autograd graph.

    - Deduplicates nodes by a stable ``debug_id``.
    - Starts from:
      - ``root.node`` when given a :class:`GradientEdge`.
      - :func:`get_gradient_edge(root).node` when given a tensor.
      - ``root`` when given a :class:`Node`.
    """

    start = _root_node(root)
    seen: set[int] = set()
    stack: list[tuple[Node, int]] = [(start, 0)]

    while stack:
        node, depth = stack.pop()
        if node._debug_id in seen:  # type: ignore[attr-defined]
            continue
        seen.add(node._debug_id)  # type: ignore[attr-defined]
        yield node

        if max_depth is not None and depth >= max_depth:
            continue

        # Push children in reverse so iteration roughly matches a
        # left-to-right DFS when popped.
        for child, _input_nr in reversed(node.next_functions):
            if child is not None:
                stack.append((child, depth + 1))


def dump_graph(
    root: Node | GradientEdge | Any,
    *,
    max_depth: int | None = 5,
    file: IO[str] | None = None,
) -> None:
    """Pretty-print a summary of the autograd graph starting at ``root``.

    The exact textual format is intentionally simple and not part of the
    stable API; tests only assert on basic invariants.
    """

    out = file if file is not None else _sys.stdout
    for node in iter_nodes(root, max_depth=max_depth):
        meta = node.metadata()
        debug_id = meta.get("debug_id", None)
        num_inputs = meta.get("num_inputs", None)
        has_meta = meta.get("has_input_meta", None)
        print(
            f"{node.name}(debug_id={debug_id}, "
            f"num_inputs={num_inputs}, has_input_meta={has_meta})",
            file=out,
        )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

_state = threading.local()


@contextlib.contextmanager
def saved_tensors_hooks(pack_hook, unpack_hook):
    """Install saved-tensor hooks for the duration of the context.

    Hooks are **instrumentation-only** currently:
    - They see cloned, detached CPU copies of tensors that are saved for
      backward.
    - They cannot change which tensors the engine uses for gradients.

    Restrictions:
    - ``pack_hook`` and ``unpack_hook`` must be callable.
    - Both observe tensors under NoGrad from the engine's point of view.
    - For non-builtin hooks, the object returned by ``pack_hook`` must not be
      a VibeTensor ``_C.Tensor`` or a ``torch.Tensor``; this is enforced in
      C++ and results in a ``RuntimeError``.
    """

    if not callable(pack_hook) or not callable(unpack_hook):
        raise TypeError(
            "saved_tensors_hooks: pack_hook and unpack_hook must be callable"
        )

    ag = _get_ag()
    push = getattr(ag, "_push_saved_tensors_hooks", None)
    pop = getattr(ag, "_pop_saved_tensors_hooks", None)
    if not callable(push) or not callable(pop):  # pragma: no cover - defensive
        # Best-effort no-op when internals are unavailable.
        yield
        return

    push(pack_hook, unpack_hook, False)
    try:
        yield
    finally:
        pop()


@contextlib.contextmanager
def save_on_cpu():
    """Convenience wrapper over :func:`saved_tensors_hooks`

    In VibeTensor all differentiable tensors are already CPU float32, so this
    context is **semantically a no-op** for gradients. It is included for API
    parity and future-proofing.
    """

    ag = _get_ag()
    push = getattr(ag, "_push_saved_tensors_hooks", None)
    pop = getattr(ag, "_pop_saved_tensors_hooks", None)
    if not callable(push) or not callable(pop):  # pragma: no cover - defensive
        yield
        return

    def _pack(t):  # noqa: D401 - simple identity hook
        # Builtin hook: observe the tensor but don't keep it.
        return None

    def _unpack(payload):  # noqa: D401 - simple identity hook
        return None

    push(_pack, _unpack, True)
    try:
        yield
    finally:
        pop()


@contextlib.contextmanager
def disable_saved_tensors_hooks(
    error_message: str = "saved_tensors_hooks are disabled",
):
    """Disable ``saved_tensors_hooks`` / ``save_on_cpu`` within this block.

    - Blocks *new* hook installations; existing graphs with hooks continue to
      behave as they were captured.
    - Contexts are nestable; the innermost ``error_message`` wins.
    """

    ag = _get_ag()
    setter = getattr(ag, "_set_saved_tensors_disabled", None)
    if not callable(setter):  # pragma: no cover - defensive
        yield
        return

    # Maintain a Python-side shadow stack of disable states per thread.
    stack = getattr(_state, "disable_stack", [])
    if not isinstance(stack, list):  # pragma: no cover - defensive
        stack = []
    stack.append((True, str(error_message)))
    _state.disable_stack = stack

    setter(True, str(error_message))
    try:
        yield
    finally:
        stack.pop()
        if stack:
            disabled, msg = stack[-1]
        else:
            disabled, msg = (False, "")
        _state.disable_stack = stack
        setter(bool(disabled), str(msg))


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


class _RegState:
    __slots__ = ("tensors", "fn", "mode", "handles", "version", "grads", "any_fired_in_run")

    def __init__(self, tensors: Sequence[_Tensor], fn, mode: str) -> None:
        self.tensors = tuple(tensors)
        self.fn = fn
        self.mode = mode
        self.handles: list[Any] = []
        # Per-backward-run state
        self.version: int = -1
        self.grads: list[Any | None] = [None] * len(self.tensors)
        self.any_fired_in_run: bool = False


class _MultiGradState:
    __slots__ = ("regs", "next_id", "version", "callback_installed")

    def __init__(self) -> None:
        self.regs: dict[int, _RegState] = {}
        self.next_id: int = 1
        self.version: int = 0
        self.callback_installed: bool = False


_state_mg = threading.local()


def _get_multi_state() -> _MultiGradState:
    st = getattr(_state_mg, "value", None)
    if st is None:
        st = _MultiGradState()
        _state_mg.value = st
    return st


def _as_tensor_sequence_for_multi(value: Any, *, arg_name: str) -> tuple[_Tensor, ...]:
    if isinstance(value, _Tensor):
        return (value,)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = tuple(value)
        if not seq:
            raise RuntimeError(
                f"register_multi_grad_hook: {arg_name} must be a non-empty sequence"
            )
        for x in seq:
            if not isinstance(x, _Tensor):
                raise TypeError(
                    "register_multi_grad_hook: tensors must be VibeTensor `_C.Tensor` objects"
                )
        return seq
    raise TypeError(
        "register_multi_grad_hook: tensors must be a VibeTensor tensor or a non-empty sequence of tensors"
    )


def _multi_grad_per_tensor_hook(reg_id: int, index: int, grad):
    st = _get_multi_state()
    reg = st.regs.get(reg_id)
    if reg is None:
        return None

    run_version = st.version
    if reg.version != run_version:
        reg.version = run_version
        reg.grads = [None] * len(reg.grads)
        reg.any_fired_in_run = False

    if reg.mode == "all":
        reg.grads[index] = grad
    else:  # "any"
        if not reg.any_fired_in_run:
            reg.any_fired_in_run = True
            ag = _get_ag()
            stats_any = getattr(ag, "_stats_multi_grad_fired_any", None)
            if callable(stats_any):
                stats_any()
            reg.fn(grad)
    return None


def _backward_complete_dispatch() -> None:
    st = _get_multi_state()
    ag = _get_ag()
    stats_all = getattr(ag, "_stats_multi_grad_fired_all", None)
    cur_version = st.version

    try:
        for reg in list(st.regs.values()):
            if reg.mode != "all":
                continue
            if reg.version != cur_version:
                continue
            if not any(g is not None for g in reg.grads):
                continue
            if callable(stats_all):
                stats_all()
            reg.fn(list(reg.grads))
            # Reset per-run state for this registration; version
            # advancement happens once per GraphTask below.
            reg.grads = [None] * len(reg.grads)
            reg.any_fired_in_run = False
    finally:
        # Always advance the run version, even if hooks raise or the
        # engine reported an error. This prevents stale per-run state
        # from leaking into the next backward.
        st.version += 1


class MultiHandle:
    def __init__(self, reg_id: int) -> None:
        self._reg_id = reg_id
        self._removed = False

    def remove(self) -> None:
        """Detach this registration from future backward runs.

        Idempotent: calling ``remove()`` multiple times is allowed.
        """

        if self._removed:
            return
        st = _get_multi_state()
        reg = st.regs.pop(self._reg_id, None)
        self._removed = True
        if reg is None:
            return
        # Detach underlying per-tensor hooks.
        for h in reg.handles:
            try:
                h.remove()
            except Exception:
                pass
        reg.handles.clear()

        if not st.regs:
            ag = _get_ag()
            setter = getattr(ag, "_set_backward_complete_callback", None)
            if callable(setter):
                setter(None)
            st.callback_installed = False


def register_multi_grad_hook(
    tensors: Sequence[_Tensor],
    fn,
    *,
    mode: str = "all",
) -> MultiHandle:
    """Register a hook observing gradients for multiple tensors.

    - ``tensors`` must be non-view leaf tensors that require grad and are in the
      supported dtype/device domain (CPU float32). Unsupported tensors raise the
      same errors as ``Tensor.register_hook``.
    - ``fn`` is only ever called with **PyTorch ``torch.Tensor`` gradients**, not
      ``_C.Tensor`` objects, for easier integration with existing tooling.
    """

    if not callable(fn):
        raise TypeError(
            "register_multi_grad_hook: fn must be callable"
        )

    mode_str = str(mode)
    if mode_str not in ("all", "any"):
        raise ValueError(
            "register_multi_grad_hook: mode must be 'all' or 'any'"
        )

    ag = _get_ag()
    if bool(getattr(ag, "is_in_backward", lambda: False)()):
        raise RuntimeError(
            "register_multi_grad_hook: cannot be called while a backward is in progress"
        )

    ts = _as_tensor_sequence_for_multi(tensors, arg_name="tensors")

    st = _get_multi_state()
    reg_id = st.next_id
    st.next_id += 1
    reg = _RegState(ts, fn, mode_str)
    st.regs[reg_id] = reg

    # Install per-tensor hooks; validation of tensor domain is delegated to
    # ``Tensor.register_hook``.
    def make_hook(index: int):
        def _hook(grad):
            return _multi_grad_per_tensor_hook(reg_id, index, grad)

        return _hook

    for idx, t in enumerate(ts):
        h = t.register_hook(make_hook(idx))
        reg.handles.append(h)

    # Ensure global backward-complete dispatcher is registered.
    setter = getattr(ag, "_set_backward_complete_callback", None)
    if callable(setter):
        setter(_backward_complete_dispatch)
        st.callback_installed = True

    # Bump registration stats when available.
    stats_reg = getattr(ag, "_stats_multi_grad_registered", None)
    if callable(stats_reg):
        stats_reg()

    return MultiHandle(reg_id)
