# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
import triton.knobs


def _hook_inactive(hook: Any) -> bool:
    if hook is None:
        return True
    calls = getattr(hook, "calls", None)
    return calls is not None and not calls


_GET_CURRENT_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)
_Tensor = torch.Tensor
_int = int
_bool = bool
_rk_global = triton.knobs.runtime
_ck_global = triton.knobs.compilation


def _arg_key(arg: Any) -> Any:
    if arg is None:
        return None
    t = type(arg)
    if t is _Tensor:
        return arg.dtype if (arg.data_ptr() & 15) == 0 else (arg.dtype, 0)
    if t is _int:
        return 100 + (arg == 1) + 2 * ((arg & 15) == 0) + 4 * (-2147483648 <= arg <= 2147483647)
    if t is _bool:
        return 200 + arg
    if isinstance(arg, _Tensor):
        return arg.dtype if (arg.data_ptr() & 15) == 0 else (arg.dtype, 0)
    return t


class FastTritonLauncher:
    """Bypass Triton's per-call Python binder/specialization on steady decode paths.

    Caches ``CompiledKernel`` keyed by device, argument specializations (dtype,
    pointer 16-byte alignment, scalar ``== 1`` / ``% 16 == 0`` / int32 range,
    ``None`` vs non-``None``), and trailing ``constexpr`` values. Automatically
    falls back to standard ``jit_fn[grid](...)`` whenever runtime hooks, debug
    mode, instrumentation, ``launch_metadata``, or cache misses occur.

    Callers MUST pass positional arguments in exact ``jit_fn.arg_names`` order,
    with any ``constexpr`` arguments placed at the end (matching
    ``num_constexpr``), because Triton's native C launcher (``extractArgs`` in
    ``driver.c``) indexes ``kernel_args`` 1-to-1 with ``arg_annotations``
    including ``ARG_CONSTEXPR`` placeholders.
    """

    def __init__(
        self,
        jit_fn: Any,
        constexpr_names: tuple[str, ...] = (),
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.jit_fn = jit_fn
        self.constexpr_names = constexpr_names
        self.num_constexpr = len(constexpr_names)
        self.compile_kwargs = compile_kwargs or {}
        self._cache: dict[tuple[Any, ...], Any] = {}

    def _can_use_fast_path(self) -> bool:
        rk = _rk_global
        jit_fn = self.jit_fn
        eh = rk.launch_enter_hook
        xh = rk.launch_exit_hook
        return (
            (eh is None or _hook_inactive(eh))
            and (xh is None or _hook_inactive(xh))
            and not jit_fn.debug
            and not jit_fn.pre_run_hooks
            and jit_fn.launch_metadata is None
            and _ck_global.instrumentation_mode == ""
        )

    def launch(
        self,
        grid_0: int,
        grid_1: int,
        grid_2: int,
        device_idx: int,
        stream_handle: int,
        *args: Any,
    ) -> None:
        if grid_0 <= 0 or grid_1 <= 0 or grid_2 <= 0:
            return
        jit_fn = self.jit_fn
        can_fast = self._can_use_fast_path()
        num_constexpr = self.num_constexpr
        n_args = len(args)
        num_runtime = n_args - num_constexpr
        if can_fast:
            key_list: list[Any] = [device_idx]
            append = key_list.append
            for i in range(num_runtime):
                arg = args[i]
                t = type(arg)
                if t is _Tensor:
                    append(arg.dtype if (arg.data_ptr() & 15) == 0 else (arg.dtype, 0))
                elif t is _int:
                    append(
                        100
                        + (arg == 1)
                        + 2 * ((arg & 15) == 0)
                        + 4 * (-2147483648 <= arg <= 2147483647)
                    )
                elif arg is None:
                    append(None)
                elif t is _bool:
                    append(200 + arg)
                elif isinstance(arg, _Tensor):
                    append(arg.dtype if (arg.data_ptr() & 15) == 0 else (arg.dtype, 0))
                else:
                    append(t)
            for i in range(num_runtime, n_args):
                append(args[i])
            key: tuple[Any, ...] | None = tuple(key_list)
            kernel = self._cache.get(key)
            if kernel is not None:
                kernel.run(
                    grid_0,
                    grid_1,
                    grid_2,
                    stream_handle,
                    kernel.function,
                    kernel.packed_metadata,
                    None,
                    None,
                    None,
                    *args,
                )
                return
        else:
            key = None

        runtime_args = args[:num_runtime] if num_constexpr else args
        constexpr_vals = args[num_runtime:] if num_constexpr else ()
        grid = (grid_0, grid_1, grid_2)
        if num_constexpr or self.compile_kwargs:
            kw = dict(self.compile_kwargs)
            for name, val in zip(self.constexpr_names, constexpr_vals):
                kw[name] = val
            kernel = jit_fn[grid](*runtime_args, **kw)
        else:
            kernel = jit_fn[grid](*runtime_args)

        if (
            key is not None
            and hasattr(kernel, "run")
            and getattr(kernel, "function", None) is not None
            and hasattr(kernel, "packed_metadata")
        ):
            self._cache[key] = kernel

    def launch_on_device(
        self,
        grid_0: int,
        grid_1: int,
        grid_2: int,
        device: torch.device,
        *args: Any,
    ) -> None:
        dev_idx = device.index
        if dev_idx is None:
            dev_idx = torch.cuda.current_device()
        if _GET_CURRENT_RAW_STREAM is not None:
            stream_handle = _GET_CURRENT_RAW_STREAM(dev_idx)
        else:
            stream_handle = torch.cuda.current_stream(dev_idx).cuda_stream
        self.launch(grid_0, grid_1, grid_2, dev_idx, stream_handle, *args)
