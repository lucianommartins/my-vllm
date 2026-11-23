# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable
from concurrent.futures import Future
from multiprocessing import Lock
from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.network_utils import (
    aiter_requires_tcp_store,
    get_distributed_init_method,
    get_file_store_init_method,
    get_ip,
    get_open_port,
)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.vllm_net_devices import set_worker_net_device
from vllm.v1.outputs import AsyncModelRunnerOutput, DraftTokenIds, ModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


_EMPTY_KWARGS: dict[str, Any] = {}


class _ImmediateFuture(Future):
    """Lock-free completed Future for synchronous UniProcExecutor RPCs."""

    def __init__(self, value: Any = None, exc: BaseException | None = None):
        # Skip Future.__init__() to avoid allocating Condition/RLock per step.
        self._imm_result = value
        self._imm_exc = exc

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def result(self, timeout: float | None = None) -> Any:
        if self._imm_exc is not None:
            raise self._imm_exc
        return self._imm_result

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._imm_exc

    def add_done_callback(self, fn: Callable[[Future], Any]) -> None:
        try:
            fn(self)
        except Exception:
            logger.exception("exception calling callback for %r", self)

    def __repr__(self) -> str:
        state = "raised" if self._imm_exc is not None else "finished"
        return f"<{type(self).__name__} at {id(self):#x} state={state}>"


_NONE_SINGLE_FUTURE = _ImmediateFuture(None)


class AsyncOutputFuture(Future):
    def __init__(self, async_output: AsyncModelRunnerOutput, single_value: bool):
        # Skip Future.__init__() to avoid Condition/RLock churn per step.
        self.async_output = async_output
        self.single_value = single_value
        self._resolved = False
        self._value: Any = None
        self._exc: BaseException | None = None
        self._done_callbacks: list[Callable[[Future], Any]] | None = None

    def _resolve(self) -> None:
        if self._resolved:
            return
        try:
            output = self.async_output.get_output()
            self._value = output if self.single_value else [output]
        except Exception as e:
            self._exc = e
        self._resolved = True
        if self._done_callbacks is not None:
            callbacks = self._done_callbacks
            self._done_callbacks = None
            for cb in callbacks:
                try:
                    cb(self)
                except Exception:
                    logger.exception("exception calling callback for %r", self)

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return self._resolved

    def set_result(self, result: Any) -> None:
        self._value = result
        self._exc = None
        self._resolved = True

    def set_exception(self, exception: BaseException | None) -> None:
        self._exc = exception
        self._resolved = True

    def result(self, timeout: float | None = None):
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not self._resolved:
            self._resolve()
        if self._exc is not None:
            raise self._exc
        return self._value

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if timeout is not None:
            raise RuntimeError("timeout not implemented")

        if not self._resolved:
            self._resolve()
        return self._exc

    def add_done_callback(self, fn: Callable[[Future], Any]) -> None:
        if self._resolved:
            try:
                fn(self)
            except Exception:
                logger.exception("exception calling callback for %r", self)
            return
        if self._done_callbacks is None:
            self._done_callbacks = [fn]
        else:
            self._done_callbacks.append(fn)

    def __repr__(self) -> str:
        if not self._resolved:
            state = "pending"
        elif self._exc is not None:
            state = "raised"
        else:
            state = "finished"
        return f"<{type(self).__name__} at {id(self):#x} state={state}>"


class UniProcExecutor(Executor):
    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        self.driver_worker = WorkerWrapperBase(rpc_rank=0)
        self._sample_tokens_fn = None
        self._cached_driver_worker = None
        distributed_init_method, rank, local_rank = self._distributed_args()
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
            shared_worker_lock=Lock(),
        )

        # Set net device env vars for the worker if VLLM_GPU_NIC_PCIE_MAPPING is set
        set_worker_net_device(local_rank, self.vllm_config)

        self.driver_worker.init_worker(all_kwargs=[kwargs])
        self.driver_worker.init_device()

        if envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
            self.driver_worker.elastic_ep_execute("load_model")
        else:
            self.driver_worker.load_model()
        current_platform.update_block_size_for_backend(self.vllm_config)
        self._cached_driver_worker = self.driver_worker
        self._execute_model_fn = None
        if (
            type(self.driver_worker) is WorkerWrapperBase
            and "sample_tokens" not in self.driver_worker.__dict__
        ):
            self._sample_tokens_fn = self.driver_worker.worker.sample_tokens
            if (
                type(self.driver_worker).execute_model is WorkerWrapperBase.execute_model
                and "execute_model" not in self.driver_worker.__dict__
            ):
                self._execute_model_fn = self.driver_worker.worker.execute_model
        else:
            self._sample_tokens_fn = self.driver_worker.sample_tokens

    def _distributed_args(self) -> tuple[str, int, int]:
        """Return (distributed_init_method, rank, local_rank)."""
        if aiter_requires_tcp_store():
            distributed_init_method = get_distributed_init_method(
                get_ip(), get_open_port()
            )
        else:
            distributed_init_method = get_file_store_init_method()
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(":")
        local_rank = int(device_info[1]) if len(device_info) > 1 else 0
        return distributed_init_method, 0, local_rank

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if kwargs is None:
            kwargs = _EMPTY_KWARGS

        if not non_block:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                result = result.get_output()
            return result if single_value else [result]

        try:
            result = run_method(self.driver_worker, method, args, kwargs)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, single_value)
            if single_value and result is None:
                return _NONE_SINGLE_FUTURE
            return _ImmediateFuture(result if single_value else [result])
        except Exception as e:
            return _ImmediateFuture(exc=e)

    def execute_model(  # type: ignore[override]
        self, scheduler_output: SchedulerOutput, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        if (
            non_block
            and self.driver_worker is getattr(self, "_cached_driver_worker", None)
            and "collective_rpc" not in self.__dict__
            and type(self).collective_rpc is UniProcExecutor.collective_rpc
        ):
            exec_fn = getattr(self, "_execute_model_fn", None)
            if exec_fn is not None and (
                self.driver_worker.mm_receiver_cache is None
                or not scheduler_output.scheduled_new_reqs
            ):
                result = exec_fn(scheduler_output)
            else:
                result = self.driver_worker.execute_model(scheduler_output)
            if result is None:
                return _NONE_SINGLE_FUTURE
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, True)
            return _ImmediateFuture(result)
        output = self.collective_rpc(
            "execute_model",
            args=(scheduler_output,),
            non_block=non_block,
            single_value=True,
        )
        # In non-blocking mode, surface any exception as early as possible.
        if non_block and output.done():
            # Raise the exception in-line if the task failed.
            output.result()
        return output

    def sample_tokens(  # type: ignore[override]
        self, grammar_output: GrammarOutput | None, non_block: bool = False
    ) -> ModelRunnerOutput | None | Future[ModelRunnerOutput | None]:
        fn = getattr(self, "_sample_tokens_fn", None)
        if (
            fn is not None
            and self.driver_worker is getattr(self, "_cached_driver_worker", None)
            and "collective_rpc" not in self.__dict__
            and type(self).collective_rpc is UniProcExecutor.collective_rpc
        ):
            if not non_block:
                result = fn(grammar_output)
                if isinstance(result, AsyncModelRunnerOutput):
                    return result.get_output()
                return result
            try:
                result = fn(grammar_output)
            except Exception as e:
                return _ImmediateFuture(exc=e)
            if isinstance(result, AsyncModelRunnerOutput):
                return AsyncOutputFuture(result, True)
            if result is None:
                return _NONE_SINGLE_FUTURE
            return _ImmediateFuture(result)
        return self.collective_rpc(
            "sample_tokens",
            args=(grammar_output,),
            non_block=non_block,
            single_value=True,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        return self.collective_rpc("take_draft_token_ids", single_value=True)

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return

    def shutdown(self) -> None:
        if worker := self.driver_worker:
            worker.shutdown()

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        return True


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/features/torchrun/torchrun_example_offline.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """

    def _init_executor(self) -> None:
        """Initialize the worker and load the model."""
        assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, (
            "To get deterministic execution, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0"
        )
        super()._init_executor()

    def _distributed_args(self) -> tuple[str, int, int]:
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        return distributed_init_method, rank, local_rank

    def determine_available_memory(self) -> list[int]:  # in bytes
        # we need to get the min across all ranks.
        memory = super().determine_available_memory()
        from vllm.distributed.parallel_state import get_world_group

        cpu_group = get_world_group().cpu_group
        memory_tensor = torch.tensor([memory], device="cpu", dtype=torch.int64)
        dist.all_reduce(memory_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return [memory_tensor.item()]

    def shutdown(self) -> None:
        super().shutdown()
        # Release vLLM's communicators (TP/PP/DP/EP groups) now, as
        # MultiprocExecutor's workers do. Left alive, they are torn down by
        # static destructors at interpreter exit, where newer RCCL aborts the
        # process ("terminate called without an active exception"). The world
        # group and the default process group may belong to the launcher, so
        # they stay.
        from vllm.distributed.parallel_state import destroy_model_parallel

        destroy_model_parallel()
